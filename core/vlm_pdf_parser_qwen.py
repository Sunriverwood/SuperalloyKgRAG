import os
import json
import time
import logging
import yaml
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI


# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/superalloyKgRAG.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("SuperalloyKgRAG")
    logging.info("VLM PDF Parser日志记录器设置完成")
    return logger


# --- 加载配置 ---
def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    logging.info(f"正在从 {config_path} 加载配置...")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")

    with open(config_path, 'r', encoding='utf-8') as f:
        # 简单替换，实际生产建议使用 os.environ.get
        raw_config = f.read()
        config = yaml.safe_load(raw_config)

    logging.info("配置加载成功。")
    return config


class VLMPdfParser:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 VLM PDF 解析器 (Qwen / OpenAI SDK 版)
        """
        parser_cfg = config["vlm_parser"]
        llm_cfg = config["llm"]

        # 设置日志
        self.logger = logging.getLogger("SuperalloyKgRAG")

        # 目录和文件路径
        self.pdf_folder = PROJECT_ROOT / parser_cfg["input_dir"]
        self.output_folder = PROJECT_ROOT / parser_cfg["output_dir"]
        self.state_file = PROJECT_ROOT / parser_cfg["state_file_path"]
        self.prompt_path = PROJECT_ROOT / parser_cfg.get("parsing_prompt_path")
        self.requests_path = PROJECT_ROOT / parser_cfg.get("requests_path")

        # 批处理配置
        self.batch_size = parser_cfg["batch_size"]
        self.timeout_seconds = parser_cfg.get("batch_polling_timeout_seconds", 86400)
        self.sleep_interval = parser_cfg.get("sleep_interval", 10)

        # LLM配置
        self.model_name = llm_cfg["model"]
        # 优先读取环境变量
        self.api_key = os.getenv("QWEN_API_KEY") or llm_cfg.get("api_key")
        if not self.api_key:
            raise ValueError("未找到 API Key，请设置 QWEN_API_KEY")

        # 初始化 OpenAI 客户端
        # 注意：使用 oss:// 链接需要开启 X-DashScope-OssResourceResolve
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            default_headers={'X-DashScope-OssResourceResolve': 'enable'}
        )

        # 初始化环境和资源
        self._initialize_environment()

        # 加载状态和指令
        self.state = self._load_state()
        self.instructions = self._load_prompt()

    def _initialize_environment(self):
        """初始化环境：创建必要的目录"""
        self.logger.info("正在初始化环境和创建目录...")
        try:
            self.pdf_folder.mkdir(parents=True, exist_ok=True)
            self.output_folder.mkdir(parents=True, exist_ok=True)
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            if self.requests_path:
                self.requests_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info("✅ 目录初始化完成。")
        except Exception as e:
            self.logger.critical(f"❌ 初始化目录失败: {e}")
            raise

    def _load_state(self) -> Dict[str, Any]:
        """加载处理状态"""
        if self.state_file.exists():
            self.logger.info(f"正在加载状态文件: {self.state_file}")
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                return state
            except Exception as e:
                self.logger.error(f"❌ 加载状态文件失败: {e}")
                return {}
        else:
            return {}

    def _load_prompt(self) -> str:
        """加载PDF解析指令"""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"关键指令文件未找到: {self.prompt_path}")
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def list_pdfs(self) -> set:
        return {f.name for f in self.pdf_folder.iterdir() if f.suffix.lower() == ".pdf"}

    def save_json(self, data: dict, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_state(self, state: dict):
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    # -------- 辅助函数：上传到阿里云临时 OSS --------
    # 参考：千问-获取URL.pdf
    def _upload_to_oss_and_get_url(self, file_path: Path) -> str:
        """获取凭证并上传文件到 OSS，返回 oss:// URL"""
        # 1. 获取凭证
        url = "https://dashscope.aliyuncs.com/api/v1/uploads"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"action": "getPolicy", "model": self.model_name}

        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            raise Exception(f"获取上传凭证失败: {resp.text}")

        policy_data = resp.json()['data']

        # 2. 上传文件
        file_name = file_path.name
        key = f"{policy_data['upload_dir']}/{file_name}"

        files = {
            'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
            'Signature': (None, policy_data['signature']),
            'policy': (None, policy_data['policy']),
            'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key': (None, key),
            'success_action_status': (None, '200'),
            'file': (file_name, open(file_path, 'rb'))
        }

        upload_resp = requests.post(policy_data['upload_host'], files=files)
        if upload_resp.status_code != 200:
            raise Exception(f"上传文件到OSS失败: {upload_resp.text}")

        return f"oss://{key}"

    # -------- 阶段1：文件上传 --------
    def upload_files(self):
        """文件发现与上传 (获取 oss:// URL)"""
        self.logger.info("[阶段1] 文件发现与上传 (阿里云OSS)...")
        current_pdfs = self.list_pdfs()
        for pdf_file in current_pdfs:
            if pdf_file not in self.state:
                self.state[pdf_file] = {"status": "pending_upload"}
        self.save_state(self.state)

        for pdf_file, data in self.state.items():
            if data["status"] in ["pending_upload", "failed_upload"]:
                pdf_path = self.pdf_folder / pdf_file
                try:
                    self.logger.info(f"正在上传到临时存储: {pdf_file}")
                    # 调用辅助函数上传
                    oss_url = self._upload_to_oss_and_get_url(pdf_path)

                    self.state[pdf_file].update({
                        "status": "uploaded",
                        "uploaded_file_uri": oss_url,  # oss://...
                        "uploaded_file_name": pdf_file
                    })
                    self.logger.info(f"✅ 上传成功: {oss_url}")
                except Exception as e:
                    self.state[pdf_file].update({"status": "failed_upload", "error": str(e)})
                    self.logger.error(f"❌ 上传失败 {pdf_file}: {e}")
                finally:
                    self.save_state(self.state)

    # -------- 阶段2：创建 Batch 作业 --------
    def create_batch_jobs(self):
        """创建 OpenAI 兼容的 Batch 作业"""
        self.logger.info("[阶段2] 创建批处理作业...")

        # 筛选已上传的文件
        files_to_process = [
            f for f, d in self.state.items()
            if d.get("status") == "uploaded"
        ]

        if not files_to_process:
            self.logger.info("无待处理文件，无需创建新作业。")
            return

        # 分批处理
        chunks = [files_to_process[i:i + self.batch_size] for i in range(0, len(files_to_process), self.batch_size)]

        for i, chunk_files in enumerate(chunks):
            job_suffix = f"{datetime.now().strftime('%Y%m%d%H%M')}-{i + 1}"

            # 1. 构建 JSONL 内容
            jsonl_lines = []
            for pdf_file in chunk_files:
                file_uri = self.state[pdf_file]["uploaded_file_uri"]

                # 构建 OpenAI Chat Completion Request
                # Qwen-VL 支持通过 image_url 传递 oss:// 链接
                request_body = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.instructions},
                                {"type": "image_url", "image_url": {"url": file_uri}}
                            ]
                        }
                    ],
                    # 强制 JSON 输出 (如果模型支持，如 qwen-plus/max；VL 模型通常也支持但需验证)
                    # "response_format": {"type": "json_object"}
                }

                jsonl_line = {
                    "custom_id": pdf_file,  # 使用文件名作为ID
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body
                }
                jsonl_lines.append(json.dumps(jsonl_line, ensure_ascii=False))

            # 2. 保存 JSONL 文件
            batch_filename = f"batch_requests_{job_suffix}.jsonl"
            batch_file_path = self.requests_path.parent / batch_filename

            try:
                with open(batch_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(jsonl_lines))

                self.logger.info(f"  - 批量请求文件已保存: {batch_file_path}")

                # 3. 上传 JSONL 文件 (OpenAI File API)
                with open(batch_file_path, "rb") as f:
                    batch_input_file = self.client.files.create(
                        file=f,
                        purpose="batch"
                    )

                # 4. 提交 Batch 作业
                batch_job = self.client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"vlm-parsing-{job_suffix}"}
                )

                self.logger.info(f"✅ 作业创建成功 ID: {batch_job.id}")

                # 更新状态
                for pdf in chunk_files:
                    self.state[pdf].update({
                        "status": "processing",
                        "batch_job_id": batch_job.id  # 记录 ID 而不是 Name
                    })

            except Exception as e:
                self.logger.error(f"❌ 批处理作业创建失败: {e}")
                for pdf in chunk_files:
                    self.state[pdf].update({"status": "failed_job_creation", "error": str(e)})
            finally:
                self.save_state(self.state)

    # -------- 阶段3：作业监控 --------
    def monitor_jobs(self):
        """监控 OpenAI Batch 作业"""
        self.logger.info("[阶段3] 监控处理中的作业...")

        # 获取所有唯一的 job_id
        active_job_ids = {
            d["batch_job_id"] for d in self.state.values()
            if d.get("status") == "processing" and "batch_job_id" in d
        }

        if not active_job_ids:
            self.logger.info("当前无活动作业。")
            return

        while active_job_ids:
            self.logger.info(f"正在监控 {len(active_job_ids)} 个作业...")
            finished_ids = set()

            for job_id in list(active_job_ids):
                try:
                    # 查询作业状态
                    job = self.client.batches.retrieve(batch_id=job_id)
                    status = job.status
                    self.logger.info(f"  - 作业 '{job_id}' 状态: {status}")

                    if status in ('completed', 'failed', 'cancelled', 'expired'):
                        finished_ids.add(job_id)

                        if status == 'completed':
                            self.process_job_results(job)
                        else:
                            # 失败处理
                            err_msg = f"作业结束状态: {status}"
                            if hasattr(job, 'errors') and job.errors:
                                err_msg += f" {job.errors}"

                            self.logger.error(f"❌ 作业失败: {err_msg}")
                            for pdf, data in self.state.items():
                                if data.get('batch_job_id') == job_id:
                                    data.update({'status': f'failed_{status}', 'error': err_msg})

                except Exception as e:
                    self.logger.error(f"❌ 监控作业 '{job_id}' 异常: {e}")
                    # 不移除，重试

            active_job_ids -= finished_ids
            self.save_state(self.state)

            if active_job_ids:
                time.sleep(self.sleep_interval)

    # -------- 阶段4：结果处理 --------
    def process_job_results(self, job):
        """下载并解析 OpenAI Batch 结果"""
        self.logger.info(f"  -> 处理作业 '{job.id}' 结果...")

        output_file_id = job.output_file_id
        if not output_file_id:
            self.logger.error("❌ 作业完成但无输出文件ID")
            return

        try:
            # 下载内容
            content = self.client.files.content(output_file_id).text

            # 解析 JSONL 结果
            for line in content.strip().split("\n"):
                try:
                    res = json.loads(line)
                    custom_id = res.get("custom_id")  # 对应 pdf 文件名
                    response = res.get("response", {})

                    if not custom_id or custom_id not in self.state:
                        continue

                    if response.get("status_code") == 200:
                        # 提取生成的文本
                        # 路径: body -> choices[0] -> message -> content
                        choices = response.get("body", {}).get("choices", [])
                        if choices:
                            text_content = choices[0].get("message", {}).get("content", "")

                            # 清洗 Markdown 代码块
                            cleaned_text = text_content.strip().replace("```json", "").replace("```", "").strip()

                            try:
                                parsed_json = json.loads(cleaned_text)
                                output_file = self.output_folder / (Path(custom_id).stem + ".json")
                                self.save_json(parsed_json, output_file)

                                self.state[custom_id].update({
                                    "status": "completed",
                                    "output_path": str(output_file)
                                })
                                self.logger.info(f"    - ✅ {custom_id} 解析成功")
                            except json.JSONDecodeError:
                                self.state[custom_id].update({
                                    "status": "failed_parsing",
                                    "error": "模型输出非有效JSON"
                                })
                                self.logger.error(f"    - ❌ {custom_id} JSON解析失败")
                        else:
                            self.state[custom_id].update({"status": "failed_empty", "error": "模型返回为空"})
                    else:
                        # 单个请求失败
                        error_info = res.get("error") or response.get("body")
                        self.state[custom_id].update({
                            "status": "failed_request",
                            "error": str(error_info)
                        })
                        self.logger.error(f"    - ❌ {custom_id} 请求失败: {error_info}")

                except Exception as e:
                    self.logger.error(f"    - 解析结果行异常: {e}")

        except Exception as e:
            self.logger.error(f"❌ 下载/处理结果文件失败: {e}")

    # -------- 报告 --------
    def generate_report(self):
        """生成简报"""
        self.logger.info("=" * 50)
        self.logger.info("📋 最终处理报告")

        counts = {"completed": 0, "failed": 0, "pending": 0}
        for data in self.state.values():
            status = data.get("status", "")
            if status == "completed":
                counts["completed"] += 1
            elif "failed" in status:
                counts["failed"] += 1
            else:
                counts["pending"] += 1

        self.logger.info(f"✅ 成功: {counts['completed']}")
        self.logger.info(f"❌ 失败: {counts['failed']}")
        self.logger.info(f"⏳ 待定: {counts['pending']}")
        self.logger.info("=" * 50)


def main():
    try:
        config = load_config()
        setup_logging(config)
        parser = VLMPdfParser(config)

        parser.upload_files()
        parser.create_batch_jobs()
        parser.monitor_jobs()
        parser.generate_report()

    except Exception as e:
        logging.critical(f"程序运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()