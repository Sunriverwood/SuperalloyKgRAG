import os
import json
import time
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from google.genai import types
from google.api_core import exceptions  # 引入特定的异常类型

from utils.client_factory import create_gemini_client

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
        raw_config = f.read()
        # 替换环境变量
        raw_config = raw_config.replace("${GEMINI_API_KEY}", os.environ.get("GEMINI_API_KEY", ""))
        config = yaml.safe_load(raw_config)

    logging.info("配置加载成功。")
    return config


class VLMPdfParser:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 VLM PDF 解析器

        Args:
            config: 配置字典
        """
        parser_cfg = config["vlm_parser"]
        llm_cfg = config["llm"]

        # 设置日志
        self.logger = logging.getLogger("SuperalloyKgRAG")

        # 目录和文件路径 (转换为绝对路径)
        self.pdf_folder = PROJECT_ROOT / parser_cfg["input_dir"]
        self.output_folder = PROJECT_ROOT / parser_cfg["output_dir"]
        self.state_file = PROJECT_ROOT / parser_cfg["state_file_path"]
        self.prompt_path = PROJECT_ROOT / parser_cfg.get("parsing_prompt_path")
        self.requests_path = PROJECT_ROOT / parser_cfg.get("requests_path")

        # 批处理配置
        self.batch_size = parser_cfg["batch_size"]
        self.timeout_seconds = parser_cfg.get("batch_polling_timeout_seconds")
        self.sleep_interval = parser_cfg.get("sleep_interval")

        # LLM配置
        self.model_name = llm_cfg["model"]
        self.api_key = llm_cfg["api_key"]
        self.proxy = config.get("proxy")

        self.client = create_gemini_client(
            api_key=self.api_key,
            proxy=self.proxy
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
                self.logger.info(f"✅ 成功加载状态文件，包含 {len(state)} 个文件记录")
                return state
            except Exception as e:
                self.logger.error(f"❌ 加载状态文件失败: {e}")
                return {}
        else:
            self.logger.info("未找到状态文件，将从空状态开始")
            return {}

    def _load_prompt(self) -> str:
        """加载PDF解析指令"""
        if not self.prompt_path.exists():
            error_msg = f"关键指令文件未找到: {self.prompt_path}"
            self.logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as f:
                instructions = f.read()
            self.logger.info(f"✅ 成功加载指令文件: {self.prompt_path}")
            return instructions
        except Exception as e:
            self.logger.critical(f"❌ 加载指令文件失败: {e}")
            raise

    def list_pdfs(self, pdf_folder: Path = None) -> set:
        """列出PDF文件夹中的所有PDF文件"""
        if pdf_folder is None:
            pdf_folder = self.pdf_folder
        return {f.name for f in pdf_folder.iterdir() if f.suffix.lower() == ".pdf"}

    def save_json(self, data: dict, path: Path):
        """保存JSON文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_state(self, state: dict, state_file: Path = None):
        """保存状态文件"""
        if state_file is None:
            state_file = self.state_file
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    # -------- 文件上传 --------
    def upload_files(self):
        """文件发现与上传"""
        self.logger.info("[阶段1] 文件发现与上传...")
        current_pdfs = self.list_pdfs()
        for pdf_file in current_pdfs:
            if pdf_file not in self.state:
                self.state[pdf_file] = {"status": "pending_upload"}
        self.save_state(self.state)

        for pdf_file, data in self.state.items():
            if data["status"] == "pending_upload":
                pdf_path = self.pdf_folder / pdf_file
                try:
                    self.logger.info(f"上传: {pdf_file}")
                    response = self.client.files.upload(file=str(pdf_path))
                    self.state[pdf_file].update({
                        "status": "uploaded",
                        "uploaded_file_uri": response.uri,
                        "uploaded_file_name": response.name
                    })
                except Exception as e:
                    self.state[pdf_file].update({"status": "failed_upload", "error": str(e)})
                    self.logger.error(f"上传失败 {pdf_file}: {e}")
                finally:
                    self.save_state(self.state)

    # -------- 批处理作业 --------
    def create_batch_jobs(self):
        """创建批处理作业"""
        self.logger.info("[阶段2] 创建批处理作业...")
        requests, files_for_jobs = [], []
        for pdf_file, data in self.state.items():
            if data["status"] == "uploaded":
                requests.append({
                    "key": pdf_file,
                    "request": {
                        "contents": [{"role": "user", "parts": [
                            {"text": self.instructions},
                            {"file_data": {"mime_type": "application/pdf", "file_uri": data["uploaded_file_uri"]}}
                        ]}],
                        "generationConfig": {"response_mime_type": "application/json"}
                    }
                })
                files_for_jobs.append(pdf_file)

        if not requests:
            self.logger.info("无待处理文件，无需创建新作业。")
            return

        # 确保 cache 目录存在
        requests_dir = self.requests_path.parent
        requests_dir.mkdir(exist_ok=True, parents=True)

        chunks = [requests[i:i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        files_chunks = [files_for_jobs[i:i + self.batch_size] for i in range(0, len(files_for_jobs), self.batch_size)]

        for i, (chunk, files_in_chunk) in enumerate(zip(chunks, files_chunks)):
            job_name = f"KG-Batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i + 1}"

            # # 为每个批次创建唯一的请求文件（使用时间戳）
            # timestamp = int(time.time())
            # batch_request_file = requests_dir / f"vlm_batch_requests_{timestamp}_{i}.jsonl"
            batch_request_file = self.requests_path

            try:
                # 写入批量请求文件
                with open(batch_request_file, "w", encoding="utf-8") as f:
                    for req in chunk:
                        f.write(json.dumps(req, ensure_ascii=False) + "\n")

                self.logger.info(f"  - 批量请求文件已保存: {batch_request_file}")

                # 上传批量请求文件
                batch_input = self.client.files.upload(
                    file=str(batch_request_file),
                    config=types.UploadFileConfig(display_name=job_name, mime_type="application/jsonl")
                )

                # 创建批处理作业
                job = self.client.batches.create(
                    model=self.model_name,
                    src=batch_input.name,
                    config={"display_name": job_name}
                )
                self.logger.info(f"✅ 作业创建成功: {job.name}")

                # 更新状态
                for pdf in files_in_chunk:
                    self.state[pdf].update({"status": "processing", "batch_job_name": job.name})

            except Exception as e:
                for pdf in files_in_chunk:
                    self.state[pdf].update({"status": "failed_job_creation", "error": str(e)})
                self.logger.error(f"❌ 批处理作业创建失败 {job_name}: {e}")
            finally:
                self.save_state(self.state)

    # -------- 作业监控 --------
    def monitor_jobs(self):
        """监控所有处理中的作业"""
        self.logger.info("[阶段3] 监控所有处理中的作业...")
        active_jobs = {d["batch_job_name"] for d in self.state.values() if d.get("status") == "processing"}

        if not active_jobs:
            self.logger.info("当前无活动作业需要监控。")
            return

        start_times = {name: datetime.now() for name in active_jobs}
        sleep_interval = self.sleep_interval

        while active_jobs:
            self.logger.info(f"正在监控 {len(active_jobs)} 个活动作业...")
            finished_jobs = set()

            for job_name in list(active_jobs):
                # 检查超时
                elapsed = datetime.now() - start_times.get(job_name, datetime.now())
                if elapsed.total_seconds() > self.timeout_seconds:
                    self.logger.warning(f"⏰ 作业 '{job_name}' 超时，正在尝试取消...")
                    try:
                        self.client.batches.cancel(name=job_name)
                    except exceptions.NotFound:
                        pass  # 作业可能已经结束或被删除

                    for pdf, data in self.state.items():
                        if data.get('batch_job_name') == job_name:
                            data.update({'status': 'failed_timeout', 'error': '批处理作业运行超时'})
                    finished_jobs.add(job_name)
                    continue

                # 获取作业状态
                try:
                    job = self.client.batches.get(name=job_name)
                    self.logger.info(f"  - 作业 '{job.name}' 当前状态: {job.state.name}")
                    if job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_EXPIRED', 'JOB_STATE_CANCELLED'):
                        self.logger.info(f"-> 作业 '{job.name}' 已完成，状态: {job.state.name}")
                        if job.state.name == 'JOB_STATE_SUCCEEDED':
                            self.process_job_results(job)
                        else:
                            error_detail = str(job.error) if job.error else f"作业以状态 {job.state.name} 结束"
                            for pdf, data in self.state.items():
                                if data.get('batch_job_name') == job_name:
                                    data.update({'status': f'failed_{job.state.name.lower()}', 'error': error_detail})
                        finished_jobs.add(job_name)

                except exceptions.NotFound:
                    self.logger.warning(f"⚠️ 作业 '{job_name}' 在API侧未找到，可能已被删除。将其标记为失败。")
                    for pdf, data in self.state.items():
                        if data.get('batch_job_name') == job_name:
                            data.update({'status': 'failed_job_not_found', 'error': '作业在API侧丢失'})
                    finished_jobs.add(job_name)
                except Exception as e:
                    self.logger.error(f"❌ 监控作业 '{job_name}' 时发生错误: {e}")

            if finished_jobs:
                active_jobs -= finished_jobs
                self.save_state(self.state)

            if active_jobs:
                self.logger.info(f"仍有 {len(active_jobs)} 个作业在运行中，将在 {sleep_interval // 60}分钟后再次检查...")
                time.sleep(sleep_interval)

    # -------- 结果处理 --------
    def process_job_results(self, job):
        """处理作业结果"""
        self.logger.info(f"  -> 正在处理作业 '{job.name}' 的结果...")
        if not (job.dest and job.dest.file_name):
            self.logger.error(f"❌ 错误：作业 '{job.name}' 成功，但未找到输出文件。")
            for pdf, data in self.state.items():
                if data.get('batch_job_name') == job.name:
                    data.update({'status': 'failed_job_no_output', 'error': '作业成功但无输出文件'})
            return

        result_file_name = job.dest.file_name
        try:
            self.logger.info(f"  - 📥 正在下载结果文件: {result_file_name}")
            content = self.client.files.download(file=result_file_name).decode("utf-8")

            for line in content.strip().split("\n"):
                result = json.loads(line)
                key = result.get("key")

                if not key or key not in self.state:
                    self.logger.warning(f"  - ⚠️ 警告：在结果文件中发现一个无效或不存在的 key '{key}'。")
                    continue

                if result.get("response"):
                    output_file = self.output_folder / (Path(key).stem + ".json")
                    try:
                        text = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
                        data = json.loads(cleaned)
                        self.save_json(data, output_file)
                        self.state[key].update({"status": "completed", "output_path": str(output_file)})
                        self.logger.info(f"    - ✅ 成功: '{key}' 的结果已保存到 {output_file}")
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        self.state[key].update({"status": "failed_parsing", "error": f"解析结果失败: {e}"})
                        self.logger.error(f"    - ❌ 失败: 解析 '{key}' 的结果时出错: {e}")
                elif result.get("error"):
                    error_message = result['error'].get('message', '未知API错误')
                    self.state[key].update({"status": "failed_in_job", "error": error_message})
                    self.logger.error(f"    - ❌ 失败: 处理 '{key}' 时API返回错误: {error_message}")
        except Exception as e:
            self.logger.critical(f"❌ 严重错误: 处理结果文件 '{result_file_name}' 时发生意外: {e}")
            for pdf, data in self.state.items():
                if data.get("batch_job_name") == job.name:
                    data.update({"status": "failed_processing_results", "error": str(e)})

    # -------- 报告 --------
    def generate_report(self):
        """生成最终处理报告"""
        self.logger.info("=" * 50)
        self.logger.info("📋 [阶段4] 最终处理报告")
        self.logger.info("=" * 50)

        successful_files = [f for f, data in self.state.items() if data.get('status') == 'completed']
        failed_files = [f for f, data in self.state.items() if 'failed' in data.get('status', '')]
        pending_files = [f for f, data in self.state.items() if
                         data.get('status') not in ['completed'] and 'failed' not in data.get('status', '')]

        self.logger.info(f"\n✅ 处理成功 ({len(successful_files)} 个文件):")
        if successful_files:
            for f in successful_files:
                self.logger.info(f"  - {f}")
        else:
            self.logger.info("  - 无")

        self.logger.error(f"\n❌ 处理失败 ({len(failed_files)} 个文件):")
        if failed_files:
            for f in failed_files:
                error = self.state[f].get('error', '未知错误')
                status = self.state[f].get('status', '未知状态')
                self.logger.error(f"  - {f} (状态: {status}, 原因: {error})")
        else:
            self.logger.info("  - 无")

        if pending_files:
            self.logger.info(f"\n⏳ 待处理/处理中 ({len(pending_files)} 个文件):")
            for f in pending_files:
                self.logger.info(f"  - {f} (状态: {self.state[f].get('status')})")

        self.logger.info("\n" + "=" * 50)


def main():
    """主执行函数"""
    try:
        # 加载配置
        config = load_config()

        # 设置日志
        setup_logging(config)

        # 创建解析器实例
        parser = VLMPdfParser(config)

        # 运行核心逻辑
        parser.upload_files()
        parser.create_batch_jobs()
        parser.monitor_jobs()
        parser.generate_report()

    except (FileNotFoundError, Exception) as e:
        logging.critical(f"程序启动或运行失败: {e}", exc_info=True)
        print(f"发生严重错误，请查看日志文件。错误: {e}")


if __name__ == "__main__":
    main()