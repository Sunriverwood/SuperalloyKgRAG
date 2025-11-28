# extraction.py
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml
# 修改：使用 OpenAI SDK
from openai import OpenAI

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/extraction.log")
    log_file = PROJECT_ROOT / relative_log_path

    Path(log_file).parent.mkdir(exist_ok=True, parents=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("日志记录器设置完成")


# --- 加载配置和 Prompt ---
def load_config_and_prompt(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件和抽取 Prompt"""
    try:
        settings_path = PROJECT_ROOT / settings_filename
        with open(settings_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("成功加载 settings.yaml 配置文件")

        relative_prompt_path = config.get("extraction", {}).get("prompt_path")
        if not relative_prompt_path:
            raise ValueError("配置文件中未找到 extraction.prompt_path")

        prompt_path = PROJECT_ROOT / relative_prompt_path
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config["extraction"]["prompt"] = f.read()
        logging.info(f"成功加载 Prompt 文件: {prompt_path}")

        return config
    except FileNotFoundError as e:
        logging.error(f"配置文件或 Prompt 文件未找到: {e}")
        raise
    except Exception as e:
        logging.error(f"加载配置时出错: {e}")
        raise


# --- 准备批量请求文件 (修改为 OpenAI 格式) ---
def prepare_batch_requests(config: Dict[str, Any]) -> Path:
    """从文本块创建批量请求的 JSONL 文件 (OpenAI 兼容格式)"""
    extraction_config = config["extraction"]
    input_path = PROJECT_ROOT / extraction_config["input_dir"] / extraction_config["input_filename"]
    requests_dir = PROJECT_ROOT / extraction_config["requests_dir"]
    requests_dir.mkdir(exist_ok=True, parents=True)
    batch_request_path = requests_dir / "extraction_requests.jsonl"

    prompt_template = extraction_config["prompt"]
    model_name = config["llm"]["model"]  # 确保是阿里云支持的模型名，如 qwen-plus

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, open(batch_request_path, 'w',
                                                                     encoding='utf-8') as outfile:
            count = 0
            for line in infile:
                chunk = json.loads(line)
                text_content = chunk.get("text", "")
                # custom_id 必须为字符串
                chunk_id = str(chunk.get("id", f"chunk-{count}"))

                if not text_content:
                    logging.warning(f"跳过ID为 {chunk_id} 的空文本块")
                    continue

                # 修改：构建 OpenAI Batch API 请求体
                # 注意：System Prompt 中必须包含 "JSON" 以启用 json_object 模式 [cite: 530]
                messages = [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": text_content}
                ]

                request_line = {
                    "custom_id": chunk_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": messages,
                        "temperature": config["llm"].get("temperature", 0.1),
                        # top_p 参数名调整 (Gemini topP -> OpenAI top_p)
                        "top_p": config["llm"].get("top_p", 0.9),
                        # 启用 JSON 模式 [cite: 529]
                        "response_format": {"type": "json_object"}
                    }
                }
                outfile.write(json.dumps(request_line, ensure_ascii=False) + "\n")
                count += 1
        logging.info(f"成功创建批量请求文件，共 {count} 条记录: {batch_request_path}")
        return batch_request_path
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {input_path}")
        raise
    except Exception as e:
        logging.error(f"准备批量请求文件时出错: {e}")
        raise


# --- 主执行函数 ---
def run_extraction():
    """执行完整的数据抽取流程"""
    config = load_config_and_prompt(settings_filename='config/settings.yaml')
    setup_logging(config)

    # 1. 配置 OpenAI 客户端 (阿里云百炼)
    try:
        # 修改：优先读取 QWEN_API_KEY，不设置代理
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("GEMINI_API_KEY") or config.get("llm", {}).get("api_key")
        if not api_key:
            raise ValueError("请在环境变量 QWEN_API_KEY 或 settings.yaml 中设置有效的 API Key")

        # 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        logging.info("阿里云百炼 (OpenAI兼容) 客户端配置成功")
    except Exception as e:
        logging.error(f"配置客户端失败: {e}")
        return

    # 2. 准备并上传批量请求文件
    try:
        batch_request_path = prepare_batch_requests(config)
    except Exception:
        return

    logging.info(f"📤 正在上传批量请求文件: {batch_request_path.name}...")
    try:
        # 修改：使用 client.files.create
        with open(batch_request_path, "rb") as file_obj:
            uploaded_file = client.files.create(
                file=file_obj,
                purpose="batch"
            )
        logging.info(f"✅ 文件上传成功: {uploaded_file.id}")
    except Exception as e:
        logging.error(f"❌ 文件上传失败: {e}")
        return

    # 3. 创建批量处理作业
    logging.info(f"🚀 正在创建批量作业...")
    try:
        # 修改：使用 client.batches.create
        # endpoint 必须为 /v1/chat/completions
        # completion_window="24h" 为必填
        file_batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                'description': f"extraction-job-{batch_request_path.stem}",
            }
        )
        logging.info(f"✅ 批量作业已创建: {file_batch_job.id}")
    except Exception as e:
        logging.error(f"❌ 创建批量作业失败: {e}")
        return

    # 4. 轮询作业状态
    job_id = file_batch_job.id
    # 阿里云 Batch 状态集 [cite: 323]
    completed_states = {'completed', 'failed', 'cancelled', 'expired'}
    sleep_interval = config.get("vlm_parser", {}).get("sleep_interval", 60)  # 建议根据实际情况调整

    logging.info(f"⏳ 开始轮询作业 '{job_id}' 状态，每 {sleep_interval} 秒检查一次...")

    batch_job_status = None
    while True:
        try:
            # 修改：使用 client.batches.retrieve
            batch_job_status = client.batches.retrieve(batch_id=job_id)
            current_state = batch_job_status.status
            logging.info(f"  - 当前状态: {current_state}")
            if current_state in completed_states:
                break
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - 轮询失败: {e}")
            time.sleep(sleep_interval * 2)

    # 5. 结果处理
    if batch_job_status and batch_job_status.status == 'completed':
        logging.info(f"✅ 作业成功完成！")
        try:
            output_file_id = batch_job_status.output_file_id
            if not output_file_id:
                logging.warning("作业完成但没有 output_file_id")
                return

            logging.info(f"📥 正在下载结果文件: {output_file_id}")
            # 修改：使用 client.files.content
            file_content = client.files.content(output_file_id).text

            extraction_config = config["extraction"]
            output_path = PROJECT_ROOT / extraction_config["output_dir"] / extraction_config["output_filename"]
            output_path.parent.mkdir(exist_ok=True, parents=True)

            processed_count = 0
            error_count = 0

            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in file_content.strip().split('\n'):
                    try:
                        result = json.loads(line)
                        chunk_id = result.get("custom_id")
                        response = result.get("response", {})

                        if chunk_id and response.get("status_code") == 200:
                            # 提取 response 中的文本内容 (OpenAI 格式)
                            # 路径: response -> body -> choices[0] -> message -> content
                            body = response.get("body", {})
                            choices = body.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content", "")
                                if content:
                                    # 尝试解析 JSON 字符串 (因为启用了 json_object 模式)
                                    try:
                                        graph_data = json.loads(content)
                                    except json.JSONDecodeError:
                                        # 如果返回的不是纯 JSON，则保留原文本或记录警告
                                        logging.warning(f"  - ⚠️ ID '{chunk_id}' 的内容不是有效 JSON")
                                        graph_data = {"raw_content": content}

                                    final_output = {"id": chunk_id, "graph": graph_data}
                                    outfile.write(json.dumps(final_output, ensure_ascii=False) + "\n")
                                    processed_count += 1
                                else:
                                    logging.warning(f"  - ⚠️ ID '{chunk_id}' 的响应内容为空。")
                                    error_count += 1
                            else:
                                logging.warning(f"  - ⚠️ ID '{chunk_id}' 没有 choices。")
                                error_count += 1
                        else:
                            # 处理错误
                            error_info = result.get("error") or response.get("body")
                            logging.error(f"  - ❌ 处理 ID '{chunk_id}' 时发生错误: {error_info}")
                            error_count += 1

                    except json.JSONDecodeError:
                        logging.warning(f"  - ⚠️ 无法解析结果行: {line[:100]}...")
                        error_count += 1
                    except Exception as e:
                        logging.warning(f"  - ⚠️ 处理 ID '{chunk_id}' 时发生未知错误: {e}")
                        error_count += 1

            logging.info(f"🎉 结果处理完成！成功处理 {processed_count} 条，失败 {error_count} 条。")
            logging.info(f"💾 最终图谱数据已保存至: {output_path}")

        except Exception as e:
            logging.error(f"❌ 下载或处理结果文件时发生严重错误: {e}")
    else:
        status = getattr(batch_job_status, 'status', 'Unknown')
        logging.error(f"❌ 作业未能成功。最终状态: {status}")
        # 尝试获取错误信息
        if hasattr(batch_job_status, 'errors') and batch_job_status.errors:
            logging.error(f"  - 错误详情: {batch_job_status.errors}")


if __name__ == "__main__":
    run_extraction()