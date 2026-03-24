# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google import genai
from google.genai import types
import yaml
import json
import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/extraction.log")
    log_file = PROJECT_ROOT / relative_log_path

    Path(log_file).parent.mkdir(exist_ok=True, parents=True)

    # 移除所有现有的处理器，以避免重复记录
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


# --- 准备批量请求文件 ---
def prepare_batch_requests(config: Dict[str, Any]) -> Path:
    """从文本块创建批量请求的 JSONL 文件"""
    extraction_config = config["extraction"]
    input_path = PROJECT_ROOT / extraction_config["input_dir"] / extraction_config["input_filename"]
    requests_dir = PROJECT_ROOT / extraction_config["requests_dir"]
    requests_dir.mkdir(exist_ok=True, parents=True)
    batch_request_path = requests_dir / "extraction_requests.jsonl"

    prompt_template = extraction_config["prompt"]

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, open(batch_request_path, 'w',
                                                                     encoding='utf-8') as outfile:
            count = 0
            for line in infile:
                chunk = json.loads(line)
                text_content = chunk.get("text", "")
                chunk_id = chunk.get("id", f"chunk-{count}")

                if not text_content:
                    logging.warning(f"跳过ID为 {chunk_id} 的空文本块")
                    continue

                # 构建 Gemini API 请求体
                request = {
                    "key": chunk_id,
                    "request": {
                        "system_instruction": {
                            "parts": [
                                {"text": prompt_template}
                            ]
                        },
                        "contents": [
                            {
                                "parts": [
                                    {"text": text_content}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "response_mime_type": "application/json",
                            "temperature": config["llm"].get("temperature", 0.1),
                            "topP": config["llm"].get("top_p", 0.9),
                        }
                    }
                }
                outfile.write(json.dumps(request, ensure_ascii=False) + "\n")
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

    # 1. 配置 Gemini 客户端
    try:
        # **优化点**: 优先从环境变量获取 API Key，增强安全性
        api_key = os.getenv("GEMINI_API_KEY", config.get("llm", {}).get("api_key"))
        if not api_key or "${" in api_key:
            raise ValueError("请在环境变量或 settings.yaml 中设置有效的 GEMINI_API_KEY")

        # **优化点**: 设置代理，提高网络访问稳定性
        proxy = config.get("proxy")
        if proxy:
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
            logging.info(f"已设置代理: {proxy}")

        client = genai.Client(api_key=api_key)
        logging.info("Gemini 客户端配置成功")
    except Exception as e:
        logging.error(f"配置 Gemini 客户端失败: {e}")
        return

    # 2. 准备并上传批量请求文件
    batch_request_path = prepare_batch_requests(config)
    logging.info(f"📤 正在上传批量请求文件: {batch_request_path.name}...")
    try:
        uploaded_file = client.files.upload(
            file=str(batch_request_path),
            config={
                "display_name": f'extraction-batch-{batch_request_path.stem}',
                "mime_type": 'application/jsonl'
            }
        )
        logging.info(f"✅ 文件上传成功: {uploaded_file.name}")
    except Exception as e:
        logging.error(f"❌ 文件上传失败: {e}")
        return

    # 3. 创建批量处理作业
    model_name = config["llm"]["model"]
    logging.info(f"🚀 正在使用模型 '{model_name}' 创建批量作业...")
    try:
        file_batch_job = client.batches.create(
            model=f"models/{model_name}",
            src=uploaded_file.name,
            config={
                'display_name': f"extraction-job-{batch_request_path.stem}",
            },
        )
        logging.info(f"✅ 批量作业已创建: {file_batch_job.name}")
    except Exception as e:
        logging.error(f"❌ 创建批量作业失败: {e}")
        return

    # 4. **优化点**: 借鉴 `gemini_ocr_batch.py` 的详细轮询逻辑
    job_name = file_batch_job.name
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    sleep_interval = config.get("vlm_parser", {}).get("sleep_interval", 300)  # 缩短轮询间隔

    logging.info(f"⏳ 开始轮询作业 '{job_name}' 状态，每 {sleep_interval} 秒检查一次...")
    while True:
        try:
            batch_job_status = client.batches.get(name=job_name)
            current_state = batch_job_status.state.name
            logging.info(f"  - 当前状态: {current_state}")
            if current_state in completed_states:
                break
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - 轮询失败: {e}")
            time.sleep(sleep_interval * 2)  # 发生错误时延长等待时间

    # 5. **优化点**: 借鉴 `gemini_ocr_batch.py` 的结果处理和错误分析逻辑
    if batch_job_status.state.name == 'JOB_STATE_SUCCEEDED':
        logging.info(f"✅ 作业成功完成！")
        try:
            result_file_name = batch_job_status.dest.file_name
            logging.info(f"📥 正在下载结果文件: {result_file_name}")
            file_content = client.files.download(file=result_file_name).decode('utf-8')

            extraction_config = config["extraction"]
            output_path = PROJECT_ROOT / extraction_config["output_dir"] / extraction_config["output_filename"]
            output_path.parent.mkdir(exist_ok=True, parents=True)

            processed_count = 0
            error_count = 0

            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in file_content.strip().split('\n'):
                    try:
                        result = json.loads(line)
                        chunk_id = result.get("key")

                        if chunk_id and "response" in result:
                            # 提取 response 中的文本内容
                            response_part = \
                            result["response"].get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0]
                            if "text" in response_part:
                                graph_text = response_part["text"]
                                # 将字符串形式的 JSON 再次解析
                                graph_data = json.loads(graph_text)
                                final_output = {"id": chunk_id, "graph": graph_data}
                                outfile.write(json.dumps(final_output, ensure_ascii=False) + "\n")
                                processed_count += 1
                            else:
                                logging.warning(f"  - ⚠️ ID '{chunk_id}' 的响应中没有找到 'text' 部分。")
                                error_count += 1
                        elif result.get("error"):
                            logging.error(f"  - ❌ 处理 ID '{chunk_id}' 时发生错误: {result['error']['message']}")
                            error_count += 1

                    except json.JSONDecodeError:
                        logging.warning(f"  - ⚠️ 无法解析结果行: {line}")
                        error_count += 1
                    except (KeyError, IndexError) as e:
                        logging.warning(f"  - ⚠️ 解析来自 ID '{chunk_id}' 的响应结构失败: {e}")
                        error_count += 1

            logging.info(f"🎉 结果处理完成！成功处理 {processed_count} 条，失败 {error_count} 条。")
            logging.info(f"💾 最终图谱数据已保存至: {output_path}")

        except Exception as e:
            logging.error(f"❌ 下载或处理结果文件时发生严重错误: {e}")
    else:
        logging.error(f"❌ 作业未能成功。最终状态: {batch_job_status.state.name}")
        if batch_job_status.error:
            logging.error(f"  - 错误详情: {batch_job_status.error}")



if __name__ == "__main__":
    run_extraction()