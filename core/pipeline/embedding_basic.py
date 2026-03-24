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

# embedding_basic.py

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import lancedb
from google import genai
from google.genai import types

from utils.client_factory import create_gemini_client

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/embedding_basic.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)
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


# --- 加载配置 ---
def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    logging.info(f"正在从 {config_path} 加载配置...")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info("配置加载成功。")
    return config


# --- 数据加载 ---
def load_text_units(file_path: Path) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载文本单元。"""
    logging.info(f"正在从 {file_path} 加载文本单元...")
    if not file_path.exists():
        raise FileNotFoundError(f"输入文件 {file_path} 未找到！")
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                # 假设每行都是一个有效的JSON对象
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"跳过无法解析的行 {i + 1}: {line.strip()}")
    logging.info(f"成功加载 {len(documents)} 个文本单元。")
    return documents


# --- 批量向量化工作流 (与原脚本相同) ---
def create_embedding_requests(documents: List[Dict], model_name: str, output_path: Path, dimensionality: int):
    """为批量嵌入创建请求并写入本地 JSONL 文件。"""
    logging.info(f"正在为 {len(documents)} 个文档创建批量嵌入请求...")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            # 确保每个文档都有 'id' 和 'text' 字段
            if "id" not in doc or "text" not in doc:
                logging.warning(f"文档缺少 'id' 或 'text' 字段，已跳过: {doc}")
                continue

            request_body = {
                "content": {"parts": [{"text": doc["text"]}]},
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": dimensionality
            }

            request_line = {
                "key": doc["id"],
                "request": request_body
            }
            f.write(json.dumps(request_line, ensure_ascii=False) + '\n')
    logging.info(f"嵌入请求已写入到 {output_path}")


def submit_and_monitor_embedding_job(client: genai.Client, requests_path: Path, model_name: str,
                                     sleep_interval: int) -> Any:
    """上传文件，创建并监控批量嵌入作业。"""
    logging.info(f"📤 正在上传请求文件: {requests_path.name}...")
    try:
        uploaded_file = client.files.upload(
            file=str(requests_path),
            config={
                "display_name": f'embedding-batch-{requests_path.stem}',
                "mime_type": 'application/jsonl'
            }
        )
        logging.info(f"✅ 文件上传成功: {uploaded_file.name}")
    except Exception as e:
        logging.error(f"❌ 文件上传失败: {e}")
        return None

    logging.info("🚀 正在创建批量嵌入作业...")
    try:
        batch_job = client.batches.create_embeddings(
            model=f"{model_name}",
            src={'file_name': uploaded_file.name},
            config={'display_name': f"embedding-job-{requests_path.stem}"},
        )
        logging.info(f"✅ 批量作业已创建: {batch_job.name}")
    except Exception as e:
        logging.error(f"❌ 创建批量作业失败: {e}")
        client.files.delete(name=uploaded_file.name)
        return None

    job_name = batch_job.name
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    if sleep_interval == 0:
        logging.info("sleep_interval=0，跳过轮询，直接返回作业对象。")
        return batch_job
    else:
        logging.info(f"⏳ [Embedding] 开始轮询作业 '{job_name}' 状态，每 {sleep_interval} 秒检查一次...")
        while True:
            try:
                batch_job = client.batches.get(name=job_name)
                current_state = batch_job.state.name
                logging.info(f"  - [Embedding] 当前状态: {current_state}")
                if current_state in completed_states:
                    break
                time.sleep(sleep_interval)
            except Exception as e:
                logging.error(f"  - [Embedding] 轮询失败: {e}")
                time.sleep(sleep_interval * 2)
        return batch_job


def process_embedding_results(batch_job: Any, client: genai.Client, original_docs: List[Dict]) -> List[Dict]:
    """下载、处理并归一化批量嵌入作业的结果。"""
    if not batch_job or batch_job.state != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 作业失败或未执行，无法处理结果。")
        if batch_job and batch_job.error: logging.error(f"  - 失败原因: {batch_job.error.message}")
        return []

    job_id = batch_job.name.split('/')[-1]
    logging.info(f"📥 [{job_id}] 正在下载结果文件: {batch_job.dest.file_name}")
    file_content = client.files.download(file=batch_job.dest.file_name).decode('utf-8')

    output_lines = file_content.strip().split('\n')
    if len(output_lines) != len(original_docs):
        logging.error(f"❌ [{job_id}] 结果数量与原始文档数量不匹配！"
                      f"原始文档: {len(original_docs)}, 返回结果: {len(output_lines)}")

    embedded_docs = []
    error_count = 0
    for i, line in enumerate(output_lines):
        try:
            if i >= len(original_docs):
                break
            result = json.loads(line)
            if "response" in result and "embedding" in result["response"]:
                embedding_values = np.array(result["response"]["embedding"]["values"])
                normed_embedding = embedding_values / np.linalg.norm(embedding_values)
                doc = original_docs[i]
                doc["vector"] = normed_embedding.tolist()
                embedded_docs.append(doc)
            else:
                error_msg = result.get("error", {}).get("message", "未知错误")
                logging.error(f"  - ❌ 文档索引 {i} (ID: {original_docs[i]['id']}) 嵌入失败: {error_msg}")
                error_count += 1
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logging.warning(f"  - ⚠️ 解析或处理结果行 {i + 1} 失败: '{line[:100]}...', 错误: {e}")
            error_count += 1

    logging.info(f"🎉 [{job_id}] 结果处理完成！成功获取并归一化 {len(embedded_docs)} 个向量，失败 {error_count} 个。")
    return embedded_docs


# --- 向量存储函数 ---
def store_embeddings_lancedb(db_path: Path, table_name: str, embedded_data: List[Dict]):
    """将嵌入数据存储到LanceDB表中"""
    if not embedded_data:
        logging.warning(f"没有可用于存储到表 '{table_name}' 的数据。")
        return

    logging.info(f"正在将 {len(embedded_data)} 条记录存入 LanceDB 表 '{table_name}'...")
    db_path.mkdir(exist_ok=True, parents=True)
    db = lancedb.connect(db_path)
    df = pd.DataFrame(embedded_data)

    try:
        if table_name in db.table_names():
            logging.warning(f"表 '{table_name}' 已存在，将被覆盖。")
            db.drop_table(table_name)
        db.create_table(table_name, data=df)
        logging.info(f"✅ LanceDB 表 '{table_name}' 创建并写入成功。")
    except Exception as e:
        logging.error(f"❌ 写入 LanceDB 表 '{table_name}' 时失败: {e}")


# --- 主执行函数 ---
def main():
    """主执行流程"""
    config = load_config()
    setup_logging(config)

    # --- 1. 初始化客户端 ---
    api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    proxy = config.get("proxy")
    client = create_gemini_client(api_key, proxy)
    logging.info("Gemini 客户端初始化完成。")

    # --- 2. 加载数据 ---
    text_units_path = PROJECT_ROOT / config["embedding"]["input_text_units_path"]
    documents_to_embed = load_text_units(text_units_path)

    if not documents_to_embed:
        logging.warning("未能加载任何文本单元，程序即将退出。")
        return

    # --- 3. 完整嵌入工作流 ---
    logging.info("\n--- 开始处理文本块嵌入 ---")

    # 从配置中提取参数
    embedding_config = config["embedding"]
    model_name = embedding_config["model"]
    dimensionality = embedding_config["dimensionality"]
    sleep_interval = embedding_config["sleep_interval"]
    requests_dir = PROJECT_ROOT / embedding_config["requests_path"]
    requests_path = requests_dir / "embedding_text_units_requests.jsonl"
    db_path = PROJECT_ROOT / embedding_config["output_db_text_path"]

    table_name = embedding_config.get("output_text_table_name", "text")

    # a. 创建批量请求
    create_embedding_requests(documents_to_embed, model_name, requests_path, dimensionality)

    # b. 提交作业并监控
    final_batch_job_status = submit_and_monitor_embedding_job(client, requests_path, model_name, sleep_interval)

    # c. 处理结果
    embedded_documents = process_embedding_results(final_batch_job_status, client, documents_to_embed)

    # d. 存储到LanceDB
    if embedded_documents:
        store_embeddings_lancedb(db_path, table_name, embedded_documents)
        logging.info("\n✅ 所有文本块已成功嵌入并存入向量数据库！")
    else:
        logging.error("\n❌ 未能获取任何有效的向量，存储过程被跳过。")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)