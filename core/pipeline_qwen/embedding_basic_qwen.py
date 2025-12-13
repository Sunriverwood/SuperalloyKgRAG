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
# 修改：使用 OpenAI SDK
from openai import OpenAI


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


# --- 批量向量化工作流 (修改为 OpenAI Batch 格式) ---
def create_embedding_requests(documents: List[Dict], model_name: str, output_path: Path, dimensionality: int):
    """为批量嵌入创建请求并写入本地 JSONL 文件 (OpenAI 兼容格式)。"""
    logging.info(f"正在为 {len(documents)} 个文档创建批量嵌入请求...")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            # 确保每个文档都有 'id' 和 'text' 字段
            if "id" not in doc or "text" not in doc:
                logging.warning(f"文档缺少 'id' 或 'text' 字段，已跳过: {doc}")
                continue

            # 修改：构建兼容 OpenAI Batch API 的请求体
            request_line = {
                "custom_id": str(doc["id"]),  # 确保是字符串
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model_name,
                    "input": doc["text"],
                    "dimensions": dimensionality  # text-embedding-v3 支持
                }
            }
            f.write(json.dumps(request_line, ensure_ascii=False) + '\n')
    logging.info(f"嵌入请求已写入到 {output_path}")


def submit_and_monitor_embedding_job(client: OpenAI, requests_path: Path, model_name: str,
                                     sleep_interval: int) -> Any:
    """上传文件，创建并监控批量嵌入作业 (OpenAI SDK)。"""
    logging.info(f"📤 正在上传请求文件: {requests_path.name}...")
    try:
        # 修改：使用 client.files.create
        with open(requests_path, "rb") as file_obj:
            uploaded_file = client.files.create(
                file=file_obj,
                purpose="batch"
            )
        logging.info(f"✅ 文件上传成功: {uploaded_file.id}")
    except Exception as e:
        logging.error(f"❌ 文件上传失败: {e}")
        return None

    logging.info("🚀 正在创建批量嵌入作业...")
    try:
        # 修改：使用 client.batches.create， endpoint 必须为 /v1/embeddings
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",  # 阿里云必填
            metadata={"description": f"embedding-job-{requests_path.stem}"}
        )
        logging.info(f"✅ 批量作业已创建: {batch_job.id}")
    except Exception as e:
        logging.error(f"❌ 创建批量作业失败: {e}")
        # OpenAI SDK 不直接支持按名称删除文件，需通过 ID 删除，这里略过清理步骤或需记录 ID
        return None

    job_id = batch_job.id
    # 阿里云 Batch 状态集
    completed_states = {'completed', 'failed', 'cancelled', 'expired'}

    if sleep_interval == 0:
        logging.info("sleep_interval=0，跳过轮询，直接返回作业对象。")
        return batch_job
    else:
        logging.info(f"⏳ [Embedding] 开始轮询作业 '{job_id}' 状态，每 {sleep_interval} 秒检查一次...")
        while True:
            try:
                # 修改：使用 client.batches.retrieve
                batch_job = client.batches.retrieve(batch_id=job_id)
                current_state = batch_job.status
                logging.info(f"  - [Embedding] 当前状态: {current_state}")
                if current_state in completed_states:
                    break
                time.sleep(sleep_interval)
            except Exception as e:
                logging.error(f"  - [Embedding] 轮询失败: {e}")
                time.sleep(sleep_interval * 2)
        return batch_job


def process_embedding_results(batch_job: Any, client: OpenAI, original_docs: List[Dict]) -> List[Dict]:
    """下载、处理并归一化批量嵌入作业的结果。"""
    if not batch_job or batch_job.status != 'completed':
        logging.error("❌ 作业失败或未执行，无法处理结果。")
        if hasattr(batch_job, 'errors') and batch_job.errors:
            logging.error(f"  - 失败原因: {batch_job.errors}")
        return []

    job_id = batch_job.id
    output_file_id = batch_job.output_file_id

    if not output_file_id:
        logging.error("❌ 作业完成但没有 output_file_id。")
        return []

    logging.info(f"📥 [{job_id}] 正在下载结果文件: {output_file_id}")
    try:
        # 修改：使用 client.files.content 下载内容
        file_content = client.files.content(output_file_id).text
    except Exception as e:
        logging.error(f"❌ 下载结果失败: {e}")
        return []

    output_lines = file_content.strip().split('\n')

    # 构建 custom_id -> embedding 的映射
    result_map = {}
    error_count = 0

    for line in output_lines:
        try:
            res_json = json.loads(line)
            c_id = res_json.get("custom_id")
            response = res_json.get("response", {})

            if response.get("status_code") == 200:
                # 解析 OpenAI Embedding 响应
                # 结构: response -> body -> data -> [ {embedding: ...} ]
                body = response.get("body", {})
                data_list = body.get("data", [])
                if data_list and "embedding" in data_list[0]:
                    result_map[c_id] = data_list[0]["embedding"]
                else:
                    error_count += 1
            else:
                logging.warning(f"请求失败 custom_id={c_id}: {response}")
                error_count += 1
        except Exception as e:
            logging.warning(f"解析结果行失败: {e}")
            error_count += 1

    embedded_docs = []

    for doc in original_docs:
        # custom_id 在 create_embedding_requests 中被转为字符串
        doc_id = str(doc["id"])

        if doc_id in result_map:
            embedding_values = np.array(result_map[doc_id])
            # 归一化 (阿里云 v3/v4 通常已归一化，保留以防万一)
            norm = np.linalg.norm(embedding_values)
            if norm > 0:
                normed_embedding = embedding_values / norm
            else:
                normed_embedding = embedding_values

            doc["vector"] = normed_embedding.tolist()
            embedded_docs.append(doc)
        else:
            # 可能是该行请求失败
            pass

    logging.info(
        f"🎉 [{job_id}] 结果处理完成！成功获取并归一化 {len(embedded_docs)} 个向量，失败/丢失 {len(original_docs) - len(embedded_docs)} 个。")
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

    # 确保只包含需要的字段，防止 LanceDB 类型推断错误
    # 这里假设 embedded_data 已经包含 'vector' 和原始 doc 的字段
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

    # --- 1. 初始化客户端 (修改：使用 OpenAI SDK) ---
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    if not api_key:
        logging.error("未找到有效的 API Key")
        return

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logging.info("阿里云百炼 (OpenAI兼容) 客户端初始化完成。")

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