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

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import networkx as nx
import pandas as pd
import yaml
import lancedb
from openai import OpenAI
import concurrent.futures
from utils.clean_embedding_text import EmbeddingTextCleaner

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/embedding.log")
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


# --- 文本清洁函数 ---
# 保留一个轻量包装器以便未来可能的统一调用
_text_cleaner_instance: EmbeddingTextCleaner | None = None


def init_text_cleaner(graph_path: Path, id_maps_path: Path) -> EmbeddingTextCleaner:
    global _text_cleaner_instance
    if _text_cleaner_instance is None:
        _text_cleaner_instance = EmbeddingTextCleaner(
            final_graph_path=graph_path,
            id_maps_path=id_maps_path
        )
    return _text_cleaner_instance


def clean_text_for_embedding(text: str, community_id: str = "") -> str:
    if _text_cleaner_instance is None:
        if not isinstance(text, str):
            return ""
        tmp = re.sub(r'\[Data:.*?]', '', text)
        tmp = re.sub(r'\s*\([ER]\d+\)', '', tmp)
        return re.sub(r'\s+', ' ', tmp).strip()
    return _text_cleaner_instance.clean_text(text, community_id)


# --- 数据加载与准备 (保持不变) ---

def load_graph_data(graph_path: Path) -> nx.DiGraph:
    """仅加载最终图谱"""
    logging.info(f"正在从 {graph_path} 加载图谱...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    graph = nx.node_link_graph(graph_data)
    logging.info("图谱加载成功。")
    return graph


def load_and_prepare_community_data(community_report_path: Path, text_cleaner: EmbeddingTextCleaner) -> List[Dict]:
    """加载并准备社区数据"""
    logging.info(f"[社区] 正在从 {community_report_path} 加载和准备社区数据...")
    communities_to_embed = []
    if not community_report_path.exists():
        logging.error(f"[社区] 社区报告文件未找到: {community_report_path}")
        return []
    with open(community_report_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                community_id = str(data["community_id"])
                report = data["report"]
                local_id_map = data.get("local_id_map", {})
                clean_title = text_cleaner.clean_text(report.get("title", ""), community_id)
                clean_summary = text_cleaner.clean_text(report.get("summary", ""), community_id)
                clean_findings_texts = []
                for finding in report.get("findings", []):
                    clean_f_summary = text_cleaner.clean_text(finding.get("summary", ""), community_id)
                    clean_f_explanation = text_cleaner.clean_text(finding.get("explanation", ""), community_id)
                    clean_findings_texts.append(f"{clean_f_summary} {clean_f_explanation}")
                full_clean_text = (f"Title: {clean_title}\nAbstract: {clean_summary}\nfindings:\n" + "\n".join(
                    clean_findings_texts))
                communities_to_embed.append({
                    "id": f"community_{community_id}",
                    "text_to_embed": full_clean_text,
                    "payload_report_json": json.dumps(report, ensure_ascii=False),
                    "payload_map_json": json.dumps(local_id_map, ensure_ascii=False)
                })
            except Exception:
                pass
    logging.info(f"[社区] 准备了 {len(communities_to_embed)} 个社区待嵌入。")
    return communities_to_embed


def prepare_embedding_data(graph: nx.DiGraph, community_report_path: Path, text_cleaner: EmbeddingTextCleaner) -> Dict[
    str, List[Dict]]:
    """准备所有层级的数据"""
    logging.info("正在准备三个的嵌入数据...")
    communities_to_embed = load_and_prepare_community_data(community_report_path, text_cleaner)
    logging.info("[实体] 正在准备实体数据...")
    entities_to_embed = []
    for node_id, data in graph.nodes(data=True):
        if data.get("is_disambiguated"):
            payload_data = data.copy()
            text_source = payload_data.pop("description", data.get("name", ""))
            text_to_embed = text_cleaner.clean_text(text_source, community_id="")
            entities_to_embed.append({
                "id": str(node_id),
                "text_to_embed": text_to_embed,
                "payload_node_data_json": json.dumps(payload_data, ensure_ascii=False)
            })
    logging.info(f"[实体] 准备了 {len(entities_to_embed)} 个实体待嵌入。")
    logging.info("[关系] 正在准备关系数据...")
    relationships_to_embed = []
    for u, v, data in graph.edges(data=True):
        source_name = graph.nodes[u].get("name", "未知实体")
        target_name = graph.nodes[v].get("name", "未知实体")
        rel_desc = data.get("description", "未知关系")
        triplet_text = f"{source_name} -> [{rel_desc}] -> {target_name}"
        payload_data = data.copy()
        payload_data["source_id"] = str(u)
        payload_data["target_id"] = str(v)
        relationships_to_embed.append({
            "id": str(data["id"]),
            "text_to_embed": triplet_text,
            "payload_edge_data_json": json.dumps(payload_data, ensure_ascii=False)
        })
    logging.info(f"[关系] 准备了 {len(relationships_to_embed)} 个关系待嵌入。")
    return {"communities": communities_to_embed, "entities": entities_to_embed, "relationships": relationships_to_embed}


# --- 批量向量化工作流 (修改重点) ---

def create_embedding_requests(documents: List[Dict], model_name: str, output_path: Path, dimensionality: int,
                              data_type: str = ""):
    """
    修改：为阿里云百炼(OpenAI兼容)批量嵌入创建请求并写入本地 JSONL 文件。
    参考:
    """
    level_tag = f"[{data_type.upper()}] " if data_type else ""
    logging.info(f"{level_tag}正在为 {len(documents)} 个文档创建批量嵌入请求...")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            text_for_embedding = doc.get("text_to_embed", "")
            if not text_for_embedding:
                logging.warning(f"{level_tag}文档 {doc['id']} 缺少 'text_to_embed' 内容，跳过。")
                continue

            # 修改：构建兼容 OpenAI Batch API 的请求体
            # 注意：Embedding 任务的 url 必须是 /v1/embeddings
            request_line = {
                "custom_id": doc["id"],
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model_name,
                    "input": text_for_embedding,
                    "dimensions": dimensionality  # 支持 text-embedding-v3/v4 [cite: 288]
                }
            }
            f.write(json.dumps(request_line, ensure_ascii=False) + '\n')
    logging.info(f"{level_tag}嵌入请求已写入到 {output_path}")


def submit_and_monitor_embedding_job(client: OpenAI, requests_path: Path, model_name: str, sleep_interval: int,
                                     data_type: str = "", batch_index: int = 0, total_batches: int = 1) -> Any:
    """
    修改：使用 OpenAI SDK 上传文件并监控阿里云百炼 Batch 任务。
    参考: [cite: 1344, 1375, 1446]
    """
    level_tag = f"[{data_type.upper()}] " if data_type else ""
    batch_tag = f"[批次 {batch_index + 1}/{total_batches}] " if total_batches > 1 else ""

    logging.info(f"{level_tag}{batch_tag}📤 正在上传请求文件: {requests_path.name}...")
    try:
        # 修改：使用 client.files.create 上传文件，purpose 必须为 'batch' [cite: 1356]
        with open(requests_path, "rb") as file_to_upload:
            uploaded_file = client.files.create(
                file=file_to_upload,
                purpose="batch"
            )
        logging.info(f"{level_tag}{batch_tag}✅ 文件上传成功: {uploaded_file.id}")
    except Exception as e:
        logging.error(f"{level_tag}{batch_tag}❌ 文件上传失败: {e}")
        return None

    logging.info(f"{level_tag}{batch_tag}🚀 正在创建批量嵌入作业...")
    try:
        # 修改：使用 client.batches.create 创建任务
        # endpoint 必须设为 /v1/embeddings [cite: 1387]
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",  # 阿里云要求必填，支持 24h [cite: 1391]
            metadata={"ds_name": f"embed-{data_type}-{batch_index}"}  # 可选：添加任务名称
        )
        logging.info(f"{level_tag}{batch_tag}✅ 批量作业已创建: {batch_job.id}")
    except Exception as e:
        logging.error(f"{level_tag}{batch_tag}❌ 创建批量作业失败: {e}")
        return None

    job_id = batch_job.id
    # 阿里云 Batch 状态：validating, failed, in_progress, finalizing, completed, expired, cancelling, cancelled [cite: 1427]
    completed_states = {'completed', 'failed', 'cancelled', 'expired'}

    if sleep_interval == 0:
        logging.info(f"{level_tag}{batch_tag}sleep_interval=0，跳过轮询，直接返回作业对象。")
        return batch_job

    else:
        logging.info(f"{level_tag}{batch_tag}⏳ 开始轮询作业 '{job_id}' 状态，每 {sleep_interval} 秒检查一次...")
        while True:
            try:
                # 修改：使用 client.batches.retrieve 查询状态 [cite: 1458]
                batch_job_status = client.batches.retrieve(batch_id=job_id)
                current_state = batch_job_status.status
                logging.info(f"{level_tag}{batch_tag}  当前状态: {current_state}")

                if current_state in completed_states:
                    break
                time.sleep(sleep_interval)
            except Exception as e:
                logging.error(f"{level_tag}{batch_tag}  轮询失败: {e}")
                time.sleep(sleep_interval * 2)
        return batch_job_status


def process_embedding_results(batch_job: Any, client: OpenAI, original_docs: List[Dict], data_type: str = "") -> List[
    Dict]:
    """
    修改：下载并处理阿里云百炼 Batch 结果。
    结果格式是 JSONL，每行包含 response.body.data[0].embedding
    """
    level_tag = f"[{data_type.upper()}] " if data_type else ""

    if not batch_job or batch_job.status != 'completed':
        logging.error(f"{level_tag}❌ 作业失败或未执行，无法处理结果。状态: {getattr(batch_job, 'status', 'Unknown')}")
        if hasattr(batch_job, 'errors') and batch_job.errors:
            logging.error(f"{level_tag}  失败详情: {batch_job.errors}")
        return []

    output_file_id = batch_job.output_file_id
    if not output_file_id:
        logging.error(f"{level_tag}❌ 作业完成但没有 output_file_id。")
        return []

    logging.info(f"{level_tag}📥 正在下载结果文件: {output_file_id}")
    try:
        # 修改：使用 client.files.content 下载结果 [cite: 1562]
        file_content = client.files.content(output_file_id).text
    except Exception as e:
        logging.error(f"{level_tag}❌ 下载结果失败: {e}")
        # 尝试重试一次
        logging.info(f"{level_tag}⏳ 等待5秒后重试下载...")
        time.sleep(5)
        try:
            file_content = client.files.content(output_file_id).text
            logging.info(f"{level_tag}✅ 重试下载成功")
        except Exception as retry_error:
            logging.error(f"{level_tag}❌ 重试下载仍然失败: {retry_error}")
            return []

    output_lines = file_content.strip().split('\n')

    # 建立 custom_id 到 result 的映射，因为 Batch 结果顺序可能不保证一致
    result_map = {}
    error_count = 0

    for line in output_lines:
        try:
            res_json = json.loads(line)
            c_id = res_json.get("custom_id")
            response = res_json.get("response", {})

            if response.get("status_code") == 200:
                # 解析 OpenAI 格式的 Embedding 响应
                # 结构: response -> body -> data -> [ {embedding: ...} ]
                body = response.get("body", {})
                data_list = body.get("data", [])
                if data_list and "embedding" in data_list[0]:
                    embedding = data_list[0]["embedding"]
                    result_map[c_id] = embedding
                else:
                    error_count += 1
            else:
                logging.warning(f"{level_tag} 请求失败: {c_id}, 状态码: {response.get('status_code')}")
                error_count += 1
        except Exception as e:
            logging.error(f"{level_tag} 解析结果行失败: {e}")
            error_count += 1

    embedded_docs = []

    for doc in original_docs:
        doc_id = doc["id"]
        if doc_id in result_map:
            embedding_values = np.array(result_map[doc_id])
            # 阿里云 text-embedding-v3/v4 通常返回已归一化的向量，
            # 但保留此步骤以确保万无一失。
            norm = np.linalg.norm(embedding_values)
            if norm > 0:
                normed_embedding = embedding_values / norm
            else:
                normed_embedding = embedding_values

            doc["vector"] = normed_embedding.tolist()
            embedded_docs.append(doc)
        else:
            logging.warning(f"{level_tag} 文档 {doc_id} 未找到对应的嵌入结果。")

    logging.info(
        f"{level_tag}🎉 结果处理完成！成功获取 {len(embedded_docs)} 个向量，失败/丢失 {len(original_docs) - len(embedded_docs)} 个。")
    return embedded_docs


# --- 向量存储函数 (保持不变) ---
def store_embeddings_lancedb(db_path: Path, table_name: str, embedded_data: List[Dict], data_type: str = ""):
    """将嵌入数据存储到LanceDB表中"""
    level_tag = f"[{data_type.upper()}] " if data_type else ""
    if not embedded_data:
        logging.warning(f"{level_tag}没有可用于存储到表 '{table_name}' 的数据。")
        return

    logging.info(f"{level_tag}正在将 {len(embedded_data)} 条记录存入 LanceDB 表 '{table_name}'...")
    db_path.mkdir(exist_ok=True, parents=True)
    db = lancedb.connect(db_path)

    df_data = []
    for doc in embedded_data:
        if "vector" not in doc: continue
        new_doc = doc.copy()
        if "text_to_embed" in new_doc:
            new_doc["text"] = new_doc.pop("text_to_embed")
        for key, val in new_doc.items():
            if key.startswith("payload_") and not isinstance(val, str):
                new_doc[key] = json.dumps(val, ensure_ascii=False)
        df_data.append(new_doc)

    if not df_data: return
    df = pd.DataFrame(df_data)

    try:
        if table_name in db.table_names():
            logging.warning(f"{level_tag}表 '{table_name}' 已存在，将删除并重建。")
            db.drop_table(table_name)
        db.create_table(table_name, data=df)
        logging.info(f"{level_tag}✅ LanceDB 表 '{table_name}' 创建并写入成功。")
    except Exception as e:
        logging.error(f"{level_tag}❌ 写入 LanceDB 表 '{table_name}' 时失败: {e}", exc_info=True)


# --- 并行任务处理单元 (SDK 类型注解修改) ---
def process_data_type(data_type: str, documents: List[Dict], config: Dict[str, Any], client: OpenAI):
    """
    处理单个数据类型的完整工作流
    """
    if not documents:
        logging.info(f"[{data_type.upper()}] 没有需要处理的数据，跳过。")
        return f"{data_type.upper()}: 无数据，跳过。"

    logging.info(f"\n{'=' * 60}\n[{data_type.upper()}] 开始处理\n{'=' * 60}")

    model_name = config["embedding"]["model"]
    dimensionality = config["embedding"]["dimensionality"]
    sleep_interval = config["embedding"]["sleep_interval"]
    requests_dir = PROJECT_ROOT / config["embedding"]["requests_path"]
    db_path = PROJECT_ROOT / config["embedding"]["output_db_path"]

    # 阿里云单文件最大 50,000 请求 [cite: 1238]，配置 batch_size 需注意
    batch_size = int(config["embedding"].get("batch_size", 2000))

    num_documents = len(documents)
    num_batches = (num_documents + batch_size - 1) // batch_size

    if num_batches > 1:
        logging.info(
            f"⚙️ [{data_type.upper()}] 文档数量 {num_documents} 超过批次大小 {batch_size}，将拆分为 {num_batches} 个批次处理")

    all_embedded_documents = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_documents)
        batch_documents = documents[start_idx:end_idx]

        if num_batches > 1:
            requests_path = requests_dir / f"embedding_{data_type}_requests_batch_{batch_idx + 1}.jsonl"
        else:
            requests_path = requests_dir / f"embedding_{data_type}_requests.jsonl"

        logging.info(
            f"🔄 [{data_type.upper()}] 处理批次 {batch_idx + 1}/{num_batches}：文档 {start_idx + 1}-{end_idx} (共 {len(batch_documents)} 个)")

        create_embedding_requests(batch_documents, model_name, requests_path, dimensionality, data_type)

        batch_job_status = submit_and_monitor_embedding_job(
            client, requests_path, model_name, sleep_interval, data_type, batch_idx, num_batches
        )

        embedded_batch = process_embedding_results(batch_job_status, client, batch_documents, data_type)

        if embedded_batch:
            all_embedded_documents.extend(embedded_batch)
            logging.info(
                f"✅ [{data_type.upper()}] 批次 {batch_idx + 1}/{num_batches} 完成，获得 {len(embedded_batch)} 个嵌入向量")
        else:
            logging.warning(f"⚠️ [{data_type.upper()}] 批次 {batch_idx + 1}/{num_batches} 未获得有效嵌入向量")

    if all_embedded_documents:
        logging.info(f"🎉 [{data_type.upper()}] 所有批次处理完成，共获得 {len(all_embedded_documents)} 个嵌入向量")
        store_embeddings_lancedb(db_path, data_type, all_embedded_documents, data_type)
        return f"[{data_type.upper()}] 成功处理 {len(all_embedded_documents)} 个文档。"
    else:
        logging.error(f"❌ [{data_type.upper()}] 所有批次均未获得有效嵌入向量")
        return f"[{data_type.upper()}] 处理失败或未返回向量。"


# --- 主执行函数 ---
def main():
    """主执行流程"""
    config = load_config()
    setup_logging(config)

    # --- 1. 初始化客户端 (修改：使用 OpenAI SDK 连接阿里云百炼) ---
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("未找到 API Key，请设置 QWEN_API_KEY 环境变量")

    # 获取代理配置
    # proxy = config.get("proxy")
    http_client_kwargs = {}
    # if proxy:
    #     import httpx
    #     http_client_kwargs["http_client"] = httpx.Client(proxy=proxy)
    #     logging.info(f"使用代理: {proxy}")

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **http_client_kwargs
    )
    logging.info("阿里云百炼 (OpenAI兼容) 客户端初始化完成。")

    # --- 2. 初始化文本清洁器 ---
    graph_path = PROJECT_ROOT / config["embedding"]["input_graph_path"]
    id_maps_path = PROJECT_ROOT / config["embedding"]["input_id_maps_path"]
    text_cleaner = init_text_cleaner(graph_path, id_maps_path)
    logging.info("文本清洁器初始化完成。")

    # --- 3. 加载数据 ---
    graph = load_graph_data(graph_path)
    community_path = PROJECT_ROOT / config["embedding"]["input_community_report_path"]

    # --- 4. 准备待嵌入数据 ---
    data_to_embed = prepare_embedding_data(graph, community_path, text_cleaner)

    # --- 5. 循环处理每个数据层级 ---
    logging.info("\n--- 开始并行处理所有数据层级 ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_datatype = {
            executor.submit(process_data_type, data_type, documents, config, client): data_type
            for data_type, documents in data_to_embed.items()
        }
        for future in concurrent.futures.as_completed(future_to_datatype):
            data_type = future_to_datatype[future]
            try:
                result = future.result()
                logging.info(f"✅ 并行任务结果 -> {result}")
            except Exception as exc:
                logging.error(f"❌  {data_type.upper()} 在执行过程中产生异常: {exc}", exc_info=True)

    logging.info("\n✅ 所有层级的知识图谱数据均已成功嵌入并存入向量数据库！")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)