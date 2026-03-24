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
from google import genai
import concurrent.futures
from utils.client_factory import create_gemini_client
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
    """
    代理到 EmbeddingTextCleaner.clean_text。
    注意：community_id 用于启用 E1/R1 -> 人类可读名称 的替换。
    如果未提供 community_id，将仅进行通用清洁（移除 [Data: ...] 和 (E1)/(R1)）。
    """
    if _text_cleaner_instance is None:
        # 若未初始化，返回基本清洁结果以避免异常（保持与旧逻辑一致）
        if not isinstance(text, str):
            return ""
        tmp = re.sub(r'\[Data:.*?]', '', text)  # remove redundant escaping of ']'
        tmp = re.sub(r'\s*\([ER]\d+\)', '', tmp)  # use char class instead of (?:E|R)
        return re.sub(r'\s+', ' ', tmp).strip()
    return _text_cleaner_instance.clean_text(text, community_id)


# --- 数据加载与准备 (修改) ---

def load_graph_data(graph_path: Path) -> nx.DiGraph:
    """
    修改：仅加载最终图谱。
    社区报告将在 'prepare_embedding_data' 中单独加载。
    """
    logging.info(f"正在从 {graph_path} 加载图谱...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    graph = nx.node_link_graph(graph_data, edges="links")
    logging.info("图谱加载成功。")
    return graph


def load_and_prepare_community_data(community_report_path: Path, text_cleaner: EmbeddingTextCleaner) -> List[Dict]:
    """
    新增：加载 community_summaries.jsonl并准备用于嵌入的数据。
    这将分离 "text_to_embed" (清洁) 和 "payload" (原始)。
    使用 EmbeddingTextCleaner.clean_text 进行文本清洁。
    """
    logging.info(f"[社区] 正在从 {community_report_path} 加载和准备社区数据...")
    communities_to_embed = []

    if not community_report_path.exists():
        logging.error(f"[社区] 社区报告文件未找到: {community_report_path}")
        return []

    with open(community_report_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                community_id = str(data["community_id"])  # 确保为字符串
                report = data["report"]  # 这是一个 dict
                local_id_map = data.get("local_id_map", {})  # 也是 dict

                # --- 1. 准备 "清洁文本" (用于嵌入) ---
                clean_title = text_cleaner.clean_text(report.get("title", ""), community_id)
                clean_summary = text_cleaner.clean_text(report.get("summary", ""), community_id)

                clean_findings_texts = []
                for finding in report.get("findings", []):
                    clean_f_summary = text_cleaner.clean_text(finding.get("summary", ""), community_id)
                    clean_f_explanation = text_cleaner.clean_text(finding.get("explanation", ""), community_id)
                    clean_findings_texts.append(f"{clean_f_summary} {clean_f_explanation}")

                # 合并为一个大文档 (Document) 以供嵌入
                full_clean_text = (
                        f"Title: {clean_title}\n"
                        f"Abstract: {clean_summary}\n"
                        f"findings:\n" + "\n".join(clean_findings_texts)
                )

                # --- 2. 准备 "元数据载荷" (用于存储和溯源) ---
                communities_to_embed.append({
                    "id": f"community_{community_id}",
                    "text_to_embed": full_clean_text,
                    "payload_report_json": json.dumps(report, ensure_ascii=False),
                    "payload_map_json": json.dumps(local_id_map, ensure_ascii=False)
                })

            except json.JSONDecodeError:
                logging.warning(f"跳过无效的JSONL行: {line[:100]}...")
            except KeyError as e:
                logging.warning(f"跳过缺少键 {e} 的JSONL行: {line[:100]}...")

    logging.info(f"[社区] 准备了 {len(communities_to_embed)} 个社区待嵌入。")
    return communities_to_embed


def prepare_embedding_data(graph: nx.DiGraph, community_report_path: Path, text_cleaner: EmbeddingTextCleaner) -> Dict[str, List[Dict]]:
    """
    修改：为社区、实体和关系准备待嵌入的文本数据和载荷。
    文本清洁使用 EmbeddingTextCleaner。
    """
    logging.info("正在准备三个的嵌入数据...")

    # 1. 社区数据 (使用新逻辑)
    communities_to_embed = load_and_prepare_community_data(community_report_path, text_cleaner)

    # 2. 实体数据 (应用新结构)
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

    # 3. 关系数据 (应用新结构)
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


# --- 批量向量化工作流 (修改) ---
def create_embedding_requests(documents: List[Dict], model_name: str, output_path: Path, dimensionality: int, data_type: str = ""):
    """为批量嵌入创建请求并写入本地 JSONL 文件。"""
    level_tag = f"[{data_type.upper()}] " if data_type else ""
    logging.info(f"{level_tag}正在为 {len(documents)} 个文档创建批量嵌入请求...")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            text_for_embedding = doc.get("text_to_embed", "")
            if not text_for_embedding:
                logging.warning(f"{level_tag}文档 {doc['id']} 缺少 'text_to_embed' 内容，跳过。")
                continue

            request_body = {
                "content": {"parts": [{"text": text_for_embedding}]},
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": dimensionality
            }

            request_line = {
                "key": doc["id"],
                "request": request_body
            }
            f.write(json.dumps(request_line, ensure_ascii=False) + '\n')
    logging.info(f"{level_tag}嵌入请求已写入到 {output_path}")


def submit_and_monitor_embedding_job(client: genai.Client, requests_path: Path, model_name: str, sleep_interval: int, data_type: str = "", batch_index: int = 0, total_batches: int = 1) -> Any:
    """
    上传文件，创建并监控批量嵌入作业。
    batch_index: 当前批次索引（从0开始）
    total_batches: 总批次数
    """
    level_tag = f"[{data_type.upper()}] " if data_type else ""
    batch_tag = f"[批次 {batch_index+1}/{total_batches}] " if total_batches > 1 else ""
    logging.info(f"{level_tag}{batch_tag}📤 正在上传请求文件: {requests_path.name}...")
    try:
        uploaded_file = client.files.upload(
            file=str(requests_path),
            config={
                "display_name": f'embedding-batch-{requests_path.stem}',
                "mime_type": 'application/jsonl'
            }
        )
        logging.info(f"{level_tag}{batch_tag}✅ 文件上传成功: {uploaded_file.name}")
    except Exception as e:
        logging.error(f"{level_tag}{batch_tag}❌ 文件上传失败: {e}")
        return None

    logging.info(f"{level_tag}{batch_tag}🚀 正在创建批量嵌入作业...")
    try:
        batch_job = client.batches.create_embeddings(
            model=f"{model_name}",
            src={'file_name': uploaded_file.name},
            config={'display_name': f"embedding-job-{requests_path.stem}"},
        )
        logging.info(f"{level_tag}{batch_tag}✅ 批量作业已创建: {batch_job.name}")
    except Exception as e:
        logging.error(f"{level_tag}{batch_tag}❌ 创建批量作业失败: {e}")
        try:
            client.files.delete(name=uploaded_file.name)  # 清理已上传的文件
        except:
            pass
        return None

    job_name = batch_job.name
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    if sleep_interval == 0:
        logging.info(f"{level_tag}{batch_tag}sleep_interval=0，跳过轮询，直接返回作业对象。")
        return batch_job

    else:
        logging.info(f"{level_tag}{batch_tag}⏳ 开始轮询作业 '{job_name}' 状态，每 {sleep_interval} 秒检查一次...")
        while True:
            try:
                batch_job_status = client.batches.get(name=job_name)
                current_state = batch_job_status.state.name
                logging.info(f"{level_tag}{batch_tag}  当前状态: {current_state}")
                if current_state in completed_states:
                    break
                time.sleep(sleep_interval)
            except Exception as e:
                logging.error(f"{level_tag}{batch_tag}  轮询失败: {e}")
                time.sleep(sleep_interval * 2)
        return batch_job_status


def process_embedding_results(batch_job: Any, client: genai.Client, original_docs: List[Dict], data_type: str = "") -> List[Dict]:
    """下载、处理并归一化批量嵌入作业的结果（基于行号索引匹配）。"""
    level_tag = f"[{data_type.upper()}] " if data_type else ""

    if not batch_job or batch_job.state != 'JOB_STATE_SUCCEEDED':
        logging.error(f"{level_tag}❌ 作业失败或未执行，无法处理结果。")
        if batch_job and batch_job.error: logging.error(f"{level_tag}  失败原因: {batch_job.error}")
        return []

    # job_id = batch_job.name.split('/')[-1]
    logging.info(f"{level_tag}📥 正在下载结果文件: {batch_job.dest.file_name}")
    file_content = client.files.download(file=batch_job.dest.file_name).decode('utf-8')

    output_lines = file_content.strip().split('\n')

    if len(output_lines) != len(original_docs):
        logging.error(f"{level_tag}❌ 结果数量与原始文档数量不匹配！"
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
                logging.error(f"{level_tag}  ❌ 文档索引 {i} (ID: {original_docs[i]['id']}) 嵌入失败: {error_msg}")
                error_count += 1

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logging.warning(f"{level_tag}  ⚠️ 解析或处理结果行 {i + 1} 失败: '{line[:100]}...', 错误: {e}")
            error_count += 1

    logging.info(f"{level_tag}🎉 结果处理完成！成功获取并归一化 {len(embedded_docs)} 个向量，失败 {error_count} 个。")
    return embedded_docs


# --- 向量存储函数 ---
def store_embeddings_lancedb(db_path: Path, table_name: str, embedded_data: List[Dict], data_type: str = ""):
    """将嵌入数据（包括向量和所有元数据/载荷）存储到LanceDB表中"""
    level_tag = f"[{data_type.upper()}] " if data_type else ""

    if not embedded_data:
        logging.warning(f"{level_tag}没有可用于存储到表 '{table_name}' 的数据。")
        return

    logging.info(f"{level_tag}正在将 {len(embedded_data)} 条记录存入 LanceDB 表 '{table_name}'...")
    db_path.mkdir(exist_ok=True, parents=True)
    db = lancedb.connect(db_path)

    df_data = []
    for doc in embedded_data:
        # 确保 'vector' 键存在
        if "vector" not in doc:
            logging.warning(f"{level_tag}文档 {doc['id']} 缺少 'vector'，无法存入 LanceDB。")
            continue

        new_doc = doc.copy()

        # 将 'text_to_embed' 重命名为 'text' (这是向量的源文本)
        if "text_to_embed" in new_doc:
            new_doc["text"] = new_doc.pop("text_to_embed")

        # 确保所有 payload 都是字符串 (它们应该是, 但以防万一)
        for key, val in new_doc.items():
            if key.startswith("payload_") and not isinstance(val, str):
                logging.warning(f"{level_tag}将 {key} 转换为字符串...")
                new_doc[key] = json.dumps(val, ensure_ascii=False)

        df_data.append(new_doc)

    if not df_data:
        logging.error(f"{level_tag}没有有效数据可供存入表 '{table_name}'。")
        return

    df = pd.DataFrame(df_data)

    try:
        if table_name in db.table_names():
            logging.warning(f"{level_tag}表 '{table_name}' 已存在，将删除并重建。")
            db.drop_table(table_name)

        # 从 DataFrame 自动创建表
        db.create_table(table_name, data=df)
        logging.info(f"{level_tag}✅ LanceDB 表 '{table_name}' 创建并写入成功。")
    except Exception as e:
        logging.error(f"{level_tag}❌ 写入 LanceDB 表 '{table_name}' 时失败: {e}", exc_info=True)


# --- 并行任务处理单元 ---
def process_data_type(data_type: str, documents: List[Dict], config: Dict[str, Any], client: genai.Client):
    """
    处理单个数据类型的完整工作流：创建请求 -> 提交和监控 -> 处理结果 -> 存储。
    支持批次分割处理，避免API配额错误。
    """
    if not documents:
        logging.info(f"[{data_type.upper()}] 没有需要处理的数据，跳过。")
        return f"{data_type.upper()}: 无数据，跳过。"

    logging.info(f"\n{'='*60}\n[{data_type.upper()}] 开始处理\n{'='*60}")

    model_name = config["embedding"]["model"]
    dimensionality = config["embedding"]["dimensionality"]
    sleep_interval = config["embedding"]["sleep_interval"]
    requests_dir = PROJECT_ROOT / config["embedding"]["requests_path"]
    db_path = PROJECT_ROOT / config["embedding"]["output_db_path"]

    # 获取批次大小配置（默认2000）
    batch_size = int(config["embedding"].get("batch_size", 2000))

    # 计算批次数量
    num_documents = len(documents)
    num_batches = (num_documents + batch_size - 1) // batch_size

    if num_batches > 1:
        logging.info(f"⚙️ [{data_type.upper()}] 文档数量 {num_documents} 超过批次大小 {batch_size}，将拆分为 {num_batches} 个批次处理")

    all_embedded_documents = []

    # 分批处理
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_documents)
        batch_documents = documents[start_idx:end_idx]

        # 为每个批次创建单独的请求文件
        if num_batches > 1:
            requests_path = requests_dir / f"embedding_{data_type}_requests_batch_{batch_idx+1}.jsonl"
        else:
            requests_path = requests_dir / f"embedding_{data_type}_requests.jsonl"

        logging.info(f"🔄 [{data_type.upper()}] 处理批次 {batch_idx+1}/{num_batches}：文档 {start_idx+1}-{end_idx} (共 {len(batch_documents)} 个)")

        # 创建批次请求
        create_embedding_requests(batch_documents, model_name, requests_path, dimensionality, data_type)

        # 提交并监控批次作业
        batch_job_status = submit_and_monitor_embedding_job(
            client, requests_path, model_name, sleep_interval, data_type, batch_idx, num_batches
        )

        # 处理批次结果
        embedded_batch = process_embedding_results(batch_job_status, client, batch_documents, data_type)

        if embedded_batch:
            all_embedded_documents.extend(embedded_batch)
            logging.info(f"✅ [{data_type.upper()}] 批次 {batch_idx+1}/{num_batches} 完成，获得 {len(embedded_batch)} 个嵌入向量")
        else:
            logging.warning(f"⚠️ [{data_type.upper()}] 批次 {batch_idx+1}/{num_batches} 未获得有效嵌入向量")

    # 合并结果并存储
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

    # --- 1. 初始化客户端 ---
    api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    proxy = config.get("proxy")
    client = create_gemini_client(api_key, proxy)
    logging.info("Gemini 客户端初始化完成。")

    # --- 2. 初始化文本清洁器 ---
    graph_path = PROJECT_ROOT / config["embedding"]["input_graph_path"]
    # 如果配置未提供，默认使用 data/cache/community_detection_id_maps.json
    id_maps_path = PROJECT_ROOT / "data" / "cache" / "community_detection_id_maps.json"
    text_cleaner = init_text_cleaner(graph_path, id_maps_path)
    logging.info("文本清洁器初始化完成。")

    # --- 3. 加载数据 ---
    graph = load_graph_data(graph_path)
    community_path = PROJECT_ROOT / config["embedding"]["input_community_report_path"]

    # --- 4. 准备待嵌入数据 (使用清洁器) ---
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