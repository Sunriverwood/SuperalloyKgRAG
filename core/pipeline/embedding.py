import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import networkx as nx
import pandas as pd
import yaml
import lancedb
from google import genai
from google.genai import types
import concurrent.futures
from utils.client_factory import create_gemini_client

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


# --- 数据加载与准备 ---
def load_graph_data(graph_path: Path, community_report_path: Path) -> (nx.DiGraph, pd.DataFrame):
    """加载最终图谱和社区报告"""
    logging.info(f"正在从 {graph_path} 加载图谱...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    graph = nx.node_link_graph(graph_data)

    logging.info(f"正在从 {community_report_path} 加载社区报告...")
    community_reports_df = pd.read_csv(community_report_path)

    logging.info("图谱和社区报告加载成功。")
    return graph, community_reports_df


def prepare_embedding_data(graph: nx.DiGraph, reports_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """为社区、实体和关系准备待嵌入的文本数据"""
    logging.info("正在准备三个层级的嵌入数据...")

    # 1. 社区数据
    communities_to_embed = [{"id": str(row["community_id"]), "text": row["summary"]} for _, row in
                            reports_df.iterrows()]
    logging.info(f"  - 准备了 {len(communities_to_embed)} 个社区待嵌入。")

    # 2. 实体数据
    entities_to_embed = [{"id": str(node_id), "text": data.get("description", ""), **data} for node_id, data in
                         graph.nodes(data=True) if data.get("is_disambiguated")]
    logging.info(f"  - 准备了 {len(entities_to_embed)} 个实体待嵌入。")

    # 3. 关系数据
    relationships_to_embed = []
    for u, v, data in graph.edges(data=True):
        source_name = graph.nodes[u].get("name", "未知实体")
        target_name = graph.nodes[v].get("name", "未知实体")
        rel_desc = data.get("description", "未知关系")
        triplet_text = f"{source_name} -> [{rel_desc}] -> {target_name}"
        relationships_to_embed.append({"id": str(data["id"]), "text": triplet_text, "source": str(u), "target": str(v)})
    logging.info(f"  - 准备了 {len(relationships_to_embed)} 个关系待嵌入。")

    return {"communities": communities_to_embed, "entities": entities_to_embed, "relationships": relationships_to_embed}


# --- 批量向量化工作流 ---
def create_embedding_requests(documents: List[Dict], model_name: str, output_path: Path, dimensionality: int):
    """为批量嵌入创建请求并写入本地 JSONL 文件。"""
    logging.info(f"正在为 {len(documents)} 个文档创建批量嵌入请求...")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
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


def submit_and_monitor_embedding_job(client: genai.Client, requests_path: Path, model_name: str, sleep_interval: int) -> (Any, Any):
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
        return None, None

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
        client.files.delete(name=uploaded_file.name)  # 清理已上传的文件
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
                batch_job_status = client.batches.get(name=job_name)
                current_state = batch_job_status.state.name
                logging.info(f"  - [Embedding] 当前状态: {current_state}")
                if current_state in completed_states:
                    break
                time.sleep(sleep_interval)
            except Exception as e:
                logging.error(f"  - [Embedding] 轮询失败: {e}")
                time.sleep(sleep_interval * 2)
        return batch_job_status


def process_embedding_results(batch_job: Any, client: genai.Client, original_docs: List[Dict]) -> List[Dict]:
    """下载、处理并归一化批量嵌入作业的结果（基于行号索引匹配）。"""
    if not batch_job or batch_job.state != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 作业失败或未执行，无法处理结果。")
        if batch_job and batch_job.error: logging.error(f"  - 失败原因: {batch_job.error}")
        return []

    job_id = batch_job.name.split('/')[-1]
    logging.info(f"📥 [{job_id}] 正在下载结果文件: {batch_job.dest.file_name}")
    file_content = client.files.download(file=batch_job.dest.file_name).decode('utf-8')

    # 1. 将输出结果按行分割
    output_lines = file_content.strip().split('\n')

    # 2. 安全检查：输入和输出的数量是否一致
    if len(output_lines) != len(original_docs):
        logging.error(f"❌ [{job_id}] 结果数量与原始文档数量不匹配！"
                      f"原始文档: {len(original_docs)}, 返回结果: {len(output_lines)}")
        # 即使数量不匹配，也尝试处理，但记录错误

    embedded_docs = []
    error_count = 0

    # 3. 使用 enumerate 同时遍历原始文档和输出结果行
    for i, line in enumerate(output_lines):
        try:
            # 确保原始文档索引在范围内
            if i >= len(original_docs):
                break
            result = json.loads(line)

            # 检查是否有 response 字段并且内容有效
            if "response" in result and "embedding" in result["response"]:
                embedding_values = np.array(result["response"]["embedding"]["values"])
                normed_embedding = embedding_values / np.linalg.norm(embedding_values)

                # 基于索引将向量添加回对应的原始文档
                doc = original_docs[i]
                doc["vector"] = normed_embedding.tolist()
                embedded_docs.append(doc)
            else:
                # 如果行中包含错误信息，则记录
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
            db.drop_table(table_name)
        db.create_table(table_name, data=df)
        logging.info(f"✅ LanceDB 表 '{table_name}' 创建并写入成功。")
    except Exception as e:
        logging.error(f"❌ 写入 LanceDB 表 '{table_name}' 时失败: {e}")


# --- 并行任务处理单元 ---
def process_data_type(data_type: str, documents: List[Dict], config: Dict[str, Any], client: genai.Client):
    """
    处理单个数据类型的完整工作流：创建请求 -> 提交和监控 -> 处理结果 -> 存储。
    这个函数将被并行执行。
    """
    if not documents:
        logging.info(f"层级 {data_type.upper()} 没有需要处理的数据，跳过。")
        return f"层级 {data_type.upper()}: 无数据，跳过。"

    logging.info(f"\n--- 开始处理层级: {data_type.upper()} ---")

    # 从主配置中提取所需参数
    model_name = config["embedding"]["model"]
    dimensionality = config["embedding"]["dimensionality"]
    sleep_interval = config["embedding"]["sleep_interval"]
    requests_dir = PROJECT_ROOT / config["embedding"]["requests_path"]
    requests_path = requests_dir / f"embedding_{data_type}_requests.jsonl"
    db_path = PROJECT_ROOT / config["embedding"]["output_db_path"]

    # a. 创建批量请求
    create_embedding_requests(documents, model_name, requests_path, dimensionality)

    # b. 提交作业并监控
    final_batch_job_status = submit_and_monitor_embedding_job(client, requests_path, model_name, sleep_interval)

    # c. 处理结果
    embedded_documents = process_embedding_results(final_batch_job_status, client, documents)

    # d. 存储到LanceDB
    store_embeddings_lancedb(db_path, data_type, embedded_documents)

    if not embedded_documents:
        return f"层级 {data_type.upper()}: 处理失败或未返回向量。"

    return f"层级 {data_type.upper()}: 成功处理 {len(embedded_documents)} 个文档。"


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
    graph_path = PROJECT_ROOT / config["embedding"]["input_graph_path"]
    community_path = PROJECT_ROOT / config["embedding"]["input_community_report_path"]
    graph, reports_df = load_graph_data(graph_path, community_path)

    # --- 3. 准备待嵌入数据 ---
    data_to_embed = prepare_embedding_data(graph, reports_df)

    # --- 4. 循环处理每个数据层级 ---
    logging.info("\n--- 开始并行处理所有数据层级 ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 为每个数据类型（communities, entities, relationships）提交一个任务
        future_to_datatype = {
            executor.submit(process_data_type, data_type, documents, config, client): data_type
            for data_type, documents in data_to_embed.items()
        }
        # 等待所有任务完成并获取结果
        for future in concurrent.futures.as_completed(future_to_datatype):
            data_type = future_to_datatype[future]
            try:
                result = future.result()
                logging.info(f"✅ 并行任务结果 -> {result}")
            except Exception as exc:
                logging.error(f"❌ 层级 {data_type.upper()} 在执行过程中产生异常: {exc}", exc_info=True)

    logging.info("\n✅ 所有层级的知识图谱数据均已成功嵌入并存入向量数据库！")

# # --- 主执行函数（用于恢复结果的临时版本） ---
# def main():
#     """主执行流程（临时：用于恢复已完成的作业结果）"""
#     config = load_config()
#     setup_logging(config)
#
#     # --- 1. 初始化客户端 ---
#     api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
#     proxy = config.get("proxy")
#     client = create_gemini_client(api_key, proxy)
#     logging.info("Gemini 客户端初始化完成。")
#
#     # --- 2. 加载本地数据（用于匹配） ---
#     graph_path = PROJECT_ROOT / config["embedding"]["input_graph_path"]
#     community_path = PROJECT_ROOT / config["embedding"]["input_community_report_path"]
#     graph, reports_df = load_graph_data(graph_path, community_path)
#     data_to_embed = prepare_embedding_data(graph, reports_df)
#     db_path = PROJECT_ROOT / config["embedding"]["output_db_path"]
#
#     # --- 3. 恢复已完成的作业 ---
#     logging.info("\n--- 开始恢复已完成的批量作业结果 ---")
#
#     try:
#         # 获取您账户上最近的批量作业列表
#         recent_jobs = client.batches.list()
#
#         # 创建一个字典以便我们通过显示名称（display_name）查找作业
#         job_map = {job.display_name: job for job in recent_jobs}
#         logging.info(f"找到了 {len(job_map)} 个最近的批量作业。")
#
#         # 遍历我们之前提交的三个层级
#         for data_type, documents in data_to_embed.items():
#             if not documents:
#                 continue
#
#             # 构建我们之前为这个作业设置的显示名称
#             expected_display_name = f"embedding-job-embedding_{data_type}_requests"
#
#             logging.info(f"--- 正在查找层级 '{data_type.upper()}' 的作业 ({expected_display_name})... ---")
#
#             # 在找到的作业中进行匹配
#             if expected_display_name in job_map:
#                 completed_job = job_map[expected_display_name]
#
#                 # 确认作业状态是 SUCCEEDED
#                 if completed_job.state == 'JOB_STATE_SUCCEEDED':
#                     logging.info(f"✅ 找到了已成功的作业: {completed_job.name}")
#
#                     # a. 处理结果（使用我们刚刚修正过的函数）
#                     embedded_documents = process_embedding_results(completed_job, client, documents)
#
#                     # b. 存储到LanceDB
#                     store_embeddings_lancedb(db_path, data_type, embedded_documents)
#
#                     logging.info(f"🎉 层级 '{data_type.upper()}' 的结果已成功恢复并存储。")
#                 else:
#                     logging.warning(f"⚠️ 作业 '{completed_job.name}' 已找到，但状态为 {completed_job.state.name}，跳过。")
#             else:
#                 logging.error(
#                     f"❌ 未能找到显示名称为 '{expected_display_name}' 的作业，请检查作业是否已过期或名称是否正确。")
#
#     except Exception as e:
#         logging.critical(f"恢复过程中发生错误: {e}", exc_info=True)
#
#     logging.info("\n✅ 所有找到的已成功作业均已恢复完毕！")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)