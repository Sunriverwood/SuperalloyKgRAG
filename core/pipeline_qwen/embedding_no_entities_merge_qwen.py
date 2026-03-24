"""
构建 no_entities_merge.db - 基于消歧图，仅包含实体和关系两个层级（无社区）

用法:
    python -m core.pipeline_qwen.embedding_no_entities_merge_qwen
"""

import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.clean_embedding_text import EmbeddingTextCleaner

# 直接导入 embedding_qwen 中经过验证的函数，保证 API 调用、结果解析、存储逻辑完全一致
from core.pipeline_qwen.embedding_qwen import (
    create_embedding_requests,
    submit_and_monitor_embedding_job,
    process_embedding_results,
    store_embeddings_lancedb,
)


def setup_logging(config: Dict[str, Any]):
    """设置日志系统"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / "embedding_no_entities_merge.log"

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


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    import yaml
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_text_cleaner(graph_path: Path, id_maps_path: Path) -> EmbeddingTextCleaner:
    """初始化文本清洁器（与 embedding_qwen.init_text_cleaner 一致）"""
    return EmbeddingTextCleaner(
        final_graph_path=graph_path,
        id_maps_path=id_maps_path
    )


def load_disambiguation_graph(graph_path: Path) -> nx.DiGraph:
    """加载消歧图"""
    logging.info(f"正在从 {graph_path} 加载消歧图...")

    if not graph_path.exists():
        raise FileNotFoundError(f"消歧图文件 {graph_path} 不存在！")

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    graph = nx.node_link_graph(graph_data, directed=True)

    logging.info(f"消歧图加载完成: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
    return graph


def prepare_no_community_embedding_data(graph: nx.DiGraph, text_cleaner: EmbeddingTextCleaner) -> Dict[str, List[Dict]]:
    """
    为无社区场景准备嵌入数据（仅实体和关系），
    数据结构与 embedding_qwen.prepare_embedding_data 返回的 entities/relationships 完全一致。
    """
    logging.info("正在准备实体和关系的嵌入数据（无社区）...")

    # --- 准备实体数据 ---
    # 与 embedding_qwen.prepare_embedding_data 中实体部分一致：
    # 字段: id, text_to_embed, payload_node_data_json
    logging.info("[实体] 正在准备实体数据...")
    entities_to_embed = []

    for node_id, data in graph.nodes(data=True):
        payload_data = data.copy()
        text_source = payload_data.pop("description", data.get("name", ""))
        text_to_embed = text_cleaner.clean_text(text_source, community_id="")

        if text_to_embed.strip():
            entities_to_embed.append({
                "id": str(node_id),
                "text_to_embed": text_to_embed,
                "payload_node_data_json": json.dumps(payload_data, ensure_ascii=False)
            })

    logging.info(f"[实体] 准备了 {len(entities_to_embed)} 个实体待嵌入。")

    # --- 准备关系数据 ---
    # 与 embedding_qwen.prepare_embedding_data 中关系部分一致：
    # 字段: id, text_to_embed, payload_edge_data_json
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
            "id": str(data.get("id", f"{u}_{v}")),
            "text_to_embed": triplet_text,
            "payload_edge_data_json": json.dumps(payload_data, ensure_ascii=False)
        })

    logging.info(f"[关系] 准备了 {len(relationships_to_embed)} 个关系待嵌入。")

    return {
        "entities": entities_to_embed,
        "relationships": relationships_to_embed
    }


def process_data_type_no_merge(data_type: str, documents: List[Dict], config: Dict[str, Any],
                               client: OpenAI, db_path: Path) -> str:
    """
    处理单个数据类型的完整工作流。
    与 embedding_qwen.process_data_type 逻辑完全一致，
    唯一区别是 db_path 由外部传入（指向 no_entities_merge.db）。
    """
    if not documents:
        logging.info(f"[{data_type.upper()}] 没有需要处理的数据，跳过。")
        return f"{data_type.upper()}: 无数据，跳过。"

    logging.info(f"\n{'=' * 60}\n[{data_type.upper()}] 开始处理\n{'=' * 60}")

    model_name = config["embedding"]["model"]
    dimensionality = config["embedding"]["dimensionality"]
    sleep_interval = config["embedding"]["sleep_interval"]
    requests_dir = PROJECT_ROOT / config["embedding"]["requests_path"]

    # 阿里云单文件最大 50,000 请求，配置 batch_size 需注意
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
            requests_path = requests_dir / f"embedding_no_merge_{data_type}_requests_batch_{batch_idx + 1}.jsonl"
        else:
            requests_path = requests_dir / f"embedding_no_merge_{data_type}_requests.jsonl"

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


def main():
    """主执行流程"""
    config = load_config()
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("开始构建 no_entities_merge.db (消歧图 + 无社区 + 仅实体和关系)")
    logging.info("=" * 80)

    # --- 1. 初始化客户端 ---
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("未找到 API Key，请设置 QWEN_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    logging.info("阿里云百炼 (OpenAI兼容) 客户端初始化完成。")

    # --- 2. 加载消歧图 ---
    graph_path = PROJECT_ROOT / config["graph_builder"].get(
        "disambiguation_graph_path", "data/graphs/disambiguation_graph.json")
    graph = load_disambiguation_graph(graph_path)

    # --- 3. 初始化文本清洁器（用消歧图代替 final_graph） ---
    id_maps_path = PROJECT_ROOT / config["embedding"]["input_id_maps_path"]
    text_cleaner = init_text_cleaner(graph_path, id_maps_path)
    logging.info("文本清洁器初始化完成。")

    # --- 4. 准备待嵌入数据（仅实体和关系，无社区） ---
    data_to_embed = prepare_no_community_embedding_data(graph, text_cleaner)

    # --- 5. 确定输出数据库路径 ---
    base_db_path = PROJECT_ROOT / config["embedding"]["output_db_path"]
    db_path = base_db_path.parent / "no_entities_merge.db"
    logging.info(f"输出数据库: {db_path}")

    # --- 6. 并行处理实体和关系 ---
    logging.info("\n--- 开始并行处理实体和关系 ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_datatype = {
            executor.submit(process_data_type_no_merge, data_type, documents, config, client, db_path): data_type
            for data_type, documents in data_to_embed.items()
        }
        for future in concurrent.futures.as_completed(future_to_datatype):
            data_type = future_to_datatype[future]
            try:
                result = future.result()
                logging.info(f"✅ 并行任务结果 -> {result}")
            except Exception as exc:
                logging.error(f"❌ {data_type.upper()} 在执行过程中产生异常: {exc}", exc_info=True)

    logging.info(f"\n✅ no_entities_merge.db 构建完成！")
    logging.info(f"数据库位置: {db_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"关键错误: {e}", exc_info=True)
        sys.exit(1)
