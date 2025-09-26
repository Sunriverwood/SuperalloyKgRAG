import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import networkx as nx
import pandas as pd
import yaml
from google import genai
from google.genai import types

from utils.client_factory import create_gemini_client

# --- 动态计算项目根目录 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/graph_builder.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)

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


# --- 图构建函数 ---
def load_and_build_initial_graph(jsonl_path: Path) -> nx.DiGraph:
    """从jsonl文件加载实体和关系，并创建全局唯一ID来构建初始图。"""
    logging.info(f"从 {jsonl_path} 加载图数据并创建全局唯一ID...")
    graph = nx.DiGraph()
    all_nodes, all_edges = [], []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunk_id = data.get("id")
            if not chunk_id:
                logging.warning("发现一行缺少'id'字段，跳过。")
                continue

            id_map = {}
            if 'graph' in data and 'entities' in data['graph']:
                for entity in data['graph']['entities']:
                    local_id = entity['id']
                    global_id = f"{chunk_id}-{local_id}"
                    id_map[local_id] = global_id
                    new_entity = {**entity, 'id': global_id, 'original_id': local_id, 'chunk_id': chunk_id}
                    all_nodes.append(new_entity)

            if 'graph' in data and 'relationships' in data['graph']:
                for rel in data['graph']['relationships']:
                    if rel['source'] in id_map and rel['target'] in id_map:
                        global_id = f"{chunk_id}-{rel['id']}"
                        new_rel = {**rel, 'id': global_id, 'source': id_map[rel['source']],
                                   'target': id_map[rel['target']], 'original_id': rel['id'], 'chunk_id': chunk_id}
                        all_edges.append(new_rel)

    if not all_nodes:
        logging.warning("未能从文件中加载任何节点。")
        return graph

    entity_df = pd.DataFrame(all_nodes).drop_duplicates(subset=['id'])
    relationship_df = pd.DataFrame(all_edges).drop_duplicates(subset=['id']) if all_edges else pd.DataFrame()

    logging.info(f"加载了 {entity_df.shape[0]} 个唯一实体和 {relationship_df.shape[0]} 个唯一关系。")

    for _, entity in entity_df.iterrows():
        graph.add_node(entity['id'], **{k: v for k, v in entity.items() if k != 'id'})

    if not relationship_df.empty:
        for _, rel in relationship_df.iterrows():
            if graph.has_node(rel['source']) and graph.has_node(rel['target']):
                graph.add_edge(rel['source'], rel['target'],
                               **{k: v for k, v in rel.items() if k not in ['source', 'target']})

    logging.info("包含全局唯一ID的初始图构建完成。")
    logging.info(f"图信息: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边。")
    return graph


def create_disambiguation_prompt(node_id: str, graph: nx.DiGraph) -> str:
    """为单个实体生成用于消歧的上下文和prompt。"""
    node_data = graph.nodes[node_id]
    context_parts = [
        f"实体名称: {node_data.get('name', 'N/A')}",
        f"实体类型: {node_data.get('type', 'N/A')}",
        f"原始描述: {node_data.get('description', 'N/A')}"
    ]
    relations_context = []
    for _, v, data in graph.edges(node_id, data=True):
        relations_context.append(
            f"- 与 '{graph.nodes[v].get('name', 'N/A')}' 的关系是 '{data.get('relationship', 'N/A')}'")
    for u, _, data in graph.in_edges(node_id, data=True):
        relations_context.append(
            f"- '{graph.nodes[u].get('name', 'N/A')}' 与它的关系是 '{data.get('relationship', 'N/A')}'")
    if relations_context:
        context_parts.append("\n关系上下文:")
        context_parts.extend(relations_context)
    full_context = "\n".join(context_parts)
    prompt = f"""
    请基于以下上下文，为实体生成一个全面、精准、消除歧义的标准化描述，像百科定义一样，不超过100字。
    上下文信息:
    {full_context}
    标准化描述:
    """
    return prompt.strip()

def create_batch_requests(graph: nx.DiGraph, model_name: str, output_path: Path) -> List[str]:
    """创建批量请求并写入本地 JSONL 文件。"""
    logging.info("正在创建批量消歧请求...")
    requests, node_ids = [], list(graph.nodes())
    for node_id in node_ids:
        prompt = create_disambiguation_prompt(node_id, graph)
        requests.append(
            {"key": node_id, "request": {"model": f"models/{model_name}", "contents": {"parts": [{"text": prompt}]}}})

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    logging.info(f"已为 {len(requests)} 个实体生成请求，并写入到 {output_path}")
    return node_ids

def submit_and_monitor_job(client: genai.Client, input_file_path: Path, model_name: str, sleep_interval: int) -> Any:
    """上传文件，创建并监控批量作业。"""
    logging.info(f"📤 正在上传请求文件: {input_file_path.name}...")
    try:
        uploaded_file = client.files.upload(
            file=str(input_file_path),
            config={
                "display_name": f'graph-batch-{input_file_path.stem}',
                "mime_type": 'application/jsonl'
            }
        )
        logging.info(f"✅ 文件上传成功: {uploaded_file.name}")
    except Exception as e:
        logging.error(f"❌ 文件上传失败: {e}")
        return None, None

    logging.info(f"🚀 正在使用模型 '{model_name}' 创建批量作业...")
    try:
        batch_job = client.batches.create(
            model=f"models/{model_name}",
            src=uploaded_file.name,
            config={
                'display_name': f"graph-job-{input_file_path.stem}",
            },
        )
        logging.info(f"✅ 批量作业已创建: {batch_job.name}")
    except Exception as e:
        logging.error(f"❌ 创建批量作业失败: {e}")
        return None, uploaded_file

    job_name = batch_job.name
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

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

    return batch_job_status, uploaded_file

def process_results(batch_job_status: Any, client: genai.Client) -> Dict[str, str]:
    """下载并处理批量作业的结果。"""
    if batch_job_status.state.name != 'JOB_STATE_SUCCEEDED':
        logging.error(f"❌ 作业失败: {batch_job_status.error}")
        return {}

    new_descriptions = {}
    result_file_name = batch_job_status.dest.file_name
    logging.info(f"📥 正在下载结果文件: {result_file_name}")
    file_content = client.files.download(file=result_file_name).decode('utf-8')

    processed_count, error_count = 0, 0
    for line in file_content.strip().split('\n'):
        try:
            result = json.loads(line)
            node_id = result.get("key")
            if node_id and result.get("response"):
                text = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                new_descriptions[node_id] = text.strip()
                processed_count += 1
            elif result.get("error"):
                logging.error(f"  - ❌ 处理 ID '{node_id}' 时发生错误: {result['error']['message']}")
                error_count += 1
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logging.warning(f"  - ⚠️ 解析结果行失败: '{line}', 错误: {e}")
            error_count += 1

    logging.info(f"🎉 结果处理完成！成功获取 {processed_count} 条描述，失败 {error_count} 条。")
    return new_descriptions


def update_graph_with_new_descriptions(graph: nx.DiGraph, new_descriptions: Dict[str, str]):
    """用Gemini生成的新描述更新图中的实体。"""
    logging.info("正在用消歧后的描述更新图谱...")
    updated_count = 0
    for node_id, new_desc in new_descriptions.items():
        if graph.has_node(node_id):
            graph.nodes[node_id]['description'] = new_desc
            graph.nodes[node_id]['is_disambiguated'] = True
            updated_count += 1
    logging.info(f"成功更新了 {updated_count} 个节点的描述。")


def save_final_graph(graph: nx.DiGraph, output_path: Path):
    """将最终的图谱保存到文件。"""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f"正在将最终图谱保存到 {output_path}...")
    graph_data = nx.node_link_data(graph)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ 图谱保存成功。")


# --- 主执行函数 ---
def main():
    """主执行流程"""
    config = load_config()
    setup_logging(config)

    # --- 1. 初始化客户端 ---
    api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    proxy = config["proxy"]
    client = create_gemini_client(api_key, proxy)
    sleep_interval = config["graph_builder"]["sleep_interval"]
    logging.info("Gemini 客户端初始化完成。")

    # --- 2. 加载和构建初始图 ---
    input_path = PROJECT_ROOT / config["graph_builder"]["input_path"]
    graph = load_and_build_initial_graph(input_path)
    if not graph.nodes:
        logging.error("图为空，无法继续。请检查输入文件。")
        return

    # --- 3. 创建批量请求 ---
    model_name = config["llm"]["model"]
    requests_path = PROJECT_ROOT / config["graph_builder"]["requests_path"]
    create_batch_requests(graph, model_name, requests_path)

    # --- 4. 提交并监控作业 ---
    batch_job, input_file = submit_and_monitor_job(client, requests_path, model_name, sleep_interval)

    # --- 5. 处理结果 ---
    new_descriptions = process_results(batch_job, client)

    # --- 6. 更新并保存最终图谱 ---
    if new_descriptions:
        update_graph_with_new_descriptions(graph, new_descriptions)
        output_path = PROJECT_ROOT / config["graph_builder"]["output_path"]
        save_final_graph(graph, output_path)
    else:
        logging.warning("未能获取任何新的实体描述，图谱未更新。")

    logging.info("图构建与实体消歧流程全部完成！")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)