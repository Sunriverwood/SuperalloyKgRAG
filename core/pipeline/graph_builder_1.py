import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List
from string import Template
import networkx as nx
import pandas as pd
import yaml
from collections import Counter
from google import genai
from google.genai import types
import leidenalg as la
import igraph as ig

from utils.client_factory import create_gemini_client
import utils.community_importance as ci

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

def load_prompt(prompt_dir: str, filename: str) -> str:
    prompt_path = PROJECT_ROOT / prompt_dir/ filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt文件未找到: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

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

def build_community_context(
    graph: nx.Graph,
    member_ids: List[str],
    max_entities: int = 25,
    max_relationships: int = 50,
    include_importance: bool = True,
) -> str:
    """
    先从社区内筛选/排序关系（边），再基于已选关系的端点节点构建“实体”清单。
    - 仅以节点ID过滤（member_ids 必须是图中的节点ID）
    - 关系优先级：按 `composite_importance` 降序（缺失时稳定退化）
    - 节点优先级：在“已选关系”中出现频次高者优先；并以社区内度数作为次级排序
    - 最终仅输出：
        Entities: （不超过 max_entities 个）
        Relationships: （端点均在最终实体集合内，且不超过 max_relationships 条）
    """
    # 0) 过滤出有效成员ID
    member_set = {n for n in member_ids if n in graph.nodes}
    if not member_set:
        return "Entities:\n\nRelationships:\n"

    # 1) 在社区内收集候选边（两端都在 member_set）
    candidate_edges = []
    for u, v, edata in graph.edges(data=True):
        if u in member_set and v in member_set:
            rel = edata.get("relationship") or edata.get("relation") or edata.get("rel_type") or "related_to"
            imp = edata.get("composite_importance")
            # 为了排序和去重的稳定性，规范端点顺序
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            rel_id = edata.get("id")
            candidate_edges.append((a, b, rel, imp, rel_id))

    if not candidate_edges:
        # 没有社区内边，则仅输出实体列表（按度数粗排）
        degrees = []
        for n in member_set:
            deg = sum(1 for nbr in graph.neighbors(n) if nbr in member_set)
            degrees.append((n, deg))
        degrees.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [n for n, _ in degrees[:max_entities]]
        entity_lines = [f"- {graph.nodes[n].get('name') or str(n)}" for n in selected_nodes]
        return "Entities:\n" + "\n".join(entity_lines) + "\n\nRelationships:\n"

    # 2) 先“选关系”——按 composite_importance 降序，缺失时退化排序
    def edge_sort_key(item):
        a, b, rel, imp = item
        # 没有 imp 时使用负无穷确保排在后面；随后再按 rel / a / b 稳定排序
        imp_key = -(imp if isinstance(imp, (int, float)) else float("-inf"))
        return (imp_key, rel, str(a), str(b))

    candidate_edges.sort(key=edge_sort_key)
    top_edges = candidate_edges[:max_relationships]

    # 3) 基于“已选关系”确定节点优先级
    #    频次（在 top_edges 中的出现次数）为主；社区内度数为辅
    freq = Counter()
    for a, b, _, _, _ in top_edges:
        freq[a] += 1
        freq[b] += 1

    deg_cache = {}
    def community_degree(nid: Any) -> int:
        if nid in deg_cache:
            return deg_cache[nid]
        d = sum(1 for nbr in graph.neighbors(nid) if nbr in member_set)
        deg_cache[nid] = d
        return d

    # 参与了已选关系的节点集合
    involved_nodes = set()
    for a, b, _, _, _ in top_edges:
        involved_nodes.add(a)
        involved_nodes.add(b)

    # 若参与节点数超过上限，则按 (频次 desc, 社区度数 desc, 节点ID) 取前 max_entities
    ordered_nodes = sorted(
        involved_nodes,
        key=lambda n: (freq[n], community_degree(n), str(n)),
        reverse=True
    )
    selected_nodes = ordered_nodes[:max_entities]
    selected_nodes_set = set(selected_nodes)

    # 4) 过滤关系：仅保留端点都落在“最终实体集合”内的 top_edges
    filtered_edges = [(a, b, rel, imp, rel_id) for (a, b, rel, imp, rel_id) in top_edges
                      if a in selected_nodes_set and b in selected_nodes_set]

    # 如果由于实体截断导致一个关系都不剩，可以放宽到“至少一端在实体内”，再取前N，尽量不空
    if not filtered_edges:
        relaxed_edges = [(a, b, rel, imp, rel_id) for (a, b, rel, imp, rel_id) in top_edges
                         if (a in selected_nodes_set or b in selected_nodes_set)]
        filtered_edges = relaxed_edges[:max_relationships]

    # 5) 渲染文本
    def label(nid: Any) -> str:
        nm = graph.nodes[nid].get("name")
        return f"{nm} ({nid})" if nm else str(nid)

    entity_lines = [f"- {label(n)}" for n in selected_nodes]

    rel_lines = []
    for a, b, rel, imp, rel_id in filtered_edges:
        if include_importance and isinstance(imp, (int, float)):
            rel_lines.append(f"- {label(a)} -[{rel} | score={imp} | id={rel_id}]-> {label(b)}")
        else:
            rel_lines.append(f"- {label(a)} -[{rel} | id={rel_id}]-> {label(b)}")

    # 6) 输出
    parts = ["Entities:", "\n".join(entity_lines) if entity_lines else "",
             "\nRelationships:", "\n".join(rel_lines) if rel_lines else ""]
    return "\n".join(parts)



def create_batch_requests(
        graph: nx.DiGraph,
        model_name: str,
        output_path: Path,
        request_type: str,
        prompt_dir: str | None = None,
        max_report_words: str | None = None,
        max_relationships: int | None = None,
        max_entities: int | None = None) -> int:
    """创建批量请求并写入本地 JSONL 文件。可用于实体消歧或社区总结。"""
    logging.info(f"正在为 '{request_type}' 创建批量请求...")
    requests = []
    if request_type == "disambiguation":
        for node_id in graph.nodes():
            prompt = create_disambiguation_prompt(node_id, graph)
            requests.append(
                {"key": node_id,
                 "request": {"model": f"models/{model_name}", "contents": {"parts": [{"text": prompt}]}}})
    elif request_type == "community_summary":
        communities = {}
        for node, data in graph.nodes(data=True):
            community_id = data.get("community")
            if community_id is not None:
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

        for comm_id, members in communities.items():
            _max_entities = 25 if (max_entities is None or max_entities <= 0) else max_entities
            _max_relationships = 50 if (max_relationships is None or max_relationships <= 0) else max_relationships

            community_prompt = Template(load_prompt(prompt_dir, "community_summary.md"))
            context = build_community_context(graph, members, _max_entities, _max_relationships)
            prompt = community_prompt.substitute(max_report_len=max_report_words, context=context)

            requests.append(
                {"key": comm_id,
                 "request": {"model": f"models/{model_name}", "contents": {"parts": [{"text": prompt}]}}})

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    logging.info(f"已为 {len(requests)} 个 '{request_type}' 任务生成请求，并写入到 {output_path}")
    return len(requests)


def submit_and_monitor_job(client: genai.Client, input_file_path: Path, model_name: str, sleep_interval: int,
                           job_type: str) -> Any:
    """上传文件，创建并监控批量作业。"""
    logging.info(f"📤 [{job_type}] 正在上传请求文件: {input_file_path.name}...")
    try:
        uploaded_file = client.files.upload(
            file=str(input_file_path),
            config={
                "display_name": f'{job_type}-batch-{input_file_path.stem}',
                "mime_type": 'application/jsonl'
            }
        )
        logging.info(f"✅ [{job_type}] 文件上传成功: {uploaded_file.name}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 文件上传失败: {e}")
        return None

    logging.info(f"🚀 [{job_type}] 正在使用模型 '{model_name}' 创建批量作业...")
    try:
        batch_job = client.batches.create(
            model=f"models/{model_name}",
            src=uploaded_file.name,
            config={
                'display_name': f"{job_type}-job-{input_file_path.stem}",
            },
        )
        logging.info(f"✅ [{job_type}] 批量作业已创建: {batch_job.name}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 创建批量作业失败: {e}")
        client.files.delete(name=uploaded_file.name)  # 清理已上传的文件
        return None

    job_name = batch_job.name
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    logging.info(f"⏳ [{job_type}] 开始轮询作业 '{job_name}' 状态，每 {sleep_interval} 秒检查一次...")
    while True:
        try:
            batch_job_status = client.batches.get(name=job_name)
            current_state = batch_job_status.state.name
            logging.info(f"  - [{job_type}] 当前状态: {current_state}")
            if current_state in completed_states:
                break
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - [{job_type}] 轮询失败: {e}")
            time.sleep(sleep_interval * 2)

    return batch_job_status


def process_results(batch_job_status: Any, client: genai.Client) -> Dict[str, str]:
    """下载并处理批量作业的结果。"""
    if not batch_job_status or batch_job_status.state.name != 'JOB_STATE_SUCCEEDED':
        logging.error(f"❌ 作业失败，无法处理结果。")
        if batch_job_status: logging.error(f"  - 失败原因: {batch_job_status.error}")
        return {}

    results_dict = {}
    result_file_name = batch_job_status.dest.file_name
    logging.info(f"📥 正在下载结果文件: {result_file_name}")
    try:
        file_content = client.files.download(file=result_file_name).decode('utf-8')
        processed_count, error_count = 0, 0
        for line in file_content.strip().split('\n'):
            try:
                result = json.loads(line)
                key = result.get("key")
                if key and result.get("response"):
                    text = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    results_dict[key] = text.strip()
                    processed_count += 1
                elif result.get("error"):
                    logging.error(f"  - ❌ 处理 ID '{key}' 时发生错误: {result['error']['message']}")
                    error_count += 1
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logging.warning(f"  - ⚠️ 解析结果行失败: '{line}', 错误: {e}")
                error_count += 1
        logging.info(f"🎉 结果处理完成！成功获取 {processed_count} 条记录，失败 {error_count} 条。")
    except Exception as e:
        logging.error(f"❌ 下载或处理结果文件失败: {e}")
        return {}

    return results_dict


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


# --- 社区发现与总结 ---
def detect_communities(graph: nx.DiGraph, weight_alpha: float) -> nx.DiGraph:
    """使用Leiden算法进行社区发现并标注图谱"""
    logging.info("开始执行社区发现 (Leiden 算法)...")
    if not graph.edges:
        logging.warning("图中没有边，无法进行社区发现。")
        return graph

    undirected_graph = graph.to_undirected()
    igraph_graph = ig.Graph.from_networkx(undirected_graph)
    partition = la.find_partition(igraph_graph, la.ModularityVertexPartition)
    logging.info(f"社区发现完成！共发现 {len(partition)} 个社区。")

    node_mapping = {i: name for i, name in enumerate(igraph_graph.vs["_nx_name"])}
    community_map = {node_mapping[node_index]: str(community_id) for community_id, community in enumerate(partition) for
                     node_index in community}

    nx.set_node_attributes(graph, community_map, "community")

    for community_id, community in enumerate(partition):
        community_nodes = [node_mapping[idx] for idx in community]
        subgraph = undirected_graph.subgraph(community_nodes)
        for node_name in community_nodes:
            degree = subgraph.degree(node_name)
            graph.nodes[node_name]["degree"] = degree

    communities = [[node_mapping[idx] for idx in community] for community in partition]
    ci.calculate_community_importance(graph, communities, weight_alpha=weight_alpha)

    logging.info("已将社区ID、节点degree和关系重要性分数标注到图谱节点。")
    return graph


def save_community_reports(reports: Dict[str, str], output_path: Path):
    """将社区报告保存为CSV文件"""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(list(reports.items()), columns=['community_id', 'summary'])
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"💾 社区报告已保存至: {output_path}")


# --- 保存最终结果 ---
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
    proxy = config.get("proxy")
    client = create_gemini_client(api_key, proxy)
    sleep_interval = config["graph_builder"]["sleep_interval"]
    model_name = config["llm"]["model"]
    prompt_dir = config["graph_builder"]["prompt_dir"]
    weight_alpha = config["graph_builder"]["community_importance_weight_alpha"]
    max_report_words = str(config["graph_builder"].get("community_summary_max_report_words", 800))
    max_entities = int(config["graph_builder"].get("community_summary_used_entities_num", 25))
    max_relationships = int(config["graph_builder"].get("community_summary_used_relationships_num", 50))
    logging.info("Gemini 客户端初始化完成。")

    # --- 2. 加载和构建初始图 ---
    input_path = PROJECT_ROOT / config["graph_builder"]["input_path"]
    graph = load_and_build_initial_graph(input_path)
    if not graph.nodes:
        logging.error("图为空，流程终止。请检查输入文件。")
        return

    # --- 3. 实体消歧 ---
    logging.info("--- 开始执行阶段1: 实体消歧 ---")
    disambiguation_requests_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_requests_path"]
    create_batch_requests(graph=graph, model_name=model_name, output_path=disambiguation_requests_path, request_type="disambiguation")

    disambiguation_job = submit_and_monitor_job(client, disambiguation_requests_path, model_name, sleep_interval,
                                                "Disambiguation")
    new_descriptions = process_results(disambiguation_job, client)

    if new_descriptions:
        update_graph_with_new_descriptions(graph, new_descriptions)
    else:
        logging.warning("未能获取任何新的实体描述，图谱未更新。")
    logging.info("--- 实体消歧阶段完成 ---")

    # --- 4. 社区发现与总结 ---
    logging.info("--- 开始执行阶段2: 社区发现与总结 ---")
    graph_with_communities = detect_communities(graph,weight_alpha)

    community_requests_path = PROJECT_ROOT / config["graph_builder"]["community_requests_path"]
    num_communities = create_batch_requests(
        graph=graph_with_communities,
        model_name=model_name,
        output_path=community_requests_path,
        request_type="community_summary",
        prompt_dir=prompt_dir,
        max_report_words=max_report_words,
        max_relationships=max_relationships,
        max_entities=max_entities,
    )

    if num_communities > 0:
        community_job = submit_and_monitor_job(client, community_requests_path, model_name, sleep_interval,
                                               "CommunitySummary")
        community_summaries = process_results(community_job, client)

        if community_summaries:
            reports_path = PROJECT_ROOT / config["graph_builder"]["community_reports_path"]
            save_community_reports(community_summaries, reports_path)
    else:
        logging.warning("图中未发现社区，跳过摘要生成步骤。")
    logging.info("--- 社区发现与总结阶段完成 ---")

    # --- 5. 保存最终图谱 ---
    final_graph_path = PROJECT_ROOT / config["graph_builder"]["output_graph_path"]
    save_final_graph(graph_with_communities, final_graph_path)

    logging.info("🎉🎉🎉 图谱构建与增强流程全部完成！ 🎉🎉🎉")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)