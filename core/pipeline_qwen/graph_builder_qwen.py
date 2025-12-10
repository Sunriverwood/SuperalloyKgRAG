import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import networkx as nx
import pandas as pd
import yaml
from collections import Counter, defaultdict
from dataclasses import dataclass

# 修改：使用 OpenAI SDK
from openai import OpenAI

# 社区发现依赖（保持不变）
import leidenalg as la
import igraph as ig

import utils.community_importance as ci

# --- 动态计算项目根目录 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================
# 通用：日志与配置
# =========================

def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/graph_builder.log")
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


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "config" / settings_filename
    logging.info(f"正在从 {cfg_path} 加载配置...")
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件 {cfg_path} 未找到！")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info("配置加载成功。")
    return config


def load_prompt(prompt_dir: str, filename: str) -> str:
    p = PROJECT_ROOT / prompt_dir / filename
    if not p.exists():
        raise FileNotFoundError(f"Prompt文件未找到: {p}")
    return p.read_text(encoding='utf-8')


# =========================
# ID规范化工具
# =========================

def normalize_entity_id(entity_id: str) -> str:
    if not entity_id:
        return entity_id
    return str(entity_id).replace('_', '-')


def try_find_node_with_normalization(G: nx.DiGraph, entity_id: str) -> str | None:
    if G.has_node(entity_id):
        return entity_id
    normalized_id = normalize_entity_id(entity_id)
    if normalized_id != entity_id and G.has_node(normalized_id):
        return normalized_id
    if '-e-' in entity_id:
        parts = entity_id.rsplit('-e-', 1)
        if len(parts) == 2:
            reverse_id = f"{parts[0]}_e-{parts[1]}"
            if G.has_node(reverse_id):
                return reverse_id
    return None


# =========================
# 阶段0：加载与初始图构建
# =========================

def load_and_build_initial_graph(jsonl_path: Path) -> nx.DiGraph:
    """从jsonl加载实体/关系并创建全局唯一ID，构建初始有向图。"""
    logging.info(f"从 {jsonl_path} 加载图数据并创建全局唯一ID...")
    G = nx.DiGraph()
    all_nodes, all_edges = [], []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk_id = data.get("id")
            if not chunk_id:
                logging.warning("发现一行缺少'id'字段，跳过。")
                continue

            id_map: Dict[str, str] = {}
            g = data.get("graph", {})

            if isinstance(g, list):
                if g and isinstance(g[0], dict):
                    g = g[0]
                else:
                    continue

            if not isinstance(g, dict):
                continue

            for ent in g.get("entities", []):
                local_id = ent.get('id')
                if local_id is None:
                    continue
                gid = f"{chunk_id}-{local_id}"
                id_map[local_id] = gid
                new_ent = {**ent, 'id': gid, 'original_id': local_id, 'chunk_id': chunk_id}
                all_nodes.append(new_ent)

            for rel in g.get("relationships", []):
                s, t = rel.get('source'), rel.get('target')
                if s in id_map and t in id_map:
                    gid = f"{chunk_id}-{rel.get('id')}"
                    new_rel = {
                        **rel,
                        'id': gid,
                        'source': id_map[s],
                        'target': id_map[t],
                        'original_id': rel.get('id'),
                        'chunk_id': chunk_id,
                    }
                    all_edges.append(new_rel)

    if not all_nodes:
        logging.warning("未能从文件中加载任何节点。")
        return G

    ent_df = pd.DataFrame(all_nodes).drop_duplicates(subset=['id'])
    rel_df = pd.DataFrame(all_edges).drop_duplicates(subset=['id']) if all_edges else pd.DataFrame()

    logging.info(f"加载了 {ent_df.shape[0]} 个唯一实体和 {rel_df.shape[0]} 个唯一关系。")

    for _, row in ent_df.iterrows():
        G.add_node(row['id'], **{k: v for k, v in row.items() if k != 'id'})
    if not rel_df.empty:
        for _, row in rel_df.iterrows():
            if G.has_node(row['source']) and G.has_node(row['target']):
                G.add_edge(row['source'], row['target'],
                           **{k: v for k, v in row.items() if k not in ('source', 'target')})

    logging.info(f"初始图构建完成：{G.number_of_nodes()} 节点, {G.number_of_edges()} 边。")
    return G


# =========================
# 阶段1：实体消歧
# =========================

def create_disambiguation_prompt(node_id: str, G: nx.DiGraph) -> str:
    ndata = G.nodes[node_id]
    ctx = [
        f"实体名称: {ndata.get('name', 'N/A')}",
        f"实体类型: {ndata.get('type', 'N/A')}",
        f"原始描述: {ndata.get('description', 'N/A')}"
    ]
    rels = []
    for _, v, ed in G.edges(node_id, data=True):
        rels.append(f"- 与 '{G.nodes[v].get('name', 'N/A')}' 的关系是 '{ed.get('relationship', 'N/A')}'")
    for u, _, ed in G.in_edges(node_id, data=True):
        rels.append(f"- '{G.nodes[u].get('name', 'N/A')}' 与它的关系是 '{ed.get('relationship', 'N/A')}'")
    if rels:
        ctx.append("\n关系上下文:")
        ctx.extend(rels)
    prompt = (
        "基于以下上下文，为实体生成一个全面、精准、消除歧义的标准化描述，像百科定义一样，不超过100字。\n"
        f"上下文:\n{'\n'.join(ctx)}\n"
        "标准化描述:"
    )
    return prompt


def create_batch_requests(
        graph: nx.DiGraph,
        model_name: str,
        output_path: Path,
        request_type: str,
        prompt_dir: str | None = None,
        max_report_words: str | None = None,
        max_relationships: int | None = None,
        max_entities: int | None = None,
        batch_size: int | None = None,
) -> int:
    """
    创建批量请求 (OpenAI 兼容格式)，支持分批处理

    Args:
        graph: 图对象
        model_name: 模型名称
        output_path: 输出路径
        request_type: 请求类型
        prompt_dir: prompt目录
        max_report_words: 最大报告字数
        max_relationships: 最大关系数
        max_entities: 最大实体数
        batch_size: 单次批量请求的最大数量，如果为None则不分批

    Returns:
        生成的请求数量
    """
    logging.info(f"正在为 '{request_type}' 创建批量请求...")
    requests: List[Dict[str, Any]] = []

    # 准备 Prompt 列表
    prompts_map = {}  # custom_id -> prompt_content

    if request_type == "disambiguation":
        for nid in graph.nodes():
            prompts_map[str(nid)] = create_disambiguation_prompt(nid, graph)

    elif request_type == "community_summary":
        communities: Dict[str, List[str]] = defaultdict(list)
        for node, data in graph.nodes(data=True):
            cid = data.get("community")
            if cid is not None:
                communities[str(cid)].append(node)

        community_prompt = Template(load_prompt(prompt_dir, "community_summary.md"))
        _max_entities = 25 if not max_entities or max_entities <= 0 else max_entities
        _max_relationships = 50 if not max_relationships or max_relationships <= 0 else max_relationships

        all_id_maps: Dict[str, Dict[str, str]] = {}

        for comm_id, members in communities.items():
            context, id_map = build_community_context(graph, members, _max_entities, _max_relationships)
            prompt = community_prompt.substitute(max_report_len=max_report_words or "1000", context=context)
            prompts_map[str(comm_id)] = prompt
            all_id_maps[comm_id] = id_map

        # 保存映射
        id_map_path = output_path.parent / f"{output_path.stem}_id_maps.json"
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(all_id_maps, f, ensure_ascii=False, indent=2)
        logging.info(f"社区ID映射已保存至: {id_map_path}")

    elif request_type == "entity_merge":
        raise RuntimeError("请使用 create_entity_merge_requests 生成实体合并仲裁请求。")
    else:
        raise ValueError(f"未知 request_type: {request_type}")

    # 修改：构建 OpenAI Batch Request
    for key, prompt_text in prompts_map.items():
        request_line = {
            "custom_id": key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.1  # 适当降低随机性
            }
        }
        requests.append(request_line)

    # 支持分批处理
    total_requests = len(requests)
    if batch_size and total_requests > batch_size:
        num_batches = (total_requests + batch_size - 1) // batch_size
        logging.info(f"⚙️ 请求数量 {total_requests} 超过批次大小 {batch_size}，将拆分为 {num_batches} 个批次处理")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_requests)
            batch_requests = requests[start_idx:end_idx]

            # 为每个批次创建单独的文件
            batch_output_path = output_path.parent / f"{output_path.stem}_batch_{batch_idx + 1}.jsonl"

            with open(batch_output_path, 'w', encoding='utf-8') as f:
                for req in batch_requests:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")

            logging.info(f"✅ 批次 {batch_idx + 1}/{num_batches} 已写入 {len(batch_requests)} 个请求: {batch_output_path}")

        return total_requests
    else:
        # 不分批，直接处理
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        logging.info(f"已为 {len(requests)} 个 '{request_type}' 任务写入 {output_path}")
        return len(requests)


def submit_and_monitor_job(client: OpenAI, input_file_path: Path, model_name: str, sleep_interval: int,
                           job_type: str, monitor: bool = True) -> Any:
    """
    提交并监控批量作业 (OpenAI SDK)。
    适用于 Chat Completion 任务 (disambiguation, community summary, merge)。

    Args:
        client: OpenAI客户端
        input_file_path: 输入文件路径
        model_name: 模型名称
        sleep_interval: 轮询间隔
        job_type: 作业类型
        monitor: 是否立即监控，False时只提交不等待完成

    Returns:
        作业状态对象
    """
    logging.info(f"📤 [{job_type}] 正在上传请求文件: {input_file_path.name}...")
    try:
        # 修改：client.files.create
        with open(input_file_path, "rb") as f:
            up = client.files.create(file=f, purpose="batch")
        logging.info(f"✅ [{job_type}] 文件上传成功: {up.id}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 文件上传失败: {e}")
        return None

    logging.info(f"🚀 [{job_type}] 正在使用模型 '{model_name}' 创建批量作业...")
    try:
        # 修改：client.batches.create, 指定 endpoint
        job = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={'description': f"{job_type}-job-{input_file_path.stem}"}
        )
        logging.info(f"✅ [{job_type}] 批量作业已创建: {job.id}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 创建批量作业失败: {e}")
        return None

    # 如果不需要立即监控，直接返回作业对象
    if not monitor:
        return job

    # 监控作业完成
    return _monitor_job_completion(client, job, sleep_interval, job_type)


def _monitor_job_completion(client: OpenAI, job: Any, sleep_interval: int, job_type: str) -> Any:
    """
    监控批量作业完成状态

    Args:
        client: OpenAI客户端
        job: 作业对象
        sleep_interval: 轮询间隔
        job_type: 作业类型

    Returns:
        完成后的作业状态
    """
    completed = {'completed', 'failed', 'cancelled', 'expired'}
    logging.info(f"⏳ [{job_type}] 开始轮询作业 '{job.id}' 状态，每 {sleep_interval} 秒检查一次...")
    while True:
        try:
            status = client.batches.retrieve(batch_id=job.id)
            state = status.status
            logging.debug(f"  - [{job_type}] 当前状态: {state}")
            if state in completed:
                logging.info(f"✅ [{job_type}] 作业完成，最终状态: {state}")
                return status
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - [{job_type}] 轮询失败: {e}")
            time.sleep(max(2, sleep_interval * 2))


def process_results(batch_job_status: Any, client: OpenAI) -> Dict[str, str]:
    """下载并处理批量作业的结果 (OpenAI 格式)，返回 {custom_id: text}。"""
    if not batch_job_status or batch_job_status.status != 'completed':
        logging.error("❌ 作业失败，无法处理结果。")
        if hasattr(batch_job_status, 'errors') and batch_job_status.errors:
            logging.error(f"  - 失败原因: {batch_job_status.errors}")
        return {}

    out: Dict[str, str] = {}
    output_file_id = batch_job_status.output_file_id
    if not output_file_id:
        logging.error("❌ 作业完成但没有 output_file_id")
        return {}

    logging.info(f"📥 正在下载结果文件: {output_file_id}")
    try:
        # 修改：client.files.content
        content = client.files.content(output_file_id).text
        ok, bad = 0, 0
        for line in str(content).strip().split('\n'):
            try:
                obj = json.loads(line)
                key = obj.get("custom_id")
                response = obj.get("response", {})

                if key and response.get("status_code") == 200:
                    # 解析 OpenAI 响应结构
                    # response -> body -> choices[0] -> message -> content
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                        # 确保text是字符串类型
                        out[key] = str(text or "").strip()
                        ok += 1
                    else:
                        logging.warning(f"ID {key} 没有 choices")
                        bad += 1
                else:
                    err = obj.get("error") or response.get("body")
                    logging.error(f"  - ❌ 处理 ID '{key}' 时发生错误: {err}")
                    bad += 1
            except Exception as e:
                logging.warning(f"  - ⚠️ 解析结果行失败: '{line[:120]}...' 错误: {e}")
                bad += 1
        logging.info(f"🎉 结果处理完成：成功 {ok} 条，失败 {bad} 条。")
    except Exception as e:
        logging.error(f"❌ 下载或处理结果文件失败: {e}")
        return {}
    return out


# =========================
# 向量聚类 + LLM 仲裁 的实体合并
# =========================

# --- A) 临时嵌入（仅用于候选簇发现，不入库） ---

def _create_temp_embedding_requests(entities: List[Tuple[str, str]], output_path: Path, model_name: str,
                                    dim: int = 768) -> None:
    """
    entities: List[(id, text)] 其中 text = name + \n + desc
    生成 Embeddings 批量请求 JSONL (OpenAI 格式)。
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid, text in entities:
            # 修改：OpenAI Batch Embedding Request
            req = {
                "custom_id": str(eid),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model_name,
                    "input": text,
                    "dimensions": dim  # text-embedding-v3 支持
                }
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _submit_and_monitor_embedding_job(client: OpenAI, requests_path: Path, model_name: str, sleep_interval: int,
                                      batch_index: int = 0, total_batches: int = 1, monitor: bool = True):
    """
    提交单个批量嵌入作业并监控 (OpenAI SDK)。

    Args:
        client: OpenAI客户端
        requests_path: 请求文件路径
        model_name: 嵌入模型名称
        sleep_interval: 轮询间隔
        batch_index: 批次索引
        total_batches: 总批次数
        monitor: 是否立即监控，False时只提交不等待完成

    Returns:
        作业状态对象
    """
    batch_tag = f"[批次 {batch_index + 1}/{total_batches}]" if total_batches > 1 else ""
    logging.info(f"📤 [EntityEmb] {batch_tag} 上传临时嵌入请求: {requests_path.name} ...")
    try:
        # 修改：client.files.create
        with open(requests_path, "rb") as f:
            up = client.files.create(file=f, purpose="batch")
        logging.info(f"✅ [EntityEmb] {batch_tag} 文件上传成功: {up.id}")
    except Exception as e:
        logging.error(f"❌ [EntityEmb] {batch_tag} 文件上传失败: {e}")
        return None

    logging.info(f"🚀 [EntityEmb] {batch_tag} 创建批量嵌入作业...")
    try:
        # 修改：client.batches.create, endpoint="/v1/embeddings"
        job = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={'description': f"emb-entity-job-{requests_path.stem}"}
        )
        logging.info(f"✅ [EntityEmb] {batch_tag} 作业已创建: {job.id}")
    except Exception as e:
        logging.error(f"❌ [EntityEmb] {batch_tag} 创建嵌入作业失败: {e}")
        return None

    # 如果不需要立即监控，直接返回作业对象
    if not monitor:
        return job

    # 监控作业完成
    return _monitor_embedding_job_completion(client, job, sleep_interval, batch_index, total_batches)


def _monitor_embedding_job_completion(client: OpenAI, job: Any, sleep_interval: int,
                                      batch_index: int = 0, total_batches: int = 1) -> Any:
    """
    监控嵌入作业完成状态

    Args:
        client: OpenAI客户端
        job: 作业对象
        sleep_interval: 轮询间隔
        batch_index: 批次索引
        total_batches: 总批次数

    Returns:
        完成后的作业状态
    """
    batch_tag = f"[批次 {batch_index + 1}/{total_batches}]" if total_batches > 1 else ""
    done = {'completed', 'failed', 'cancelled', 'expired'}
    logging.info(f"⏳ [EntityEmb] {batch_tag} 轮询 '{job.id}' 状态，每 {sleep_interval} 秒...")
    while True:
        try:
            st = client.batches.retrieve(batch_id=job.id)
            if st.status in done:
                logging.info(f"✅ [EntityEmb] {batch_tag} 作业完成，最终状态: {st.status}")
                return st
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - [EntityEmb] {batch_tag} 轮询失败: {e}")
            time.sleep(max(2, sleep_interval * 2))


def _process_embedding_results(batch_job: Any, client: OpenAI, id_order: List[str]) -> np.ndarray:
    if not batch_job or getattr(batch_job, 'status', None) != 'completed':
        logging.error("❌ 临时嵌入作业失败或未执行。")
        if batch_job and getattr(batch_job, 'errors', None):
            logging.error(f"  - 失败原因: {batch_job.errors}")
        return np.zeros((0, 1), dtype=float)

    output_file_id = batch_job.output_file_id
    if not output_file_id:
        return np.zeros((0, 1), dtype=float)

    logging.info(f"📥 [{batch_job.id}] 正在下载结果文件: {output_file_id}")

    # 修改：client.files.content
    try:
        file_content = client.files.content(output_file_id).text
    except Exception as e:
        logging.error(f"下载 Embedding 结果失败: {e}")
        return np.zeros((0, 1), dtype=float)

    lines = str(file_content).strip().split('\n')

    # 建立映射以保证顺序
    res_map = {}
    for line in lines:
        try:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            response = obj.get("response", {})
            if response.get("status_code") == 200:
                body = response.get("body", {})
                data = body.get("data", [])
                if data:
                    res_map[cid] = np.array(data[0]["embedding"], dtype=float)
        except:
            pass

    vecs: List[np.ndarray] = []
    # 如果 dimensions 参数生效，这里维度应该一致；初始化一个默认以防万一
    # 但由于我们无法预知维度（除非传入），这里假设第一条成功数据的维度
    default_dim = 768
    if res_map:
        default_dim = next(iter(res_map.values())).shape[0]

    for eid in id_order:
        if str(eid) in res_map:
            emb = res_map[str(eid)]
            # 归一化 (阿里云 text-embedding-v3/v4 通常已归一化，但保留逻辑)
            n = np.linalg.norm(emb)
            vecs.append(emb / (n if n > 0 else 1.0))
        else:
            logging.warning(f"⚠️ 未找到实体 {eid} 的嵌入结果")
            vecs.append(np.zeros(default_dim, dtype=float))

    if not vecs:
        return np.zeros((0, 1), dtype=float)

    return np.vstack(vecs)


# --- B) 候选簇发现（互为近邻 + 相似度阈值） ---
# (此部分逻辑是纯数学计算，不需要修改)

def _cosine_sim_matrix(A: np.ndarray) -> np.ndarray:
    return np.clip(A @ A.T, -1.0, 1.0)


def build_candidate_clusters(vecs: np.ndarray, ids: List[str], topk: int = 10, min_sim: float = 0.9) -> List[
    List[str]]:
    n = vecs.shape[0]
    if n == 0: return []
    sims = _cosine_sim_matrix(vecs)
    np.fill_diagonal(sims, -1.0)
    nn_graph = nx.Graph()
    for i in range(n):
        row = sims[i]
        idx = np.argpartition(row, -topk)[-topk:]
        for j in idx:
            if row[j] >= min_sim:
                nn_graph.add_edge(i, j, sim=float(row[j]))
    to_remove = []
    for u, v in nn_graph.edges():
        if not (sims[u, v] >= min_sim and sims[v, u] >= min_sim):
            to_remove.append((u, v))
    nn_graph.remove_edges_from(to_remove)
    clusters: List[List[str]] = []
    for comp in nx.connected_components(nn_graph):
        c = sorted(list(comp))
        if len(c) > 12:
            sub = _split_large_component(vecs[c], [ids[i] for i in c], base_threshold=min_sim + 0.03)
            clusters.extend(sub)
        elif len(c) >= 2:
            clusters.append([ids[i] for i in c])
    return clusters


def _split_large_component(vecs: np.ndarray, ids: List[str], base_threshold: float) -> List[List[str]]:
    sims = _cosine_sim_matrix(vecs)
    np.fill_diagonal(sims, -1.0)
    G = nx.Graph()
    m, _ = sims.shape
    for i in range(m):
        for j in range(i + 1, m):
            if sims[i, j] >= base_threshold:
                G.add_edge(i, j)
    out: List[List[str]] = []
    for comp in nx.connected_components(G):
        cc = list(comp)
        if len(cc) >= 2:
            out.append([ids[i] for i in cc])
    return out


# --- C) LLM 仲裁 ---

@dataclass
class LLMResolutionGroup:
    canonical_name: str
    member_ids: List[str]
    rationale: str | None = None


def _build_merge_prompt_for_cluster(G: nx.DiGraph, member_ids: List[str], prompt_dir: str) -> str:
    tpl = load_prompt(prompt_dir, "entity_disambiguation.md")
    payload = []
    for eid in member_ids:
        n = G.nodes[eid]
        item = {
            "id": eid,
            "name": n.get("name", ""),
            "aliases": list(set(n.get("aliases", []) + ([n.get("name")] if n.get("name") else []))),
            "desc": n.get("description", ""),
            "type": n.get("type", "")
        }
        payload.append(item)
    body = json.dumps({"entities": payload}, ensure_ascii=False)
    return Template(tpl).substitute(payload=body)


def create_entity_merge_requests(G: nx.DiGraph, clusters: List[List[str]], model_name: str, prompt_dir: str,
                                 output_path: Path, batch_size: int = None) -> int:
    """
    创建实体合并请求，支持分批处理

    Args:
        G: 图对象
        clusters: 候选簇列表
        model_name: 模型名称
        prompt_dir: prompt目录
        output_path: 输出路径
        batch_size: 单次批量请求的最大数量，如果为None则不分批

    Returns:
        生成的请求数量
    """
    # 修改：OpenAI Batch Request Format
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cnt = 0

    # 如果指定了batch_size，则分批处理
    if batch_size and len(clusters) > batch_size:
        num_batches = (len(clusters) + batch_size - 1) // batch_size
        logging.info(f"⚙️ 候选簇数量 {len(clusters)} 超过批次大小 {batch_size}，将拆分为 {num_batches} 个批次处理")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(clusters))
            batch_clusters = clusters[start_idx:end_idx]

            # 为每个批次创建单独的文件
            batch_output_path = output_path.parent / f"{output_path.stem}_batch_{batch_idx + 1}.jsonl"

            with open(batch_output_path, 'w', encoding='utf-8') as f:
                for local_idx, member_ids in enumerate(batch_clusters):
                    global_idx = start_idx + local_idx
                    prompt = _build_merge_prompt_for_cluster(G, member_ids, prompt_dir)
                    req = {
                        "custom_id": f"cluster_{global_idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "response_format": {"type": "json_object"}  # 强制 JSON
                        }
                    }
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    cnt += 1

            logging.info(f"✅ 批次 {batch_idx + 1}/{num_batches} 已生成 {len(batch_clusters)} 个请求: {batch_output_path}")
    else:
        # 不分批，直接处理
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, member_ids in enumerate(clusters):
                prompt = _build_merge_prompt_for_cluster(G, member_ids, prompt_dir)
                req = {
                    "custom_id": f"cluster_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"}  # 强制 JSON
                    }
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                cnt += 1
        logging.info(f"为 {cnt} 个候选簇生成了 LLM 仲裁请求: {output_path}")

    return cnt


def _safe_json_loads(text: str) -> Dict[str, Any] | None:
    # ... (原有逻辑保持不变)
    try:
        return json.loads(text)
    except:
        pass
    if "```" in text:
        patterns = [r'```json\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```', r'```json\s*(.*?)```', r'```(.*?)```']
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    content = str(match).strip()
                    if content: return json.loads(content)
                except:
                    continue
        parts = str(text).split("```")
        for chunk in parts:
            chunk = str(chunk).strip()
            if not chunk or chunk.lower() in ['json', 'json']: continue
            try:
                return json.loads(chunk)
            except:
                continue
    try:
        json_match = re.search(r'\{.*}', text, re.DOTALL)
        if json_match: return json.loads(json_match.group())
    except:
        pass
    return None


def parse_entity_merge_results(raw_texts: Dict[str, str]) -> List[LLMResolutionGroup]:
    groups: List[LLMResolutionGroup] = []
    for key, text in raw_texts.items():
        obj = _safe_json_loads(text)
        if not obj or "groups" not in obj:
            logging.warning(f"LLM 返回非期望JSON，key={key}，已跳过。片段: {text[:100]}...")
            continue
        for g in obj.get("groups", []):
            try:
                cname = str(g.get("canonical_name") or '').strip()
                mids = [str(x) for x in g.get("member_ids", []) if x]
                rationale = str(g.get("rationale") or '').strip() or None
                if cname and len(mids) >= 1:
                    groups.append(LLMResolutionGroup(cname, mids, rationale))
            except Exception as e:
                logging.warning(f"解析分组失败: {e}")
    return groups


# --- D) 应用合并到图 (保持不变) ---

def _choose_representative(G: nx.DiGraph, group: LLMResolutionGroup) -> str | None:
    lc = group.canonical_name.lower()
    valid_member_ids = []
    for eid in group.member_ids:
        actual_id = try_find_node_with_normalization(G, eid)
        if actual_id:
            valid_member_ids.append(actual_id)
    if not valid_member_ids:
        logging.warning(f"⚠️ 分组 '{group.canonical_name}' 的所有成员ID都不存在于图中: {group.member_ids}")
        return None
    for eid in valid_member_ids:
        if (G.nodes[eid].get('name') or '').lower() == lc:
            return eid
    best = (-1, None)
    for eid in valid_member_ids:
        deg = G.degree(eid)
        if deg > best[0]:
            best = (deg, eid)
    return best[1] or valid_member_ids[0]


def build_merge_map(G: nx.DiGraph, groups: List[LLMResolutionGroup]) -> Tuple[Dict[str, str], Dict[str, str]]:
    alias2canon: Dict[str, str] = {}
    canon_name: Dict[str, str] = {}
    skipped_groups = 0
    for g in groups:
        try:
            cid = _choose_representative(G, g)
            if cid is None:
                skipped_groups += 1
                continue
            valid_members = []
            for mid in g.member_ids:
                actual_id = try_find_node_with_normalization(G, mid)
                if actual_id:
                    valid_members.append(actual_id)
            if not valid_members:
                skipped_groups += 1
                continue
            canon_name[cid] = g.canonical_name
            for mid in valid_members:
                alias2canon[mid] = cid
        except Exception as e:
            logging.error(f"❌ 处理分组 '{g.canonical_name}' 时出错: {e}")
            skipped_groups += 1
            continue
    if skipped_groups > 0:
        logging.warning(f"⚠️ 共跳过 {skipped_groups} 个无效分组")
    return alias2canon, canon_name


def apply_entity_merge(G: nx.DiGraph, alias2canon: Dict[str, str], canon_name_map: Dict[str, str],
                       edge_agg: str = "max") -> nx.DiGraph:
    if not alias2canon: return G
    groups: Dict[str, List[str]] = defaultdict(list)
    for aid, cid in alias2canon.items():
        groups[cid].append(aid)
    for cid, members in groups.items():
        members = list(dict.fromkeys(members))
        if cid not in members: members.insert(0, cid)
        aliases: set[str] = set()
        descs: List[str] = []
        provenance: List[Any] = []
        types: Counter = Counter()
        names_in_group: List[str] = []
        for eid in members:
            nd = G.nodes[eid]
            nm = str(nd.get('name') or '').strip()
            if nm:
                names_in_group.append(nm)
                aliases.add(nm)
            aliases.update(nd.get('aliases', []) or [])
            d = str(nd.get('description') or '').strip()
            if d: descs.append(d)
            pv = nd.get('provenance') or []
            if isinstance(pv, list):
                provenance.extend(pv)
            elif isinstance(pv, dict):
                provenance.append(pv)
            tp = str(nd.get('type') or '').strip()
            if tp: types[tp] += 1
        main_name = str(canon_name_map.get(cid) or '').strip()
        if not main_name:
            if names_in_group:
                main_name = max(Counter(names_in_group).items(), key=lambda x: (x[1], len(x[0])))[0]
            else:
                main_name = cid
        merged_desc = max(descs, key=len) if descs else ''
        merged_aliases = sorted(a for a in aliases if a and a != main_name)
        dominant_type = types.most_common(1)[0][0] if types else (G.nodes[cid].get('type') or '')
        nd0 = G.nodes[cid]
        nd0['name'] = main_name
        nd0['aliases'] = merged_aliases
        nd0['description'] = merged_desc
        nd0['type'] = dominant_type
        nd0['is_disambiguated'] = True
        prov_ids = [{'merged_from': members}]
        if provenance: prov_ids.extend(provenance)
        nd0['provenance'] = prov_ids

    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        if alias2canon.get(n, n) != n: continue
        H.add_node(n, **data)
    tmp_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for u, v, ed in G.edges(data=True):
        uu = alias2canon.get(u, u)
        vv = alias2canon.get(v, v)
        if uu == vv: continue
        key = (uu, vv, ed.get('relationship') or ed.get('rel_type') or 'related_to')
        if key not in tmp_edges:
            tmp_edges[key] = dict(ed)
            tmp_edges[key]['source'] = uu
            tmp_edges[key]['target'] = vv
            try:
                tmp_edges[key]['weight'] = float(ed.get('weight', 1.0))
            except:
                tmp_edges[key]['weight'] = 1.0
            tmp_edges[key]['id'] = ed.get('id') or f"{uu}::{key[2]}::{vv}"
        else:
            if edge_agg == 'sum':
                try:
                    tmp_edges[key]['weight'] += float(ed.get('weight', 1.0))
                except:
                    pass
            elif edge_agg == 'max':
                try:
                    tmp_edges[key]['weight'] = max(float(tmp_edges[key]['weight']), float(ed.get('weight', 1.0)))
                except:
                    pass
            if ed.get('description'):
                prev = tmp_edges[key].get('description', '')
                cur = ed.get('description', '')
                tmp_edges[key]['description'] = prev if len(prev) >= len(cur) else cur
    for (_, _, _), ed in tmp_edges.items():
        H.add_edge(ed['source'], ed['target'], **ed)
    logging.info(
        f"实体合并完成：节点 {G.number_of_nodes()}→{H.number_of_nodes()}，边 {G.number_of_edges()}→{H.number_of_edges()}。")
    return H


# =========================
# 社区发现 + 报告 (保持不变，除请求生成外)
# =========================

def build_community_context(graph: nx.Graph, member_ids: List[str], max_entities: int = 25, max_relationships: int = 50,
                            include_importance: bool = True) -> Tuple[str, Dict[str, str]]:
    # (此函数逻辑保持不变，用于生成 Prompt 的 Context 部分)
    member_set = {n for n in member_ids if n in graph.nodes}
    if not member_set: return "Entities:\n\nRelationships:\n", {}
    local_id_map: Dict[str, str] = {}
    entity_counter = 1
    relation_counter = 1
    candidate_edges = []
    for u, v, ed in graph.edges(data=True):
        if u in member_set and v in member_set:
            rel = ed.get("relationship") or ed.get("relation") or ed.get("rel_type") or "related_to"
            imp = ed.get("composite_importance")
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            rid = ed.get("id")
            candidate_edges.append((a, b, rel, imp, rid))
    if not candidate_edges:
        degrees = []
        for n in member_set:
            deg = sum(1 for nbr in graph.neighbors(n) if nbr in member_set)
            degrees.append((n, deg))
        degrees.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [n for n, _ in degrees[:max_entities]]
        for node_id in selected_nodes:
            if node_id not in local_id_map:
                local_id_map[node_id] = f"E{entity_counter}"
                entity_counter += 1
        entity_lines = [f"- {graph.nodes[n].get('name') or str(n)} ({local_id_map[n]})" for n in selected_nodes]
        return "Entities:\n" + "\n".join(entity_lines) + "\n\nRelationships:\n", local_id_map

    def edge_sort_key(item):
        a, b, rel, imp, _ = item
        imp_key = -(imp if isinstance(imp, (int, float)) else float("-inf"))
        return (imp_key, rel, str(a), str(b))

    candidate_edges.sort(key=edge_sort_key)
    top_edges = candidate_edges[:max_relationships]
    freq = Counter()
    for a, b, *_ in top_edges:
        freq[a] += 1;
        freq[b] += 1
    deg_cache = {}

    def community_degree(nid: Any) -> int:
        if nid in deg_cache: return deg_cache[nid]
        d = sum(1 for nbr in graph.neighbors(nid) if nbr in member_set)
        deg_cache[nid] = d
        return d

    involved = set()
    for a, b, *_ in top_edges: involved.add(a); involved.add(b)
    ordered_nodes = sorted(involved, key=lambda n: (freq[n], community_degree(n), str(n)), reverse=True)
    selected_nodes = ordered_nodes[:max_entities]
    S = set(selected_nodes)
    filtered = [(a, b, rel, imp, rid) for (a, b, rel, imp, rid) in top_edges if a in S and b in S]
    if not filtered:
        relaxed = [(a, b, rel, imp, rid) for (a, b, rel, imp, rid) in top_edges if (a in S or b in S)]
        filtered = relaxed[:max_relationships]
    for node_id in selected_nodes:
        if node_id not in local_id_map:
            local_id_map[node_id] = f"E{entity_counter}"
            entity_counter += 1
    for a, b, rel, imp, rid in filtered:
        if rid and rid not in local_id_map:
            local_id_map[rid] = f"R{relation_counter}"
            relation_counter += 1

    def label(nid: Any) -> str:
        nm = graph.nodes[nid].get("name")
        local_id = local_id_map.get(nid, nid)
        return f"{nm} ({local_id})" if nm else str(local_id)

    entity_lines = [f"- {label(n)}" for n in selected_nodes]
    rel_lines = []
    for a, b, rel, imp, rid in filtered:
        local_rid = local_id_map.get(rid, rid) if rid else "?"
        if include_importance and isinstance(imp, (int, float)):
            rel_lines.append(f"- {label(a)} -[{rel} | score={imp} | id={local_rid}]-> {label(b)}")
        else:
            rel_lines.append(f"- {label(a)} -[{rel} | id={local_rid}]-> {label(b)}")
    parts = ["Entities:", "\n".join(entity_lines) if entity_lines else "", "\nRelationships:",
             "\n".join(rel_lines) if rel_lines else ""]
    reverse_id_map = {v: k for k, v in local_id_map.items()}
    return "\n".join(parts), reverse_id_map


def detect_communities(graph: nx.DiGraph, weight_alpha: float) -> nx.DiGraph:
    logging.info("开始执行社区发现 (Leiden 算法)...")
    if not graph.edges:
        logging.warning("图中没有边，无法进行社区发现。")
        return graph
    UG = graph.to_undirected()
    ig_graph = ig.Graph.from_networkx(UG)
    partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
    logging.info(f"社区发现完成！共发现 {len(partition)} 个社区。")
    node_mapping = {i: name for i, name in enumerate(ig_graph.vs["_nx_name"])}
    community_map = {node_mapping[node_index]: str(community_id) for community_id, community in enumerate(partition) for
                     node_index in community}
    nx.set_node_attributes(graph, community_map, "community")
    for community in partition:
        nodes = [node_mapping[idx] for idx in community]
        sub = UG.subgraph(nodes)
        for nid in nodes: graph.nodes[nid]["degree"] = sub.degree(nid)
    communities = [[node_mapping[idx] for idx in c] for c in partition]
    ci.calculate_community_importance(graph, communities, weight_alpha=weight_alpha)
    logging.info("已将社区ID、节点degree和关系重要性分数标注到图谱。")
    return graph


def save_community_reports(reports: Dict[str, str], output_path: Path, id_map_path: Path = None):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    id_maps: Dict[str, Dict[str, str]] = {}
    if id_map_path and id_map_path.exists():
        try:
            with open(id_map_path, 'r', encoding='utf-8') as f:
                id_maps = json.load(f)
            logging.info(f"已加载 {len(id_maps)} 个社区的ID映射")
        except Exception as e:
            logging.warning(f"加载ID映射失败: {e}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for cid, text in reports.items():
            obj = _safe_json_loads(text) if isinstance(text, str) else None
            record = {"community_id": str(cid)}
            if obj is None:
                record["report_raw"] = text
            else:
                record["report"] = obj
            if str(cid) in id_maps: record["local_id_map"] = id_maps[str(cid)]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info(f"💾 社区报告(JSONL)已保存至: {output_path}")


def save_graph(graph: nx.DiGraph, output_path: Path):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f"正在将图谱保存到 {output_path}...")
    data = nx.node_link_data(graph)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info("✅ 图谱保存成功。")


# =========================
# 可复用的阶段封装 (调用端点调整)
# =========================

def run_disambiguation_stage(client: OpenAI, graph: nx.DiGraph, config: Dict[str, Any], model_name: str,
                             sleep_interval: int, disamb_requests_path: Path, save_path: Path = None) -> Dict[str, str]:
    """
    运行实体消歧阶段，支持分批处理

    Args:
        client: OpenAI客户端
        graph: 图对象
        config: 配置字典
        model_name: 模型名称
        sleep_interval: 轮询间隔
        disamb_requests_path: 请求文件路径
        save_path: 保存路径

    Returns:
        消歧结果字典
    """
    logging.info("--- 阶段1: 实体消歧 ---")

    # 获取batch_size配置
    embedding_batch_size = int(config["graph_builder"].get("embedding_batch_size", 5000))

    # 创建批量请求，使用batch_size控制
    create_batch_requests(graph=graph, model_name=model_name, output_path=disamb_requests_path,
                          request_type="disambiguation", batch_size=embedding_batch_size)

    # 处理分批生成的请求文件
    results = {}
    num_nodes = graph.number_of_nodes()
    if embedding_batch_size and num_nodes > embedding_batch_size:
        num_batches = (num_nodes + embedding_batch_size - 1) // embedding_batch_size
        logging.info(f"🚀 并行提交 {num_batches} 个消歧批次作业...")

        # 并行提交所有批次
        batch_jobs = []
        for batch_idx in range(num_batches):
            batch_disamb_path = disamb_requests_path.parent / f"{disamb_requests_path.stem}_batch_{batch_idx + 1}.jsonl"
            logging.info(f"📤 提交消歧批次 {batch_idx + 1}/{num_batches}")
            job = submit_and_monitor_job(client, batch_disamb_path, model_name, sleep_interval,
                                        f"Disambiguation-Batch{batch_idx + 1}", monitor=False)
            if job:
                batch_jobs.append((batch_idx + 1, job))

        # 并行监控所有批次
        logging.info(f"⏳ 开始监控 {len(batch_jobs)} 个批次作业的完成状态...")
        completed_jobs = []

        with ThreadPoolExecutor(max_workers=min(len(batch_jobs), 10)) as executor:
            # 提交监控任务
            future_to_batch = {
                executor.submit(_monitor_job_completion, client, job, sleep_interval, f"Disambiguation-Batch{batch_idx}"): (batch_idx, job)
                for batch_idx, job in batch_jobs
            }

            # 等待所有任务完成
            for future in as_completed(future_to_batch):
                batch_idx, job = future_to_batch[future]
                try:
                    completed_job = future.result()
                    if completed_job:
                        completed_jobs.append((batch_idx, completed_job))
                        logging.info(f"✅ 消歧批次 {batch_idx} 已完成")
                except Exception as e:
                    logging.error(f"❌ 消歧批次 {batch_idx} 处理失败: {e}")

        # 处理所有批次的结果
        for batch_idx, completed_job in sorted(completed_jobs, key=lambda x: x[0]):
            batch_results = process_results(completed_job, client)
            results.update(batch_results)
            logging.info(f"📊 批次 {batch_idx} 返回 {len(batch_results)} 个结果")

        logging.info(f"🎉 所有消歧批次处理完成，共获得 {len(results)} 个结果")
    else:
        job = submit_and_monitor_job(client, disamb_requests_path, model_name, sleep_interval, "Disambiguation")
        results = process_results(job, client)

    if results:
        matched = 0
        for nid, desc in results.items():
            actual_id = try_find_node_with_normalization(graph, nid)
            if actual_id:
                graph.nodes[actual_id]['description'] = desc
                graph.nodes[actual_id]['is_disambiguated'] = True
                matched += 1
            else:
                logging.warning(f"⚠️ 消歧结果中的节点ID '{nid}' 在图中不存在")
        logging.info(f"已更新 {matched}/{len(results)} 个节点的消歧描述。")
        if save_path is not None:
            save_graph(graph, save_path)
            logging.info(f"消歧后图已保存到: {save_path}")
    else:
        logging.warning("未能获取新的实体描述。")
    return results


def run_entity_merge_stage(client: OpenAI, graph: nx.DiGraph, config: Dict[str, Any], model_name: str, prompt_dir: str,
                           sleep_interval: int, tmp_emb_req_path: Path, merge_req_path: Path,
                           save_path: Path = None) -> nx.DiGraph:
    enable_entity_merge = bool(config["graph_builder"].get("enable_entity_merge", True))
    if not enable_entity_merge:
        logging.info("已禁用实体合并阶段。")
        return graph

    embed_model = config.get("embedding", {}).get("model", "text-embedding-v3")  # 默认改为千问模型
    embed_dim = int(config.get("embedding", {}).get("dimensionality", 1024))
    entity_topk = int(config["graph_builder"].get("entity_merge_topk", 10))
    entity_min_sim = float(config["graph_builder"].get("entity_merge_min_sim", 0.82))
    embedding_batch_size = int(config["graph_builder"].get("embedding_batch_size", 5000))

    logging.info("--- 阶段2: 实体合并（聚类 + LLM 仲裁） ---")
    ent_ids: List[str] = []
    ent_texts: List[Tuple[str, str]] = []
    for nid, nd in graph.nodes(data=True):
        if nd.get('is_disambiguated'):
            name_str = str(nd.get('name', '')).strip() if nd.get('name') is not None else ''
            desc_str = str(nd.get('description', '')).strip() if nd.get('description') is not None else ''
            text = f"{name_str}\n{desc_str}".strip()
            ent_ids.append(nid)
            ent_texts.append((nid, text or (nd.get('name') or nid)))

    if len(ent_ids) < 2:
        logging.info("可用于合并的实体数量不足，跳过该阶段。")
        return graph

    num_entities = len(ent_texts)
    num_batches = (num_entities + embedding_batch_size - 1) // embedding_batch_size

    all_embeddings = []
    if num_batches > 1:
        logging.info(f"⚙️ 实体数量 {num_entities} 超过批次大小，拆分为 {num_batches} 个批次")
        logging.info(f"🚀 并行提交 {num_batches} 个嵌入批次作业...")

        # 准备所有批次的请求文件
        batch_jobs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * embedding_batch_size
            end_idx = min((batch_idx + 1) * embedding_batch_size, num_entities)
            batch_ent_texts = ent_texts[start_idx:end_idx]
            batch_ent_ids = ent_ids[start_idx:end_idx]
            batch_req_path = tmp_emb_req_path.parent / f"{tmp_emb_req_path.stem}_batch_{batch_idx + 1}.jsonl"

            logging.info(f"📝 准备批次 {batch_idx + 1}/{num_batches}：实体 {start_idx + 1}-{end_idx}")
            _create_temp_embedding_requests(batch_ent_texts, batch_req_path, model_name=embed_model, dim=embed_dim)
            batch_jobs.append((batch_idx, batch_req_path, batch_ent_ids))

        # 并行提交所有嵌入作业
        submitted_jobs = []
        for batch_idx, batch_req_path, batch_ent_ids in batch_jobs:
            logging.info(f"📤 提交嵌入批次 {batch_idx + 1}/{num_batches}")
            emb_job = _submit_and_monitor_embedding_job(client, batch_req_path, embed_model, sleep_interval,
                                                       batch_idx, num_batches, monitor=False)
            if emb_job:
                submitted_jobs.append((batch_idx, emb_job, batch_ent_ids))

        # 并行监控所有批次
        logging.info(f"⏳ 开始监控 {len(submitted_jobs)} 个嵌入批次作业的完成状态...")
        completed_jobs = []

        with ThreadPoolExecutor(max_workers=min(len(submitted_jobs), 10)) as executor:
            # 提交监控任务
            future_to_batch = {
                executor.submit(_monitor_embedding_job_completion, client, job, sleep_interval, batch_idx, num_batches): (batch_idx, job, batch_ent_ids)
                for batch_idx, job, batch_ent_ids in submitted_jobs
            }

            # 等待所有任务完成
            for future in as_completed(future_to_batch):
                batch_idx, job, batch_ent_ids = future_to_batch[future]
                try:
                    completed_job = future.result()
                    if completed_job:
                        completed_jobs.append((batch_idx, completed_job, batch_ent_ids))
                        logging.info(f"✅ 嵌入批次 {batch_idx + 1} 已完成")
                except Exception as e:
                    logging.error(f"❌ 嵌入批次 {batch_idx + 1} 处理失败: {e}")

        # 处理所有批次的结果
        for batch_idx, completed_job, batch_ent_ids in sorted(completed_jobs, key=lambda x: x[0]):
            batch_V = _process_embedding_results(completed_job, client, batch_ent_ids)
            if batch_V.shape[0] > 0:
                all_embeddings.append(batch_V)
                logging.info(f"📊 批次 {batch_idx + 1} 获得 {batch_V.shape[0]} 个嵌入向量")
            else:
                logging.warning(f"⚠️ 批次 {batch_idx + 1} 未获得有效嵌入向量")

        if all_embeddings:
            V = np.vstack(all_embeddings)
            logging.info(f"🎉 所有嵌入批次处理完成，共获得 {V.shape[0]} 个嵌入向量")
        else:
            logging.error("❌ 所有批次均未获得有效嵌入向量")
            V = np.zeros((0, embed_dim), dtype=float)
    else:
        # 单批次处理
        batch_req_path = tmp_emb_req_path
        logging.info(f"🔄 处理单批次：{num_entities} 个实体")
        _create_temp_embedding_requests(ent_texts, batch_req_path, model_name=embed_model, dim=embed_dim)
        emb_job = _submit_and_monitor_embedding_job(client, batch_req_path, embed_model, sleep_interval, 0, 1)
        V = _process_embedding_results(emb_job, client, ent_ids)

    if all_embeddings:
        V = np.vstack(all_embeddings)
        logging.info(f"🎉 所有批次处理完成，共获得 {V.shape[0]} 个嵌入向量")
    else:
        logging.error("❌ 所有批次均未获得有效嵌入向量")
        V = np.zeros((0, embed_dim), dtype=float)

    clusters = build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim)
    logging.info(f"候选同义簇数量: {len(clusters)}")
    if not clusters:
        logging.info("未发现候选合并簇，跳过 LLM 仲裁。")
        return graph

    # 使用embedding_batch_size控制单次批量请求数量
    create_entity_merge_requests(graph, clusters, model_name=model_name, prompt_dir=prompt_dir,
                                 output_path=merge_req_path, batch_size=embedding_batch_size)

    # 处理分批生成的请求文件
    merge_texts = {}
    if embedding_batch_size and len(clusters) > embedding_batch_size:
        num_merge_batches = (len(clusters) + embedding_batch_size - 1) // embedding_batch_size
        logging.info(f"🚀 并行提交 {num_merge_batches} 个LLM仲裁批次作业...")

        # 并行提交所有批次
        batch_jobs = []
        for batch_idx in range(num_merge_batches):
            batch_merge_path = merge_req_path.parent / f"{merge_req_path.stem}_batch_{batch_idx + 1}.jsonl"
            logging.info(f"📤 提交LLM仲裁批次 {batch_idx + 1}/{num_merge_batches}")
            merge_job = submit_and_monitor_job(client, batch_merge_path, model_name, sleep_interval,
                                              f"EntityMerge-Batch{batch_idx + 1}", monitor=False)
            if merge_job:
                batch_jobs.append((batch_idx + 1, merge_job))

        # 并行监控所有批次
        logging.info(f"⏳ 开始监控 {len(batch_jobs)} 个LLM仲裁批次作业的完成状态...")
        completed_jobs = []

        with ThreadPoolExecutor(max_workers=min(len(batch_jobs), 10)) as executor:
            # 提交监控任务
            future_to_batch = {
                executor.submit(_monitor_job_completion, client, job, sleep_interval, f"EntityMerge-Batch{batch_idx}"): (batch_idx, job)
                for batch_idx, job in batch_jobs
            }

            # 等待所有任务完成
            for future in as_completed(future_to_batch):
                batch_idx, job = future_to_batch[future]
                try:
                    completed_job = future.result()
                    if completed_job:
                        completed_jobs.append((batch_idx, completed_job))
                        logging.info(f"✅ LLM仲裁批次 {batch_idx} 已完成")
                except Exception as e:
                    logging.error(f"❌ LLM仲裁批次 {batch_idx} 处理失败: {e}")

        # 处理所有批次的结果
        for batch_idx, completed_job in sorted(completed_jobs, key=lambda x: x[0]):
            batch_texts = process_results(completed_job, client)
            merge_texts.update(batch_texts)
            logging.info(f"📊 批次 {batch_idx} 返回 {len(batch_texts)} 个结果")

        logging.info(f"🎉 所有LLM仲裁批次处理完成，共获得 {len(merge_texts)} 个结果")
    else:
        merge_job = submit_and_monitor_job(client, merge_req_path, model_name, sleep_interval, "EntityMerge")
        merge_texts = process_results(merge_job, client)

    groups = parse_entity_merge_results(merge_texts)
    logging.info(f"LLM 确认的分组数量: {len(groups)}")

    # 人工审核功能（可选）
    enable_manual_review = config["graph_builder"].get("enable_manual_review", False)
    if enable_manual_review:
        try:
            from utils.entity_merge_review import run_entity_merge_review
            review_sample_size = config["graph_builder"].get("manual_review_sample_size", 5)
            review_output_dir = PROJECT_ROOT / config["graph_builder"].get("manual_review_output_dir",
                                                                           "data/reports/manual_review")

            logging.info(f"🔍 启动人工审核流程，抽样数量: {review_sample_size}")
            review_report = run_entity_merge_review(
                graph=graph,
                clusters=clusters,
                llm_groups=groups,
                sample_size=review_sample_size,
                output_dir=review_output_dir
            )
            logging.info("✅ 人工审核完成")
        except Exception as e:
            logging.warning(f"⚠️ 人工审核过程出现异常，已跳过: {e}")

    alias2canon, canon_name_map = build_merge_map(graph, groups)

    if alias2canon:
        graph = apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
        if save_path is not None:
            save_graph(graph, save_path)
            logging.info(f"消歧后图已保存到: {save_path}")
    else:
        logging.info("LLM 未给出任何可用合并映射，跳过合并应用。")
    return graph


def run_community_detection_and_importance(graph: nx.DiGraph, weight_alpha: float) -> nx.DiGraph:
    logging.info("--- 阶段3: 社区发现 ---")
    return detect_communities(graph, weight_alpha)


def run_community_summaries(client: OpenAI, graph: nx.DiGraph, model_name: str, prompt_dir: str, config: Dict[str, Any],
                            sleep_interval: int, community_requests_path: Path) -> Dict[str, str]:
    max_report_words = str(config["graph_builder"].get("community_summary_max_report_words", 800))
    max_entities = int(config["graph_builder"].get("community_summary_used_entities_num", 25))
    max_relationships = int(config["graph_builder"].get("community_summary_used_relationships_num", 50))

    num_comms = create_batch_requests(
        graph=graph,
        model_name=model_name,
        output_path=community_requests_path,
        request_type="community_summary",
        prompt_dir=prompt_dir,
        max_report_words=max_report_words,
        max_relationships=max_relationships,
        max_entities=max_entities,
    )

    if num_comms <= 0:
        logging.warning("图中未发现社区，跳过摘要生成步骤。")
        return {}

    job = submit_and_monitor_job(client, community_requests_path, model_name, sleep_interval, "CommunitySummary")
    summaries = process_results(job, client)
    return summaries


def build_pipeline_from_config(config: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[str, str]]:
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    if not api_key:
        raise ValueError("请设置 QWEN_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logging.info("阿里云百炼 (OpenAI兼容) 客户端初始化完成。")

    sleep_interval = int(config["graph_builder"].get("sleep_interval", 5))
    model_name = config["llm"]["model"]
    prompt_dir = config["graph_builder"].get("prompt_dir", "prompts")
    weight_alpha = float(config["graph_builder"].get("community_importance_weight_alpha", 0.6))

    input_path = PROJECT_ROOT / config["graph_builder"]["input_path"]
    disambiguation_graph_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_graph_path"]
    merge_graph_path = PROJECT_ROOT / config["graph_builder"]["merged_graph_path"]
    disamb_requests_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_requests_path"]
    community_requests_path = PROJECT_ROOT / config["graph_builder"]["community_requests_path"]
    tmp_emb_req_path = PROJECT_ROOT / config["graph_builder"]["embedding_requests_path"]
    merge_req_path = PROJECT_ROOT / config["graph_builder"]["merge_requests_path"]

    graph = load_and_build_initial_graph(input_path)
    if not graph.nodes:
        raise RuntimeError("图为空，流程终止。请检查输入文件。")

    run_disambiguation_stage(client, graph, config, model_name, sleep_interval, disamb_requests_path,
                             save_path=disambiguation_graph_path)
    graph = run_entity_merge_stage(client, graph, config, model_name, prompt_dir, sleep_interval, tmp_emb_req_path,
                                   merge_req_path, save_path=merge_graph_path)
    graph = run_community_detection_and_importance(graph, weight_alpha)
    summaries = run_community_summaries(client, graph, model_name, prompt_dir, config, sleep_interval,
                                        community_requests_path)
    return graph, summaries


# =========================
# 主流程
# =========================

def main():
    config = load_config()
    setup_logging(config)

    graph, summaries = build_pipeline_from_config(config)

    reports_path = PROJECT_ROOT / config["graph_builder"]["community_reports_path"]
    final_graph_path = PROJECT_ROOT / config["graph_builder"]["output_graph_path"]
    community_requests_path = PROJECT_ROOT / config["graph_builder"]["community_requests_path"]

    # 构建ID映射文件路径
    id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"

    if summaries:
        save_community_reports(summaries, reports_path, id_map_path)
    save_graph(graph, final_graph_path)

    logging.info("🎉🎉🎉 全流程完成：消歧 → 实体合并 → 社区发现/摘要 → 保存 🎉🎉🎉")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}")
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)