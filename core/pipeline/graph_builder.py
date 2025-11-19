import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from string import Template
import numpy as np
import networkx as nx
import pandas as pd
import yaml
from collections import Counter, defaultdict
from dataclasses import dataclass

from google import genai

# 社区发现依赖（保持不变）
import leidenalg as la
import igraph as ig

from utils.client_factory import create_gemini_client
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
    """
    规范化实体ID，将下划线统一转换为连字符。
    用于处理历史遗留数据（使用_）和新数据（使用-）的兼容性问题。

    例如：chunk-71d90a3f3b4c2a7f57191246fa17a016_e-1 -> chunk-71d90a3f3b4c2a7f57191246fa17a016-e-1
    """
    if not entity_id:
        return entity_id
    return str(entity_id).replace('_', '-')

def try_find_node_with_normalization(G: nx.DiGraph, entity_id: str) -> str | None:
    """
    尝试在图中查找节点，支持ID规范化。
    首先尝试原始ID，如果找不到则尝试规范化后的ID。

    返回：图中实际存在的节点ID，如果都不存在则返回None
    """
    if G.has_node(entity_id):
        return entity_id

    # 尝试规范化后的ID
    normalized_id = normalize_entity_id(entity_id)
    if normalized_id != entity_id and G.has_node(normalized_id):
        return normalized_id

    # 反向尝试：如果输入是规范化的，尝试带下划线的版本
    if '-e-' in entity_id:
        # 尝试将最后一个 -e- 替换为 _e-
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
            data = json.loads(line)
            chunk_id = data.get("id")
            if not chunk_id:
                logging.warning("发现一行缺少'id'字段，跳过。")
                continue

            id_map: Dict[str, str] = {}
            g = data.get("graph", {})

            # 处理 graph 可能是列表的情况（LLM有时返回列表而非字典）
            if isinstance(g, list):
                logging.warning(f"chunk {chunk_id}: graph 是列表而非字典，尝试转换")
                # 如果是列表，尝试找到包含 entities 和 relationships 的元素
                if g and isinstance(g[0], dict):
                    g = g[0]  # 取第一个元素
                else:
                    logging.warning(f"chunk {chunk_id}: 无法从列表中提取有效的图数据，跳过")
                    continue

            # 确保 g 是字典
            if not isinstance(g, dict):
                logging.warning(f"chunk {chunk_id}: graph 既非字典也非列表，类型为 {type(g)}，跳过")
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
                G.add_edge(row['source'], row['target'], **{k: v for k, v in row.items() if k not in ('source','target')})

    logging.info(f"初始图构建完成：{G.number_of_nodes()} 节点, {G.number_of_edges()} 边。")
    return G

# =========================
# 阶段1：实体消歧（保持原设计）
# =========================

def create_disambiguation_prompt(node_id: str, G: nx.DiGraph) -> str:
    ndata = G.nodes[node_id]
    ctx = [
        f"实体名称: {ndata.get('name','N/A')}",
        f"实体类型: {ndata.get('type','N/A')}",
        f"原始描述: {ndata.get('description','N/A')}"
    ]
    rels = []
    for _, v, ed in G.edges(node_id, data=True):
        rels.append(f"- 与 '{G.nodes[v].get('name','N/A')}' 的关系是 '{ed.get('relationship','N/A')}'")
    for u, _, ed in G.in_edges(node_id, data=True):
        rels.append(f"- '{G.nodes[u].get('name','N/A')}' 与它的关系是 '{ed.get('relationship','N/A')}'")
    if rels:
        ctx.append("\n关系上下文:")
        ctx.extend(rels)
    prompt = (
        "请基于以下上下文，为实体生成一个全面、精准、消除歧义的标准化描述，像百科定义一样，不超过100字。\n"
        f"上下文信息:\n{'\n'.join(ctx)}\n"
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
) -> int:
    """创建批量请求（实体消歧 / 社区摘要 / 实体合并仲裁）。"""
    logging.info(f"正在为 '{request_type}' 创建批量请求...")
    requests: List[Dict[str, Any]] = []

    if request_type == "disambiguation":
        for nid in graph.nodes():
            prompt = create_disambiguation_prompt(nid, graph)
            requests.append({"key": nid, "request": {"model": f"models/{model_name}", "contents": {"parts": [{"text": prompt}]}}})

    elif request_type == "community_summary":
        # 根据节点属性 'community' 组织成员
        communities: Dict[str, List[str]] = defaultdict(list)
        for node, data in graph.nodes(data=True):
            cid = data.get("community")
            if cid is not None:
                communities[str(cid)].append(node)

        # 组装 prompt
        community_prompt = Template(load_prompt(prompt_dir, "community_summary.md"))
        _max_entities = 25 if not max_entities or max_entities <= 0 else max_entities
        _max_relationships = 50 if not max_relationships or max_relationships <= 0 else max_relationships

        # 保存所有社区的ID映射
        all_id_maps: Dict[str, Dict[str, str]] = {}

        for comm_id, members in communities.items():
            context, id_map = build_community_context(graph, members, _max_entities, _max_relationships)
            prompt = community_prompt.substitute(max_report_len=max_report_words or "1000", context=context)
            requests.append({"key": comm_id, "request": {"model": f"models/{model_name}", "contents": {"parts": [{"text": prompt}]}}})
            all_id_maps[comm_id] = id_map

        # 将ID映射保存到与请求文件同目录的JSON文件
        id_map_path = output_path.parent / f"{output_path.stem}_id_maps.json"
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(all_id_maps, f, ensure_ascii=False, indent=2)
        logging.info(f"社区ID映射已保存至: {id_map_path}")

    elif request_type == "entity_merge":
        raise RuntimeError("请使用 create_entity_merge_requests 生成实体合并仲裁请求。")

    else:
        raise ValueError(f"未知 request_type: {request_type}")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    logging.info(f"已为 {len(requests)} 个 '{request_type}' 任务写入 {output_path}")
    return len(requests)


def submit_and_monitor_job(client: genai.Client, input_file_path: Path, model_name: str, sleep_interval: int, job_type: str) -> Any:
    logging.info(f"📤 [{job_type}] 正在上传请求文件: {input_file_path.name}...")
    try:
        up = client.files.upload(file=str(input_file_path), config={"display_name": f'{job_type}-batch-{input_file_path.stem}', "mime_type": 'application/jsonl'})
        logging.info(f"✅ [{job_type}] 文件上传成功: {up.name}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 文件上传失败: {e}")
        return None

    logging.info(f"🚀 [{job_type}] 正在使用模型 '{model_name}' 创建批量作业...")
    try:
        job = client.batches.create(model=f"models/{model_name}", src=up.name, config={'display_name': f"{job_type}-job-{input_file_path.stem}"})
        logging.info(f"✅ [{job_type}] 批量作业已创建: {job.name}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 创建批量作业失败: {e}")
        client.files.delete(name=up.name)
        return None

    completed = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    logging.info(f"⏳ [{job_type}] 开始轮询作业 '{job.name}' 状态，每 {sleep_interval} 秒检查一次...")
    while True:
        try:
            status = client.batches.get(name=job.name)
            state = status.state.name
            logging.info(f"  - [{job_type}] 当前状态: {state}")
            if state in completed:
                return status
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - [{job_type}] 轮询失败: {e}")
            time.sleep(max(2, sleep_interval * 2))


def process_results(batch_job_status: Any, client: genai.Client) -> Dict[str, str]:
    """下载并处理批量作业的结果，返回 {key: text}。"""
    if not batch_job_status or batch_job_status.state.name != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 作业失败，无法处理结果。")
        if batch_job_status:
            logging.error(f"  - 失败原因: {batch_job_status.error}")
        return {}

    out: Dict[str, str] = {}
    fn = batch_job_status.dest.file_name
    logging.info(f"📥 正在下载结果文件: {fn}")
    try:
        content = client.files.download(file=fn).decode('utf-8')
        ok, bad = 0, 0
        for line in content.strip().split('\n'):
            try:
                obj = json.loads(line)
                key = obj.get("key")
                if key and obj.get("response"):
                    text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    out[key] = (text or "").strip()
                    ok += 1
                elif obj.get("error"):
                    logging.error(f"  - ❌ 处理 ID '{key}' 时发生错误: {obj['error'].get('message')}")
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

def _create_temp_embedding_requests(entities: List[Tuple[str, str]], output_path: Path, model_name: str, dim: int = 768) -> None:
    """
    entities: List[(id, text)] 其中 text = name + \n + desc
    生成 Gemini Embeddings 批量请求 JSONL。
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for eid, text in entities:
            req = {
                "key": eid,
                "request": {
                    "content": {"parts": [{"text": text}]},
                    "task_type": "RETRIEVAL_DOCUMENT",
                    "output_dimensionality": dim
                }
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def _submit_and_monitor_embedding_job(client: genai.Client, requests_path: Path, model_name: str, sleep_interval: int):
    logging.info(f"📤 [EntityEmb] 上传临时嵌入请求: {requests_path.name} ...")
    try:
        up = client.files.upload(file=str(requests_path), config={"display_name": f'emb-entity-{requests_path.stem}', "mime_type": 'application/jsonl'})
        logging.info(f"✅ [EntityEmb] 文件上传成功: {up.name}")
    except Exception as e:
        logging.error(f"❌ [EntityEmb] 文件上传失败: {e}")
        return None

    logging.info("🚀 [EntityEmb] 创建批量嵌入作业...")
    try:
        job = client.batches.create_embeddings(model=f"{model_name}", src={'file_name': up.name}, config={'display_name': f"emb-entity-job-{requests_path.stem}"})
        logging.info(f"✅ [EntityEmb] 作业已创建: {job.name}")
    except Exception as e:
        logging.error(f"❌ [EntityEmb] 创建嵌入作业失败: {e}")
        client.files.delete(name=up.name)
        return None

    done = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    logging.info(f"⏳ [EntityEmb] 轮询 '{job.name}' 状态，每 {sleep_interval} 秒...")
    while True:
        try:
            st = client.batches.get(name=job.name)
            if st.state.name in done:
                return st
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - [EntityEmb] 轮询失败: {e}")
            time.sleep(max(2, sleep_interval * 2))


def _process_embedding_results(batch_job: Any, client: genai.Client, id_order: List[str]) -> np.ndarray:
    if not batch_job or getattr(batch_job, 'state', None) != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 临时嵌入作业失败或未执行。")
        if batch_job and getattr(batch_job, 'error', None):
            logging.error(f"  - 失败原因: {batch_job.error}")
        return np.zeros((0, 1), dtype=float)

    job_id = batch_job.name.split('/')[-1]
    logging.info(f"📥 [{job_id}] 正在下载结果文件: {batch_job.dest.file_name}")
    file_content = client.files.download(file=batch_job.dest.file_name).decode('utf-8')

    lines = file_content.strip().split('\n')
    if len(lines) != len(id_order):
        logging.warning(f"⚠️ 嵌入结果数量({len(lines)})与输入实体数({len(id_order)})不一致，尝试按行序对齐。")

    vecs: List[np.ndarray] = []
    for i, line in enumerate(lines[:len(id_order)]):
        try:
            obj = json.loads(line)
            emb = np.array(obj["response"]["embedding"]["values"], dtype=float)
            # 归一化
            n = np.linalg.norm(emb)
            vecs.append(emb / (n if n > 0 else 1.0))
        except Exception as e:
            logging.warning(f"  - ⚠️ 解析嵌入结果失败 (行 {i+1}): {e}")
            vecs.append(np.zeros(768, dtype=float))

    return np.vstack(vecs)


# --- B) 候选簇发现（互为近邻 + 相似度阈值） ---

def _cosine_sim_matrix(A: np.ndarray) -> np.ndarray:
    # A 已经单位化；cosine = dot(A, A.T)
    return np.clip(A @ A.T, -1.0, 1.0)


def build_candidate_clusters(vecs: np.ndarray, ids: List[str], topk: int = 10, min_sim: float = 0.82) -> List[List[str]]:
    """基于互为近邻构图，连通分量即候选簇。"""
    n = vecs.shape[0]
    if n == 0:
        return []

    sims = _cosine_sim_matrix(vecs)
    np.fill_diagonal(sims, -1.0)  # 排除自身

    # 每个点挑 topk 相邻并过滤阈值
    nn_graph = nx.Graph()
    for i in range(n):
        row = sims[i]
        idx = np.argpartition(row, -topk)[-topk:]
        for j in idx:
            if row[j] >= min_sim:
                nn_graph.add_edge(i, j, sim=float(row[j]))
    # 仅保留“互为近邻”的边
    to_remove = []
    for u, v in nn_graph.edges():
        if not (sims[u, v] >= min_sim and sims[v, u] >= min_sim):
            to_remove.append((u, v))
    nn_graph.remove_edges_from(to_remove)

    clusters: List[List[str]] = []
    for comp in nx.connected_components(nn_graph):
        c = sorted(list(comp))
        # 保证簇尺寸不至于过大；过大则按阈值再切分（简单策略：提高阈值0.03重切）
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
        for j in range(i+1, m):
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


def create_entity_merge_requests(G: nx.DiGraph, clusters: List[List[str]], model_name: str, prompt_dir: str, output_path: Path) -> int:
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cnt = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, member_ids in enumerate(clusters):
            prompt = _build_merge_prompt_for_cluster(G, member_ids, prompt_dir)
            line = {
                "key": f"cluster_{idx}",
                "request": {"model": f"models/{model_name}", "contents": {"parts": [{"text": prompt}]}}
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            cnt += 1
    logging.info(f"为 {cnt} 个候选簇生成了 LLM 仲裁请求: {output_path}")
    return cnt


def _safe_json_loads(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass

    if "```" in text:
        # 提取代码块内容
        # 匹配 ```json ... ``` 或 ``` ... ```
        patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',      # ``` ... ```
            r'```json\s*(.*?)```',      # ```json...``` (无换行)
            r'```(.*?)```'              # ```...``` (无换行)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    content = match.strip()
                    if content:
                        return json.loads(content)
                except Exception:
                    continue

        # 如果上面的正则都失败了，尝试简单分割
        parts = text.split("```")
        for i, chunk in enumerate(parts):
            # 跳过空块和可能的语言标识符（如 "json"）
            chunk = chunk.strip()
            if not chunk or chunk.lower() in ['json', 'JSON']:
                continue
            try:
                return json.loads(chunk)
            except Exception:
                continue

    # 最后尝试查找文本中的JSON对象
    try:
        # 尝试找到 { ... } 形式的JSON
        json_match = re.search(r'\{.*}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return None


def parse_entity_merge_results(raw_texts: Dict[str, str]) -> List[LLMResolutionGroup]:
    groups: List[LLMResolutionGroup] = []
    for key, text in raw_texts.items():
        obj = _safe_json_loads(text)
        if not obj or "groups" not in obj:
            logging.warning(f"LLM 返回非期望JSON，key={key}，已跳过。片段: {text[:800]}...")
            continue
        for g in obj.get("groups", []):
            try:
                cname = (g.get("canonical_name") or '').strip()
                mids = [str(x) for x in g.get("member_ids", []) if x]
                rationale = (g.get("rationale") or '').strip() or None
                if cname and len(mids) >= 1:
                    groups.append(LLMResolutionGroup(cname, mids, rationale))
            except Exception as e:
                logging.warning(f"解析分组失败: {e}")
    return groups

# --- D) 应用合并到图 ---


def _choose_representative(G: nx.DiGraph, group: LLMResolutionGroup) -> str | None:
    """从成员中选择一个作为 canonical_id：
    - 若存在 name 等于 canonical_name（忽略大小写）的节点，优先选它；
    - 否则选度数最高者（信息更丰富）；
    - 再否则选择列表第一个。
    """
    lc = group.canonical_name.lower()
    candidate = None

    # 过滤出存在于图中的member_ids，支持ID规范化（兼容_和-）
    valid_member_ids = []
    for eid in group.member_ids:
        actual_id = try_find_node_with_normalization(G, eid)
        if actual_id:
            valid_member_ids.append(actual_id)

    if not valid_member_ids:
        logging.warning(f"⚠️ 分组 '{group.canonical_name}' 的所有成员ID都不存在于图中: {group.member_ids}")
        return None

    # 首选：名称匹配
    for eid in valid_member_ids:
        if (G.nodes[eid].get('name') or '').lower() == lc:
            return eid
    # 次选：度数
    best = (-1, None)
    for eid in valid_member_ids:
        deg = G.degree(eid)
        if deg > best[0]:
            best = (deg, eid)
    candidate = best[1] or valid_member_ids[0]
    return candidate


def build_merge_map(G: nx.DiGraph, groups: List[LLMResolutionGroup]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    返回 (alias_id -> canonical_id 映射, canonical_id -> canonical_name)。
    同一实体可能被多个 group 覆盖，后到的覆盖先到的（通常问题不大，因为簇内互斥）。
    """
    alias2canon: Dict[str, str] = {}
    canon_name: Dict[str, str] = {}
    skipped_groups = 0

    for g in groups:
        try:
            cid = _choose_representative(G, g)
            if cid is None:
                logging.warning(f"⚠️ 无法为分组 '{g.canonical_name}' 选择代表节点，跳过此分组")
                skipped_groups += 1
                continue

            # 只为存在于图中的成员ID建立映射，支持ID规范化
            valid_members = []
            for mid in g.member_ids:
                actual_id = try_find_node_with_normalization(G, mid)
                if actual_id:
                    valid_members.append(actual_id)

            if not valid_members:
                logging.warning(f"⚠️ 分组 '{g.canonical_name}' 没有有效的成员节点，跳过")
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


def apply_entity_merge(G: nx.DiGraph, alias2canon: Dict[str, str], canon_name_map: Dict[str, str], edge_agg: str = "max") -> nx.DiGraph:
    if not alias2canon:
        return G

    # 1) 为每个 canonical_id 收集成员
    groups: Dict[str, List[str]] = defaultdict(list)
    for aid, cid in alias2canon.items():
        groups[cid].append(aid)

    # 2) 合并节点属性
    for cid, members in groups.items():
        members = list(dict.fromkeys(members))  # 去重，保序
        if cid not in members:
            members.insert(0, cid)
        aliases: set[str] = set()
        descs: List[str] = []
        provenance: List[Any] = []
        types: Counter = Counter()
        names_in_group: List[str] = []

        for eid in members:
            nd = G.nodes[eid]
            nm = (nd.get('name') or '').strip()
            if nm:
                names_in_group.append(nm)
                aliases.add(nm)
            aliases.update(nd.get('aliases', []) or [])
            d = (nd.get('description') or '').strip()
            if d:
                descs.append(d)
            pv = nd.get('provenance') or []
            if isinstance(pv, list):
                provenance.extend(pv)
            elif isinstance(pv, dict):
                provenance.append(pv)
            tp = (nd.get('type') or '').strip()
            if tp:
                types[tp] += 1

        # 主名：采用 LLM 指定；若缺，退回出现频次最高/最长名称
        main_name = (canon_name_map.get(cid) or '').strip()
        if not main_name:
            if names_in_group:
                main_name = max(Counter(names_in_group).items(), key=lambda x: (x[1], len(x[0])))[0]
            else:
                main_name = cid  # 兜底

        # 描述：选信息量最大的一条（长度最大），防止无界拼接
        merged_desc = max(descs, key=len) if descs else ''
        merged_aliases = sorted(a for a in aliases if a and a != main_name)
        dominant_type = types.most_common(1)[0][0] if types else (G.nodes[cid].get('type') or '')

        # 写回代表节点
        nd0 = G.nodes[cid]
        nd0['name'] = main_name
        nd0['aliases'] = merged_aliases
        nd0['description'] = merged_desc
        nd0['type'] = dominant_type
        nd0['is_disambiguated'] = True
        # provenance 合并（去重）
        # 将成员ID作为来源之一，便于溯源
        prov_ids = [{'merged_from': members}]
        if provenance:
            prov_ids.extend(provenance)
        nd0['provenance'] = prov_ids

    # 3) 重写边：把 alias 端点替换为 canonical，随后聚合多边
    H = nx.DiGraph()
    # 先复制节点
    for n, data in G.nodes(data=True):
        # 若该节点并入了其他 canonical，则跳过（只保留 canonical 节点）
        if alias2canon.get(n, n) != n:
            continue
        H.add_node(n, **data)

    # 边重写
    tmp_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for u, v, ed in G.edges(data=True):
        uu = alias2canon.get(u, u)
        vv = alias2canon.get(v, v)
        if uu == vv:
            # 自环边通常无信息，跳过
            continue
        key = (uu, vv, ed.get('relationship') or ed.get('rel_type') or 'related_to')
        if key not in tmp_edges:
            tmp_edges[key] = dict(ed)
            tmp_edges[key]['source'] = uu
            tmp_edges[key]['target'] = vv
            # weight 初始化
            try:
                tmp_edges[key]['weight'] = float(ed.get('weight', 1.0))
            except Exception:
                tmp_edges[key]['weight'] = 1.0
            # 关系ID重置为稳定的三元组Key（避免冲突）
            tmp_edges[key]['id'] = ed.get('id') or f"{uu}::{key[2]}::{vv}"
        else:
            # 聚合策略
            if edge_agg == 'sum':
                try:
                    tmp_edges[key]['weight'] += float(ed.get('weight', 1.0))
                except Exception:
                    pass
            elif edge_agg == 'max':
                try:
                    tmp_edges[key]['weight'] = max(float(tmp_edges[key]['weight']), float(ed.get('weight', 1.0)))
                except Exception:
                    pass
            # 合并来源/描述
            if ed.get('description'):
                prev = tmp_edges[key].get('description', '')
                cur = ed.get('description', '')
                tmp_edges[key]['description'] = prev if len(prev) >= len(cur) else cur

    for (_, _, _), ed in tmp_edges.items():
        H.add_edge(ed['source'], ed['target'], **ed)

    logging.info(f"实体合并完成：节点 {G.number_of_nodes()}→{H.number_of_nodes()}，边 {G.number_of_edges()}→{H.number_of_edges()}。")
    return H

# =========================
# 社区发现 + 报告（保持不变）
# =========================

def build_community_context(
    graph: nx.Graph,
    member_ids: List[str],
    max_entities: int = 25,
    max_relationships: int = 50,
    include_importance: bool = True,
) -> Tuple[str, Dict[str, str]]:
    member_set = {n for n in member_ids if n in graph.nodes}
    if not member_set:
        return "Entities:\n\nRelationships:\n", {}

    # 创建本地ID映射 (全局ID -> 本地ID)
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

        # 为选中的节点分配本地ID
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
        freq[a] += 1
        freq[b] += 1

    deg_cache = {}
    def community_degree(nid: Any) -> int:
        if nid in deg_cache:
            return deg_cache[nid]
        d = sum(1 for nbr in graph.neighbors(nid) if nbr in member_set)
        deg_cache[nid] = d
        return d

    involved = set()
    for a, b, *_ in top_edges:
        involved.add(a); involved.add(b)

    ordered_nodes = sorted(involved, key=lambda n: (freq[n], community_degree(n), str(n)), reverse=True)
    selected_nodes = ordered_nodes[:max_entities]
    S = set(selected_nodes)

    filtered = [(a, b, rel, imp, rid) for (a, b, rel, imp, rid) in top_edges if a in S and b in S]
    if not filtered:
        relaxed = [(a, b, rel, imp, rid) for (a, b, rel, imp, rid) in top_edges if (a in S or b in S)]
        filtered = relaxed[:max_relationships]

    # 为选中的节点和关系分配本地ID
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

    parts = ["Entities:", "\n".join(entity_lines) if entity_lines else "", "\nRelationships:", "\n".join(rel_lines) if rel_lines else ""]

    # 反转ID映射: 本地ID -> 全局ID
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
    community_map = {node_mapping[node_index]: str(community_id) for community_id, community in enumerate(partition) for node_index in community}
    nx.set_node_attributes(graph, community_map, "community")

    for community in partition:
        nodes = [node_mapping[idx] for idx in community]
        sub = UG.subgraph(nodes)
        for nid in nodes:
            graph.nodes[nid]["degree"] = sub.degree(nid)

    communities = [[node_mapping[idx] for idx in c] for c in partition]
    ci.calculate_community_importance(graph, communities, weight_alpha=weight_alpha)
    logging.info("已将社区ID、节点degree和关系重要性分数标注到图谱。")
    return graph


def save_community_reports(reports: Dict[str, str], output_path: Path, id_map_path: Path = None):
    """将社区报告保存为 JSONL，每行一个社区对象。
    - 若返回文本不是合法 JSON，则以 `report_raw` 字段原样存储。
    - 如果提供了 id_map_path，会添加 local_id_map 字段。
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # 加载ID映射（如果提供）
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

            # 添加local_id_map（如果有）
            if str(cid) in id_maps:
                record["local_id_map"] = id_maps[str(cid)]

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
# 可复用的阶段封装
# =========================

def run_disambiguation_stage(client: genai.Client, graph: nx.DiGraph, model_name: str, sleep_interval: int, disamb_requests_path: Path, save_path: Path = None) -> Dict[str, str]:
    logging.info("--- 阶段1: 实体消歧 ---")
    create_batch_requests(graph=graph, model_name=model_name, output_path=disamb_requests_path, request_type="disambiguation")
    job = submit_and_monitor_job(client, disamb_requests_path, model_name, sleep_interval, "Disambiguation")
    results = process_results(job, client)
    if results:
        matched = 0
        for nid, desc in results.items():
            # 支持ID规范化：尝试查找节点
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


def run_entity_merge_stage(client: genai.Client, graph: nx.DiGraph, config: Dict[str, Any], model_name: str, prompt_dir: str, sleep_interval: int, tmp_emb_req_path: Path, merge_req_path: Path, save_path: Path = None) -> nx.DiGraph:
    enable_entity_merge = bool(config["graph_builder"].get("enable_entity_merge", True))
    if not enable_entity_merge:
        logging.info("已禁用实体合并阶段。")
        return graph

    embed_model = config.get("embedding", {}).get("model", "gemini-embedding-001")
    embed_dim = int(config.get("embedding", {}).get("dimensionality", 768))
    entity_topk = int(config["graph_builder"].get("entity_merge_topk", 10))
    entity_min_sim = float(config["graph_builder"].get("entity_merge_min_sim", 0.82))

    logging.info("--- 阶段2: 实体合并（聚类 + LLM 仲裁） ---")
    ent_ids: List[str] = []
    ent_texts: List[Tuple[str, str]] = []
    for nid, nd in graph.nodes(data=True):
        if nd.get('is_disambiguated'):
            text = f"{nd.get('name', '').strip()}\n{nd.get('description', '').strip()}".strip()
            ent_ids.append(nid)
            ent_texts.append((nid, text or (nd.get('name') or nid)))

    if len(ent_ids) < 2:
        logging.info("可用于合并的实体数量不足，跳过该阶段。")
        return graph

    _create_temp_embedding_requests(ent_texts, tmp_emb_req_path, model_name=embed_model, dim=embed_dim)
    emb_job = _submit_and_monitor_embedding_job(client, tmp_emb_req_path, embed_model, sleep_interval)
    V = _process_embedding_results(emb_job, client, ent_ids)

    clusters = build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim)
    logging.info(f"候选同义簇数量: {len(clusters)}")
    if not clusters:
        logging.info("未发现候选合并簇，跳过 LLM 仲裁。")
        return graph

    create_entity_merge_requests(graph, clusters, model_name=model_name, prompt_dir=prompt_dir, output_path=merge_req_path)
    merge_job = submit_and_monitor_job(client, merge_req_path, model_name, sleep_interval, "EntityMerge")
    merge_texts = process_results(merge_job, client)
    groups = parse_entity_merge_results(merge_texts)
    logging.info(f"LLM 确认的分组数量: {len(groups)}")

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


def run_community_summaries(client: genai.Client, graph: nx.DiGraph, model_name: str, prompt_dir: str, config: Dict[str, Any], sleep_interval: int, community_requests_path: Path) -> Dict[str, str]:
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
    api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    proxy = config.get("proxy")
    client = create_gemini_client(api_key, proxy)
    logging.info("Gemini 客户端初始化完成。")

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

    run_disambiguation_stage(client, graph, model_name, sleep_interval, disamb_requests_path, save_path=disambiguation_graph_path)
    graph = run_entity_merge_stage(client, graph, config, model_name, prompt_dir, sleep_interval, tmp_emb_req_path, merge_req_path, save_path=merge_graph_path)
    graph = run_community_detection_and_importance(graph, weight_alpha)
    summaries = run_community_summaries(client, graph, model_name, prompt_dir, config, sleep_interval, community_requests_path)
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