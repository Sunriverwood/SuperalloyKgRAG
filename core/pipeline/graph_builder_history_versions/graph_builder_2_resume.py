"""
临时恢复脚本 - 从云端恢复已完成的批处理作业并继续执行

使用场景：
1. 实体消歧批处理作业已完成（在云端有结果文件）
2. 实体向量化批处理作业已完成（在云端有结果文件）
3. 本地没有保存这两份结果，需要从云端下载并继续后续流程

执行步骤：
- 从云端查找并下载实体消歧作业结果
- 从云端查找并下载实体向量化作业结果
- 继续执行实体合并、社区发现、社区摘要等后续流程
"""

import json
import logging
import os
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

# 社区发现依赖
import leidenalg as la
import igraph as ig

from utils.client_factory import create_gemini_client
import utils.community_importance as ci

# --- 动态计算项目根目录 ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# =========================
# 通用函数（从 graph_builder_2.py 复制）
# =========================

def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/graph_builder_resume.log")
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
    logging.info("日志记录器设置完成（恢复模式）")


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
# 从云端恢复作业结果
# =========================

def find_batch_job_by_display_name(client: genai.Client, display_name_pattern: str) -> Any:
    """
    根据显示名称模式查找批处理作业

    Args:
        client: Gemini 客户端
        display_name_pattern: 显示名称模式（支持部分匹配）

    Returns:
        找到的作业对象，如果未找到则返回 None
    """
    logging.info(f"🔍 正在查找包含 '{display_name_pattern}' 的批处理作业...")
    try:
        jobs = list(client.batches.list())
        logging.info(f"找到 {len(jobs)} 个批处理作业")

        for job in jobs:
            if display_name_pattern in job.display_name:
                logging.info(f"✅ 找到匹配作业: {job.display_name} (状态: {job.state.name})")
                return job

        logging.warning(f"⚠️ 未找到包含 '{display_name_pattern}' 的作业")
        return None
    except Exception as e:
        logging.error(f"❌ 查找作业失败: {e}")
        return None


def download_and_process_disambiguation_results(client: genai.Client, job: Any) -> Dict[str, str]:
    """
    从云端下载并处理实体消歧结果

    Returns:
        Dict[node_id, description] - 实体ID到消歧描述的映射
    """
    if not job or job.state.name != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 消歧作业未成功完成")
        return {}

    out: Dict[str, str] = {}
    fn = job.dest.file_name
    logging.info(f"📥 正在下载消歧结果文件: {fn}")

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
        logging.info(f"🎉 消歧结果处理完成：成功 {ok} 条，失败 {bad} 条。")
    except Exception as e:
        logging.error(f"❌ 下载或处理消歧结果文件失败: {e}")
        return {}

    return out


def download_and_process_embedding_results(client: genai.Client, job: Any, id_order: List[str]) -> np.ndarray:
    """
    从云端下载并处理实体向量化结果

    Args:
        job: 嵌入作业对象
        id_order: 实体ID的顺序列表

    Returns:
        归一化后的嵌入向量矩阵 (n_entities, embedding_dim)
    """
    if not job or job.state.name != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 嵌入作业未成功完成")
        return np.zeros((0, 1), dtype=float)

    job_id = job.name.split('/')[-1]
    logging.info(f"📥 [{job_id}] 正在下载嵌入结果文件: {job.dest.file_name}")

    try:
        file_content = client.files.download(file=job.dest.file_name).decode('utf-8')
        lines = file_content.strip().split('\n')

        if len(lines) != len(id_order):
            logging.warning(f"⚠️ 嵌入结果数量({len(lines)})与输入实体数({len(id_order)})不一致")

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

        logging.info(f"🎉 嵌入结果处理完成：{len(vecs)} 个向量")
        return np.vstack(vecs)
    except Exception as e:
        logging.error(f"❌ 下载或处理嵌入结果文件失败: {e}")
        return np.zeros((0, 1), dtype=float)


# =========================
# 加载图数据
# =========================

def load_graph_from_jsonl(input_path: Path) -> nx.DiGraph:
    """从 JSONL 文件加载图数据"""
    logging.info(f"正在从 {input_path} 加载图数据...")

    G = nx.DiGraph()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)

                # 检查是否有 graph 字段（extracted_graph.jsonl 格式）
                if 'graph' in obj:
                    graph_data = obj['graph']
                    chunk_id = obj.get('id', f'chunk-{line_num}')  # 使用chunk ID作为前缀

                    # ID映射：将局部ID映射到全局唯一ID
                    id_mapping = {}

                    # 添加实体（作为节点）
                    if 'entities' in graph_data:
                        for entity in graph_data['entities']:
                            local_id = entity['id']
                            # 生成全局唯一ID：chunk_id-local_id (使用短横线匹配原始格式)
                            global_id = f"{chunk_id}-{local_id}"
                            id_mapping[local_id] = global_id

                            # 构建节点属性
                            node_attrs = {
                                'name': entity.get('name', ''),
                                'type': entity.get('type', ''),
                                'description': entity.get('description', ''),
                                'attributes': entity.get('attributes', {}),
                                'chunk_id': chunk_id,  # 记录来源chunk
                                'original_id': local_id   # 记录原始ID（字段名与原始代码一致）
                            }
                            # 添加节点
                            G.add_node(global_id, **node_attrs)

                    # 添加关系（作为边）
                    if 'relationships' in graph_data:
                        for rel in graph_data['relationships']:
                            local_source = rel.get('source')
                            local_target = rel.get('target')

                            # 使用ID映射转换为全局ID
                            if local_source and local_target:
                                global_source = id_mapping.get(local_source, f"{chunk_id}_{local_source}")
                                global_target = id_mapping.get(local_target, f"{chunk_id}_{local_target}")

                                # 确保源和目标节点存在
                                if not G.has_node(global_source):
                                    G.add_node(global_source, name=local_source, type='UNKNOWN',
                                             description='', chunk_id=chunk_id, local_id=local_source)
                                if not G.has_node(global_target):
                                    G.add_node(global_target, name=local_target, type='UNKNOWN',
                                             description='', chunk_id=chunk_id, local_id=local_target)

                                # 添加边
                                edge_attrs = {
                                    'id': rel.get('id', ''),
                                    'relationship': rel.get('relationship', ''),
                                    'description': rel.get('description', ''),
                                    'weight': rel.get('weight', 1.0),
                                    'source_sentence': rel.get('source_sentence', ''),
                                    'chunk_id': chunk_id
                                }
                                G.add_edge(global_source, global_target, **edge_attrs)

                # 兼容其他格式（如果有 nodes 和 edges 字段）
                elif 'nodes' in obj:
                    for node in obj['nodes']:
                        G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})

                    if 'edges' in obj:
                        for edge in obj['edges']:
                            if G.has_node(edge['source']) and G.has_node(edge['target']):
                                G.add_edge(edge['source'], edge['target'],
                                         **{k: v for k, v in edge.items() if k not in ('source', 'target')})

            except json.JSONDecodeError as e:
                logging.warning(f"⚠️  解析第 {line_num} 行失败: {e}")
                continue
            except Exception as e:
                logging.warning(f"⚠️  处理第 {line_num} 行时出错: {e}")
                continue

    logging.info(f"图加载完成：{len(G.nodes)} 个节点，{len(G.edges)} 条边")
    return G


# =========================
# 从 graph_builder_2.py 导入的关键函数
# =========================

def _cosine_sim_matrix(A: np.ndarray) -> np.ndarray:
    return np.clip(A @ A.T, -1.0, 1.0)


def build_candidate_clusters(vecs: np.ndarray, ids: List[str], topk: int = 10, min_sim: float = 0.82) -> List[List[str]]:
    """基于互为近邻构图，连通分量即候选簇。"""
    n = vecs.shape[0]
    if n == 0:
        return []

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
        if len(comp) >= 2:
            clusters.append([ids[i] for i in comp])

    return clusters


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

    logging.info(f"已生成 {cnt} 个实体合并请求 -> {output_path}")
    return cnt


def submit_and_monitor_job(client: genai.Client, input_file_path: Path, model_name: str, sleep_interval: int, job_type: str) -> Any:
    logging.info(f"📤 [{job_type}] 正在上传请求文件: {input_file_path.name}...")
    try:
        up = client.files.upload(file=str(input_file_path),
                                config={"display_name": f'{job_type}-batch-{input_file_path.stem}',
                                       "mime_type": 'application/jsonl'})
        logging.info(f"✅ [{job_type}] 文件上传成功: {up.name}")
    except Exception as e:
        logging.error(f"❌ [{job_type}] 文件上传失败: {e}")
        return None

    logging.info(f"🚀 [{job_type}] 正在使用模型 '{model_name}' 创建批量作业...")
    try:
        job = client.batches.create(model=f"models/{model_name}", src=up.name,
                                   config={'display_name': f"{job_type}-job-{input_file_path.stem}"})
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
    """下载并处理批量作业的结果"""
    if not batch_job_status or batch_job_status.state.name != 'JOB_STATE_SUCCEEDED':
        logging.error("❌ 作业失败，无法处理结果。")
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


@dataclass
class LLMResolutionGroup:
    canonical_name: str
    member_ids: List[str]
    rationale: str | None = None


def parse_entity_merge_results(merge_texts: Dict[str, str]) -> List[LLMResolutionGroup]:
    """解析 LLM 返回的合并分组结果"""
    groups: List[LLMResolutionGroup] = []
    for k, txt in merge_texts.items():
        # 移除可能的代码块标记
        txt = txt.strip()
        if txt.startswith("```"):
            lines = txt.split('\n')
            txt = '\n'.join(lines[1:-1]) if len(lines) > 2 else txt

        try:
            data = json.loads(txt)
            for g in data.get("groups", []):
                grp = LLMResolutionGroup(
                    canonical_name=g["canonical_name"],
                    member_ids=g["member_ids"],
                    rationale=g.get("rationale")
                )
                groups.append(grp)
        except Exception as e:
            logging.warning(f"解析合并结果 '{k}' 失败: {e}")
            continue

    return groups


def build_merge_map(G: nx.DiGraph, groups: List[LLMResolutionGroup]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """构建实体合并映射"""
    alias2canon: Dict[str, str] = {}
    canon_name_map: Dict[str, str] = {}

    for g in groups:
        if len(g.member_ids) < 2:
            continue

        # 选择规范ID（度数最大的节点）
        best = (0, g.member_ids[0])
        for mid in g.member_ids:
            if G.has_node(mid):
                deg = G.degree(mid)
                if deg > best[0]:
                    best = (deg, mid)

        canon_id = best[1]
        canon_name_map[canon_id] = g.canonical_name

        for mid in g.member_ids:
            if mid != canon_id:
                alias2canon[mid] = canon_id

    logging.info(f"构建了 {len(alias2canon)} 个别名映射，{len(canon_name_map)} 个规范名称")
    return alias2canon, canon_name_map


def apply_entity_merge(G: nx.DiGraph, alias2canon: Dict[str, str], canon_name_map: Dict[str, str],
                       edge_agg: str = 'max') -> nx.DiGraph:
    """应用实体合并到图"""
    logging.info(f"正在应用实体合并：{len(alias2canon)} 个别名 -> 规范实体")

    G2 = nx.DiGraph()

    # 复制节点
    for nid, nd in G.nodes(data=True):
        if nid in alias2canon:
            continue
        new_data = nd.copy()
        if nid in canon_name_map:
            new_data['name'] = canon_name_map[nid]
            new_data['aliases'] = list(set(new_data.get('aliases', []) +
                                          [nd.get('name', '')]))
        G2.add_node(nid, **new_data)

    # 合并边
    edge_dict: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for u, v, ed in G.edges(data=True):
        u2 = alias2canon.get(u, u)
        v2 = alias2canon.get(v, v)
        if u2 == v2:
            continue
        edge_dict[(u2, v2)].append(ed)

    for (u2, v2), eds in edge_dict.items():
        if edge_agg == 'max':
            best = max(eds, key=lambda x: x.get('weight', 0.0))
        else:
            best = eds[0]
        G2.add_edge(u2, v2, **best)

    logging.info(f"合并后图：{len(G2.nodes)} 个节点（减少 {len(G.nodes) - len(G2.nodes)}），"
                f"{len(G2.edges)} 条边")
    return G2


# 导入社区发现相关函数
def detect_communities(G: nx.DiGraph, weight_alpha: float = 0.6) -> nx.DiGraph:
    """社区发现并计算重要性"""
    from core.pipeline.graph_builder_2 import detect_communities as original_detect_communities
    return original_detect_communities(G, weight_alpha)


def run_community_summaries(client: genai.Client, graph: nx.DiGraph, model_name: str,
                           prompt_dir: str, config: Dict[str, Any],
                           sleep_interval: int, community_requests_path: Path) -> Dict[str, str]:
    """生成社区摘要"""
    from core.pipeline.graph_builder_2 import run_community_summaries as original_run_community_summaries
    return original_run_community_summaries(client, graph, model_name, prompt_dir,
                                           config, sleep_interval, community_requests_path)


def save_community_reports(summaries: Dict[str, str], output_path: Path):
    """保存社区摘要报告"""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    records = []
    for cid, summary_text in summaries.items():
        record = {"community_id": cid, "summary": ""}
        try:
            summary_text = summary_text.strip()
            if summary_text.startswith("```"):
                lines = summary_text.split('\n')
                summary_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else summary_text

            obj = json.loads(summary_text)
            record["title"] = obj.get("title", "")
            record["summary"] = obj.get("summary", "")
            record["rating"] = obj.get("rating", 0.0)
            record["rating_explanation"] = obj.get("rating_explanation", "")
            record["findings"] = json.dumps(obj.get("findings", []), ensure_ascii=False)
            record["report"] = obj
        except Exception as e:
            logging.warning(f"解析社区 {cid} 摘要失败: {e}")
            record["summary"] = summary_text

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"✅ 社区报告已保存到: {output_path}")


def save_final_graph(graph: nx.DiGraph, output_path: Path):
    """保存最终图"""
    output_path.parent.mkdir(exist_ok=True, parents=True)
    graph_data = nx.node_link_data(graph)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ 最终图已保存到: {output_path}")


# =========================
# 主恢复流程
# =========================

def main():
    """
    主恢复流程

    步骤：
    1. 从云端恢复实体消歧结果
    2. 应用消歧描述到图
    3. 从云端恢复实体向量化结果
    4. 执行实体合并
    5. 执行社区发现
    6. 生成社区摘要
    7. 保存最终结果
    """
    config = load_config()
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("开始执行恢复流程：从云端下载已完成的批处理作业结果并继续执行")
    logging.info("=" * 80)

    # 初始化客户端
    api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    proxy = config.get("proxy")
    client = create_gemini_client(api_key, proxy)
    logging.info("✅ Gemini 客户端初始化完成")

    # 加载配置参数
    sleep_interval = int(config["graph_builder"].get("sleep_interval", 5))
    model_name = config["llm"]["model"]
    prompt_dir = config["graph_builder"].get("prompt_dir", "prompts")
    weight_alpha = float(config["graph_builder"].get("community_importance_weight_alpha", 0.6))

    input_path = PROJECT_ROOT / config["graph_builder"]["input_path"]
    merge_req_path = PROJECT_ROOT / config["graph_builder"]["merge_requests_path"]
    community_requests_path = PROJECT_ROOT / config["graph_builder"]["community_requests_path"]
    reports_path = PROJECT_ROOT / config["graph_builder"]["community_reports_path"]
    final_graph_path = PROJECT_ROOT / config["graph_builder"]["output_graph_path"]

    embed_model = config.get("embedding", {}).get("model", "gemini-embedding-001")
    entity_topk = int(config["graph_builder"].get("entity_merge_topk", 10))
    entity_min_sim = float(config["graph_builder"].get("entity_merge_min_sim", 0.82))

    # ========== 步骤1：加载初始图 ==========
    logging.info("\n" + "=" * 80)
    logging.info("步骤1：加载初始图数据")
    logging.info("=" * 80)
    graph = load_graph_from_jsonl(input_path)

    # ========== 步骤2：恢复实体消歧结果 ==========
    logging.info("\n" + "=" * 80)
    logging.info("步骤2：从云端恢复实体消歧结果")
    logging.info("=" * 80)

    # 查找消歧作业（根据实际的显示名称模式调整）
    disamb_job = find_batch_job_by_display_name(client, "Disambiguation-job")

    if disamb_job:
        disamb_results = download_and_process_disambiguation_results(client, disamb_job)

        # 应用消歧结果到图
        if disamb_results:
            for nid, desc in disamb_results.items():
                if graph.has_node(nid):
                    graph.nodes[nid]['description'] = desc
                    graph.nodes[nid]['is_disambiguated'] = True
            logging.info(f"✅ 已更新 {len(disamb_results)} 个节点的消歧描述")
        else:
            logging.warning("⚠️ 未获取到消歧结果")
    else:
        logging.error("❌ 未找到消歧作业，请检查作业名称或手动指定")
        # 如果没有消歧作业，可以选择继续或退出
        # return

    # ========== 步骤3：准备实体数据并恢复向量化结果 ==========
    logging.info("\n" + "=" * 80)
    logging.info("步骤3：准备实体数据并从云端恢复向量化结果")
    logging.info("=" * 80)

    # 收集已消歧的实体
    ent_ids: List[str] = []
    ent_texts: List[Tuple[str, str]] = []
    for nid, nd in graph.nodes(data=True):
        if nd.get('is_disambiguated'):
            text = f"{nd.get('name', '').strip()}\n{nd.get('description', '').strip()}".strip()
            ent_ids.append(nid)
            ent_texts.append((nid, text or (nd.get('name') or nid)))

    logging.info(f"找到 {len(ent_ids)} 个已消歧的实体")

    if len(ent_ids) < 2:
        logging.warning("⚠️ 可用于合并的实体数量不足，跳过实体合并阶段")
        V = np.zeros((0, 1), dtype=float)
    else:
        # 查找嵌入作业（根据实际的显示名称模式调整）
        emb_job = find_batch_job_by_display_name(client, "emb-entity-job")

        if emb_job:
            V = download_and_process_embedding_results(client, emb_job, ent_ids)
        else:
            logging.error("❌ 未找到嵌入作业")
            V = np.zeros((0, 1), dtype=float)

    # ========== 步骤4：执行实体合并 ==========
    if V.shape[0] >= 2:
        logging.info("\n" + "=" * 80)
        logging.info("步骤4：执行实体合并")
        logging.info("=" * 80)

        # 构建候选簇
        clusters = build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim)
        logging.info(f"候选同义簇数量: {len(clusters)}")

        if clusters:
            # 创建合并请求
            create_entity_merge_requests(graph, clusters, model_name=model_name,
                                        prompt_dir=prompt_dir, output_path=merge_req_path)

            # 提交并监控作业
            merge_job = submit_and_monitor_job(client, merge_req_path, model_name,
                                              sleep_interval, "EntityMerge")

            # 处理结果
            merge_texts = process_results(merge_job, client)
            groups = parse_entity_merge_results(merge_texts)
            logging.info(f"LLM 确认的分组数量: {len(groups)}")

            # 应用合并
            alias2canon, canon_name_map = build_merge_map(graph, groups)
            if alias2canon:
                graph = apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
            else:
                logging.info("⚠️ LLM 未给出任何可用合并映射，跳过合并应用")
        else:
            logging.info("⚠️ 未发现候选合并簇，跳过 LLM 仲裁")
    else:
        logging.info("⚠️ 跳过实体合并阶段")

    # ========== 步骤5：社区发现 ==========
    logging.info("\n" + "=" * 80)
    logging.info("步骤5：执行社区发现")
    logging.info("=" * 80)
    graph = detect_communities(graph, weight_alpha)

    # ========== 步骤6：生成社区摘要 ==========
    logging.info("\n" + "=" * 80)
    logging.info("步骤6：生成社区摘要")
    logging.info("=" * 80)
    summaries = run_community_summaries(client, graph, model_name, prompt_dir,
                                       config, sleep_interval, community_requests_path)

    # ========== 步骤7：保存结果 ==========
    logging.info("\n" + "=" * 80)
    logging.info("步骤7：保存最终结果")
    logging.info("=" * 80)

    if summaries:
        save_community_reports(summaries, reports_path)
    save_final_graph(graph, final_graph_path)

    logging.info("\n" + "=" * 80)
    logging.info("🎉🎉🎉 恢复流程完成！")
    logging.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logging.critical(f"关键文件未找到: {e}", exc_info=True)
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)

