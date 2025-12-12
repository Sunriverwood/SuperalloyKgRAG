"""
快速恢复脚本 - 从云端恢复已完成的批处理作业并继续执行（直接使用graph_builder）
使用时移动到core/pipeline文件夹下

使用方法：
1. 查看日志找到各个作业ID：
   - 消歧作业ID：搜索 "disambiguation" 或 "消歧"
   - 嵌入作业ID：搜索 "emb-entity-job" 或 "embedding"
   - 合并作业ID：搜索 "EntityMerge" 或 "entity_merge"
   - 社区摘要作业ID：搜索 "community" 或 "社区"
2. 在下面的配置区域设置 START_MODE 和对应的作业ID
3. 运行脚本: python core/pipeline/graph_builder_debug_qwen.py

支持的恢复模式（智能检测已有图文件）：
- "DISAMBIGUATION": 从消歧结果开始
  * 如果已有消歧图：直接加载 → 嵌入 → 合并 → 社区发现 → 社区摘要
  * 否则：【下载并应用消歧】→ 嵌入 → 合并 → 社区发现 → 社区摘要

- "EMBEDDING": 从嵌入结果开始
  * 如果已有合并图：直接加载 → 社区发现 → 社区摘要
  * 如果已有消歧图：加载消歧图 →【下载并应用嵌入】→ 合并 → 社区发现 → 社区摘要
  * 否则：【下载并应用消歧】→【下载并应用嵌入】→ 合并 → 社区发现 → 社区摘要

- "MERGE": 从合并结果开始
  * 如果已有合并图：直接加载 → 社区发现 → 社区摘要
  * 如果已有消歧图：加载消歧图 →【验证嵌入】→【下载并应用合并】→ 社区发现 → 社区摘要
  * 否则：【下载并应用消歧】→【验证嵌入】→【下载并应用合并】→ 社区发现 → 社区摘要

- "COMMUNITY": 从社区发现开始
  * 如果已有合并图：直接加载 → 社区发现 →【下载并应用社区摘要或重新生成】
  * 如果已有消歧图：加载消歧图 →【应用前置步骤】→ 社区发现 →【下载并应用社区摘要或重新生成】
  * 否则：【应用所有前置步骤】→ 社区发现 →【下载并应用社区摘要或重新生成】

工作原理：
- 每个模式都会首先检查对应阶段的图文件是否存在（消歧图、合并图）
- 如果存在，直接加载该图并跳过前置步骤
- 如果不存在，才从云端作业下载并应用结果，然后保存中间图文件供下次使用
- 这样可以最大程度复用已有结果，避免重复下载和处理

中间图文件：
- 消歧图: data/graphs/disambiguation_graph.json
- 合并图: data/graphs/merged_graph.json
- 最终图: data/graphs/final_graph.json

此版本直接使用 graph_builder 的函数，用于验证 graph_builder 代码的正确性。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import os
import numpy as np
import networkx as nx
from typing import Dict, List, Any
import yaml
from openai import OpenAI

# 统一导入 graph_builder_qwen，所有功能函数都调用它
from core.pipeline_qwen import graph_builder_qwen as gb

# ========== 配置区域 - 请根据实际情况修改 ==========
# 恢复模式选择（必填）：
START_MODE = "EMBEDDING"

# 作业ID配置（根据选择的模式填写对应的ID）
# 支持单个ID（字符串）或多个批次ID（列表）
DISAMBIGUATION_JOB_ID = "batch_req_xxxx" # 示例 ID
# 如果任务被拆分成多个批次，使用列表格式：
# DISAMBIGUATION_JOB_ID = ["batch_xxx_1", "batch_xxx_2", "batch_xxx_3"]
# EMBEDDING_JOB_ID = "batch_req_xxxx"
EMBEDDING_JOB_ID = ["batch_76179c74-35d1-403e-8b62-f500696b0b99","batch_a378cdeb-e0a2-4a58-84e4-f4bdb67bb692","batch_f28d2645-cb36-4e6e-a01b-895737ecd149",
                    "batch_13466778-4838-4510-a4fa-7381ce0584aa","batch_02258542-e3ae-4dfb-8496-898daac5638f","batch_b1104eba-27cc-4a0a-beea-e443c17a2ff6",
                    "batch_60ff8b20-722c-40c2-a329-2f806b73fec5","batch_bb6f09d1-f750-419a-8a8c-4217294033a5","batch_6591e08c-6e2d-4da4-930d-ee00ef3d80df"]
# EMBEDDING_JOB_ID = ["batch_xxx_1", "batch_xxx_2"]

ENTITY_MERGE_JOB_ID = "batch_b7"
# ENTITY_MERGE_JOB_ID = ["batch_xxx_1", "batch_xxx_2"]

COMMUNITY_SUMMARY_JOB_ID = "batch_98"
# COMMUNITY_SUMMARY_JOB_ID = ["batch_xxx_1", "batch_xxx_2"]

# ==================================================

def setup_logging():
    log_file = PROJECT_ROOT / "logs" / "superalloyKgRAG.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_config():
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_job_id(job_id):
    """
    将作业ID标准化为列表格式

    Args:
        job_id: 可以是字符串（单个ID）或列表（多个批次ID）

    Returns:
        列表格式的作业ID
    """
    if job_id is None:
        return []
    if isinstance(job_id, str):
        return [job_id] if job_id and not job_id.startswith("batch_req_xxxx") else []
    if isinstance(job_id, list):
        return [jid for jid in job_id if jid and not jid.startswith("batch_req_xxxx")]
    return []

def is_valid_job_id(job_id):
    """检查作业ID是否有效（非空且非占位符）"""
    if isinstance(job_id, str):
        return job_id and not job_id.startswith("batch_req_xxxx")
    if isinstance(job_id, list):
        return any(jid and not jid.startswith("batch_req_xxxx") for jid in job_id)
    return False

def download_and_process_disambiguation_results(client: OpenAI, job: Any) -> Dict[str, str]:
    """从已完成的消歧作业下载并处理结果 - 调用 graph_builder_qwen.process_results"""
    return gb.process_results(job, client)

def download_and_process_multiple_disambiguation_batches(client: OpenAI, job_ids: List[str]) -> Dict[str, str]:
    """
    从多个消歧批次作业下载并合并结果

    Args:
        client: OpenAI客户端
        job_ids: 批次作业ID列表

    Returns:
        合并后的消歧结果字典
    """
    all_results = {}
    successful_batches = 0
    failed_batches = 0

    for idx, job_id in enumerate(job_ids, 1):
        logging.info(f"\n📦 处理消歧批次 {idx}/{len(job_ids)}: {job_id}")
        try:
            job = client.batches.retrieve(batch_id=job_id)
            logging.info(f"   作业状态: {job.status}")

            if job.status == 'completed':
                batch_results = download_and_process_disambiguation_results(client, job)
                if batch_results:
                    all_results.update(batch_results)
                    successful_batches += 1
                    logging.info(f"   ✅ 批次 {idx} 成功获取 {len(batch_results)} 个结果")
                else:
                    failed_batches += 1
                    logging.warning(f"   ⚠️ 批次 {idx} 未获取到结果")
            else:
                failed_batches += 1
                logging.error(f"   ❌ 批次 {idx} 状态不是成功: {job.status}")
        except Exception as e:
            failed_batches += 1
            logging.error(f"   ❌ 批次 {idx} 处理失败: {e}")

    logging.info(f"\n🎉 消歧批次处理完成: 成功 {successful_batches}/{len(job_ids)}, 失败 {failed_batches}/{len(job_ids)}")
    logging.info(f"   总计获得 {len(all_results)} 个消歧结果")
    return all_results

def download_and_process_embedding_results(client: OpenAI, job: Any, id_order: List[str]) -> np.ndarray:
    """从已完成的嵌入作业下载并处理结果 - 调用 graph_builder_qwen._process_embedding_results"""
    return gb._process_embedding_results(job, client, id_order)

def download_and_process_multiple_embedding_batches(client: OpenAI, job_ids: List[str], id_order: List[str]) -> np.ndarray:
    """
    从多个嵌入批次作业下载并合并结果（修复版：从结果中提取实际ID并匹配）

    Args:
        client: OpenAI客户端
        job_ids: 批次作业ID列表
        id_order: 实体ID顺序列表（完整列表）

    Returns:
        合并后的嵌入矩阵（按 id_order 顺序）
    """
    # 第一步：收集所有批次的嵌入结果到字典中
    logging.info(f"\n🔄 开始处理 {len(job_ids)} 个嵌入批次，收集所有结果...")
    all_embeddings_dict = {}  # {entity_id: embedding_vector}
    successful_batches = 0
    failed_batches = 0

    for idx, job_id in enumerate(job_ids, 1):
        logging.info(f"\n📦 处理嵌入批次 {idx}/{len(job_ids)}: {job_id}")

        try:
            job = client.batches.retrieve(batch_id=job_id)
            logging.info(f"   作业状态: {job.status}")

            if job.status == 'completed':
                # 下载结果文件
                output_file_id = job.output_file_id
                if not output_file_id:
                    logging.warning(f"   ⚠️ 批次 {idx} 没有输出文件")
                    failed_batches += 1
                    continue

                logging.info(f"   📥 下载批次 {idx} 的结果文件...")
                content = client.files.content(output_file_id).text
                lines = content.strip().split('\n')

                # 解析每一行，提取 custom_id 和 embedding
                batch_count = 0
                for line in lines:
                    try:
                        obj = json.loads(line)
                        cid = obj.get("custom_id")
                        response = obj.get("response", {})

                        if response.get("status_code") == 200:
                            body = response.get("body", {})
                            data = body.get("data", [])
                            if data and cid:
                                # 归一化嵌入向量
                                emb = np.array(data[0]["embedding"], dtype=float)
                                n = np.linalg.norm(emb)
                                all_embeddings_dict[cid] = emb / (n if n > 0 else 1.0)
                                batch_count += 1
                    except Exception as e:
                        logging.debug(f"   解析行失败: {e}")
                        continue

                successful_batches += 1
                logging.info(f"   ✅ 批次 {idx} 成功获取 {batch_count} 个嵌入向量")
            else:
                failed_batches += 1
                logging.error(f"   ❌ 批次 {idx} 状态不是成功: {job.status}")
        except Exception as e:
            failed_batches += 1
            logging.error(f"   ❌ 批次 {idx} 处理失败: {e}")

    logging.info(f"\n🎉 嵌入批次收集完成: 成功 {successful_batches}/{len(job_ids)}, 失败 {failed_batches}/{len(job_ids)}")
    logging.info(f"   总计收集到 {len(all_embeddings_dict)} 个实体的嵌入向量")

    # 第二步：按照 id_order 的顺序组装嵌入矩阵
    if not all_embeddings_dict:
        logging.error("   ❌ 所有批次均未获得有效嵌入向量")
        return np.zeros((0, 1), dtype=float)

    logging.info(f"\n🔧 按照 {len(id_order)} 个实体的顺序组装嵌入矩阵...")

    # 获取嵌入向量的维度
    default_dim = next(iter(all_embeddings_dict.values())).shape[0]

    vecs = []
    matched_count = 0
    unmatched_count = 0
    unmatched_samples = []

    for eid in id_order:
        str_eid = str(eid)
        found = False

        # 尝试1: 直接匹配
        if str_eid in all_embeddings_dict:
            vecs.append(all_embeddings_dict[str_eid])
            matched_count += 1
            found = True
        # 尝试2: 规范化匹配（_ 转 -）
        elif str_eid.replace('_', '-') in all_embeddings_dict:
            normalized_id = str_eid.replace('_', '-')
            vecs.append(all_embeddings_dict[normalized_id])
            matched_count += 1
            found = True
            if matched_count == 1:
                logging.info(f"   ℹ️ 使用 ID 规范化匹配 (下划线→连字符)")
        # 尝试3: 反向规范化匹配（- 转 _）
        elif str_eid.replace('-', '_') in all_embeddings_dict:
            normalized_id = str_eid.replace('-', '_')
            vecs.append(all_embeddings_dict[normalized_id])
            matched_count += 1
            found = True
            if matched_count == 1:
                logging.info(f"   ℹ️ 使用 ID 规范化匹配 (连字符→下划线)")

        if not found:
            if unmatched_count < 5:
                logging.warning(f"   ⚠️ 未找到实体 '{eid}' 的嵌入结果")
            if unmatched_count < 10:
                unmatched_samples.append(str_eid)
            vecs.append(np.zeros(default_dim, dtype=float))
            unmatched_count += 1

    if unmatched_count > 0:
        logging.warning(f"   ⚠️ 总计 {unmatched_count}/{len(id_order)} 个实体未找到嵌入结果")
        if unmatched_count > 5:
            logging.warning(f"      （仅显示前5个警告，还有 {unmatched_count - 5} 个未显示）")
        if unmatched_samples:
            logging.warning(f"      未匹配样本: {unmatched_samples}")

    logging.info(f"   ✅ 成功匹配 {matched_count}/{len(id_order)} 个实体的嵌入向量")

    V = np.vstack(vecs)
    logging.info(f"   📊 最终嵌入矩阵形状: {V.shape}")
    return V

def download_and_process_community_summary_results(client: OpenAI, job: Any) -> Dict[str, str]:
    """从已完成的社区摘要作业下载并处理结果 - 调用 graph_builder_qwen.process_results"""
    return gb.process_results(job, client)

def download_and_process_multiple_merge_batches(client: OpenAI, job_ids: List[str]) -> Dict[str, str]:
    """
    从多个实体合并批次作业下载并合并结果

    Args:
        client: OpenAI客户端
        job_ids: 批次作业ID列表

    Returns:
        合并后的merge结果字典
    """
    all_results = {}
    successful_batches = 0
    failed_batches = 0

    for idx, job_id in enumerate(job_ids, 1):
        logging.info(f"\n📦 处理合并批次 {idx}/{len(job_ids)}: {job_id}")
        try:
            job = client.batches.retrieve(batch_id=job_id)
            logging.info(f"   作业状态: {job.status}")

            if job.status == 'completed':
                batch_results = download_and_process_disambiguation_results(client, job)  # merge也使用相同格式
                if batch_results:
                    all_results.update(batch_results)
                    successful_batches += 1
                    logging.info(f"   ✅ 批次 {idx} 成功获取 {len(batch_results)} 个结果")
                else:
                    failed_batches += 1
                    logging.warning(f"   ⚠️ 批次 {idx} 未获取到结果")
            else:
                failed_batches += 1
                logging.error(f"   ❌ 批次 {idx} 状态不是成功: {job.status}")
        except Exception as e:
            failed_batches += 1
            logging.error(f"   ❌ 批次 {idx} 处理失败: {e}")

    logging.info(f"\n🎉 合并批次处理完成: 成功 {successful_batches}/{len(job_ids)}, 失败 {failed_batches}/{len(job_ids)}")
    logging.info(f"   总计获得 {len(all_results)} 个合并结果")
    return all_results

def download_and_process_multiple_community_summary_batches(client: OpenAI, job_ids: List[str]) -> Dict[str, str]:
    """
    从多个社区摘要批次作业下载并合并结果

    Args:
        client: OpenAI客户端
        job_ids: 批次作业ID列表

    Returns:
        合并后的社区摘要结果字典
    """
    all_summaries = {}
    successful_batches = 0
    failed_batches = 0

    for idx, job_id in enumerate(job_ids, 1):
        logging.info(f"\n📦 处理社区摘要批次 {idx}/{len(job_ids)}: {job_id}")
        try:
            job = client.batches.retrieve(batch_id=job_id)
            logging.info(f"   作业状态: {job.status}")

            if job.status == 'completed':
                batch_summaries = download_and_process_community_summary_results(client, job)
                if batch_summaries:
                    all_summaries.update(batch_summaries)
                    successful_batches += 1
                    logging.info(f"   ✅ 批次 {idx} 成功获取 {len(batch_summaries)} 个摘要")
                else:
                    failed_batches += 1
                    logging.warning(f"   ⚠️ 批次 {idx} 未获取到摘要")
            else:
                failed_batches += 1
                logging.error(f"   ❌ 批次 {idx} 状态不是成功: {job.status}")
        except Exception as e:
            failed_batches += 1
            logging.error(f"   ❌ 批次 {idx} 处理失败: {e}")

    logging.info(f"\n🎉 社区摘要批次处理完成: 成功 {successful_batches}/{len(job_ids)}, 失败 {failed_batches}/{len(job_ids)}")
    logging.info(f"   总计获得 {len(all_summaries)} 个摘��")
    return all_summaries

def main():
    setup_logging()
    logging.info("=" * 80)
    logging.info("快速恢复脚本启动（使用 graph_builder_qwen 函数）")
    logging.info(f"启动模式: {START_MODE}")
    logging.info("=" * 80)

    # 验证模式
    valid_modes = ["DISAMBIGUATION", "EMBEDDING", "MERGE", "COMMUNITY"]
    if START_MODE not in valid_modes:
        logging.error(f"❌ 无效的启动模式: {START_MODE}")
        logging.error(f"请选择以下模式之一: {', '.join(valid_modes)}")
        return

    config = load_config()

    # 初始化客户端 (OpenAI / Qwen)
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        logging.error("❌ 未找到 QWEN_API_KEY 环境变量")
        return

    # 不使用代理
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logging.info("✅ 阿里云百炼 (OpenAI兼容) 客户端初始化完成")

    # graph_builder_qwen 已在文件顶部导入为 gb

    # 加载配置参数
    sleep_interval = int(config["graph_builder"].get("sleep_interval", 5))
    model_name = config["llm"]["model"]
    prompt_dir = config["graph_builder"].get("prompt_dir", "prompts")
    weight_alpha = float(config["graph_builder"].get("community_importance_weight_alpha", 0.6))
    entity_topk = int(config["graph_builder"].get("entity_merge_topk", 10))
    entity_min_sim = float(config["graph_builder"].get("entity_merge_min_sim", 0.82))

    input_path = PROJECT_ROOT / config["graph_builder"]["input_path"]
    merge_req_path = PROJECT_ROOT / config["graph_builder"]["merge_requests_path"]
    community_requests_path = PROJECT_ROOT / config["graph_builder"]["community_requests_path"]
    reports_path = PROJECT_ROOT / config["graph_builder"]["community_reports_path"]
    final_graph_path = PROJECT_ROOT / config["graph_builder"]["output_graph_path"]

    # ========== 模式1: 从消歧结果开始 ==========
    if START_MODE == "DISAMBIGUATION":
        logging.info("\n🚀 模式: 从消歧结果开始恢复")
        logging.info("执行流程: 应用消歧 → 嵌入 → 合并 → 社区发现 → 社区摘要")

        # 步骤1: 检查是否已有消歧图文件
        disamb_graph_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_graph_path"]
        if disamb_graph_path.exists():
            logging.info(f"\n步骤1: 发现已有消歧图文件: {disamb_graph_path}")
            logging.info("直接加载该图，跳过消歧步骤")
            with open(disamb_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True)
            logging.info(f"✅ 成功加载消歧图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            logging.info("\n步骤1: 未找到消歧图文件，将从初始图开始并应用消歧结果")

            if not is_valid_job_id(DISAMBIGUATION_JOB_ID):
                logging.error("❌ 请先设置 DISAMBIGUATION_JOB_ID!")
                return

            # 加载初始图
            logging.info("加载初始图...")
            graph = gb.load_and_build_initial_graph(input_path)

            # 步骤2: 恢复并应用消歧结果（支持单个或多个批次）
            job_ids = normalize_job_id(DISAMBIGUATION_JOB_ID)

            if len(job_ids) == 1:
                # 单个作业ID
                logging.info(f"\n步骤2: 从作业 {job_ids[0]} 恢复消歧结果...")
                try:
                    disamb_job = client.batches.retrieve(batch_id=job_ids[0])
                    logging.info(f"作业状态: {disamb_job.status}")

                    if disamb_job.status == 'completed':
                        disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                        if disamb_results:
                            updated_count = apply_disambiguation_results(graph, disamb_results)
                            logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                            # 保存消歧图
                            gb.save_graph(graph, disamb_graph_path)
                            logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                        else:
                            logging.error("❌ 未获取到消歧结果")
                            return
                    else:
                        logging.error(f"❌ 作业状态不是成功: {disamb_job.status}")
                        return
                except Exception as e:
                    logging.error(f"❌ 获取消歧作业失败: {e}")
                    return
            else:
                # 多个批次作业ID
                logging.info(f"\n步骤2: 从 {len(job_ids)} 个消歧批次作业恢复结果...")
                logging.info(f"批次ID列表: {job_ids}")

                try:
                    disamb_results = download_and_process_multiple_disambiguation_batches(client, job_ids)
                    if disamb_results:
                        updated_count = apply_disambiguation_results(graph, disamb_results)
                        logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                        # 保存消歧图
                        gb.save_graph(graph, disamb_graph_path)
                        logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                    else:
                        logging.error("❌ 未获取到消歧结果")
                        return
                except Exception as e:
                    logging.error(f"❌ 获取消歧批次作业失败: {e}")
                    return

        # 步骤3: 生成嵌入（支持批次拆分）
        logging.info("\n步骤3: 生成实体嵌入...")
        ent_ids = [nid for nid, nd in graph.nodes(data=True) if nd.get('is_disambiguated')]
        ent_texts = []
        for nid in ent_ids:
            nd = graph.nodes[nid]
            text = f"{nd.get('name', '').strip()}\n{nd.get('description', '').strip()}".strip()
            ent_texts.append((nid, text or (nd.get('name') or nid)))

        logging.info(f"找到 {len(ent_ids)} 个已消歧的实体")

        if len(ent_ids) < 2:
            logging.warning("⚠️ 已消歧实体数量不足，无法继续")
            return

        # 使用配置中的嵌入模型和批次大小
        embed_model = config.get("embedding", {}).get("model", "text-embedding-v3")
        embed_dim = int(config.get("embedding", {}).get("dimensionality", 768))
        embedding_batch_size = int(config["graph_builder"].get("embedding_batch_size", 5000))
        tmp_emb_req_path = PROJECT_ROOT / config["graph_builder"]["embedding_requests_path"]

        num_entities = len(ent_texts)
        num_batches = (num_entities + embedding_batch_size - 1) // embedding_batch_size

        all_embeddings = []
        if num_batches > 1:
            logging.info(f"⚙️ 实体数量 {num_entities} 超过批次大小 {embedding_batch_size}，拆分为 {num_batches} 个批次")
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
                logging.info(f"   批次 {batch_idx + 1} 的 ID 样本: {batch_ent_ids[:3] if len(batch_ent_ids) >= 3 else batch_ent_ids}")
                gb._create_temp_embedding_requests(batch_ent_texts, batch_req_path, model_name=embed_model, dim=embed_dim)
                batch_jobs.append((batch_idx, batch_req_path, batch_ent_ids))

            # 并行提交所有嵌入作业
            submitted_jobs = []
            for batch_idx, batch_req_path, batch_ent_ids in batch_jobs:
                logging.info(f"📤 提交嵌入批次 {batch_idx + 1}/{num_batches}")
                emb_job = gb._submit_and_monitor_embedding_job(client, batch_req_path, embed_model, sleep_interval,
                                                              batch_idx, num_batches, monitor=False)
                if emb_job:
                    logging.info(f"   作业 ID: {emb_job.id}, 对应批次索引: {batch_idx}, ID 数量: {len(batch_ent_ids)}")
                    submitted_jobs.append((batch_idx, emb_job, batch_ent_ids))

            # 批量监控所有批次（一次轮询查询所有）
            logging.info(f"⏳ 开始批量监控 {len(submitted_jobs)} 个嵌入批次作业的完成状态...")
            completed_jobs = gb._monitor_multiple_jobs_completion(
                client,
                submitted_jobs,  # [(batch_idx, job, batch_ent_ids), ...]
                sleep_interval,
                job_type="EntityEmb"
            )

            # 处理所有批次的结果
            for batch_idx, completed_job, batch_ent_ids in sorted(completed_jobs, key=lambda x: x[0]):
                logging.info(f"📊 处理批次 {batch_idx + 1} 的结果，作业 ID: {completed_job.id}")
                logging.info(f"   该批次预期的 ID 数量: {len(batch_ent_ids)}")
                logging.info(f"   该批次 ID 样本: {batch_ent_ids[:3] if len(batch_ent_ids) >= 3 else batch_ent_ids}")
                batch_V = gb._process_embedding_results(completed_job, client, batch_ent_ids)
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
            logging.info(f"🔄 处理单批次：{num_entities} 个实体")
            gb._create_temp_embedding_requests(ent_texts, tmp_emb_req_path, model_name=embed_model, dim=embed_dim)
            emb_job = gb._submit_and_monitor_embedding_job(client, tmp_emb_req_path, embed_model, sleep_interval, 0, 1)
            V = gb._process_embedding_results(emb_job, client, ent_ids)

        logging.info(f"✅ 完成 {V.shape[0]} 个嵌入向量")

        # 步骤4: 实体合并
        logging.info("\n步骤4: 执行实体合并...")
        clusters = gb.build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim)
        logging.info(f"候选同义簇数量: {len(clusters)}")

        if clusters:
            gb.create_entity_merge_requests(graph, clusters, model_name=model_name,
                                        prompt_dir=prompt_dir, output_path=merge_req_path)
            merge_job = gb.submit_and_monitor_job(client, merge_req_path, model_name,
                                              sleep_interval, "EntityMerge")
            merge_texts = gb.process_results(merge_job, client)
            groups = gb.parse_entity_merge_results(merge_texts)
            logging.info(f"LLM 确认的分组数量: {len(groups)}")

            alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
            if alias2canon:
                logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')

        # 步骤5: 社区发现
        logging.info("\n步骤5: 执行社区发现...")
        graph = gb.detect_communities(graph, weight_alpha)

        # 步骤6: 社区摘要
        logging.info("\n步骤6: 生成社区摘要...")
        summaries = gb.run_community_summaries(client, graph, model_name, prompt_dir,
                                           config, sleep_interval, community_requests_path)

        # 保存结果
        logging.info("\n步骤7: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path)
        gb.save_graph(graph, final_graph_path)

    # ========== 模式2: 从嵌入结果开始 ==========
    elif START_MODE == "EMBEDDING":
        logging.info("\n🚀 模式: 从嵌入结果开始恢复")
        logging.info("执行流程: 应用消歧 → 应用嵌入 → 合并 → 社区发现 → 社区摘要")

        # 步骤1: 检查是否已有合并图文件
        merged_graph_path = PROJECT_ROOT / config["graph_builder"]["merged_graph_path"]
        if merged_graph_path.exists():
            logging.info(f"\n步骤1: 发现已有合并图文件: {merged_graph_path}")
            logging.info("直接加载该图，跳过消歧、嵌入、合并步骤")
            with open(merged_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True)
            logging.info(f"✅ 成功加载合并图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            logging.info("\n步骤1: 未找到合并图文件，需要执行嵌入和合并流程")

            # 检查消歧图
            disamb_graph_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_graph_path"]
            if disamb_graph_path.exists():
                logging.info(f"发现已有消歧图文件: {disamb_graph_path}")
                logging.info("直接加载消歧图，跳过消歧步骤")
                with open(disamb_graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                graph = nx.node_link_graph(data, directed=True)
                logging.info(f"✅ 成功加载消歧图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
            else:
                logging.info("未找到消歧图文件，从初始图开始")

                # 加载初始图
                graph = gb.load_and_build_initial_graph(input_path)

                # 步骤2: 应用消歧结果（从DISAMBIGUATION_JOB_ID，支持单个或多个批次）
                if is_valid_job_id(DISAMBIGUATION_JOB_ID):
                    job_ids = normalize_job_id(DISAMBIGUATION_JOB_ID)

                    if len(job_ids) == 1:
                        logging.info(f"\n步骤2: 从作业 {job_ids[0]} 应用消歧结果...")
                        try:
                            disamb_job = client.batches.retrieve(batch_id=job_ids[0])
                            if disamb_job.status == 'completed':
                                disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                                if disamb_results:
                                    updated_count = apply_disambiguation_results(graph, disamb_results)
                                    logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                    gb.save_graph(graph, disamb_graph_path)
                                    logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                                else:
                                    logging.warning("⚠️ 未获取到消歧结果")
                            else:
                                logging.warning(f"⚠️ 消歧作业状态不是成功: {disamb_job.status}")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取消歧作业失败: {e}")
                    else:
                        logging.info(f"\n步骤2: 从 {len(job_ids)} 个消歧批次作业应用结果...")
                        try:
                            disamb_results = download_and_process_multiple_disambiguation_batches(client, job_ids)
                            if disamb_results:
                                updated_count = apply_disambiguation_results(graph, disamb_results)
                                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                gb.save_graph(graph, disamb_graph_path)
                                logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到消歧结果")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取消歧批次作业失败: {e}")
                else:
                    logging.warning("\n步骤2: 未提供消歧作业ID，跳过消歧结果应用")

            # 步骤3: 准备实体列表
            logging.info("\n步骤3: 准备实体数据...")
            ent_ids = [nid for nid, nd in graph.nodes(data=True) if nd.get('is_disambiguated')]
            logging.info(f"找到 {len(ent_ids)} 个已消歧的实体")

            if len(ent_ids) < 2:
                logging.error("❌ 已消歧实体数量不足")
                return

            # 步骤4: 恢复嵌入结果（支持单个或多个批次）
            if not is_valid_job_id(EMBEDDING_JOB_ID):
                logging.error("❌ 请先设置 EMBEDDING_JOB_ID!")
                return

            emb_job_ids = normalize_job_id(EMBEDDING_JOB_ID)

            if len(emb_job_ids) == 1:
                logging.info(f"\n步骤4: 从作业 {emb_job_ids[0]} 恢复嵌入结果...")
                try:
                    emb_job = client.batches.retrieve(batch_id=emb_job_ids[0])
                    logging.info(f"作业状态: {emb_job.status}")

                    if emb_job.status == 'completed':
                        V = download_and_process_embedding_results(client, emb_job, ent_ids)
                        if V.shape[0] < 2:
                            logging.error("❌ 嵌入向量数量不足")
                            return
                        logging.info(f"✅ 成功恢复 {V.shape[0]} 个嵌入向量")
                    else:
                        logging.error(f"❌ 作业状态不是成功: {emb_job.status}")
                        return
                except Exception as e:
                    logging.error(f"❌ 获取嵌入作业失败: {e}")
                    return
            else:
                logging.info(f"\n步骤4: 从 {len(emb_job_ids)} 个嵌入批次作业恢复结果...")
                try:
                    V = download_and_process_multiple_embedding_batches(client, emb_job_ids, ent_ids)
                    if V.shape[0] < 2:
                        logging.error("❌ 嵌入向量数量不足")
                        return
                    logging.info(f"✅ 成功恢复 {V.shape[0]} 个嵌入向量")
                except Exception as e:
                    logging.error(f"❌ 获取嵌入批次作业失败: {e}")
                    return

            # 步骤5: 实体合并
            logging.info("\n步骤5: 执行实体合并...")
            clusters = gb.build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim)
            logging.info(f"候选同义簇数量: {len(clusters)}")

            if clusters:
                gb.create_entity_merge_requests(graph, clusters, model_name=model_name,
                                            prompt_dir=prompt_dir, output_path=merge_req_path)
                merge_job = gb.submit_and_monitor_job(client, merge_req_path, model_name,
                                                  sleep_interval, "EntityMerge")
                merge_texts = gb.process_results(merge_job, client)
                groups = gb.parse_entity_merge_results(merge_texts)
                logging.info(f"LLM 确认的分组数量: {len(groups)}")

                alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
                if alias2canon:
                    logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                    graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')

            # 保存合并后的图
            gb.save_graph(graph, merged_graph_path)
            logging.info(f"✅ 合并图已保存到: {merged_graph_path}")

        # 步骤6: 社区发现
        logging.info("\n步骤6: 执行社区发现...")
        graph = gb.detect_communities(graph, weight_alpha)

        # 步骤7: 社区摘要
        logging.info("\n步骤7: 生成社区摘要...")
        summaries = gb.run_community_summaries(client, graph, model_name, prompt_dir,
                                           config, sleep_interval, community_requests_path)

        # 保存结果
        logging.info("\n步骤8: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path)
        gb.save_graph(graph, final_graph_path)

    # ========== 模式3: 从合并结果开始 ==========
    elif START_MODE == "MERGE":
        logging.info("\n🚀 模式: 从合并结果开始恢复")
        logging.info("执行流程: 应用消歧 → 应用嵌入 → 应用合并 → 社区发现 → 社区摘要")

        # 步骤1: 检查是否已有合并图文件
        merged_graph_path = PROJECT_ROOT / config["graph_builder"]["merged_graph_path"]
        if merged_graph_path.exists():
            logging.info(f"\n步骤1: 发现已有合并图文件: {merged_graph_path}")
            logging.info("直接加载该图，跳过消歧、嵌入、合并步骤")
            with open(merged_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True)
            logging.info(f"✅ 成功加载合并图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            logging.info("\n步骤1: 未找到合并图文件，需要执行合并流程")

            if not is_valid_job_id(ENTITY_MERGE_JOB_ID):
                logging.error("❌ 请先设置 ENTITY_MERGE_JOB_ID!")
                return

            # 检查消歧图
            disamb_graph_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_graph_path"]
            if disamb_graph_path.exists():
                logging.info(f"发现已有消歧图文件: {disamb_graph_path}")
                logging.info("直接加载消歧图，跳过消歧步骤")
                with open(disamb_graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                graph = nx.node_link_graph(data, directed=True)
                logging.info(f"✅ 成功加载消歧图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
            else:
                logging.info("未找到消歧图文件，从初始图开始")

                # 加载初始图
                graph = gb.load_and_build_initial_graph(input_path)

                # 步骤2: 应用消歧结果（从DISAMBIGUATION_JOB_ID，支持单个或多个批次）
                if is_valid_job_id(DISAMBIGUATION_JOB_ID):
                    job_ids = normalize_job_id(DISAMBIGUATION_JOB_ID)

                    if len(job_ids) == 1:
                        logging.info(f"\n步骤2: 从作业 {job_ids[0]} 应用消歧结果...")
                        try:
                            disamb_job = client.batches.retrieve(batch_id=job_ids[0])
                            if disamb_job.status == 'completed':
                                disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                                if disamb_results:
                                    updated_count = apply_disambiguation_results(graph, disamb_results)
                                    logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                    gb.save_graph(graph, disamb_graph_path)
                                    logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                                else:
                                    logging.warning("⚠️ 未获取到消歧结果")
                            else:
                                logging.warning(f"⚠️ 消歧作业状态不是成功: {disamb_job.status}")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取消歧作业失败: {e}")
                    else:
                        logging.info(f"\n步骤2: 从 {len(job_ids)} 个消歧批次作业应用结果...")
                        try:
                            disamb_results = download_and_process_multiple_disambiguation_batches(client, job_ids)
                            if disamb_results:
                                updated_count = apply_disambiguation_results(graph, disamb_results)
                                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                gb.save_graph(graph, disamb_graph_path)
                                logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到消歧结果")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取消歧批次作业失败: {e}")
                else:
                    logging.warning("\n步骤2: 未提供消歧作业ID，跳过消歧结果应用")

            # 步骤3: 准备实体数据并验证
            ent_ids = [nid for nid, nd in graph.nodes(data=True) if nd.get('is_disambiguated')]
            logging.info(f"\n步骤3: 找到 {len(ent_ids)} 个已消歧的实体")

            if len(ent_ids) < 2:
                logging.error("❌ 已消歧实体数量不足，无法继续")
                return

            # 步骤4: 应用嵌入结果（从EMBEDDING_JOB_ID，支持单个或多个批次）
            # 注意：这里不需要实际应用嵌入向量到图中，只是为了后续合并验证
            if is_valid_job_id(EMBEDDING_JOB_ID):
                emb_job_ids = normalize_job_id(EMBEDDING_JOB_ID)
                logging.info(f"\n步骤4: 验证 {len(emb_job_ids)} 个嵌入作业...")

                for idx, job_id in enumerate(emb_job_ids, 1):
                    try:
                        emb_job = client.batches.retrieve(batch_id=job_id)
                        if emb_job.status == 'completed':
                            logging.info(f"   ✅ 嵌入作业 {idx}/{len(emb_job_ids)} 已完成")
                        else:
                            logging.warning(f"   ⚠️ 嵌入作业 {idx}/{len(emb_job_ids)} 状态: {emb_job.status}")
                    except Exception as e:
                        logging.warning(f"   ⚠️ 验证嵌入作业 {idx}/{len(emb_job_ids)} 失败: {e}")

            # 步骤5: 恢复并应用合并结果（支持单个或多个批次）
            merge_job_ids = normalize_job_id(ENTITY_MERGE_JOB_ID)

            if len(merge_job_ids) == 1:
                logging.info(f"\n步骤5: 从作业 {merge_job_ids[0]} 应用实体合并结果...")
                try:
                    merge_job = client.batches.retrieve(batch_id=merge_job_ids[0])
                    logging.info(f"作业状态: {merge_job.status}")

                    if merge_job.status == 'completed':
                        merge_texts = gb.process_results(merge_job, client)
                        if merge_texts:
                            groups = gb.parse_entity_merge_results(merge_texts)
                            logging.info(f"✅ 成功恢复 {len(groups)} 个LLM确认的分组")

                            alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
                            if alias2canon:
                                logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                                graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
                            else:
                                logging.warning("⚠️ 未找到需要合并的实体")

                            # 保存合并后的图
                            gb.save_graph(graph, merged_graph_path)
                            logging.info(f"✅ 合并图已保存到: {merged_graph_path}")
                        else:
                            logging.error("❌ 未获取到合并结果")
                            return
                    else:
                        logging.error(f"❌ 作业状态不是成功: {merge_job.status}")
                        return
                except Exception as e:
                    logging.error(f"❌ 获取合并作业失败: {e}")
                    return
            else:
                logging.info(f"\n步骤5: 从 {len(merge_job_ids)} 个合并批次作业应用结果...")
                try:
                    merge_texts = download_and_process_multiple_merge_batches(client, merge_job_ids)
                    if merge_texts:
                        groups = gb.parse_entity_merge_results(merge_texts)
                        logging.info(f"✅ 成功恢复 {len(groups)} 个LLM确认的分组")

                        alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
                        if alias2canon:
                            logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                            graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
                        else:
                            logging.warning("⚠️ 未找到需要合并的实体")

                        # 保存合并后的图
                        gb.save_graph(graph, merged_graph_path)
                        logging.info(f"✅ 合并图已保存到: {merged_graph_path}")
                    else:
                        logging.error("❌ 未获取到合并结果")
                        return
                except Exception as e:
                    logging.error(f"❌ 获取合并批次作业失败: {e}")
                    return

        # 步骤6: 社区发现
        logging.info("\n步骤6: 执行社区发现...")
        graph = gb.detect_communities(graph, weight_alpha)

        # 步骤7: 社区摘要
        logging.info("\n步骤7: 生成社区摘要...")
        summaries = gb.run_community_summaries(client, graph, model_name, prompt_dir,
                                           config, sleep_interval, community_requests_path)

        # 保存结果
        logging.info("\n步骤8: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path)
        gb.save_graph(graph, final_graph_path)

    # ========== 模式4: 从社区发现/摘要开始 ==========
    elif START_MODE == "COMMUNITY":
        logging.info("\n🚀 模式: 从社区发现开始恢复")
        logging.info("执行流程: 加载图 → 社区发现 → 社区摘要")

        # 步骤1: 检查并加载图（优先使用已合并的图，否则加载消歧图，最后才是初始图）
        logging.info("\n步骤1: 加载图...")
        merged_graph_path = PROJECT_ROOT / config["graph_builder"]["merged_graph_path"]
        disamb_graph_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_graph_path"]

        skip_preprocessing = False

        if merged_graph_path.exists():
            logging.info(f"发现已有合并图文件: {merged_graph_path}")
            logging.info("直接加载合并图，跳过所有前置步骤")
            with open(merged_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True)
            logging.info(f"✅ 成功加载合并图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
            skip_preprocessing = True
        elif disamb_graph_path.exists():
            logging.info(f"发现已有消歧图文件: {disamb_graph_path}")
            logging.info("加载消歧图，需要应用嵌入和合并步骤")
            with open(disamb_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True)
            logging.info(f"✅ 成功加载消歧图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            logging.info("未找到任何中间图文件，从初始图开始")
            graph = gb.load_and_build_initial_graph(input_path)

        # 如果需要，应用前置步骤
        if not skip_preprocessing:
            # 步骤2: 应用消歧结果（如果还没有消歧图，支持单个或多个批次）
            if not disamb_graph_path.exists():
                if is_valid_job_id(DISAMBIGUATION_JOB_ID):
                    job_ids = normalize_job_id(DISAMBIGUATION_JOB_ID)

                    if len(job_ids) == 1:
                        logging.info(f"\n步骤2: 从作业 {job_ids[0]} 应用消歧结果...")
                        try:
                            disamb_job = client.batches.retrieve(batch_id=job_ids[0])
                            if disamb_job.status == 'completed':
                                disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                                if disamb_results:
                                    updated_count = apply_disambiguation_results(graph, disamb_results)
                                    logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                    gb.save_graph(graph, disamb_graph_path)
                                    logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                                else:
                                    logging.warning("⚠️ 未获取到消歧结果")
                            else:
                                logging.warning(f"⚠️ 消歧作业状态不是成功: {disamb_job.status}")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取消歧作业失败: {e}")
                    else:
                        logging.info(f"\n步骤2: 从 {len(job_ids)} 个消歧批次作业应用结果...")
                        try:
                            disamb_results = download_and_process_multiple_disambiguation_batches(client, job_ids)
                            if disamb_results:
                                updated_count = apply_disambiguation_results(graph, disamb_results)
                                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                gb.save_graph(graph, disamb_graph_path)
                                logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到消歧结果")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取消歧批次作业失败: {e}")
                else:
                    logging.warning("\n步骤2: 未提供消歧作业ID，跳过消歧结果应用")

            # 步骤3: 验证嵌入作业（支持单个或多个批次）
            if is_valid_job_id(EMBEDDING_JOB_ID):
                emb_job_ids = normalize_job_id(EMBEDDING_JOB_ID)
                logging.info(f"\n步骤3: 验证 {len(emb_job_ids)} 个嵌入作业...")

                for idx, job_id in enumerate(emb_job_ids, 1):
                    try:
                        emb_job = client.batches.retrieve(batch_id=job_id)
                        if emb_job.status == 'completed':
                            logging.info(f"   ✅ 嵌入作业 {idx}/{len(emb_job_ids)} 已完成")
                        else:
                            logging.warning(f"   ⚠️ 嵌入作业 {idx}/{len(emb_job_ids)} 状态: {emb_job.status}")
                    except Exception as e:
                        logging.warning(f"   ⚠️ 验证嵌入作业 {idx}/{len(emb_job_ids)} 失败: {e}")

            # 步骤4: 应用合并结果（如果还没有合并图，支持单个或多个批次）
            if not merged_graph_path.exists():
                if is_valid_job_id(ENTITY_MERGE_JOB_ID):
                    merge_job_ids = normalize_job_id(ENTITY_MERGE_JOB_ID)

                    if len(merge_job_ids) == 1:
                        logging.info(f"\n步骤4: 从作业 {merge_job_ids[0]} 应用实体合并结果...")
                        try:
                            merge_job = client.batches.retrieve(batch_id=merge_job_ids[0])
                            if merge_job.status == 'completed':
                                merge_texts = gb.process_results(merge_job, client)
                                if merge_texts:
                                    groups = gb.parse_entity_merge_results(merge_texts)
                                    logging.info(f"✅ 成功恢复 {len(groups)} 个LLM确认的分组")

                                    alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
                                    if alias2canon:
                                        logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                                        graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
                                        gb.save_graph(graph, merged_graph_path)
                                        logging.info(f"✅ 合并图已保存到: {merged_graph_path}")
                                else:
                                    logging.warning("⚠️ 未获取到合并结果")
                            else:
                                logging.warning(f"⚠️ 合并作业状态不是成功: {merge_job.status}")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取合并作业失败: {e}")
                    else:
                        logging.info(f"\n步骤4: 从 {len(merge_job_ids)} 个合并批次作业应用结果...")
                        try:
                            merge_texts = download_and_process_multiple_merge_batches(client, merge_job_ids)
                            if merge_texts:
                                groups = gb.parse_entity_merge_results(merge_texts)
                                logging.info(f"✅ 成功恢复 {len(groups)} 个LLM确认的分组")

                                alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
                                if alias2canon:
                                    logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                                    graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
                                    gb.save_graph(graph, merged_graph_path)
                                    logging.info(f"✅ 合并图已保存到: {merged_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到合并结果")
                        except Exception as e:
                            logging.warning(f"⚠️ 获取合并批次作业失败: {e}")
                else:
                    logging.warning("\n步骤4: 未提供合并作业ID，跳过合并结果应用")

        # 步骤5: 检查/执行社区发现
        has_community = any('community' in data for _, data in graph.nodes(data=True))
        if not has_community:
            logging.info("\n步骤5: 图中没有社区信息，执行社区发现...")
            graph = gb.detect_communities(graph, weight_alpha)
            logging.info(f"✅ 社区发现完成")
        else:
            logging.info("\n步骤5: 图中已有社区信息，跳过社区发现")
            communities = set(data.get('community') for _, data in graph.nodes(data=True) if 'community' in data)
            logging.info(f"发现 {len(communities)} 个社区")

        # 步骤6: 社区摘要（恢复或生成，支持单个或多个批次）
        summaries = {}
        if is_valid_job_id(COMMUNITY_SUMMARY_JOB_ID):
            summary_job_ids = normalize_job_id(COMMUNITY_SUMMARY_JOB_ID)

            if len(summary_job_ids) == 1:
                logging.info(f"\n步骤6: 从作业 {summary_job_ids[0]} 恢复社区摘要结果...")
                try:
                    summary_job = client.batches.retrieve(batch_id=summary_job_ids[0])
                    logging.info(f"作业状态: {summary_job.status}")

                    if summary_job.status == 'completed':
                        summaries = download_and_process_community_summary_results(client, summary_job)
                        if summaries:
                            logging.info(f"✅ 成功恢复 {len(summaries)} 个社区的摘要")
                        else:
                            logging.warning("⚠️ 未获取到社区摘要结果，将重新生成")
                    else:
                        logging.warning(f"⚠️ 作业状态不是成功: {summary_job.status}，将重新生成")
                except Exception as e:
                    logging.warning(f"⚠️ 获取社区摘要作业失败: {e}，将重新生成")
            else:
                logging.info(f"\n步骤6: 从 {len(summary_job_ids)} 个社区摘要批次作业恢复结果...")
                try:
                    summaries = download_and_process_multiple_community_summary_batches(client, summary_job_ids)
                    if summaries:
                        logging.info(f"✅ 成功恢复 {len(summaries)} 个社区的摘要")
                    else:
                        logging.warning("⚠️ 未获取到社区摘要结果，将重新生成")
                except Exception as e:
                    logging.warning(f"⚠️ 获取社区摘要批次作业失败: {e}，将重新生成")

        # 如果没有成功恢复摘要，则生成新的
        if not summaries:
            logging.info("\n步骤6: 生成社区摘要...")
            summaries = gb.run_community_summaries(client, graph, model_name, prompt_dir,
                                               config, sleep_interval, community_requests_path)

        # 步骤7: 保存最终结果
        logging.info("\n步骤7: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path)
            logging.info(f"✅ 社区报告已保存到: {reports_path}")
        else:
            logging.warning("⚠️ 没有社区摘要可保存")

        gb.save_graph(graph, final_graph_path)
        logging.info(f"✅ 最终图已保存到: {final_graph_path}")

    logging.info("\n" + "=" * 80)
    logging.info("🎉🎉🎉 快速恢复流程完成！")
    logging.info("=" * 80)
    logging.info(f"社区报告: {reports_path}")
    logging.info(f"最终图: {final_graph_path}")

# apply_disambiguation_results 函数使用 gb（已在顶部导入）
def apply_disambiguation_results(graph: nx.DiGraph, disamb_results: Dict[str, str]) -> int:
    """应用消歧结果到图中的节点，支持ID规范化"""
    updated_count = 0
    for result_id, desc in disamb_results.items():
        # 使用 graph_builder 的 ID 规范化工具
        actual_id = gb.try_find_node_with_normalization(graph, result_id)
        if actual_id:
            graph.nodes[actual_id]['description'] = desc
            graph.nodes[actual_id]['is_disambiguated'] = True
            updated_count += 1
        else:
            # 方法3: 提取local_id并匹配 (保留作为降级方案)
            if '-e-' in result_id:
                local_id = result_id[result_id.rfind('-e-')+1:]
            else:
                local_id = result_id

            for node_id, node_data in graph.nodes(data=True):
                if node_data.get('local_id') == local_id or node_data.get('original_id') == local_id:
                    graph.nodes[node_id]['description'] = desc
                    graph.nodes[node_id]['is_disambiguated'] = True
                    updated_count += 1
                    break

    if updated_count < len(disamb_results) * 0.5:
        logging.warning(f"⚠️ 匹配率较低: {updated_count}/{len(disamb_results)} ({updated_count/len(disamb_results)*100:.1f}%)")

    return updated_count


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)