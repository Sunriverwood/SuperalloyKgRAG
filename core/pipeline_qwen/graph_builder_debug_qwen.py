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

# ========== 配置区域 - 请根据实际情况修改 ==========
# 恢复模式选择（必填）：
START_MODE = "EMBEDDING"

# 作业ID配置（根据选择的模式填写对应的ID）
DISAMBIGUATION_JOB_ID = "batch_req_xxxx" # 示例 ID
EMBEDDING_JOB_ID = "batch_7f3f7022-7cff-4509-9b00-7f69873d6540"
ENTITY_MERGE_JOB_ID = "batch_req_zzzz"
COMMUNITY_SUMMARY_JOB_ID = "batch_req_aaaa"

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

def download_and_process_disambiguation_results(client: OpenAI, job: Any) -> Dict[str, str]:
    """从已完成的消歧作业下载并处理结果 (OpenAI Format)"""
    if job.status != 'completed':
        logging.error(f"作业状态不是成功: {job.status}")
        return {}

    output_file_id = job.output_file_id
    if not output_file_id:
        logging.error("作业无输出文件ID")
        return {}

    try:
        logging.info(f"📥 正在下载消歧结果文件: {output_file_id}")
        content = client.files.content(output_file_id).text

        results = {}
        ok, bad = 0, 0
        for line in content.strip().split('\n'):
            try:
                obj = json.loads(line)
                key = obj.get("custom_id") # OpenAI Batch 使用 custom_id
                response = obj.get("response", {})

                if key and response.get("status_code") == 200:
                    # 提取 Chat Completion 内容
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                        results[key] = (text or "").strip()
                        ok += 1
                    else:
                        bad += 1
                else:
                    err = obj.get("error") or response.get("body")
                    logging.error(f"  - ❌ 处理 ID '{key}' 时发生错误: {err}")
                    bad += 1
            except Exception as e:
                logging.warning(f"  - ⚠️ 解析结果行失败: {e}")
                bad += 1

        logging.info(f"🎉 消歧结果处理完成：成功 {ok} 条，失败 {bad} 条。")
        return results
    except Exception as e:
        logging.error(f"❌ 下载消歧结果失败: {e}")
        return {}

def download_and_process_embedding_results(client: OpenAI, job: Any, id_order: List[str]) -> np.ndarray:
    """从已完成的嵌入作业下载并处理结果 (OpenAI Format)"""
    if job.status != 'completed':
        logging.error(f"嵌入作业状态不是成功: {job.status}")
        return np.zeros((0, 1), dtype=float)

    output_file_id = job.output_file_id
    if not output_file_id:
        return np.zeros((0, 1), dtype=float)

    try:
        logging.info(f"📥 正在下载嵌入结果文件: {output_file_id}")
        content = client.files.content(output_file_id).text

        lines = content.strip().split('\n')
        # 构建 map 以应对乱序
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
        default_dim = 768
        if res_map:
            default_dim = next(iter(res_map.values())).shape[0]

        for eid in id_order:
            if str(eid) in res_map:
                emb = res_map[str(eid)]
                n = np.linalg.norm(emb)
                vecs.append(emb / (n if n > 0 else 1.0))
            else:
                logging.warning(f"⚠️ 未找到实体 {eid} 的嵌入结果")
                vecs.append(np.zeros(default_dim, dtype=float))

        V = np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=float)
        logging.info(f"✅ 成功处理 {V.shape[0]} 个嵌入向量")
        return V
    except Exception as e:
        logging.error(f"❌ 下载嵌入结果失败: {e}")
        return np.zeros((0, 1), dtype=float)

def download_and_process_community_summary_results(client: OpenAI, job: Any) -> Dict[str, str]:
    """从已完成的社区摘要作业下载并处理结果 (OpenAI Format)"""
    if job.status != 'completed':
        logging.error(f"社区摘要作业状态不是成功: {job.status}")
        return {}

    output_file_id = job.output_file_id
    if not output_file_id:
        return {}

    try:
        logging.info(f"📥 正在下载社区摘要结果文件: {output_file_id}")
        content = client.files.content(output_file_id).text

        summaries = {}
        ok, bad = 0, 0
        for line in content.strip().split('\n'):
            try:
                obj = json.loads(line)
                key = obj.get("custom_id")
                response = obj.get("response", {})

                if key and response.get("status_code") == 200:
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                        summaries[key] = (text or "").strip()
                        ok += 1
                    else:
                        bad += 1
                else:
                    err = obj.get("error") or response.get("body")
                    logging.error(f"  - ❌ 处理社区 '{key}' 时发生错误: {err}")
                    bad += 1
            except Exception as e:
                logging.warning(f"  - ⚠️ 解析结果行失败: {e}")
                bad += 1

        logging.info(f"🎉 社区摘要结果处理完成：成功 {ok} 条，失败 {bad} 条。")
        return summaries
    except Exception as e:
        logging.error(f"❌ 下载社区摘要结果失败: {e}")
        return {}

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

    # 导入 graph_builder_qwen 的函数
    # 假设此时 graph_builder_qwen 已经迁移为 OpenAI SDK 版本
    try:
        from core.pipeline_qwen import graph_builder_qwen as gb
    except ImportError:
        logging.error("❌ 无法导入 core.pipeline_qwen.graph_builder_qwen，请确保文件存在且路径正确")
        return

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

            if not DISAMBIGUATION_JOB_ID or DISAMBIGUATION_JOB_ID.startswith("batch_req_xxxx"):
                logging.error("❌ 请先设置 DISAMBIGUATION_JOB_ID!")
                return

            # 加载初始图
            logging.info("加载初始图...")
            graph = gb.load_and_build_initial_graph(input_path)

            # 步骤2: 恢复并应用消歧结果
            logging.info(f"\n步骤2: 从作业 {DISAMBIGUATION_JOB_ID} 恢复消歧结果...")
            try:
                disamb_job = client.batches.retrieve(batch_id=DISAMBIGUATION_JOB_ID)
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

        # 步骤3: 生成嵌入
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

        # 使用配置中的嵌入模型
        embed_model = config.get("embedding", {}).get("model", "text-embedding-v3")
        embed_dim = int(config.get("embedding", {}).get("dimensionality", 768))
        tmp_emb_req_path = PROJECT_ROOT / config["graph_builder"]["embedding_requests_path"]

        gb._create_temp_embedding_requests(ent_texts, tmp_emb_req_path, model_name=embed_model, dim=embed_dim)
        emb_job = gb._submit_and_monitor_embedding_job(client, tmp_emb_req_path, embed_model, sleep_interval)
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

                # 步骤2: 应用消歧结果（从DISAMBIGUATION_JOB_ID）
                if DISAMBIGUATION_JOB_ID and not DISAMBIGUATION_JOB_ID.startswith("batch_req_"):
                    logging.info(f"\n步骤2: 从作业 {DISAMBIGUATION_JOB_ID} 应用消歧结果...")
                    try:
                        disamb_job = client.batches.retrieve(batch_id=DISAMBIGUATION_JOB_ID)
                        if disamb_job.status == 'completed':
                            disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                            if disamb_results:
                                updated_count = apply_disambiguation_results(graph, disamb_results)
                                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                # 保存消歧图
                                gb.save_graph(graph, disamb_graph_path)
                                logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到消歧结果")
                        else:
                            logging.warning(f"⚠️ 消歧作业状态不是成功: {disamb_job.status}")
                    except Exception as e:
                        logging.warning(f"⚠️ 获取消歧作业失败: {e}")
                else:
                    logging.warning("\n步骤2: 未提供消歧作业ID，跳过消歧结果应用")

            # 步骤3: 准备实体列表
            logging.info("\n步骤3: 准备实体数据...")
            ent_ids = [nid for nid, nd in graph.nodes(data=True) if nd.get('is_disambiguated')]
            logging.info(f"找到 {len(ent_ids)} 个已消歧的实体")

            if len(ent_ids) < 2:
                logging.error("❌ 已消歧实体数量不足")
                return

            # 步骤4: 恢复嵌入结果
            if not EMBEDDING_JOB_ID or EMBEDDING_JOB_ID.startswith("batch_req_"):
                logging.error("❌ 请先设置 EMBEDDING_JOB_ID!")
                return

            logging.info(f"\n步骤4: 从作业 {EMBEDDING_JOB_ID} 恢复嵌入结果...")
            try:
                emb_job = client.batches.retrieve(batch_id=EMBEDDING_JOB_ID)
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

            if not ENTITY_MERGE_JOB_ID or ENTITY_MERGE_JOB_ID.startswith("batch_req_"):
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

                # 步骤2: 应用消歧结果（从DISAMBIGUATION_JOB_ID）
                if DISAMBIGUATION_JOB_ID and not DISAMBIGUATION_JOB_ID.startswith("batch_req_"):
                    logging.info(f"\n步骤2: 从作业 {DISAMBIGUATION_JOB_ID} 应用消歧结果...")
                    try:
                        disamb_job = client.batches.retrieve(batch_id=DISAMBIGUATION_JOB_ID)
                        if disamb_job.status == 'completed':
                            disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                            if disamb_results:
                                updated_count = apply_disambiguation_results(graph, disamb_results)
                                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                # 保存消歧图
                                gb.save_graph(graph, disamb_graph_path)
                                logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到消歧结果")
                        else:
                            logging.warning(f"⚠️ 消歧作业状态不是成功: {disamb_job.status}")
                    except Exception as e:
                        logging.warning(f"⚠️ 获取消歧作业失败: {e}")
                else:
                    logging.warning("\n步骤2: 未提供消歧作业ID，跳过消歧结果应用")

            # 步骤3: 准备实体数据并验证
            ent_ids = [nid for nid, nd in graph.nodes(data=True) if nd.get('is_disambiguated')]
            logging.info(f"\n步骤3: 找到 {len(ent_ids)} 个已消歧的实体")

            if len(ent_ids) < 2:
                logging.error("❌ 已消歧实体数量不足，无法继续")
                return

            # 步骤4: 应用嵌入结果（从EMBEDDING_JOB_ID）
            # 注意：这里不需要实际应用嵌入向量到图中，只是为了后续合并验证
            logging.info(f"\n步骤4: 验证嵌入作业 {EMBEDDING_JOB_ID}...")
            if EMBEDDING_JOB_ID and not EMBEDDING_JOB_ID.startswith("batch_req_"):
                try:
                    emb_job = client.batches.retrieve(batch_id=EMBEDDING_JOB_ID)
                    if emb_job.status == 'completed':
                        logging.info("✅ 嵌入作业已完成")
                    else:
                        logging.warning(f"⚠️ 嵌入作业状态: {emb_job.status}")
                except Exception as e:
                    logging.warning(f"⚠️ 验证嵌入作业失败: {e}")

            # 步骤5: 恢复并应用合并结果
            logging.info(f"\n步骤5: 从作业 {ENTITY_MERGE_JOB_ID} 应用实体合并结果...")
            try:
                merge_job = client.batches.retrieve(batch_id=ENTITY_MERGE_JOB_ID)
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
            # 步骤2: 应用消歧结果（如果还没有消歧图）
            if not disamb_graph_path.exists():
                if DISAMBIGUATION_JOB_ID and not DISAMBIGUATION_JOB_ID.startswith("batch_req_"):
                    logging.info(f"\n步骤2: 从作业 {DISAMBIGUATION_JOB_ID} 应用消歧结果...")
                    try:
                        disamb_job = client.batches.retrieve(batch_id=DISAMBIGUATION_JOB_ID)
                        if disamb_job.status == 'completed':
                            disamb_results = download_and_process_disambiguation_results(client, disamb_job)
                            if disamb_results:
                                updated_count = apply_disambiguation_results(graph, disamb_results)
                                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述")
                                # 保存消歧图
                                gb.save_graph(graph, disamb_graph_path)
                                logging.info(f"✅ 消歧图已保存到: {disamb_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到消歧结果")
                        else:
                            logging.warning(f"⚠️ 消歧作业状态不是成功: {disamb_job.status}")
                    except Exception as e:
                        logging.warning(f"⚠️ 获取消歧作业失败: {e}")
                else:
                    logging.warning("\n步骤2: 未提供消歧作业ID，跳过消歧结果应用")

            # 步骤3: 验证嵌入作业
            if EMBEDDING_JOB_ID and not EMBEDDING_JOB_ID.startswith("batch_req_"):
                logging.info(f"\n步骤3: 验证嵌入作业 {EMBEDDING_JOB_ID}...")
                try:
                    emb_job = client.batches.retrieve(batch_id=EMBEDDING_JOB_ID)
                    if emb_job.status == 'completed':
                        logging.info("✅ 嵌入作业已完成")
                    else:
                        logging.warning(f"⚠️ 嵌入作业状态: {emb_job.status}")
                except Exception as e:
                    logging.warning(f"⚠️ 验证嵌入作业失败: {e}")

            # 步骤4: 应用合并结果（如果还没有合并图）
            if not merged_graph_path.exists():
                if ENTITY_MERGE_JOB_ID and not ENTITY_MERGE_JOB_ID.startswith("batch_req_"):
                    logging.info(f"\n步骤4: 从作业 {ENTITY_MERGE_JOB_ID} 应用实体合并结果...")
                    try:
                        merge_job = client.batches.retrieve(batch_id=ENTITY_MERGE_JOB_ID)
                        if merge_job.status == 'completed':
                            merge_texts = gb.process_results(merge_job, client)
                            if merge_texts:
                                groups = gb.parse_entity_merge_results(merge_texts)
                                logging.info(f"✅ 成功恢复 {len(groups)} 个LLM确认的分组")

                                alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
                                if alias2canon:
                                    logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                                    graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
                                    # 保存合并后的图
                                    gb.save_graph(graph, merged_graph_path)
                                    logging.info(f"✅ 合并图已保存到: {merged_graph_path}")
                            else:
                                logging.warning("⚠️ 未获取到合并结果")
                        else:
                            logging.warning(f"⚠️ 合并作业状态不是成功: {merge_job.status}")
                    except Exception as e:
                        logging.warning(f"⚠️ 获取合并作业失败: {e}")
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

        # 步骤6: 社区摘要（恢复或生成）
        summaries = {}
        if COMMUNITY_SUMMARY_JOB_ID and not COMMUNITY_SUMMARY_JOB_ID.startswith("batch_req_"):
            logging.info(f"\n步骤6: 从作业 {COMMUNITY_SUMMARY_JOB_ID} 恢复社区摘要结果...")
            try:
                summary_job = client.batches.retrieve(batch_id=COMMUNITY_SUMMARY_JOB_ID)
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

from core.pipeline_qwen import graph_builder_qwen as gb # 再次导入确保引用
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