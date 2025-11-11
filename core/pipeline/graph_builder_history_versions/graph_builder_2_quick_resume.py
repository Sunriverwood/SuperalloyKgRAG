"""
完整恢复脚本 - 从云端恢复已完成的批处理作业并继续执行

使用方法：
1. 查看日志找到各个作业ID：
   - 消歧作业ID：搜索 "disambiguation" 或 "消歧"
   - 嵌入作业ID：搜索 "emb-entity-job" 或 "embedding"
   - 合并作业ID：搜索 "EntityMerge" 或 "entity_merge"
2. 在下面的配置区域填入对应的作业ID
3. 运行脚本: python core/pipeline/graph_builder_2_quick_resume.py

脚本会自动：
- 从云端下载已完成的消歧结果
- 从云端下载已完成的嵌入结果
- 从云端下载已完成的合并结果（如果提供了ENTITY_MERGE_JOB_ID）
- 执行社区发现
- 生成社区摘要
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import os
import numpy as np
import networkx as nx
from typing import Dict, List
import yaml
from google import genai
from utils.client_factory import create_gemini_client

# ========== 配置区域 - 请根据实际情况修改 ==========
DISAMBIGUATION_JOB_ID = "batches/lvrc1rylh8kv0ugjvh9zfods3fh6djk6yi63"  # 已知的消歧作业ID
EMBEDDING_JOB_ID = "batches/vzi891d2d5qoese284prt86d5wye2vz0vens"  # 实际的嵌入作业ID
ENTITY_MERGE_JOB_ID = "batches/ee8eeksp53mo4aucnk2kkvzkt6ph9pe5825i"  # 实体合并作业ID

# 如果你不知道某个作业ID，设置为 None，脚本会尝试查找或执行该步骤
# EMBEDDING_JOB_ID = None
# ENTITY_MERGE_JOB_ID = None
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

def main():
    setup_logging()
    logging.info("=" * 80)
    logging.info("快速恢复脚本启动")
    logging.info("=" * 80)

    # 检查配置
    if EMBEDDING_JOB_ID == "batches/YOUR_EMBEDDING_JOB_ID_HERE":
        logging.error("❌ 请先在脚本中设置 EMBEDDING_JOB_ID!")
        logging.info("提示：查看之前的日志，搜索包含 'emb-entity-job' 或 'embedding' 的作业ID")
        logging.info("格式应该是: batches/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return

    config = load_config()

    # 初始化客户端
    api_key = os.getenv("GEMINI_API_KEY") or config["llm"]["api_key"]
    proxy = config.get("proxy")
    client = create_gemini_client(api_key, proxy)
    logging.info("✅ Gemini 客户端初始化完成")

    # 导入完整的恢复脚本
    from core.pipeline import graph_builder_2_resume

    # 步骤1: 加载初始图
    logging.info("\n步骤1: 加载初始图...")
    input_path = PROJECT_ROOT / config["graph_builder"]["input_path"]
    graph = graph_builder_2_resume.load_graph_from_jsonl(input_path)

    # 步骤2: 使用已知的消歧作业ID恢复结果
    logging.info(f"\n步骤2: 从作业 {DISAMBIGUATION_JOB_ID} 恢复消歧结果...")
    try:
        disamb_job = client.batches.get(name=DISAMBIGUATION_JOB_ID)
        logging.info(f"作业状态: {disamb_job.state.name}")

        if disamb_job.state.name == 'JOB_STATE_SUCCEEDED':
            disamb_results = graph_builder_2_resume.download_and_process_disambiguation_results(client, disamb_job)

            if disamb_results:
                # 消歧结果的key格式可能是: chunk-xxx-e-1 (使用短横线)
                # 图节点ID格式是: chunk-xxx_e-1 (使用下划线)
                updated_count = 0
                for result_id, desc in disamb_results.items():
                    # 方法1: 直接匹配
                    if graph.has_node(result_id):
                        graph.nodes[result_id]['description'] = desc
                        graph.nodes[result_id]['is_disambiguated'] = True
                        updated_count += 1
                    else:
                        # 方法2: 转换ID格式 (hyphen -> underscore)
                        # 从 chunk-xxx-e-1 转换为 chunk-xxx_e-1
                        # 策略: 找到最后一个 '-e-' 并替换为 '_e-'
                        if '-e-' in result_id:
                            # 找到最后一个 '-e-' 的位置
                            last_e_pos = result_id.rfind('-e-')
                            converted_id = result_id[:last_e_pos] + '_' + result_id[last_e_pos+1:]

                            if graph.has_node(converted_id):
                                graph.nodes[converted_id]['description'] = desc
                                graph.nodes[converted_id]['is_disambiguated'] = True
                                updated_count += 1
                                continue

                        # 方法3: 提取local_id并匹配
                        # 从 chunk-xxx-e-1 提取 e-1
                        if '-e-' in result_id:
                            local_id = result_id[result_id.rfind('-e-')+1:]  # 获取最后的 'e-X' 部分
                        else:
                            local_id = result_id

                        # 遍历图节点查找匹配的local_id
                        for node_id, node_data in graph.nodes(data=True):
                            if node_data.get('local_id') == local_id:
                                graph.nodes[node_id]['description'] = desc
                                graph.nodes[node_id]['is_disambiguated'] = True
                                updated_count += 1
                                break

                logging.info(f"✅ 已更新 {updated_count} 个节点的消歧描述（共收到 {len(disamb_results)} 条结果）")

                # 如果匹配率低，给出警告
                if updated_count < len(disamb_results) * 0.5:
                    logging.warning(f"⚠️  匹配率较低: {updated_count}/{len(disamb_results)} ({updated_count/len(disamb_results)*100:.1f}%)")
                    logging.warning("   可能原因: 消歧结果ID格式与图节点ID格式不匹配")
            else:
                logging.warning("⚠️ 未获取到消歧结果")
        else:
            logging.error(f"❌ 作业状态不是成功: {disamb_job.state.name}")
    except Exception as e:
        logging.error(f"❌ 获取消歧作业失败: {e}")
        logging.info("提示：请检查作业ID是否正确")

    # 步骤3: 准备实体数据
    logging.info("\n步骤3: 准备实体数据...")
    ent_ids: List[str] = []
    for nid, nd in graph.nodes(data=True):
        if nd.get('is_disambiguated'):
            ent_ids.append(nid)

    logging.info(f"找到 {len(ent_ids)} 个已消歧的实体")

    # 步骤4: 恢复嵌入结果
    V = np.zeros((0, 1), dtype=float)
    if EMBEDDING_JOB_ID and EMBEDDING_JOB_ID != "None":
        logging.info(f"\n步骤4: 从作业 {EMBEDDING_JOB_ID} 恢复嵌入结果...")
        try:
            emb_job = client.batches.get(name=EMBEDDING_JOB_ID)
            logging.info(f"作业状态: {emb_job.state.name}")

            if emb_job.state.name == 'JOB_STATE_SUCCEEDED':
                V = graph_builder_2_resume.download_and_process_embedding_results(client, emb_job, ent_ids)
                logging.info(f"✅ 成功恢复 {V.shape[0]} 个嵌入向量")
            else:
                logging.error(f"❌ 作业状态不是成功: {emb_job.state.name}")
        except Exception as e:
            logging.error(f"❌ 获取嵌入作业失败: {e}")
            logging.info("提示：请检查嵌入作业ID是否正确")
    else:
        logging.info("\n步骤4: 尝试自动查找嵌入作业...")
        emb_job = graph_builder_2_resume.find_batch_job_by_display_name(client, "emb-entity")
        if emb_job:
            logging.info(f"找到作业: {emb_job.name}")
            V = graph_builder_2_resume.download_and_process_embedding_results(client, emb_job, ent_ids)
        else:
            logging.warning("⚠️ 未找到嵌入作业，将跳过实体合并")

    # 步骤5-8: 调用完整流程的后续步骤（实体合并、社区发现、摘要生成）
    logging.info("\n准备执行步骤5-8: 实体合并、社区发现和摘要生成...")

    from core.pipeline.graph_builder_2_resume import (
        build_candidate_clusters, create_entity_merge_requests,
        submit_and_monitor_job, process_results, parse_entity_merge_results,
        build_merge_map, apply_entity_merge, detect_communities,
        run_community_summaries, save_community_reports, save_final_graph
    )

    # 加载配置参数
    sleep_interval = int(config["graph_builder"].get("sleep_interval", 5))
    model_name = config["llm"]["model"]
    prompt_dir = config["graph_builder"].get("prompt_dir", "prompts")
    weight_alpha = float(config["graph_builder"].get("community_importance_weight_alpha", 0.6))
    entity_topk = int(config["graph_builder"].get("entity_merge_topk", 10))
    entity_min_sim = float(config["graph_builder"].get("entity_merge_min_sim", 0.82))

    merge_req_path = PROJECT_ROOT / config["graph_builder"]["merge_requests_path"]
    community_requests_path = PROJECT_ROOT / config["graph_builder"]["community_requests_path"]
    reports_path = PROJECT_ROOT / config["graph_builder"]["community_reports_path"]
    final_graph_path = PROJECT_ROOT / config["graph_builder"]["output_graph_path"]

    # 实体合并
    if V.shape[0] >= 2:
        logging.info("\n步骤5: 执行/恢复实体合并...")

        # 尝试从云端恢复已完成的合并结果
        merge_completed = False
        if ENTITY_MERGE_JOB_ID and ENTITY_MERGE_JOB_ID != "None":
            logging.info(f"从作业 {ENTITY_MERGE_JOB_ID} 恢复实体合并结果...")
            try:
                merge_job = client.batches.get(name=ENTITY_MERGE_JOB_ID)
                logging.info(f"作业状态: {merge_job.state.name}")

                if merge_job.state.name == 'JOB_STATE_SUCCEEDED':
                    logging.info("✅ 作业已完成，正在下载合并结果...")
                    merge_texts = process_results(merge_job, client)

                    if merge_texts:
                        groups = parse_entity_merge_results(merge_texts)
                        logging.info(f"✅ 成功恢复 {len(groups)} 个LLM确认的分组")

                        alias2canon, canon_name_map = build_merge_map(graph, groups)
                        if alias2canon:
                            logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
                            graph = apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
                            merge_completed = True
                        else:
                            logging.warning("⚠️  未生成合并映射")
                    else:
                        logging.warning("⚠️  未获取到合并结果")
                else:
                    logging.warning(f"⚠️  作业状态不是成功: {merge_job.state.name}")
            except Exception as e:
                logging.error(f"❌ 从云端恢复合并结果失败: {e}")
                logging.info("将尝试重新执行实体合并流程...")

        # 如果没有从云端恢复，则执行新的合并流程
        if not merge_completed:
            logging.info("执行新的实体合并流程...")
            clusters = build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim)
            logging.info(f"候选同义簇数量: {len(clusters)}")

            if clusters:
                create_entity_merge_requests(graph, clusters, model_name=model_name,
                                            prompt_dir=prompt_dir, output_path=merge_req_path)
                merge_job = submit_and_monitor_job(client, merge_req_path, model_name,
                                                  sleep_interval, "EntityMerge")
                merge_texts = process_results(merge_job, client)
                groups = parse_entity_merge_results(merge_texts)
                logging.info(f"LLM 确认的分组数量: {len(groups)}")

                alias2canon, canon_name_map = build_merge_map(graph, groups)
                if alias2canon:
                    graph = apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
            else:
                logging.info("未发现需要合并的实体簇")
    else:
        logging.warning("⚠️  向量数据不足，跳过实体合并")

    # 社区发现
    logging.info("\n步骤6: 执行社区发现...")
    graph = detect_communities(graph, weight_alpha)

    # 社区摘要
    logging.info("\n步骤7: 生成社区摘要...")
    summaries = run_community_summaries(client, graph, model_name, prompt_dir,
                                       config, sleep_interval, community_requests_path)

    # 保存结果
    logging.info("\n步骤8: 保存最终结果...")
    if summaries:
        save_community_reports(summaries, reports_path)
    save_final_graph(graph, final_graph_path)

    logging.info("\n" + "=" * 80)
    logging.info("🎉🎉🎉 快速恢复流程完成！")
    logging.info("=" * 80)
    logging.info(f"社区报告: {reports_path}")
    logging.info(f"最终图: {final_graph_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"程序执行时发生致命错误: {e}", exc_info=True)

