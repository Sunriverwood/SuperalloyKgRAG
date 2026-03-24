# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

- "RESUBMIT_MERGE": 重新提交已有的合并请求文件
  * 用途：合并LLM仲裁批次因余额不足/API限流等原因失败后，直接重提交已有的请求文件
  * 不重新下载嵌入向量，不重新构建候选簇
  * 自动查找 data/cache/entities_merge_requests_batch_*.jsonl 文件并提交
  * 提交完成后继续执行：合并 → 社区发现 → 社区摘要

  使用方法：
  1. 设置 START_MODE = "COMMUNITY"
  2. 确保已有 data/graphs/disambiguation_graph.json
  3. 确保已有 data/cache/entities_merge_requests_batch_*.jsonl
  4. 运行脚本

- "COMMUNITY": 从社区发现开始
  * 如果已有合并图：直接加载 → 社区发现 →【下载并应用社区摘要或重新生成】
  * 如果已有消歧图：加载消歧图 →【应用前置步骤】→ 社区发现 →【下载并应用社区摘要或重新生成】
  * 否则：【应用所有前置步骤】→ 社区发现 →【下载并应用社区摘要或重新生成】

- "COMMUNITY_LEVEL_RESUME": 从指定层级恢复社区报告生成【新增】
  * 用途：当某个层级（如Level 3）完成后，下一层级（如Level 2）失败时使用
  * 优先从本地检查点加载已完成报告
  * 如果没有检查点，自动从 COMMUNITY_SUMMARY_JOB_ID 下载云端作业结果
  * 从指定层级（RESUME_FROM_LEVEL）继续生成报告

  使用方法：
  1. 设置 START_MODE = "COMMUNITY_LEVEL_RESUME"
  2. 设置 RESUME_FROM_LEVEL = 2  # 从Level 2开始恢复（Level 3已完成）
  3. 设置 COMMUNITY_SUMMARY_JOB_ID = ["batch_xxx"]  # Level 3的作业ID
  4. 确保已有 data/graphs/merged_graph.json
  5. 运行脚本，系统会：
     - 首先尝试加载本地检查点
     - 如果没有检查点，从云端作业下载 Level 3 的结果
     - 保存为检查点（方便下次恢复）
     - 从 Level 2 继续生成

- "DOWNLOAD_COMMUNITY_REPORTS": 从云端下载已完成的社区报告并保存到本地【新增】
  * 用途：当所有层级（Level 3到Level 0）的社区报告都在云端生成完成后，
         批量下载并保存为本地的社区报告文件
  * 自动解析每个层级的作业结果
  * 合并所有层级的报告到一个文件
  * 按层级排序保存（Level 0在前，Level 3在后）

  使用方法：
  1. 设置 START_MODE = "DOWNLOAD_COMMUNITY_REPORTS"
  2. 设置 COMMUNITY_SUMMARY_JOB_IDS_BY_LEVEL = {
         3: ["batch_level3_xxx"],  # Level 3的作业ID列表
         2: ["batch_level2_xxx"],  # Level 2的作业ID列表
         1: ["batch_level1_xxx"],  # Level 1的作业ID列表
         0: ["batch_level0_xxx"],  # Level 0的作业ID列表
     }
  3. 确保已有 data/graphs/final_graph.json（用于获取社区元数据）
  4. 运行脚本，系统会：
     - 依次从各层级的云端作业下载报告
     - 合并所有报告
     - 保存为本地的社区报告文件

工作原理：
- 每个模式都会首先检查对应阶段的图文件是否存在（消歧图、合并图）
- 如果存在，直接加载该图并跳过前置步骤
- 如果不存在，才从云端作业下载并应用结果，然后保存中间图文件供下次使用
- 这样可以最大程度复用已有结果，避免重复下载和处理
- 每个社区层级完成后，会保存检查点文件，支持从任意层级恢复

中间图文件：
- 消歧图: data/graphs/disambiguation_graph.json
- 合并图: data/graphs/merged_graph.json
- 最终图: data/graphs/final_graph.json

此版本直接使用 graph_builder 的函数，用于验证 graph_builder 代码的正确性。
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import collections.abc
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

# 导入社区报告模块的恢复功能
from utils.community_reports import (
    load_level_checkpoint,
    run_hierarchical_community_summaries_with_resume,
    merge_all_level_id_maps
)

# ========== 配置区域 - 请根据实际情况修改 ==========
# 恢复模式选择（必填）：
# 可选值: "DISAMBIGUATION", "EMBEDDING", "MERGE", "RESUBMIT_MERGE", "COMMUNITY", "COMMUNITY_LEVEL_RESUME", "DOWNLOAD_COMMUNITY_REPORTS"
START_MODE = "COMMUNITY"

# 层级恢复配置（仅在 START_MODE = "COMMUNITY_LEVEL_RESUME" 时使用）
# 从哪个层级开始恢复（例如：如果Level 3已完成但Level 2失败，设置为2）
RESUME_FROM_LEVEL = 2

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

COMMUNITY_SUMMARY_JOB_ID = ["batch_93c74a8c-233b-44bc-ae8b-9d2805f52cf4"]
# COMMUNITY_SUMMARY_JOB_ID = ["batch_xxx_1", "batch_xxx_2"]

# 各层级社区报告作业ID配置（仅在 START_MODE = "DOWNLOAD_COMMUNITY_REPORTS" 时使用）
# 格式：{层级: [作业ID列表]}，从高层级到低层级配置
COMMUNITY_SUMMARY_JOB_IDS_BY_LEVEL = {
    3: ["batch_d7ad65d0-d1c4-45f9-bcdb-8e8396182a5d"],  # Level 3的作业ID列表
    2: ["batch_e3c4ce99-5825-469a-8216-42bbd02c00c9"],  # Level 2的作业ID列表
    1: ["batch_7f357a99-5c13-4569-b5bc-a3c627f144d2"],  # Level 1的作业ID列表
    0: ["batch_0563d1a7-ba22-46e1-aa79-12935253580a"],  # Level 0的作业ID列表
}

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


def deep_update(d: dict, u: dict) -> dict:
    """递归深度更新字典"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def apply_ablation_config(config: Dict[str, Any], ablation_name: str) -> Dict[str, Any]:
    """将消融实验配置深层覆盖到主配置"""
    ablation_profiles = config.get("ablation", {})
    if ablation_name not in ablation_profiles:
        raise ValueError(f"未找到消融实验配置: '{ablation_name}'，可用: {list(ablation_profiles.keys())}")
    profile = ablation_profiles[ablation_name]
    logging.info(f"🔬 应用消融实验配置: '{ablation_name}' - {profile.get('description', '')}")
    for section, overrides in profile.items():
        if section in ['description', 'multidimensional_evaluation']:
            continue
        if section in config and isinstance(config[section], dict) and isinstance(overrides, dict):
            deep_update(config[section], overrides)
            logging.info(f"   ✅ 已深层覆盖 '{section}' 配置")
    return config

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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Graph Builder Debug/恢复脚本")
    parser.add_argument('--ablation', type=str, default=None,
                        help='消融实验名称，如 no_entity_merge')
    args = parser.parse_args()

    setup_logging()
    logging.info("=" * 80)
    logging.info("快速恢复脚本启动（使用 graph_builder_qwen 函数）")
    logging.info(f"启动模式: {START_MODE}")
    if args.ablation:
        logging.info(f"消融实验: {args.ablation}")
    logging.info("=" * 80)

    # 验证模式
    valid_modes = ["DISAMBIGUATION", "EMBEDDING", "MERGE", "RESUBMIT_MERGE", "COMMUNITY", "COMMUNITY_LEVEL_RESUME", "DOWNLOAD_COMMUNITY_REPORTS"]
    if START_MODE not in valid_modes:
        logging.error(f"❌ 无效的启动模式: {START_MODE}")
        logging.error(f"请选择以下模式之一: {', '.join(valid_modes)}")
        return

    config = load_config()

    # 应用消融实验配置（如果指定）
    if args.ablation:
        config = apply_ablation_config(config, args.ablation)

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

    # 社区检测配置参数
    use_hierarchical = config["graph_builder"].get("use_hierarchical_communities", True)
    max_level = int(config["graph_builder"].get("max_community_level", 10))
    min_community_size = int(config["graph_builder"].get("min_community_size", 10))
    logging.info(f"社区检测配置: use_hierarchical={use_hierarchical}, max_level={max_level}, min_community_size={min_community_size}")
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
        clusters = gb.build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim, config=config)
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
        graph, communities_list = gb.detect_communities(
            graph=graph,
            weight_alpha=weight_alpha,
            use_hierarchical=use_hierarchical,
            max_level=max_level,
            min_community_size=min_community_size
        )

        # 步骤6: 社区摘要
        logging.info("\n步骤6: 生成社区摘要...")
        summaries = gb.run_community_summaries(
            client=client,
            graph=graph,
            model_name=model_name,
            prompt_dir=prompt_dir,
            config=config,
            sleep_interval=sleep_interval,
            community_requests_path=community_requests_path,
            communities_list=communities_list
        )

        # 保存结果
        logging.info("\n步骤7: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path, communities_list)
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
            clusters = gb.build_candidate_clusters(V, ent_ids, topk=entity_topk, min_sim=entity_min_sim, config=config)
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
        graph, communities_list = gb.detect_communities(
            graph=graph,
            weight_alpha=weight_alpha,
            use_hierarchical=use_hierarchical,
            max_level=max_level,
            min_community_size=min_community_size
        )

        # 步骤7: 社区摘要
        logging.info("\n步骤7: 生成社区摘要...")
        summaries = gb.run_community_summaries(
            client=client,
            graph=graph,
            model_name=model_name,
            prompt_dir=prompt_dir,
            config=config,
            sleep_interval=sleep_interval,
            community_requests_path=community_requests_path,
            communities_list=communities_list
        )

        # 保存结果
        logging.info("\n步骤8: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path, communities_list)
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
        graph, communities_list = gb.detect_communities(
            graph=graph,
            weight_alpha=weight_alpha,
            use_hierarchical=use_hierarchical,
            max_level=max_level,
            min_community_size=min_community_size
        )

        # 步骤7: 社区摘要
        logging.info("\n步骤7: 生成社区摘要...")
        summaries = gb.run_community_summaries(
            client=client,
            graph=graph,
            model_name=model_name,
            prompt_dir=prompt_dir,
            config=config,
            sleep_interval=sleep_interval,
            community_requests_path=community_requests_path,
            communities_list=communities_list
        )

        # 保存结果
        logging.info("\n步骤8: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path, communities_list)
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
            graph = nx.node_link_graph(data, directed=True, edges="links")
            logging.info(f"✅ 成功加载合并图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
            skip_preprocessing = True
        elif disamb_graph_path.exists():
            logging.info(f"发现已有消歧图文件: {disamb_graph_path}")
            logging.info("加载消歧图，需要应用嵌入和合并步骤")
            with open(disamb_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True, edges="links")
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
        communities_list = None  # 初始化
        if not has_community:
            logging.info("\n步骤5: 图中没有社区信息，执行社区发现...")
            graph, communities_list = gb.detect_communities(
                graph=graph,
                weight_alpha=weight_alpha,
                use_hierarchical=use_hierarchical,
                max_level=max_level,
                min_community_size=min_community_size
            )
            logging.info(f"✅ 社区发现完成")
        else:
            logging.info("\n步骤5: 图中已有社区信息，跳过社区发现")
            communities = set(data.get('community') for _, data in graph.nodes(data=True) if 'community' in data)
            logging.info(f"发现 {len(communities)} 个社区")
            logging.info("⚠️ 注意：使用已有社区信息时，社区报告将使用传统扁平模式")

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
            summaries = gb.run_community_summaries(
                client=client,
                graph=graph,
                model_name=model_name,
                prompt_dir=prompt_dir,
                config=config,
                sleep_interval=sleep_interval,
                community_requests_path=community_requests_path,
                communities_list=communities_list
            )

        # 步骤7: 保存最终结果
        logging.info("\n步骤7: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path, communities_list)
            logging.info(f"✅ 社区报告已保存到: {reports_path}")
        else:
            logging.warning("⚠️ 没有社区摘要可保存")

        gb.save_graph(graph, final_graph_path)
        logging.info(f"✅ 最终图已保存到: {final_graph_path}")

    # ========== 模式5: 从指定层级恢复社区报告生成 ==========
    elif START_MODE == "COMMUNITY_LEVEL_RESUME":
        logging.info("\n🚀 模式: 从指定层级恢复社区报告生成")
        logging.info(f"执行流程: 加载已完成报告 → 从 Level {RESUME_FROM_LEVEL} 继续 → 完成所有层级")

        # 步骤1: 加载图
        logging.info("\n步骤1: 加载图...")
        merged_graph_path = PROJECT_ROOT / config["graph_builder"]["merged_graph_path"]

        if merged_graph_path.exists():
            logging.info(f"加载合并图文件: {merged_graph_path}")
            with open(merged_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True, edges="links")
            logging.info(f"✅ 成功加载合并图：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            logging.error(f"❌ 未找到合并图文件: {merged_graph_path}")
            logging.error("   请先运行 MERGE 或 COMMUNITY 模式生成合并图")
            return

        # 步骤2: 尝试加载已完成的报告（优先检查点，其次云端作业）
        logging.info(f"\n步骤2: 加载已完成的报告...")
        preloaded_summaries = {}

        # 2.1 首先尝试从本地检查点加载
        checkpoint_summaries, actual_level = load_level_checkpoint(
            community_requests_path,
            RESUME_FROM_LEVEL + 1  # 尝试加载比目标层级更深一层的检查点
        )

        if checkpoint_summaries:
            preloaded_summaries = checkpoint_summaries
            logging.info(f"✅ 从检查点加载了 {len(preloaded_summaries)} 个已完成的报告")
            if actual_level is not None:
                logging.info(f"   检查点来自层级: {actual_level}")

        # 2.2 如果没有检查点，尝试从云端作业下载
        if not preloaded_summaries and is_valid_job_id(COMMUNITY_SUMMARY_JOB_ID):
            logging.info("   未找到本地检查点，尝试从云端作业下载已完成的报告...")
            summary_job_ids = normalize_job_id(COMMUNITY_SUMMARY_JOB_ID)

            try:
                if len(summary_job_ids) == 1:
                    logging.info(f"   从作业 {summary_job_ids[0]} 下载报告...")
                    summary_job = client.batches.retrieve(batch_id=summary_job_ids[0])
                    logging.info(f"   作业状态: {summary_job.status}")

                    if summary_job.status == 'completed':
                        preloaded_summaries = download_and_process_community_summary_results(client, summary_job)
                        if preloaded_summaries:
                            logging.info(f"   ✅ 从云端下载了 {len(preloaded_summaries)} 个报告")

                            # 保存为检查点，方便下次恢复
                            checkpoint_path = community_requests_path.parent / f"community_summaries_checkpoint_level{RESUME_FROM_LEVEL + 1}.json"
                            try:
                                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                                    json.dump({
                                        "level": RESUME_FROM_LEVEL + 1,
                                        "total_summaries": len(preloaded_summaries),
                                        "summaries": preloaded_summaries,
                                        "source": "cloud_job",
                                        "job_ids": summary_job_ids
                                    }, f, ensure_ascii=False, indent=2)
                                logging.info(f"   💾 已保存检查点: {checkpoint_path.name}")
                            except Exception as e:
                                logging.warning(f"   ⚠️ 保存检查点失败: {e}")
                        else:
                            logging.warning("   ⚠️ 云端作业未返回有效报告")
                    else:
                        logging.warning(f"   ⚠️ 云端作业状态不是已完成: {summary_job.status}")
                else:
                    logging.info(f"   从 {len(summary_job_ids)} 个批次作业下载报告...")
                    preloaded_summaries = download_and_process_multiple_community_summary_batches(client, summary_job_ids)
                    if preloaded_summaries:
                        logging.info(f"   ✅ 从云端下载了 {len(preloaded_summaries)} 个报告")

                        # 保存为检查点
                        checkpoint_path = community_requests_path.parent / f"community_summaries_checkpoint_level{RESUME_FROM_LEVEL + 1}.json"
                        try:
                            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                                json.dump({
                                    "level": RESUME_FROM_LEVEL + 1,
                                    "total_summaries": len(preloaded_summaries),
                                    "summaries": preloaded_summaries,
                                    "source": "cloud_job",
                                    "job_ids": summary_job_ids
                                }, f, ensure_ascii=False, indent=2)
                            logging.info(f"   💾 已保存检查点: {checkpoint_path.name}")
                        except Exception as e:
                            logging.warning(f"   ⚠️ 保存检查点失败: {e}")
                    else:
                        logging.warning("   ⚠️ 云端作业未返回有效报告")
            except Exception as e:
                logging.error(f"   ❌ 从云端下载报告失败: {e}")

        if not preloaded_summaries:
            logging.warning("⚠️ 未能加载任何已完成的报告，将从头开始生成所有层级")

        # 步骤3: 执行社区发现（如果图中没有社区信息）
        has_community = any('community' in data for _, data in graph.nodes(data=True))
        communities_list = None

        if not has_community:
            logging.info("\n步骤3: 图中没有社区信息，执行社区发现...")
            graph, communities_list = gb.detect_communities(
                graph=graph,
                weight_alpha=weight_alpha,
                use_hierarchical=use_hierarchical,
                max_level=max_level,
                min_community_size=min_community_size
            )
            logging.info(f"✅ 社区发现完成")
        else:
            logging.info("\n步骤3: 图中已有社区信息，尝试从图中重建communities_list...")
            # 尝试从图节点中重建communities_list
            communities_by_level = {}
            for node_id, node_data in graph.nodes(data=True):
                if 'community' in node_data:
                    comm_id = str(node_data['community'])
                    level = node_data.get('community_level', 0)
                    if comm_id not in communities_by_level:
                        communities_by_level[comm_id] = {
                            'community_id': comm_id,
                            'level': level,
                            'node_ids': [],
                            'children_ids': node_data.get('community_children', []),
                            'parent_id': node_data.get('community_parent')
                        }
                    communities_by_level[comm_id]['node_ids'].append(node_id)

            if communities_by_level:
                communities_list = list(communities_by_level.values())
                logging.info(f"✅ 从图中重建了 {len(communities_list)} 个社区的信息")
            else:
                logging.warning("⚠️ 无法从图中重建社区信息")

        # 步骤4: 从已加载的报告恢复，继续生成剩余层级
        logging.info(f"\n步骤4: 从 Level {RESUME_FROM_LEVEL} 继续生成社区摘要...")
        logging.info(f"   已预加载 {len(preloaded_summaries)} 个报告")

        if communities_list:
            max_report_words = str(config["graph_builder"].get("community_summary_max_report_words", 800))
            max_entities = int(config["graph_builder"].get("community_summary_used_entities_num", 25))
            max_relationships = int(config["graph_builder"].get("community_summary_used_relationships_num", 50))

            summaries = run_hierarchical_community_summaries_with_resume(
                client=client,
                graph=graph,
                model_name=model_name,
                prompt_dir=prompt_dir,
                config=config,
                sleep_interval=sleep_interval,
                community_requests_path=community_requests_path,
                communities_list=communities_list,
                max_report_words=max_report_words,
                max_entities=max_entities,
                max_relationships=max_relationships,
                load_prompt_func=gb.load_prompt,
                build_context_func=gb.build_community_context,
                submit_job_func=gb.submit_and_monitor_job,
                process_results_func=gb.process_results,
                resume_from_level=RESUME_FROM_LEVEL,
                preloaded_summaries=preloaded_summaries
            )
        else:
            logging.error("❌ 无法获取社区列表，无法继续")
            return

        # 步骤5: 保存最终结果
        logging.info("\n步骤5: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path, communities_list)
            logging.info(f"✅ 社区报告已保存到: {reports_path}")
        else:
            logging.warning("⚠️ 没有社区摘要可保存")

        gb.save_graph(graph, final_graph_path)
        logging.info(f"✅ 最终图已保存到: {final_graph_path}")

    # ========== 模式6: 从云端下载所有层级的社区报告 ==========
    elif START_MODE == "DOWNLOAD_COMMUNITY_REPORTS":
        logging.info("\n🚀 模式: 从云端下载所有层级的社区报告")
        logging.info("执行流程: 加载最终图谱 → 构建社区结构 → 创建请求文件 → 下载云端结果 → 保存报告")

        # 步骤1: 加载最终图谱
        logging.info("\n步骤1: 加载最终图谱...")

        if final_graph_path.exists():
            logging.info(f"加载最终图谱文件: {final_graph_path}")
            with open(final_graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = nx.node_link_graph(data, directed=True, edges="links")
            logging.info(f"✅ 成功加载最终图谱：{graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            logging.error(f"❌ 未找到最终图谱文件: {final_graph_path}")
            logging.error("   请先运行 COMMUNITY 模式生成最终图谱")
            return

        # 步骤2: 从图中重建communities_list（使用community_levels字段）
        logging.info("\n步骤2: 从图中重建社区结构...")
        communities_by_id = {}  # {community_id: community_info}

        for node_id, node_data in graph.nodes(data=True):
            community_levels = node_data.get('community_levels', {})
            if not community_levels:
                # 兼容旧格式
                if 'community' in node_data:
                    comm_id = str(node_data['community'])
                    level = node_data.get('community_level', 0)
                    if comm_id not in communities_by_id:
                        communities_by_id[comm_id] = {
                            'community_id': comm_id,
                            'level': level,
                            'node_ids': [],
                            'children_ids': node_data.get('community_children', []),
                            'parent_id': node_data.get('community_parent'),
                            'title': ''
                        }
                    communities_by_id[comm_id]['node_ids'].append(node_id)
            else:
                # 新格式：使用 community_levels 字段
                max_level = max(int(k.replace('level_', '')) for k in community_levels.keys())
                for level_key, comm_id in community_levels.items():
                    try:
                        level = int(level_key.replace('level_', ''))
                    except ValueError:
                        continue

                    comm_id_str = str(comm_id)
                    if comm_id_str not in communities_by_id:
                        parent_id = None
                        if level > 0:
                            parent_key = f"level_{level - 1}"
                            parent_id = community_levels.get(parent_key)

                        communities_by_id[comm_id_str] = {
                            'community_id': comm_id_str,
                            'level': level,
                            'node_ids': [],
                            'children_ids': [],
                            'parent_id': parent_id,
                            'title': ''
                        }

                    # 只在最底层添加节点ID
                    if level == max_level:
                        if node_id not in communities_by_id[comm_id_str]['node_ids']:
                            communities_by_id[comm_id_str]['node_ids'].append(node_id)

        # 构建父子关系
        for comm_id, comm_info in communities_by_id.items():
            parent_id = comm_info.get('parent_id')
            if parent_id and str(parent_id) in communities_by_id:
                parent_comm = communities_by_id[str(parent_id)]
                if comm_id not in parent_comm['children_ids']:
                    parent_comm['children_ids'].append(comm_id)

        communities_list = list(communities_by_id.values())

        # 按层级分组
        communities_by_level = {}
        for comm in communities_list:
            level = comm['level']
            if level not in communities_by_level:
                communities_by_level[level] = []
            communities_by_level[level].append(comm)

        max_level = max(communities_by_level.keys()) if communities_by_level else 0
        logging.info(f"✅ 从图中重建了 {len(communities_list)} 个社区")
        logging.info(f"   层级范围: 0 到 {max_level}")
        for level in sorted(communities_by_level.keys()):
            logging.info(f"   Level {level}: {len(communities_by_level[level])} 个社区")

        # 步骤3: 验证作业ID配置
        logging.info("\n步骤3: 验证各层级作业ID配置...")
        valid_levels = []
        for level, job_ids in sorted(COMMUNITY_SUMMARY_JOB_IDS_BY_LEVEL.items(), reverse=True):
            normalized_ids = normalize_job_id(job_ids)
            if normalized_ids:
                valid_levels.append((level, normalized_ids))
                logging.info(f"   Level {level}: {len(normalized_ids)} 个作业ID")
            else:
                logging.warning(f"   Level {level}: 未配置有效的作业ID")

        if not valid_levels:
            logging.error("❌ 未配置任何有效的作业ID，请设置 COMMUNITY_SUMMARY_JOB_IDS_BY_LEVEL")
            return

        # 步骤4: 为每个层级创建请求文件、下载结果、保存检查点
        logging.info("\n步骤4: 逐层处理社区报告...")

        # 导入Template
        from string import Template

        # 获取配置参数
        max_report_words = str(config["graph_builder"].get("community_summary_max_report_words", 800))
        max_entities = int(config["graph_builder"].get("community_summary_used_entities_num", 25))
        max_relationships = int(config["graph_builder"].get("community_summary_used_relationships_num", 50))

        all_summaries = {}
        level_summaries = {}

        # 按层级从高到低处理（自底向上）
        for level, job_ids in sorted(valid_levels, key=lambda x: x[0], reverse=True):
            logging.info(f"\n{'='*60}")
            logging.info(f"🔄 处理 Level {level}")
            logging.info(f"{'='*60}")

            level_communities = communities_by_level.get(level, [])
            if not level_communities:
                logging.warning(f"   ⚠️ Level {level} 没有社区，跳过")
                continue

            # 分离叶子社区和非叶子社区
            leaf_communities = [c for c in level_communities if not c['children_ids']]
            non_leaf_communities = [c for c in level_communities if c['children_ids']]

            logging.info(f"   叶子社区: {len(leaf_communities)} 个")
            logging.info(f"   中间层社区: {len(non_leaf_communities)} 个")

            # 4.1 创建该层级的请求文件（level{n}.jsonl）
            level_temp_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}.jsonl"
            level_id_maps = {}

            # 加载prompt模板
            community_prompt = Template(gb.load_prompt(prompt_dir, "community_summary.md"))
            hierarchical_prompt = Template(gb.load_prompt(prompt_dir, "hierarchical_community_summary.md"))

            with open(level_temp_path, 'w', encoding='utf-8') as f:
                # 处理叶子社区（基于节点上下文）
                for comm in leaf_communities:
                    comm_id = comm['community_id']
                    members = comm['node_ids']

                    # 构建社区上下文
                    context, id_map = gb.build_community_context(graph, members, max_entities, max_relationships)
                    level_id_maps[comm_id] = id_map
                    prompt = community_prompt.substitute(max_report_len=max_report_words, context=context)

                    # 写入请求
                    request_line = {
                        "custom_id": comm_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1
                        }
                    }
                    f.write(json.dumps(request_line, ensure_ascii=False) + '\n')

                # 处理非叶子社区（基于子社区报告或节点上下文）
                for comm in non_leaf_communities:
                    comm_id = comm['community_id']
                    children_ids = comm['children_ids']

                    # 收集子社区报告
                    sub_reports = []
                    for child_id in children_ids:
                        if child_id in all_summaries:
                            child_text = all_summaries[child_id]
                            # 解析JSON
                            try:
                                if isinstance(child_text, str):
                                    cleaned = child_text.strip()
                                    if cleaned.startswith("```json"):
                                        cleaned = cleaned[7:]
                                    elif cleaned.startswith("```"):
                                        cleaned = cleaned[3:]
                                    if cleaned.endswith("```"):
                                        cleaned = cleaned[:-3]
                                    child_obj = json.loads(cleaned.strip())
                                else:
                                    child_obj = child_text
                                sub_reports.append({"community_id": child_id, "report": child_obj})
                            except:
                                pass

                    # 构建上下文
                    if sub_reports:
                        # 使用子社区报告作为上下文
                        context_text = ""
                        for sr in sub_reports:
                            report = sr['report']
                            context_text += f"\n## Sub-Community {sr['community_id']}\n"
                            context_text += f"**Title**: {report.get('title', 'N/A')}\n"
                            context_text += f"**Summary**: {report.get('summary', 'N/A')}\n"
                            context_text += f"**Rating**: {report.get('rating', 0)}/10 - {report.get('rating_explanation', 'N/A')}\n"
                            findings = report.get('findings', [])
                            if findings:
                                context_text += f"**Key Findings** ({len(findings)} findings):\n"
                                for f_idx, finding in enumerate(findings[:3], 1):
                                    context_text += f"{f_idx}. {finding.get('summary', 'N/A')}\n"
                            context_text += "\n"
                        prompt = hierarchical_prompt.substitute(
                            max_report_len=max_report_words,
                            sub_community_reports=context_text
                        )
                    else:
                        # 没有子报告，使用节点上下文（叶子社区或投影社区）
                        members = comm['node_ids']
                        context, id_map = gb.build_community_context(graph, members, max_entities, max_relationships)
                        level_id_maps[comm_id] = id_map
                        prompt = community_prompt.substitute(max_report_len=max_report_words, context=context)

                    # 写入请求
                    request_line = {
                        "custom_id": comm_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1
                        }
                    }
                    f.write(json.dumps(request_line, ensure_ascii=False) + '\n')

            logging.info(f"   📝 已创建请求文件: {level_temp_path.name} ({len(level_communities)} 个社区)")

            # 4.2 保存该层级的ID映射
            level_id_map_path = level_temp_path.parent / f"{level_temp_path.stem}_id_maps.json"
            with open(level_id_map_path, 'w', encoding='utf-8') as f:
                json.dump(level_id_maps, f, ensure_ascii=False, indent=2)

            # 详细统计ID映射情况
            total_mappings = sum(len(id_map) for id_map in level_id_maps.values())
            logging.info(f"   📝 已保存ID映射: {level_id_map_path.name}")
            logging.info(f"      - {len(level_id_maps)} 个社区有映射（叶子:{len(leaf_communities)}, 使用节点上下文的非叶子:{len([c for c in non_leaf_communities if c['community_id'] in level_id_maps])}）")
            logging.info(f"      - 总计 {total_mappings} 个ID映射条目")

            # 样本展示
            if level_id_maps:
                sample_comm_id = list(level_id_maps.keys())[0]
                sample_map = level_id_maps[sample_comm_id]
                logging.info(f"      - 映射样本 ({sample_comm_id}): {dict(list(sample_map.items())[:2])}...")

            # 4.3 从云端下载该层级的报告
            logging.info(f"\n   📥 从云端下载 Level {level} 的报告...")
            level_reports = {}

            for idx, job_id in enumerate(job_ids, 1):
                try:
                    job = client.batches.retrieve(batch_id=job_id)
                    logging.info(f"      批次 {idx}/{len(job_ids)} 状态: {job.status}")

                    if job.status == 'completed':
                        batch_results = download_and_process_community_summary_results(client, job)
                        if batch_results:
                            level_reports.update(batch_results)
                            logging.info(f"      ✅ 批次 {idx} 获取了 {len(batch_results)} 个报告")
                        else:
                            logging.warning(f"      ⚠️ 批次 {idx} 未获取到报告")
                    elif job.status == 'in_progress':
                        logging.warning(f"      ⏳ 批次 {idx} 仍在进行中")
                    else:
                        logging.error(f"      ❌ 批次 {idx} 状态异常: {job.status}")
                except Exception as e:
                    logging.error(f"      ❌ 批次 {idx} 下载失败: {e}")

            if level_reports:
                level_summaries[level] = level_reports
                all_summaries.update(level_reports)
                logging.info(f"   ✅ Level {level} 共获取 {len(level_reports)} 个报告")
            else:
                logging.warning(f"   ⚠️ Level {level} 未获取到任何报告")

            # 4.5 保存层级检查点
            level_checkpoint_path = community_requests_path.parent / f"community_summaries_checkpoint_level{level}.json"
            try:
                with open(level_checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "level": level,
                        "total_summaries": len(all_summaries),
                        "summaries": all_summaries,
                        "source": "cloud_download",
                        "job_ids": job_ids
                    }, f, ensure_ascii=False, indent=2)
                logging.info(f"   💾 层级 {level} 检查点已保存: {level_checkpoint_path.name}")
            except Exception as e:
                logging.warning(f"   ⚠️ 保存层级检查点失败: {e}")

        # 步骤5: 合并所有层级的ID映射并统一保存各层级报告文件
        logging.info("\n步骤5: 合并所有层级的ID映射...")
        merged_id_maps = {}
        try:
            merged_id_maps = merge_all_level_id_maps(community_requests_path)
            logging.info(f"   ✅ 成功合并 ID 映射，共 {len(merged_id_maps)} 个社区有映射")

            # 验证：检查哪些社区有ID映射
            communities_with_maps = set(merged_id_maps.keys())
            communities_with_reports = set(all_summaries.keys())

            logging.info(f"\n   🔍 ID映射覆盖率验证:")
            logging.info(f"   - 有报告的社区数: {len(communities_with_reports)}")
            logging.info(f"   - 有ID映射的社区数: {len(communities_with_maps)}")
            logging.info(f"   - 覆盖率: {len(communities_with_maps & communities_with_reports)}/{len(communities_with_reports)} ({len(communities_with_maps & communities_with_reports)/len(communities_with_reports)*100:.1f}%)")

            # 按层级统计ID映射覆盖率
            if communities_list:
                for level in sorted(set(c['level'] for c in communities_list)):
                    level_comms = [c for c in communities_list if c['level'] == level]
                    level_comm_ids = set(c['community_id'] for c in level_comms)
                    level_with_maps = level_comm_ids & communities_with_maps
                    level_with_reports = level_comm_ids & communities_with_reports

                    # 统计该层级的叶子社区和非叶子社区
                    level_leaf = [c for c in level_comms if not c['children_ids']]
                    level_nonleaf = [c for c in level_comms if c['children_ids']]

                    logging.info(f"   Level {level}: {len(level_with_maps)}/{len(level_with_reports)} 有映射 (叶子:{len(level_leaf)}, 非叶子:{len(level_nonleaf)})")

            # 5.1 使用合并后的ID映射生成各层级报告文件，并记录实际使用的映射
            logging.info(f"\n   🔄 使用合并后的ID映射生成各层级报告文件...")
            updated_level_id_maps = {}  # {level: {comm_id: id_map}} - 记录各层级实际使用的映射

            for level in sorted(communities_by_level.keys(), reverse=True):
                level_communities = communities_by_level[level]
                level_reports_path = community_requests_path.parent / f"community_reports_level{level}.jsonl"
                level_actual_maps = {}  # 该层级实际使用的ID映射

                try:
                    level_community_ids = set(c['community_id'] for c in level_communities)
                    level_report_items = {cid: text for cid, text in all_summaries.items() if cid in level_community_ids}

                    # 构建社区元数据映射
                    community_metadata = {}
                    for comm in level_communities:
                        community_metadata[comm['community_id']] = {
                            'level': comm['level'],
                            'title': comm.get('title', ''),
                            'parent_id': comm.get('parent_id'),
                            'children_ids': comm.get('children_ids', []),
                            'node_count': len(comm.get('node_ids', []))
                        }

                    with open(level_reports_path, 'w', encoding='utf-8') as f:
                        for cid, text in level_report_items.items():
                            try:
                                if isinstance(text, str):
                                    cleaned = text.strip()
                                    if cleaned.startswith("```json"):
                                        cleaned = cleaned[7:]
                                    elif cleaned.startswith("```"):
                                        cleaned = cleaned[3:]
                                    if cleaned.endswith("```"):
                                        cleaned = cleaned[:-3]
                                    report_obj = json.loads(cleaned.strip())
                                else:
                                    report_obj = text

                                # 构建完整的记录结构
                                record = {"community_id": cid}
                                record["report"] = report_obj

                                # 使用合并后的ID映射（如果该社区有映射）
                                if cid in merged_id_maps:
                                    record["local_id_map"] = merged_id_maps[cid]
                                    # 记录该层级实际使用的映射
                                    level_actual_maps[cid] = merged_id_maps[cid]

                                # 添加层级元数据
                                if cid in community_metadata:
                                    metadata = community_metadata[cid]
                                    record["level"] = metadata['level']
                                    record["title"] = metadata['title']
                                    record["parent_id"] = metadata['parent_id']
                                    record["children_ids"] = metadata['children_ids']
                                    record["node_count"] = metadata['node_count']

                            except:
                                record = {"community_id": cid, "level": level, "report_raw": text}
                                # 即使解析失败，也尝试添加ID映射
                                if cid in merged_id_maps:
                                    record["local_id_map"] = merged_id_maps[cid]
                                    level_actual_maps[cid] = merged_id_maps[cid]

                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # 保存该层级实际使用的映射
                    updated_level_id_maps[level] = level_actual_maps

                    # 统计该层级有ID映射的报告数量
                    reports_with_maps = len(level_actual_maps)
                    logging.info(f"      Level {level}: 生成了 {len(level_report_items)} 个报告，其中 {reports_with_maps} 个包含 local_id_map")

                except Exception as e:
                    logging.warning(f"      ⚠️ Level {level} 报告保存失败: {e}")
                    updated_level_id_maps[level] = {}

            # 5.2 反向更新各层级的detection_level_id_maps文件，使其与reports_level一致
            logging.info(f"\n   🔄 反向更新各层级的 detection_level_id_maps 文件（与 reports_level 保持一致）...")
            for level, level_maps in updated_level_id_maps.items():
                level_id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_id_maps.json"

                try:
                    # 保存为该层级的ID映射文件
                    with open(level_id_map_path, 'w', encoding='utf-8') as f:
                        json.dump(level_maps, f, ensure_ascii=False, indent=2)

                    # 统计映射条目数
                    total_entries = sum(len(id_map) for id_map in level_maps.values())
                    logging.info(f"      Level {level}: 已更新 {level_id_map_path.name} ({len(level_maps)} 个社区, {total_entries} 个ID条目)")

                except Exception as e:
                    logging.warning(f"      ⚠️ Level {level} ID映射文件更新失败: {e}")

        except Exception as e:
            logging.warning(f"   ⚠️ 合并ID映射失败: {e}")
            import traceback
            logging.warning(f"   详细错误: {traceback.format_exc()}")

        # 步骤6: 汇总统计并验证与图谱的一致性
        logging.info("\n步骤6: 汇总下载结果并验证与图谱的一致性...")
        logging.info(f"📊 下载统计:")
        total_reports = 0
        for level in sorted(level_summaries.keys(), reverse=True):
            count = len(level_summaries[level])
            total_reports += count
            logging.info(f"   Level {level}: {count} 个报告")
        logging.info(f"   总计: {total_reports} 个报告")

        if not all_summaries:
            logging.error("❌ 未能从云端下载到任何社区报告")
            return

        # 验证报告与图谱社区的一致性
        graph_community_ids = set(comm['community_id'] for comm in communities_list)
        report_community_ids = set(all_summaries.keys())

        matched_ids = graph_community_ids & report_community_ids
        missing_in_reports = graph_community_ids - report_community_ids
        extra_in_reports = report_community_ids - graph_community_ids

        logging.info(f"\n🔍 社区ID一致性验证:")
        logging.info(f"   图谱中的社区数: {len(graph_community_ids)}")
        logging.info(f"   报告中的社区数: {len(report_community_ids)}")
        logging.info(f"   匹配的社区数: {len(matched_ids)}")

        if missing_in_reports:
            logging.warning(f"   ⚠️ 图谱中有 {len(missing_in_reports)} 个社区在报告中缺失")
            missing_by_level = {}
            for comm in communities_list:
                if comm['community_id'] in missing_in_reports:
                    lvl = comm['level']
                    missing_by_level[lvl] = missing_by_level.get(lvl, 0) + 1
            for lvl in sorted(missing_by_level.keys()):
                logging.warning(f"      Level {lvl}: {missing_by_level[lvl]} 个缺失")

        if extra_in_reports:
            logging.warning(f"   ⚠️ 报告中有 {len(extra_in_reports)} 个社区不在图谱中")

        if not missing_in_reports and not extra_in_reports:
            logging.info(f"   ✅ 报告与图谱社区完全一致！")

        # 步骤7: 保存最终的社区报告
        logging.info("\n步骤7: 保存最终的社区报告...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"

        # 验证ID映射文件是否存在
        if not id_map_path.exists():
            logging.warning(f"   ⚠️ ID映射文件不存在: {id_map_path}")
            logging.warning(f"   将尝试直接使用合并后的映射数据")
            # 如果合并映射文件不存在，尝试手动保存
            if merged_id_maps:
                try:
                    with open(id_map_path, 'w', encoding='utf-8') as f:
                        json.dump(merged_id_maps, f, ensure_ascii=False, indent=2)
                    logging.info(f"   ✅ 已保存ID映射文件: {id_map_path.name}")
                except Exception as e:
                    logging.error(f"   ❌ 保存ID映射文件失败: {e}")

        gb.save_community_reports(all_summaries, reports_path, id_map_path, communities_list)
        logging.info(f"✅ 社区报告已保存到: {reports_path}")

        # 验证最终报告中的ID映射情况
        logging.info(f"\n   🔍 验证最终报告中的 local_id_map 情况...")
        try:
            with open(reports_path, 'r', encoding='utf-8') as f:
                report_lines = f.readlines()

            total_reports = len(report_lines)
            reports_with_maps = 0
            reports_without_maps = 0

            for line in report_lines:
                try:
                    record = json.loads(line)
                    if 'local_id_map' in record and record['local_id_map']:
                        reports_with_maps += 1
                    else:
                        reports_without_maps += 1
                except:
                    pass

            logging.info(f"   - 总报告数: {total_reports}")
            logging.info(f"   - 有 local_id_map: {reports_with_maps}")
            logging.info(f"   - 无 local_id_map: {reports_without_maps}")
            logging.info(f"   - 覆盖率: {reports_with_maps}/{total_reports} ({reports_with_maps/total_reports*100:.1f}%)")

            # 按层级统计
            if communities_list:
                for level in sorted(set(c['level'] for c in communities_list)):
                    level_comms = [c for c in communities_list if c['level'] == level]
                    level_leaf = [c for c in level_comms if not c['children_ids']]
                    level_nonleaf = [c for c in level_comms if c['children_ids']]

                    logging.info(f"   Level {level}: 叶子社区 {len(level_leaf)} 个（应有ID映射）, 非叶子社区 {len(level_nonleaf)} 个（使用子报告者无需ID映射）")

        except Exception as e:
            logging.warning(f"   ⚠️ 验证最终报告失败: {e}")

        logging.info(f"\n🎉 所有社区报告下载完成！")
        logging.info(f"   生成的文件:")
        logging.info(f"   - 社区报告: {reports_path}")
        logging.info(f"   - 各层级检查点: community_summaries_checkpoint_level*.json")
        logging.info(f"   - 各层级报告: community_reports_level*.jsonl")
        logging.info(f"   - 各层级请求文件: {community_requests_path.stem}_level*.jsonl")
        logging.info(f"   - ID映射文件: *_id_maps.json")

    # ========== 模式7: 重新提交合并请求文件 ==========
    elif START_MODE == "RESUBMIT_MERGE":
        logging.info("\n🚀 模式: 重新提交已有的合并请求文件")
        logging.info("执行流程: 加载消歧图 → 重提交合并请求 → 应用合并 → 社区发现 → 社区摘要")

        # 步骤1: 加载消歧图
        disamb_graph_path = PROJECT_ROOT / config["graph_builder"]["disambiguation_graph_path"]
        if not disamb_graph_path.exists():
            logging.error(f"❌ 消歧图不存在: {disamb_graph_path}")
            return
        with open(disamb_graph_path, 'r', encoding='utf-8') as f:
            graph = nx.node_link_graph(json.load(f), directed=True)
        logging.info(f"✅ 加载消歧图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")

        # 步骤2: 查找并提交合并请求文件
        cache_dir = PROJECT_ROOT / "data" / "cache"
        batch_files = sorted(cache_dir.glob("entities_merge_requests_batch_*.jsonl"))
        if not batch_files:
            logging.error("❌ 未找到合并请求文件 entities_merge_requests_batch_*.jsonl")
            return
        logging.info(f"找到 {len(batch_files)} 个合并请求文件")

        submitted_jobs = []
        for idx, bf in enumerate(batch_files):
            logging.info(f"📤 上传批次 {idx + 1}/{len(batch_files)}: {bf.name}")
            try:
                with open(bf, "rb") as fobj:
                    uploaded = client.files.create(file=fobj, purpose="batch")
                logging.info(f"   文件上传成功: {uploaded.id}")

                job = client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"entity-merge-resubmit-{bf.stem}"}
                )
                logging.info(f"   ✅ 作业已创建: {job.id}")
                submitted_jobs.append((idx, job, bf.name))
            except Exception as e:
                logging.error(f"   ❌ 提交失败: {e}")

        if not submitted_jobs:
            logging.error("❌ 没有成功提交的作业")
            return

        # 步骤3: 监控所有作业
        logging.info(f"⏳ 监控 {len(submitted_jobs)} 个作业...")
        completed_jobs = gb._monitor_multiple_jobs_completion(
            client, submitted_jobs, sleep_interval, job_type="EntityMerge-Resubmit"
        )

        # 步骤4: 下载并合并结果
        all_merge_texts = {}
        for batch_idx, completed_job, fname in sorted(completed_jobs, key=lambda x: x[0]):
            batch_results = gb.process_results(completed_job, client)
            all_merge_texts.update(batch_results)
            logging.info(f"📊 批次 {batch_idx + 1} ({fname}) 返回 {len(batch_results)} 个结果")

        logging.info(f"🎉 合计获得 {len(all_merge_texts)} 个合并仲裁结果")

        if not all_merge_texts:
            logging.error("❌ 没有获得任何合并结果，终止")
            return

        # 步骤5: 解析并应用合并
        groups = gb.parse_entity_merge_results(all_merge_texts)
        logging.info(f"LLM 确认的分组数量: {len(groups)}")

        alias2canon, canon_name_map = gb.build_merge_map(graph, groups)
        if alias2canon:
            logging.info(f"应用实体合并: {len(alias2canon)} 个别名 → {len(canon_name_map)} 个规范名")
            graph = gb.apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='max')
        else:
            logging.warning("⚠️ 未找到需要合并的实体")

        merged_graph_path = PROJECT_ROOT / config["graph_builder"]["merged_graph_path"]
        gb.save_graph(graph, merged_graph_path)
        logging.info(f"✅ 合并图已保存: {merged_graph_path}")

        # 步骤6: 社区发现
        logging.info("\n步骤6: 执行社区发现...")
        graph, communities_list = gb.detect_communities(
            graph=graph,
            weight_alpha=weight_alpha,
            use_hierarchical=use_hierarchical,
            max_level=max_level,
            min_community_size=min_community_size
        )

        # 步骤7: 社区摘要
        logging.info("\n步骤7: 生成社区摘要...")
        summaries = gb.run_community_summaries(
            client=client,
            graph=graph,
            model_name=model_name,
            prompt_dir=prompt_dir,
            config=config,
            sleep_interval=sleep_interval,
            community_requests_path=community_requests_path,
            communities_list=communities_list
        )

        # 步骤8: 保存最终结果
        logging.info("\n步骤8: 保存最终结果...")
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"
        if summaries:
            gb.save_community_reports(summaries, reports_path, id_map_path, communities_list)
        gb.save_graph(graph, final_graph_path)

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