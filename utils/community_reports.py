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
社区报告生成模块

实现 GraphRAG 的自底向上社区报告生成策略：
1. 叶子社区：基于节点、边、声明生成详细报告
2. 上层社区：基于子社区报告生成汇总报告
3. 支持 Token 管理和替换机制
"""

import json
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from string import Template
from collections import defaultdict
import networkx as nx
from openai import OpenAI
import yaml


def _parse_report_json(text: Any, community_id: str = "") -> Optional[Dict[str, Any]]:
    """
    健壮地解析社区报告 JSON，处理各种常见的格式问题

    Args:
        text: 报告文本（可以是字符串或字典）
        community_id: 社区ID（用于日志）

    Returns:
        解析后的字典，如果解析失败返回 None
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        return None

    # 1. 清理 markdown 代码块标记
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # 2. 尝试直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 尝试修复常见的 JSON 格式问题
    fixed = cleaned

    # 3.1 修复单引号 -> 双引号（但保留字符串内容中的单引号）
    # 这是一个简化的修复，可能不适用于所有情况
    try:
        # 尝试使用 ast.literal_eval 处理 Python 字典格式
        import ast
        obj = ast.literal_eval(fixed)
        if isinstance(obj, dict):
            return obj
    except:
        pass

    # 3.2 移除尾随逗号（JSON 不允许）
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 3.3 尝试修复缺少引号的键
    # 匹配 { key: 或 , key: 的模式，其中 key 没有引号
    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 3.4 尝试只提取 JSON 对象部分
    # 有时候 LLM 会在 JSON 前后添加额外的文本
    json_match = re.search(r'\{[\s\S]*\}', fixed)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 3.5 尝试更激进的修复：替换所有单引号
    # 注意：这可能会破坏包含单引号的字符串值
    aggressive_fix = fixed.replace("'", '"')
    try:
        return json.loads(aggressive_fix)
    except json.JSONDecodeError as e:
        logging.warning(f"      ⚠️ 无法解析子社区 {community_id} 的报告: JSON解析错误 - {e}")
        logging.debug(f"         原始文本前200字符: {text[:200] if len(text) > 200 else text}...")
        return None


def load_level_checkpoint(community_requests_path: Path, target_level: int) -> Tuple[Dict[str, str], int]:
    """
    从检查点文件加载已完成的层级报告

    Args:
        community_requests_path: 社区请求文件路径（用于定位检查点目录）
        target_level: 目标恢复层级（从这个层级开始继续）

    Returns:
        Tuple[已加载的报告字典, 实际恢复的层级]
    """
    checkpoint_dir = community_requests_path.parent
    all_summaries = {}
    actual_resume_level = None

    # 从最高层级开始，寻找最近的有效检查点
    for level in range(10, target_level - 1, -1):  # 假设最高10层
        checkpoint_path = checkpoint_dir / f"community_summaries_checkpoint_level{level}.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_summaries = data.get("summaries", {})
                actual_resume_level = data.get("level")
                logging.info(f"✅ 已加载层级 {actual_resume_level} 的检查点: {len(all_summaries)} 个报告")
                break
            except Exception as e:
                logging.warning(f"⚠️ 加载检查点 level{level} 失败: {e}")
                continue

    return all_summaries, actual_resume_level


def merge_all_level_id_maps(community_requests_path: Path, output_path: Path = None) -> Dict[str, Dict[str, str]]:
    """
    合并所有层级的 ID 映射文件为统一的映射文件

    在多层社区结构中，每个层级的 ID 映射会单独保存为:
    - community_detection_level{N}_id_maps.json
    - community_detection_level{N}_id_maps.json

    此函数将它们合并为一个统一的文件，供 embedding 阶段使用。

    正确的格式: {community_id: {local_id: global_id}}
    例如: {"comm_1": {"E1": "chunk-xxx-e-1", "R1": "chunk-xxx-r-1"}}

    Args:
        community_requests_path: 社区请求文件路径（用于定位映射文件目录）
        output_path: 输出文件路径（默认为 community_detection_id_maps.json）

    Returns:
        合并后的 ID 映射字典 {community_id: {local_id: global_id}}
    """
    maps_dir = community_requests_path.parent
    all_id_maps: Dict[str, Dict[str, str]] = {}

    # 查找所有层级的 ID 映射文件（只匹配 *_level{n}_id_maps.json 格式）
    # 排除合并文件本身（community_detection_id_maps.json）避免循环包含
    id_map_files = list(maps_dir.glob("*_level*_id_maps.json"))

    if not id_map_files:
        logging.warning("未找到任何层级 ID 映射文件")
        return all_id_maps

    logging.info(f"🔄 正在合并 {len(id_map_files)} 个层级 ID 映射文件...")

    def is_local_id(key: str) -> bool:
        """检查是否是本地ID格式 (E1, E2, R1, R2, ...)"""
        if not key:
            return False
        return bool(key[0] in ('E', 'R') and key[1:].isdigit())

    def is_global_id(key: str) -> bool:
        """检查是否是全局ID格式 (chunk-xxx-e-1, ...)"""
        if not key:
            return False
        return key.startswith('chunk-') or '-e-' in key or '-r-' in key

    def normalize_id_map(id_map: Dict[str, str]) -> Dict[str, str]:
        """
        标准化 ID 映射，确保格式为 {local_id: global_id}

        处理两种可能的错误格式:
        1. {global_id: local_id} - 键值颠倒
        2. 混合格式
        """
        if not id_map:
            return {}

        normalized = {}
        inverted_count = 0
        normal_count = 0

        for k, v in id_map.items():
            k_is_local = is_local_id(k)
            v_is_local = is_local_id(v)
            k_is_global = is_global_id(k)
            v_is_global = is_global_id(v)

            if k_is_local and v_is_global:
                # 正确格式: E1 -> chunk-xxx
                normalized[k] = v
                normal_count += 1
            elif k_is_global and v_is_local:
                # 颠倒格式: chunk-xxx -> E1，需要反转
                normalized[v] = k
                inverted_count += 1
            elif k_is_local:
                # k 是本地ID，v 格式不明确，假设是正确的
                normalized[k] = v
                normal_count += 1
            elif v_is_local:
                # v 是本地ID，k 可能是全局ID，需要反转
                normalized[v] = k
                inverted_count += 1
            else:
                # 无法判断，保持原样
                normalized[k] = v

        if inverted_count > 0:
            logging.debug(f"      修正了 {inverted_count} 个颠倒的映射条目")

        return normalized

    for map_file in id_map_files:
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                level_maps = json.load(f)

            if isinstance(level_maps, dict):
                # 合并到总映射中
                for comm_id, id_map in level_maps.items():
                    if isinstance(id_map, dict):
                        # 标准化 ID 映射格式
                        normalized_map = normalize_id_map(id_map)

                        if comm_id not in all_id_maps:
                            all_id_maps[comm_id] = {}
                        all_id_maps[comm_id].update(normalized_map)

                logging.debug(f"   已合并 {map_file.name}: {len(level_maps)} 个社区映射")
        except Exception as e:
            logging.warning(f"   ⚠️ 加载 {map_file.name} 失败: {e}")

    # 保存合并后的映射
    if output_path is None:
        output_path = community_requests_path.parent / f"{community_requests_path.stem}_id_maps.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_id_maps, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ 已合并 {len(all_id_maps)} 个社区的 ID 映射到: {output_path.name}")
    except Exception as e:
        logging.error(f"❌ 保存合并后的 ID 映射失败: {e}")

    return all_id_maps


def reverse_sync_level_reports_with_id_maps(
    community_requests_path: Path,
    communities_by_level: Dict[int, List[Dict[str, Any]]],
    all_summaries: Dict[str, str],
    merged_id_maps: Dict[str, Dict[str, str]]
) -> None:
    """
    反向同步：使用合并后的ID映射更新各层级报告文件，并更新各层级的ID映射文件

    Args:
        community_requests_path: 社区请求文件路径
        communities_by_level: 按层级分组的社区列表
        all_summaries: 所有社区的报告
        merged_id_maps: 合并后的ID映射
    """
    if not merged_id_maps:
        return

    logging.info(f"\n   🔄 使用合并的ID映射更新各层级报告文件...")

    for level in sorted(communities_by_level.keys(), reverse=True):
        level_communities = communities_by_level[level]
        level_reports_path = community_requests_path.parent / f"community_reports_level{level}.jsonl"
        level_actual_maps = {}  # 该层级实际使用的ID映射

        try:
            level_community_ids = set(c['community_id'] for c in level_communities)
            level_reports = {cid: text for cid, text in all_summaries.items() if cid in level_community_ids}

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
                for cid, text in level_reports.items():
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

                        record = {"community_id": cid}
                        record["report"] = report_obj

                        # 使用合并后的ID映射
                        if cid in merged_id_maps:
                            record["local_id_map"] = merged_id_maps[cid]
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
                        if cid in merged_id_maps:
                            record["local_id_map"] = merged_id_maps[cid]
                            level_actual_maps[cid] = merged_id_maps[cid]

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(f"      Level {level}: 已更新 {len(level_reports)} 个报告，其中 {len(level_actual_maps)} 个包含 local_id_map")

            # 反向保存该层级的ID映射
            level_id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_id_maps.json"
            with open(level_id_map_path, 'w', encoding='utf-8') as f:
                json.dump(level_actual_maps, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logging.warning(f"      ⚠️ Level {level} 报告更新失败: {e}")


def run_hierarchical_community_summaries_with_resume(
    client: OpenAI,
    graph: nx.DiGraph,
    model_name: str,
    prompt_dir: str,
    config: Dict[str, Any],
    sleep_interval: int,
    community_requests_path: Path,
    communities_list: List[Dict[str, Any]],
    max_report_words: str,
    max_entities: int,
    max_relationships: int,
    load_prompt_func,
    build_context_func,
    submit_job_func,
    process_results_func,
    resume_from_level: int = None,
    preloaded_summaries: Dict[str, str] = None
) -> Dict[str, str]:
    """
    生成分层社区报告，支持从指定层级恢复

    Args:
        resume_from_level: 从哪个层级开始恢复（None表示从头开始）
        preloaded_summaries: 预加载的报告字典（用于恢复时）
        其他参数与run_hierarchical_community_summaries相同

    Returns:
        社区ID到报告的映射
    """
    logging.info(f"📊 开始生成分层社区报告（共 {len(communities_list)} 个社区）")

    # 1. 按层级分组
    communities_by_level = defaultdict(list)
    for comm in communities_list:
        communities_by_level[comm['level']].append(comm)

    max_level = max(communities_by_level.keys())
    logging.info(f"   层级范围: 0 到 {max_level}")
    for level in sorted(communities_by_level.keys()):
        logging.info(f"   Level {level}: {len(communities_by_level[level])} 个社区")

    # 2. 初始化报告字典
    if preloaded_summaries:
        all_summaries = dict(preloaded_summaries)
        logging.info(f"   📥 已预加载 {len(all_summaries)} 个报告")
    else:
        all_summaries = {}

    # 3. 确定起始层级
    if resume_from_level is not None:
        start_level = resume_from_level
        logging.info(f"\n🔄 从 Level {start_level} 恢复，跳过更深层级")
    else:
        start_level = max_level
        logging.info(f"\n🔄 采用自底向上策略：从 Level {max_level} → Level 0")

    # 4. 自底向上生成报告
    for current_level in range(start_level, -1, -1):
        level_communities = communities_by_level[current_level]
        logging.info(f"\n{'='*60}")
        logging.info(f"🔄 处理 Level {current_level} ({len(level_communities)} 个社区)")
        logging.info(f"{'='*60}")

        leaf_communities = [c for c in level_communities if not c['children_ids']]
        non_leaf_communities = [c for c in level_communities if c['children_ids']]

        # 处理叶子社区
        if leaf_communities:
            # 检查是否已有这些社区的报告
            missing_leaf = [c for c in leaf_communities if c['community_id'] not in all_summaries]
            if missing_leaf:
                logging.info(f"   📝 生成 {len(missing_leaf)} 个叶子社区的报告（跳过已有的 {len(leaf_communities) - len(missing_leaf)} 个）")
                leaf_summaries = generate_leaf_community_summaries(
                    client=client,
                    graph=graph,
                    communities=missing_leaf,
                    model_name=model_name,
                    prompt_dir=prompt_dir,
                    sleep_interval=sleep_interval,
                    community_requests_path=community_requests_path,
                    max_report_words=max_report_words,
                    max_entities=max_entities,
                    max_relationships=max_relationships,
                    level=current_level,
                    load_prompt_func=load_prompt_func,
                    build_context_func=build_context_func,
                    submit_job_func=submit_job_func,
                    process_results_func=process_results_func
                )
                all_summaries.update(leaf_summaries)
            else:
                logging.info(f"   ✅ 所有 {len(leaf_communities)} 个叶子社区已有报告，跳过")

        # 处理非叶子社区
        if non_leaf_communities:
            missing_parent = [c for c in non_leaf_communities if c['community_id'] not in all_summaries]
            if missing_parent:
                logging.info(f"   🌳 生成 {len(missing_parent)} 个中间层社区的报告（跳过已有的 {len(non_leaf_communities) - len(missing_parent)} 个）")
                parent_summaries = generate_parent_community_summaries(
                    client=client,
                    graph=graph,
                    communities=missing_parent,
                    child_summaries=all_summaries,
                    model_name=model_name,
                    prompt_dir=prompt_dir,
                    sleep_interval=sleep_interval,
                    community_requests_path=community_requests_path,
                    max_report_words=max_report_words,
                    max_entities=max_entities,
                    max_relationships=max_relationships,
                    level=current_level,
                    load_prompt_func=load_prompt_func,
                    build_context_func=build_context_func,
                    submit_job_func=submit_job_func,
                    process_results_func=process_results_func
                )
                all_summaries.update(parent_summaries)
            else:
                logging.info(f"   ✅ 所有 {len(non_leaf_communities)} 个中间层社区已有报告，跳过")

        # 保存检查点
        level_checkpoint_path = community_requests_path.parent / f"community_summaries_checkpoint_level{current_level}.json"
        level_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(level_checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "level": current_level,
                    "total_summaries": len(all_summaries),
                    "summaries": all_summaries
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"   💾 层级 {current_level} 检查点已保存: {level_checkpoint_path.name}")
        except Exception as e:
            logging.warning(f"   ⚠️ 保存层级检查点失败: {e}")

        # 保存当前层级的报告到单独文件（JSONL格式）
        level_reports_path = community_requests_path.parent / f"community_reports_level{current_level}.jsonl"
        level_reports_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # 筛选出当前层级的社区报告
            level_community_ids = set(c['community_id'] for c in level_communities)
            level_reports = {cid: text for cid, text in all_summaries.items() if cid in level_community_ids}

            with open(level_reports_path, 'w', encoding='utf-8') as f:
                for cid, text in level_reports.items():
                    # 尝试解析JSON
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
                        record = {"community_id": cid, "level": current_level, "report": report_obj}
                    except:
                        record = {"community_id": cid, "level": current_level, "report_raw": text}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(f"   📄 层级 {current_level} 报告已保存: {level_reports_path.name} ({len(level_reports)} 条)")
        except Exception as e:
            logging.warning(f"   ⚠️ 保存层级报告失败: {e}")

    # 合并所有层级的 ID 映射文件（第一次，基于原始层级文件）
    logging.info(f"\n   🔄 合并所有层级的ID映射...")
    merged_id_maps = merge_all_level_id_maps(community_requests_path)

    # 反向同步：使用合并后的ID映射更新各层级报告文件和ID映射文件
    reverse_sync_level_reports_with_id_maps(
        community_requests_path=community_requests_path,
        communities_by_level=communities_by_level,
        all_summaries=all_summaries,
        merged_id_maps=merged_id_maps
    )
    
    # 重新合并ID映射（第二次，基于反向同步后的层级文件，确保完整性）
    logging.info(f"\n   🔄 重新合并ID映射文件（基于反向同步后的层级文件）...")
    final_merged_maps = merge_all_level_id_maps(community_requests_path)
    logging.info(f"   ✅ 最终合并文件包含 {len(final_merged_maps)} 个社区的ID映射")

    logging.info(f"\n✅ 分层社区报告生成完成！共生成 {len(all_summaries)} 个报告")
    return all_summaries


def _monitor_multiple_jobs(client: OpenAI, submitted_jobs: List[Tuple], sleep_interval: int, job_type: str) -> List[Tuple]:
    """并行监控多个批次作业"""
    import time
    completed_states = {'completed', 'failed', 'cancelled', 'expired'}
    pending_jobs = list(submitted_jobs)
    completed_jobs = []
    
    while pending_jobs:
        still_pending = []
        for batch_idx, job, extra in pending_jobs:
            try:
                status = client.batches.retrieve(batch_id=job.id)
                current_state = status.status
                if current_state in completed_states:
                    logging.info(f"      ✅ [{job_type}] 批次 {batch_idx} 作业结束，状态: {current_state}")
                    completed_jobs.append((batch_idx, status, extra))
                else:
                    still_pending.append((batch_idx, job, extra))
            except Exception as e:
                logging.error(f"      ❌ [{job_type}] 批次 {batch_idx} 状态查询失败: {e}")
                still_pending.append((batch_idx, job, extra))
        
        pending_jobs = still_pending
        if pending_jobs:
            logging.info(f"      ⏳ [{job_type}] 仍有 {len(pending_jobs)} 个批次作业在进行中，等待 {sleep_interval} 秒...")
            time.sleep(sleep_interval)
            
    return completed_jobs


def generate_leaf_community_summaries(
    client: OpenAI,
    graph: nx.DiGraph,
    communities: List[Dict[str, Any]],
    model_name: str,
    prompt_dir: str,
    sleep_interval: int,
    community_requests_path: Path,
    max_report_words: str,
    max_entities: int,
    max_relationships: int,
    level: int,
    load_prompt_func,
    build_context_func,
    submit_job_func,
    process_results_func
) -> Dict[str, str]:
    """
    生成叶子社区的报告（基于节点和边）

    Args:
        client: OpenAI 客户端
        graph: 知识图谱
        communities: 叶子社区列表
        model_name: 模型名称
        prompt_dir: Prompt 目录
        sleep_interval: 轮询间隔
        community_requests_path: 请求文件路径
        max_report_words: 最大报告字数
        max_entities: 最大实体数
        max_relationships: 最大关系数
        level: 当前层级
        load_prompt_func: 加载 prompt 的函数
        build_context_func: 构建上下文的函数
        submit_job_func: 提交作业的函数
        process_results_func: 处理结果的函数

    Returns:
        社区ID到报告的映射
    """
    MAX_BATCH_SIZE = 45000
    all_summaries = {}
    all_id_maps: Dict[str, Dict[str, str]] = {}
    
    total_communities = len(communities)
    num_batches = (total_communities + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

    logging.info(f"      📦 社区总数 {total_communities}，将拆分为 {num_batches} 个批次处理（每批最多 {MAX_BATCH_SIZE}）")

    single_full_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}.jsonl"
    id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_id_maps.json"
    skip_build_context = False

    if single_full_path.exists() and single_full_path.stat().st_size > 1024 * 1024:
        batches_exist = True
        for b_idx in range(num_batches):
            b_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_batch{b_idx}.jsonl"
            if not b_path.exists():
                batches_exist = False
                break
                
        if id_map_path.exists() and batches_exist:
            logging.info("      ✅ 发现已生成的批次请求与 ID Maps，跳过上下文构建。")
            with open(id_map_path, 'r', encoding='utf-8') as f:
                all_id_maps = json.load(f)
            skip_build_context = True
        else:
            logging.info(f"      🔍 发现上次生成的完整请求文件: {single_full_path.name}，正在解析拆分并提取 ID Maps...")
            import re
            batch_files = []
            for b_idx in range(num_batches):
                b_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_batch{b_idx}.jsonl"
                b_path.parent.mkdir(parents=True, exist_ok=True)
                batch_files.append(open(b_path, 'w', encoding='utf-8'))
                
            with open(single_full_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    b_idx = line_idx // MAX_BATCH_SIZE
                    if b_idx < len(batch_files):
                        batch_files[b_idx].write(line)
                        
                    try:
                        obj = json.loads(line)
                        c_id = obj.get("custom_id")
                        content = obj.get("body", {}).get("messages", [{}])[0].get("content", "")
                        
                        entities_match = re.search(r'Entities:\n(.*?)(?:\n\nRelationships:|$)', content, re.DOTALL)
                        id_map = {}
                        if entities_match:
                            for e_line in entities_match.group(1).split('\n'):
                                e_line = e_line.strip()
                                if e_line.startswith('- '):
                                    m = re.match(r'- (.*?) \((E\d+)\)', e_line)
                                    if m:
                                        id_map[m.group(1).strip()] = m.group(2)
                        all_id_maps[c_id] = id_map
                    except Exception:
                        pass
                        
            for bf in batch_files:
                bf.close()
                
            id_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(id_map_path, 'w', encoding='utf-8') as f:
                json.dump(all_id_maps, f, ensure_ascii=False, indent=2)
            logging.info("      ✅ 大文件拆分与 ID Maps 恢复完成！")
            skip_build_context = True

    submitted_jobs = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * MAX_BATCH_SIZE
        end_idx = min((batch_idx + 1) * MAX_BATCH_SIZE, total_communities)
        batch_communities = communities[start_idx:end_idx]
        
        # 创建临时请求文件
        temp_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_batch{batch_idx}.jsonl"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not skip_build_context:
            # 准备请求
            community_prompt = Template(load_prompt_func(prompt_dir, "community_summary.md"))
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                for comm in batch_communities:
                    comm_id = comm['community_id']
                    members = comm['node_ids']

                    # 构建社区上下文
                    context, id_map = build_context_func(graph, members, max_entities, max_relationships)
                    prompt = community_prompt.substitute(max_report_len=max_report_words, context=context)
                    all_id_maps[comm_id] = id_map

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

        logging.info(f"      ✅ 已准备好批次 {batch_idx+1}/{num_batches} ({len(batch_communities)} 个社区) 的请求: {temp_path.name}")

        # 提交批量作业
        job_name = f"L{level}Leaf_B{batch_idx}" if num_batches > 1 else f"Level{level}LeafSummary"
        try:
            job = submit_job_func(client, temp_path, model_name, sleep_interval, job_name, monitor=False)
            if job:
                submitted_jobs.append((batch_idx, job, None))
        except TypeError:
            logging.warning("      ⚠️ submit_job_func 不支持 monitor=False 参数，退化为串行提交")
            job = submit_job_func(client, temp_path, model_name, sleep_interval, job_name)
            batch_summaries = process_results_func(job, client)
            if batch_summaries:
                all_summaries.update(batch_summaries)

    if submitted_jobs:
        logging.info(f"      ⏳ 开始并行轮询 {len(submitted_jobs)} 个叶子社区作业...")
        completed_jobs = _monitor_multiple_jobs(client, submitted_jobs, sleep_interval, f"Level{level}Leaf")
        for batch_idx, comp_job, _ in sorted(completed_jobs, key=lambda x: x[0]):
            batch_summaries = process_results_func(comp_job, client)
            if batch_summaries:
                all_summaries.update(batch_summaries)

    # 保存ID映射（所有批次合并保存）
    id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_id_maps.json"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(all_id_maps, f, ensure_ascii=False, indent=2)

    return all_summaries


def generate_parent_community_summaries(
    client: OpenAI,
    graph: nx.DiGraph,
    communities: List[Dict[str, Any]],
    child_summaries: Dict[str, str],
    model_name: str,
    prompt_dir: str,
    sleep_interval: int,
    community_requests_path: Path,
    max_report_words: str,
    max_entities: int,
    max_relationships: int,
    level: int,
    load_prompt_func,
    build_context_func,
    submit_job_func,
    process_results_func
) -> Dict[str, str]:
    """
    生成父社区的报告（基于子社区报告或节点信息）

    采用 GraphRAG 的自底向上策略：
    1. 优先使用子社区报告（如果存在）
    2. 如果没有子社区报告，则使用节点和边的信息（叶子社区或投影社区）
    3. 自适应处理两种上下文来源

    Args:
        client: OpenAI 客户端
        graph: 知识图谱（用于提取节点信息）
        communities: 父社区列表
        child_summaries: 子社区报告字典
        model_name: 模型名称
        prompt_dir: Prompt 目录
        sleep_interval: 轮询间隔
        community_requests_path: 请求文件路径
        max_report_words: 最大报告字数
        max_entities: 最大实体数（用于节点上下文）
        max_relationships: 最大关系数（用于节点上下文）
        level: 当前层级
        load_prompt_func: 加载 prompt 的函数
        build_context_func: 构建节点上下文的函数
        submit_job_func: 提交作业的函数
        process_results_func: 处理结果的函数

    Returns:
        社区ID到报告的映射
    """
    MAX_BATCH_SIZE = 45000
    all_summaries = {}
    
    total_communities = len(communities)
    num_batches = (total_communities + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

    logging.info(f"      📦 父社区总数 {total_communities}，将拆分为 {num_batches} 个批次处理（每批最多 {MAX_BATCH_SIZE}）")

    # 用于保存节点ID映射（当使用节点上下文时）
    all_id_maps: Dict[str, Dict[str, str]] = {}
    
    # 加载层级社区prompt
    hierarchical_prompt = Template(load_prompt_func(prompt_dir, "hierarchical_community_summary.md"))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * MAX_BATCH_SIZE
        end_idx = min((batch_idx + 1) * MAX_BATCH_SIZE, total_communities)
        batch_communities = communities[start_idx:end_idx]
        
        # 创建临时请求文件
        temp_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_batch{batch_idx}.jsonl"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            for comm in batch_communities:
                comm_id = comm['community_id']
                children_ids = comm['children_ids']

                # 尝试收集子社区的报告
                sub_reports = []
                for child_id in children_ids:
                    if child_id in child_summaries:
                        child_text = child_summaries[child_id]
                        # 解析JSON（更健壮的解析逻辑）
                        child_obj = _parse_report_json(child_text, child_id)

                        if child_obj:
                            sub_reports.append({
                                "community_id": child_id,
                                "report": child_obj
                            })

                # 决定使用哪种上下文
                if sub_reports:
                    # 情况1: 有子社区报告，使用报告上下文
                    context_text = ""
                    for idx, sr in enumerate(sub_reports, 1):
                        report = sr['report']
                        context_text += f"\n## Sub-Community {sr['community_id']}\n"
                        context_text += f"**Title**: {report.get('title', 'N/A')}\n"
                        context_text += f"**Summary**: {report.get('summary', 'N/A')}\n"
                        context_text += f"**Rating**: {report.get('rating', 0)}/10 - {report.get('rating_explanation', 'N/A')}\n"

                        findings = report.get('findings', [])
                        if findings:
                            context_text += f"**Key Findings** ({len(findings)} findings):\n"
                            for f_idx, finding in enumerate(findings[:3], 1):  # 限制每个子社区最多3个findings
                                context_text += f"{f_idx}. {finding.get('summary', 'N/A')}\n"
                        context_text += "\n"

                    logging.debug(f"      社区 {comm_id}: 使用 {len(sub_reports)} 个子社区报告作为上下文")

                else:
                    # 情况2: 没有子社区报告，使用节点信息（叶子社区或投影社区）
                    members = comm['node_ids']
                    context_text, id_map = build_context_func(graph, members, max_entities, max_relationships)
                    all_id_maps[comm_id] = id_map

                    logging.debug(f"      社区 {comm_id}: 使用 {len(members)} 个节点作为上下文（叶子/投影社区）")

                # 生成prompt
                prompt = hierarchical_prompt.substitute(
                    max_report_len=max_report_words,
                    sub_community_reports=context_text
                )

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

        logging.info(f"      ✅ 已准备好父社区请求批次 {batch_idx+1}/{num_batches}: {temp_path.name}")

    # 并发提交并监控所有批次
    submitted_jobs = []
    for batch_idx in range(num_batches):
        temp_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_batch{batch_idx}.jsonl"
        job_name = f"L{level}Parent_B{batch_idx}" if num_batches > 1 else f"Level{level}ParentSummary"
        try:
            job = submit_job_func(client, temp_path, model_name, sleep_interval, job_name, monitor=False)
            if job:
                submitted_jobs.append((batch_idx, job, None))
        except TypeError:
            logging.warning("      ⚠️ submit_job_func 不支持 monitor=False 参数，退化为串行提交")
            job = submit_job_func(client, temp_path, model_name, sleep_interval, job_name)
            batch_summaries = process_results_func(job, client)
            if batch_summaries:
                all_summaries.update(batch_summaries)

    if submitted_jobs:
        logging.info(f"      ⏳ 开始并行轮询 {len(submitted_jobs)} 个父社区作业...")
        completed_jobs = _monitor_multiple_jobs(client, submitted_jobs, sleep_interval, f"Level{level}Parent")
        for batch_idx, comp_job, _ in sorted(completed_jobs, key=lambda x: x[0]):
            batch_summaries = process_results_func(comp_job, client)
            if batch_summaries:
                all_summaries.update(batch_summaries)

    # 如果使用了节点上下文，保存ID映射
    if all_id_maps:
        id_map_path = community_requests_path.parent / f"{community_requests_path.stem}_level{level}_id_maps.json"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(all_id_maps, f, ensure_ascii=False, indent=2)

    return all_summaries


def run_hierarchical_community_summaries(
    client: OpenAI,
    graph: nx.DiGraph,
    model_name: str,
    prompt_dir: str,
    config: Dict[str, Any],
    sleep_interval: int,
    community_requests_path: Path,
    communities_list: List[Dict[str, Any]],
    max_report_words: str,
    max_entities: int,
    max_relationships: int,
    load_prompt_func,
    build_context_func,
    submit_job_func,
    process_results_func
) -> Dict[str, str]:
    """
    生成分层社区报告

    采用 GraphRAG 的自底向上策略：
    1. 从最深层级（叶子社区）开始生成报告 - 基于节点、边、声明
    2. 逐层向上，使用下层社区报告生成上层报告
    3. 按照 GraphRAG 的策略：如果子社区报告总长度在 token 限制内，直接汇总；否则排序并选择性替换

    Args:
        client: OpenAI 客户端
        graph: 知识图谱
        model_name: 模型名称
        prompt_dir: Prompt 目录
        config: 配置字典
        sleep_interval: 轮询间隔
        community_requests_path: 请求文件路径
        communities_list: 社区列表（包含层级信息）
        max_report_words: 最大报告字数
        max_entities: 最大实体数
        max_relationships: 最大关系数
        load_prompt_func: 加载 prompt 的函数
        build_context_func: 构建上下文的函数
        submit_job_func: 提交作业的函数
        process_results_func: 处理结果的函数

    Returns:
        社区ID到报告的映射
    """
    logging.info(f"📊 开始生成分层社区报告（共 {len(communities_list)} 个社区）")

    # 1. 按层级分组
    communities_by_level = defaultdict(list)
    for comm in communities_list:
        communities_by_level[comm['level']].append(comm)

    max_level = max(communities_by_level.keys())
    logging.info(f"   层级范围: 0 到 {max_level}")
    for level in sorted(communities_by_level.keys()):
        logging.info(f"   Level {level}: {len(communities_by_level[level])} 个社区")

    # 2. 存储所有社区的报告
    all_summaries: Dict[str, str] = {}

    # 3. 自底向上生成报告：从最深层级（叶子）到 Level 0（根）
    # 这符合 GraphRAG 的标准策略：叶子社区基于节点生成，上层社区基于子社区报告生成
    logging.info(f"\n🔄 采用自底向上策略：从 Level {max_level} → Level 0")

    for current_level in range(max_level, -1, -1):  # 从 max_level 递减到 0
        level_communities = communities_by_level[current_level]
        logging.info(f"\n{'='*60}")
        logging.info(f"🔄 处理 Level {current_level} ({len(level_communities)} 个社区)")
        logging.info(f"{'='*60}")

        # 判断这些社区是否是叶子社区（没有子社区）
        # 注意：由于投影机制，同一层级可能既有真实社区（有子社区）也有投影社区（叶子）
        leaf_communities = [c for c in level_communities if not c['children_ids']]
        non_leaf_communities = [c for c in level_communities if c['children_ids']]

        # 3.1 处理叶子社区（基于节点和边）
        if leaf_communities:
            logging.info(f"   📝 生成 {len(leaf_communities)} 个叶子社区的报告（基于节点和边）")
            leaf_summaries = generate_leaf_community_summaries(
                client=client,
                graph=graph,
                communities=leaf_communities,
                model_name=model_name,
                prompt_dir=prompt_dir,
                sleep_interval=sleep_interval,
                community_requests_path=community_requests_path,
                max_report_words=max_report_words,
                max_entities=max_entities,
                max_relationships=max_relationships,
                level=current_level,
                load_prompt_func=load_prompt_func,
                build_context_func=build_context_func,
                submit_job_func=submit_job_func,
                process_results_func=process_results_func
            )
            all_summaries.update(leaf_summaries)

        # 3.2 处理非叶子社区（基于子社区报告或节点）
        if non_leaf_communities:
            logging.info(f"   🌳 生成 {len(non_leaf_communities)} 个中间层社区的报告（基于子社区报告或节点）")
            parent_summaries = generate_parent_community_summaries(
                client=client,
                graph=graph,
                communities=non_leaf_communities,
                child_summaries=all_summaries,
                model_name=model_name,
                prompt_dir=prompt_dir,
                sleep_interval=sleep_interval,
                community_requests_path=community_requests_path,
                max_report_words=max_report_words,
                max_entities=max_entities,
                max_relationships=max_relationships,
                level=current_level,
                load_prompt_func=load_prompt_func,
                build_context_func=build_context_func,
                submit_job_func=submit_job_func,
                process_results_func=process_results_func
            )
            all_summaries.update(parent_summaries)

        # 每个层级完成后保存中间结果（用于恢复）
        level_checkpoint_path = community_requests_path.parent / f"community_summaries_checkpoint_level{current_level}.json"
        try:
            with open(level_checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "level": current_level,
                    "total_summaries": len(all_summaries),
                    "summaries": all_summaries
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"   💾 层级 {current_level} 检查点已保存: {level_checkpoint_path.name}")
        except Exception as e:
            logging.warning(f"   ⚠️ 保存层级检查点失败: {e}")

        # 保存当前层级的报告到单独文件（JSONL格式）
        level_reports_path = community_requests_path.parent / f"community_reports_level{current_level}.jsonl"
        try:
            # 筛选出当前层级的社区报告
            level_community_ids = set(c['community_id'] for c in level_communities)
            level_reports = {cid: text for cid, text in all_summaries.items() if cid in level_community_ids}

            with open(level_reports_path, 'w', encoding='utf-8') as f:
                for cid, text in level_reports.items():
                    # 尝试解析JSON
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
                        record = {"community_id": cid, "level": current_level, "report": report_obj}
                    except:
                        record = {"community_id": cid, "level": current_level, "report_raw": text}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(f"   📄 层级 {current_level} 报告已保存: {level_reports_path.name} ({len(level_reports)} 条)")
        except Exception as e:
            logging.warning(f"   ⚠️ 保存层级报告失败: {e}")

    # 合并所有层级的 ID 映射文件（第一次，基于原始层级文件）
    logging.info(f"\n   🔄 合并所有层级的ID映射...")
    merged_id_maps = merge_all_level_id_maps(community_requests_path)

    # 反向同步：使用合并后的ID映射更新各层级报告文件和ID映射文件
    reverse_sync_level_reports_with_id_maps(
        community_requests_path=community_requests_path,
        communities_by_level=communities_by_level,
        all_summaries=all_summaries,
        merged_id_maps=merged_id_maps
    )
    
    # 重新合并ID映射（第二次，基于反向同步后的层级文件，确保完整性）
    logging.info(f"\n   🔄 重新合并ID映射文件（基于反向同步后的层级文件）...")
    final_merged_maps = merge_all_level_id_maps(community_requests_path)
    logging.info(f"   ✅ 最终合并文件包含 {len(final_merged_maps)} 个社区的ID映射")

    logging.info(f"\n✅ 分层社区报告生成完成！共生成 {len(all_summaries)} 个报告")
    return all_summaries


def run_community_summaries(
    client: OpenAI,
    graph: nx.DiGraph,
    model_name: str,
    prompt_dir: str,
    config: Dict[str, Any],
    sleep_interval: int,
    community_requests_path: Path,
    communities_list: List[Dict[str, Any]],
    load_prompt_func,
    build_context_func,
    create_batch_requests_func,
    submit_job_func,
    process_results_func
) -> Dict[str, str]:
    """
    生成社区摘要，支持分层社区和传统扁平社区

    Args:
        client: OpenAI 客户端
        graph: 知识图谱
        model_name: 模型名称
        prompt_dir: Prompt 目录
        config: 配置字典
        sleep_interval: 轮询间隔
        community_requests_path: 请求文件路径
        communities_list: 社区列表（包含层级信息），如果为None则使用传统扁平方式
        load_prompt_func: 加载 prompt 的函数
        build_context_func: 构建上下文的函数
        create_batch_requests_func: 创建批量请求的函数
        submit_job_func: 提交作业的函数
        process_results_func: 处理结果的函数

    Returns:
        社区ID到报告的映射
    """
    max_report_words = str(config["graph_builder"].get("community_summary_max_report_words", 800))
    max_entities = int(config["graph_builder"].get("community_summary_used_entities_num", 25))
    max_relationships = int(config["graph_builder"].get("community_summary_used_relationships_num", 50))

    # 判断是否使用分层模式
    use_hierarchical = communities_list is not None and len(communities_list) > 0

    if use_hierarchical:
        logging.info("🌳 使用分层社区报告生成模式")
        return run_hierarchical_community_summaries(
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
            load_prompt_func=load_prompt_func,
            build_context_func=build_context_func,
            submit_job_func=submit_job_func,
            process_results_func=process_results_func
        )
    else:
        # 传统扁平模式
        logging.info("📊 使用传统扁平社区报告生成模式")
        num_comms = create_batch_requests_func(
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

        job = submit_job_func(client, community_requests_path, model_name, sleep_interval, "CommunitySummary")
        summaries = process_results_func(job, client)
        return summaries

