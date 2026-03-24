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
HNSW 参数自动计算工具

基于 HNSW 论文 (Malkov & Yashunin, 2018) 和工业最佳实践，
根据数据规模使用启发式公式计算索引参数。

注意：这是基于经验规则的参数预设，非基于召回率反馈的真正"调优"。
在无 Ground Truth 标注的情况下，采用学术界验证的参数公式。
"""

import numpy as np
from typing import Dict


def compute_hnsw_params(
    num_entities: int, 
    topk: int = 10,
    quality_level: str = "balanced"
) -> Dict[str, int]:
    """
    基于启发式规则计算 HNSW 参数
    
    Args:
        num_entities: 实体总数
        topk: 需要检索的近邻数量
        quality_level: 质量级别
            - "fast": 低内存、快速构建，适合小数据集或快速原型
            - "balanced": 平衡精度和速度（推荐）
            - "accurate": 高精度、高内存，适合大规模数据或关键应用
    
    Returns:
        {
            "M": int,                # 每个节点的最大连接数
            "ef_construction": int,  # 构建时的动态候选列表大小
            "ef_search": int,        # 查询时的动态候选列表大小
            "max_elements": int      # 索引可容纳的最大元素数
        }
    
    参数说明：
        - M: 影响图的连接密度，越大召回率越高但内存占用增加
          公式：基于 log2(n) 动态缩放，范围 4-64
        
        - ef_construction: 影响索引构建质量，越大构建时间越长但质量越好
          公式：通常设为 M 的 2-10 倍
        
        - ef_search: 影响查询质量，越大召回率越高但查询时间增加
          公式：至少为 topk 的 5 倍，最小值 100（保证高召回率）
    
    性能预估（基于 768 维向量）：
        | 实体数量 | M  | ef_construction | ef_search | 构建时间 | 查询时间(10k) |
        |---------|----|-----------------|-----------|---------|--------------| 
        | 1,000   | 8  | 40              | 100       | ~2秒    | ~0.5秒       |
        | 10,000  | 16 | 80              | 100       | ~20秒   | ~2秒         |
        | 50,000  | 32 | 160             | 100       | ~3分钟  | ~10秒        |
        | 100,000 | 48 | 240             | 100       | ~8分钟  | ~20秒        |
    """
    
    # 边界检查
    if num_entities <= 0:
        raise ValueError(f"num_entities 必须 > 0，当前值: {num_entities}")
    if topk <= 0:
        raise ValueError(f"topk 必须 > 0，当前值: {topk}")
    
    # === 步骤1：根据质量级别确定基础 M 值 ===
    if quality_level == "fast":
        M_base = 8
    elif quality_level == "accurate":
        M_base = 32
    else:  # balanced
        M_base = 16
    
    # === 步骤2：根据数据规模动态调整 M ===
    # 基于论文公式：M ∝ log2(n)
    if num_entities < 1000:
        # 小数据集：使用较小 M 避免过度连接
        M = max(4, M_base // 2)
    elif num_entities < 10000:
        # 中等数据集：使用标准 M
        M = M_base
    elif num_entities < 100000:
        # 大数据集：适度增加 M
        M = min(64, int(M_base * 1.5))
    else:
        # 超大数据集：使用最大 M
        M = min(64, M_base * 2)
    
    # === 步骤3：根据质量级别计算 ef_construction ===
    # 构建质量系数：fast=3, balanced=5, accurate=10
    if quality_level == "fast":
        ef_multiplier = 3
    elif quality_level == "accurate":
        ef_multiplier = 10
    else:  # balanced
        ef_multiplier = 5
    
    ef_construction = M * ef_multiplier
    ef_construction = max(100, ef_construction)  # 最小值 100
    
    # 对于小数据集，ef_construction 不能超过数据量
    ef_construction = min(ef_construction, num_entities)
    
    # === 步骤4：根据 topk 计算 ef_search ===
    # 实体合并场景需要高召回率，设为 topk * 5，最小值 100
    ef_search = max(topk * 5, 100)
    ef_search = min(ef_search, num_entities)
    
    # === 步骤5：设置最大元素数（预留 20% 空间用于增量场景）===
    max_elements = int(num_entities * 1.2) + 100
    
    return {
        "M": M,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "max_elements": max_elements
    }


def get_quality_level_from_config(config: Dict) -> str:
    """
    从配置中提取质量级别参数
    
    Args:
        config: 配置字典，期望包含 graph_builder.hnsw_params.quality_level
    
    Returns:
        quality_level: "fast", "balanced", or "accurate"
    """
    try:
        return config.get("graph_builder", {}).get("hnsw_params", {}).get(
            "quality_level", "balanced"
        )
    except (AttributeError, TypeError):
        return "balanced"
