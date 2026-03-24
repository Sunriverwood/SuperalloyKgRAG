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

import json
import logging
import sys
from pathlib import Path
import networkx as nx

# 确保能找到项目根目录模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 从已有代码导入 社区发现 和 中心性计算 方法
from core.pipeline_qwen.graph_builder_qwen import detect_communities
from utils.community_importance import calculate_community_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s")

def main():
    """
    独立脚本：针对巨大未合并图 (disambiguation_graph)
    1. 基于图拓扑进行快速社区发现 (仅仅聚类，不请求 LLM 总结)
    2. 基于分离出的若干小社区，安全、高效地计算边介数中心性与 composite_importance
    3. 输出带有 reasoning 需要的权重特征的 graph.json
    """
    input_path = PROJECT_ROOT / "data/graphs/disambiguation_graph.json"
    output_path = input_path  # 直接覆盖原文件
    
    if not input_path.exists():
        logging.error(f"找不到文件: {input_path}")
        return

    # 1. 加载图
    logging.info(f"正在加载未合并图谱: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 构建 NetworkX DiGraph
    G = nx.node_link_graph(data, directed=True, edges="links")
    logging.info(f"✅ 图谱加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 2. 纯粹社区聚类 (仅仅为了分治大图)
    # 调用 graph_builder 里的核心算法（这里默认使用分层或扁平，传参可以调整以优化大图发现速度）
    # 大图可考虑暂时关闭分层 (use_hierarchical=False) 来加快单纯分组速度
    logging.info("开始多模块社区发现(仅拓扑聚类)...")
    try:
        # 为了极速分组，这里建议关闭分层，只做扁平聚类拿到社区划分列表即可
        G_clustered, communities_list = detect_communities(
            graph=G, 
            weight_alpha=0.6, 
            use_hierarchical=False 
        )
        logging.info(f"✅ 社区聚类完成，共划分为 {len(communities_list)} 个小社区。")
    except Exception as e:
        logging.error(f"社区聚类失败: {e}", exc_info=True)
        return
    
    # 获取叶子/所有社区对应的节点列表
    community_nodes_list = [comm_info['node_ids'] for comm_info in communities_list]
    
    # 3. 分组并行计算边介数中心性与 composite_importance
    # 原有的 calculate_community_importance 其实内部自带进程池，传进去社区划分即可
    logging.info(f"开始利用进程池基于 {len(community_nodes_list)} 个社区的划分，计算复合重要性分数...")
    G_updated = calculate_community_importance(
        graph=G_clustered,
        communities=community_nodes_list,
        weight_alpha=0.6
    )
    logging.info("✅ 复合重要性计算完成！")
    
    # 4. 保存新图谱数据
    logging.info(f"正在保存计算好的新图谱至: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = nx.node_link_data(G_updated, edges="links")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logging.info("✅ 所有步骤已完成！你可以使用新图重新运行模型训练了。")

if __name__ == "__main__":
    main()
