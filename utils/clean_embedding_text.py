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

import re
import json
import logging
import networkx as nx
from pathlib import Path
from typing import Dict


class EmbeddingTextCleaner:
    """
    一个高级文本清洁器，用于在嵌入之前“去污染”文本。

    它利用图谱和ID映射表将本地ID（如 E1）替换回
    它们的人类可读名称（如“相接触规则”），
    并移除所有溯源标记。
    """

    def __init__(self, final_graph_path: Path, id_maps_path: Path):
        """
        初始化清洁器。

        参数:
            final_graph_path (Path): 指向 final_graph.json  的路径。
            id_maps_path (Path): 指向 community_detection_id_maps.json 的路径。
        """
        logging.info("初始化 EmbeddingTextCleaner...")
        self.global_node_name_map: Dict[str, str] = {}
        self.global_edge_name_map: Dict[str, str] = {}
        self.local_id_maps: Dict[str, Dict[str, str]] = {}
        self.rich_map_cache: Dict[str, Dict[str, str]] = {}  # 社区ID -> 富映射表 的缓存

        self._load_global_graph(final_graph_path)
        self._load_local_id_maps(id_maps_path)
        logging.info(
            f"Cleaner 初始化成功。加载了 {len(self.global_node_name_map)} 个节点和 {len(self.global_edge_name_map)} 个关系名称。")

    def _load_global_graph(self, final_graph_path: Path):
        """从 final_graph.json 加载节点和边的名称映射。"""
        try:
            with open(final_graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # 使用 networkx 正确解析图谱结构
            graph = nx.node_link_graph(graph_data)

            for node_id, data in graph.nodes(data=True):
                # 映射: 全局ID -> 实体名称
                self.global_node_name_map[node_id] = data.get('name', node_id)

            for u, v, data in graph.edges(data=True):
                edge_id = data.get('id')
                if edge_id:
                    # 映射: 全局ID -> 关系类型
                    self.global_edge_name_map[edge_id] = data.get('relationship', edge_id)

            logging.info(f"成功从 {final_graph_path.name} 加载全局名称映射。")
        except FileNotFoundError:
            logging.error(f"关键文件未找到: {final_graph_path}")
            raise
        except Exception as e:
            logging.error(f"加载或解析 final_graph.json 失败: {e}", exc_info=True)
            raise

    def _load_local_id_maps(self, id_maps_path: Path):
        """加载 community_detection_id_maps.json文件。"""
        try:
            with open(id_maps_path, 'r', encoding='utf-8') as f:
                self.local_id_maps = json.load(
                    f)  # 结构: {comm_id: {local_id: global_id}}
            logging.info(f"成功从 {id_maps_path.name} 加载本地ID映射。")
        except FileNotFoundError:
            logging.error(f"关键文件未找到: {id_maps_path}")
            raise
        except Exception as e:
            logging.error(f"加载 community_detection_id_maps.json 失败: {e}",
                          exc_info=True)
            raise

    def _get_rich_lookup(self, community_id: str) -> Dict[str, str]:
        """
        动态构建（或从缓存获取）一个“富查找表”。
        例如：{"E1": "相接触规则", "R1": "APPLIES_TO"}
        """
        # 检查缓存
        if community_id in self.rich_map_cache:
            return self.rich_map_cache[community_id]

        simple_map = self.local_id_maps.get(community_id)
        if not simple_map:
            logging.debug(f"社区ID '{community_id}' 没有ID映射表。代词替换将被跳过。")
            return {}

        rich_map = {}
        for local_id, global_id in simple_map.items():
            if local_id.startswith('E'):
                # 这是一个实体/节点
                # 从 全局ID 查找 名称
                rich_map[local_id] = self.global_node_name_map.get(global_id, global_id)
            elif local_id.startswith('R'):
                # 这是一个关系/边
                rich_map[local_id] = self.global_edge_name_map.get(global_id, global_id)

        # 缓存结果以便将来使用
        self.rich_map_cache[community_id] = rich_map
        return rich_map

    def clean_text(self, text: str, community_id: str) -> str:
        """
        (公共方法)
        对单个文本字符串执行完整的清洁操作。

        参数:
            text (str): 从社区报告中提取的原始文本（例如 finding ）。
            community_id (str): 该文本所属的社区ID。

        返回:
            str: 准备好进行嵌入的纯净语义文本。
        """
        if not isinstance(text, str):
            return ""

        # 1. 获取该社区的 {E1: "实体名", R1: "关系名"} 映射表
        rich_map = self._get_rich_lookup(community_id)

        # 2. 移除 [Data: ...] 溯源块
        cleaned_text = re.sub(r'\[Data:.*?]', '', text)

        # 3. 移除括号标签, e.g., "相接触规则 (E1)" -> "相接触规则"
        cleaned_text = re.sub(r'\s*\([ER]\d+\)', '', cleaned_text)

        # 4. (升级) 替换独立的 "代词" ID, e.g., "E1" -> "相接触规则"
        # 若无富映射，但文本中仍包含独立的 [ER]\d+ 代词，则发出告警

        if not rich_map and re.search(r'\b[ER]\d+\b', cleaned_text):
            logging.warning(f"社区ID '{community_id}' 缺少ID映射表；检测到待替换的代词，将跳过代词替换。")

        elif rich_map:
            # 按键长度降序排序 (确保 "E10" 在 "E1" 之前被替换)
            sorted_keys = sorted(rich_map.keys(), key=len, reverse=True)

            # 创建一个高效的正则表达式，匹配所有独立的键
            # \b 确保我们只匹配完整的单词 (e.g., "E1" 而不是 "THEOREM1")
            pattern = r'\b(' + r'|'.join(re.escape(k) for k in sorted_keys) + r')\b'

            # 定义一个替换函数，用于在 rich_map 中查找匹配项
            replacer = lambda m: rich_map.get(m.group(0), m.group(0))

            cleaned_text = re.sub(pattern, replacer, cleaned_text)

        # 5. 清理可能残留的多余空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text


# --- 示例用法 ---
if __name__ == "__main__":
    # 假设此脚本在 core/pipeline/ 目录下
    # 并且 final_graph.json 和 id_maps.json 在 data/graphs/ 目录下

    # 1. 定义路径
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    GRAPH_PATH = PROJECT_ROOT / "data" / "graphs" / "final_graph.json"
    ID_MAPS_PATH = PROJECT_ROOT / "data" / "cache" / "community_detection_id_maps.json"  # 假设您把它存在这里
    SUMMARIES_PATH = PROJECT_ROOT / "data" / "reports" / "community_summaries.jsonl"  # 假设

    # 2. 初始化清洁器 (一次性加载所有映射)
    try:
        cleaner = EmbeddingTextCleaner(
            final_graph_path=GRAPH_PATH,
            id_maps_path=ID_MAPS_PATH
        )

        # 3. 模拟从 community_summaries.jsonl 读取数据并清洁
        logging.info("\n" + "=" * 30 + " 开始清洁测试 " + "=" * 30)

        # --- 测试用例 1: 您提供的例子 ---
        community_id_test_1 = "20"  # 假设 "相接触规则" 在社区 69
        text_to_clean_1 = "本报告分析了一个由两个实体组成的简单社区：相接触规则 (E1) 和多组分图 (E2)。社区的核心结构是 E1 对 E2 的直接应用关系 (R1) [Data: Entities (E1, E2); Relationships (R1)]。"

        cleaned_text_1 = cleaner.clean_text(text_to_clean_1, community_id_test_1)

        print(f"\n--- 社区 {community_id_test_1} 测试 ---")
        print(f"原始文本: {text_to_clean_1}")
        print(f"清洁文本: {cleaned_text_1}")

        # --- 测试用例 2: 从 community_summaries.jsonl 中提取一个真实案例 ---
        community_id_test_2 = "8"
        text_to_clean_2 = "个人 R. Raj (E2) 和 M. F. Ashby (E3) 作为参与者与该会议相关联 [Data: Entities (E1, E2, E3); Relationships (R1, R2)]。该会议在地理上与伦敦 (E4) 相关...。它还通过组织关系与金属学会 (E5) 和物理学会 (E6) 相关联。"

        cleaned_text_2 = cleaner.clean_text(text_to_clean_2, community_id_test_2)

        print(f"\n--- 社区 {community_id_test_2} 测试 ---")
        print(f"原始文本: {text_to_clean_2}")
        print(f"清洁文本: {cleaned_text_2}")

        # --- 测试用例 3: 另一个真实案例 (混合了代词和括号) ---
        community_id_test_3 = "5"
        text_to_clean_3 = "核心实体是等温截面 (E3)，它代表了组分 A、B 和 C 的三元相图。E3 被明确地置于三个主要组分中：组分 A (E1)、组分 B (E2) 和组分 C (E4) [Data: Entities (E3, E1, E2, E4); Relationships (R1, R2, R3)]。"

        cleaned_text_3 = cleaner.clean_text(text_to_clean_3, community_id_test_3)

        print(f"\n--- 社区 {community_id_test_3} 测试 ---")
        print(f"原始文本: {text_to_clean_3}")
        print(f"清洁文本: {cleaned_text_3}")

    except FileNotFoundError:
        logging.error("测试失败：未能找到所需的JSON文件。请确保路径正确。")
    except Exception as e:
        logging.error(f"测试失败：{e}", exc_info=True)