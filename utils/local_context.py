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

import logging
import networkx as nx
from typing import List, Dict, Any, Set


class LocalSearchContextBuilder:
    """
    负责从种子实体出发，扩展图结构，获取实体、关系、文本块和社区报告，并组装成 Prompt 上下文。
    """

    def __init__(self,
                 graph_data: Dict[str, Any],
                 text_units: Dict[str, str],
                 community_reports: Dict[str, str] = None,
                 context_token_limit: int = 12000):
        """
        Args:
            graph_data: 加载自 final_graph.json 的字典数据
            text_units: chunk_id 到 文本内容 的映射
            community_reports: community_id 到 报告内容 的映射 (可选)
            context_token_limit: 最大 Token 限制
        """
        self.G = self._build_networkx_graph(graph_data)
        self.text_units = text_units
        self.community_reports = community_reports or {}
        self.context_limit = context_token_limit

        # 预处理：建立 chunk_id 到 entity 的反向索引，用于快速查找
        self.chunk_to_entities = {}

    def _build_networkx_graph(self, data: Dict[str, Any]) -> nx.Graph:
        """将 JSON 数据转换为 NetworkX 图对象，方便遍历"""
        logging.info("正在构建内存图结构...")
        G = nx.Graph()
        # 添加节点
        for node in data.get("nodes", []):
            # 使用 id 作为键，存储所有属性
            G.add_node(node["id"], **node)

        # 添加边 (兼容 links 或 relationships 字段)
        edges = data.get("links", []) or data.get("relationships", [])
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                G.add_edge(source, target, **edge)
        return G

    def build(self,
              selected_entities: List[Dict[str, Any]],
              k_hop: int = 1) -> str:
        """
        构建完整的上下文。

        Args:
            selected_entities: 向量检索召回的 Top-K 实体 (LanceDB 结果)
            k_hop: 邻居扩展深度 (通常为 1 或 2)
        """
        # 1. 提取种子节点 ID (注意对齐 ID 格式，final_graph.json 中的 id 通常包含 chunk 前缀)
        # 假设 LanceDB 中的 id 与 final_graph.json 中的 id 一致
        seed_ids = [e["id"] for e in selected_entities if e.get("id") in self.G.nodes]

        if not seed_ids:
            logging.warning("未在图中找到向量检索出的任何实体 ID。请检查 ID 格式是否匹配。")
            # 尝试诊断信息
            if selected_entities:
                sample_id = selected_entities[0].get("id", "N/A")
                graph_sample = list(self.G.nodes)[:3] if self.G.nodes else []
                logging.warning(f"样本 LanceDB ID: {sample_id}")
                logging.warning(f"样本 Graph IDs: {graph_sample}")
            return ""

        logging.info(f"ContextBuilder: 基于 {len(seed_ids)} 个种子实体进行扩展...")

        # 2. 子图扩展 (K-Hop Neighbors)
        candidate_nodes: Set[str] = set(seed_ids)
        current_layer = set(seed_ids)

        for hop in range(k_hop):
            next_layer = set()
            for node_id in current_layer:
                neighbors = list(self.G.neighbors(node_id))
                next_layer.update(neighbors)
            candidate_nodes.update(next_layer)
            current_layer = next_layer
            logging.info(f"  第 {hop+1} 跳扩展: 新增 {len(next_layer)} 个邻居节点")

        logging.info(f"ContextBuilder: 扩展后包含 {len(candidate_nodes)} 个候选节点。")

        # 3. 收集数据组件
        entities_section = self._format_entities(candidate_nodes, seed_ids)
        relationships_section = self._format_relationships(candidate_nodes)
        sources_section = self._format_text_units(candidate_nodes)
        reports_section = self._format_community_reports(candidate_nodes)

        # 统计信息
        num_entities = len([line for line in entities_section.split('\n') if line.strip().startswith('-')])
        num_relationships = len([line for line in relationships_section.split('\n') if line.strip().startswith('-')])

        logging.info(f"ContextBuilder: 收集了 {num_entities} 个实体, {num_relationships} 个关系")
        logging.info(f"ContextBuilder: 实体名称列表: {', '.join(self.entities_info)}")

        # 4. 组合与截断 (简单策略：按顺序拼接，直到超限)
        # 优先级: 实体 > 关系 > 来源原文 > 社区报告
        full_context = []
        current_tokens = 0

        sections = [
            ("Entities", entities_section),
            ("Relationships", relationships_section),
            ("Sources (Text Units)", sources_section),
            ("Community Reports", reports_section)
        ]

        for title, content in sections:
            if not content:
                logging.info(f"ContextBuilder: {title} 为空，跳过")
                continue
            # 粗略估算 Token (1 char ≈ 0.3 token)
            estimated_tokens = int(len(content) * 0.3)
            if current_tokens + estimated_tokens > self.context_limit:
                logging.warning(f"ContextBuilder: Token 限制 ({self.context_limit}) 即将超出，舍弃了 {title}")
                break

            full_context.append(f"{title}\n{content}\n")
            current_tokens += estimated_tokens
            logging.info(f"ContextBuilder: 添加 {title}, 估算 Token: {estimated_tokens}")

        final_context = "\n".join(full_context)
        logging.info(f"ContextBuilder: 最终上下文总 Token 估算: {current_tokens}")

        return final_context

    def _format_entities(self, node_ids: Set[str], seed_ids: List[str]) -> str:
        """格式化实体信息，包含来源引用"""
        lines = []
        entities_info = []  # 存储实体名称
        # 优先展示种子节点
        sorted_nodes = sorted(list(node_ids), key=lambda x: 0 if x in seed_ids else 1)

        for nid in sorted_nodes:
            node = self.G.nodes[nid]
            name = node.get("name", "Unk")
            desc = node.get("description", "No desc")
            entity_type = node.get("type", "")

            # 收集实体名称
            entities_info.append(name)

            # 获取关联的 chunk_ids 用于引用
            c_id = node.get("chunk_id") or node.get("text_unit_ids", [])
            if isinstance(c_id, str):
                c_id = [c_id]
            elif c_id is None:
                c_id = []

            # 标记种子节点（向量检索直接召回的节点）
            marker = "🎯 " if nid in seed_ids else ""

            # 构建实体条目
            type_str = f" ({entity_type})" if entity_type else ""
            if c_id:
                source_str = ", ".join(c_id)
                lines.append(f"{marker}- **{name}**{type_str}: {desc} [cite: {source_str}]")
            else:
                lines.append(f"{marker}- **{name}**{type_str}: {desc}")

        # 将实体名称存储到实例变量中供后续使用
        self.entities_info = entities_info
        return "\n".join(lines)

    def _format_relationships(self, node_ids: Set[str]) -> str:
        """格式化关系信息，包含来源引用和权重"""
        lines = []
        subgraph = self.G.subgraph(node_ids)
        # 按权重排序
        edges = sorted(subgraph.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)

        for u, v, data in edges:
            src = self.G.nodes[u].get("name", u)
            tgt = self.G.nodes[v].get("name", v)
            desc = data.get("description", "Related")
            weight = data.get("weight", 0)
            chunk_id = data.get("chunk_id") or data.get("source_id", "")

            # 构建关系条目
            if chunk_id:
                weight_str = f" (weight: {weight:.2f})" if weight > 0 else ""
                lines.append(f"- **{src} → {tgt}**{weight_str}: {desc} [cite: {chunk_id}]")
            else:
                lines.append(f"- **{src} → {tgt}**: {desc}")

        return "\n".join(lines)

    def _format_text_units(self, node_ids: Set[str]) -> str:
        """根据节点反查 Text Units 原文"""
        relevant_chunk_ids = set()
        for nid in node_ids:
            node = self.G.nodes[nid]
            # 兼容不同的字段名
            c_ids = node.get("chunk_id") or node.get("text_unit_ids")
            if c_ids:
                if isinstance(c_ids, str):
                    relevant_chunk_ids.add(c_ids)
                elif isinstance(c_ids, list):
                    relevant_chunk_ids.update(c_ids)

        lines = []
        # 限制文本单元数量，避免上下文过长
        max_text_units = 15
        for idx, cid in enumerate(list(relevant_chunk_ids)[:max_text_units]):
            text = self.text_units.get(cid)
            if text:
                # 显示完整文本，但如果太长则截断
                if len(text) > 800:
                    preview = text[:800].replace("\n", " ") + "..."
                else:
                    preview = text.replace("\n", " ")
                lines.append(f"**[{cid}]**\n{preview}\n")

        if len(relevant_chunk_ids) > max_text_units:
            lines.append(f"... ({len(relevant_chunk_ids) - max_text_units} more text units omitted)")

        return "\n".join(lines)

    def _format_community_reports(self, node_ids: Set[str]) -> str:
        """获取相关节点的社区报告"""
        community_ids = set()
        for nid in node_ids:
            comm = self.G.nodes[nid].get("community")
            if comm:
                community_ids.add(str(comm))

        lines = []
        # 限制社区报告数量
        max_communities = 5
        for idx, cid in enumerate(list(community_ids)[:max_communities]):
            report = self.community_reports.get(cid)
            if report:
                lines.append(f"**Community {cid}:**\n{report}\n")

        if len(community_ids) > max_communities:
            lines.append(f"... ({len(community_ids) - max_communities} more communities omitted)")

        return "\n".join(lines)