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
知识图谱统计与可视化模块
用于分析和展示知识图谱的拓扑结构特征
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scienceplots
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib后端和样式
plt.style.use(['science', 'no-latex'])  # 使用scienceplots风格


class GraphStatistics:
    """知识图谱统计分析类"""

    def __init__(self, graph_path: str, output_dir: str = "../visualizations"):
        """
        初始化图谱统计分析器

        Args:
            graph_path: 图谱JSON文件路径
            output_dir: 输出目录路径
        """
        self.graph_path = graph_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 支持的图片格式（优先SVG）
        self.image_format = 'svg'
        self.dpi = 300

        # 加载图谱
        self.G = self._load_graph()
        logger.info(f"图谱加载完成: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")

    def _load_graph(self) -> nx.Graph:
        """加载图谱数据（性能优化版本）"""
        logger.info(f"正在加载图谱: {self.graph_path}")

        try:
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 使用networkx的高效加载方法
            G = nx.node_link_graph(data)
            return G

        except Exception as e:
            logger.error(f"加载图谱失败: {e}")
            raise

    def _save_figure(self, fig, name: str, tight: bool = True):
        """
        保存图片（优先SVG格式）

        Args:
            fig: matplotlib图形对象
            name: 文件名（不含扩展名）
            tight: 是否使用紧凑布局
        """
        save_path = self.output_dir / f"{name}.{self.image_format}"

        if tight:
            fig.savefig(save_path, format=self.image_format, dpi=self.dpi,
                        bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(save_path, format=self.image_format, dpi=self.dpi,
                        facecolor='white')

        logger.info(f"图片已保存: {save_path}")

        # 如果SVG失败，尝试PNG
        if self.image_format == 'svg':
            try:
                plt.savefig(self.output_dir / f"{name}.png", dpi=self.dpi,
                            bbox_inches='tight', facecolor='white')
            except Exception as e:
                logger.warning(f"PNG备份保存失败: {e}")

        plt.close(fig)

    @lru_cache(maxsize=1)
    def _compute_node_metrics(self) -> Dict:
        """计算节点度量（缓存优化）"""
        logger.info("计算节点度量...")

        metrics = {
            'degree': dict(self.G.degree()),
            'in_degree': dict(self.G.in_degree()) if self.G.is_directed() else None,
            'out_degree': dict(self.G.out_degree()) if self.G.is_directed() else None,
        }

        # 只对不太大的图计算中心性（性能考虑）
        if self.G.number_of_nodes() < 10000:
            logger.info("计算中心性指标...")
            metrics['betweenness'] = nx.betweenness_centrality(self.G)
            metrics['closeness'] = nx.closeness_centrality(self.G)
            if not self.G.is_directed():
                metrics['pagerank'] = nx.pagerank(self.G, max_iter=50)
        else:
            logger.info("图谱过大，跳过中心性计算")

        return metrics

    def plot_degree_distribution(self):
        """度分布图（对数-对数坐标）"""
        logger.info("绘制度分布图...")

        degrees = [d for n, d in self.G.degree()]
        degree_count = Counter(degrees)

        # 排序
        degrees_sorted = sorted(degree_count.keys())
        counts = [degree_count[d] for d in degrees_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 线性坐标
        axes[0].bar(degrees_sorted, counts, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Degree', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Degree Distribution (Linear)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 对数坐标
        axes[1].loglog(degrees_sorted, counts, 'o-', alpha=0.7, markersize=4)
        axes[1].set_xlabel('Degree (log)', fontsize=12)
        axes[1].set_ylabel('Count (log)', fontsize=12)
        axes[1].set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        self._save_figure(fig, 'degree_distribution')

    def plot_node_type_distribution(self):
        """节点类型分布饼图"""
        logger.info("绘制节点类型分布...")

        node_types = [data.get('type', 'Unknown') for n, data in self.G.nodes(data=True)]
        type_count = Counter(node_types)

        # 取最小的前 n 个类型，使其计数和占比达到总数的 10\%
        total = sum(type_count.values())
        if total == 0:
            logger.warning("未找到节点类型，跳过节点类型分布图")
            return

        most_common = type_count.most_common()
        cum = 0
        top_types = []
        for t, cnt in most_common:
            top_types.append((t, cnt))
            cum += cnt
            if cum / total >= 0.25:
                break

        labels = [t[0] for t in top_types]
        sizes = [t[1] for t in top_types]

        fig, ax = plt.subplots(figsize=(10, 8))

        # 使用科学配色
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors,
            textprops={'fontsize': 18, 'fontfamily': 'Arial'}
        )

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(18)

        plt.tight_layout()
        self._save_figure(fig, 'node_type_distribution')

    def plot_relationship_type_distribution(self):
        """关系类型分布"""
        logger.info("绘制关系类型分布...")

        relationships = []
        for u, v, data in self.G.edges(data=True):
            rel = data.get('relationship') or data.get('label') or data.get('relation') or 'UNKNOWN'
            relationships.append(rel)

        rel_count = Counter(relationships)
        top_rels = rel_count.most_common(20)

        labels = [r[0] for r in top_rels]
        counts = [r[1] for r in top_rels]

        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(labels))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))

        bars = ax.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Relationship Type Distribution (Top 20)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f' {int(width)}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, 'relationship_type_distribution')

    def plot_community_distribution(self):
        """社区分布图（单图：社区序号 vs 社区大小，纵坐标对数）"""
        logger.info("绘制社区分布...")

        communities = [data.get('community', 'None') for n, data in self.G.nodes(data=True)]

        # 过滤掉None
        communities = [c for c in communities if c != 'None' and c is not None]

        if not communities:
            logger.warning("未找到社区信息，跳过社区分布图")
            return

        comm_count = Counter(communities)

        # 仅保留社区内节点数大于5的社区
        filtered_comm = {comm: cnt for comm, cnt in comm_count.items() if cnt > 5}

        if not filtered_comm:
            logger.warning("没有社区的节点数超过5，跳过社区分布图")
            return

        # 按社区大小从大到小排序，社区序号即排名
        comm_sizes = sorted(filtered_comm.values(), reverse=True)

        # 全局字体设置为 18（影响所有图表）
        plt.rcParams.update({"font.size": 18, "font.family": "Arial"})

        fig, ax = plt.subplots(figsize=(6, 6))

        x = range(1, len(comm_sizes) + 1)
        ax.plot(x, comm_sizes, linestyle='-', linewidth=2, color='#2B579A', alpha=0.8)

        ax.set_xlabel('Community Rank')
        ax.set_ylabel('Community Size')
        ax.set_yscale('log')

        plt.tight_layout()
        self._save_figure(fig, 'community_distribution')

    def plot_centrality_comparison(self):
        """中心性指标对比（仅小图谱）"""
        if self.G.number_of_nodes() >= 10000:
            logger.info("图谱过大，跳过中心性对比图")
            return

        logger.info("绘制中心性对比...")

        metrics = self._compute_node_metrics()

        if 'betweenness' not in metrics:
            logger.warning("未计算中心性，跳过")
            return

        # 选择Top K节点
        top_k = 30
        top_nodes_betweenness = sorted(metrics['betweenness'].items(),
                                        key=lambda x: x[1], reverse=True)[:top_k]
        node_names = [self.G.nodes[n].get('name', n) for n, _ in top_nodes_betweenness]

        betweenness_vals = [v for _, v in top_nodes_betweenness]
        closeness_vals = [metrics['closeness'][n] for n, _ in top_nodes_betweenness]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Betweenness Centrality
        axes[0].barh(range(len(node_names)), betweenness_vals, alpha=0.8,
                     color='coral', edgecolor='black', linewidth=0.5)
        axes[0].set_yticks(range(len(node_names)))
        axes[0].set_yticklabels(node_names, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Betweenness Centrality', fontsize=12)
        axes[0].set_title('Top 30 Nodes by Betweenness Centrality', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)

        # Closeness Centrality
        axes[1].barh(range(len(node_names)), closeness_vals, alpha=0.8,
                     color='skyblue', edgecolor='black', linewidth=0.5)
        axes[1].set_yticks(range(len(node_names)))
        axes[1].set_yticklabels(node_names, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Closeness Centrality', fontsize=12)
        axes[1].set_title('Top 30 Nodes by Closeness Centrality', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'centrality_comparison')

    def plot_graph_connectivity(self):
        """图连通性分析"""
        logger.info("绘制图连通性分析...")

        if self.G.is_directed():
            # 弱连通分量
            weak_components = list(nx.weakly_connected_components(self.G))
            component_sizes = sorted([len(c) for c in weak_components], reverse=True)
            title_suffix = "(Weakly Connected)"
        else:
            # 连通分量
            components = list(nx.connected_components(self.G))
            component_sizes = sorted([len(c) for c in components], reverse=True)
            title_suffix = "(Connected)"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 连通分量大小分布
        if len(component_sizes) > 1:
            axes[0].bar(range(len(component_sizes)), component_sizes,
                        alpha=0.7, color='teal', edgecolor='black', linewidth=0.5)
            axes[0].set_xlabel('Component Rank', fontsize=12)
            axes[0].set_ylabel('Component Size', fontsize=12)
            axes[0].set_title(f'Connected Component Sizes {title_suffix}',
                              fontsize=14, fontweight='bold')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Single Connected Component',
                         ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
            axes[0].set_title(f'Graph Connectivity {title_suffix}', fontsize=14, fontweight='bold')

        # 节点度vs连通性
        degrees = dict(self.G.degree())
        degree_values = list(degrees.values())

        axes[1].hist(degree_values, bins=50, alpha=0.7, color='purple',
                     edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Node Degree', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Degree Distribution for Connectivity', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'graph_connectivity')

    def plot_edge_weight_distribution(self):
        """边权重分布（如果有）"""
        logger.info("检查边权重...")

        weights = []
        for u, v, data in self.G.edges(data=True):
            weight = data.get('weight')
            if weight is not None:
                try:
                    weights.append(float(weight))
                except (ValueError, TypeError):
                    pass

        if not weights:
            logger.info("未找到边权重信息，跳过边权重分布图")
            return

        logger.info("绘制边权重分布...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 直方图
        axes[0].hist(weights, bins=50, alpha=0.7, color='gold',
                     edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Edge Weight', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Edge Weight Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 累积分布
        sorted_weights = np.sort(weights)
        cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)

        axes[1].plot(sorted_weights, cumulative, linewidth=2, color='crimson')
        axes[1].set_xlabel('Edge Weight', fontsize=12)
        axes[1].set_ylabel('Cumulative Probability', fontsize=12)
        axes[1].set_title('Cumulative Edge Weight Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, 'edge_weight_distribution')

    def plot_schema_heatmap(self):
        """实体类型关系热力图"""
        logger.info("绘制Schema热力图...")

        # 统计实体类型之间的关系
        edge_type_matrix = defaultdict(lambda: defaultdict(int))

        for u, v, data in self.G.edges(data=True):
            src_type = self.G.nodes[u].get('type', 'Unknown')
            tgt_type = self.G.nodes[v].get('type', 'Unknown')
            edge_type_matrix[src_type][tgt_type] += 1

        # 取前15个最常见类型
        node_types = [data.get('type', 'Unknown') for n, data in self.G.nodes(data=True)]
        type_count = Counter(node_types)
        top_types = [t[0] for t in type_count.most_common(15)]

        # 构建矩阵
        matrix = np.zeros((len(top_types), len(top_types)))
        for i, src_type in enumerate(top_types):
            for j, tgt_type in enumerate(top_types):
                matrix[i, j] = edge_type_matrix[src_type][tgt_type]

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        # 设置刻度
        ax.set_xticks(np.arange(len(top_types)))
        ax.set_yticks(np.arange(len(top_types)))
        ax.set_xticklabels(top_types, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(top_types, fontsize=10)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Relationships', fontsize=12)

        # 添加数值标注（仅显示非零值）
        for i in range(len(top_types)):
            for j in range(len(top_types)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, int(matrix[i, j]),
                                   ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Entity Type Relationship Heatmap (Top 15 Types)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Type', fontsize=12)
        ax.set_ylabel('Source Type', fontsize=12)

        plt.tight_layout()
        self._save_figure(fig, 'schema_heatmap')

    def generate_summary_statistics(self):
        """生成统计摘要文本"""
        logger.info("生成统计摘要...")

        summary = {
            "基本信息": {
                "节点数": self.G.number_of_nodes(),
                "边数": self.G.number_of_edges(),
                "是否有向图": self.G.is_directed(),
                "平均度": np.mean([d for n, d in self.G.degree()])
            },
            "度统计": {
                "最大度": max([d for n, d in self.G.degree()]),
                "最小度": min([d for n, d in self.G.degree()]),
                "度中位数": np.median([d for n, d in self.G.degree()])
            }
        }

        # 连通性
        if self.G.is_directed():
            weak_components = list(nx.weakly_connected_components(self.G))
            summary["连通性"] = {
                "弱连通分量数": len(weak_components),
                "最大弱连通分量大小": max(len(c) for c in weak_components)
            }
        else:
            components = list(nx.connected_components(self.G))
            summary["连通性"] = {
                "连通分量数": len(components),
                "最大连通分量大小": max(len(c) for c in components)
            }

        # 节点类型统计
        node_types = [data.get('type', 'Unknown') for n, data in self.G.nodes(data=True)]
        type_count = Counter(node_types)
        summary["节点类型统计"] = {
            "节点类型数": len(type_count),
            "最常见类型": type_count.most_common(5)
        }

        # 社区统计
        communities = [data.get('community') for n, data in self.G.nodes(data=True)
                       if data.get('community') is not None]
        if communities:
            summary["社区统计"] = {
                "社区数": len(set(communities)),
                "平均社区大小": len(communities) / len(set(communities))
            }

        # 保存到文件
        summary_path = self.output_dir / "statistics_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("知识图谱统计摘要\n")
            f.write("=" * 50 + "\n\n")

            for category, stats in summary.items():
                f.write(f"{category}:\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.2f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

        logger.info(f"统计摘要已保存: {summary_path}")

    def run_all_visualizations(self):
        """运行所有可视化"""
        logger.info("=" * 60)
        logger.info("开始生成知识图谱可视化")
        logger.info("=" * 60)

        visualizations = [
            self.plot_degree_distribution,
            self.plot_node_type_distribution,
            self.plot_relationship_type_distribution,
            self.plot_community_distribution,
            self.plot_centrality_comparison,
            self.plot_graph_connectivity,
            self.plot_edge_weight_distribution,
            self.plot_schema_heatmap,
            self.generate_summary_statistics
        ]

        for viz_func in visualizations:
            try:
                viz_func()
            except Exception as e:
                logger.error(f"执行 {viz_func.__name__} 时出错: {e}")
                continue

        logger.info("=" * 60)
        logger.info(f"所有可视化已完成！输出目录: {self.output_dir}")
        logger.info("=" * 60)


def main():
    """主函数"""
    # 配置路径
    graph_path = "../data/graphs/final_graph.json"
    output_dir = "../visualizations"

    # 创建统计分析器
    stats = GraphStatistics(graph_path, output_dir)

    # 运行所有可视化
    stats.run_all_visualizations()


if __name__ == "__main__":
    main()

