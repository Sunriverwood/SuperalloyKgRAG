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
分析超合金知识图谱中composition, processing, structure, property, performance之间的关系
python ontology.py --mode full
python ontology.py --mode viz



"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Set
import argparse
import pickle

# 导入并应用scienceplots样式
import scienceplots  # noqa: F401
plt.style.use(['science', 'no-latex'])

# 定义关键节点类型
KEY_NODE_TYPES = {
    'composition': ['COMPOSITION', 'ELEMENT', 'ALLOY', 'CHEMICAL_COMPOSITION'],
    'processing': ['PROCESSING', 'HEAT_TREATMENT', 'MANUFACTURING', 'PROCESS'],
    'structure': ['STRUCTURE', 'MICROSTRUCTURE', 'PHASE', 'GRAIN'],
    'property': ['PROPERTY', 'MECHANICAL_PROPERTY', 'PHYSICAL_PROPERTY'],
    'performance': ['PERFORMANCE', 'BEHAVIOR', 'RESPONSE']
}


class SuperalloyRelationshipAnalyzer:
    """超合金关系分析器"""

    def __init__(self, graph_path: str, output_dir: str = "../visualizations"):
        """初始化分析器"""
        self.graph_path = graph_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载图谱
        print(f"正在加载图谱: {graph_path}")
        self.G = self._load_graph()
        print(f"图谱加载完成: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")

        # 分类节点
        self.categorized_nodes = self._categorize_nodes()
        self._print_category_stats()

    def _load_graph(self) -> nx.Graph:
        """加载图谱数据"""
        try:
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return nx.node_link_graph(data)
        except Exception as e:
            print(f"加载图谱失败: {e}")
            raise

    def _categorize_nodes(self) -> Dict[str, Set[str]]:
        """将节点按类型分类"""
        categorized = defaultdict(set)

        for node_id, node_data in self.G.nodes(data=True):
            node_type = node_data.get('type', '').upper()

            # 检查节点类型属于哪个大类
            for category, types in KEY_NODE_TYPES.items():
                if any(t in node_type for t in types):
                    categorized[category].add(node_id)
                    break

        return dict(categorized)

    def _print_category_stats(self):
        """打印各类别统计信息"""
        print("\n" + "="*60)
        print("节点分类统计:")
        print("="*60)
        for category, nodes in self.categorized_nodes.items():
            print(f"{category:15s}: {len(nodes):6d} 个节点")
        print("="*60 + "\n")

    def analyze_pairwise_relationships(self) -> Dict[Tuple[str, str], Dict]:
        """分析两两类别之间的关系（区分方向）"""
        print("分析两两类别之间的关系（区分方向）...")

        categories = list(self.categorized_nodes.keys())
        results = {}

        # 分析所有方向的关系，包括 cat1->cat2 和 cat2->cat1
        for cat1 in categories:
            for cat2 in categories:
                key = (cat1, cat2)
                results[key] = self._analyze_category_pair(cat1, cat2)

        return results

    def _analyze_category_pair(self, cat1: str, cat2: str) -> Dict:
        """分析从cat1到cat2的单向关系"""
        nodes1 = self.categorized_nodes.get(cat1, set())
        nodes2 = self.categorized_nodes.get(cat2, set())

        relationships = []
        relationship_types = Counter()
        edge_weights = []

        # 只遍历从 cat1 到 cat2 的边
        for node1 in nodes1:
            for node2 in nodes2:
                # 只检查从 node1 到 node2 的边
                if self.G.has_edge(node1, node2):
                    edge_data = self.G.get_edge_data(node1, node2)
                    rel_type = edge_data.get('type', edge_data.get('relationship', 'UNKNOWN'))
                    weight = edge_data.get('weight', 1.0)

                    relationships.append({
                        'source': node1,
                        'target': node2,
                        'source_label': self.G.nodes[node1].get('label', node1),
                        'target_label': self.G.nodes[node2].get('label', node2),
                        'type': rel_type,
                        'weight': weight,
                        'direction': f'{cat1} -> {cat2}'
                    })
                    relationship_types[rel_type] += 1
                    edge_weights.append(weight)

        return {
            'count': len(relationships),
            'relationships': relationships,
            'relationship_types': dict(relationship_types),
            'avg_weight': np.mean(edge_weights) if edge_weights else 0,
            'max_weight': max(edge_weights) if edge_weights else 0,
            'min_weight': min(edge_weights) if edge_weights else 0
        }

    def save_results_to_excel(self, results: Dict, filename: str = "superalloy_relationships.xlsx"):
        """将结果保存到Excel文件"""
        output_path = self.output_dir / filename
        print(f"\n保存结果到: {output_path}")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 总览sheet
            summary_data = []
            for (cat1, cat2), data in results.items():
                summary_data.append({
                    '源类别': cat1,
                    '目标类别': cat2,
                    '关系数量': data['count'],
                    '平均权重': data['avg_weight'],
                    '最大权重': data['max_weight'],
                    '最小权重': data['min_weight'],
                    '关系类型数': len(data['relationship_types'])
                })

            df_summary = pd.DataFrame(summary_data)
            df_summary = df_summary.sort_values('关系数量', ascending=False)
            df_summary.to_excel(writer, sheet_name='总览', index=False)

            # 2. 为每对类别创建详细sheet
            for (cat1, cat2), data in results.items():
                if data['count'] > 0:
                    sheet_name = f"{cat1[:10]}-{cat2[:10]}"[:31]  # Excel sheet名称限制
                    df_detail = pd.DataFrame(data['relationships'])
                    df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

            # 3. 关系类型统计sheet
            all_rel_types = Counter()
            for data in results.values():
                all_rel_types.update(data['relationship_types'])

            df_rel_types = pd.DataFrame([
                {'关系类型': k, '出现次数': v}
                for k, v in all_rel_types.most_common()
            ])
            df_rel_types.to_excel(writer, sheet_name='关系类型统计', index=False)

        print(f"结果已保存!")

    def save_analysis_results(self, results: Dict, filename: str = "analysis_results.pkl"):
        """保存分析结果到文件以便后续使用"""
        output_path = self.output_dir / filename
        # 保存分析结果和类别信息
        cache_data = {
            'results': results,
            'categorized_nodes': self.categorized_nodes
        }
        with open(output_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"分析结果已缓存到: {output_path}")

    def load_analysis_results(self, filename: str = "analysis_results.pkl") -> Dict:
        """从文件加载之前保存的分析结果"""
        input_path = self.output_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"未找到缓存文件: {input_path}")

        with open(input_path, 'rb') as f:
            cache_data = pickle.load(f)

        # 兼容旧格式和新格式
        if isinstance(cache_data, dict) and 'results' in cache_data:
            # 新格式：包含 results 和 categorized_nodes
            self.categorized_nodes = cache_data['categorized_nodes']
            results = cache_data['results']
        else:
            # 旧格式：只有 results
            results = cache_data
            # 如果是旧格式，无法获取 categorized_nodes，需要重新分析
            print("警告: 使用旧版本缓存文件，缺少类别信息")
            self.categorized_nodes = {}

        print(f"已加载缓存的分析结果: {input_path}")
        return results

    def visualize_relationship_matrix(self, results: Dict):
        """可视化关系矩阵热图"""
        print("\n绘制关系矩阵热图...")

        categories = sorted(self.categorized_nodes.keys())
        n = len(categories)
        matrix = np.zeros((n, n))

        # 构建矩阵（行=源类别，列=目标类别）
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                key = (cat1, cat2)
                if key in results:
                    matrix[i, j] = results[key]['count']

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))

        # 设置全局字体为 Arial 且字号为 18（局部修改 rcParams）
        plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})

        # 使用 Arial 字体对象（可用于显式 fontproperties）
        from matplotlib import font_manager as fm
        arial = fm.FontProperties(family='Arial', size=18)

        # 根据可选属性来自定义首尾颜色，默认保留原始色阶
        start_color = getattr(self, "cmap_start", "#C8D9F7")  # 浅色（最小值）
        end_color = getattr(self, "cmap_end", "#C4BADF")  # 深色（最大值）

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        # 设置刻度
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(categories, rotation=30, ha="right", fontproperties=arial)
        ax.set_yticklabels(categories, fontproperties=arial)

        # 添加数值标注
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(matrix[i, j])),
                        ha="center", va="center", color="black",
                        fontsize=18, fontproperties=arial)

        ax.set_xlabel('Target Category', fontsize=18, fontproperties=arial)
        ax.set_ylabel('Source Category', fontsize=18, fontproperties=arial)

        # 添加colorbar
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        # 保存
        output_path = self.output_dir / "relationship_matrix.svg"
        fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"热图已保存: {output_path}")

        # 同时保存PNG备份
        png_path = self.output_dir / "relationship_matrix.png"
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')

        plt.close(fig)

    def visualize_relationship_types_distribution(self, results: Dict):
        """可视化关系类型分布"""
        print("\n绘制关系类型分布图...")

        # 收集所有关系类型
        all_rel_types = Counter()
        for data in results.values():
            all_rel_types.update(data['relationship_types'])

        # 取前20个最常见的关系类型
        top_rel_types = all_rel_types.most_common(20)
        types, counts = zip(*top_rel_types) if top_rel_types else ([], [])

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(range(len(types)), counts, color='steelblue')
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types)
        ax.set_xlabel('Count')
        ax.set_title('Top 20 Relationship Types in Superalloy Knowledge Graph')
        ax.invert_yaxis()

        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center')

        # 保存
        output_path = self.output_dir / "relationship_types_distribution.svg"
        fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"关系类型分布图已保存: {output_path}")

        # PNG备份
        png_path = self.output_dir / "relationship_types_distribution.png"
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')

        plt.close(fig)

    def print_summary(self, results: Dict):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("超合金知识图谱关系分析摘要")
        print("="*80)

        # 按关系数量排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)

        for (cat1, cat2), data in sorted_results:
            if data['count'] > 0:
                print(f"\n【{cat1} <-> {cat2}】")
                print(f"  关系总数: {data['count']}")
                print(f"  平均权重: {data['avg_weight']:.4f}")
                print(f"  权重范围: [{data['min_weight']:.4f}, {data['max_weight']:.4f}]")
                print(f"  关系类型: {len(data['relationship_types'])} 种")

                # 显示前5个最常见的关系类型
                top_types = sorted(data['relationship_types'].items(),
                                 key=lambda x: x[1], reverse=True)[:5]
                print(f"  主要关系类型:")
                for rel_type, count in top_types:
                    print(f"    - {rel_type}: {count}")

        print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='超合金知识图谱关系分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整分析流程（默认）
  python ontology.py
  
  # 只进行可视化（使用缓存的分析结果）
  python ontology.py --mode viz
  
  # 完整分析并保存结果
  python ontology.py --mode full
  
  # 指定图谱路径
  python ontology.py --graph-path ../data/graphs/final_graph.json
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'viz', 'analyze'],
        default='full',
        help="""运行模式:
        full - 完整流程（分析+可视化+保存）[默认]
        analyze - 仅分析并保存结果（不可视化）
        viz - 仅可视化（使用缓存结果）"""
    )

    parser.add_argument(
        '--graph-path',
        default='../data/graphs/final_graph.json',
        help='图谱文件路径（默认: ../data/graphs/final_graph.json）'
    )

    parser.add_argument(
        '--output-dir',
        default='../visualizations',
        help='输出目录路径（默认: ../visualizations）'
    )

    parser.add_argument(
        '--cache-file',
        default='analysis_results.pkl',
        help='缓存文件名（默认: analysis_results.pkl）'
    )

    args = parser.parse_args()

    # 检查图谱文件是否存在（viz模式不需要）
    if args.mode != 'viz' and not os.path.exists(args.graph_path):
        print(f"错误: 找不到图谱文件 {args.graph_path}")
        return

    # 创建分析器
    if args.mode == 'viz':
        # 仅可视化模式：只需要output_dir，不加载图谱
        print("=" * 80)
        print("运行模式: 仅可视化（使用缓存结果）")
        print("=" * 80)
        # 创建一个临时的分析器实例用于可视化
        class VizOnlyAnalyzer:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = VizOnlyAnalyzer(args.output_dir)
        # 手动添加方法
        analyzer.load_analysis_results = lambda filename: SuperalloyRelationshipAnalyzer.load_analysis_results(analyzer, filename)
        analyzer.visualize_relationship_matrix = lambda results: SuperalloyRelationshipAnalyzer.visualize_relationship_matrix(analyzer, results)
        analyzer.visualize_relationship_types_distribution = lambda results: SuperalloyRelationshipAnalyzer.visualize_relationship_types_distribution(analyzer, results)

        # 加载缓存结果
        try:
            results = analyzer.load_analysis_results(args.cache_file)
        except FileNotFoundError as e:
            print(f"\n错误: {e}")
            print("请先运行完整分析模式生成缓存文件:")
            print(f"  python ontology.py --mode full")
            return

        # 只进行可视化
        analyzer.visualize_relationship_matrix(results)
        analyzer.visualize_relationship_types_distribution(results)
        print("\n可视化完成!")

    else:
        # 分析模式或完整模式
        print("=" * 80)
        if args.mode == 'full':
            print("运行模式: 完整流程（分析 + 可视化 + 保存）")
        else:
            print("运行模式: 仅分析并保存结果")
        print("=" * 80)

        analyzer = SuperalloyRelationshipAnalyzer(args.graph_path, args.output_dir)

        # 分析两两关系
        print("\n开始分析...")
        results = analyzer.analyze_pairwise_relationships()

        # 打印摘要
        analyzer.print_summary(results)

        # 保存详细结果到Excel
        analyzer.save_results_to_excel(results)

        # 保存分析结果供后续使用
        analyzer.save_analysis_results(results, args.cache_file)

        # 如果是完整模式，进行可视化
        if args.mode == 'full':
            print("\n开始可视化...")
            analyzer.visualize_relationship_matrix(results)
            analyzer.visualize_relationship_types_distribution(results)

        print("\n分析完成!")


if __name__ == "__main__":
    main()

