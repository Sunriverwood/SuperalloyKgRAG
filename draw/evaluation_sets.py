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
评测集题目类型可视化模块
用于分析和展示评测集中题目的类型、难度、领域等分布情况
"""

import json
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import rcParams

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use(['science', 'no-latex'])

# 全局字体设置
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20


class EvaluationSetVisualizer:
    """评测集题目类型可视化类"""

    def __init__(self, report_file: str, output_dir: str = "../visualizations/evaluation_sets",
                 eval_sets_dir: str = None):
        """
        初始化可视化器

        Args:
            report_file: 评测报告JSON文件路径
            output_dir: 输出目录
            eval_sets_dir: 评测集数据目录（用于生成交叉统计热力图）
        """
        self.report_file = Path(report_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 评测集数据目录
        if eval_sets_dir:
            self.eval_sets_dir = Path(eval_sets_dir)
        else:
            # 默认使用相对于报告文件的路径
            self.eval_sets_dir = self.report_file.parent.parent / "evaluation_sets"

        # 加载数据
        self.data = self._load_data()
        self.overall_stats = self.data.get('overall_statistics', {})
        self.by_difficulty = self.data.get('by_difficulty', {})
        self.by_domain = self.data.get('by_domain', {})
        self.by_type = self.data.get('by_type', {})

        # 加载题目详细数据（用于交叉统计）
        self.questions = self._load_questions()

    def _load_data(self) -> Dict:
        """加载评测报告数据"""
        with open(self.report_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_questions(self) -> List[Dict]:
        """加载所有题目的详细数据"""
        questions = []

        # 读取所有评测集文件
        if self.eval_sets_dir.exists():
            for json_file in self.eval_sets_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            questions.extend(data)
                except Exception as e:
                    print(f"警告: 读取文件 {json_file} 失败: {e}")

        return questions

    def plot_difficulty_distribution(self, save_name: str = "difficulty_distribution.png"):
        """绘制难度级别分布"""
        difficulties = []
        counts = []

        # 按L1, L2, L3, L4排序
        difficulty_order = ['L1', 'L2', 'L3', 'L4']
        for diff in difficulty_order:
            if diff in self.by_difficulty:
                difficulties.append(diff)
                counts.append(self.by_difficulty[diff]['count'])

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(difficulties, counts, color=colors, alpha=0.8)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{count}', ha='center', va='bottom', fontsize=18)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Number of Questions')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_type_distribution(self, save_name: str = "type_distribution.png"):
        """绘制题目类型分布"""
        types = list(self.by_type.keys())
        counts = [self.by_type[t]['count'] for t in types]

        # 按数量排序
        sorted_indices = np.argsort(counts)[::-1]
        types = [types[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(types)), counts, color='steelblue', alpha=0.8)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{count}', ha='center', va='bottom', fontsize=18)

        ax.set_xlabel('Question Type')
        ax.set_ylabel('Number of Questions')
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_domain_distribution(self, save_name: str = "domain_distribution.png"):
        """绘制领域分布"""
        domains = list(self.by_domain.keys())
        counts = [self.by_domain[d]['count'] for d in domains]

        # 按数量排序
        sorted_indices = np.argsort(counts)[::-1]
        domains = [domains[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(len(domains)), counts, color='coral', alpha=0.8)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{count}', ha='center', va='bottom', fontsize=18)

        ax.set_xlabel('Domain')
        ax.set_ylabel('Number of Questions')
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_type_pie_chart(self, save_name: str = "type_pie_chart.png"):
        """绘制题目类型饼图"""
        types = list(self.by_type.keys())
        counts = [self.by_type[t]['count'] for t in types]

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 18})

        # 设置百分比文字颜色和大小
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(18)
            autotext.set_weight('bold')

        ax.axis('equal')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_pie_chart(self, save_name: str = "difficulty_pie_chart.png"):
        """绘制难度级别饼图"""
        difficulty_order = ['L1', 'L2', 'L3', 'L4']
        difficulties = []
        counts = []

        for diff in difficulty_order:
            if diff in self.by_difficulty:
                difficulties.append(diff)
                counts.append(self.by_difficulty[diff]['count'])

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        wedges, texts, autotexts = ax.pie(counts, labels=difficulties, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 20})

        # 设置百分比文字颜色和大小
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(20)
            autotext.set_weight('bold')

        ax.axis('equal')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_score_by_difficulty(self, save_name: str = "score_by_difficulty.png"):
        """绘制不同难度的平均得分"""
        difficulty_order = ['L1', 'L2', 'L3', 'L4']
        difficulties = []
        avg_scores = []

        for diff in difficulty_order:
            if diff in self.by_difficulty:
                difficulties.append(diff)
                avg_scores.append(self.by_difficulty[diff]['avg_score'])

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(difficulties, avg_scores, color=colors, alpha=0.8)

        # 添加数值标签
        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=18)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_score_by_type(self, save_name: str = "score_by_type.png"):
        """绘制不同题目类型的平均得分"""
        types = list(self.by_type.keys())
        avg_scores = [self.by_type[t]['avg_score'] for t in types]

        # 按得分排序
        sorted_indices = np.argsort(avg_scores)[::-1]
        types = [types[i] for i in sorted_indices]
        avg_scores = [avg_scores[i] for i in sorted_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(types)), avg_scores, color='steelblue', alpha=0.8)

        # 添加数值标签
        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=18)

        ax.set_xlabel('Question Type')
        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_score_by_domain(self, save_name: str = "score_by_domain.png"):
        """绘制不同领域的平均得分"""
        domains = list(self.by_domain.keys())
        avg_scores = [self.by_domain[d]['avg_score'] for d in domains]

        # 按得分排序
        sorted_indices = np.argsort(avg_scores)[::-1]
        domains = [domains[i] for i in sorted_indices]
        avg_scores = [avg_scores[i] for i in sorted_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(len(domains)), avg_scores, color='coral', alpha=0.8)

        # 添加数值标签
        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=18)

        ax.set_xlabel('Domain')
        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_combined_distribution(self, save_name: str = "combined_distribution.png"):
        """绘制综合分布图（2x2子图）"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 难度分布
        ax = axes[0, 0]
        difficulty_order = ['L1', 'L2', 'L3', 'L4']
        difficulties = []
        counts = []
        for diff in difficulty_order:
            if diff in self.by_difficulty:
                difficulties.append(diff)
                counts.append(self.by_difficulty[diff]['count'])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(difficulties, counts, color=colors, alpha=0.8)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{count}', ha='center', va='bottom', fontsize=18)
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Number of Questions')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 2. 题目类型分布
        ax = axes[0, 1]
        types = list(self.by_type.keys())
        counts = [self.by_type[t]['count'] for t in types]
        sorted_indices = np.argsort(counts)[::-1]
        types = [types[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        bars = ax.bar(range(len(types)), counts, color='steelblue', alpha=0.8)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{count}', ha='center', va='bottom', fontsize=16)
        ax.set_xlabel('Question Type')
        ax.set_ylabel('Number of Questions')
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 3. 难度得分
        ax = axes[1, 0]
        difficulties = []
        avg_scores = []
        for diff in difficulty_order:
            if diff in self.by_difficulty:
                difficulties.append(diff)
                avg_scores.append(self.by_difficulty[diff]['avg_score'])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(difficulties, avg_scores, color=colors, alpha=0.8)
        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=18)
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 4. 题目类型得分
        ax = axes[1, 1]
        types = list(self.by_type.keys())
        avg_scores = [self.by_type[t]['avg_score'] for t in types]
        sorted_indices = np.argsort(avg_scores)[::-1]
        types = [types[i] for i in sorted_indices]
        avg_scores = [avg_scores[i] for i in sorted_indices]
        bars = ax.bar(range(len(types)), avg_scores, color='steelblue', alpha=0.8)
        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=16)
        ax.set_xlabel('Question Type')
        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_type_heatmap(self, save_name: str = "difficulty_type_heatmap.png"):
        """绘制难度-类型交叉统计热力图"""
        if not self.questions:
            print("警告: 未找到题目详细数据，跳过难度-类型热力图生成")
            return

        # 收集所有类型和难度
        difficulty_order = ['L1', 'L2', 'L3', 'L4']
        types = sorted(list(set(q['type'] for q in self.questions)))

        # 构建交叉统计矩阵
        data_matrix = np.zeros((len(types), len(difficulty_order)))

        for question in self.questions:
            q_type = question['type']
            q_diff = question['difficulty']
            if q_type in types and q_diff in difficulty_order:
                type_idx = types.index(q_type)
                diff_idx = difficulty_order.index(q_diff)
                data_matrix[type_idx, diff_idx] += 1

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

        # 设置刻度
        ax.set_xticks(np.arange(len(difficulty_order)))
        ax.set_yticks(np.arange(len(types)))
        ax.set_xticklabels(difficulty_order)
        ax.set_yticklabels(types)

        # 添加数值标签
        for i in range(len(types)):
            for j in range(len(difficulty_order)):
                count = int(data_matrix[i, j])
                text = ax.text(j, i, str(count),
                             ha="center", va="center",
                             color="white" if count > data_matrix.max()/2 else "black",
                             fontsize=18)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Questions', rotation=270, labelpad=30)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Question Type')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_domain_heatmap(self, save_name: str = "difficulty_domain_heatmap.png"):
        """绘制难度-领域交叉统计热力图"""
        if not self.questions:
            print("警告: 未找到题目详细数据，跳过难度-领域热力图生成")
            return

        # 收集所有领域和难度
        difficulty_order = ['L1', 'L2', 'L3', 'L4']
        domains = sorted(list(set(q['domain'] for q in self.questions)))

        # 构建交叉统计矩阵
        data_matrix = np.zeros((len(domains), len(difficulty_order)))

        for question in self.questions:
            q_domain = question['domain']
            q_diff = question['difficulty']
            if q_domain in domains and q_diff in difficulty_order:
                domain_idx = domains.index(q_domain)
                diff_idx = difficulty_order.index(q_diff)
                data_matrix[domain_idx, diff_idx] += 1

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(data_matrix, cmap='YlGnBu', aspect='auto')

        # 设置刻度
        ax.set_xticks(np.arange(len(difficulty_order)))
        ax.set_yticks(np.arange(len(domains)))
        ax.set_xticklabels(difficulty_order)
        ax.set_yticklabels(domains)

        # 添加数值标签
        for i in range(len(domains)):
            for j in range(len(difficulty_order)):
                count = int(data_matrix[i, j])
                text = ax.text(j, i, str(count),
                             ha="center", va="center",
                             color="white" if count > data_matrix.max()/2 else "black",
                             fontsize=18)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Questions', rotation=270, labelpad=30)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Domain')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def generate_all_plots(self):
        """生成所有可视化图表"""
        print("开始生成评测集可视化图表...")
        print(f"总题目数: {self.overall_stats.get('total_questions', 0)}")
        print(f"平均得分: {self.overall_stats.get('avg_score', 0):.4f}")
        print()

        self.plot_difficulty_distribution()
        print("✓ 难度级别分布图")

        self.plot_type_distribution()
        print("✓ 题目类型分布图")

        self.plot_domain_distribution()
        print("✓ 领域分布图")

        self.plot_type_pie_chart()
        print("✓ 题目类型饼图")

        self.plot_difficulty_pie_chart()
        print("✓ 难度级别饼图")

        self.plot_score_by_difficulty()
        print("✓ 难度级别平均得分图")

        self.plot_score_by_type()
        print("✓ 题目类型平均得分图")

        self.plot_score_by_domain()
        print("✓ 领域平均得分图")

        self.plot_combined_distribution()
        print("✓ 综合分布图")

        self.plot_difficulty_type_heatmap()
        print("✓ 难度-题目类型热力图")

        self.plot_difficulty_domain_heatmap()
        print("✓ 难度-领域热力图")

        print(f"\n所有图表已保存至: {self.output_dir}")


def main():
    """主函数"""
    # 设置文件路径
    report_file = "D:/Pycharm/Projects/SuperalloyKgRAG/data/reports/evaluation_report_20251229_113957_local.json"
    output_dir = "D:/Pycharm/Projects/SuperalloyKgRAG/visualizations/evaluation_sets"

    # 创建可视化器并生成图表
    visualizer = EvaluationSetVisualizer(report_file, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()

