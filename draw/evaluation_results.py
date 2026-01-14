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
评测结果可视化模块
用于绘制baseline对比实验的评测结果
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
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


class EvaluationVisualizer:
    """评测结果可视化类"""

    def __init__(self, baseline_file: str, output_dir: str = "../visualizations"):
        """
        初始化可视化器

        Args:
            baseline_file: baseline对比JSON文件路径
            output_dir: 输出目录
        """
        self.baseline_file = Path(baseline_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.data = self._load_data()
        self.summary = self.data.get('summary', {})
        self.details = self.data.get('details', {})

    def _load_data(self) -> Dict:
        """加载baseline数据"""
        with open(self.baseline_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_overall_scores(self, save_name: str = "overall_scores.png"):
        """绘制各模型的总体平均分数对比"""
        models = []
        scores = []

        for model, stats in self.summary.items():
            models.append(model)
            scores.append(stats['avg_score'])

        # 按分数排序
        sorted_indices = np.argsort(scores)[::-1]
        models = [models[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), scores, color='steelblue', alpha=0.8)

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=18)

        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_comparison(self, save_name: str = "difficulty_comparison.png"):
        """绘制各模型在不同难度级别的表现对比"""
        difficulties = ['L1', 'L2', 'L3', 'L4']
        models = list(self.summary.keys())

        # 准备数据
        scores_by_difficulty = {diff: [] for diff in difficulties}

        for model in models:
            by_diff = self.summary[model]['by_difficulty']
            for diff in difficulties:
                scores_by_difficulty[diff].append(by_diff.get(diff, 0))

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(models))
        width = 0.2

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (diff, color) in enumerate(zip(difficulties, colors)):
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, scores_by_difficulty[diff], width,
                         label=diff, color=color, alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_heatmap(self, save_name: str = "difficulty_heatmap.png"):
        """绘制难度级别表现热力图"""
        difficulties = ['L1', 'L2', 'L3', 'L4']
        models = list(self.summary.keys())

        # 准备数据矩阵
        data_matrix = []
        for model in models:
            by_diff = self.summary[model]['by_difficulty']
            row = [by_diff.get(diff, 0) for diff in difficulties]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # 设置刻度
        ax.set_xticks(np.arange(len(difficulties)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(difficulties)
        ax.set_yticklabels(models)

        # 添加数值标签
        for i in range(len(models)):
            for j in range(len(difficulties)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=16)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=30)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Model')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_score_distribution(self, save_name: str = "score_distribution.png"):
        """绘制各模型的分数分布箱线图"""
        models = list(self.details.keys())

        # 收集每个模型的所有分数
        all_scores = []
        for model in models:
            scores = [item['scores']['overall_score'] for item in self.details[model]]
            all_scores.append(scores)

        # 绘图
        fig, ax = plt.subplots(figsize=(14, 7))

        bp = ax.boxplot(all_scores, labels=models, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

        ax.set_ylabel('Overall Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_score_components(self, save_name: str = "score_components.png"):
        """绘制各模型的分数组成（关键词分数 vs 语义分数）"""
        models = list(self.details.keys())

        # 收集数据
        keyword_scores = []
        semantic_scores = []

        for model in models:
            # 只收集有keyword_scores和semantic_score的项
            kw_scores = [item['scores']['keyword_scores']['weighted_f1']
                        for item in self.details[model]
                        if 'keyword_scores' in item['scores']]
            sem_scores = [item['scores']['semantic_score']
                         for item in self.details[model]
                         if 'semantic_score' in item['scores']]
            keyword_scores.append(np.mean(kw_scores) if kw_scores else 0)
            semantic_scores.append(np.mean(sem_scores) if sem_scores else 0)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, keyword_scores, width, label='Keyword Score',
                      color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, semantic_scores, width, label='Semantic Score',
                      color='#ff7f0e', alpha=0.8)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=16)

        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper left', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_score_scatter(self, save_name: str = "difficulty_score_scatter.png"):
        """绘制难度与分数的散点图"""
        difficulty_map = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4}

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, (model, color) in enumerate(zip(self.details.keys(), colors)):
            difficulties = []
            scores = []

            for item in self.details[model]:
                diff = item['difficulty']
                score = item['scores']['overall_score']
                difficulties.append(difficulty_map[diff])
                scores.append(score)

            ax.scatter(difficulties, scores, label=model, alpha=0.6,
                      s=50, color=color)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Overall Score')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
        ax.set_ylim(0, 1.0)
        ax.legend(loc='best', frameon=True, fontsize=16)
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def generate_all_plots(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")

        self.plot_overall_scores()
        print("✓ 总体分数对比图")

        self.plot_difficulty_comparison()
        print("✓ 难度级别对比图")

        self.plot_difficulty_heatmap()
        print("✓ 难度表现热力图")

        self.plot_score_distribution()
        print("✓ 分数分布箱线图")

        self.plot_score_components()
        print("✓ 分数组成对比图")

        self.plot_difficulty_score_scatter()
        print("✓ 难度-分数散点图")

        print(f"\n所有图表已保存至: {self.output_dir}")


def main():
    """主函数"""
    # 设置文件路径
    baseline_file = "D:/Pycharm/Projects/SuperalloyKgRAG/data/reports/baseline/baseline_comparison_20251229_014021.json"
    output_dir = "D:/Pycharm/Projects/SuperalloyKgRAG/visualizations/evaluation"

    # 创建可视化器并生成图表
    visualizer = EvaluationVisualizer(baseline_file, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()

