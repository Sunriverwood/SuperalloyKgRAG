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
Baseline vs 方法对比可视化模块
用于对比baseline模型和不同查询方法的评测结果
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


class BaselineVsMethodsVisualizer:
    """Baseline vs 方法对比可视化类"""

    def __init__(self, baseline_file: str, reports_dir: str, output_dir: str = "../visualizations/baseline_comparison"):
        """
        初始化可视化器

        Args:
            baseline_file: baseline对比JSON文件路径
            reports_dir: 评测报告文件所在目录
            output_dir: 输出目录
        """
        self.baseline_file = Path(baseline_file)
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.baseline_data = self._load_baseline()
        self.method_reports = self._load_method_reports()

    def _load_baseline(self) -> Dict:
        """加载baseline数据"""
        with open(self.baseline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('summary', {})

    def _load_method_reports(self) -> Dict:
        """加载所有方法的评测报告"""
        reports = {}

        # 查找所有evaluation_report_数字_数字_标签.json文件
        for json_file in self.reports_dir.glob("evaluation_report_*_*_*.json"):
            # 从文件名提取方法标签
            filename = json_file.stem  # evaluation_report_20251229_113957_local
            parts = filename.split('_')
            if len(parts) >= 4:
                method_name = '_'.join(parts[4:])  # local, global, drift, etc.

                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reports[method_name] = data

        return reports

    def plot_overall_comparison(self, save_name: str = "overall_comparison.png"):
        """绘制整体平均分对比图（baseline模型 + 各方法）"""
        # 收集baseline模型数据
        baseline_names = []
        baseline_scores = []

        for model, stats in self.baseline_data.items():
            baseline_names.append(model)
            baseline_scores.append(stats['avg_score'])

        # 收集各方法数据
        method_names = []
        method_scores = []

        for method, data in self.method_reports.items():
            method_names.append(method.upper())
            method_scores.append(data['overall_statistics']['avg_score'])

        # 组合所有数据
        all_names = baseline_names + method_names
        all_scores = baseline_scores + method_scores

        # 按分数排序
        sorted_indices = np.argsort(all_scores)[::-1]
        all_names = [all_names[i] for i in sorted_indices]
        all_scores = [all_scores[i] for i in sorted_indices]

        # 区分baseline和方法
        colors = []
        for name in all_names:
            if name in baseline_names:
                colors.append('#1f77b4')  # 蓝色 - baseline模型
            else:
                colors.append('#ff7f0e')  # 橙色 - 方法

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 7))
        bars = ax.bar(range(len(all_names)), all_scores, color=colors, alpha=0.8)

        # 添加数值标签
        for bar, score in zip(bars, all_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=16)

        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.8, label='Baseline Models'),
            Patch(facecolor='#ff7f0e', alpha=0.8, label='Our Methods')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_comparison(self, save_name: str = "difficulty_comparison.png"):
        """绘制按难度级别的对比图"""
        difficulties = ['L1', 'L2', 'L3', 'L4']

        # 收集baseline模型数据
        baseline_data = {}
        for model, stats in self.baseline_data.items():
            baseline_data[model] = [stats['by_difficulty'].get(diff, 0) for diff in difficulties]

        # 收集各方法数据
        method_data = {}
        for method, data in self.method_reports.items():
            method_data[method.upper()] = [
                data['by_difficulty'].get(diff, {}).get('avg_score', 0)
                for diff in difficulties
            ]

        # 合并所有数据
        all_data = {**baseline_data, **method_data}
        all_names = list(all_data.keys())

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 7))

        x = np.arange(len(difficulties))
        n_items = len(all_names)
        width = 0.8 / n_items  # 总宽度0.8，除以项目数

        # 使用不同颜色
        colors = plt.cm.tab20(np.linspace(0, 1, n_items))

        for i, (name, scores) in enumerate(all_data.items()):
            offset = width * (i - n_items / 2 + 0.5)
            bars = ax.bar(x + offset, scores, width, label=name,
                          color=colors[i], alpha=0.8)

        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.set_xlabel('Difficulty Level')
        ax.legend(loc='upper left', frameon=True, ncol=2, fontsize=16)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_difficulty_heatmap(self, save_name: str = "difficulty_heatmap.png"):
        """绘制难度级别表现热力图"""
        difficulties = ['L1', 'L2', 'L3', 'L4']

        # 收集所有数据
        all_names = []
        all_scores = []

        # Baseline模型
        for model, stats in self.baseline_data.items():
            all_names.append(model)
            scores = [stats['by_difficulty'].get(diff, 0) for diff in difficulties]
            all_scores.append(scores)

        # 各方法
        for method, data in self.method_reports.items():
            all_names.append(method.upper())
            scores = [
                data['by_difficulty'].get(diff, {}).get('avg_score', 0)
                for diff in difficulties
            ]
            all_scores.append(scores)

        data_matrix = np.array(all_scores)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 12))

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # 设置刻度
        ax.set_xticks(np.arange(len(difficulties)))
        ax.set_yticks(np.arange(len(all_names)))
        ax.set_xticklabels(difficulties)
        ax.set_yticklabels(all_names)

        # 添加数值标签
        for i in range(len(all_names)):
            for j in range(len(difficulties)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                               ha="center", va="center",
                               color="white" if data_matrix[i, j] > 0.5 else "black",
                               fontsize=16)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Score', rotation=270, labelpad=30)

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Model / Method')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_by_domain_comparison(self, save_name: str = "domain_comparison.png"):
        """绘制按领域的对比图（仅针对有domain数据的方法）"""
        # 只收集有domain数据的方法
        method_data = {}

        for method, data in self.method_reports.items():
            if 'by_domain' in data and data['by_domain']:
                method_data[method.upper()] = data['by_domain']

        if not method_data:
            print("警告: 没有找到包含领域数据的方法，跳过领域对比图")
            return

        # 获取所有领域
        all_domains = set()
        for domains_dict in method_data.values():
            all_domains.update(domains_dict.keys())
        all_domains = sorted(list(all_domains))

        # 准备数据
        method_names = list(method_data.keys())
        scores_by_domain = {domain: [] for domain in all_domains}

        for method in method_names:
            for domain in all_domains:
                score = method_data[method].get(domain, {}).get('avg_score', 0)
                scores_by_domain[domain].append(score)

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(all_domains))
        n_methods = len(method_names)
        width = 0.8 / n_methods

        colors = plt.cm.Set3(np.linspace(0, 1, n_methods))

        for i, method in enumerate(method_names):
            offset = width * (i - n_methods / 2 + 0.5)
            scores = [scores_by_domain[domain][i] for domain in all_domains]
            bars = ax.bar(x + offset, scores, width, label=method,
                          color=colors[i], alpha=0.8)

        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(all_domains, rotation=45, ha='right')
        ax.set_xlabel('Domain')
        ax.legend(loc='upper left', frameon=True, fontsize=18)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_by_type_comparison(self, save_name: str = "type_comparison.png"):
        """绘制按题目类型的对比图（包含baseline模型和方法）"""
        from collections import defaultdict

        # 收集baseline模型的按type数据（从details中统计）
        baseline_data = {}

        # 需要从baseline_file重新加载完整数据（包含details）
        with open(self.baseline_file, 'r', encoding='utf-8') as f:
            full_baseline = json.load(f)

        if 'details' in full_baseline:
            for model, items in full_baseline['details'].items():
                type_scores = defaultdict(list)
                for item in items:
                    if 'type' in item:
                        qtype = item['type']
                        score = item['scores']['overall_score']
                        type_scores[qtype].append(score)

                # 计算每个type的平均分
                baseline_data[model] = {
                    qtype: sum(scores) / len(scores) if scores else 0
                    for qtype, scores in type_scores.items()
                }

        # 收集方法的按type数据
        method_data = {}
        for method, data in self.method_reports.items():
            if 'by_type' in data and data['by_type']:
                method_data[method.upper()] = {
                    qtype: stats['avg_score']
                    for qtype, stats in data['by_type'].items()
                }

        # 合并所有数据
        all_data = {**baseline_data, **method_data}

        if not all_data:
            print("警告: 没有找到包含题目类型数据，跳过类型对比图")
            return

        # 获取所有类型
        all_types = set()
        for type_dict in all_data.values():
            all_types.update(type_dict.keys())
        all_types = sorted(list(all_types))

        # 准备数据
        all_names = list(all_data.keys())
        scores_by_type = {qtype: [] for qtype in all_types}

        for name in all_names:
            for qtype in all_types:
                score = all_data[name].get(qtype, 0)
                scores_by_type[qtype].append(score)

        # 绘图
        fig, ax = plt.subplots(figsize=(16, 7))

        x = np.arange(len(all_types))
        n_items = len(all_names)
        width = 0.8 / n_items

        colors = plt.cm.tab20(np.linspace(0, 1, n_items))

        for i, name in enumerate(all_names):
            offset = width * (i - n_items / 2 + 0.5)
            scores = [scores_by_type[qtype][i] for qtype in all_types]
            bars = ax.bar(x + offset, scores, width, label=name,
                          color=colors[i], alpha=0.8)

        ax.set_ylabel('Average Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=30, ha='right')
        ax.set_xlabel('Question Type')
        ax.legend(loc='upper left', frameon=True, fontsize=16, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def plot_radar_chart(self, save_name: str = "radar_chart.png"):
        """绘制雷达图对比各难度级别的表现"""
        difficulties = ['L1', 'L2', 'L3', 'L4']

        # 收集数据（只选择部分代表性模型/方法）
        selected_items = {}

        # 选择最好的baseline模型
        best_baseline = max(self.baseline_data.items(),
                            key=lambda x: x[1]['avg_score'])
        selected_items[best_baseline[0]] = [
            best_baseline[1]['by_difficulty'].get(diff, 0)
            for diff in difficulties
        ]

        # 选择所有方法
        for method, data in self.method_reports.items():
            selected_items[method.upper()] = [
                data['by_difficulty'].get(diff, {}).get('avg_score', 0)
                for diff in difficulties
            ]

        # 绘图
        angles = np.linspace(0, 2 * np.pi, len(difficulties), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_items)))

        for i, (name, scores) in enumerate(selected_items.items()):
            scores_plot = scores + scores[:1]  # 闭合
            ax.plot(angles, scores_plot, 'o-', linewidth=2, label=name,
                    color=colors[i])
            ax.fill(angles, scores_plot, alpha=0.15, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(difficulties, fontsize=20)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=18)

        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()

    def generate_all_plots(self):
        """生成所有对比可视化图表"""
        print("开始生成Baseline vs 方法对比可视化图表...")
        print(f"Baseline模型数量: {len(self.baseline_data)}")
        print(f"方法数量: {len(self.method_reports)}")
        print()

        self.plot_overall_comparison()
        print("✓ 整体平均分对比图")

        self.plot_difficulty_comparison()
        print("✓ 难度级别对比图")

        self.plot_difficulty_heatmap()
        print("✓ 难度级别热力图")

        self.plot_by_domain_comparison()
        print("✓ 领域对比图")

        self.plot_by_type_comparison()
        print("✓ 题目类型对比图")

        self.plot_radar_chart()
        print("✓ 雷达图")

        print(f"\n所有图表已保存至: {self.output_dir}")


def main():
    """主函数"""
    # 设置文件路径
    baseline_file = "D:/Pycharm/Projects/SuperalloyKgRAG/data/reports/baseline/baseline_comparison_20251229_014021.json"
    reports_dir = "D:/Pycharm/Projects/SuperalloyKgRAG/data/reports"
    output_dir = "D:/Pycharm/Projects/SuperalloyKgRAG/visualizations/baseline_comparison"

    # 创建可视化器并生成图表
    visualizer = BaselineVsMethodsVisualizer(baseline_file, reports_dir, output_dir)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
