"""
实体合并人工审核模块
用于随机抽样候选簇进行人工审核，并对比大模型合并效果与人工合并效果
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class ClusterReviewResult:
    """单个簇的审核结果"""
    cluster_id: int
    member_ids: List[str]
    member_names: List[str]
    member_descriptions: List[str]
    # 人工决策
    human_decision: str  # "merge" or "keep_separate"
    human_canonical_name: str = ""
    human_rationale: str = ""
    # LLM决策
    llm_decision: str = ""
    llm_canonical_name: str = ""
    llm_rationale: str = ""
    # 对比结果
    agreement: bool = False


class EntityMergeReviewer:
    """实体合并审核器"""

    def __init__(self, graph: nx.DiGraph, clusters: List[List[str]], sample_size: int = 5):
        """
        初始化审核器

        Args:
            graph: NetworkX图对象
            clusters: 候选合并簇列表
            sample_size: 抽样数量
        """
        self.graph = graph
        self.clusters = clusters
        self.sample_size = min(sample_size, len(clusters))
        self.review_results: List[ClusterReviewResult] = []

        # 设置中文字体
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """设置matplotlib中文字体"""
        try:
            # 尝试使用常见的中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.warning(f"设置中文字体失败: {e}")

    def sample_clusters(self) -> List[Tuple[int, List[str]]]:
        """
        随机抽样候选簇

        Returns:
            抽样的簇列表，每个元素为 (cluster_id, member_ids)
        """
        if not self.clusters:
            logging.warning("没有候选簇可供抽样")
            return []

        # 随机抽样
        sampled_indices = random.sample(range(len(self.clusters)), self.sample_size)
        sampled_clusters = [(idx, self.clusters[idx]) for idx in sampled_indices]

        logging.info(f"已从 {len(self.clusters)} 个候选簇中随机抽样 {len(sampled_clusters)} 个")
        return sampled_clusters

    def _get_entity_info(self, entity_id: str) -> Dict[str, Any]:
        """获取实体信息"""
        if not self.graph.has_node(entity_id):
            return {
                "id": entity_id,
                "name": "未找到",
                "description": "节点不存在于图中",
                "type": "未知"
            }

        node = self.graph.nodes[entity_id]
        return {
            "id": entity_id,
            "name": node.get("name", ""),
            "description": node.get("description", ""),
            "type": node.get("type", ""),
            "aliases": node.get("aliases", [])
        }

    def display_cluster(self, cluster_id: int, member_ids: List[str]) -> None:
        """
        展示候选簇信息供人工审核

        Args:
            cluster_id: 簇ID
            member_ids: 成员实体ID列表
        """
        print("\n" + "=" * 80)
        print(f"候选簇 #{cluster_id}")
        print("=" * 80)

        for i, entity_id in enumerate(member_ids, 1):
            info = self._get_entity_info(entity_id)
            print(f"\n实体 {i}:")
            print(f"  ID: {info['id']}")
            print(f"  名称: {info['name']}")
            print(f"  类型: {info['type']}")
            print(f"  描述: {info['description'][:200]}..." if len(info['description']) > 200 else f"  描述: {info['description']}")
            if info.get('aliases'):
                print(f"  别名: {', '.join(info['aliases'][:5])}")

        print("\n" + "-" * 80)

    def collect_human_decision(self, cluster_id: int, member_ids: List[str]) -> ClusterReviewResult:
        """
        收集人工审核决策

        Args:
            cluster_id: 簇ID
            member_ids: 成员实体ID列表

        Returns:
            审核结果
        """
        # 展示簇信息
        self.display_cluster(cluster_id, member_ids)

        # 收集实体信息
        member_names = []
        member_descriptions = []
        for eid in member_ids:
            info = self._get_entity_info(eid)
            member_names.append(info['name'])
            member_descriptions.append(info['description'])

        # 询问是否合并
        print("\n请判断这些实体是否应该合并:")
        print("1. 合并 (这些实体指向同一概念)")
        print("2. 保持分离 (这些实体是不同的概念)")

        while True:
            choice = input("\n请输入选择 (1/2): ").strip()
            if choice in ['1', '2']:
                break
            print("无效输入，请重新输入")

        result = ClusterReviewResult(
            cluster_id=cluster_id,
            member_ids=member_ids,
            member_names=member_names,
            member_descriptions=member_descriptions,
            human_decision="merge" if choice == '1' else "keep_separate"
        )

        # 如果选择合并，收集规范名称和理由
        if result.human_decision == "merge":
            canonical_name = input("请输入合并后的规范名称: ").strip()
            rationale = input("请简要说明合并理由: ").strip()
            result.human_canonical_name = canonical_name
            result.human_rationale = rationale
        else:
            rationale = input("请简要说明保持分离的理由: ").strip()
            result.human_rationale = rationale

        return result

    def run_manual_review(self) -> List[ClusterReviewResult]:
        """
        运行人工审核流程

        Returns:
            审核结果列表
        """
        print("\n" + "=" * 80)
        print("开始实体合并人工审核")
        print("=" * 80)

        sampled_clusters = self.sample_clusters()

        for idx, (cluster_id, member_ids) in enumerate(sampled_clusters, 1):
            print(f"\n进度: {idx}/{len(sampled_clusters)}")
            result = self.collect_human_decision(cluster_id, member_ids)
            self.review_results.append(result)

        logging.info(f"人工审核完成，共审核 {len(self.review_results)} 个候选簇")
        return self.review_results

    def compare_with_llm(self, llm_groups: List[Any]) -> None:
        """
        对比LLM决策与人工决策

        Args:
            llm_groups: LLM返回的合并分组 (LLMResolutionGroup列表)
        """
        if not self.review_results:
            logging.warning("没有人工审核结果可供对比")
            return

        # 将LLM结果转换为查找字典
        llm_decisions = {}
        for group in llm_groups:
            # 对于LLM合并的每个组，记录其成员ID
            member_set = frozenset(group.member_ids)
            llm_decisions[member_set] = {
                "decision": "merge",
                "canonical_name": group.canonical_name,
                "rationale": group.rationale or ""
            }

        # 对比每个审核结果
        for result in self.review_results:
            member_set = frozenset(result.member_ids)

            # 查找LLM对该簇的决策
            if member_set in llm_decisions:
                llm_decision = llm_decisions[member_set]
                result.llm_decision = llm_decision["decision"]
                result.llm_canonical_name = llm_decision["canonical_name"]
                result.llm_rationale = llm_decision["rationale"]
            else:
                # LLM未合并该簇
                result.llm_decision = "keep_separate"
                result.llm_canonical_name = ""
                result.llm_rationale = "LLM未将此簇标记为需要合并"

            # 判断是否一致
            result.agreement = (result.human_decision == result.llm_decision)

        logging.info("LLM决策对比完成")

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        生成对比报告

        Returns:
            包含统计信息的报告字典
        """
        if not self.review_results:
            return {"error": "没有审核结果"}

        total = len(self.review_results)
        agreements = sum(1 for r in self.review_results if r.agreement)
        disagreements = total - agreements

        # 统计决策分布
        human_merge = sum(1 for r in self.review_results if r.human_decision == "merge")
        llm_merge = sum(1 for r in self.review_results if r.llm_decision == "merge")

        # 详细对比
        details = []
        for result in self.review_results:
            details.append({
                "cluster_id": result.cluster_id,
                "member_count": len(result.member_ids),
                "member_names": result.member_names,
                "human_decision": result.human_decision,
                "human_canonical_name": result.human_canonical_name,
                "human_rationale": result.human_rationale,
                "llm_decision": result.llm_decision,
                "llm_canonical_name": result.llm_canonical_name,
                "llm_rationale": result.llm_rationale,
                "agreement": result.agreement
            })

        report = {
            "summary": {
                "total_reviewed": total,
                "agreements": agreements,
                "disagreements": disagreements,
                "agreement_rate": agreements / total if total > 0 else 0,
                "human_merge_count": human_merge,
                "llm_merge_count": llm_merge,
                "human_merge_rate": human_merge / total if total > 0 else 0,
                "llm_merge_rate": llm_merge / total if total > 0 else 0
            },
            "details": details
        }

        return report

    def save_report(self, output_path: Path) -> None:
        """
        保存对比报告到文件

        Args:
            output_path: 输出文件路径
        """
        report = self.generate_comparison_report()

        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info(f"对比报告已保存到: {output_path}")

    def visualize_comparison(self, output_path: Path = None) -> None:
        """
        可视化对比结果

        Args:
            output_path: 图片保存路径，如果为None则显示图片
        """
        if not self.review_results:
            logging.warning("没有审核结果可供可视化")
            return

        report = self.generate_comparison_report()
        summary = report['summary']

        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 图1: 一致性对比
        ax1 = axes[0]
        labels = ['一致', '不一致']
        sizes = [summary['agreements'], summary['disagreements']]
        colors = ['#66b3ff', '#ff9999']
        explode = (0.1, 0)

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title(f'人工审核与LLM决策一致性\n(总样本: {summary["total_reviewed"]})',
                     fontsize=14, fontweight='bold')

        # 图2: 决策对比
        ax2 = axes[1]
        categories = ['人工决策', 'LLM决策']
        merge_counts = [summary['human_merge_count'], summary['llm_merge_count']]
        keep_counts = [
            summary['total_reviewed'] - summary['human_merge_count'],
            summary['total_reviewed'] - summary['llm_merge_count']
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax2.bar(x - width/2, merge_counts, width, label='合并', color='#66b3ff')
        bars2 = ax2.bar(x + width/2, keep_counts, width, label='保持分离', color='#ff9999')

        ax2.set_ylabel('数量', fontsize=12)
        ax2.set_title('决策分布对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

        # 图3: 合并率对比
        ax3 = axes[2]
        merge_rates = [summary['human_merge_rate'] * 100, summary['llm_merge_rate'] * 100]
        bars = ax3.bar(categories, merge_rates, color=['#66b3ff', '#99ff99'], width=0.6)

        ax3.set_ylabel('合并率 (%)', fontsize=12)
        ax3.set_title('合并率对比', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"可视化结果已保存到: {output_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self) -> None:
        """打印对比摘要"""
        if not self.review_results:
            print("没有审核结果")
            return

        report = self.generate_comparison_report()
        summary = report['summary']

        print("\n" + "=" * 80)
        print("实体合并审核对比摘要")
        print("=" * 80)
        print(f"\n总审核样本数: {summary['total_reviewed']}")
        print(f"一致数量: {summary['agreements']}")
        print(f"不一致数量: {summary['disagreements']}")
        print(f"一致率: {summary['agreement_rate']:.2%}")
        print(f"\n人工决策:")
        print(f"  合并: {summary['human_merge_count']} ({summary['human_merge_rate']:.2%})")
        print(f"  保持分离: {summary['total_reviewed'] - summary['human_merge_count']}")
        print(f"\nLLM决策:")
        print(f"  合并: {summary['llm_merge_count']} ({summary['llm_merge_rate']:.2%})")
        print(f"  保持分离: {summary['total_reviewed'] - summary['llm_merge_count']}")

        # 打印不一致的案例
        disagreements = [r for r in self.review_results if not r.agreement]
        if disagreements:
            print(f"\n不一致案例详情 ({len(disagreements)} 个):")
            print("-" * 80)
            for i, result in enumerate(disagreements, 1):
                print(f"\n案例 {i} (簇 #{result.cluster_id}):")
                print(f"  成员: {', '.join(result.member_names)}")
                print(f"  人工决策: {result.human_decision}")
                if result.human_canonical_name:
                    print(f"    规范名称: {result.human_canonical_name}")
                print(f"    理由: {result.human_rationale}")
                print(f"  LLM决策: {result.llm_decision}")
                if result.llm_canonical_name:
                    print(f"    规范名称: {result.llm_canonical_name}")
                print(f"    理由: {result.llm_rationale}")

        print("\n" + "=" * 80)


def run_entity_merge_review(graph: nx.DiGraph, clusters: List[List[str]],
                            llm_groups: List[Any],
                            sample_size: int = 5,
                            output_dir: Path = None) -> Dict[str, Any]:
    """
    运行实体合并人工审核流程

    Args:
        graph: NetworkX图对象
        clusters: 候选合并簇列表
        llm_groups: LLM返回的合并分组
        sample_size: 抽样数量
        output_dir: 输出目录

    Returns:
        对比报告字典
    """
    if not clusters:
        logging.warning("没有候选簇，跳过人工审核")
        return {}

    # 创建审核器
    reviewer = EntityMergeReviewer(graph, clusters, sample_size)

    # 运行人工审核
    reviewer.run_manual_review()

    # 对比LLM决策
    reviewer.compare_with_llm(llm_groups)

    # 打印摘要
    reviewer.print_summary()

    # 保存报告和可视化
    if output_dir:
        output_dir = Path(output_dir)
        reviewer.save_report(output_dir / "entity_merge_review_report.json")
        reviewer.visualize_comparison(output_dir / "entity_merge_review_comparison.png")

    return reviewer.generate_comparison_report()

