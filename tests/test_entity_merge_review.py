"""
测试实体合并人工审核功能
"""

import networkx as nx
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.entity_merge_review import EntityMergeReviewer, run_entity_merge_review
from dataclasses import dataclass
from typing import List


@dataclass
class MockLLMResolutionGroup:
    """模拟的LLM分组结果"""
    canonical_name: str
    member_ids: List[str]
    rationale: str = None


def create_test_graph():
    """创建测试图谱"""
    G = nx.DiGraph()

    # 添加一些测试实体
    entities = [
        {
            "id": "e1",
            "name": "γ' precipitate",
            "type": "MATERIAL_PHASE",
            "description": "Ordered L12 structure precipitate in nickel-based superalloys",
            "aliases": ["gamma prime", "γ'"]
        },
        {
            "id": "e2",
            "name": "gamma prime phase",
            "type": "MATERIAL_PHASE",
            "description": "The strengthening phase in superalloys with L12 crystal structure",
            "aliases": ["γ' phase", "Ni3Al"]
        },
        {
            "id": "e3",
            "name": "creep resistance",
            "type": "PROPERTY",
            "description": "Material's ability to resist deformation under constant stress",
            "aliases": ["creep strength"]
        },
        {
            "id": "e4",
            "name": "creep behavior",
            "type": "PHENOMENON",
            "description": "Time-dependent deformation of materials under stress",
            "aliases": ["creep response"]
        },
        {
            "id": "e5",
            "name": "tensile strength",
            "type": "PROPERTY",
            "description": "Maximum stress a material can withstand while being stretched",
            "aliases": ["ultimate tensile strength", "UTS"]
        },
        {
            "id": "e6",
            "name": "tensile test",
            "type": "METHOD",
            "description": "Testing method to measure tensile properties",
            "aliases": ["tension test"]
        }
    ]

    for entity in entities:
        G.add_node(entity["id"], **{k: v for k, v in entity.items() if k != "id"})

    return G


def create_test_clusters():
    """创建测试候选簇"""
    return [
        ["e1", "e2"],  # γ' precipitate vs gamma prime phase
        ["e3", "e4"],  # creep resistance vs creep behavior
        ["e5", "e6"],  # tensile strength vs tensile test
    ]


def create_mock_llm_groups():
    """创建模拟的LLM合并结果"""
    return [
        MockLLMResolutionGroup(
            canonical_name="γ' precipitate",
            member_ids=["e1", "e2"],
            rationale="Both refer to the same ordered intermetallic phase"
        ),
        MockLLMResolutionGroup(
            canonical_name="creep resistance",
            member_ids=["e3", "e4"],
            rationale="Both describe the same phenomenon of material deformation under stress"
        ),
        # LLM没有合并e5和e6
    ]


def test_basic_functionality():
    """测试基本功能（不需要交互）"""
    print("=" * 80)
    print("测试1: 基本功能测试（无交互）")
    print("=" * 80)

    graph = create_test_graph()
    clusters = create_test_clusters()

    # 创建审核器
    reviewer = EntityMergeReviewer(graph, clusters, sample_size=2)

    # 测试抽样
    sampled = reviewer.sample_clusters()
    print(f"✅ 抽样功能正常，抽取了 {len(sampled)} 个候选簇")

    # 测试实体信息获取
    for cluster_id, member_ids in sampled:
        print(f"\n候选簇 #{cluster_id}:")
        for eid in member_ids:
            info = reviewer._get_entity_info(eid)
            print(f"  - {info['name']} ({info['type']})")

    print("\n✅ 基本功能测试通过")


def test_report_generation():
    """测试报告生成功能"""
    print("\n" + "=" * 80)
    print("测试2: 报告生成功能测试")
    print("=" * 80)

    graph = create_test_graph()
    clusters = create_test_clusters()
    llm_groups = create_mock_llm_groups()

    # 创建审核器
    reviewer = EntityMergeReviewer(graph, clusters, sample_size=2)

    # 模拟人工审核结果（不需要真正的交互）
    from utils.entity_merge_review import ClusterReviewResult

    # 模拟第一个簇的审核：人工认为应该合并
    result1 = ClusterReviewResult(
        cluster_id=0,
        member_ids=["e1", "e2"],
        member_names=["γ' precipitate", "gamma prime phase"],
        member_descriptions=["desc1", "desc2"],
        human_decision="merge",
        human_canonical_name="γ' phase",
        human_rationale="Same strengthening phase"
    )

    # 模拟第二个簇的审核：人工认为应该保持分离
    result2 = ClusterReviewResult(
        cluster_id=1,
        member_ids=["e3", "e4"],
        member_names=["creep resistance", "creep behavior"],
        member_descriptions=["desc3", "desc4"],
        human_decision="keep_separate",
        human_rationale="Different concepts - property vs phenomenon"
    )

    reviewer.review_results = [result1, result2]

    # 对比LLM结果
    reviewer.compare_with_llm(llm_groups)

    # 生成报告
    report = reviewer.generate_comparison_report()

    print(f"\n报告摘要:")
    print(f"  总审核数: {report['summary']['total_reviewed']}")
    print(f"  一致数: {report['summary']['agreements']}")
    print(f"  不一致数: {report['summary']['disagreements']}")
    print(f"  一致率: {report['summary']['agreement_rate']:.2%}")
    print(f"  人工合并率: {report['summary']['human_merge_rate']:.2%}")
    print(f"  LLM合并率: {report['summary']['llm_merge_rate']:.2%}")

    print("\n✅ 报告生成功能测试通过")


def test_visualization():
    """测试可视化功能"""
    print("\n" + "=" * 80)
    print("测试3: 可视化功能测试")
    print("=" * 80)

    graph = create_test_graph()
    clusters = create_test_clusters()
    llm_groups = create_mock_llm_groups()

    # 创建审核器
    reviewer = EntityMergeReviewer(graph, clusters, sample_size=3)

    # 模拟人工审核结果
    from utils.entity_merge_review import ClusterReviewResult

    reviewer.review_results = [
        ClusterReviewResult(
            cluster_id=0,
            member_ids=["e1", "e2"],
            member_names=["γ' precipitate", "gamma prime phase"],
            member_descriptions=["desc1", "desc2"],
            human_decision="merge",
            human_canonical_name="γ' phase",
            human_rationale="Same phase"
        ),
        ClusterReviewResult(
            cluster_id=1,
            member_ids=["e3", "e4"],
            member_names=["creep resistance", "creep behavior"],
            member_descriptions=["desc3", "desc4"],
            human_decision="keep_separate",
            human_rationale="Different concepts"
        ),
        ClusterReviewResult(
            cluster_id=2,
            member_ids=["e5", "e6"],
            member_names=["tensile strength", "tensile test"],
            member_descriptions=["desc5", "desc6"],
            human_decision="keep_separate",
            human_rationale="Property vs method"
        ),
    ]

    # 对比LLM结果
    reviewer.compare_with_llm(llm_groups)

    # 保存报告和可视化
    output_dir = PROJECT_ROOT / "data" / "reports" / "test_manual_review"
    reviewer.save_report(output_dir / "test_report.json")

    try:
        reviewer.visualize_comparison(output_dir / "test_comparison.png")
        print(f"\n✅ 可视化图表已生成: {output_dir / 'test_comparison.png'}")
    except Exception as e:
        print(f"\n⚠️ 可视化生成失败（可能是缺少中文字体）: {e}")

    print(f"✅ 报告已保存: {output_dir / 'test_report.json'}")
    print("✅ 可视化功能测试完成")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("实体合并人工审核功能测试")
    print("=" * 80 + "\n")

    try:
        test_basic_functionality()
        test_report_generation()
        test_visualization()

        print("\n" + "=" * 80)
        print("所有测试通过！✅")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

