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
聚类算法对比测试脚本

功能：
1. 从 merged_graph.json 加载合并后的图
2. 使用多种聚类算法进行社区发现
3. 为每种算法输出带社区标签的图谱（JSON + GEXF 格式）
4. 生成聚类统计摘要（CSV），包括社区数量、模块度等指标

输出目录结构：
data/graphs/clustering/
├── leiden/
│   ├── community_graph.json
│   └── community_graph.gexf
├── louvain/
│   ├── community_graph.json
│   └── community_graph.gexf
├── ...
└── summary.csv

使用方法：
    python run_clustering_comparison.py
    python run_clustering_comparison.py --algorithms leiden,louvain
    python run_clustering_comparison.py --resolution 1.5
    python run_clustering_comparison.py --n-clusters 50
"""

import sys
import json
import math
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from copy import deepcopy
from datetime import datetime

import networkx as nx
import numpy as np

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.community_clustering import (
    CommunityDetector,
    get_available_algorithms
)


# =========================
# 配置与日志
# =========================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志记录器"""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 移除所有现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler()]

    # 如果指定了日志文件，添加文件处理器
    if log_file is None:
        log_path = PROJECT_ROOT / "logs" / "superalloyKgRAG.log"
    else:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_path, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


def clean_value(v):
    """
    把属性值清洗成 GEXF 可接受的简单类型
    复用自 gephi.py
    """
    if v is None:
        return ""
    if isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    # list/dict/tuple/set/其他复杂对象 -> 转成字符串
    try:
        return json.dumps(v, ensure_ascii=False)
    except TypeError:
        return str(v)


# =========================
# 图加载与保存
# =========================

def load_graph(input_path: Path) -> nx.DiGraph:
    """
    加载 NetworkX node-link 格式的 JSON 图
    """
    logging.info(f"正在加载图谱: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"图谱文件不存在: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    graph = nx.node_link_graph(data, edges="links")
    logging.info(f"图谱加载成功: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")

    return graph


def save_graph_json(graph: nx.DiGraph, output_path: Path):
    """
    保存图为 JSON 格式（NetworkX node-link）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = nx.node_link_data(graph, edges="links")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"JSON 图谱已保存: {output_path}")


def save_graph_gexf(graph: nx.DiGraph, output_path: Path):
    """
    保存图为 GEXF 格式（用于 Gephi 可视化）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建适合 GEXF 的图副本（清洗属性）
    directed = isinstance(graph, nx.DiGraph)
    G = nx.DiGraph() if directed else nx.Graph()

    # 复制节点（清洗属性）
    for node, attrs in graph.nodes(data=True):
        clean_attrs = {}
        for k, v in attrs.items():
            clean_attrs[k] = clean_value(v)

        # 确保有 label 属性
        if 'label' not in clean_attrs:
            clean_attrs['label'] = str(attrs.get('name', node))

        G.add_node(node, **clean_attrs)

    # 复制边（清洗属性）
    for src, tgt, attrs in graph.edges(data=True):
        clean_attrs = {}
        for k, v in attrs.items():
            clean_attrs[k] = clean_value(v)
        G.add_edge(src, tgt, **clean_attrs)

    nx.write_gexf(G, str(output_path), encoding="utf-8")
    logging.info(f"GEXF 图谱已保存: {output_path}")


# =========================
# 聚类统计计算
# =========================

def calculate_modularity(graph: nx.Graph, communities: List[List[str]]) -> float:
    """
    计算模块度 (Modularity)
    """
    if not communities:
        return 0.0

    # 转为无向图
    if isinstance(graph, nx.DiGraph):
        UG = graph.to_undirected()
    else:
        UG = graph

    # 构建社区划分（集合列表）
    community_sets = [set(c) for c in communities]

    try:
        modularity = nx.algorithms.community.modularity(UG, community_sets)
        return modularity
    except Exception as e:
        logging.warning(f"模块度计算失败: {e}")
        return 0.0


def calculate_statistics(communities: List[List[str]]) -> Dict[str, Any]:
    """
    计算社区统计信息
    """
    if not communities:
        return {
            "num_communities": 0,
            "min_size": 0,
            "max_size": 0,
            "avg_size": 0.0,
            "median_size": 0.0,
            "std_size": 0.0,
            "singleton_count": 0
        }

    sizes = [len(c) for c in communities]

    return {
        "num_communities": len(communities),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": round(np.mean(sizes), 2),
        "median_size": round(np.median(sizes), 2),
        "std_size": round(np.std(sizes), 2),
        "singleton_count": sum(1 for s in sizes if s == 1)
    }


# =========================
# 聚类执行
# =========================

def run_clustering(
    graph: nx.DiGraph,
    algorithm: str,
    embedding_db_path: Optional[Path] = None,
    embedding_table: str = "entities",
    **params
) -> Dict[str, Any]:
    """
    对指定算法执行聚类并返回结果

    Returns:
        dict: 包含 graph, communities, community_map, stats, modularity
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"开始执行聚类算法: {algorithm.upper()}")
    logging.info(f"{'='*60}")

    # 深拷贝图以避免修改原图
    graph_copy = deepcopy(graph)

    try:
        # 创建社区发现器
        detector = CommunityDetector(
            algorithm=algorithm,
            embedding_db_path=embedding_db_path,
            embedding_table=embedding_table,
            **params
        )

        # 执行社区发现
        community_map, communities = detector.detect(graph_copy)

        # 将社区 ID 写入图节点
        nx.set_node_attributes(graph_copy, community_map, "community")

        # 计算节点度数
        UG = graph_copy.to_undirected()
        for node in graph_copy.nodes():
            graph_copy.nodes[node]["degree"] = UG.degree(node)

        # 计算统计信息
        stats = calculate_statistics(communities)
        modularity = calculate_modularity(graph_copy, communities)

        logging.info(f"✅ {algorithm} 聚类完成:")
        logging.info(f"   - 社区数量: {stats['num_communities']}")
        logging.info(f"   - 最大社区: {stats['max_size']} 节点")
        logging.info(f"   - 最小社区: {stats['min_size']} 节点")
        logging.info(f"   - 平均大小: {stats['avg_size']}")
        logging.info(f"   - 模块度: {modularity:.4f}")

        return {
            "success": True,
            "algorithm": algorithm,
            "graph": graph_copy,
            "communities": communities,
            "community_map": community_map,
            "stats": stats,
            "modularity": modularity,
            "error": None
        }

    except Exception as e:
        logging.error(f"❌ {algorithm} 聚类失败: {e}")
        return {
            "success": False,
            "algorithm": algorithm,
            "graph": None,
            "communities": [],
            "community_map": {},
            "stats": calculate_statistics([]),
            "modularity": 0.0,
            "error": str(e)
        }


def save_summary_csv(results: List[Dict[str, Any]], output_path: Path):
    """
    保存聚类统计摘要为 CSV 文件
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")

    fieldnames = [
        "algorithm",
        "success",
        "num_communities",
        "min_size",
        "max_size",
        "avg_size",
        "median_size",
        "std_size",
        "singleton_count",
        "modularity",
        "error"
    ]

    rows = []
    for r in results:
        row = {
            "algorithm": r["algorithm"],
            "success": r["success"],
            "modularity": round(r["modularity"], 4) if r["modularity"] else 0.0,
            "error": r.get("error", "")
        }
        row.update(r["stats"])
        rows.append(row)

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"统计摘要已保存: {summary_path}")


def print_summary_table(results: List[Dict[str, Any]]):
    """
    打印统计摘要表格
    """
    logging.info("\n" + "=" * 100)
    logging.info("聚类算法对比统计摘要")
    logging.info("=" * 100)

    # 表头
    headers = ["算法", "状态", "社区数", "最小", "最大", "平均", "中位数", "单点社区", "模块度"]
    widths = [20, 8, 8, 8, 8, 8, 8, 10, 10]

    header_line = ""
    for h, w in zip(headers, widths):
        header_line += f"{h:^{w}}"
    logging.info(header_line)
    logging.info("-" * 100)

    # 数据行
    for r in results:
        status = "✅" if r["success"] else "❌"
        stats = r["stats"]

        row = [
            r["algorithm"],
            status,
            str(stats["num_communities"]),
            str(stats["min_size"]),
            str(stats["max_size"]),
            f"{stats['avg_size']:.1f}",
            f"{stats['median_size']:.1f}",
            str(stats["singleton_count"]),
            f"{r['modularity']:.4f}" if r["modularity"] else "N/A"
        ]

        row_line = ""
        for val, w in zip(row, widths):
            row_line += f"{val:^{w}}"
        logging.info(row_line)

    logging.info("=" * 100)

    # 找出模块度最高的算法
    successful = [r for r in results if r["success"] and r["modularity"]]
    if successful:
        best = max(successful, key=lambda x: x["modularity"])
        logging.info(f"\n🏆 模块度最高的算法: {best['algorithm'].upper()} (Modularity = {best['modularity']:.4f})")


# =========================
# 主函数
# =========================

def parse_resolution_sweep(value: Optional[str]) -> Optional[List[float]]:
    """解析分辨率 sweep 参数，支持 0.5,1.0,1.5 或 0.5:2.0:0.25"""
    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    if ":" in raw:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError("resolution sweep 格式应为 start:end:step")
        start, end, step = [float(p) for p in parts]
        if step == 0:
            raise ValueError("resolution sweep step 不能为 0")
        values = list(np.arange(start, end + (step * 0.5), step))
    else:
        values = [float(p.strip()) for p in raw.split(",") if p.strip()]

    return values if values else None


def format_resolution_label(resolution: float) -> str:
    """格式化分辨率用于输出路径/标签"""
    return f"{resolution:g}"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="聚类算法对比测试工具 - 从 merged_graph 运行多种聚类算法并输出结果供可视化对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 运行所有可用算法
  %(prog)s --algorithms leiden,louvain        # 只运行指定算法
  %(prog)s --resolution 1.5                   # 设置 Leiden/Louvain 分辨率
  %(prog)s --resolution-sweep 0.5,1.0,1.5     # Leiden/Louvain/HDBSCAN-Leiden 多分辨率 sweep
  %(prog)s --n-clusters 50                    # 设置 KMeans/Spectral 簇数
  %(prog)s --list-algorithms                  # 列出可用算法
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/graphs/merged_graph.json",
        help="输入图路径（默认: data/graphs/merged_graph.json）"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/graphs/clustering",
        help="输出目录（默认: data/graphs/clustering）"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="data/embeddings/multicommunities.db",
        help="嵌入数据库路径（默认: data/embeddings/multicommunities.db）"
    )

    parser.add_argument(
        "--table",
        type=str,
        default="entities",
        help="嵌入表名（默认: entities）"
    )

    parser.add_argument(
        "--algorithms", "-a",
        type=str,
        default=None,
        help="指定运行的算法列表（逗号分隔，默认: 全部可用算法）"
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden/Louvain/HDBSCAN-Leiden 分辨率参数（默认: 1.0）"
    )

    parser.add_argument(
        "--resolution-sweep",
        type=str,
        default=None,
        help="分辨率 sweep（逗号列表或 start:end:step，仅对 Leiden/Louvain/HDBSCAN-Leiden 生效）"
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="KMeans/Spectral/Agglomerative 簇数量（默认: 自动估计）"
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="HDBSCAN 最小簇大小（默认: 自动估计）"
    )

    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="列出可用的聚类算法并退出"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）"
    )

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 列出可用算法
    available_algorithms = get_available_algorithms()

    if args.list_algorithms:
        print("可用的聚类算法:")
        for algo in available_algorithms:
            requires_embedding = algo in ["kmeans", "hdbscan", "agglomerative", "spectral", "hdbscan_leiden"]
            note = " (需要嵌入向量)" if requires_embedding else ""
            print(f"  - {algo}{note}")
        return

    # 确定要运行的算法
    if args.algorithms:
        algorithms = [a.strip().lower() for a in args.algorithms.split(",")]
        # 验证算法有效性
        invalid = [a for a in algorithms if a not in available_algorithms]
        if invalid:
            logging.error(f"无效的算法: {invalid}")
            logging.info(f"可用算法: {available_algorithms}")
            return
    else:
        algorithms = available_algorithms

    logging.info(f"将运行以下聚类算法: {algorithms}")

    # 路径配置
    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir
    db_path = PROJECT_ROOT / args.db_path

    # 检查嵌入数据库
    embedding_algorithms = ["kmeans", "hdbscan", "agglomerative", "spectral", "hdbscan_leiden"]
    needs_embedding = any(a in algorithms for a in embedding_algorithms)

    if needs_embedding and not db_path.exists():
        logging.warning(f"嵌入数据库不存在: {db_path}")
        logging.warning("基于向量的算法可能会失败")

    # 解析分辨率 sweep
    try:
        resolution_sweep = parse_resolution_sweep(args.resolution_sweep)
    except ValueError as e:
        logging.error(str(e))
        return

    if resolution_sweep:
        logging.info(f"分辨率 sweep 列表: {resolution_sweep}")

    # 加载图
    try:
        graph = load_graph(input_path)
    except FileNotFoundError as e:
        logging.error(str(e))
        return

    # 准备算法参数
    algo_params = {}
    if args.resolution:
        algo_params["resolution"] = args.resolution
    if args.n_clusters:
        algo_params["n_clusters"] = args.n_clusters
    if args.min_cluster_size:
        algo_params["min_cluster_size"] = args.min_cluster_size

    # 运行所有算法
    results = []

    for algorithm in algorithms:
        sweep_targets = ["leiden", "louvain", "hdbscan_leiden"]
        if resolution_sweep and algorithm in sweep_targets:
            run_configs = [
                {
                    "label": f"{algorithm}@res={format_resolution_label(r)}",
                    "output_dir": output_dir / algorithm / f"res_{format_resolution_label(r)}",
                    "params": {**algo_params, "resolution": r}
                }
                for r in resolution_sweep
            ]
        else:
            run_configs = [
                {
                    "label": algorithm,
                    "output_dir": output_dir / algorithm,
                    "params": algo_params
                }
            ]

        for run_cfg in run_configs:
            result = run_clustering(
                graph=graph,
                algorithm=algorithm,
                embedding_db_path=db_path if algorithm in embedding_algorithms else None,
                embedding_table=args.table,
                **run_cfg["params"]
            )

            # 使用带分辨率的标签
            result["algorithm"] = run_cfg["label"]
            results.append(result)

            # 保存成功的结果
            if result["success"]:
                algo_dir = run_cfg["output_dir"]

                # 保存 JSON
                json_path = algo_dir / "community_graph.json"
                save_graph_json(result["graph"], json_path)

                # 保存 GEXF
                gexf_path = algo_dir / "community_graph.gexf"
                save_graph_gexf(result["graph"], gexf_path)

    # 保存统计摘要
    summary_path = output_dir / "summary.csv"
    save_summary_csv(results, summary_path)

    # 打印汇总表格
    print_summary_table(results)

    logging.info(f"\n所有结果已保存到: {output_dir}")
    logging.info("可以使用 Gephi 打开 .gexf 文件进行可视化对比")


if __name__ == "__main__":
    main()
