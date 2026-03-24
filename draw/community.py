import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

# ================= 配置参数 =================
# 图谱数据路径
GRAPH_FILE = Path("../data/graphs/final_graph.json")
# 输出图片路径
OUTPUT_IMG = Path("../visualizations/superalloy_community_structure.png")

# 宏观展示过滤器
MIN_DEGREE = 3  # 仅展示连接数 >= 3 的节点（调节此参数控制图的复杂度）
MIN_EDGE_WEIGHT = 3  # 仅展示权重 >= 2 的边
LABEL_TOP_K = 15  # 仅标注度最高的前K个节点名称


# ===========================================

def load_graph(filepath):
    """加载自定义格式的 JSON 图谱"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.Graph()

    # 加载节点
    for node in data.get('nodes', []):
        # 提取社区ID，默认为 '0'
        comm_id = node.get('community', '0')
        G.add_node(node['id'],
                   label=node.get('name', ''),
                   community=comm_id,
                   size=node.get('degree', 1))

    # 加载边 (检查 links 或 edges 字段，或者根据你提供的snippet，直接解析列表)
    # 注意：根据你的JSON结构，边可能直接在根列表或 'links' 中
    edges_list = data.get('links', []) if 'links' in data else data.get('edges', [])

    # 如果 JSON 结构是 networkx 的 node-link 格式
    if not edges_list and 'graph' in data:
        # 尝试从 adjacency 恢复，或者如果结构不同，需根据实际情况调整
        pass

        # 针对你的数据结构片段，如果边是单独的列表对象：
    # 假设边数据在 data['links'] 中 (这是标准 export)
    for edge in edges_list:
        if edge.get('weight', 1) >= MIN_EDGE_WEIGHT:
            G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))

    return G


def draw_community_graph(G):
    print(f"原始节点数: {G.number_of_nodes()}, 原始边数: {G.number_of_edges()}")

    # 1. 过滤：只保留核心架构（高频节点）
    # 创建子图，只包含度大于阈值的节点
    nodes_to_keep = [n for n, d in G.degree() if d >= MIN_DEGREE]
    G_sub = G.subgraph(nodes_to_keep)

    print(f"过滤后 (宏观视图) - 节点数: {G_sub.number_of_nodes()}, 边数: {G_sub.number_of_edges()}")

    if G_sub.number_of_nodes() == 0:
        print("警告：过滤条件过高，没有节点被选中。请降低 MIN_DEGREE。")
        return

    # 2. 准备颜色映射
    communities = sorted(list(set(nx.get_node_attributes(G_sub, 'community').values())),
                         key=lambda x: int(x) if x.isdigit() else 0)
    color_map = plt.get_cmap('tab20')  # 使用包含20种颜色的调色板
    # 创建 社区ID -> 颜色 的字典
    comm_to_color = {comm: color_map(i % 20) for i, comm in enumerate(communities)}

    node_colors = [comm_to_color[G_sub.nodes[n]['community']] for n in G_sub.nodes()]

    # 3. 节点大小：基于度中心性，并进行缩放
    node_sizes = [G_sub.degree(n) * 20 for n in G_sub.nodes()]

    # 4. 布局算法：使用弹簧布局，k参数控制节点间距（越大越松散）
    pos = nx.spring_layout(G_sub, k=0.5, iterations=50, seed=42)

    # 5. 绘图
    plt.figure(figsize=(16, 12))

    # 绘制边 (透明度降低以突出结构)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3, edge_color='gray')

    # 绘制节点
    nx.draw_networkx_nodes(G_sub, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors='white')  # 白色边框增加对比度

    # 6. 智能标注：只标注最重要的节点（Degree Top K）
    node_degrees = dict(G_sub.degree())
    top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:LABEL_TOP_K]
    labels = {n: G_sub.nodes[n]['label'] for n in top_nodes}

    # 使用文本路径效果增加可读性
    import matplotlib.patheffects as pe
    text_items = nx.draw_networkx_labels(G_sub, pos, labels, font_size=12, font_weight='bold', font_color='black')
    for t in text_items.values():
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    plt.title("Hierarchical Community Structure of Superalloy Knowledge Graph", fontsize=20)
    plt.axis('off')

    # 添加图例 (仅显示主要社区)
    legend_elements = []
    # 仅展示节点最多的前5个社区作为图例示例
    from collections import Counter
    comm_counts = Counter(nx.get_node_attributes(G_sub, 'community').values())
    top_comms = [c for c, _ in comm_counts.most_common(5)]

    for comm in top_comms:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          label=f'Cluster {comm}',
                                          markerfacecolor=comm_to_color[comm], markersize=10))

    plt.legend(handles=legend_elements, loc='upper right', title="Major Alloy Themes")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {OUTPUT_IMG}")


if __name__ == "__main__":
    if GRAPH_FILE.exists():
        g = load_graph(GRAPH_FILE)
        draw_community_graph(g)
    else:
        print(f"错误：找不到文件 {GRAPH_FILE}")