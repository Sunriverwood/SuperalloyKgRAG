import json
import networkx as nx
from collections import defaultdict, Counter


def generate_labeled_cpspp_schema():
    # 1. 加载图数据
    file_path = '../data/graphs/final_graph.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 2. 定义 CPSPP 映射字典 (保持不变)
    category_map = {
        # Core
        "Superalloy": "Core", "Material": "Core", "Alloy": "Core",
        # Composition
        "Element": "Composition", "Chemical Element": "Composition", "Composition": "Composition",
        # Processing
        "Process": "Processing", "Heat Treatment": "Processing", "Manufacturing": "Processing",
        "Technique": "Processing",
        # Structure
        "Microstructure": "Structure", "Phase": "Structure", "Defect": "Structure", "Grain": "Structure",
        # Property
        "Property": "Property", "Mechanical Property": "Property", "Physical Property": "Property",
        # Performance/Application
        "Performance": "Performance", "Application": "Performance", "Metric": "Performance", "Failure": "Performance"
    }

    # 3. 统计类间关系及其名称 (Schema Discovery)
    # 结构: {(SourceCat, TargetCat): Counter({'relationship_name': count})}
    edge_label_stats = defaultdict(Counter)

    for u, v, edata in G.edges(data=True):
        src_type = G.nodes[u].get('type', 'Unknown')
        tgt_type = G.nodes[v].get('type', 'Unknown')

        src_cat = category_map.get(src_type)
        tgt_cat = category_map.get(tgt_type)

        # 获取关系名称，通常在 'label', 'relationship' 或 'relation' 字段中
        rel_label = edata.get('relationship') or edata.get('label') or edata.get('relation') or 'RELATED_TO'
        # 清理关系名称：转大写，去空格，统一格式
        rel_label = rel_label.upper().replace(" ", "_")

        # 记录核心类之间的关系
        if src_cat and tgt_cat and src_cat != tgt_cat:
            # 这里我们要保留方向，因为关系动词通常是有方向的 (如 Process -> AFFECTS -> Structure)
            edge_label_stats[(src_cat, tgt_cat)][rel_label] += 1

    # 4. 生成 Mermaid 代码
    print("### Generated Mermaid Code with Labels ###\n")
    print("```mermaid")
    print("graph TD")
    print("    %% 定义节点")
    print("    Core((Superalloy)):::core")
    print("    Comp(Composition):::ring")
    print("    Proc(Processing):::ring")
    print("    Struc(Structure):::ring")
    print("    Prop(Property):::ring")
    print("    Perf(Performance):::ring")

    print("\n    %% 样式定义")
    print("    classDef core fill:#ff9900,stroke:#333,stroke-width:4px,color:white,font-size:16px;")
    print("    classDef ring fill:#0099cc,stroke:#333,stroke-width:2px,color:white;")

    # 辅助函数：获取某对连接中最频繁出现的标签
    def get_top_label(s_cat, t_cat, default_label):
        stats = edge_label_stats.get((s_cat, t_cat))
        if not stats:
            # 尝试反向查找（如果数据中方向定义不一致）
            stats_rev = edge_label_stats.get((t_cat, s_cat))
            if stats_rev:
                return stats_rev.most_common(1)[0][0]
            return default_label
        return stats.most_common(1)[0][0]  # 返回频次最高的标签

    print("\n    %% 1. 核心辐射关系 (Hub)")
    # 动态获取 Core 与各部分的具体关系词
    print(f'    Core -- "{get_top_label("Core", "Composition", "HAS_COMPOSITION")}" --> Comp')
    print(f'    Core -- "{get_top_label("Core", "Processing", "UNDERGOES")}" --> Proc')
    print(f'    Core -- "{get_top_label("Core", "Structure", "HAS_MICROSTRUCTURE")}" --> Struc')
    print(f'    Core -- "{get_top_label("Core", "Property", "EXHIBITS")}" --> Prop')
    print(f'    Core -- "{get_top_label("Core", "Performance", "USED_IN")}" --> Perf')

    print("\n    %% 2. CPSPP 环形逻辑链 (Ring)")
    print(f'    Comp -- "{get_top_label("Composition", "Processing", "AFFECTS")}" --> Proc')
    print(f'    Proc -- "{get_top_label("Processing", "Structure", "DETERMINES")}" --> Struc')
    print(f'    Struc -- "{get_top_label("Structure", "Property", "INFLUENCES")}" --> Prop')
    print(f'    Prop -- "{get_top_label("Property", "Performance", "ENABLES")}" --> Perf')

    # 闭环关系 (Performance -> Composition 通常是设计反馈，数据中可能没有，保留虚线或默认词)
    print(f'    Perf -. "GUIDES_DESIGN" .-> Comp')

    print("\n    %% 3. 常见跨层级强关联 (Cross-Links)")
    # 检查 Processing -> Property 是否有强关联
    if edge_label_stats.get(('Processing', 'Property')):
        label = get_top_label("Processing", "Property", "IMPROVES")
        print(f'    Proc -. "{label}" .-> Prop')

    print("```")

    # 打印统计，方便您在论文正文中引用
    print("\n### Top Relationships Found (for Paper Text) ###")
    for (src, tgt), counter in sorted(edge_label_stats.items()):
        top_rel, count = counter.most_common(1)[0]
        print(f"{src} -> {tgt}: '{top_rel}' (count={count})")


if __name__ == "__main__":
    generate_labeled_cpspp_schema()