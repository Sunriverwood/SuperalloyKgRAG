# 多层社区报告生成指南 (Community Report Generation Guide)

本文档详细说明GraphRAG中多层社区报告的生成方法，这是实现宏观-微观多粒度检索的关键环节。

## 一、多层社区的构建方法 (Structure Construction)

### 目标
将庞大的知识图谱划分为结构化的、具有层级关系的模块，以便于"分而治之"。

### 基础输入
- **实体 (Entities)**：从源文档中提取的命名实体
- **关系 (Relationships)**：实体之间的关系
- **声明 (Claims)**：从文档中提取的事实性陈述

这些元素共同构成知识图谱索引。

### 核心算法
采用 **Leiden 社区检测算法** (Leiden community detection algorithm)

### 构建机制

#### 1. 递归检测 (Recursive Detection)
算法以层级方式运行：
- **首次检测**：在整个图谱上检测出第一层的大社区
- **递归分割**：在每个检测到的社区内部，递归地检测子社区 (sub-communities)
- **终止条件**：递归持续进行，直到达到"叶子社区"(leaf communities)

#### 2. 终止条件
递归在以下情况下停止：
1. **最大层级限制**：达到预设的 `max_level`
2. **最小社区规模**：社区节点数少于 `min_community_size`
3. **无法进一步分割**：Leiden算法返回单一社区，表示无法再细分

⚠️ **重要变更**：已移除 `max_cluster_size` 限制，允许算法自然分割直到无法继续为止。

#### 3. 覆盖原则
**相互排斥、集体穷尽** (mutually exclusive, collectively exhaustive)：
- 每一个层级的划分都保证覆盖图中的所有节点
- 节点不会重复出现在同一层级的不同社区中
- 每一层都是对完整数据集的独立划分

### 层级定义
- **Level 0**：根层级，整个图作为1个社区
- **Level 1**：对Level 0使用Leiden算法进行首次分割
- **Level 2+**：对每个社区继续使用Leiden算法递归分割
- **Level N**：叶子层级，最细粒度的社区（无法再分割的社区）

---

## 二、多层社区报告的生成方法 (Report Generation)

### 目标
为每一个社区生成详细的文本摘要 (Summary)，作为后续回答问题的语料库。

### 总体策略
**自底向上 (Bottom-up)**：
1. 先生成叶子社区的摘要
2. 利用叶子社区的摘要来生成高层社区的摘要

### 上下文管理机制
为了应对 LLM 的上下文窗口 (Context Window) 限制，系统设计了特定的优先级和替换机制。

---

## 三、叶子社区 (Leaf-level) 报告生成

### 特点
- 叶子社区没有子社区
- 只能利用原始的元素信息（节点、边、声明）

### 生成流程

#### 1. 优先级排序 (Prioritization)
系统根据重要性对社区内的元素进行排序。排序依据包括：
- 节点的度中心性 (Degree Centrality)
- 边的权重 (Edge Weight)
- 其他图论指标

#### 2. 填充规则
按照降序排列，迭代地将以下信息添加到 LLM 的上下文窗口中：
- **源节点描述** (Source Node Description)
- **目标节点描述** (Target Node Description)
- **边描述** (Edge Description)
- **相关声明** (Related Claims)

#### 3. 截断机制
- 一旦添加的内容达到 **Token 限制** (token limit)，即停止添加
- 忽略剩余的低优先级信息
- 确保最重要的信息被包含在摘要中

### 伪代码
```python
def generate_leaf_community_report(community, token_limit):
    """生成叶子社区报告"""
    # 1. 获取社区内的所有元素
    elements = get_community_elements(community)
    
    # 2. 优先级排序
    sorted_elements = prioritize_elements(elements)
    
    # 3. 填充上下文
    context = []
    current_tokens = 0
    
    for element in sorted_elements:
        element_tokens = count_tokens(element)
        if current_tokens + element_tokens <= token_limit:
            context.append(element)
            current_tokens += element_tokens
        else:
            break  # 达到token限制，停止添加
    
    # 4. 生成摘要
    report = llm_generate_summary(context)
    return report
```

---

## 四、高层社区 (Higher-level) 报告生成

### 特点
- 高层社区包含多个子社区
- 需要权衡"细节"与"概括"

### 生成流程

#### 条件判断 A：空间充足
**场景**：社区内所有原始元素摘要 (Element Summaries) 的总 Token 数 < 上下文窗口限制

**处理**：
- 按照叶子社区的处理方式
- 直接使用所有元素摘要生成报告

```python
if total_element_tokens < token_limit:
    # 使用所有原始元素摘要
    report = generate_from_elements(all_elements)
```

#### 条件判断 B：空间不足 - 替换机制
**场景**：原始元素摘要超出了窗口限制

**处理策略**：

##### 1. 子社区排序
按照子社区包含的元素摘要的 Token 数量进行降序排列

```python
sub_communities = sorted(
    community.children,
    key=lambda x: count_tokens(x.element_summaries),
    reverse=True
)
```

##### 2. 迭代缩减 (Substitution)
按顺序将 Token 占用较多的子社区的"原始元素列表"替换为该子社区已经生成的较短的"社区摘要" (Community Summary)

```python
context = []
current_tokens = 0

for sub_community in sub_communities:
    element_tokens = count_tokens(sub_community.element_summaries)
    summary_tokens = count_tokens(sub_community.community_summary)
    
    # 优先使用元素摘要，如果空间不足则使用社区摘要
    if current_tokens + element_tokens <= token_limit:
        context.extend(sub_community.element_summaries)
        current_tokens += element_tokens
    elif current_tokens + summary_tokens <= token_limit:
        context.append(sub_community.community_summary)
        current_tokens += summary_tokens
    else:
        break  # 无法继续添加
```

### 完整伪代码
```python
def generate_higher_level_community_report(community, token_limit):
    """生成高层社区报告"""
    # 1. 获取所有原始元素
    all_elements = get_community_elements(community)
    total_element_tokens = count_tokens(all_elements)
    
    # 条件判断 A：空间充足
    if total_element_tokens <= token_limit:
        return llm_generate_summary(all_elements)
    
    # 条件判断 B：空间不足，启动替换机制
    sub_communities = sorted(
        community.children,
        key=lambda x: count_tokens(x.element_summaries),
        reverse=True
    )
    
    context = []
    current_tokens = 0
    
    for sub_community in sub_communities:
        element_summaries = sub_community.element_summaries
        community_summary = sub_community.community_summary
        
        element_tokens = count_tokens(element_summaries)
        summary_tokens = count_tokens(community_summary)
        
        # 优先使用详细的元素摘要
        if current_tokens + element_tokens <= token_limit:
            context.extend(element_summaries)
            current_tokens += element_tokens
        # 空间不足，使用压缩的社区摘要
        elif current_tokens + summary_tokens <= token_limit:
            context.append(community_summary)
            current_tokens += summary_tokens
        else:
            # 无法继续添加，停止
            break
    
    # 生成报告
    report = llm_generate_summary(context)
    return report
```

---

## 五、实现要点

### 1. 数据结构设计
```python
class HierarchicalCommunity:
    """社区层级结构"""
    def __init__(self, community_id, level, parent_id=None):
        self.community_id = community_id
        self.level = level
        self.parent_id = parent_id
        self.children_ids = []  # 子社区ID列表
        self.node_ids = []  # 节点ID列表
        
        # 报告生成相关
        self.element_summaries = []  # 原始元素摘要列表
        self.community_summary = ""  # 社区摘要（生成后填充）
```

### 2. Token 计数
需要实现准确的 Token 计数函数：
```python
def count_tokens(text):
    """计算文本的token数量"""
    # 使用与LLM相同的tokenizer
    return len(tokenizer.encode(text))
```

### 3. 元素优先级
实现元素重要性评分：
```python
def prioritize_elements(elements):
    """对元素进行优先级排序"""
    # 可以使用多种指标的组合
    scored_elements = []
    for element in elements:
        score = calculate_importance(element)
        scored_elements.append((score, element))
    
    # 按分数降序排列
    scored_elements.sort(reverse=True)
    return [elem for score, elem in scored_elements]
```

### 4. 报告生成提示词
设计有效的提示词模板：
```python
REPORT_PROMPT = """
根据以下信息，生成一个全面的社区摘要报告：

{context}

要求：
1. 概括社区的主要主题和内容
2. 识别关键实体及其关系
3. 总结重要的事实声明
4. 保持简洁但信息丰富

摘要：
"""
```

---

## 六、处理流程图

```
[所有社区] 
    ↓
[识别叶子社区]
    ↓
[生成叶子社区报告] ← 使用原始元素 + 优先级排序 + Token截断
    ↓
[自底向上遍历]
    ↓
[对每个高层社区]
    ├─ 计算元素摘要总Token数
    ├─ 判断：Token数 < 限制？
    │   ├─ 是 → 使用所有元素摘要生成
    │   └─ 否 → 启动替换机制
    │       ├─ 排序子社区（按Token数降序）
    │       ├─ 迭代添加：优先使用元素摘要
    │       ├─ 空间不足时：使用社区摘要替换
    │       └─ 生成高层报告
    ↓
[完成所有层级的报告生成]
```

---

## 七、优势分析

### 1. 多粒度检索
- **宏观视角**：高层社区报告提供全局概览
- **微观视角**：叶子社区报告提供详细信息
- **灵活性**：可以根据查询需求选择合适的层级

### 2. 上下文管理
- **优先级机制**：确保重要信息被保留
- **替换策略**：在有限的上下文窗口内最大化信息密度
- **自底向上**：逐层抽象，保持信息连贯性

### 3. 可扩展性
- **分层结构**：适用于大规模知识图谱
- **递归分割**：自动适应图谱的复杂度
- **模块化设计**：易于维护和扩展

---

## 八、注意事项

### 1. Token 计数准确性
确保使用与 LLM 相同的 tokenizer 进行 token 计数，避免超出窗口限制。

### 2. 优先级策略
根据具体应用场景调整元素优先级排序策略，可能需要多次实验优化。

### 3. 报告质量
定期评估生成的报告质量，根据反馈调整提示词和生成策略。

### 4. 性能优化
- 缓存已生成的报告
- 并行处理独立的社区
- 使用批处理减少 API 调用次数

---

## 九、相关文档

- [HIERARCHICAL_COMMUNITIES_GUIDE.md](HIERARCHICAL_COMMUNITIES_GUIDE.md) - 分层社区检测详细指南
- [RUN_INDEXING_GUIDE.md](RUN_INDEXING_GUIDE.md) - 索引构建流程指南
- `utils/community_reports.py` - 社区报告生成模块
- `utils/recursive_leiden.py` - 递归Leiden算法实现

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-14 | 文档整理，更新相关链接 |
| 2025-12-25 | 初始版本 |

