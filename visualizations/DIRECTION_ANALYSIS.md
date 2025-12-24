# 方向性关系矩阵分析说明

## 修改内容

修改了 `draw/ontology.py` 脚本，使其能够区分关系的方向性，生成非对称的关系矩阵。

## 主要改进

### 1. 方向区分
- **之前**: 矩阵是对称的，`(cat1, cat2)` 和 `(cat2, cat1)` 被视为同一个关系
- **现在**: 严格区分方向，`(cat1 → cat2)` 和 `(cat2 → cat1)` 分别统计

### 2. 关系矩阵解读

矩阵中的位置 `[i, j]` 表示从**行类别 i** 到**列类别 j** 的关系数量。

例如从分析结果可以看出明显的方向性差异：

| 源类别 → 目标类别 | 关系数量 | 反向关系数量 |
|-------------------|---------|-------------|
| processing → composition | 316 | 244 |
| composition → structure | 926 | 224 |
| processing → structure | 954 | 176 |
| processing → property | 1415 | 141 |
| composition → property | 725 | 64 |

### 3. 关键发现

#### 强方向性关系对：
1. **composition → structure** (926) vs **structure → composition** (224)
   - 成分主要**形成**结构，反向关系较少
   
2. **processing → property** (1415) vs **property → processing** (141)
   - 工艺主要**影响**性能，反向依赖较少

3. **processing → structure** (954) vs **structure → processing** (176)
   - 工艺主要**产生**结构，结构影响工艺较少

#### 相对平衡的关系对：
1. **processing → composition** (316) vs **composition → processing** (244)
   - 双向关系较为平衡
   
2. **property ↔ performance** (130 vs 75)
   - 性能和性质相互影响

## 文件输出

### 生成的文件：
1. **relationship_matrix.svg/png**: 方向性关系矩阵热图
   - 行：源类别
   - 列：目标类别
   
2. **superalloy_relationships.xlsx**: 详细的关系数据
   - 总览sheet：包含所有方向的关系统计
   - 详细sheets：每对类别的具体关系列表
   
3. **relationship_types_distribution.svg/png**: Top 20 关系类型分布

## 技术细节

### 修改的函数：

1. **`analyze_pairwise_relationships()`**
   - 遍历所有类别对的所有方向 (cat1 → cat2 和 cat2 → cat1)
   
2. **`_analyze_category_pair(cat1, cat2)`**
   - 只分析从 cat1 到 cat2 的单向关系
   - 移除了之前的反向边检查逻辑

3. **`visualize_relationship_matrix()`**
   - 矩阵构建时直接使用 `(cat1, cat2)` 键
   - 不再尝试查找反向键

## 使用方法

```bash
cd draw
python ontology.py
```

结果将保存在 `visualizations/` 目录下。

