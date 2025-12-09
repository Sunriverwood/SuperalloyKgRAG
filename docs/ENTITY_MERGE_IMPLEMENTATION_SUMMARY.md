# 实体合并功能实现总结

## 实现的功能

### 功能1: 批量请求数量控制

在实体消歧与合并阶段，单次批量请求控制数量最多为 `embedding_batch_size`。

#### 修改的文件

**`core/pipeline_qwen/graph_builder_qwen.py`**

1. **`create_batch_requests` 函数** (第206行)
   - 新增 `batch_size` 参数
   - 当请求数量超过 `batch_size` 时，自动拆分为多个批次
   - 每个批次生成独立的 `.jsonl` 文件
   
2. **`create_entity_merge_requests` 函数** (第587行)
   - 新增 `batch_size` 参数
   - 支持将候选簇拆分为多个批次处理
   - 每个批次生成独立的请求文件

3. **`run_disambiguation_stage` 函数** (第1045行)
   - 新增 `config` 参数，用于获取 `embedding_batch_size` 配置
   - 支持分批处理消歧请求
   - 自动合并多个批次的结果

4. **`run_entity_merge_stage` 函数** (第1080行)
   - 使用 `embedding_batch_size` 控制嵌入向量生成批次
   - 使用 `embedding_batch_size` 控制LLM仲裁请求批次
   - 自动合并多个批次的结果

5. **`build_pipeline_from_config` 函数** (第1289行)
   - 更新 `run_disambiguation_stage` 调用，传递 `config` 参数

#### 配置文件修改

**`config/settings.yaml`**

- 更新了 `embedding_batch_size` 的注释说明，明确其同时控制嵌入向量数量和LLM仲裁数量

```yaml
embedding_batch_size: 5000 # 每批次处理的实体嵌入数量和LLM仲裁数量，避免API配额错误
```

### 功能2: 人工审核功能

实体合并阶段增加人工审核功能，可以随机抽样候选簇进行人工审核，并对比大模型合并效果与人工合并效果。

#### 新建的文件

**`utils/entity_merge_review.py`** (新建)

核心类和函数：

1. **`ClusterReviewResult` 类**
   - 数据类，用于存储单个簇的审核结果
   - 包含人工决策、LLM决策、对比结果等信息

2. **`EntityMergeReviewer` 类**
   - 主要功能类，负责整个人工审核流程
   - 主要方法：
     - `sample_clusters()`: 随机抽样候选簇
     - `display_cluster()`: 展示候选簇信息
     - `collect_human_decision()`: 收集人工审核决策
     - `run_manual_review()`: 运行完整的人工审核流程
     - `compare_with_llm()`: 对比LLM决策与人工决策
     - `generate_comparison_report()`: 生成对比报告
     - `save_report()`: 保存报告到JSON文件
     - `visualize_comparison()`: 生成可视化对比图表
     - `print_summary()`: 打印对比摘要

3. **`run_entity_merge_review` 函数**
   - 便捷函数，封装完整的人工审核流程
   - 支持自动生成报告和可视化

#### 修改的文件

**`core/pipeline_qwen/graph_builder_qwen.py`**

在 `run_entity_merge_stage` 函数中（第1195-1213行）添加了人工审核功能集成：

```python
# 人工审核功能（可选）
enable_manual_review = config["graph_builder"].get("enable_manual_review", False)
if enable_manual_review:
    try:
        from utils.entity_merge_review import run_entity_merge_review
        review_sample_size = config["graph_builder"].get("manual_review_sample_size", 5)
        review_output_dir = PROJECT_ROOT / config["graph_builder"].get("manual_review_output_dir", 
                                                                       "data/reports/manual_review")
        
        logging.info(f"🔍 启动人工审核流程，抽样数量: {review_sample_size}")
        review_report = run_entity_merge_review(
            graph=graph,
            clusters=clusters,
            llm_groups=groups,
            sample_size=review_sample_size,
            output_dir=review_output_dir
        )
        logging.info("✅ 人工审核完成")
    except Exception as e:
        logging.warning(f"⚠️ 人工审核过程出现异常，已跳过: {e}")
```

**`config/settings.yaml`**

新增人工审核相关配置：

```yaml
graph_builder:
  # 人工审核配置
  enable_manual_review: False  # 是否启用人工审核功能（默认关闭）
  manual_review_sample_size: 5  # 人工审核抽样数量
  manual_review_output_dir: "data/reports/manual_review"  # 人工审核输出目录
```

#### 文档文件

**`docs/ENTITY_MERGE_REVIEW_GUIDE.md`** (新建)

详细的使用指南，包括：
- 功能概述
- 配置方式
- 使用方法
- 审核流程说明
- 输出结果解读
- 常见问题解答

**`tests/test_entity_merge_review.py`** (新建)

测试脚本，包含：
- 基本功能测试
- 报告生成功能测试
- 可视化功能测试

## 工作流程

### 功能1: 批量请求控制流程

```
1. 加载图谱
2. 实体消歧阶段
   ├─ 检查节点数量是否超过 embedding_batch_size
   ├─ 如果超过，拆分为多个批次
   │  ├─ 生成 disambiguation_requests_batch_1.jsonl
   │  ├─ 生成 disambiguation_requests_batch_2.jsonl
   │  └─ ...
   ├─ 依次提交每个批次的作业
   └─ 合并所有批次的结果
3. 实体合并阶段
   ├─ 生成嵌入向量
   │  ├─ 检查实体数量是否超过 embedding_batch_size
   │  ├─ 如果超过，拆分为多个批次
   │  └─ 合并所有批次的嵌入向量
   ├─ 构建候选簇
   ├─ LLM仲裁
   │  ├─ 检查候选簇数量是否超过 embedding_batch_size
   │  ├─ 如果超过，拆分为多个批次
   │  │  ├─ 生成 entities_merge_requests_batch_1.jsonl
   │  │  ├─ 生成 entities_merge_requests_batch_2.jsonl
   │  │  └─ ...
   │  ├─ 依次提交每个批次的作业
   │  └─ 合并所有批次的结果
   └─ 应用合并结果
```

### 功能2: 人工审核流程

```
1. 实体合并阶段
2. 检查 enable_manual_review 配置
3. 如果启用人工审核
   ├─ 从候选簇中随机抽样（默认5个）
   ├─ 对每个抽样簇
   │  ├─ 显示实体详细信息
   │  ├─ 收集人工决策（合并/保持分离）
   │  └─ 记录人工理由
   ├─ 对比LLM决策与人工决策
   ├─ 生成对比报告
   │  ├─ 统计一致率
   │  ├─ 统计合并率
   │  └─ 记录不一致案例
   ├─ 保存JSON报告
   ├─ 生成可视化图表
   │  ├─ 一致性对比饼图
   │  ├─ 决策分布柱状图
   │  └─ 合并率对比柱状图
   └─ 打印对比摘要
```

## 特点

### 功能1特点

1. **透明性**: 完全兼容原有流程，不影响最终结果
2. **自动化**: 自动检测是否需要分批，无需手动干预
3. **可配置**: 通过 `embedding_batch_size` 灵活控制批次大小
4. **容错性**: 即使单个批次失败，其他批次可以继续处理
5. **可追溯**: 每个批次生成独立文件，便于调试和问题排查

### 功能2特点

1. **可选性**: 默认关闭，仅在需要时启用
2. **随机性**: 随机抽样保证评估的客观性
3. **交互性**: 通过控制台交互式收集人工决策
4. **可视化**: 自动生成直观的对比图表
5. **详细性**: 提供详细的对比报告和不一致案例分析
6. **非侵入性**: 不影响实际的合并结果，仅用于评估

## 使用示例

### 启用批量控制（已默认启用）

```yaml
# config/settings.yaml
graph_builder:
  embedding_batch_size: 5000  # 根据API配额调整
```

### 启用人工审核

```yaml
# config/settings.yaml
graph_builder:
  enable_manual_review: True  # 启用人工审核
  manual_review_sample_size: 5  # 抽样5个候选簇
  manual_review_output_dir: "data/reports/manual_review"
```

然后正常运行索引流程：

```bash
python app/run_indexing.py
```

## 输出示例

### 批量控制日志

```
⚙️ 请求数量 8000 超过批次大小 5000，将拆分为 2 个批次处理
✅ 批次 1/2 已写入 5000 个请求: data/cache/disambiguation_requests_batch_1.jsonl
✅ 批次 2/2 已写入 3000 个请求: data/cache/disambiguation_requests_batch_2.jsonl
🔄 提交消歧批次 1/2
✅ 批次 1/2 完成
🔄 提交消歧批次 2/2
✅ 批次 2/2 完成
```

### 人工审核输出

```
🔍 启动人工审核流程，抽样数量: 5
已从 156 个候选簇中随机抽样 5 个

================================================================================
候选簇 #42
================================================================================

实体 1:
  ID: paper_123-e-45
  名称: γ' precipitate
  类型: MATERIAL_PHASE
  描述: Ordered L12 structure precipitate...

实体 2:
  ID: paper_234-e-67
  名称: gamma prime phase
  类型: MATERIAL_PHASE
  描述: The strengthening phase in superalloys...

--------------------------------------------------------------------------------
请判断这些实体是否应该合并:
1. 合并 (这些实体指向同一概念)
2. 保持分离 (这些实体是不同的概念)

请输入选择 (1/2): 1
请输入合并后的规范名称: γ' phase
请简要说明合并理由: Same strengthening phase, different naming conventions

[继续其他簇的审核...]

================================================================================
实体合并审核对比摘要
================================================================================

总审核样本数: 5
一致数量: 4
不一致数量: 1
一致率: 80.00%

人工决策:
  合并: 3 (60.00%)
  保持分离: 2

LLM决策:
  合并: 4 (80.00%)
  保持分离: 1

✅ 人工审核完成
对比报告已保存到: data/reports/manual_review/entity_merge_review_report.json
可视化结果已保存到: data/reports/manual_review/entity_merge_review_comparison.png
```

## 注意事项

1. **批量控制功能**已经在代码中实现，默认启用，无需额外配置
2. **人工审核功能**默认关闭，需要在配置文件中显式启用
3. 人工审核需要在控制台进行交互，不适合后台运行
4. 建议先使用小批量数据测试人工审核功能
5. 可视化图表需要系统安装中文字体，否则中文可能显示为方块

## 依赖要求

已在 `utils/entity_merge_review.py` 中使用的依赖：
- `networkx`: 图谱操作
- `matplotlib`: 可视化
- `numpy`: 数值计算

这些依赖应该已经在项目的 `requirements.txt` 中。

## 后续扩展建议

1. **断点续传**: 支持人工审核过程中断后继续
2. **批量审核**: 支持一次展示多个候选簇，提高审核效率
3. **决策应用**: 支持将人工决策应用到实际合并过程中
4. **审核历史**: 保存历史审核记录，支持回溯和分析
5. **Web界面**: 开发Web界面替代控制台交互
6. **多人协作**: 支持多人并行审核，提高审核效率

## 版本信息

- 实现日期: 2025-01-09
- 实现版本: v1.0
- 对应文件版本: graph_builder_qwen.py (已修改)

