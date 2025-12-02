# 路由器集成推理功能 - 使用CoT技术

## 更新概述

已成功将推理查询功能集成到智能路由器 (`router_qwen.py`) 中，并采用 Chain of Thought (CoT) 技术实现智能分类和自动方法选择。

## 新增功能

### 1. 推理查询模式 (REASONING)

路由器现在支持四种查询模式：

| 模式 | 描述 | 示例查询 |
|------|------|----------|
| **GLOBAL** | 全局摘要/概览 | "总结所有超合金的主要应用" |
| **LOCAL** | 特定实体属性 | "镍基超合金的成分是什么？" |
| **REASONING** | 多跳推理/关系发现 | "镍和涡轮叶片之间有什么关系？" |
| **DRIFT** | 上下文扩展搜索 | "全面分析高温合金的性能" |

### 2. CoT (Chain of Thought) 分类

使用思维链技术进行智能分类，包含以下步骤：

```
1. Query Analysis (查询分析)
   - 用户在问什么？
   - 需要什么类型的信息？

2. Complexity Assessment (复杂度评估)
   - 简单事实查找 → LOCAL
   - 关系发现/因果推理 → REASONING
   - 广泛概览/摘要 → GLOBAL
   - 需要迭代细化 → DRIFT

3. Keywords Identification (关键词识别)
   - 识别特定模式的关键词
   - 判断查询意图

4. Method Selection (方法选择，仅针对REASONING)
   - PPR: 适合发现相关实体
   - GNN: 适合复杂多跳推理
```

### 3. 自动方法选择

当路由器判定为推理查询时，CoT会自动选择最佳方法：

**PPR (Personalized PageRank)**：
- 适用于："find related", "what connects", "entities similar to"
- 优势：基于图结构发现关联实体
- 示例："找到与镍相关的所有应用"

**GNN (Graph Neural Network)**：
- 适用于："why", "causal relationship", "complex interaction"
- 优势：学习复杂的多跳模式
- 示例："为什么高温会影响蠕变抗性？"

## 实现细节

### 推理处理器初始化

```python
# 智能检测模型是否存在
model_path = PROJECT_ROOT / "data/reasoning/model.pt"
if model_path.exists():
    self.reasoning_handler = ReasoningQueryHandler(config, load_trained_model=True)
    self.reasoning_enabled = True
else:
    logging.warning("未找到推理模型，推理功能将被禁用")
    self.reasoning_enabled = False
```

**优势**：
- ✅ 自动检测模型可用性
- ✅ 优雅降级到DRIFT模式
- ✅ 不会因模型缺失而崩溃

### CoT分类流程

```python
async def _cot_classify_intent(self, query: str) -> Dict[str, Any]:
    """
    使用CoT技术进行意图分类
    
    Returns:
        {
            'intent': 'REASONING',  # GLOBAL/LOCAL/REASONING/DRIFT
            'reasoning': '...',      # CoT推理过程
            'method': 'ppr'          # ppr/gnn (仅REASONING模式)
        }
    """
```

### 查询路由逻辑

```python
async def route_and_answer(self, query: str) -> str:
    # 1. CoT分类
    result = await self._cot_classify_intent(query)
    intent = result['intent']
    method = result['method']
    
    # 2. 分发执行
    if intent == "REASONING":
        if self.reasoning_enabled:
            # 使用推理处理器（带方法选择）
            return self.reasoning_handler.query(query, method=method)
        else:
            # 降级为漂移搜索
            return await self.drift_handler.perform_drift_search(query)
```

## 使用方法

### 基本用法

```bash
# 交互模式
python core/query_qwen/router_qwen.py

# 命令行模式
python core/query_qwen/router_qwen.py "What is the relationship between nickel and turbine blades?"
```

### 查询示例

#### 示例1：推理查询（自动选择PPR）

**查询**：
```
What connects nickel-based superalloys to turbine blades?
```

**CoT分析**：
```
REASONING: The query asks "what connects", indicating a relationship discovery task.
This requires finding paths in the knowledge graph between two entities.
INTENT: REASONING
METHOD: ppr
```

**执行流程**：
1. 路由器识别为REASONING模式
2. 自动选择PPR方法（适合"连接"类查询）
3. 推理处理器执行PPR图传播
4. 提取推理路径并生成答案

#### 示例2：推理查询（自动选择GNN）

**查询**：
```
Why does high temperature affect creep resistance in superalloys?
```

**CoT分析**：
```
REASONING: The query asks "why", indicating causal reasoning is needed.
This requires understanding complex interactions and learned patterns.
INTENT: REASONING
METHOD: gnn
```

**执行流程**：
1. 路由器识别为REASONING模式
2. 自动选择GNN方法（适合"为什么"类查询）
3. 推理处理器使用query-aware GNN
4. 生成基于学习模式的推理答案

#### 示例3：降级处理（模型不存在）

**查询**：
```
What is the relationship between temperature and creep?
```

**日志输出**：
```
CoT 分类结果:
  意图: REASONING
  推理: Relationship query requiring multi-hop reasoning
  推理方法: ppr

路由判定: 推理查询 (Reasoning) with method=ppr
⚠ 推理功能未启用，降级为漂移搜索模式
```

## 配置说明

### 必需配置 (settings.yaml)

```yaml
reasoning:
  output:
    model_path: "data/reasoning/model.pt"  # 推理模型路径
  inference:
    max_path_length: 3
    max_paths_per_query: 5
    min_path_score: 0.01
  model:
    hidden_dim: 256
    num_layers: 3
    num_heads: 4

query:
  generation_model: "qwen-plus"  # 用于CoT分类
  temperature: 0.7
  drift_k_followups: 2  # 漂移搜索参数
  drift_max_steps: 2
```

### 可选配置

```yaml
logging:
  level: "INFO"
  log_file: "logs/router_qwen.log"
```

## CoT提示词模板

系统使用的CoT提示词包含以下要素：

1. **角色定义**：Knowledge Graph RAG系统的智能分类器
2. **思考步骤**：4步分析流程
3. **分类标准**：明确的判断依据
4. **方法选择**：针对REASONING的子分类
5. **输出格式**：结构化的分类结果

完整提示词见 `_cot_classify_intent()` 方法。

## 性能优化

### 1. 异步执行

```python
# 推理处理器的同步方法包装为异步
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(
    None, 
    lambda: self.reasoning_handler.query(query, method=method)
)
```

### 2. 降级策略

```python
# CoT失败时自动降级
try:
    return await self._cot_classify_intent(query)
except Exception as e:
    logging.error(f"CoT分类失败: {e}")
    return await self._simple_classify_intent(query)
```

### 3. 智能缓存

推理模型只在路由器初始化时加载一次，所有查询共享。

## 日志示例

### 成功的推理查询

```
2025-12-02 20:30:15 - INFO - [Router] CoT 分类结果:
2025-12-02 20:30:15 - INFO - [Router]   意图: REASONING
2025-12-02 20:30:15 - INFO - [Router]   推理: Query asks about relationship, requires multi-hop reasoning
2025-12-02 20:30:15 - INFO - [Router]   推理方法: ppr
2025-12-02 20:30:15 - INFO - [Router] 路由判定: 推理查询 (Reasoning) with method=ppr
2025-12-02 20:30:16 - INFO - Reasoning for query: What is the relationship between nickel and turbine blades?
2025-12-02 20:30:17 - INFO - Top-20 nodes retrieved, score range: [2.7760, 3.8909]
2025-12-02 20:30:18 - INFO - PPR converged at iteration 47
2025-12-02 20:30:18 - INFO - Extracted and ranked 5 paths
```

### 降级处理

```
2025-12-02 20:31:00 - WARNING - [Router] 未找到推理模型: data/reasoning/model.pt
2025-12-02 20:31:00 - INFO - [Router] CoT 分类结果:
2025-12-02 20:31:00 - INFO - [Router]   意图: REASONING
2025-12-02 20:31:00 - WARNING - [Router] 推理功能未启用，降级为漂移搜索模式
2025-12-02 20:31:00 - INFO - [Router] 启动漂移检索 (Drift Search)
```

## 测试验证

### 测试用例

```python
# 测试1: 推理查询 + PPR
query1 = "What is the relationship between nickel and turbine blades?"
# 期望: REASONING + ppr

# 测试2: 推理查询 + GNN
query2 = "Why does temperature affect creep resistance?"
# 期望: REASONING + gnn

# 测试3: 局部查询
query3 = "What is the composition of Inconel 718?"
# 期望: LOCAL

# 测试4: 全局查询
query4 = "Summarize the main applications of superalloys"
# 期望: GLOBAL

# 测试5: 漂移查询
query5 = "Comprehensive analysis of high-temperature alloys"
# 期望: DRIFT
```

### 运行测试

```bash
# 单个查询
python core/query_qwen/router_qwen.py "What connects nickel to turbine blades?"

# 交互模式测试多个查询
python core/query_qwen/router_qwen.py
```

## 优势总结

### ✅ 智能分类
- 使用CoT技术提高分类准确性
- 详细的推理过程可追溯
- 4种模式精准覆盖各类查询

### ✅ 自动方法选择
- PPR vs GNN 自动判定
- 基于查询语义和关键词
- 无需用户手动指定

### ✅ 优雅降级
- 模型不存在时自动降级
- CoT失败时使用简单分类
- 保证系统稳定性

### ✅ 模块化设计
- 各处理器独立可测试
- 易于扩展新的查询模式
- 统一的接口和日志

## 未来改进

1. **学习优化**：收集分类数据，fine-tune分类模型
2. **混合策略**：对复杂查询同时使用多种方法
3. **用户反馈**：根据用户满意度调整方法选择
4. **缓存机制**：缓存常见查询的分类结果

## 故障排查

### 问题1：推理功能总是被禁用

**检查**：
```bash
ls data/reasoning/model.pt
```

**解决**：
```bash
python core/query_qwen/reasoning_query_qwen.py --train
```

### 问题2：CoT分类总是失败

**检查日志**：
```
tail -f logs/router_qwen.log | grep "CoT"
```

**可能原因**：
- API密钥未设置
- 模型名称错误
- 网络连接问题

### 问题3：方法选择不准确

**临时解决**：修改 `_cot_classify_intent()` 的提示词
**长期方案**：收集数据进行模型fine-tuning

---

**版本**: 1.0  
**日期**: 2025-12-02  
**作者**: Graph Reasoning Team

