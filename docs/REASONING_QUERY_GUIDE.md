# 图推理查询系统 - 使用指南

## 概述

`reasoning_query_qwen.py` 现在是一个统一的图推理查询系统入口，支持三种运行模式：
1. **交互模式** - 用户友好的问答界面
2. **命令行模式** - 脚本化查询
3. **训练模式** - 模型训练

## 功能特性

### ✨ 新增功能

1. **智能模型检测**
   - 自动检测 `data/reasoning/model.pt` 是否存在
   - 如果模型已训练，直接加载，**不会重复训练**
   - 如果模型不存在，提示用户训练或在交互模式中引导训练

2. **交互式运行**
   - 点击运行即可使用，无需命令行参数
   - 友好的问答界面
   - 逐步选择推理方法、LLM生成、保存结果等

3. **合并功能**
   - 整合了 `run_reasoning_query.py` 的所有功能
   - 统一的结果格式化输出
   - 支持所有原有的命令行参数

## 使用方法

### 方式1：直接点击运行（交互模式）

**最简单的使用方式** - 在IDE中直接运行 `reasoning_query_qwen.py`：

```bash
# 直接运行，进入交互模式
python core/query_qwen/reasoning_query_qwen.py
```

**交互流程**：

```
================================================================================
Graph Reasoning Query System - Interactive Mode
================================================================================

✓ Found trained model: D:\...\data\reasoning\model.pt

Loading reasoning model...
✓ Model loaded successfully!

--------------------------------------------------------------------------------

Enter your query (or 'quit' to exit): What is the relationship between nickel and turbine blades?

Reasoning method (ppr/gnn) [ppr]: ppr
Generate LLM answer? (yes/no) [yes]: yes
Save results to file? (yes/no) [no]: no

================================================================================
Processing query...
================================================================================
...
```

**如果模型不存在**，会提示：

```
⚠ No trained model found at: D:\...\data\reasoning\model.pt
Do you want to train the model now? (yes/no) [yes]: yes
Number of training epochs [50]: 50

================================================================================
Starting Model Training...
================================================================================
...
```

### 方式2：命令行查询模式

**快速查询**：

```bash
# 基本查询
python core/query_qwen/reasoning_query_qwen.py --query "What is nickel used for?"

# 使用GNN方法
python core/query_qwen/reasoning_query_qwen.py --query "..." --method gnn

# 不生成LLM答案（只显示推理路径）
python core/query_qwen/reasoning_query_qwen.py --query "..." --no-llm

# 保存结果到文件
python core/query_qwen/reasoning_query_qwen.py --query "..." --output results.json
```

**短参数形式**：

```bash
python core/query_qwen/reasoning_query_qwen.py -q "What is nickel?" -m ppr -o result.json
```

### 方式3：训练模式

**智能训练**（跳过已存在的模型）：

```bash
# 如果model.pt存在，跳过训练
python core/query_qwen/reasoning_query_qwen.py --train

# 指定训练轮数
python core/query_qwen/reasoning_query_qwen.py --train --epochs 100

# 强制重新训练（即使模型存在）
python core/query_qwen/reasoning_query_qwen.py --train --force-train --epochs 100
```

**输出示例**（模型已存在时）：

```
✓ Model already exists at: D:\...\data\reasoning\model.pt
  Use --force-train to retrain anyway.
  Skipping training.
```

### 方式4：明确指定交互模式

```bash
# 使用 -i 或 --interactive
python core/query_qwen/reasoning_query_qwen.py --interactive
python core/query_qwen/reasoning_query_qwen.py -i
```

## 完整参数列表

### 模式选择（互斥）

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `--interactive` | `-i` | 启动交互模式 |
| `--train` | - | 训练模式（智能跳过） |

### 查询参数

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--query` | `-q` | None | 查询文本 |
| `--method` | `-m` | `ppr` | 推理方法：ppr/gnn |
| `--no-llm` | - | False | 跳过LLM答案生成 |

### 训练参数

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--epochs` | `-e` | 50 | 训练轮数 |
| `--force-train` | - | False | 强制重新训练 |

### 输出参数

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `--output` | `-o` | 保存结果到JSON文件 |

## 智能模型管理

### 模型检测逻辑

```python
model_path = PROJECT_ROOT / "data/reasoning/model.pt"

if model_path.exists():
    # 模型存在 -> 直接加载
    load_trained_model = True
else:
    # 模型不存在 -> 提示训练
    # 交互模式：询问是否训练
    # 命令行模式：显示错误，提示使用 --train
```

### 训练行为

**普通训练** (`--train`):
- ✅ 检查模型是否存在
- ✅ 如果存在 → 跳过训练，显示提示
- ✅ 如果不存在 → 开始训练

**强制训练** (`--train --force-train`):
- ✅ 忽略现有模型
- ✅ 总是重新训练
- ✅ 覆盖旧模型

## 输出格式

### 屏幕输出

```
================================================================================
REASONING RESULTS
================================================================================

Query: What is the relationship between nickel and turbine blades?

Top Relevant Entities:
--------------------------------------------------------------------------------
 1. Nickel-based superalloys                           (score: 3.8909)
 2. Turbine blades                                      (score: 3.6543)
 3. High-temperature alloys                             (score: 3.2156)
...

Reasoning Paths:
--------------------------------------------------------------------------------

Path 1 (confidence: 1.0000):
Single-node path: Nickel-based superalloys (no edges)

Path 2 (confidence: 1.0000):
Single-node path: Turbine blades (no edges)

Final Answer:
--------------------------------------------------------------------------------
Nickel-based superalloys are extensively used in turbine blades...

================================================================================
```

### JSON输出（--output）

```json
{
  "query": "What is the relationship between nickel and turbine blades?",
  "top_nodes": [
    {
      "id": "node_123",
      "name": "Nickel-based superalloys",
      "score": 3.8909
    }
  ],
  "paths": [
    {
      "path": ["node_123"],
      "score": 1.0,
      "edge_types": [],
      "explanation": "Single-node path: Nickel-based superalloys (no edges)"
    }
  ],
  "num_paths": 2,
  "answer": "Nickel-based superalloys are extensively used..."
}
```

## 使用场景

### 场景1：日常查询（推荐）

**直接运行，交互式查询**：

```bash
# 在IDE中点击运行，或命令行执行
python core/query_qwen/reasoning_query_qwen.py
```

- ✅ 自动检测模型
- ✅ 友好的输入提示
- ✅ 可以连续查询多次
- ✅ 适合探索和测试

### 场景2：批量查询

**编写脚本批量处理**：

```bash
#!/bin/bash
queries=(
    "What is nickel used for?"
    "Properties of turbine blades"
    "Superalloy composition"
)

for query in "${queries[@]}"; do
    python core/query_qwen/reasoning_query_qwen.py \
        --query "$query" \
        --output "results/${query//[^a-zA-Z0-9]/_}.json"
done
```

### 场景3：首次使用

**训练模型**：

```bash
# 检查并训练（如果需要）
python core/query_qwen/reasoning_query_qwen.py --train --epochs 50

# 然后查询
python core/query_qwen/reasoning_query_qwen.py --interactive
```

### 场景4：重新训练

**更新模型**：

```bash
# 图谱更新后，重新训练
python core/query_qwen/reasoning_query_qwen.py --train --force-train --epochs 100
```

## 与旧版本的区别

### ✅ 改进点

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 运行方式 | 需要命令行参数 | 点击即可运行 |
| 模型管理 | 每次都可能重新训练 | 智能跳过已训练模型 |
| 用户体验 | 需要记忆参数 | 交互式引导 |
| 文件整合 | 两个文件分开 | 统一入口 |
| 错误提示 | 直接报错 | 友好提示和引导 |

### 🔄 兼容性

**完全向后兼容**，所有旧的命令行参数仍然有效：

```bash
# 旧的 run_reasoning_query.py 命令
python core/reasoning/run_reasoning_query.py --query "..." --method ppr

# 新的 reasoning_query_qwen.py 等效命令
python core/query_qwen/reasoning_query_qwen.py --query "..." --method ppr
```

## 文件变更

### 修改的文件

- ✅ `core/query_qwen/reasoning_query_qwen.py` - 主要更新
  - 新增 `print_results()` 函数
  - 新增 `interactive_mode()` 函数
  - 新增 `command_line_mode()` 函数
  - 改进 `main()` 函数

### 可以删除的文件

- ⚠️ `core/reasoning/run_reasoning_query.py` - 功能已合并，可选择删除
- ⚠️ `core/reasoning/train_reasoning.py` - 功能已合并，可选择删除

**建议**：先保留这些文件作为备份，确认新版本稳定后再删除。

## 常见问题

### Q1: 直接运行没有任何参数会怎样？

A: 自动进入交互模式，引导用户逐步操作。

### Q2: 如何确认模型是否已训练？

A: 检查文件 `data/reasoning/model.pt` 是否存在，或运行任何命令都会显示模型状态。

### Q3: 训练被中断了怎么办？

A: 重新运行 `--train`，会检测到模型不存在（或不完整），自动重新训练。

### Q4: 如何查看帮助信息？

A: 
```bash
python core/query_qwen/reasoning_query_qwen.py --help
```

### Q5: 可以在Jupyter Notebook中使用吗？

A: 可以，导入并使用：
```python
from core.query_qwen.reasoning_query_qwen import ReasoningQueryHandler, load_config

config = load_config()
handler = ReasoningQueryHandler(config, load_trained_model=True)
results = handler.query("Your question", method='ppr')
```

## 总结

新版本的 `reasoning_query_qwen.py` 提供了：
- ✅ 更友好的用户界面（交互模式）
- ✅ 智能的模型管理（自动跳过训练）
- ✅ 灵活的使用方式（交互/命令行/API）
- ✅ 完整的功能整合（查询+训练）
- ✅ 向后兼容性（支持所有旧参数）

推荐使用交互模式进行日常查询，使用命令行模式进行脚本化处理！

