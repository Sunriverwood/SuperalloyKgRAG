# 图推理查询系统更新日志

## 版本 2.0 - 2025-12-02

### 🎉 主要更新

#### 1. 统一入口 + 交互模式
- **合并文件**：`run_reasoning_query.py` 和 `train_reasoning.py` 的功能已整合到 `reasoning_query_qwen.py`
- **交互模式**：直接点击运行即可使用，无需记忆命令行参数
- **友好界面**：逐步引导用户输入查询、选择方法、保存结果

#### 2. 智能模型管理 🧠
- **自动检测**：系统自动检测 `data/reasoning/model.pt` 是否存在
- **跳过训练**：如果模型已存在，`--train` 命令会自动跳过训练
- **智能提示**：清晰提示模型状态，避免不必要的重复训练

#### 3. 灵活的使用方式
支持三种模式，满足不同使用场景：

**模式1：交互模式**（推荐日常使用）
```bash
python core/query_qwen/reasoning_query_qwen.py
# 或
python core/query_qwen/reasoning_query_qwen.py --interactive
```

**模式2：命令行模式**（批量查询）
```bash
python core/query_qwen/reasoning_query_qwen.py --query "..."
```

**模式3：训练模式**（首次使用或更新模型）
```bash
python core/query_qwen/reasoning_query_qwen.py --train
```

### 🔧 新增功能

#### 命令行参数

| 参数 | 功能 | 示例 |
|------|------|------|
| `--interactive` / `-i` | 启动交互模式 | `-i` |
| `--train` | 训练模式（智能跳过） | `--train --epochs 100` |
| `--force-train` | 强制重新训练 | `--train --force-train` |
| `--query` / `-q` | 查询文本 | `-q "What is nickel?"` |
| `--method` / `-m` | 推理方法 | `-m ppr` / `-m gnn` |
| `--no-llm` | 跳过LLM生成 | `--no-llm` |
| `--output` / `-o` | 保存结果 | `-o result.json` |
| `--epochs` / `-e` | 训练轮数 | `-e 100` |

#### 函数更新

**新增函数**：
- `print_results()` - 统一的结果格式化输出
- `interactive_mode()` - 交互式问答循环
- `command_line_mode()` - 命令行参数解析和处理

**改进函数**：
- `main()` - 现在是统一的入口点，处理所有模式

### 🎨 用户体验改进

#### 训练智能跳过

**之前**：
```bash
python core/reasoning/train_reasoning.py  # 总是训练
```

**现在**：
```bash
python core/query_qwen/reasoning_query_qwen.py --train
# 输出：
# ✓ Model already exists at: .../model.pt
#   Use --force-train to retrain anyway.
#   Skipping training.
```

#### 交互式引导

**之前**：需要记忆并输入完整命令
```bash
python core/reasoning/run_reasoning_query.py \
    --query "What is nickel used for?" \
    --method ppr \
    --output result.json
```

**现在**：逐步交互式输入
```
Enter your query: What is nickel used for?
Reasoning method (ppr/gnn) [ppr]: ppr
Generate LLM answer? (yes/no) [yes]: yes
Save results to file? (yes/no) [no]: yes
Output file path: result.json
```

#### 无参数运行

**之前**：没有参数会报错
```bash
python core/reasoning/run_reasoning_query.py
# error: the following arguments are required: --query
```

**现在**：自动进入交互模式
```bash
python core/query_qwen/reasoning_query_qwen.py
# Starting interactive mode...
```

### 📝 向后兼容

**完全兼容旧的命令行参数**：

| 旧命令 | 新命令（等效） | 状态 |
|--------|---------------|------|
| `run_reasoning_query.py --query "..."` | `reasoning_query_qwen.py --query "..."` | ✅ 兼容 |
| `train_reasoning.py --epochs 100` | `reasoning_query_qwen.py --train --epochs 100` | ✅ 兼容 |

### 🗂️ 文件变更

**修改的文件**：
- ✅ `core/query_qwen/reasoning_query_qwen.py` - 主要更新

**新增的文件**：
- ✅ `docs/REASONING_QUERY_GUIDE.md` - 详细使用指南
- ✅ `demo_reasoning_usage.py` - 使用方式演示脚本

**可选删除的文件**（功能已合并）：
- ⚠️ `core/reasoning/run_reasoning_query.py`
- ⚠️ `core/reasoning/train_reasoning.py`

建议：先保留作为备份，确认新版本稳定后再删除。

### 🐛 Bug修复

- 修复：单节点路径显示为空的问题（已在之前版本修复）
- 改进：路径格式化函数，更清晰地显示单节点路径
- 优化：诊断日志，帮助排查图谱连通性问题

### 📖 文档更新

**新增文档**：
1. `docs/REASONING_QUERY_GUIDE.md` - 完整使用指南
   - 三种运行模式详解
   - 智能模型管理说明
   - 所有参数的详细说明
   - 常见使用场景示例

2. `docs/ALLOW_SINGLE_NODE_PATHS.md` - 单节点路径说明
   - 为什么允许单节点路径
   - 如何解读结果
   - 图谱质量诊断方法

3. `demo_reasoning_usage.py` - 快速演示脚本
   - 展示所有使用方式
   - 检查模型状态
   - 提供快速开始指引

### 🚀 性能优化

- 智能跳过训练：避免重复训练已存在的模型
- 统一模型加载：所有模式共用同一加载逻辑
- 优化日志输出：更清晰的进度提示

### 🔍 已知问题

1. **图谱连通性**：某些查询可能只返回单节点路径（非代码问题，是图谱质量问题）
2. **GPU支持**：当前PyTorch为CPU版本，需要手动安装CUDA版本以使用GPU

### 📊 使用建议

**推荐工作流程**：

1. **首次使用**：
   ```bash
   python core/query_qwen/reasoning_query_qwen.py --train
   ```

2. **日常查询**：
   ```bash
   python core/query_qwen/reasoning_query_qwen.py
   # 进入交互模式，输入问题
   ```

3. **批量处理**：
   ```bash
   python core/query_qwen/reasoning_query_qwen.py -q "..." -o result.json
   ```

4. **更新模型**（图谱更新后）：
   ```bash
   python core/query_qwen/reasoning_query_qwen.py --train --force-train
   ```

### 🎯 下一步计划

- [ ] 添加批量查询支持（读取CSV/JSON文件）
- [ ] 实现查询历史记录
- [ ] 添加结果比较功能（PPR vs GNN）
- [ ] 优化图谱连通性检测
- [ ] 支持图谱可视化

### 💡 反馈和贡献

如有问题或建议，请：
1. 查看文档：`docs/REASONING_QUERY_GUIDE.md`
2. 运行演示：`python demo_reasoning_usage.py`
3. 检查日志：`logs/reasoning_query.log`

---

**版本**: 2.0  
**日期**: 2025-12-02  
**作者**: Graph Reasoning Team  
**兼容性**: 向后兼容 v1.0

