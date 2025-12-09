# 批量处理并行化改进说明

## 改进概述

将原有的**串行批量处理**改为**并行批量处理**，大幅提升处理效率。

## 改进内容

### 1. 实体消歧阶段 (Disambiguation Stage)

**改进前**：
- 依次串行提交每个批次
- 等待一个批次完成后才提交下一个批次
- 总时间 = 批次1时间 + 批次2时间 + ... + 批次N时间

**改进后**：
- **同时提交**所有批次作业
- 使用 `ThreadPoolExecutor` 并行监控所有批次
- 总时间 ≈ max(批次1时间, 批次2时间, ..., 批次N时间)

**代码位置**：`run_disambiguation_stage()` 函数

### 2. 实体合并 - 嵌入向量生成 (Embedding Generation)

**改进前**：
- 依次串行处理每个批次的嵌入请求
- 等待一个批次完成后才处理下一个批次

**改进后**：
- **同时提交**所有批次的嵌入作业
- 并行监控所有批次完成状态
- 效率提升与批次数量成正比

**代码位置**：`run_entity_merge_stage()` 函数（嵌入部分）

### 3. 实体合并 - LLM仲裁 (LLM Arbitration)

**改进前**：
- 依次串行提交LLM仲裁请求
- 等待一个批次完成后才提交下一个批次

**改进后**：
- **同时提交**所有批次的LLM仲裁作业
- 并行监控所有批次完成状态

**代码位置**：`run_entity_merge_stage()` 函数（LLM仲裁部分）

## 技术实现

### 核心函数改造

#### 1. `submit_and_monitor_job()` 函数
- 新增 `monitor` 参数（默认 `True`）
- `monitor=False` 时只提交作业不等待完成
- `monitor=True` 时立即监控直到完成

#### 2. 新增 `_monitor_job_completion()` 函数
- 独立的监控函数，用于并行监控
- 可被 `ThreadPoolExecutor` 调用

#### 3. `_submit_and_monitor_embedding_job()` 函数
- 新增 `monitor` 参数
- 支持分离提交和监控逻辑

#### 4. 新增 `_monitor_embedding_job_completion()` 函数
- 独立的嵌入作业监控函数
- 支持并行监控多个嵌入批次

### 并行处理流程

```python
# 1. 同时提交所有批次
batch_jobs = []
for batch in batches:
    job = submit_job(batch, monitor=False)  # 只提交不等待
    batch_jobs.append(job)

# 2. 并行监控所有批次
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(monitor_job, job): job 
        for job in batch_jobs
    }
    
    for future in as_completed(futures):
        result = future.result()
        # 处理完成的批次
```

## 性能提升

### 理论提升

假设有N个批次，每个批次耗时T：

- **串行处理**：总时间 = N × T
- **并行处理**：总时间 ≈ T + 监控开销

**提升倍数** ≈ N倍（理想情况）

### 实际效果

以 15000 个实体、批次大小 5000 为例：

| 阶段 | 批次数 | 串行时间（估算） | 并行时间（估算） | 提升 |
|------|--------|-----------------|-----------------|------|
| 消歧 | 3批次 | 60分钟 | 25分钟 | 2.4x |
| 嵌入 | 3批次 | 45分钟 | 20分钟 | 2.3x |
| 仲裁 | 2批次 | 40分钟 | 25分钟 | 1.6x |
| **总计** | - | **145分钟** | **70分钟** | **2.1x** |

## 并发控制

### 线程池大小

```python
max_workers=min(len(batch_jobs), 10)
```

- 最多10个并发监控线程
- 避免过多线程导致资源竞争
- 可根据实际情况调整

### API限制考虑

- 批量作业API通常有并发限制
- 建议监控API响应，必要时调整 `max_workers`
- 错误处理机制确保单个批次失败不影响整体

## 日志输出示例

### 并行提交阶段

```
🚀 并行提交 3 个消歧批次作业...
📤 提交消歧批次 1/3
📤 提交消歧批次 2/3
📤 提交消歧批次 3/3
⏳ 开始监控 3 个批次作业的完成状态...
```

### 并行监控阶段

```
⏳ [EntityEmb] [批次 1/3] 轮询 'batch_xxx1' 状态，每 60 秒...
⏳ [EntityEmb] [批次 2/3] 轮询 'batch_xxx2' 状态，每 60 秒...
⏳ [EntityEmb] [批次 3/3] 轮询 'batch_xxx3' 状态，每 60 秒...
```

### 批次完成阶段

```
✅ 嵌入批次 2 已完成
✅ 嵌入批次 1 已完成
✅ 嵌入批次 3 已完成
📊 批次 1 获得 5000 个嵌入向量
📊 批次 2 获得 5000 个嵌入向量
📊 批次 3 获得 5000 个嵌入向量
🎉 所有嵌入批次处理完成，共获得 15000 个嵌入向量
```

## 配置要求

无需额外配置，使用原有配置即可：

```yaml
graph_builder:
  embedding_batch_size: 5000  # 批次大小
```

## 代码改动总结

### 新增导入

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

### 修改的函数

1. `submit_and_monitor_job()` - 新增 `monitor` 参数
2. `_submit_and_monitor_embedding_job()` - 新增 `monitor` 参数
3. `run_disambiguation_stage()` - 改为并行提交
4. `run_entity_merge_stage()` - 嵌入和仲裁都改为并行

### 新增的函数

1. `_monitor_job_completion()` - LLM作业监控
2. `_monitor_embedding_job_completion()` - 嵌入作业监控

## 向后兼容性

✅ **完全兼容**

- 保持原有函数签名（除了新增可选参数）
- 单批次情况下行为不变
- 配置文件无需修改

## 错误处理

### 单个批次失败

- 不影响其他批次继续处理
- 记录错误日志
- 只处理成功完成的批次结果

### 全部批次失败

- 返回空结果
- 记录详细错误信息
- 流程继续（跳过该阶段）

## 测试建议

### 功能测试

1. 单批次场景（< embedding_batch_size）
2. 多批次场景（> embedding_batch_size）
3. 部分批次失败场景

### 性能测试

1. 记录串行vs并行的实际耗时
2. 监控API调用频率
3. 观察系统资源占用

## 注意事项

1. **API配额**：并行提交会快速消耗API配额，需注意监控
2. **网络稳定性**：并发监控需要稳定的网络连接
3. **日志输出**：并行执行时日志可能交织，使用批次标签区分
4. **资源占用**：并发线程会占用一定内存，通常可忽略

## 后续优化建议

1. 添加自适应并发控制（根据API响应动态调整）
2. 支持批次优先级（重要批次优先处理）
3. 添加批次重试机制（失败批次自动重试）
4. 提供并发度配置选项（让用户控制max_workers）

---

**实现日期**：2025-01-09  
**改进版本**：v2.0 - 并行批量处理

