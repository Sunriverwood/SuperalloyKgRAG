# 文档目录

本目录包含 SuperalloyKgRAG 项目的所有技术文档，专注于系统架构和工作流程说明。

## 📖 文档索引

### 🏗️ 核心架构文档

| 文档 | 内容描述 |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | **项目架构总览** - 系统架构图、目录结构、四种查询模式详细流程图（LOCAL/GLOBAL/REASONING/DRIFT） |
| [DEPENDENCIES.md](DEPENDENCIES.md) | **依赖说明** - 项目所需的所有依赖包及其用途 |
| [ID_MANAGEMENT_GUIDE.md](ID_MANAGEMENT_GUIDE.md) | **ID管理指南** - 图谱构建、存储、查询全流程的ID生成、转换和使用详解 |
| [ID_FLOW_DIAGRAM.md](ID_FLOW_DIAGRAM.md) | **ID流转全景图** - 可视化展示ID从文档到查询的完整生命周期 |
| [ID_QUICK_REFERENCE.md](ID_QUICK_REFERENCE.md) | **ID快速参考卡** - 常用函数、问题排查、最佳实践速查手册 |

### 📥 数据处理流程

| 文档 | 内容描述 |
|------|---------|
| [RUN_INDEXING_GUIDE.md](RUN_INDEXING_GUIDE.md) | **索引流水线指南** - PDF → JSON → 文本块 → 三元组 → 知识图谱 → 向量存储的完整流程 |

### 🔍 查询系统流程

| 文档 | 内容描述 |
|------|---------|
| [REASONING_GUIDE.md](REASONING_GUIDE.md) | **图推理系统指南** - 架构、自监督训练、PPR/GNN 推理流程、使用方法 |

### 📊 评测系统

| 文档 | 内容描述 |
|------|---------|
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | **评测系统指南** - 多级评分机制（L1-L4）、使用方法、评测报告详解 |

### 🛠️ 工具与集成

| 文档 | 内容描述 |
|------|---------|
| [IMPORT_TO_NEO4J.md](IMPORT_TO_NEO4J.md) | **Neo4j 导入指南** - 如何将知识图谱导入 Neo4j 进行可视化 |

## 🚀 快速入门

### 新用户推荐阅读顺序

1. **了解架构** 📖
   - 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解项目整体架构
   - 查看系统流程图和查询模式说明

2. **安装依赖** 📦
   - 参考 [DEPENDENCIES.md](DEPENDENCIES.md) 安装所需包
   ```bash
   pip install -r requirements.txt
   ```

3. **构建知识图谱** 🔨
   - 按照 [RUN_INDEXING_GUIDE.md](RUN_INDEXING_GUIDE.md) 执行索引流水线
   ```bash
   python app/run_indexing.py
   ```

4. **开始查询** 🔍
   - 使用统一查询入口（路由器会自动选择最优查询模式）
   ```bash
   python core/query_qwen/router_qwen.py
   ```

5. **系统评测** 📊
   - 参考 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) 运行评测系统
   ```bash
   python evaluation/auto_evaluator.py --difficulty L3
   ```

6. **高级功能** 🚀
   - 阅读 [REASONING_GUIDE.md](REASONING_GUIDE.md) 了解图推理功能

## 📊 查询模式说明

项目支持四种智能查询模式，由路由器自动判断：

| 模式 | 适用场景 | 流程图位置 |
|------|---------|-----------|
| **LOCAL** | 特定实体属性查询 | [ARCHITECTURE.md § LOCAL 查询](ARCHITECTURE.md#local-局部查询) |
| **GLOBAL** | 摘要、概览类查询 | [ARCHITECTURE.md § GLOBAL 查询](ARCHITECTURE.md#global-全局查询) |
| **REASONING** | 多跳推理、关系发现 | [ARCHITECTURE.md § REASONING 查询](ARCHITECTURE.md#reasoning-图推理查询) + [REASONING_GUIDE.md](REASONING_GUIDE.md) |
| **DRIFT** | 复杂、迭代细化查询 | [ARCHITECTURE.md § DRIFT 查询](ARCHITECTURE.md#drift-漂移搜索) |
