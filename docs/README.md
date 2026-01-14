# 文档目录

本目录包含 SuperalloyKgRAG 项目的所有技术文档，专注于系统架构和工作流程说明。

## 📖 文档索引

### 🏗️ 核心架构文档

| 文档 | 内容描述 |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | **项目架构总览** - 系统架构图、目录结构、四种查询模式详细流程图（LOCAL/GLOBAL/REASONING/DRIFT） |
| [DEPENDENCIES.md](DEPENDENCIES.md) | **依赖说明** - 项目所需的所有依赖包及其用途 |

### 📥 数据处理流程

| 文档 | 内容描述 |
|------|---------|
| [RUN_INDEXING_GUIDE.md](RUN_INDEXING_GUIDE.md) | **索引流水线指南** - PDF → JSON → 文本块 → 三元组 → 知识图谱 → 向量存储的完整流程 |
| [不同类型图谱的三元组提取方式详解.md](不同类型图谱的三元组提取方式详解.md) | **三元组提取详解** - 纯文本、摘要、图片、表格四种图谱的提取方式 |

### 📊 知识图谱构建

| 文档 | 内容描述 |
|------|---------|
| [ENTITY_MERGE_GUIDE.md](ENTITY_MERGE_GUIDE.md) | **实体合并指南** - 基于嵌入相似度+LLM判断的实体消歧合并机制 |
| [HNSW_USAGE_GUIDE.md](HNSW_USAGE_GUIDE.md) | **HNSW优化指南** - 大规模实体合并的近似最近邻算法配置 |
| [HIERARCHICAL_COMMUNITIES_GUIDE.md](HIERARCHICAL_COMMUNITIES_GUIDE.md) | **分层社区指南** - 递归Leiden社区发现、层级结构、配置说明 |
| [COMMUNITY_REPORT_GENERATION.md](COMMUNITY_REPORT_GENERATION.md) | **社区报告生成** - 多层社区报告生成方法、叶子/父社区策略 |

### 🔍 查询系统

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
| [IMPORT_TO_GEPHI.md](IMPORT_TO_GEPHI.md) | **Gephi 可视化指南** - 使用Gephi进行知识图谱可视化分析 |

---

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

---

## 📊 查询模式说明

项目支持四种智能查询模式，由路由器自动判断：

| 模式 | 适用场景 | 流程图位置 |
|------|---------|-----------|
| **LOCAL** | 特定实体属性查询 | [ARCHITECTURE.md § LOCAL 查询](ARCHITECTURE.md#local-局部查询) |
| **GLOBAL** | 摘要、概览类查询 | [ARCHITECTURE.md § GLOBAL 查询](ARCHITECTURE.md#global-全局查询) |
| **REASONING** | 多跳推理、关系发现 | [ARCHITECTURE.md § REASONING 查询](ARCHITECTURE.md#reasoning-图推理查询) + [REASONING_GUIDE.md](REASONING_GUIDE.md) |
| **DRIFT** | 复杂、迭代细化查询 | [ARCHITECTURE.md § DRIFT 查询](ARCHITECTURE.md#drift-漂移搜索) |

---

## 📁 文档更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-14 | 整理文档结构，删除过时的修复/调试文档，保留13个核心文档 |
