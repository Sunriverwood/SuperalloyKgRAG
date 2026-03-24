# SuperalloyKgRAG 依赖说明

> 更新时间：2026-03-24（与当前主流程入口对齐）
> 
> 说明：本轮文档整理不包含 `draw/` 与 `visualizations/` 目录。

## 概述

本文档按**当前主代码链路**整理依赖（索引、查询、推理、评测）。

- 依赖来源：`app/`、`core/`、`evaluation/`、`utils/` 中实际导入
- 明确排除：`draw/`、`visualizations/` 目录
- 安装入口：`requirements.txt`

## 快速安装

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 核心运行依赖

以下为主流程运行所需依赖：

- `PyYAML`：读取 `config/settings.yaml`
- `openai`：Qwen/OpenAI 兼容接口调用
- `google-genai`、`google-api-core`：Gemini 与批处理相关接口
- `numpy`、`pandas`：数据处理与数值计算
- `networkx`：图结构构建与路径处理
- `igraph`、`leidenalg`：社区发现
- `lancedb`：向量存储与检索
- `hnswlib`：实体合并 ANN 加速
- `pydantic`：数据模型与校验
- `torch`、`scipy`：图推理训练与推理
- `pyarrow`：LanceDB/Pandas 相关数据格式支持

## 可选依赖

以下依赖仅在特定工具或增强流程中使用：

- `scikit-learn`、`hdbscan`：`utils/community_clustering.py` 中的聚类对比
- `matplotlib`：`utils/entity_merge_review.py` 中的人工复核可视化

## 与代码入口的对应关系

- 索引主入口：`app/run_index_qwen.py`
- 查询主入口：`core/query_qwen/router_qwen.py`
- 推理入口（训练/查询）：`core/query_qwen/reasoning_query_qwen.py`
- 评测入口：`evaluation/auto_evaluator.py`

## 版本策略

当前 `requirements.txt` 采用“核心依赖固定版本”的策略，目标是降低环境漂移风险。

如果你需要更灵活的升级策略，可在分支中改为 `>=` 约束并配套回归测试。

## 验证建议

安装后建议先验证 CLI 是否可用：

```bash
python app/run_index_qwen.py --help
$env:PYTHONPATH="."; python core/query_qwen/router_qwen.py --help
python core/query_qwen/reasoning_query_qwen.py --help
python evaluation/auto_evaluator.py --help
```
