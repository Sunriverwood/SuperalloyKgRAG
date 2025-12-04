# SuperalloyKgRAG: 基于知识图谱的高温合金领域 RAG 推理框架

## 1. 项目概述

**SuperalloyKgRAG** 是一个针对高温合金（Superalloys）领域的垂直检索增强生成（RAG）系统。本项目旨在解决传统文本切片 RAG 在处理复杂材料科学问题时的局限性。

通过引入多模态解析（OCR/VLM）与图神经网络（GNN），本框架实现了从非结构化 PDF 文档到结构化知识图谱的端到端构建。系统内置**无监督图推理引擎**，利用 RGAT（关系图注意力网络）与 Personalized PageRank 算法，支持多跳推理（Multi-hop Reasoning）与可解释性路径生成，有效提升了针对“成分-工艺-结构-性能”复杂关联问题的回答准确率。

### 核心特性

1.  **端到端数据流水线**：利用多模态大模型（VLM/OCR）处理原始 PDF 文档，将其转化为结构化的实体与关系三元组。
2.  **知识图谱构建**：自动化进行实体消歧、合并及社区发现（Community Detection），生成高质量的领域知识图谱。
3.  **图推理引擎**：基于 RGAT（关系图注意力网络）和个性化 PageRank（PPR）算法，实现多跳路径推理，能够回答复杂的“关系型”问题。
4.  **混合检索**：结合了向量检索（LanceDB）与图结构检索（Neo4j/NetworkX），提供可解释的推理路径。

------

## 2. 环境依赖与安装

本框架基于 Python 开发，推荐在 **Windows** 或 **Linux** 环境下运行。

### 前置要求

- **Python**: 3.12
- **IDE**: PyCharm (推荐) 或 VS Code
- **硬件**: 建议配置 NVIDIA GPU (支持 CUDA) 以加速推理模型训练；支持 CPU 运行。
- **数据库**:
  - **LanceDB**: 用于向量存储 (内嵌式，无需独立安装)。
  - **Neo4j**: (可选) 仅用于图谱可视化分析，不影响核心推理功能。

### 部署步骤

1. **克隆/下载项目代码**

   ```bash
   git clone [repository_url]
   cd SuperalloyKgRAG
   ```

2. **推荐使用 Conda 或 venv 创建虚拟环境**

   ```bash
   # venv 配置方法
   # 创建虚拟环境
   python -m venv venv
   # 激活环境 (Windows)
   venv\Scripts\activate
   # 激活环境 (Linux/Mac)
   source venv/bin/activate
   
   
   # conda 配置方法
   # 1. 创建名为 superalloy 的虚拟环境，指定 Python 3.12
   conda create -n superalloy python=3.12 -y
   
   # 2. 激活环境
   conda activate superalloy
   
   # 3. (可选但推荐) 安装 Intel MKL 数学核心库以优化科学计算性能
   # 注意：这是 requirements.txt 中建议的步骤，用于加速 numpy 和 pytorch
   conda install mkl mkl_fft mkl_random mkl-service -y
   ```

   

3. **安装依赖库**

   ```bash
   pip install -r requirements.txt
   ```

------

## 3. 配置与参数

系统核心配置位于 `config/settings.yaml`。在运行前，请完成 LLM 服务的鉴权配置。

### API 密钥配置

本系统支持 OpenAI 兼容接口（如 Qwen, Gemini）。请将 API Key 注入环境变量，或直接修改配置文件。

**方式一：系统环境变量 (推荐)**

- **Windows**: 在“编辑系统环境变量”中新建，变量名 `QWEN_API_KEY`，值为你的密钥。
- **Linux/Mac**: `export QWEN_API_KEY="sk-..."`

**方式二：IDE 运行配置**

- 在 PyCharm/VS Code 的 Run/Debug Configurations 中，添加 Environment variables: `QWEN_API_KEY=sk-...`。

------

## 4. 数据索引流水线 (Indexing Pipeline)

该模块负责将原始 PDF 文档转化为知识图谱与向量索引。

### 数据准备

将待处理的 PDF 文献放入 `data/raw_pdfs/` 目录。

### 执行索引

在 IDE 中运行 `app/run_indexing.py` 或使用命令行：

```bash
python app/run_indexing.py
```

**流水线阶段说明：** 系统将自动按序执行以下处理：

1. **VLM Parsing**: 基于视觉大模型解析 PDF，保留版面结构信息。
2. **Chunking**: 语义级文本切分。
3. **Triple Extraction**: 抽取实体与关系 (Entity-Relation Extraction)。
4. **Graph Construction**: 实体消歧、融合及社区发现 (Community Detection)。
5. **Vector Embedding**: 生成文本与图谱节点的向量表示 (LanceDB)。

*注：支持断点续传，若中断可再次运行，脚本将自动跳过已完成步骤。*

------

## 5. 推理模型训练 (Graph Reasoning)

在图谱构建完成后，需训练图推理模型以捕捉节点间的潜在语义关联。本模块采用自监督学习方案，无需人工标注数据。

### 启动训练

运行 `core/reasoning/train_reasoning.py`：

```bash
# 推荐使用 GPU
python core/reasoning/train_reasoning.py --epochs 100 --device cuda

# CPU 训练
python core/reasoning/train_reasoning.py --epochs 100 --device cpu
```

训练产出的模型权重将保存至 `data/reasoning/model.pt`。

------

## 6. 查询与推理 (Inference)

系统提供交互式与命令行两种查询模式，支持输出推理路径与参考文献。

### 模式一：IDE 交互式查询 (调试推荐)

1. 在 IDE 中打开 `core/reasoning/run_reasoning_query.py`。

2. 右键点击 **Run** (或 `Shift+F10`)。

3. 在控制台输入问题，例如：

   > *Inconel 718 的主要强化相是什么？*

4. 系统将实时输出：

   - **Top Entities**: 召回的关键实体。
   - **Reasoning Paths**: 推理链条 (e.g., *Inconel 718 -> contains -> Niobium -> forms -> Gamma'' Phase*).
   - **Answer**: 综合生成的最终回复。

### 模式二：命令行批处理

适用于批量测试或评估：

```bash
# 单次查询
python core/reasoning/run_reasoning_query.py --query "高温合金蠕变机制有哪些？"

# 指定推理算法 (支持 ppr / gnn) 并保存结果
python core/reasoning/run_reasoning_query.py \
    --query "Re 元素对单晶高温合金的影响" \
    --method ppr \
    --output results.json
```

------

## 7. 项目文件结构详解

为了便于二次开发与深度理解，以下列出项目中关键文件的功能说明。

### app/ (应用层)

- **`run_indexing.py`**: **[核心入口]** 索引流水线主程序。负责编排从 PDF 解析到向量入库的完整流程，支持断点续传。
- **`run_index_qwen.py`**: 针对 Qwen 模型的特定索引启动脚本，预设了 Qwen 相关的参数配置。
- **`API_test.py`**: 简单的 API 连通性测试脚本，用于验证 Key 是否有效。
- **`cloud_manager.py`**: 云存储管理工具，处理与 Google Cloud Storage 等云服务的交互（如需）。
- **`formatting.py`**: 用于将final_graph.json变为可以导入Neo4j的json格式。

### config/ (配置层)

- **`settings.yaml`**: **[全局配置]** 包含所有模块的参数设置（API Key、模型名称、路径、阈值等）。
- **`prompts/\*.md`**: 存放各类任务的 Prompt 模板（如 `text_to_graph.md` 用于三元组抽取，`basic_rag.md` 用于问答）。

### core/ (核心算法层)

#### vlm_pdf_parser (文档解析)

- **`vlm_pdf_parser.py`**: 基于视觉语言模型 (VLM) 的通用 PDF 解析器，处理图表和复杂排版。
- **`vlm_pdf_parser_qwen.py`**: 针对 Qwen-VL 优化的 PDF 解析实现。

#### pipeline/ (通用索引流水线组件)

- **`loader.py`**: 数据加载与切分模块，将 JSON 数据转换为文本块 (Chunks)。
- **`extraction.py`**: 信息抽取模块，调用 LLM 从文本块中提取实体和关系。
- **`graph_builder.py`**: 图谱构建模块，处理实体对齐、消歧、合并及社区发现算法。
- **`embedding.py`**: 向量化模块，调用 Embedding API 为文本和图节点生成向量。
- **`embedding_basic.py`**: 基础向量化实现，仅处理文本向量。

#### pipeline_qwen/ (Qwen 专用流水线)

- 包含 `loader.py`, `extraction_qwen.py`, `graph_builder_qwen.py` 等，均为针对 Qwen 模型特性的定制化实现，功能与 `pipeline/` 对应。

#### query/ & query_qwen/ (查询与检索)

- **`router.py` / `router_qwen.py`**: 意图路由模块，判断用户问题类型（全局、局部或推理）。
- **`basic_rag.py` / `basic_rag_qwen.py`**: 基础 RAG 检索实现，仅基于向量相似度。
- **`global_query.py`**: 全局查询实现，基于社区摘要回答宏观问题。
- **`local_query.py`**: 局部查询实现，基于实体邻域信息回答具体问题。
- **`reasoning_query_qwen.py`**: 推理查询实现，集成图推理功能，执行“检索-推理-生成”全流程。

#### reasoning/ (图推理引擎)

- **`data_loader.py`**: 图数据加载器，负责从 JSON/DB 加载图谱并转化为 PyTorch Geometric 数据格式。
- **`models/rgat.py`**: **[模型定义]** 关系图注意力网络 (RGAT) 的模型架构代码。
- **`training/trainer.py`**: **[训练逻辑]** 实现自监督训练循环，包括负采样、损失计算 (Link Prediction/Contrastive) 和反向传播。
- **`inference/reasoner.py`**: **[推理逻辑]** 实现推理时的节点打分、PPR 传播算法及路径搜索策略。

### utils/ (工具库)

- **`client_factory.py`**: gemini客户端工厂模式实现。
- **`community_importance.py`**: 计算图谱中社区和节点重要性的算法工具。
- **`clean_embedding_text.py`**: 文本预处理脚本，用于清洗脏数据以提升 Embedding 质量。
- **`graph_reasoning_utils.py`**: 图推理相关的辅助函数库（如 Adjacency Mask 生成）。
- **`local_context.py`**: 用于构建查询时的局部上下文窗口。

### data/ (数据存储 - 自动生成)

- `raw_pdfs/`: 存放原始 PDF 输入文件。
- `processed_jsons/`: OCR 解析后的中间 JSON 文件。
- `chunks/`: 文本切片文件 (`text_units.jsonl`)。
- `graphs/`: 构建完成的知识图谱 (`final_graph.json`)。
- `embeddings/`: LanceDB 向量数据库文件。
- `reasoning/`: 存放训练好的模型权重 (`model.pt`) 和推理结果。