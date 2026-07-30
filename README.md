# SuperalloyKgRAG: 基于知识图谱的高温合金领域 RAG 推理框架

> 文档状态：已于 2026-03-24 对齐当前主流程（索引、查询、推理、评测）。
> 
> 文档维护范围：`README.md` 与 `docs/`；本次更新按要求忽略 `draw/` 与 `visualizations/` 目录内容。

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
  - **Gephi**: (可选) 仅用于图谱可视化分析，不影响核心推理功能。

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

   如需完整复现当前混合环境（含 conda 包 + pip 包），优先使用：

   ```bash
   conda env create -f environment.yml
   conda activate deepseek
   ```

------

## 3. 配置与参数

系统核心配置位于 `config/settings.yaml`。在运行前，请完成 LLM 服务的鉴权配置。

### API 密钥配置

本系统支持 OpenAI 兼容接口（如 Qwen, Gemini），但需要特别注意：

- **OCR/PDF 解析阶段目前仅支持 Gemini**（`core/vlm_pdf_parser.py`）。
- **Qwen 当前不支持直接输入 PDF 做 OCR**，因此不能替代步骤1的 PDF 解析。

请将 API Key 注入环境变量，或直接修改配置文件。

**方式一：系统环境变量 (推荐)**

- **Windows**: 在“编辑系统环境变量”中新建，变量名 `GEMINI_API_KEY`（OCR 必需）和 `QWEN_API_KEY`（Qwen 查询可选）。
- **Linux/Mac**: `export GEMINI_API_KEY="..."`，如需 Qwen 再配置 `export QWEN_API_KEY="sk-..."`

**方式二：IDE 运行配置**

- 在 PyCharm/VS Code 的 Run/Debug Configurations 中，至少添加 `GEMINI_API_KEY=...`（用于 OCR）。若使用 Qwen 查询，再添加 `QWEN_API_KEY=sk-...`。

------

## 4. 数据索引流水线 (Indexing Pipeline)

该模块负责将原始 PDF 文档转化为知识图谱与向量索引。

### 数据准备

默认请将待处理 PDF 放入 `data/original_data/books/` 目录（可在 `config/settings.yaml` 的 `vlm_parser.input_dir` 中修改）。

### 执行索引

在 IDE 中运行 `app/run_index_qwen.py` 或使用命令行：

```bash
python app/run_index_qwen.py
```

**流水线阶段说明（Qwen 主流程）：** 系统将自动按序执行以下处理：

1. **VLM Parsing**: 基于视觉大模型解析 PDF，保留版面结构信息。
2. **Enrich Extraction**: 融合文本/摘要/图片/表格抽取结果，生成富化图谱。
3. **Graph Construction**: 实体消歧、融合及社区发现 (Community Detection)。
4. **Vector Embedding**: 生成图谱节点与社区向量表示 (LanceDB)。

*注：支持断点续传，若中断可再次运行，脚本将自动跳过已完成步骤。*

*补充：步骤1当前通过 Gemini API 执行 PDF OCR，Qwen API 不支持直接 PDF 输入。*

------

## 5. 推理模型训练 (Graph Reasoning)

在图谱构建完成后，需训练图推理模型以捕捉节点间的潜在语义关联。本模块采用自监督学习方案，无需人工标注数据。

### 启动训练

推荐通过 `core/query_qwen/reasoning_query_qwen.py` 的训练模式启动：

```bash
# 推荐使用 GPU
python core/query_qwen/reasoning_query_qwen.py --train --epochs 100 --device cuda

# CPU 训练
python core/query_qwen/reasoning_query_qwen.py --train --epochs 100 --device cpu
```

训练产出的模型权重路径由 `config/settings.yaml` 的 `reasoning.output.model_path` 控制（当前默认：`data/reasoning/develop.pt`）。

------

## 6. 查询与推理 (Inference)

系统推荐统一使用路由入口，支持交互式与命令行两种查询模式，并可在需要时切换到推理专用入口。

### 模式一：IDE 交互式查询 (调试推荐)

1. 在 IDE 中打开 `core/query_qwen/router_qwen.py`。

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
# 单次查询（自动路由，需在项目根目录执行）
$env:PYTHONPATH="."; python core/query_qwen/router_qwen.py --query "高温合金蠕变机制有哪些？"

# 强制推理模式
$env:PYTHONPATH="."; python core/query_qwen/router_qwen.py --mode reasoning --query "Re 元素对单晶高温合金的影响"

# 调试用：直接调用推理模块并指定算法
python core/query_qwen/reasoning_query_qwen.py --query "Re 元素对单晶高温合金的影响" --method ppr --output results.json
```

------

## 7. 系统评估 (Evaluation)

本项目内置了一套完整的自动评估系统，用于测试 RAG 系统在不同难度问题上的表现。评估系统支持并发执行、多维度打分和详细的结果分析。

### 评估数据准备

评估问题集存放在 `data/evaluation_sets/`：

- **L12.json**: L1（事实）+ L2（简单推理）
- **L3.json / L4.json**: 综合分析与设计/发现
- **hard.json**: 教材深推理子集

完整数据布局见 [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)。

### 运行评估

```bash
# 单方法自动评测
python evaluation/auto_evaluator.py --difficulty L3

# 多维对比（写入命名实验目录，推荐）
python -m evaluation.multidimensional_evaluator --run-dir new-baseline

# 消融实验（自动写入 ablation_<name>/）
python -m evaluation.multidimensional_evaluator --ablation text_only --methods local,reasoning

# 对已有答案重打分 / 分层分析
python -m evaluation.rescore --answers_dir new-baseline
python -m evaluation.rescore_level_analysis --dir new-baseline
```

### 评估指标

系统会自动生成多维度评估报告，包括：

- **准确性 (Accuracy)**: 答案与标准答案的匹配度
- **完整性 (Completeness)**: 是否涵盖所有关键信息点
- **推理路径质量**: 推理链条的合理性和可解释性
- **召回率**: 检索到的相关实体/文档的覆盖率

多维评测答案按实验子目录保存在  
`data/answers/multidimensional_evaluation/<run_dir>/`（如 `old-baseline`、`new-baseline`、`ablation_*`）。

------

## 8. 知识图谱可视化

### 导出到 Gephi

[Gephi](https://gephi.org/) 是一款专业的图可视化工具，适合用于展示知识图谱的宏观结构、社区聚类和节点重要性。

#### 导出步骤

1. **生成 GEXF 格式文件**

   运行图谱转换脚本：

   ```bash
   python app/gephi.py
   ```

   该脚本会将 `data/graphs/final_graph.json` 转换为 `final_graph.gexf` 文件，保留所有节点属性（社区ID、度中心性等）。

2. **导入到 Gephi**

   - 打开 Gephi，点击 **"Open Graph File..."**
   - 选择生成的 `.gexf` 文件
   - Graph Type 选择 **"Directed"**（有向图）
   - 点击 **OK** 完成导入

3. **可视化美化**

   推荐的可视化流程：
   
   - **布局算法**: 使用 **ForceAtlas 2**，勾选 "Prevent Overlap" 和 "Dissuade Hubs"
   - **节点颜色**: 按 **Modularity Class**（社区）着色
   - **节点大小**: 按 **Degree**（度数）或 **PageRank** 调整大小
   - **边透明度**: 设置为 20-30% 以避免视觉混乱
   - **导出**: 使用 PDF/SVG 矢量格式，适合学术论文插图

**详细教程**: 参见 [docs/IMPORT_TO_GEPHI.md](docs/IMPORT_TO_GEPHI.md)

### 导出到 Neo4j

Neo4j 是一款图数据库，支持 Cypher 查询语言，适合进行交互式图谱探索和复杂查询。

#### 导出步骤

1. **格式转换**

   ```bash
   python app/formatting.py
   ```

   将 `final_graph.json` 转换为 Neo4j 兼容的格式。

2. **导入 Neo4j**

   详细导入步骤和 Cypher 查询示例，请参见 [docs/IMPORT_TO_NEO4J.md](docs/IMPORT_TO_NEO4J.md)

------

## 9. 项目文件结构详解

为了便于二次开发与深度理解，以下列出项目中关键文件的功能说明。

### app/ (应用层)

- **`run_index_qwen.py`**: **[核心入口]** Qwen 主索引流水线程序（4 步），支持断点续传与消融配置。
- **`run_indexing.py`**: 通用索引流水线脚本（兼容保留入口）。
- **`API_test.py`**: 简单的 API 连通性测试脚本，用于验证 Key 是否有效。
- **`cloud_manager.py`**: Gemini 文件与批处理作业管理脚本（清理上传文件、作业结果等）。
- **`formatting.py`**: 用于将final_graph.json变为可以导入Neo4j的json格式。

### config/ (配置层)

- **`settings.yaml`**: **[全局配置]** 包含所有模块的参数设置（API Key、模型名称、路径、阈值等）。
- **`prompts/\*.md`**: 存放各类任务的 Prompt 模板（如 `text_to_graph.md` 用于三元组抽取，`basic_rag.md` 用于问答）。

### core/ (核心算法层)

#### vlm_pdf_parser (文档解析)

- **`vlm_pdf_parser.py`**: 基于视觉语言模型 (VLM) 的通用 PDF 解析器，处理图表和复杂排版。
- **`paper_pdf_parser.py`**: 面向论文全文场景的 PDF 解析实现（与 `vlm_pdf_parser.py` 输入目录可分别配置）。

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

### evaluation/ (评估系统)

- **`multidimensional_evaluator.py`**: **[多维/消融入口]** 支持 `--run-dir`、`--ablation`，答案写入实验子目录。
- **`auto_evaluator.py`**: 单方法自动评测入口。
- **`rescore.py` / `rescore_level_analysis.py`**: 对已有 answers 重打分与按难度导出 Excel。
- **`scoring.py`**: 分级评分逻辑。
- **`distribution_of_questions.py`**: 评测集难度分布统计。

### data/ (数据存储 - 本地生成，gitignore)

- `original_data/`: 书籍 / 论文全文 / 摘要语料。
- `processed_jsons/`、`chunks/`、`graphs/`、`embeddings/`、`cache/`：索引流水线产物（含 `text_only`、`no_entities_merge` 消融变体）。
- `reasoning/`：主权重 `develop.pt` 及消融权重。
- `evaluation_sets/`：`L12.json`、`L3.json`、`L4.json`、`hard.json`。
- `answers/multidimensional_evaluation/<run_dir>/`：按实验组织的评测答案。
- `reports/`：社区报告、`rescore/`、`analysis/`。

另见顶层 `history/`（历史归档）、`research_paper/`（汇报材料）。细节：[docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)。

------

## 10. 相关文档

本项目提供了完善的技术文档，位于 `docs/` 目录：

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: 系统架构详细说明（推荐首先阅读）
- **[DATA_LAYOUT.md](docs/DATA_LAYOUT.md)**: 数据与实验目录布局
- **[EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)**: 评测与消融实验指南
- **[RUN_INDEXING_GUIDE.md](docs/RUN_INDEXING_GUIDE.md)**: 索引流水线详细使用指南
- **[REASONING_GUIDE.md](docs/REASONING_GUIDE.md)**: 图推理系统完整指南
- **[IMPORT_TO_GEPHI.md](docs/IMPORT_TO_GEPHI.md)**: Gephi 可视化详细教程
- **[IMPORT_TO_NEO4J.md](docs/IMPORT_TO_NEO4J.md)**: Neo4j 导入与查询指南
- **[DEPENDENCIES.md](docs/DEPENDENCIES.md)**: 项目依赖说明

------

## 11. 常见问题 (FAQ)

### Q1: 如何选择合适的 LLM 模型？

**A**: 本项目支持任何兼容 OpenAI API 格式的模型。推荐选择：
- **Qwen-3Max**: 性价比最高
- **GPT-5.2**: 最新的多模态大模型，成本较高
- **Gemini-3-Pro**: 多模态能力出色，上下文长度最大，但成本较高

### Q2: 图推理模型训练需要多长时间？

**A**: 取决于图谱规模和硬件配置：

- **小规模**（< 1000 节点）：CPU 训练约 10-20 分钟
- **中规模**（1000-5000 节点）：GPU 训练约 30-60 分钟
- **大规模**（> 5000 节点）：GPU 训练约 1-3 小时

### Q3: 如何提升检索准确率？

**A**: 建议优化以下方面：
1. **提高文本切片质量**: 调整 `settings.yaml` 中的 chunk_size 参数
2. **改进实体抽取**: 优化 `prompts/text_to_graph.md` 中的提示词
3. **增强图推理**: 增加训练轮数（epochs），使用 GPU 加速
4. **调整检索策略**: 根据问题类型选择合适的查询模式（全局/局部/推理）

------

## 12. 贡献与支持

欢迎提交 Issue 和 Pull Request！

如果本项目对您的研究有帮助，请考虑引用我们的工作。

------

## 13. 许可证

本项目采用 **Apache License 2.0** 开源许可证。

### 许可证要点

- ✅ **允许商业使用**: 可以在商业项目中使用本项目代码
- ✅ **允许修改**: 可以修改源代码并分发
- ✅ **允许分发**: 可以自由分发原始或修改后的代码
- ✅ **专利授权**: 自动获得贡献者的相关专利授权
- ⚠️ **保留声明**: 必须保留版权声明和许可证文本
- ⚠️ **声明修改**: 修改文件时建议说明变更

### 详细文档

- 📄 完整许可证文本: [LICENSE](LICENSE)

### 版权信息

```
Copyright 2025 SUNRIVERWOOD

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
