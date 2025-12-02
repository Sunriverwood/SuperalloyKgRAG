# SuperalloyKgRAG 依赖说明文档

## 📋 概述

本文档详细说明了 SuperalloyKgRAG 项目的所有依赖包及其用途。所有版本号均与当前开发环境保持一致。

## 🚀 快速安装

```bash
# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 安装所有依赖
pip install -r requirements.txt
```

## 📦 核心依赖分类

### 1. 深度学习框架 (Deep Learning)
- **torch** (2.5.1) - PyTorch 核心库，用于 GNN 推理模块
- **torchvision** (0.20.1) - 计算机视觉工具
- **torchaudio** (2.5.1) - 音频处理工具
- **transformers** (4.49.0) - Hugging Face 预训练模型库
- **tokenizers** (0.21.0) - 快速文本分词器
- **safetensors** (0.5.3) - 安全的模型权重存储格式

**用途**: 主要用于 `core/reasoning/models/rgat.py` 中的图推理模型训练和推理

---

### 2. LLM API 客户端 (LLM Clients)
- **openai** (2.8.1) - OpenAI API 客户端（用于 Qwen 系列模型）
- **google-genai** (1.34.0) - Google Gemini API 新版客户端
- **google-generativeai** (0.8.5) - Google 生成式 AI SDK

**用途**: 
- `core/*_qwen/` - 使用 OpenAI SDK 调用 Qwen 模型
- `core/vlm_pdf_parser.py` - 使用 Gemini 进行 PDF OCR 解析
- `core/query/` - 查询处理使用 Gemini API

---

### 3. Google Cloud 服务 (Google Cloud)
- **google-cloud-vision** (3.10.2) - 视觉识别服务
- **google-cloud-storage** (3.3.1) - 云存储服务
- **google-api-python-client** (2.181.0) - Google API 通用客户端
- **google-auth** (2.40.3) - 认证库
- **grpcio** (1.74.0) - gRPC 通信协议

**用途**: PDF 文档的高级视觉解析和云端存储管理

---

### 4. 图处理与社区发现 (Graph Processing)
- **networkx** (3.4.2) - 图数据结构和算法
- **igraph** (0.11.9) - 高性能图计算库
- **leidenalg** (0.10.2) - Leiden 社区发现算法

**用途**:
- `core/pipeline*/graph_builder*.py` - 知识图谱构建与社区发现
- `core/reasoning/` - 图推理路径计算
- `utils/graph_reasoning_utils.py` - 图工具函数

---

### 5. 向量数据库 (Vector Database)
- **lancedb** (0.16.0) - 高性能向量数据库

**用途**: 
- `core/pipeline*/embedding*.py` - 存储实体、关系、社区的向量嵌入
- 支持高效的相似度检索

---

### 6. 数据处理与分析 (Data Processing)
- **numpy** (1.26.4) - 数值计算基础库
- **pandas** (2.2.3) - 数据分析和处理
- **pyarrow** (13.0.0) - 高性能列式数据格式

**用途**: 
- 文本单元处理
- 图谱数据转换
- 嵌入向量计算

---

### 7. 配置与数据验证 (Configuration)
- **PyYAML** (6.0.2) - YAML 配置文件解析
- **pydantic** (2.11.7) - 数据验证和序列化
- **pydantic_core** (2.33.2) - Pydantic 核心库

**用途**: 
- `config/settings.yaml` - 项目配置管理
- `core/pipeline*/loader.py` - 文档加载器数据模型

---

### 8. HTTP 与网络 (Networking)
- **requests** (2.32.3) - HTTP 请求库
- **httpx** (0.28.1) - 异步 HTTP 客户端
- **aiohttp** (3.11.10) - 异步 HTTP 框架

**用途**: 
- API 调用
- 文件下载
- 代理配置

---

### 9. PDF 处理 (PDF Processing)
- **PyPDF2** (3.0.1) - PDF 文件读取
- **pillow** (10.4.0) - 图像处理

**用途**: 
- `core/vlm_pdf_parser*.py` - PDF 文档解析

---

### 10. 数据集与模型仓库 (Datasets & Hub)
- **datasets** (2.19.2) - Hugging Face 数据集库
- **huggingface_hub** (0.29.2) - 模型下载与管理

**用途**: 可能用于加载预训练模型和数据集

---

### 11. 工具库 (Utilities)
- **tqdm** (4.67.1) - 进度条显示
- **tenacity** (9.1.2) - 重试机制
- **retry** (0.9.2) - 函数重试装饰器
- **colorama** (0.4.6) - 终端彩色输出

**用途**: 提升代码健壮性和用户体验

---

### 12. 日志与监控 (Logging & Monitoring)
- **tensorboard** (2.17.0) - 训练可视化
- **tensorboard_data_server** (0.7.0) - TensorBoard 后端

**用途**: 
- `core/reasoning/training/` - 监控 GNN 模型训练过程

---

### 13. 图数据库 (Graph Database)
- **neo4j** (5.28.1) - Neo4j Python 驱动

**用途**: 可选的图数据库后端（当前主要使用 NetworkX）

---

### 14. 模板引擎 (Templating)
- **Jinja2** (3.1.6) - 模板引擎
- **MarkupSafe** (3.0.2) - 字符串转义

**用途**: 
- `config/prompts/` - Prompt 模板渲染

---

## ⚠️ 特殊依赖说明

### Conda 特定包 (仅限 Conda 环境)
以下包在 `requirements.txt` 中已注释，需要通过 conda 单独安装：

```bash
conda install mkl mkl_fft mkl_random mkl-service bottleneck numexpr gmpy2 brotli
```

- **mkl** - Intel 数学核心库，显著加速 NumPy/PyTorch 计算
- **numexpr** (2.10.1) - 快速数值表达式计算
- **bottleneck** (1.4.2) - NumPy 优化函数

---

## 🔍 依赖关系图

```
SuperalloyKgRAG
├── PDF 解析流水线
│   ├── google-generativeai (Gemini Vision API)
│   ├── pillow (图像处理)
│   └── PyPDF2 (PDF 读取)
│
├── 文本分块与加载
│   ├── pydantic (数据验证)
│   └── pandas (数据处理)
│
├── 三元组提取
│   ├── openai (Qwen API)
│   └── PyYAML (配置管理)
│
├── 图谱构建
│   ├── networkx (图数据结构)
│   ├── igraph + leidenalg (社区发现)
│   └── openai (消歧与摘要)
│
├── 向量化存储
│   ├── lancedb (向量数据库)
│   ├── openai (生成嵌入)
│   └── numpy (向量计算)
│
├── 图推理模块
│   ├── torch (GNN 模型)
│   ├── transformers (预训练模型)
│   └── tensorboard (训练监控)
│
└── 查询系统
    ├── openai / google-genai (LLM API)
    ├── lancedb (向量检索)
    └── networkx (路径推理)
```

---

## 📊 统计信息

- **总依赖包数量**: 约 150+ 个
- **直接依赖**: ~40 个核心包
- **间接依赖**: ~110 个传递依赖
- **总安装大小**: 约 5-8 GB（包括 PyTorch 和模型）

---

## 🛠️ 版本兼容性

### Python 版本要求
- **推荐**: Python 3.9 - 3.12
- **当前环境**: Python 3.12

### 操作系统
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS 11+

### CUDA 支持
- PyTorch 2.5.1 支持 CUDA 11.8 / 12.1+
- 如需 GPU 加速，请安装对应 CUDA 版本

---

## 🔄 更新依赖

### 查看过期包
```bash
pip list --outdated
```

### 升级特定包
```bash
pip install --upgrade <package-name>
```

### 重新生成 requirements.txt
```bash
pip freeze > requirements_new.txt
```

---

## 🐛 常见问题

### Q1: 安装 igraph 或 leidenalg 失败
**A**: 需要 C++ 编译器。Windows 用户安装 Visual Studio Build Tools。

### Q2: torch 下载速度慢
**A**: 使用清华镜像源：
```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: Google 相关包安装报错
**A**: 检查网络连接，必要时配置代理。

---

## 📝 维护建议

1. **定期更新**: 每月检查关键依赖的安全更新
2. **版本锁定**: 生产环境使用 `==` 固定版本
3. **虚拟环境**: 始终在虚拟环境中开发，避免污染全局环境
4. **依赖审计**: 定期使用 `pip check` 检查依赖冲突

---

## 📄 许可证信息

各依赖包遵循不同的开源许可证：
- PyTorch: BSD-3-Clause
- Transformers: Apache 2.0
- NetworkX: BSD-3-Clause
- OpenAI SDK: MIT
- Google Client Libraries: Apache 2.0

请在商业使用前检查各包的许可证要求。

---

**文档版本**: 1.0  
**最后更新**: 2025-12-03  
**维护者**: SuperalloyKgRAG Team

