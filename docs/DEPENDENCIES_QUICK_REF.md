# 🚀 SuperalloyKgRAG 依赖快速参考

## 核心包速查表

| 类别 | 包名 | 版本 | 用途 |
|-----|------|------|------|
| 🧠 **深度学习** | torch | 2.5.1 | GNN 推理模型 |
| 🤖 **LLM API** | openai | 2.8.1 | Qwen 模型调用 |
| 🤖 **LLM API** | google-genai | 1.34.0 | Gemini 模型调用 |
| 🕸️ **图处理** | networkx | 3.4.2 | 图数据结构 |
| 🕸️ **社区发现** | igraph | 0.11.9 | 高性能图计算 |
| 🕸️ **社区发现** | leidenalg | 0.10.2 | Leiden 算法 |
| 💾 **向量库** | lancedb | 0.16.0 | 向量存储与检索 |
| 📊 **数据处理** | pandas | 2.2.3 | 数据分析 |
| 📊 **数值计算** | numpy | 1.26.4 | 科学计算 |
| ⚙️ **配置** | PyYAML | 6.0.2 | YAML 解析 |
| ✅ **验证** | pydantic | 2.11.7 | 数据验证 |
| 📄 **PDF** | PyPDF2 | 3.0.1 | PDF 读取 |
| 🌐 **HTTP** | requests | 2.32.3 | HTTP 请求 |
| 🔄 **异步** | aiohttp | 3.11.10 | 异步 HTTP |
| 🤗 **模型库** | transformers | 4.49.0 | 预训练模型 |

## 🎯 快速安装命令

```bash
# 完整安装
pip install -r requirements.txt

# 仅核心依赖（最小化安装）
pip install torch networkx igraph leidenalg lancedb openai google-genai pydantic PyYAML pandas numpy requests

# 开发环境（包含测试工具）
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

## 📦 按模块分组的依赖

### 1️⃣ PDF 解析模块
```txt
google-generativeai==0.8.5
google-cloud-vision==3.10.2
PyPDF2==3.0.1
pillow==10.4.0
openai==2.8.1
```

### 2️⃣ 文本处理模块
```txt
pydantic==2.11.7
pandas==2.2.3
PyYAML==6.0.2
```

### 3️⃣ 图谱构建模块
```txt
networkx==3.4.2
igraph==0.11.9
leidenalg==0.10.2
openai==2.8.1
```

### 4️⃣ 向量化模块
```txt
lancedb==0.16.0
numpy==1.26.4
openai==2.8.1
```

### 5️⃣ 图推理模块
```txt
torch==2.5.1
transformers==4.49.0
networkx==3.4.2
tensorboard==2.17.0
```

### 6️⃣ 查询模块
```txt
openai==2.8.1
google-genai==1.34.0
lancedb==0.16.0
networkx==3.4.2
```

## 🔧 常用命令

```bash
# 查看已安装包
pip list

# 查看特定包信息
pip show torch

# 验证依赖
pip check

# 导出当前环境
pip freeze > requirements_freeze.txt

# 对比版本差异
pip list --outdated

# 升级 pip 自身
python -m pip install --upgrade pip
```

## ⚡ 性能优化建议

### Windows 用户
```bash
# 使用清华镜像加速
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 PyTorch（CPU 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 PyTorch（CUDA 12.1 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Conda 用户
```bash
# 先安装 MKL 加速库
conda install mkl mkl_fft mkl_random mkl-service bottleneck numexpr

# 再安装其他依赖
pip install -r requirements.txt
```

## 🐛 故障排除

### 问题：igraph 安装失败
```bash
# Windows: 安装 Visual Studio Build Tools
# 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Linux: 安装编译依赖
sudo apt-get install build-essential python3-dev libxml2-dev libz-dev

# macOS: 安装 Xcode Command Line Tools
xcode-select --install
```

### 问题：torch 版本冲突
```bash
# 卸载现有版本
pip uninstall torch torchvision torchaudio

# 重新安装指定版本
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

### 问题：lancedb 安装失败
```bash
# 升级 pip 和 setuptools
pip install --upgrade pip setuptools wheel

# 重试安装
pip install lancedb==0.16.0
```

## 📊 环境信息检查

```bash
# Python 版本
python --version

# Pip 版本
pip --version

# 已安装包数量
pip list | wc -l  # Linux/Mac
pip list | Measure-Object -Line  # Windows PowerShell

# 检查 CUDA 是否可用（如果安装了 GPU 版本 PyTorch）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 💡 最佳实践

1. ✅ 始终使用虚拟环境
2. ✅ 锁定版本号（生产环境）
3. ✅ 定期更新安全补丁
4. ✅ 使用 `pip check` 验证依赖
5. ✅ 保留 `requirements.txt` 在版本控制中
6. ✅ 为不同环境创建不同的 requirements 文件

```
requirements/
├── base.txt          # 基础依赖
├── dev.txt           # 开发依赖（测试、格式化等）
├── prod.txt          # 生产依赖
└── docs.txt          # 文档生成依赖
```

---

**快速参考版本**: 1.0  
**对应 requirements.txt**: 2025-12-03

