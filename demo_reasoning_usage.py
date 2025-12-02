#!/usr/bin/env python
"""
快速演示：图推理查询系统的多种使用方式

运行此脚本了解如何使用新版本的 reasoning_query_qwen.py
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           图推理查询系统 - 使用方式演示                                      ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("1️⃣  方式一：直接点击运行（交互模式）")
print("=" * 80)
print("""
   在IDE中打开: core/query_qwen/reasoning_query_qwen.py
   点击运行按钮或按 F5
   
   自动进入交互模式：
   - 自动检测模型是否已训练
   - 如果已训练 → 直接开始查询
   - 如果未训练 → 询问是否训练
   
   然后：
   - 输入问题
   - 选择推理方法（ppr/gnn）
   - 选择是否生成LLM答案
   - 选择是否保存结果
   - 可以连续查询多次
""")

print("\n2️⃣  方式二：命令行快速查询")
print("=" * 80)
print("""
   基本查询：
   python core/query_qwen/reasoning_query_qwen.py --query "What is nickel?"
   
   使用GNN方法：
   python core/query_qwen/reasoning_query_qwen.py -q "..." -m gnn
   
   不生成LLM答案：
   python core/query_qwen/reasoning_query_qwen.py -q "..." --no-llm
   
   保存结果：
   python core/query_qwen/reasoning_query_qwen.py -q "..." -o result.json
""")

print("\n3️⃣  方式三：训练模式（智能跳过）")
print("=" * 80)
print("""
   智能训练（如果模型已存在，自动跳过）：
   python core/query_qwen/reasoning_query_qwen.py --train
   
   指定训练轮数：
   python core/query_qwen/reasoning_query_qwen.py --train --epochs 100
   
   强制重新训练（即使模型存在）：
   python core/query_qwen/reasoning_query_qwen.py --train --force-train
   
   ✅ 新功能：自动检测 data/reasoning/model.pt
   - 如果存在 → 显示消息并跳过训练
   - 如果不存在 → 开始训练
   - 使用 --force-train 强制重新训练
""")

print("\n4️⃣  方式四：明确指定交互模式")
print("=" * 80)
print("""
   python core/query_qwen/reasoning_query_qwen.py --interactive
   或
   python core/query_qwen/reasoning_query_qwen.py -i
""")

print("\n💡 实用技巧")
print("=" * 80)
print("""
   1. 首次使用：
      python core/query_qwen/reasoning_query_qwen.py --train
      python core/query_qwen/reasoning_query_qwen.py -i
   
   2. 日常查询（推荐）：
      直接点击运行 reasoning_query_qwen.py
   
   3. 批量查询：
      写脚本循环调用 --query 参数
   
   4. 图谱更新后：
      python core/query_qwen/reasoning_query_qwen.py --train --force-train
   
   5. 查看帮助：
      python core/query_qwen/reasoning_query_qwen.py --help
""")

print("\n📊 模型状态检查")
print("=" * 80)

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = PROJECT_ROOT / "data" / "reasoning" / "model.pt"

if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ 模型已存在: {model_path}")
    print(f"   📦 文件大小: {size_mb:.2f} MB")
    print(f"   🕐 修改时间: {os.path.getmtime(model_path)}")
    print("\n   → 直接运行即可查询，无需重新训练")
else:
    print(f"   ⚠️  模型不存在: {model_path}")
    print("\n   → 首次使用需要训练:")
    print("      python core/query_qwen/reasoning_query_qwen.py --train")

print("\n" + "=" * 80)
print("📖 详细文档: docs/REASONING_QUERY_GUIDE.md")
print("=" * 80)

print("""
现在就试试吧！

快速开始：
    python core/query_qwen/reasoning_query_qwen.py

或者直接在IDE中运行 reasoning_query_qwen.py 文件！
""")

