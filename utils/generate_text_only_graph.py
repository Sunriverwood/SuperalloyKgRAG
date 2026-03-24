import sys
import logging
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline_qwen.enrich_extraction import EnrichExtractor, GraphType, load_config, setup_logging

def generate_text_only_graph():
    config = load_config()
    setup_logging(config)

    # 临时覆盖合并后文件的输出名称
    original_filename = config["enrich_extraction"]["enriched_filename"]
    config["enrich_extraction"]["enriched_filename"] = "enriched_graph_text_only.jsonl"
    
    extractor = EnrichExtractor(config)
    
    # 强制只合并 TEXT 和 ABSTRACT
    GraphType.ALL_TYPES = [GraphType.TEXT, GraphType.ABSTRACT]
    
    logging.info("="*80)
    logging.info("🚀 开始生成纯文本富化图谱 (仅包含 Text 和 Abstract)")
    logging.info(f"📁 输出文件将保存为: {config['enrich_extraction']['enriched_filename']}")
    logging.info("="*80)
    
    # 执行合并（包含自动清洗和修复 validate_and_fix_graph）
    output_path = extractor.merge_graphs()
    
    # 恢复原配置（可选，因为脚本马上结束）
    config["enrich_extraction"]["enriched_filename"] = original_filename
    
    logging.info("🎉 生成完毕！现在你可以使用 `--ablation text_only` 运行索引管线了。")
    return output_path

if __name__ == "__main__":
    generate_text_only_graph()
