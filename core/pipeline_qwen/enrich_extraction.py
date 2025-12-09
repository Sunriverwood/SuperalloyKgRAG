# enrich_extraction.py
"""
富化提取模块：检查并合并四种类型的知识图谱
- 纯文本图谱 (extracted_graph.jsonl)
- 摘要图谱 (extracted_abstract_graph.jsonl)
- 图片图谱 (extracted_image_graph.jsonl)
- 表格图谱 (extracted_table_graph.jsonl)

如果某种图谱缺失，自动执行相应的提取流程。
最终将所有图谱合并为 enriched_graph.jsonl
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 导入各提取模块
from core.pipeline_qwen.loader import DocumentLoader
from core.pipeline_qwen.extraction_qwen import run_extraction
from core.pipeline_qwen.abstract_extraction import run_abstract_extraction
from core.pipeline_qwen.image_extraction import ImageProcessor
from core.pipeline_qwen.table_extraction import TableProcessor


# =========================
# 配置与日志
# =========================

def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/superalloyKgRAG.log")
    log_file = PROJECT_ROOT / relative_log_path

    Path(log_file).parent.mkdir(exist_ok=True, parents=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("日志记录器设置完成 - enrich_extraction")


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# =========================
# 图谱类型定义
# =========================

class GraphType:
    """定义四种图谱类型及其对应的文件名和提取函数"""

    TEXT = {
        'name': 'text',
        'filename': 'extracted_graph.jsonl',
        'extract_func': 'extract_text_graph',
        'config_key': 'extraction'
    }

    ABSTRACT = {
        'name': 'abstract',
        'filename': 'extracted_abstract_graph.jsonl',
        'extract_func': 'extract_abstract_graph',
        'config_key': 'abstract_extraction'
    }

    IMAGE = {
        'name': 'image',
        'filename': 'extracted_image_graph.jsonl',
        'extract_func': 'extract_image_graph',
        'config_key': 'image_extraction'
    }

    TABLE = {
        'name': 'table',
        'filename': 'extracted_table_graph.jsonl',
        'extract_func': 'extract_table_graph',
        'config_key': 'table_extraction'
    }

    ALL_TYPES = [TEXT, ABSTRACT, IMAGE, TABLE]


# =========================
# 图谱检查与提取
# =========================

class EnrichExtractor:
    """富化提取器：管理四种图谱的检查、提取和合并"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extracted_dir = PROJECT_ROOT / "data/graphs/extracted"
        self.enriched_dir = PROJECT_ROOT / "data/graphs/enriched"

        # 确保目录存在
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.enriched_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"提取图谱目录: {self.extracted_dir}")
        logging.info(f"富化图谱目录: {self.enriched_dir}")

    def check_graph_exists(self, graph_type: Dict[str, str]) -> bool:
        """检查指定类型的图谱是否存在且非空"""
        graph_path = self.extracted_dir / graph_type['filename']

        if not graph_path.exists():
            logging.warning(f"❌ {graph_type['name']} 不存在: {graph_path}")
            return False

        # 检查文件是否为空
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line.strip():
                    logging.warning(f"⚠️ {graph_type['name']} 为空文件: {graph_path}")
                    return False

            logging.info(f"✅ {graph_type['name']} 已存在: {graph_path}")
            return True
        except Exception as e:
            logging.error(f"❌ 读取 {graph_type['name']} 时出错: {e}")
            return False

    def extract_text_graph(self):
        """提取纯文本图谱：loader + extraction_qwen"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始提取纯文本图谱 (loader + extraction)")
        logging.info("=" * 80)

        # 步骤1: 运行 loader
        loader_config = self.config["loader"]
        loader = DocumentLoader(
            source_dir=str(PROJECT_ROOT / loader_config["source_json_dir"]),
            output_dir=str(PROJECT_ROOT / loader_config["output_dir"]),
            chunk_size=loader_config["chunk_size"],
            chunk_overlap=loader_config["chunk_overlap"]
        )

        try:
            output_path = loader.run(output_format=loader_config["output_format"])
            logging.info(f"✅ Loader 完成，输出: {output_path}")
        except Exception as e:
            logging.error(f"❌ Loader 执行失败: {e}")
            raise

        # 步骤2: 运行 extraction
        try:
            run_extraction()
            logging.info("✅ 纯文本图谱提取完成")
        except Exception as e:
            logging.error(f"❌ 纯文本图谱提取失败: {e}")
            raise

    def extract_abstract_graph(self):
        """提取摘要图谱"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始提取摘要图谱 (abstract_extraction)")
        logging.info("=" * 80)

        try:
            run_abstract_extraction()
            logging.info("✅ 摘要图谱提取完成")
        except Exception as e:
            logging.error(f"❌ 摘要图谱提取失败: {e}")
            raise

    def extract_image_graph(self):
        """提取图片图谱"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始提取图片图谱 (image_extraction)")
        logging.info("=" * 80)

        try:
            processor = ImageProcessor()
            processor.run()
            logging.info("✅ 图片图谱提取完成")
        except Exception as e:
            logging.error(f"❌ 图片图谱提取失败: {e}")
            raise

    def extract_table_graph(self):
        """提取表格图谱"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始提取表格图谱 (table_extraction)")
        logging.info("=" * 80)

        try:
            processor = TableProcessor()
            processor.run()
            logging.info("✅ 表格图谱提取完成")
        except Exception as e:
            logging.error(f"❌ 表格图谱提取失败: {e}")
            raise

    def check_and_extract_all(self) -> Dict[str, bool]:
        """检查所有图谱，缺失则提取"""
        logging.info("\n" + "=" * 80)
        logging.info("📊 开始检查四种图谱...")
        logging.info("=" * 80)

        graph_status = {}

        for graph_type in GraphType.ALL_TYPES:
            exists = self.check_graph_exists(graph_type)
            graph_status[graph_type['name']] = exists

            if not exists:
                logging.info(f"🔧 缺少 {graph_type['name']}，开始提取...")
                extract_func = getattr(self, graph_type['extract_func'])
                try:
                    extract_func()
                    graph_status[graph_type['name']] = True
                except Exception as e:
                    logging.error(f"❌ 提取 {graph_type['name']} 失败: {e}")
                    graph_status[graph_type['name']] = False

        logging.info("\n" + "=" * 80)
        logging.info("📊 图谱检查结果:")
        for name, status in graph_status.items():
            status_str = "✅ 存在" if status else "❌ 缺失"
            logging.info(f"  {name}: {status_str}")
        logging.info("=" * 80 + "\n")

        return graph_status

    def merge_graphs(self) -> Path:
        """合并所有图谱为 enriched_graph.jsonl"""
        logging.info("\n" + "=" * 80)
        logging.info("🔗 开始合并图谱...")
        logging.info("=" * 80)

        enriched_graph_path = self.enriched_dir / "enriched_graph.jsonl"
        total_count = 0

        try:
            with open(enriched_graph_path, 'w', encoding='utf-8') as outfile:
                for graph_type in GraphType.ALL_TYPES:
                    graph_path = self.extracted_dir / graph_type['filename']

                    if not graph_path.exists():
                        logging.warning(f"⚠️ 跳过不存在的图谱: {graph_type['name']}")
                        continue

                    logging.info(f"📥 合并 {graph_type['name']}...")
                    count = 0

                    try:
                        with open(graph_path, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                line = line.strip()
                                if line:
                                    # 解析并添加来源标记
                                    try:
                                        data = json.loads(line)
                                        # 为每个图谱条目添加来源类型标记
                                        if 'graph' in data and isinstance(data['graph'], dict):
                                            data['graph']['source_type'] = graph_type['name']
                                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                                        count += 1
                                    except json.JSONDecodeError as e:
                                        logging.warning(f"  ⚠️ 跳过无效JSON行: {e}")
                                        continue

                        logging.info(f"  ✅ {graph_type['name']}: 合并 {count} 条记录")
                        total_count += count

                    except Exception as e:
                        logging.error(f"  ❌ 读取 {graph_type['name']} 时出错: {e}")
                        continue

            logging.info("\n" + "=" * 80)
            logging.info(f"✅ 图谱合并完成！")
            logging.info(f"📊 总计合并: {total_count} 条记录")
            logging.info(f"📁 输出文件: {enriched_graph_path}")
            logging.info("=" * 80 + "\n")

            return enriched_graph_path

        except Exception as e:
            logging.error(f"❌ 合并图谱时出错: {e}")
            raise

    def run(self) -> Path:
        """执行完整的富化提取流程"""
        logging.info("\n" + "=" * 80)
        logging.info("🚀 开始执行富化提取流程")
        logging.info("=" * 80)

        # 步骤1: 检查并提取缺失的图谱
        graph_status = self.check_and_extract_all()

        # 步骤2: 合并所有图谱
        enriched_graph_path = self.merge_graphs()

        # 步骤3: 验证结果
        all_exists = all(graph_status.values())
        if all_exists:
            logging.info("✅ 所有图谱均已成功提取和合并！")
        else:
            missing_graphs = [name for name, status in graph_status.items() if not status]
            logging.warning(f"⚠️ 以下图谱仍然缺失: {', '.join(missing_graphs)}")

        return enriched_graph_path


# =========================
# 主函数
# =========================

def run_enrich_extraction():
    """执行富化提取的主函数"""
    # 加载配置
    config = load_config()
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("富化提取流程启动")
    logging.info("=" * 80)

    # 创建提取器并运行
    extractor = EnrichExtractor(config)
    enriched_graph_path = extractor.run()

    logging.info("\n" + "=" * 80)
    logging.info("🎉 富化提取流程完成！")
    logging.info(f"📁 富化图谱路径: {enriched_graph_path}")
    logging.info("=" * 80)

    return enriched_graph_path


if __name__ == "__main__":
    run_enrich_extraction()

