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
        self.extracted_dir = PROJECT_ROOT / self.config["enrich_extraction"]["extracted_dir"]
        self.enriched_dir = PROJECT_ROOT / self.config["enrich_extraction"]["enriched_dir"]

        # 确保目录存在
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.enriched_dir.mkdir(parents=True, exist_ok=True)

        # 统计修复信息
        self.fix_stats = {
            'total_entities': 0,
            'fixed_entities': 0,
            'error_types': {}
        }

        logging.info(f"提取图谱目录: {self.extracted_dir}")
        logging.info(f"富化图谱目录: {self.enriched_dir}")

    def validate_and_fix_entity(self, entity: Dict[str, Any], chunk_id: str = "") -> Dict[str, Any]:
        """
        验证并修复实体格式错误

        处理的错误类型：
        1. "nameXXX: " 类型的异常字段名 (如 "nameAlloy 706: ")
        2. 重复的字段 (如同时有 "e-6" 和 "id": "e-6")
        3. 缺失的必需字段 (name, type, description)
        4. 实体ID作为字段名但值为NaN或None
        5. 其他格式异常
        """
        import re

        self.fix_stats['total_entities'] += 1
        fixed = {}
        needs_fix = False
        error_type = []

        # 标准字段
        standard_fields = {'id', 'name', 'type', 'description', 'attributes'}

        # 步骤1: 提取id字段
        if 'id' in entity:
            fixed['id'] = entity['id']
        else:
            # 尝试从其他字段中找到id
            for key in entity.keys():
                if key.startswith('e-') and isinstance(entity[key], str):
                    fixed['id'] = key
                    needs_fix = True
                    error_type.append('missing_id_field')
                    break
            if 'id' not in fixed:
                fixed['id'] = 'e-unknown'
                needs_fix = True
                error_type.append('missing_id_completely')

        # 步骤2: 处理name字段
        name_found = False
        if 'name' in entity and entity['name'] and entity['name'] not in [None, 'NaN']:
            fixed['name'] = entity['name']
            name_found = True
        else:
            # 检查是否有"nameXXX: "这样的异常字段
            for key in entity.keys():
                if key.startswith('name') and key != 'name':
                    # 情况1: "nameAlloy 706: " 这种格式
                    match = re.match(r'name\s*(.+?)\s*:\s*$', key)
                    if match:
                        fixed['name'] = match.group(1).strip()
                        # description在值中
                        if entity[key] and entity[key] not in [None, 'NaN']:
                            fixed['description'] = str(entity[key])
                        name_found = True
                        needs_fix = True
                        error_type.append('name_merged_with_description')
                        break
                    # 情况2: "nameValue" 这种格式（无冒号）
                    elif len(key) > 4:
                        extracted = key[4:].strip()
                        if extracted:
                            fixed['name'] = extracted
                            if entity[key] and entity[key] not in [None, 'NaN']:
                                fixed['description'] = str(entity[key])
                            name_found = True
                            needs_fix = True
                            error_type.append('name_without_separator')
                            break

            # 检查实体ID是否被用作字段名且包含实际名称
            if not name_found and fixed['id'] in entity:
                value = entity[fixed['id']]
                if value and value not in [None, 'NaN'] and isinstance(value, str):
                    fixed['name'] = value
                    name_found = True
                    needs_fix = True
                    error_type.append('name_in_id_field')

        # 如果仍未找到name，使用id或默认值
        if not name_found:
            if fixed['id'] and fixed['id'] != 'e-unknown':
                fixed['name'] = f"Entity_{fixed['id']}"
            else:
                fixed['name'] = "Unknown Entity"
            needs_fix = True
            error_type.append('name_default_fallback')

        # 步骤3: 处理type字段
        if 'type' in entity and entity['type'] and entity['type'] not in [None, 'NaN']:
            fixed['type'] = entity['type']
        else:
            # 推断type
            name_lower = fixed['name'].lower()
            if any(word in name_lower for word in ['alloy', 'superalloy', 'material', 'steel', 'metal']):
                fixed['type'] = 'Material'
            elif any(word in name_lower for word in ['temperature', 'stress', 'strength', 'hardness', 'toughness', 'property']):
                fixed['type'] = 'Property'
            elif any(word in name_lower for word in ['phase', 'γ', 'gamma', 'precipitate', 'carbide']):
                fixed['type'] = 'Phase'
            elif any(word in name_lower for word in ['process', 'treatment', 'machining', 'forging', 'heat']):
                fixed['type'] = 'Process'
            elif any(word in name_lower for word in ['crack', 'defect', 'void', 'porosity']):
                fixed['type'] = 'Defect'
            elif any(word in name_lower for word in ['test', 'measurement', 'analysis', 'characterization']):
                fixed['type'] = 'Method'
            else:
                fixed['type'] = 'Entity'
            needs_fix = True
            error_type.append('type_inferred')

        # 步骤4: 处理description字段
        if 'description' not in fixed:
            if 'description' in entity and entity['description'] and entity['description'] not in [None, 'NaN']:
                fixed['description'] = entity['description']
            else:
                # 生成默认描述
                fixed['description'] = f"{fixed['type']} entity: {fixed['name']}"
                needs_fix = True
                error_type.append('description_generated')

        # 步骤5: 处理attributes字段
        if 'attributes' in entity and isinstance(entity['attributes'], dict):
            fixed['attributes'] = entity['attributes']
        else:
            fixed['attributes'] = {}
            if 'attributes' not in entity:
                needs_fix = True
                error_type.append('attributes_missing')

        # 步骤6: 检查是否有其他非标准字段（可能是数据错误）
        for key in entity.keys():
            if key not in standard_fields and key != fixed['id']:
                # 非标准字段，可能是错误
                value = entity[key]
                if value and value not in [None, 'NaN']:
                    # 记录到attributes中
                    fixed['attributes'][f'_raw_{key}'] = value
                needs_fix = True
                error_type.append(f'non_standard_field_{key}')

        # 更新统计
        if needs_fix:
            self.fix_stats['fixed_entities'] += 1
            for err_type in error_type:
                self.fix_stats['error_types'][err_type] = \
                    self.fix_stats['error_types'].get(err_type, 0) + 1

            logging.debug(f"修复实体 {fixed['id']} (chunk: {chunk_id}): {', '.join(error_type)}")

        return fixed

    def validate_and_fix_relationship(self, relationship: Dict[str, Any], chunk_id: str = "") -> Dict[str, Any]:
        """
        验证并修复关系格式错误
        """
        fixed = {}

        # 标准字段
        required_fields = {'id', 'source', 'target', 'relationship', 'description'}

        # 复制所有标准字段
        for field in required_fields:
            if field in relationship and relationship[field] not in [None, 'NaN']:
                fixed[field] = relationship[field]
            else:
                # 提供默认值
                if field == 'id':
                    fixed['id'] = 'r-unknown'
                elif field == 'description':
                    fixed['description'] = 'Relationship'
                else:
                    fixed[field] = 'unknown'

        # 可选字段
        if 'weight' in relationship:
            try:
                fixed['weight'] = float(relationship['weight'])
            except (ValueError, TypeError):
                fixed['weight'] = 1.0

        if 'source_sentence' in relationship:
            fixed['source_sentence'] = relationship['source_sentence']

        if 'attributes' in relationship and isinstance(relationship['attributes'], dict):
            fixed['attributes'] = relationship['attributes']

        return fixed

    def validate_and_fix_graph(self, graph_data: Dict[str, Any], chunk_id: str = "") -> Dict[str, Any]:
        """
        验证并修复整个图谱数据
        """
        if 'graph' not in graph_data:
            logging.warning(f"⚠️ Chunk {chunk_id} 缺少 'graph' 字段")
            return graph_data

        graph = graph_data['graph']

        # 修复entities
        if 'entities' in graph and isinstance(graph['entities'], list):
            fixed_entities = []
            for entity in graph['entities']:
                if isinstance(entity, dict):
                    fixed_entity = self.validate_and_fix_entity(entity, chunk_id)
                    fixed_entities.append(fixed_entity)
                else:
                    logging.warning(f"⚠️ Chunk {chunk_id}: 实体不是字典类型，跳过")
            graph['entities'] = fixed_entities

        # 修复relationships
        if 'relationships' in graph and isinstance(graph['relationships'], list):
            fixed_relationships = []
            for rel in graph['relationships']:
                if isinstance(rel, dict):
                    fixed_rel = self.validate_and_fix_relationship(rel, chunk_id)
                    fixed_relationships.append(fixed_rel)
                else:
                    logging.warning(f"⚠️ Chunk {chunk_id}: 关系不是字典类型，跳过")
            graph['relationships'] = fixed_relationships

        graph_data['graph'] = graph
        return graph_data

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
        """合并所有图谱为 enriched_graph.jsonl，并在合并前验证和修复格式错误"""
        logging.info("\n" + "=" * 80)
        logging.info("🔗 开始合并图谱（含验证和修复）...")
        logging.info("=" * 80)

        enriched_graph_path = self.enriched_dir / self.config["enrich_extraction"]["enriched_filename"]
        total_count = 0

        # 重置修复统计
        self.fix_stats = {
            'total_entities': 0,
            'fixed_entities': 0,
            'error_types': {}
        }

        try:
            with open(enriched_graph_path, 'w', encoding='utf-8') as outfile:
                for graph_type in GraphType.ALL_TYPES:
                    graph_path = self.extracted_dir / graph_type['filename']

                    if not graph_path.exists():
                        logging.warning(f"⚠️ 跳过不存在的图谱: {graph_type['name']}")
                        continue

                    logging.info(f"📥 合并 {graph_type['name']}...")
                    count = 0
                    fixed_in_this_graph = 0

                    try:
                        with open(graph_path, 'r', encoding='utf-8') as infile:
                            for line_num, line in enumerate(infile, 1):
                                line = line.strip()
                                if not line:
                                    continue

                                try:
                                    data = json.loads(line)
                                    chunk_id = data.get('id', f'line_{line_num}')

                                    # 记录修复前的实体数量
                                    before_fix = self.fix_stats['fixed_entities']

                                    # 验证并修复图谱数据
                                    fixed_data = self.validate_and_fix_graph(data, chunk_id)

                                    # 记录此行是否有修复
                                    if self.fix_stats['fixed_entities'] > before_fix:
                                        fixed_in_this_graph += 1

                                    # 为每个图谱条目添加来源类型标记
                                    if 'graph' in fixed_data and isinstance(fixed_data['graph'], dict):
                                        fixed_data['graph']['source_type'] = graph_type['name']

                                    outfile.write(json.dumps(fixed_data, ensure_ascii=False) + '\n')
                                    count += 1

                                except json.JSONDecodeError as e:
                                    logging.warning(f"  ⚠️ 跳过无效JSON行 (line {line_num}): {e}")
                                    continue
                                except Exception as e:
                                    logging.error(f"  ❌ 处理行 {line_num} 时出错: {e}")
                                    continue

                        logging.info(f"  ✅ {graph_type['name']}: 合并 {count} 条记录，修复 {fixed_in_this_graph} 条")
                        total_count += count

                    except Exception as e:
                        logging.error(f"  ❌ 读取 {graph_type['name']} 时出错: {e}")
                        continue

            # 输出详细的修复统计
            logging.info("\n" + "=" * 80)
            logging.info("📊 图谱验证与修复统计:")
            logging.info(f"  总实体数: {self.fix_stats['total_entities']}")
            logging.info(f"  已修复实体数: {self.fix_stats['fixed_entities']}")

            if self.fix_stats['error_types']:
                logging.info("\n  错误类型分布:")
                for error_type, count in sorted(self.fix_stats['error_types'].items(),
                                               key=lambda x: x[1], reverse=True):
                    logging.info(f"    - {error_type}: {count}")

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

