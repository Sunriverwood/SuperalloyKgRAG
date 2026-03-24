# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
清理disambiguation_graph.json中所有含有NaN的字段

处理两类问题：
1. 节点(nodes)中的异常字段，如 "nameAlloy 706: ": NaN, "e-6": NaN
2. 关系(relationships)中的异常字段，如描述为NaN的字段
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def clean_nan_fields(obj: Any) -> Any:
    """
    递归清理对象中所有值为NaN的字段

    Args:
        obj: 要清理的对象（可以是dict, list或其他类型）

    Returns:
        清理后的对象
    """
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # 跳过值为NaN、None或"NaN"字符串的字段
            if value is None or value == "NaN" or \
               (isinstance(value, float) and str(value) == 'nan'):
                logging.debug(f"移除字段: {key} = {value}")
                continue

            # 递归清理嵌套对象
            cleaned[key] = clean_nan_fields(value)

        return cleaned

    elif isinstance(obj, list):
        # 清理列表中的每个元素
        return [clean_nan_fields(item) for item in obj]

    else:
        # 其他类型直接返回
        return obj


def clean_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清理节点列表中的异常字段

    移除的字段类型：
    1. "nameXXX: " 这种异常字段名
    2. "e-X" 实体ID作为字段名且值为NaN
    3. 任何值为NaN的字段
    """
    cleaned_nodes = []
    total_removed = 0

    for node in nodes:
        if not isinstance(node, dict):
            cleaned_nodes.append(node)
            continue

        original_keys = set(node.keys())

        # 清理NaN值的字段
        cleaned_node = clean_nan_fields(node)

        # 额外检查：移除以"name"开头但不是"name"的异常字段
        keys_to_remove = []
        for key in cleaned_node.keys():
            if key.startswith('name') and key != 'name' and ':' in key:
                keys_to_remove.append(key)
                logging.debug(f"移除异常name字段: {key}")

        for key in keys_to_remove:
            cleaned_node.pop(key, None)

        # 额外检查：移除实体ID作为字段名的情况（如"e-6": NaN）
        if 'id' in cleaned_node:
            entity_id = cleaned_node['id']
            if entity_id in cleaned_node and entity_id != 'id':
                cleaned_node.pop(entity_id, None)
                logging.debug(f"移除重复的实体ID字段: {entity_id}")

        removed = len(original_keys) - len(cleaned_node.keys())
        total_removed += removed

        if removed > 0:
            logging.debug(f"节点 {cleaned_node.get('id', 'unknown')}: 移除了 {removed} 个异常字段")

        cleaned_nodes.append(cleaned_node)

    logging.info(f"节点清理完成: 共移除 {total_removed} 个异常字段")
    return cleaned_nodes


def clean_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    清理关系列表中的异常字段

    移除的字段：
    1. 描述过长且值为NaN的字段
    2. "relationships": NaN
    3. "attributes": NaN
    4. 任何其他值为NaN的字段
    """
    cleaned_relationships = []
    total_removed = 0

    for rel in relationships:
        if not isinstance(rel, dict):
            cleaned_relationships.append(rel)
            continue

        original_keys = set(rel.keys())

        # 清理NaN值的字段
        cleaned_rel = clean_nan_fields(rel)

        # 额外检查：移除描述过长的异常字段（通常是错误的）
        keys_to_remove = []
        for key in cleaned_rel.keys():
            # 如果字段名超过100字符且不是标准字段，移除
            if len(key) > 100 and key not in ['id', 'source', 'target', 'relationship', 'description', 'weight', 'source_sentence', 'attributes']:
                keys_to_remove.append(key)
                logging.debug(f"移除异常长字段: {key[:50]}...")

        for key in keys_to_remove:
            cleaned_rel.pop(key, None)

        removed = len(original_keys) - len(cleaned_rel.keys())
        total_removed += removed

        if removed > 0:
            logging.debug(f"关系 {cleaned_rel.get('id', 'unknown')}: 移除了 {removed} 个异常字段")

        cleaned_relationships.append(cleaned_rel)

    logging.info(f"关系清理完成: 共移除 {total_removed} 个异常字段")
    return cleaned_relationships


def clean_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """清理整个图谱对象"""
    cleaned = {}

    for key, value in graph.items():
        if key == 'nodes' and isinstance(value, list):
            cleaned['nodes'] = clean_nodes(value)
        elif key == 'links' and isinstance(value, list):
            cleaned['links'] = clean_relationships(value)
        elif key == 'relationships' and isinstance(value, list):
            cleaned['relationships'] = clean_relationships(value)
        else:
            # 其他字段也清理NaN
            cleaned[key] = clean_nan_fields(value)

    return cleaned


def clean_disambiguation_graph(input_path: Path, output_path: Path = None, backup: bool = True):
    """
    清理disambiguation_graph.json文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（如果为None，则覆盖原文件）
        backup: 是否备份原文件
    """
    if not input_path.exists():
        logging.error(f"文件不存在: {input_path}")
        return False

    logging.info(f"开始清理文件: {input_path}")

    # 读取原始文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}")
        return False
    except Exception as e:
        logging.error(f"读取文件错误: {e}")
        return False

    # 统计原始数据
    original_stats = {
        'nodes': len(data.get('nodes', [])),
        'links': len(data.get('links', [])),
        'relationships': len(data.get('relationships', []))
    }
    logging.info(f"原始数据: {original_stats}")

    # 清理数据
    cleaned_data = clean_graph(data)

    # 统计清理后数据
    cleaned_stats = {
        'nodes': len(cleaned_data.get('nodes', [])),
        'links': len(cleaned_data.get('links', [])),
        'relationships': len(cleaned_data.get('relationships', []))
    }
    logging.info(f"清理后数据: {cleaned_stats}")

    # 备份原文件
    if backup and output_path is None:
        backup_path = input_path.with_suffix('.json.backup')
        try:
            import shutil
            shutil.copy2(input_path, backup_path)
            logging.info(f"已备份原文件至: {backup_path}")
        except Exception as e:
            logging.warning(f"备份失败: {e}")

    # 确定输出路径
    if output_path is None:
        output_path = input_path

    # 写入清理后的数据
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        logging.info(f"清理完成，已保存至: {output_path}")
        return True
    except Exception as e:
        logging.error(f"写入文件错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("清理 disambiguation_graph.json 中的 NaN 字段")
    print("=" * 80)

    # 文件路径
    input_file = PROJECT_ROOT / "data" / "graphs" / "disambiguation_graph.json"

    # 执行清理
    success = clean_disambiguation_graph(
        input_path=input_file,
        output_path=None,  # 覆盖原文件
        backup=True  # 创建备份
    )

    if success:
        print("\n" + "=" * 80)
        print("✅ 清理成功！")
        print("=" * 80)
        print("\n说明:")
        print("  - 原文件已备份为: disambiguation_graph.json.backup")
        print("  - 所有含有 NaN 的字段已被移除")
        print("  - 异常的 name 字段（如 'nameAlloy 706: '）已被移除")
        print("  - 重复的实体ID字段（如 'e-6': NaN）已被移除")
        print("  - 关系中的异常长字段已被移除")
        print("\n如需恢复:")
        print("  cp data/graphs/disambiguation_graph.json.backup data/graphs/disambiguation_graph.json")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 清理失败，请查看日志")
        print("=" * 80)


if __name__ == "__main__":
    main()

