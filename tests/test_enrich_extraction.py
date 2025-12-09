"""
测试富化提取模块
验证 enrich_extraction.py 的功能是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline_qwen.enrich_extraction import EnrichExtractor, GraphType, load_config, setup_logging


def test_graph_type_definitions():
    """测试图谱类型定义"""
    print("\n" + "=" * 80)
    print("测试1: 图谱类型定义")
    print("=" * 80)

    assert len(GraphType.ALL_TYPES) == 4, "应该有4种图谱类型"

    for graph_type in GraphType.ALL_TYPES:
        assert 'name' in graph_type, f"缺少 'name' 字段"
        assert 'filename' in graph_type, f"缺少 'filename' 字段"
        assert 'extract_func' in graph_type, f"缺少 'extract_func' 字段"
        assert 'config_key' in graph_type, f"缺少 'config_key' 字段"
        print(f"✅ {graph_type['name']}: {graph_type['filename']}")

    print("✅ 图谱类型定义测试通过")


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 80)
    print("测试2: 配置加载")
    print("=" * 80)

    try:
        config = load_config()
        assert config is not None, "配置不应为空"
        assert 'enrich_extraction' in config, "配置中应包含 enrich_extraction"
        assert 'extraction' in config, "配置中应包含 extraction"
        assert 'abstract_extraction' in config, "配置中应包含 abstract_extraction"
        assert 'image_extraction' in config, "配置中应包含 image_extraction"
        assert 'table_extraction' in config, "配置中应包含 table_extraction"

        print(f"✅ 配置加载成功")
        print(f"   - 提取图谱目录: {config['enrich_extraction']['extracted_dir']}")
        print(f"   - 富化图谱目录: {config['enrich_extraction']['enriched_dir']}")
        print(f"   - 富化图谱文件: {config['enrich_extraction']['enriched_filename']}")

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        raise

    print("✅ 配置加载测试通过")


def test_extractor_initialization():
    """测试提取器初始化"""
    print("\n" + "=" * 80)
    print("测试3: 提取器初始化")
    print("=" * 80)

    try:
        config = load_config()
        setup_logging(config)

        extractor = EnrichExtractor(config)

        assert extractor.extracted_dir.exists(), "提取图谱目录应存在"
        assert extractor.enriched_dir.exists(), "富化图谱目录应存在"

        print(f"✅ 提取器初始化成功")
        print(f"   - 提取目录: {extractor.extracted_dir}")
        print(f"   - 富化目录: {extractor.enriched_dir}")

    except Exception as e:
        print(f"❌ 提取器初始化失败: {e}")
        raise

    print("✅ 提取器初始化测试通过")


def test_check_graph_exists():
    """测试图谱存在性检查"""
    print("\n" + "=" * 80)
    print("测试4: 图谱存在性检查")
    print("=" * 80)

    try:
        config = load_config()
        setup_logging(config)
        extractor = EnrichExtractor(config)

        print("\n检查各类图谱是否存在:")
        for graph_type in GraphType.ALL_TYPES:
            exists = extractor.check_graph_exists(graph_type)
            status = "✅ 存在" if exists else "❌ 不存在"
            print(f"   {graph_type['name']}: {status}")

    except Exception as e:
        print(f"❌ 图谱存在性检查失败: {e}")
        raise

    print("\n✅ 图谱存在性检查测试完成")


def test_graph_builder_input_path():
    """测试 graph_builder 的输入路径是否正确配置"""
    print("\n" + "=" * 80)
    print("测试5: Graph Builder 输入路径配置")
    print("=" * 80)

    try:
        config = load_config()

        expected_path = "data/graphs/enriched/enriched_graph.jsonl"
        actual_path = config['graph_builder']['input_path']

        assert actual_path == expected_path, \
            f"Graph Builder 输入路径应为 {expected_path}, 实际为 {actual_path}"

        print(f"✅ Graph Builder 输入路径配置正确: {actual_path}")

    except Exception as e:
        print(f"❌ Graph Builder 输入路径配置检查失败: {e}")
        raise

    print("✅ Graph Builder 输入路径配置测试通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始测试富化提取模块")
    print("=" * 80)

    tests = [
        test_graph_type_definitions,
        test_config_loading,
        test_extractor_initialization,
        test_check_graph_exists,
        test_graph_builder_input_path
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {test.__name__}")
            print(f"   错误: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())

