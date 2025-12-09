"""
图片提取模块测试脚本
用于验证不同类型图片的提取功能
"""
import json
from pathlib import Path
from core.pipeline_qwen.image_extraction import ImageProcessor

def create_test_json():
    """创建测试用的 JSON 文件，包含不同类型的图片"""
    test_data = [
        {
            "page_number": 1,
            "content_blocks": [
                {
                    "block_id": "page_1_block_1",
                    "type": "image",
                    "caption": "Effect of temperature on yield strength of Alloy 718",
                    "image_type": "chart",
                    "content": {
                        "title": "Temperature vs. Yield Strength",
                        "x_axis_label": "Temperature (°C)",
                        "y_axis_label": "Yield Strength (MPa)",
                        "legend": ["Alloy 718"],
                        "trend_description": "Yield strength decreases from 1200 MPa at 20°C to 800 MPa at 600°C showing significant thermal softening effect",
                        "extracted_data": [
                            {"x": 20, "y": 1200},
                            {"x": 200, "y": 1100},
                            {"x": 400, "y": 950},
                            {"x": 600, "y": 800}
                        ]
                    }
                },
                {
                    "block_id": "page_1_block_2",
                    "type": "image",
                    "caption": "Microstructure zones of welded Inconel 625",
                    "image_type": "schematic",
                    "content": {
                        "description": "Shows three distinct zones: base metal with original grain structure, heat-affected zone (HAZ) with grain growth, and weld fusion zone with dendritic solidification structure",
                        "ocr_text": "Base Metal | HAZ | Fusion Zone"
                    }
                },
                {
                    "block_id": "page_1_block_3",
                    "type": "image",
                    "caption": "SEM image of Alloy 718 showing δ-phase precipitation",
                    "image_type": "photograph",
                    "content": {
                        "description": "Needle-shaped δ-phase precipitates with length 2-5 μm are distributed along grain boundaries in γ matrix. The precipitates show preferential orientation and form a continuous network at high aging temperatures."
                    }
                }
            ]
        }
    ]

    # 创建测试目录和文件
    test_dir = Path("data/processed_jsons")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file = test_dir / "test_image_extraction.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 创建测试文件: {test_file}")
    return test_file


def test_context_building():
    """测试上下文构建功能"""
    print("\n" + "="*60)
    print("���试上下文构建功能")
    print("="*60)

    processor = ImageProcessor()

    # 测试 chart 类型
    chart_block = {
        "block_id": "test_chart",
        "type": "image",
        "image_type": "chart",
        "caption": "Test chart",
        "content": {
            "title": "Test Title",
            "x_axis_label": "X (unit)",
            "y_axis_label": "Y (unit)",
            "legend": ["Series A", "Series B"],
            "trend_description": "Increasing trend observed"
        }
    }

    context = processor._build_image_context(chart_block)
    print("\n[Chart Context]")
    print(context)

    # 测试 schematic 类型
    schematic_block = {
        "block_id": "test_schematic",
        "type": "image",
        "image_type": "schematic",
        "caption": "Test schematic",
        "content": {
            "description": "Shows component A connected to component B",
            "ocr_text": "A → B"
        }
    }

    context = processor._build_image_context(schematic_block)
    print("\n[Schematic Context]")
    print(context)

    # 测试 photograph 类型
    photo_block = {
        "block_id": "test_photo",
        "type": "image",
        "image_type": "photograph",
        "caption": "Test photograph",
        "content": {
            "description": "Needle-shaped precipitates visible in matrix"
        }
    }

    context = processor._build_image_context(photo_block)
    print("\n[Photograph Context]")
    print(context)


def test_batch_request_preparation():
    """测试批量请求准备功能"""
    print("\n" + "="*60)
    print("测试批量请求准备功能")
    print("="*60)

    # 创建测试数据
    test_file = create_test_json()

    # 初始化处理器
    processor = ImageProcessor()

    # 准备批量请求
    image_count = processor.prepare_batch_requests()

    print(f"\n✅ 成功准备 {image_count} 个图片的批量请求")

    # 检查生成的文件
    if processor.batch_request_path.exists():
        print(f"✅ 批量请求文件已生成: {processor.batch_request_path}")

        # 读取并显示前几行
        with open(processor.batch_request_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   共 {len(lines)} 行请求")
            if lines:
                print("\n   第一个请求示例:")
                first_request = json.loads(lines[0])
                print(f"   - custom_id: {first_request['custom_id']}")
                print(f"   - model: {first_request['body']['model']}")
                print(f"   - messages: {len(first_request['body']['messages'])} 条")

    if processor.images_units_file.exists():
        print(f"✅ Image units 文件已生成: {processor.images_units_file}")

        # 读取并显示
        with open(processor.images_units_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   共 {len(lines)} 个 image units")
            if lines:
                print("\n   第一个 unit 示例:")
                first_unit = json.loads(lines[0])
                print(f"   - id: {first_unit['id']}")
                print(f"   - image_type: {first_unit['metadata'].get('image_type')}")
                print(f"   - text length: {len(first_unit['text'])} 字符")
                print(f"   - text preview: {first_unit['text'][:150]}...")


def test_prompt_loading():
    """测试 prompt 加载功能"""
    print("\n" + "="*60)
    print("测试 Prompt 加载功能")
    print("="*60)

    processor = ImageProcessor()

    for img_type, prompt in processor.prompts.items():
        print(f"\n[{img_type.upper()}]")
        print(f"  Prompt length: {len(prompt)} 字符")
        print(f"  First 100 chars: {prompt[:100]}...")

        # 验证 prompt 包含关键信息
        if img_type == "chart":
            assert "trend" in prompt.lower() or "chart" in prompt.lower()
        elif img_type == "schematic":
            assert "schematic" in prompt.lower() or "component" in prompt.lower()
        elif img_type == "photograph":
            assert "photograph" in prompt.lower() or "phase" in prompt.lower()

    print("\n✅ 所有 prompts 加载成功")


def main():
    """运行所有测试"""
    print("="*60)
    print("图片提取模块测试")
    print("="*60)

    try:
        # 测试 1: Prompt 加载
        test_prompt_loading()

        # 测试 2: 上下文构建
        test_context_building()

        # 测试 3: 批量请求准备
        test_batch_request_preparation()

        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)

        print("\n下一步:")
        print("1. 检查生成的文件:")
        print("   - data/cache/extraction_image_requests.jsonl")
        print("   - data/chunks/image_units.jsonl")
        print("\n2. 如需执行完整流程（调用 API），运行:")
        print("   python core/pipeline_qwen/image_extraction.py")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

