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

import os
import asyncio
import functools
from google.genai import types
import yaml
from pathlib import Path
from utils.client_factory import create_gemini_client

# 加载配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
config_path = PROJECT_ROOT / "config" / "settings.yaml"

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 获取 API 密钥和代理
api_key = os.environ.get("GEMINI_API_KEY")
proxy = config.get("proxy", {})
client=create_gemini_client(api_key,proxy)

async def test_model(model_name: str):
    """测试指定模型的可用性"""
    print(f"\n{'=' * 60}")
    print(f"测试模型: {model_name}")
    print(f"{'=' * 60}")

    try:
        # 同步调用包装为异步
        generate = functools.partial(
            client.models.generate_content,
            model=model_name,
            config=types.GenerateContentConfig(temperature=0.7)
        )

        loop = asyncio.get_running_loop()

        # 简单测试问题
        test_prompt = "请回答你好。"
        print(f"📤 发送测试问题: {test_prompt}")

        response = await loop.run_in_executor(None, lambda: generate(contents=test_prompt))

        print(f"✅ 成功! 模型响应:")
        print(f"📥 {response.text[:200]}...")
        return True

    except Exception as e:
        print(f"❌ 失败! 错误信息:")
        print(f"   类型: {type(e).__name__}")
        print(f"   详情: {str(e)}")
        return False


async def main():
    """测试多个模型"""
    print("\n🚀 开始测试 Gemini API 连接...")
    print(f"API Key 前缀: {api_key[:20]}..." if api_key else "⚠️  未找到 API Key")

    # 要测试的模型列表
    models_to_test = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        # config["query"]["generation_model"]  # 配置文件中的模型
    ]

    results = {}
    for model in set(models_to_test):  # 去重
        success = await test_model(model)
        results[model] = "✅ 可用" if success else "❌ 不可用"
        await asyncio.sleep(2)  # 避免请求过快

    # 汇总结果
    print(f"\n{'=' * 60}")
    print("📊 测试结果汇总")
    print(f"{'=' * 60}")
    for model, status in results.items():
        print(f"{status} | {model}")

    print(f"\n{'=' * 60}")
    print("💡 建议:")
    if results.get(config["query"]["generation_model"]) == "❌ 不可用":
        print("⚠️  当前配置的模型不可用,建议在 settings.yaml 中修改为:")
        available = [m for m, s in results.items() if s == "✅ 可用"]
        if available:
            print(f"   generation_model: \"{available[0]}\"")
    else:
        print("✅ 当前配置的模型工作正常!")


if __name__ == "__main__":
    asyncio.run(main())
