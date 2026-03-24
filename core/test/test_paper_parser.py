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
论文 PDF 解析器 — 直接提交测试脚本 (Gemini 版)

不走 Batch API，直接调用 Gemini generate_content 接口，
用于快速验证 prompt 和解析效果。
"""

import json
import logging
import os
import sys
from pathlib import Path
from google.genai import types
from utils.client_factory import create_gemini_client
import yaml

# --- 项目根目录 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PaperParserTest")


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = f.read()
        raw = raw.replace("${GEMINI_API_KEY}", os.environ.get("GEMINI_API_KEY", ""))
        return yaml.safe_load(raw)


def load_prompt(config: dict) -> str:
    prompt_path = PROJECT_ROOT / config["paper_parser"]["parsing_prompt_path"]
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def test_single_paper(pdf_path: Path = None):
    """直接提交单篇论文 PDF 进行测试"""
    config = load_config()
    paper_cfg = config["paper_parser"]
    llm_cfg = config["llm"]

    # 确定测试文件
    if pdf_path is None:
        input_dir = PROJECT_ROOT / "core" / "test"
        pdfs = list(input_dir.glob("*.pdf"))
        if not pdfs:
            logger.error(f"❌ {input_dir} 下没有找到 PDF 文件")
            return
        pdf_path = pdfs[0]

    logger.info(f"📄 测试文件: {pdf_path.name}")

    # 初始化客户端
    api_key = llm_cfg["api_key"]
    proxy = config.get("proxy")
    model_name = llm_cfg["model"]
    instructions = load_prompt(config)

    client = create_gemini_client(api_key=api_key, proxy=proxy)

    # 1. 上传 PDF 文件
    logger.info("⬆️ 上传 PDF 到 Gemini Files API...")
    try:
        uploaded_file = client.files.upload(file=str(pdf_path))
        logger.info(f"✅ 上传成功: {uploaded_file.uri}")
    except Exception as e:
        logger.error(f"❌ 上传失败: {e}")
        return

    # 2. 流式调用 generate_content（避免 2.5-pro 长推理时服务端断连）
    logger.info(f"🚀 直接提交到 {model_name}，流式等待响应...")
    try:
        chunks = []
        usage_metadata = None
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=instructions),
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type="application/pdf"
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        ):
            if chunk.text:
                chunks.append(chunk.text)
                logger.info(f"  📶 已接收 {sum(len(c) for c in chunks)} 字符...")
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata
    except Exception as e:
        logger.error(f"❌ API 调用失败: {e}")
        return

    # 打印 Token 用量与费用估算
    if usage_metadata:
        prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) or 0
        thinking_tokens = getattr(usage_metadata, 'thoughts_token_count', 0) or 0
        candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0) or 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) or 0

        # Gemini 2.5 Pro 定价 ($/M tokens): input $0.625, output $5.00, thinking $5
        input_cost = prompt_tokens / 1_000_000 * 0.625
        thinking_cost = thinking_tokens / 1_000_000 * 5.00
        output_cost = candidates_tokens / 1_000_000 * 5.00
        total_cost = input_cost + thinking_cost + output_cost

        logger.info("=" * 60)
        logger.info("💰 Token 用量与费用估算:")
        logger.info(f"  输入 tokens:   {prompt_tokens:>10,}  (${input_cost:.4f})")
        logger.info(f"  思考 tokens:   {thinking_tokens:>10,}  (${thinking_cost:.4f})")
        logger.info(f"  输出 tokens:   {candidates_tokens:>10,}  (${output_cost:.4f})")
        logger.info(f"  合计 tokens:   {total_tokens:>10,}")
        logger.info(f"  💲 预估总费用: ${total_cost:.4f}")
        logger.info("=" * 60)

    # 3. 解析结果
    raw_content = "".join(chunks)
    logger.info(f"📥 收到响应，长度: {len(raw_content)} 字符")

    # 清洗 Markdown 代码块
    cleaned = raw_content.strip().replace("```json", "").replace("```", "").strip()

    # 4. 保存结果
    output_dir = PROJECT_ROOT / paper_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        parsed_json = json.loads(cleaned)
        output_file = output_dir / (pdf_path.stem + ".json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ JSON 解析成功，已保存到: {output_file}")

        # 打印元数据
        metadata = parsed_json.get("paper_metadata", {})
        if metadata:
            logger.info("=" * 60)
            logger.info("📋 提取到的论文元数据:")
            logger.info(f"  标题:   {metadata.get('title', 'N/A')}")
            logger.info(f"  期刊:   {metadata.get('journal', 'N/A')}")
            logger.info(f"  年份:   {metadata.get('year', 'N/A')}")
            logger.info(f"  作者:   {metadata.get('authors', 'N/A')}")
            logger.info(f"  DOI:    {metadata.get('doi', 'N/A')}")
            abstract = str(metadata.get('abstract', 'N/A') or 'N/A')
            logger.info(f"  摘要:   {abstract[:200]}{'...' if len(abstract) > 200 else ''}")
            logger.info("=" * 60)

        # 打印页面统计
        pages = parsed_json.get("pages", [])
        logger.info(f"📑 共解析 {len(pages)} 页")
        for page in pages:
            blocks = page.get("content_blocks", [])
            logger.info(f"  第 {page.get('page_number', '?')} 页: {len(blocks)} 个内容块")

    except json.JSONDecodeError:
        raw_file = output_dir / (pdf_path.stem + "_raw.txt")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(cleaned)
        logger.warning(f"⚠️ JSON 解析失败，原始文本已保存到: {raw_file}")
        logger.info(f"原始响应前500字符:\n{cleaned[:500]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_paper(Path(sys.argv[1]))
    else:
        test_single_paper()
