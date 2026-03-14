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

import json
import logging
import os
import hashlib
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/extraction.log")
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
    logging.info("日志记录器设置完成")


def load_prompt(prompt_path: str) -> str:
    """从文件加载 prompt"""
    full_path = PROJECT_ROOT / prompt_path
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    图片提取处理器：仿照表格提取方法，对不同类型的图片提取三元组
    主要从图片的 caption、description、trend、summary 方面提取信息
    """

    # 图片类型到 prompt 的映射
    IMAGE_TYPE_PROMPTS = {
        "chart": "config/prompts/chart_to_graph.md",
        "schematic": "config/prompts/schematic_to_graph.md",
        "photograph": "config/prompts/photograph_to_graph.md",
        "other": "config/prompts/photograph_to_graph.md"  # 使用 photograph 的 prompt 作为默认
    }

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        setup_logging(self.config)

        # 从配置中读取路径
        image_config = self.config.get("image_extraction", {})

        # 加载所有图片类型的 prompts
        self.prompts = {}
        for img_type, prompt_path in self.IMAGE_TYPE_PROMPTS.items():
            try:
                self.prompts[img_type] = load_prompt(prompt_path)
                logging.info(f"已加载 {img_type} 类型的 prompt: {prompt_path}")
            except Exception as e:
                logging.warning(f"加载 {img_type} prompt 失败: {e}")
                # 如果加载失败，使用默认的通用 prompt
                self.prompts[img_type] = "Extract entities and relationships from the image content."

        self.client = self._init_client()

        # 路径配置（从 settings 读取）
        self.input_dir = PROJECT_ROOT / image_config.get("input_dir", "data/processed_jsons")

        # 中间文件路径（image units，与 loader 输出格式一致）
        self.images_units_dir = PROJECT_ROOT / image_config.get("output_dir", "data/chunks")
        self.images_units_dir.mkdir(parents=True, exist_ok=True)
        self.images_units_file = self.images_units_dir / image_config.get("output_filename", "image_units.jsonl")

        # 图谱输出路径
        self.graph_output_dir = PROJECT_ROOT / image_config.get("graph_output_dir", "data/graphs/extracted")
        self.graph_output_dir.mkdir(parents=True, exist_ok=True)
        self.graph_output_file = self.graph_output_dir / image_config.get("graph_output_filename", "extracted_image_graph.jsonl")

        # 批量请求文件路径
        self.requests_dir = PROJECT_ROOT / image_config.get("requests_dir", "data/cache")
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.batch_request_path = self.requests_dir / image_config.get("requests_filename", "extraction_image_requests.jsonl")

        logging.info("ImageProcessor 初始化完成")
        logging.info(f"输入目录: {self.input_dir}")
        logging.info(f"图片 units 文件: {self.images_units_file}")
        logging.info(f"图谱输出文件: {self.graph_output_file}")
        logging.info(f"批量请求文件: {self.batch_request_path}")

    def _load_config(self, path: str) -> Dict[str, Any]:
        full_path = PROJECT_ROOT / path
        with open(full_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_client(self):
        """初始化 OpenAI 兼容客户端 (Qwen)"""
        # 优先从环境变量获取 QWEN_API_KEY
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("GEMINI_API_KEY") or self.config.get("llm", {}).get("api_key")

        if not api_key:
            logging.warning("未找到 QWEN_API_KEY，图片处理可能无法运行")
            return None

        # 配置 Qwen 的 Base URL
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _build_image_context(self, block: Dict) -> str:
        """
        从图片 block 构建上下文，提取 caption、description、trend、summary 等信息
        """
        context_parts = []

        # 基本信息
        block_id = block.get("block_id", "unknown")
        image_type = block.get("image_type", "other")
        caption = block.get("caption", "")

        context_parts.append(f"Block ID: {block_id}")
        context_parts.append(f"Image Type: {image_type}")

        if caption:
            context_parts.append(f"Caption: {caption}")

        # 获取 content 字段（不同类型的图片有不同的内容结构）
        content = block.get("content", {})

        if image_type == "chart":
            # 图表类型：提取标题、坐标轴、图例、趋势描述、数据点
            title = content.get("title", "")
            x_axis = content.get("x_axis_label", "")
            y_axis = content.get("y_axis_label", "")
            legend = content.get("legend", [])
            trend = content.get("trend_description", "")
            data = content.get("extracted_data", "")

            if title:
                context_parts.append(f"Title: {title}")
            if x_axis:
                context_parts.append(f"X-axis: {x_axis}")
            if y_axis:
                context_parts.append(f"Y-axis: {y_axis}")
            if legend:
                context_parts.append(f"Legend: {', '.join(legend)}")
            if trend:
                context_parts.append(f"Trend Description: {trend}")
            if data:
                context_parts.append(f"Data Points: {json.dumps(data, ensure_ascii=False)}")

        elif image_type == "schematic":
            # 示意图/图解类型：提取描述和 OCR 文字
            description = content.get("description", "")
            ocr_text = content.get("ocr_text", "")

            if description:
                context_parts.append(f"Description: {description}")
            if ocr_text:
                context_parts.append(f"OCR Text: {ocr_text}")

        elif image_type in ["photograph", "other"]:
            # 照片/其他类型：提取描述
            description = content.get("description", "")

            if description:
                context_parts.append(f"Description: {description}")

        # 返回完整上下文
        return "\n".join(context_parts)

    def prepare_batch_requests(self) -> int:
        """
        从 JSON 文件中收集所有图片，创建批量请求 JSONL 文件
        同时生成与 loader 格式一致的 image_units.jsonl 文件
        返回：图片数量
        """
        logging.info("开始准备图片批量请求...")

        model_name = self.config["llm"].get("model", "qwen-plus")
        image_count = 0

        # 清空之前的文件
        with open(self.batch_request_path, 'w', encoding='utf-8') as req_file, \
             open(self.images_units_file, 'w', encoding='utf-8') as units_file:

            # 遍历所有 JSON 文件
            json_files = list(self.input_dir.glob("*.json"))
            logging.info(f"找到 {len(json_files)} 个 JSON 文件")

            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # data 可能是列表（多页）或字典
                    pages = data if isinstance(data, list) else [data]

                    for page_idx, page in enumerate(pages):
                        page_number = page.get("page_number", page_idx + 1)

                        for block in page.get("content_blocks", []):
                            # 只处理图片类型的 block
                            if block.get("type") != "image":
                                continue

                            block_id = block.get("block_id", "unknown-block")
                            image_type = block.get("image_type", "other")

                            # 检查是否有有效内容
                            content = block.get("content", {})
                            if not content:
                                logging.warning(f"图片 {block_id} 没有 content，跳过")
                                continue

                            # 生成 chunk_id（与 loader 方法一致）
                            doc_id_hash = hashlib.md5(json_file.name.encode('utf-8')).hexdigest()
                            doc_id = f"doc-{doc_id_hash}"
                            block_repr = json.dumps(block, ensure_ascii=False, sort_keys=True)
                            chunk_id_hash = hashlib.md5(f"{doc_id}-image-{block_id}-{block_repr}".encode('utf-8')).hexdigest()
                            chunk_id = f"chunk-{chunk_id_hash}"

                            # 构建图片上下文
                            image_context = self._build_image_context(block)

                            # 选择合适的 prompt
                            prompt = self.prompts.get(image_type, self.prompts["other"])

                            # 构建批量请求
                            messages = [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": image_context}
                            ]

                            request_line = {
                                "custom_id": chunk_id,
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {
                                    "model": model_name,
                                    "messages": messages,
                                    "temperature": self.config["llm"].get("temperature", 0.1),
                                    "top_p": self.config["llm"].get("top_p", 0.9),
                                    "response_format": {"type": "json_object"}
                                }
                            }
                            req_file.write(json.dumps(request_line, ensure_ascii=False) + "\n")

                            # 生成与 loader 一致的 text unit 格式
                            # text 字段包含 caption + description/trend + content 的关键信息
                            text_parts = []
                            caption = block.get("caption", "")

                            if caption:
                                text_parts.append(f"Caption: {caption}")

                            # 根据图片类型添加不同的文本内容
                            if image_type == "chart":
                                title = content.get("title", "")
                                trend = content.get("trend_description", "")
                                if title:
                                    text_parts.append(f"Title: {title}")
                                if trend:
                                    text_parts.append(f"Trend: {trend}")

                            elif image_type == "schematic":
                                description = content.get("description", "")
                                if description:
                                    text_parts.append(f"Description: {description}")

                            elif image_type in ["photograph", "other"]:
                                description = content.get("description", "")
                                if description:
                                    text_parts.append(f"Description: {description}")

                            # 添加完整的 content JSON（用于检索）
                            text_parts.append(f"Full Content: {json.dumps(content, ensure_ascii=False)}")

                            text_content = "\n".join(text_parts)

                            # 元数据（与 loader 格式一致）
                            metadata = {
                                "source_filename": json_file.name,
                                "pages": [page_number],
                                "blocks": [block_id],
                                "image_type": image_type
                            }

                            # 保存 image unit（与 loader 输出格式完全一致）
                            image_unit = {
                                "id": chunk_id,
                                "document_id": doc_id,
                                "text": text_content,
                                "metadata": metadata
                            }
                            units_file.write(json.dumps(image_unit, ensure_ascii=False) + "\n")

                            image_count += 1

                except Exception as e:
                    logging.error(f"处理文件 {json_file.name} 时出错: {e}", exc_info=True)

        logging.info(f"✅ 成功创建批量请求文件，共 {image_count} 个图片: {self.batch_request_path}")
        logging.info(f"✅ 成功创建 image units 文件: {self.images_units_file}")
        return image_count

    def run(self):
        """执行图片提取流程（批量推理模式）"""
        logging.info("=" * 60)
        logging.info("开始图片批量提取流程")
        logging.info("=" * 60)

        if not self.client:
            logging.error("客户端初始化失败，无法继续")
            return

        # 步骤 1: 准备批量请求
        image_count = self.prepare_batch_requests()
        if image_count == 0:
            logging.warning("未找到任何图片，流程结束")
            return

        # 步骤 2: 上传批量请求文件
        logging.info(f"📤 正在上传批量请求文件: {self.batch_request_path.name}...")
        try:
            with open(self.batch_request_path, "rb") as file_obj:
                uploaded_file = self.client.files.create(
                    file=file_obj,
                    purpose="batch"
                )
            logging.info(f"✅ 文件上传成功: {uploaded_file.id}")
        except Exception as e:
            logging.error(f"❌ 文件上传失败: {e}")
            return

        # 步骤 3: 创建批量作业
        logging.info(f"🚀 正在创建批量作业...")
        try:
            file_batch_job = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    'description': f"image-extraction-job-{self.batch_request_path.stem}",
                }
            )
            logging.info(f"✅ 批量作业已创建: {file_batch_job.id}")
        except Exception as e:
            logging.error(f"❌ 创建批量作业失败: {e}")
            return

        # 步骤 4: 轮询作业状态
        job_id = file_batch_job.id
        completed_states = {'completed', 'failed', 'cancelled', 'expired'}
        sleep_interval = self.config.get("vlm_parser", {}).get("sleep_interval", 60)

        logging.info(f"⏳ 开始轮询作业 '{job_id}' 状态，每 {sleep_interval} 秒检查一次...")

        batch_job_status = None
        while True:
            try:
                batch_job_status = self.client.batches.retrieve(batch_id=job_id)
                current_state = batch_job_status.status
                logging.info(f"  - 当前状态: {current_state}")
                if current_state in completed_states:
                    break
                time.sleep(sleep_interval)
            except Exception as e:
                logging.error(f"  - 轮询失败: {e}")
                time.sleep(sleep_interval * 2)

        # 步骤 5: 处理结果
        if batch_job_status and batch_job_status.status == 'completed':
            logging.info(f"✅ 作业成功完成！")
            try:
                output_file_id = batch_job_status.output_file_id
                if not output_file_id:
                    logging.warning("作业完成但没有 output_file_id")
                    return

                logging.info(f"📥 正在下载结果文件: {output_file_id}")
                file_content = self.client.files.content(output_file_id).text

                # 加载图片 units（用于获取元数据）
                images_units = {}
                with open(self.images_units_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        image_unit = json.loads(line)
                        images_units[image_unit["id"]] = image_unit

                # 处理结果并生成图谱输出
                processed_count = 0
                error_count = 0

                with open(self.graph_output_file, 'w', encoding='utf-8') as outfile:
                    for line in file_content.strip().split('\n'):
                        try:
                            result = json.loads(line)
                            chunk_id = result.get("custom_id")
                            response = result.get("response", {})

                            if chunk_id and response.get("status_code") == 200:
                                # 提取 LLM 返回的图谱数据
                                body = response.get("body", {})
                                choices = body.get("choices", [])
                                if choices:
                                    content = choices[0].get("message", {}).get("content", "")
                                    if content:
                                        try:
                                            graph_data = json.loads(content)
                                        except json.JSONDecodeError:
                                            logging.warning(f"  - ⚠️ ID '{chunk_id}' 的内容不是有效 JSON")
                                            error_count += 1
                                            continue

                                        # 从 image units 中获取信息
                                        image_unit = images_units.get(chunk_id)
                                        if not image_unit:
                                            logging.warning(f"  - ⚠️ ID '{chunk_id}' 未找到 image unit")
                                            error_count += 1
                                            continue

                                        # 构建最终输出（与 extraction_qwen 格式一致）
                                        final_output = {
                                            "id": chunk_id,
                                            "graph": graph_data
                                        }
                                        outfile.write(json.dumps(final_output, ensure_ascii=False) + "\n")
                                        processed_count += 1
                                    else:
                                        logging.warning(f"  - ⚠️ ID '{chunk_id}' 的响应内容为空")
                                        error_count += 1
                                else:
                                    logging.warning(f"  - ⚠️ ID '{chunk_id}' 没有 choices")
                                    error_count += 1
                            else:
                                # 处理错误
                                error_info = result.get("error") or response.get("body")
                                logging.error(f"  - ❌ 处理 ID '{chunk_id}' 时发生错误: {error_info}")
                                error_count += 1

                        except json.JSONDecodeError:
                            logging.warning(f"  - ⚠️ 无法解析结果行: {line[:100]}...")
                            error_count += 1
                        except Exception as e:
                            logging.warning(f"  - ⚠️ 处理时发生未知错误: {e}")
                            error_count += 1

                logging.info(f"🎉 结果处理完成！成功处理 {processed_count} 个图片，失败 {error_count} 个。")
                logging.info(f"💾 Image units 已保存至: {self.images_units_file}")
                logging.info(f"💾 图谱数据已保存至: {self.graph_output_file}")

            except Exception as e:
                logging.error(f"❌ 下载或处理结果文件时发生严重错误: {e}", exc_info=True)
        else:
            status = getattr(batch_job_status, 'status', 'Unknown')
            logging.error(f"❌ 作业未能成功。最终状态: {status}")
            if hasattr(batch_job_status, 'errors') and batch_job_status.errors:
                logging.error(f"  - 错误详情: {batch_job_status.errors}")


if __name__ == "__main__":
    processor = ImageProcessor()
    processor.run()

