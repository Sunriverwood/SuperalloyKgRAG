# abstract_extraction.py
import json
import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import yaml
from openai import OpenAI

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
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
    logging.info("日志记录器设置完成")


# --- 加载配置和 Prompt ---
def load_config_and_prompt(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件和抽取 Prompt"""
    try:
        settings_path = PROJECT_ROOT / settings_filename
        with open(settings_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("成功加载 settings.yaml 配置文件")

        # 使用 text_to_graph.md 作为 prompt
        prompt_path = PROJECT_ROOT / "config/prompts/text_to_graph.md"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config["abstract_extraction"] = config.get("abstract_extraction", {})
            config["abstract_extraction"]["prompt"] = f.read()
        logging.info(f"成功加载 Prompt 文件: {prompt_path}")

        return config
    except FileNotFoundError as e:
        logging.error(f"配置文件或 Prompt 文件未找到: {e}")
        raise
    except Exception as e:
        logging.error(f"加载配置时出错: {e}")
        raise


# --- 从 Excel 读取摘要并生成 abstract_units.jsonl ---
def extract_abstracts_from_excel(config: Dict[str, Any]) -> Path:
    """从 Excel 文件读取摘要，生成 abstract_units.jsonl"""
    try:
        import pandas as pd
    except ImportError:
        logging.error("需要安装 pandas 库: pip install pandas openpyxl")
        raise

    abstract_config = config.get("abstract_extraction", {})

    # 输入 Excel 文件路径
    input_excel = PROJECT_ROOT / abstract_config.get("input_excel", "data/papers/superalloy_research.xlsx")

    # 输出 abstract_units.jsonl 路径
    output_dir = PROJECT_ROOT / abstract_config.get("output_dir", "data/chunks")
    output_dir.mkdir(exist_ok=True, parents=True)
    abstract_units_path = output_dir / abstract_config.get("output_filename", "abstract_units.jsonl")

    logging.info(f"正在从 {input_excel} 读取摘要...")

    try:
        # 读取 Excel 文件
        df = pd.read_excel(input_excel)

        # 检查必需的列是否存在（根据实际 Excel 文件调整列名）
        # 常见的列名可能是: Title, Abstract, Journal, Year, Author/Authors, DOI
        # 这里使用灵活的列名匹配
        column_mapping = {}

        # 查找 Title 列
        for col in df.columns:
            col_lower = str(col).lower()
            if 'title' in col_lower:
                column_mapping['title'] = col
            elif 'abstract' in col_lower:
                column_mapping['abstract'] = col
            elif 'journal' in col_lower:
                column_mapping['journal'] = col
            elif 'year' in col_lower:
                column_mapping['year'] = col
            elif 'author' in col_lower:
                column_mapping['author'] = col
            elif 'doi' in col_lower:
                column_mapping['doi'] = col

        if 'abstract' not in column_mapping:
            logging.error(f"Excel 文件中未找到 'Abstract' 列。现有列: {df.columns.tolist()}")
            raise ValueError("Excel 文件缺少 'Abstract' 列")

        if 'title' not in column_mapping:
            logging.warning("Excel 文件中未找到 'Title' 列，将使用行号作为标识")

        logging.info(f"找到的列映射: {column_mapping}")

        # 生成 abstract_units.jsonl
        count = 0
        with open(abstract_units_path, 'w', encoding='utf-8') as outfile:
            for idx, row in df.iterrows():
                # 获取摘要内容
                abstract_col = column_mapping.get('abstract')
                abstract_text = row[abstract_col] if abstract_col else None

                # 跳过空摘要
                if pd.isna(abstract_text) or str(abstract_text).strip() == '':
                    logging.warning(f"第 {int(idx)+2} 行的摘要为空，跳过")
                    continue

                # 生成 doc_id (使用 Title 生成哈希)
                title_col = column_mapping.get('title', '')
                title = row[title_col] if title_col and title_col in row.index else f"abstract_{idx}"
                if pd.isna(title):
                    title = f"abstract_{idx}"

                # 使用 title 生成稳定的 doc_id
                doc_id_hash = hashlib.md5(str(title).encode('utf-8')).hexdigest()
                doc_id = f"doc-{doc_id_hash}"

                chunk_id_hash = hashlib.md5(f"{doc_id}-abstract-{abstract_text}".encode('utf-8')).hexdigest()
                chunk_id = f"chunk-{chunk_id_hash}"


                # 构建 metadata
                metadata = {
                    "source_filename": str(title),
                    "type": "abstract"
                }

                journal_col = column_mapping.get('journal')
                if journal_col:
                    try:
                        val = row[journal_col]
                        if not pd.isna(val):
                            metadata["journal"] = str(val)
                    except KeyError:
                        pass

                year_col = column_mapping.get('year')
                if year_col:
                    try:
                        val = row[year_col]
                        if not pd.isna(val):
                            metadata["year"] = str(val)
                    except KeyError:
                        pass

                author_col = column_mapping.get('author')
                if author_col:
                    try:
                        val = row[author_col]
                        if not pd.isna(val):
                            metadata["author"] = str(val)
                    except KeyError:
                        pass

                doi_col = column_mapping.get('doi')
                if doi_col:
                    try:
                        val = row[doi_col]
                        if not pd.isna(val):
                            metadata["DOI"] = str(val)
                    except KeyError:
                        pass

                # 构建 chunk 对象
                chunk = {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "text": str(abstract_text).strip(),
                    "metadata": metadata
                }

                outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1

        logging.info(f"✅ 成功生成 abstract_units.jsonl，共 {count} 条摘要记录: {abstract_units_path}")
        return abstract_units_path

    except FileNotFoundError:
        logging.error(f"Excel 文件未找到: {input_excel}")
        raise
    except Exception as e:
        logging.error(f"读取 Excel 文件时出错: {e}")
        raise


# --- 准备批量请求文件 ---
def prepare_batch_requests(config: Dict[str, Any], abstract_units_path: Path) -> Path:
    """从 abstract_units.jsonl 创建批量请求的 JSONL 文件"""
    abstract_config = config["abstract_extraction"]
    requests_dir = PROJECT_ROOT / abstract_config.get("requests_dir", "data/cache")
    requests_dir.mkdir(exist_ok=True, parents=True)
    batch_request_path = requests_dir / "extraction_abstract_requests.jsonl"

    prompt_template = abstract_config["prompt"]
    model_name = config["llm"]["model"]

    try:
        with open(abstract_units_path, 'r', encoding='utf-8') as infile, \
             open(batch_request_path, 'w', encoding='utf-8') as outfile:
            count = 0
            for line in infile:
                chunk = json.loads(line)
                text_content = chunk.get("text", "")
                chunk_id = chunk.get("id")

                if not text_content:
                    logging.warning(f"跳过ID为 {chunk_id} 的空文本块")
                    continue

                # 构建 OpenAI Batch API 请求体
                messages = [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": text_content}
                ]

                request_line = {
                    "custom_id": chunk_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": messages,
                        "temperature": config["llm"].get("temperature", 0.1),
                        "top_p": config["llm"].get("top_p", 0.9),
                        "response_format": {"type": "json_object"}
                    }
                }
                outfile.write(json.dumps(request_line, ensure_ascii=False) + "\n")
                count += 1

        logging.info(f"✅ 成功创建批量请求文件，共 {count} 条记录: {batch_request_path}")
        return batch_request_path
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {abstract_units_path}")
        raise
    except Exception as e:
        logging.error(f"准备批量请求文件时出错: {e}")
        raise


# --- 主执行函数 ---
def run_abstract_extraction():
    """执行完整的摘要抽取流程"""
    config = load_config_and_prompt(settings_filename='config/settings.yaml')
    setup_logging(config)

    # 1. 从 Excel 提取摘要，生成 abstract_units.jsonl
    try:
        abstract_units_path = extract_abstracts_from_excel(config)
    except Exception as e:
        logging.error(f"提取摘要失败: {e}")
        return

    # 2. 配置 OpenAI 客户端 (阿里云百炼)
    try:
        api_key = os.getenv("QWEN_API_KEY") or os.getenv("GEMINI_API_KEY") or config.get("llm", {}).get("api_key")
        if not api_key:
            raise ValueError("请在环境变量 QWEN_API_KEY 或 settings.yaml 中设置有效的 API Key")

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        logging.info("阿里云百炼 (OpenAI兼容) 客户端配置成功")
    except Exception as e:
        logging.error(f"配置客户端失败: {e}")
        return

    # 3. 准备并上传批量请求文件
    try:
        batch_request_path = prepare_batch_requests(config, abstract_units_path)
    except Exception:
        return

    logging.info(f"📤 正在上传批量请求文件: {batch_request_path.name}...")
    try:
        with open(batch_request_path, "rb") as file_obj:
            uploaded_file = client.files.create(
                file=file_obj,
                purpose="batch"
            )
        logging.info(f"✅ 文件上传成功: {uploaded_file.id}")
    except Exception as e:
        logging.error(f"❌ 文件上传失败: {e}")
        return

    # 4. 创建批量处理作业
    logging.info(f"🚀 正在创建批量作业...")
    try:
        file_batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                'description': f"abstract-extraction-{batch_request_path.stem}",
            }
        )
        logging.info(f"✅ 批量作业已创建: {file_batch_job.id}")
    except Exception as e:
        logging.error(f"❌ 创建批量作业失败: {e}")
        return

    # 5. 轮询作业状态
    job_id = file_batch_job.id
    completed_states = {'completed', 'failed', 'cancelled', 'expired'}
    sleep_interval = config.get("vlm_parser", {}).get("sleep_interval", 60)

    logging.info(f"⏳ 开始轮询作业 '{job_id}' 状态，每 {sleep_interval} 秒检查一次...")

    batch_job_status = None
    while True:
        try:
            batch_job_status = client.batches.retrieve(batch_id=job_id)
            current_state = batch_job_status.status
            logging.info(f"  - 当前状态: {current_state}")
            if current_state in completed_states:
                break
            time.sleep(sleep_interval)
        except Exception as e:
            logging.error(f"  - 轮询失败: {e}")
            time.sleep(sleep_interval * 2)

    # 6. 结果处理
    if batch_job_status and batch_job_status.status == 'completed':
        logging.info(f"✅ 作业成功完成！")
        try:
            output_file_id = batch_job_status.output_file_id
            if not output_file_id:
                logging.warning("作业完成但没有 output_file_id")
                return

            logging.info(f"📥 正在下载结果文件: {output_file_id}")
            file_content = client.files.content(output_file_id).text

            abstract_config = config.get("abstract_extraction", {})
            graph_output_dir = PROJECT_ROOT / abstract_config.get("graph_output_dir", "data/graphs/extracted")
            graph_output_dir.mkdir(exist_ok=True, parents=True)
            output_path = graph_output_dir / abstract_config.get("graph_output_filename", "extracted_abstract_graph.jsonl")

            processed_count = 0
            error_count = 0

            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in file_content.strip().split('\n'):
                    try:
                        result = json.loads(line)
                        chunk_id = result.get("custom_id")
                        response = result.get("response", {})

                        if chunk_id and response.get("status_code") == 200:
                            body = response.get("body", {})
                            choices = body.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content", "")
                                if content:
                                    try:
                                        graph_data = json.loads(content)
                                    except json.JSONDecodeError:
                                        logging.warning(f"  - ⚠️ ID '{chunk_id}' 的内容不是有效 JSON")
                                        graph_data = {"raw_content": content}

                                    final_output = {"id": chunk_id, "graph": graph_data}
                                    outfile.write(json.dumps(final_output, ensure_ascii=False) + "\n")
                                    processed_count += 1
                                else:
                                    logging.warning(f"  - ⚠️ ID '{chunk_id}' 的响应内容为空。")
                                    error_count += 1
                            else:
                                logging.warning(f"  - ⚠️ ID '{chunk_id}' 没有 choices。")
                                error_count += 1
                        else:
                            error_info = result.get("error") or response.get("body")
                            logging.error(f"  - ❌ 处理 ID '{chunk_id}' 时发生错误: {error_info}")
                            error_count += 1

                    except json.JSONDecodeError:
                        logging.warning(f"  - ⚠️ 无法解析结果行: {line[:100]}...")
                        error_count += 1
                    except Exception as e:
                        logging.warning(f"  - ⚠️ 处理时发生未知错误: {e}")
                        error_count += 1

            logging.info(f"🎉 结果处理完成！成功处理 {processed_count} 条，失败 {error_count} 条。")
            logging.info(f"💾 最终图谱数据已保存至: {output_path}")

        except Exception as e:
            logging.error(f"❌ 下载或处理结果文件时发生严重错误: {e}")
    else:
        status = getattr(batch_job_status, 'status', 'Unknown')
        logging.error(f"❌ 作业未能成功。最终状态: {status}")
        if hasattr(batch_job_status, 'errors') and batch_job_status.errors:
            logging.error(f"  - 错误详情: {batch_job_status.errors}")


if __name__ == "__main__":
    run_abstract_extraction()

