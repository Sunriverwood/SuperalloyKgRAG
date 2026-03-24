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

import logging
from typing import Dict, Any

import json

from core.vlm_pdf_parser import VLMPdfParser, load_config, setup_logging


class PaperPdfParser(VLMPdfParser):
    """
    学术论文 PDF 解析器 (Gemini 版)

    完全继承 VLMPdfParser，仅通过 config_key="paper_parser" 切换配置节点。
    差异点由配置驱动：
    - 输入目录指向 data/original_data/full_text/
    - 使用论文专用 prompt (paper_pdf_parsing.md)
    - 状态文件独立 (paper_processing_state.json)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, config_key="paper_parser")
        self.logger = logging.getLogger("paper_pdf_parser")

    def process_job_results(self, job):
        """处理作业结果，额外打印论文元数据日志"""
        super().process_job_results(job)

        # 遍历本次作业已完成的文件，打印论文元数据
        for pdf_file, file_state in self.state.items():
            if file_state.get("batch_job_name") != job.name:
                continue
            if file_state.get("status") != "completed":
                continue

            output_path = file_state.get("output_path")
            if not output_path:
                continue

            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                metadata = data.get("paper_metadata", {})
                if metadata:
                    title = metadata.get("title", "N/A")
                    self.logger.info(f"    - 📄 论文: {title[:80]}")
                    self.logger.info(f"    - 📋 期刊: {metadata.get('journal', 'N/A')}")
                    self.logger.info(f"    - 📅 年份: {metadata.get('year', 'N/A')}")
                    self.logger.info(f"    - 🔗 DOI: {metadata.get('doi', 'N/A')}")
                    pages = data.get("pages", [])
                    self.logger.info(f"    - 📑 共解析 {len(pages)} 页内容")
            except Exception as e:
                self.logger.warning(f"    - ⚠️ 读取论文元数据失败 ({pdf_file}): {e}")


def main():
    try:
        config = load_config()
        setup_logging(config)
        parser = PaperPdfParser(config)

        parser.upload_files()
        parser.wait_for_files_active()
        parser.create_batch_jobs()
        parser.monitor_jobs()
        parser.generate_report()

    except Exception as e:
        logging.critical(f"程序运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()
