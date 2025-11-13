import os
import json
import time
import logging
from datetime import datetime, timedelta

from google.genai import types
from google.api_core import exceptions  # 引入特定的异常类型

from utils.client_factory import create_gemini_client


class VLMPdfParser:
    def __init__(self, config, logger: logging.Logger, initial_state: dict, instructions: str):
        parser_cfg = config["vlm_parser"]
        llm_cfg = config["llm"]

        self.logger = logger

        # 目录和文件路径仍然作为属性，但不再进行IO操作
        self.pdf_folder = parser_cfg["input_dir"]
        self.output_folder = parser_cfg["output_dir"]
        self.state_file = parser_cfg["state_file_path"]

        # 批处理配置
        self.batch_size = parser_cfg["batch_size"]
        self.timeout_seconds = parser_cfg.get("batch_polling_timeout_seconds")
        self.sleep_interval = parser_cfg.get("sleep_interval")

        # LLM配置
        self.model_name = llm_cfg["model"]
        self.client = create_gemini_client(
            api_key=llm_cfg["api_key"],
            proxy=config.get("proxy")
        )

        # 直接使用传入的初始状态和指令，不再自己加载
        self.state = initial_state
        self.instructions = instructions

    def list_pdfs(self, pdf_folder: str):
        return {f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")}

    def save_json(self, data: dict, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_state(self, state: dict, state_file: str):
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    # -------- 文件上传 --------
    def upload_files(self):
        self.logger.info("[阶段1] 文件发现与上传...")
        current_pdfs = self.list_pdfs(self.pdf_folder)
        for pdf_file in current_pdfs:
            if pdf_file not in self.state:
                self.state[pdf_file] = {"status": "pending_upload"}
        self.save_state(self.state, self.state_file)

        for pdf_file, data in self.state.items():
            if data["status"] == "pending_upload":
                pdf_path = os.path.join(self.pdf_folder, pdf_file)
                try:
                    self.logger.info(f"上传: {pdf_file}")
                    response = self.client.files.upload(file=pdf_path)
                    self.state[pdf_file].update({
                        "status": "uploaded",
                        "uploaded_file_uri": response.uri,
                        "uploaded_file_name": response.name
                    })
                except Exception as e:
                    self.state[pdf_file].update({"status": "failed_upload", "error": str(e)})
                    self.logger.error(f"上传失败 {pdf_file}: {e}")
                finally:
                    self.save_state(self.state, self.state_file)

    # -------- 批处理作业 --------
    def create_batch_jobs(self):
        self.logger.info("[阶段2] 创建批处理作业...")
        requests, files_for_jobs = [], []
        for pdf_file, data in self.state.items():
            if data["status"] == "uploaded":
                requests.append({
                    "key": pdf_file,
                    "request": {
                        "contents": [{"role": "user", "parts": [
                            {"text": self.instructions},
                            {"file_data": {"mime_type": "application/pdf", "file_uri": data["uploaded_file_uri"]}}
                        ]}],
                        "generationConfig": {"response_mime_type": "application/json"}
                    }
                })
                files_for_jobs.append(pdf_file)

        if not requests:
            self.logger.info("无待处理文件，无需创建新作业。")
            return

        chunks = [requests[i:i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        files_chunks = [files_for_jobs[i:i + self.batch_size] for i in range(0, len(files_for_jobs), self.batch_size)]

        for i, (chunk, files_in_chunk) in enumerate(zip(chunks, files_chunks)):
            job_name = f"KG-Batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i + 1}"
            tmp_file = f"temp_batch_requests_{i}.jsonl"
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    for req in chunk:
                        f.write(json.dumps(req) + "\n")

                batch_input = self.client.files.upload(
                    file=tmp_file,
                    config=types.UploadFileConfig(display_name=job_name, mime_type="jsonl")
                )
                job = self.client.batches.create(
                    model=self.model_name,
                    src=batch_input.name,
                    config={"display_name": job_name}
                )
                self.logger.info(f"✅ 作业创建成功: {job.name}")
                for pdf in files_in_chunk:
                    self.state[pdf].update({"status": "processing", "batch_job_name": job.name})
            except Exception as e:
                for pdf in files_in_chunk:
                    self.state[pdf].update({"status": "failed_job_creation", "error": str(e)})
                self.logger.error(f"❌ 批处理作业创建失败 {job_name}: {e}")
            finally:
                self.save_state(self.state, self.state_file)
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

    # -------- 作业监控 (已更新) --------
    def monitor_jobs(self):
        self.logger.info("[阶段3] 监控所有处理中的作业...")
        active_jobs = {d["batch_job_name"] for d in self.state.values() if d.get("status") == "processing"}

        if not active_jobs:
            self.logger.info("当前无活动作业需要监控。")
            return

        start_times = {name: datetime.now() for name in active_jobs}
        sleep_interval = self.sleep_interval

        while active_jobs:
            self.logger.info(f"正在监控 {len(active_jobs)} 个活动作业...")
            finished_jobs = set()

            for job_name in list(active_jobs):
                # 检查超时
                elapsed = datetime.now() - start_times.get(job_name, datetime.now())
                if elapsed.total_seconds() > self.timeout_seconds:
                    self.logger.warning(f"⏰ 作业 '{job_name}' 超时，正在尝试取消...")
                    try:
                        self.client.batches.cancel(name=job_name)
                    except exceptions.NotFound:
                        pass  # 作业可能已经结束或被删除

                    for pdf, data in self.state.items():
                        if data.get('batch_job_name') == job_name:
                            data.update({'status': 'failed_timeout', 'error': '批处理作业运行超时'})
                    finished_jobs.add(job_name)
                    continue

                # 获取作业状态
                try:
                    job = self.client.batches.get(name=job_name)
                    self.logger.info(f"  - 作业 '{job.name}' 当前状态: {job.state.name}")
                    if job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_EXPIRED', 'JOB_STATE_CANCELLED'):
                        self.logger.info(f"-> 作业 '{job.name}' 已完成，状态: {job.state.name}")
                        if job.state.name == 'JOB_STATE_SUCCEEDED':
                            self.process_job_results(job)
                        else:
                            error_detail = str(job.error) if job.error else f"作业以状态 {job.state.name} 结束"
                            for pdf, data in self.state.items():
                                if data.get('batch_job_name') == job_name:
                                    data.update({'status': f'failed_{job.state.name.lower()}', 'error': error_detail})
                        finished_jobs.add(job_name)

                except exceptions.NotFound:
                    self.logger.warning(f"⚠️ 作业 '{job_name}' 在API侧未找到，可能已被删除。将其标记为失败。")
                    for pdf, data in self.state.items():
                        if data.get('batch_job_name') == job_name:
                            data.update({'status': 'failed_job_not_found', 'error': '作业在API侧丢失'})
                    finished_jobs.add(job_name)
                except Exception as e:
                    self.logger.error(f"❌ 监控作业 '{job_name}' 时发生错误: {e}")

            if finished_jobs:
                active_jobs -= finished_jobs
                self.save_state(self.state, self.state_file)

            if active_jobs:
                self.logger.info(f"仍有 {len(active_jobs)} 个作业在运行中，将在 {sleep_interval // 60}分钟后再次检查...")
                time.sleep(sleep_interval)

    # -------- 结果处理 (已更新) --------
    def process_job_results(self, job):
        self.logger.info(f"  -> 正在处理作业 '{job.name}' 的结果...")
        if not (job.dest and job.dest.file_name):
            self.logger.error(f"❌ 错误：作业 '{job.name}' 成功，但未找到输出文件。")
            for pdf, data in self.state.items():
                if data.get('batch_job_name') == job.name:
                    data.update({'status': 'failed_job_no_output', 'error': '作业成功但无输出文件'})
            return

        result_file_name = job.dest.file_name
        try:
            self.logger.info(f"  - 📥 正在下载结果文件: {result_file_name}")
            content = self.client.files.download(file=result_file_name).decode("utf-8")

            for line in content.strip().split("\n"):
                result = json.loads(line)
                key = result.get("key")

                if not key or key not in self.state:
                    self.logger.warning(f"  - ⚠️ 警告：在结果文件中发现一个无效或不存在的 key '{key}'。")
                    continue

                if result.get("response"):
                    output_file = os.path.join(self.output_folder, os.path.splitext(key)[0] + ".json")
                    try:
                        text = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
                        data = json.loads(cleaned)
                        self.save_json(data, output_file)
                        self.state[key].update({"status": "completed", "output_path": output_file})
                        self.logger.info(f"    - ✅ 成功: '{key}' 的结果已保存到 {output_file}")
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        self.state[key].update({"status": "failed_parsing", "error": f"解析结果失败: {e}"})
                        self.logger.error(f"    - ❌ 失败: 解析 '{key}' 的结果时出错: {e}")
                elif result.get("error"):
                    error_message = result['error'].get('message', '未知API错误')
                    self.state[key].update({"status": "failed_in_job", "error": error_message})
                    self.logger.error(f"    - ❌ 失败: 处理 '{key}' 时API返回错误: {error_message}")
        except Exception as e:
            self.logger.critical(f"❌ 严重错误: 处理结果文件 '{result_file_name}' 时发生意外: {e}")
            for pdf, data in self.state.items():
                if data.get("batch_job_name") == job.name:
                    data.update({"status": "failed_processing_results", "error": str(e)})

    # -------- 报告 --------
    def generate_report(self):
        self.logger.info("=" * 50)
        self.logger.info("📋 [阶段4] 最终处理报告")
        self.logger.info("=" * 50)

        successful_files = [f for f, data in self.state.items() if data.get('status') == 'completed']
        failed_files = [f for f, data in self.state.items() if 'failed' in data.get('status', '')]
        pending_files = [f for f, data in self.state.items() if
                         data.get('status') not in ['completed'] and 'failed' not in data.get('status', '')]

        self.logger.info(f"\n✅ 处理成功 ({len(successful_files)} 个文件):")
        if successful_files:
            for f in successful_files:
                self.logger.info(f"  - {f}")
        else:
            self.logger.info("  - 无")

        self.logger.error(f"\n❌ 处理失败 ({len(failed_files)} 个文件):")
        if failed_files:
            for f in failed_files:
                error = self.state[f].get('error', '未知错误')
                status = self.state[f].get('status', '未知状态')
                self.logger.error(f"  - {f} (状态: {status}, 原因: {error})")
        else:
            self.logger.info("  - 无")

        if pending_files:
            self.logger.info(f"\n⏳ 待处理/处理中 ({len(pending_files)} 个文件):")
            for f in pending_files:
                self.logger.info(f"  - {f} (状态: {self.state[f].get('status')})")

        self.logger.info("\n" + "=" * 50)