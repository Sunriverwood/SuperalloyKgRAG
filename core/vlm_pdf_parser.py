import os, json, time, logging
from datetime import datetime
from google.genai import types

from utils.state_manager import load_state, save_state
from utils.file_utils import list_pdfs, save_json, load_prompt
from utils.client_factory import create_gemini_client


class VLMPdfParser:
    def __init__(self, config, logger: logging.Logger):
        parser_cfg = config["vlm_parser"]
        llm_cfg = config["llm"]

        self.logger = logger

        # 目录配置
        self.pdf_folder = parser_cfg["input_dir"]
        self.output_folder = parser_cfg["output_dir"]
        self.state_file = parser_cfg["state_file_path"]
        os.makedirs(self.pdf_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        # 批处理配置
        self.batch_size = parser_cfg["batch_size"]
        self.timeout_seconds = parser_cfg["batch_polling_timeout_seconds"]

        # LLM配置
        self.model_name = llm_cfg["model"]
        self.client = create_gemini_client(
            api_key=llm_cfg["api_key"],
            proxy=config.get("proxy")
        )

        # 加载状态与Prompt
        self.state = load_state(self.state_file)
        self.instructions = load_prompt("config/prompts/pdf_parsing.md")
        if not self.instructions:
            raise FileNotFoundError("未找到解析指令文件 pdf_parsing.md")

    # -------- 文件上传 --------
    def upload_files(self):
        self.logger.info("[阶段1] 文件发现与上传...")
        current_pdfs = list_pdfs(self.pdf_folder)
        for pdf_file in current_pdfs:
            if pdf_file not in self.state:
                self.state[pdf_file] = {"status": "pending_upload"}
        save_state(self.state, self.state_file)

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
                save_state(self.state, self.state_file)

    # -------- 批处理作业 --------
    def create_batch_jobs(self):
        self.logger.info("[阶段2] 创建批处理作业...")
        requests, files = [], []
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
                files.append(pdf_file)

        if not requests:
            self.logger.info("无待处理文件")
            return

        chunks = [requests[i:i+self.batch_size] for i in range(0, len(requests), self.batch_size)]
        files_chunks = [files[i:i+self.batch_size] for i in range(0, len(files), self.batch_size)]

        for i, (chunk, files_in_chunk) in enumerate(zip(chunks, files_chunks)):
            job_name = f"KG-Batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i+1}"
            tmp_file = f"temp_batch_{i}.jsonl"
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    for req in chunk: f.write(json.dumps(req) + "\n")
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
                save_state(self.state, self.state_file)
                if os.path.exists(tmp_file): os.remove(tmp_file)

    # -------- 作业监控 --------
    def monitor_jobs(self):
        self.logger.info("[阶段3] 监控作业...")
        active_jobs = {d["batch_job_name"] for d in self.state.values() if d.get("status") == "processing"}
        start_times = {name: datetime.now() for name in active_jobs}

        while active_jobs:
            for job_name in list(active_jobs):
                try:
                    job = self.client.batches.get(name=job_name)
                    if job.state.name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                                          "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"):
                        self.logger.info(f"-> 作业 {job.name} 结束: {job.state.name}")
                        if job.state.name == "JOB_STATE_SUCCEEDED":
                            self.process_job_results(job)
                        else:
                            for pdf, d in self.state.items():
                                if d.get("batch_job_name") == job_name:
                                    d.update({"status": "failed", "error": f"{job.state.name}"})
                        save_state(self.state, self.state_file)
                        active_jobs.remove(job_name)
                except Exception as e:
                    self.logger.error(f"作业 {job_name} 查询失败: {e}")
            time.sleep(600)

    # -------- 结果处理 --------
    def process_job_results(self, job):
        if not (job.dest and job.dest.file_name):
            return
        try:
            content = self.client.files.download(file=job.dest.file_name).decode("utf-8")
            for line in content.strip().split("\n"):
                result = json.loads(line)
                key = result.get("key")
                if not key or key not in self.state:
                    self.logger.warning("结果文件中发现无效 key")
                    continue
                if result.get("response"):
                    text = result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
                    try:
                        data = json.loads(cleaned)
                        out_file = os.path.join(self.output_folder, os.path.splitext(key)[0] + ".json")
                        save_json(data, out_file)
                        self.state[key].update({"status": "completed", "output": out_file})
                        self.logger.info(f"结果已保存: {out_file}")
                    except Exception as e:
                        self.state[key].update({"status": "failed_parsing", "error": str(e)})
                        self.logger.error(f"解析失败 {key}: {e}")
                elif result.get("error"):
                    self.state[key].update({"status": "failed_in_job", "error": result["error"].get("message")})
                    self.logger.error(f"API返回错误 {key}: {result['error'].get('message')}")
        except Exception as e:
            for pdf, d in self.state.items():
                if d.get("batch_job_name") == job.name:
                    d.update({"status": "failed_processing_results", "error": str(e)})
            self.logger.critical(f"处理结果文件失败: {e}")

    # -------- 报告 --------
    def generate_report(self):
        self.logger.info("[阶段4] 最终报告")
        success = [f for f,d in self.state.items() if d.get("status")=="completed"]
        failed  = [f for f,d in self.state.items() if "failed" in d.get("status","")]
        pending = [f for f,d in self.state.items() if d.get("status") not in ["completed"] and "failed" not in d.get("status","")]

        self.logger.info(f"✅ 成功 {len(success)} 个")
        for f in success: self.logger.info(f" - {f}")
        self.logger.info(f"❌ 失败 {len(failed)} 个")
        for f in failed: self.logger.error(f" - {f}, {self.state[f].get('error')}")
        self.logger.info(f"⏳ 待处理 {len(pending)} 个")
        for f in pending: self.logger.info(f" - {f}")
