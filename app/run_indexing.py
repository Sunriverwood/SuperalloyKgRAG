import yaml
from core.vlm_pdf_parser import VLMPdfParser
from utils.logging_config import setup_logging

if __name__ == "__main__":
    config = yaml.safe_load(open("config/settings.yaml", "r", encoding="utf-8"))
    logger = setup_logging(config.get("logging", {}))

    parser = VLMPdfParser(config, logger)
    parser.upload_files()
    parser.create_batch_jobs()
    parser.monitor_jobs()
    parser.generate_report()
