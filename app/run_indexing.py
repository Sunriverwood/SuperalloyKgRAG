import os
import sys
import yaml
from core.vlm_pdf_parser import VLMPdfParser
import logging
import json

def load_prompt(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def setup_logging(logging_cfg: dict):
    """
    根据配置初始化日志系统
    """
    level = getattr(logging, logging_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = logging_cfg.get("log_file")

    # 创建日志目录
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers
    )

    logger = logging.getLogger("SuperalloyKgRAG")
    return logger


def load_state(state_file: str):
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    # 1. 动态计算项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 可以使用pathlib.Path替代

    # 2. 使用绝对路径加载配置文件
    config_path = os.path.join(PROJECT_ROOT, "config/settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = f.read()
        raw_config = raw_config.replace("${GEMINI_API_KEY}", os.environ.get("GEMINI_API_KEY", ""))
        config = yaml.safe_load(raw_config)

    # 3. 将配置文件中所有的相对路径转换为绝对路径
    parser_cfg = config["vlm_parser"]
    parser_cfg["state_file_path"] = os.path.join(PROJECT_ROOT, parser_cfg["state_file_path"])
    parser_cfg["input_dir"] = os.path.join(PROJECT_ROOT, parser_cfg["input_dir"])
    parser_cfg["output_dir"] = os.path.join(PROJECT_ROOT, parser_cfg["output_dir"])
    parser_cfg["prompt_path"] = os.path.join(PROJECT_ROOT, "config/prompts/pdf_parsing.md")
    if "logging" in config and "log_file" in config["logging"]:
        config["logging"]["log_file"] = os.path.join(PROJECT_ROOT, config["logging"]["log_file"])

    # 4. 初始化日志 (不变)
    logger = setup_logging(config.get("logging", {}))

    # 5. 【新增】执行所有环境设置和初始数据加载
    logger.info("正在初始化环境和加载资源...")
    try:
        # 创建所有需要的目录
        os.makedirs(parser_cfg["input_dir"], exist_ok=True)
        os.makedirs(parser_cfg["output_dir"], exist_ok=True)
        # 确保cache目录存在
        os.makedirs(os.path.dirname(parser_cfg["state_file_path"]), exist_ok=True)

        # 加载初始状态
        initial_state = load_state(parser_cfg["state_file_path"])

        # 加载指令
        instructions = load_prompt(parser_cfg["prompt_path"])
        if not instructions:
            # 如果指令文件不存在，这是严重错误，直接退出
            raise FileNotFoundError(f"关键指令文件未找到: {parser_cfg['prompt_path']}")

    except Exception as e:
        logger.critical(f"初始化失败，程序退出: {e}")
        sys.exit(1)  # 程序异常退出

    logger.info("初始化完成。")

    # 6. 【修改】将加载好的数据"注入"到解析器中
    parser = VLMPdfParser(config, logger, initial_state, instructions)

    # 7. 运行核心逻辑 (不变)
    parser.upload_files()
    parser.create_batch_jobs()
    parser.monitor_jobs()
    parser.generate_report()