import os
import sys
import yaml
from core.vlm_pdf_parser import VLMPdfParser
from utils.logging_config import setup_logging
from utils.state_manager import load_state  # 引入 state_manager
from utils.file_utils import load_prompt  # 引入 file_utils

if __name__ == "__main__":
    # 1. 动态计算项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. 使用绝对路径加载配置文件
    config_path = os.path.join(PROJECT_ROOT, "config/settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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