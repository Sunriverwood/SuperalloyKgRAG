import logging
import os

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
