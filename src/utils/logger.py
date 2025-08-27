import os
import logging
from typing import Literal
from pydantic import BaseModel


class LoggerConfig(BaseModel):
    level: str = "INFO"
    handler: Literal["stream", "file"] = "stream"
    file_path: str = os.path.join(os.getcwd(), "app.log")


log_levels_map = {
        "debug": logging.DEBUG, # ok for dev
        "info": logging.INFO, # ok for dev and staging
        "warning": logging.WARNING, # ok for production
        "error": logging.ERROR,
        "critical": logging.CRITICAL
}


def get_logger(name: str, logger_config: LoggerConfig = None) -> logging.Logger:
    config = logger_config
    if logger_config is not None:
        update_logger_level = log_levels_map.get(config.level.lower(), logging.INFO)
        config = config.model_copy(update={"level": update_logger_level})
    else:
        config = LoggerConfig()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(filename)-15s:%(lineno)-4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(config.level)
    if config.handler == "stream":
        handler = logging.StreamHandler()
    else:
        assert config.handler == "file"
        handler = logging.FileHandler(filename=config.file_path)
    handler.setLevel(config.level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

