import logging
from typing import Literal

LOGGING_LEVELS = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def set_logging(level: LOGGING_LEVELS = "WARNING") -> None:
    logging.basicConfig(level=level, format="%(asctime)s : %(name)s - %(levelname)s - %(message)s")


def get_logger(name: str, level: LOGGING_LEVELS = "DEBUG") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
