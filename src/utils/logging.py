# src/utils/logging.py
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str | Path, level: str = "INFO", fmt: str = "json") -> None:
    """
    Configure loguru for structured logging.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # remove default handler

    if fmt == "json":
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,   # outputs JSON to stdout
        )
        logger.add(
            log_dir / "run_{time}.log",
            level=level,
            serialize=True,
            rotation="100 MB",
        )
    else:
        logger.add(sys.stdout, level=level)
        logger.add(log_dir / "run_{time}.log", level=level, rotation="100 MB")