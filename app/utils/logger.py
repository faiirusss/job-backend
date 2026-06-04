import json
import sys
from typing import Any, TextIO

from loguru import logger


def _json_sink(message: Any) -> str:
    record = message.record
    payload = {
        "time": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "extra": record["extra"],
    }
    return json.dumps(payload, default=str)


_DEV_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(
    level: str = "INFO",
    sink: TextIO | None = None,
    app_env: str = "production",
) -> None:
    logger.remove()
    target = sink if sink is not None else sys.stdout

    if app_env == "development":
        # Human-readable, colorized console output for live `make dev` monitoring.
        logger.add(
            target,
            level=level,
            format=_DEV_FORMAT,
            colorize=True,
            serialize=False,
            enqueue=False,
        )
        return

    def _emit(message: Any) -> None:
        target.write(_json_sink(message) + "\n")
        target.flush()

    logger.add(_emit, level=level, format="{message}", enqueue=False)
