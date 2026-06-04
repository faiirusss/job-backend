import io
import json

from loguru import logger

from app.utils.logger import configure_logging


def test_configure_logging_emits_json():
    sink = io.StringIO()
    configure_logging(level="INFO", sink=sink)
    logger.info("hello", query_id=42, portal="glints")
    line = sink.getvalue().strip()
    parsed = json.loads(line)
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["extra"]["query_id"] == 42
    assert parsed["extra"]["portal"] == "glints"
