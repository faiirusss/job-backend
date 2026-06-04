from pathlib import Path

import pytest

from app.utils.pdf import PDFEmptyError, extract_text

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_cv.pdf"


def test_extract_text_returns_visible_text():
    text = extract_text(FIXTURE.read_bytes())
    assert "ANDI PUTRA" in text
    assert "Python" in text
    assert "FastAPI" in text


def test_extract_text_empty_pdf_raises(tmp_path):
    empty_pdf_bytes = b"%PDF-1.4\n%%EOF\n"
    with pytest.raises(PDFEmptyError):
        extract_text(empty_pdf_bytes)
