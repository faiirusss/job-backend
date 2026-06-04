import io
import re
import unicodedata

import pdfplumber


class PDFEmptyError(Exception):
    pass


_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE.sub(" ", ln).rstrip() for ln in text.split("\n")]
    out = "\n".join(lines)
    return _MULTI_BLANK.sub("\n\n", out).strip()


def extract_text(pdf_bytes: bytes) -> str:
    parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                if t.strip():
                    parts.append(t)
    except Exception as e:
        raise PDFEmptyError(f"Failed to parse PDF: {e}") from e

    text = _clean("\n\n".join(parts))
    if not text:
        raise PDFEmptyError("PDF contains no extractable text")
    return text
