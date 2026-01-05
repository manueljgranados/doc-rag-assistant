from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


def load_pdf_pages(path: Path) -> Iterable[PageText]:
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", " ").strip()
        if text:
            yield PageText(page_number=i, text=text)
