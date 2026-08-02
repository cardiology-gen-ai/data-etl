"""Utilities for reading externally generated MinerU Markdown.

The KG preprocessing path treats these files as manual inputs.  This module
does not invoke MinerU, convert PDFs, or download any parsing resources.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class MinerUMarkdownDocument:
    doc_id: str
    path: Path
    text: str
    sha256: str


def normalize_for_matching(text: str) -> str:
    """Normalize text for robust heading comparisons, not for chunk output."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\u00ad", "")
    value = re.sub(r"(\w)[-\u2010\u2011\u2012\u2013\u2014\u2212]\s*\n\s*(\w)", r"\1\2", value)
    value = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    return value.strip()


def normalize_heading_for_matching(text: str) -> str:
    """Return a compact lowercase form suitable for heading title matching."""
    value = re.sub(r"<[^>]+>", " ", text or "")
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[*_`#~]+", " ", value)
    value = normalize_for_matching(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .:-").casefold()


def _candidate_paths(
    doc_id: str,
    markdown_root: Optional[Path],
    markdown_path: Optional[Path],
) -> Iterable[Path]:
    if markdown_path is not None:
        if markdown_path.is_dir():
            yield markdown_path / f"{doc_id}.md"
            yield markdown_path / doc_id / f"{doc_id}.md"
            yield markdown_path / doc_id / "full.md"
        else:
            yield markdown_path

    if markdown_root is not None:
        yield markdown_root / f"{doc_id}.md"
        yield markdown_root / doc_id / f"{doc_id}.md"
        yield markdown_root / doc_id / "full.md"


def resolve_mineru_markdown_path(
    doc_id: str,
    markdown_root: Optional[Path],
    markdown_path: Optional[Path] = None,
) -> Path:
    """Resolve the manual MinerU Markdown path for one document id."""
    candidates = []
    for candidate in _candidate_paths(doc_id, markdown_root, markdown_path):
        candidate = Path(candidate).expanduser()
        if candidate not in candidates:
            candidates.append(candidate)
        if candidate.is_file():
            return candidate

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Expected externally generated MinerU Markdown for "
        f"doc_id={doc_id!r}, but no file was found. Checked: {expected}"
    )


def load_mineru_markdown(
    doc_id: str,
    markdown_root: Optional[Path],
    markdown_path: Optional[Path] = None,
) -> MinerUMarkdownDocument:
    """Read an already generated MinerU Markdown file for ``doc_id``."""
    path = resolve_mineru_markdown_path(
        doc_id=doc_id,
        markdown_root=markdown_root,
        markdown_path=markdown_path,
    )
    text = path.read_text(encoding="utf-8")
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return MinerUMarkdownDocument(
        doc_id=doc_id,
        path=path,
        text=text,
        sha256=sha256,
    )
