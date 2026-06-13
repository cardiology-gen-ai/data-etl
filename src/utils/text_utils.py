import re
import unicodedata
from typing import Optional, List


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def title_overlap_score(line_title: str, section_title: Optional[str]) -> float:
    """Compute Jaccard overlap score between line title and section title"""
    if not section_title:
        return 0.0
    lt = set(_normalize_tokens(line_title))
    st = set(_normalize_tokens(section_title))
    if not lt or not st:
        return 0.0
    return len(lt & st) / len(lt | st)


def post_process_markdown(text: str) -> str:
    """Clean and normalize Markdown produced by parsing."""
    lines = [
        line
        for line in text.split("\n")
        if not re.search(r"\[\.+\]", line)
    ]
    md = "\n".join(lines)
    md = re.sub(r"<!--\s*image\s*-->", "", md)
    md = re.sub(r"(#{1,6}\s+)\*\*(.+?)\*\*", r"\1\2", md)
    md = unicodedata.normalize("NFKC", md)
    md = md.replace("\r\n", "\n")
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.replace("\u00AD", "")
    md = re.sub(r"(\w)[-\u2010\u2011\u2212]\n(\w)", r"\1\2", md)
    return md.strip()