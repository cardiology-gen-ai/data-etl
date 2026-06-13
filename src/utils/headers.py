import json
import pathlib
from typing import List

from managers.parsing.parsing_manager import ParsedHeading


def headings_sidecar_path(cache_dir: pathlib.Path, stem: str) -> pathlib.Path:
    return pathlib.Path(cache_dir) / f"{stem}.headings.json"


def dump_headings(
        cache_dir: pathlib.Path,
        stem: str,
        headings: List[ParsedHeading],
) -> pathlib.Path:
    """Persist headings to the sidecar JSON.  Returns the file path."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = headings_sidecar_path(cache_dir, stem)
    payload = [h.model_dump() for h in headings]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_headings(cache_dir: pathlib.Path, stem: str) -> List[ParsedHeading]:
    """Load headings from the sidecar, or return ``[]`` if absent."""
    path = headings_sidecar_path(cache_dir, stem)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParsedHeading(**d) for d in data]