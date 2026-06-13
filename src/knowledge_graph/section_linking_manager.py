import json
import pathlib
import pickle
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from cardiology_gen_ai import Singleton

from config.manager import PreprocessingConfig
from managers.chunking.chunking_manager import ChunkMetadata
from cardiology_gen_ai.utils.logger import get_logger

from managers.tables.table_manager import TableManager

LINKING_VERSION = "v1.0"


@dataclass
class SectionLink:
    """Links one recommendation to its supporting chunks.

    `match_strategy` records HOW the chunks were selected -- useful when
    auditing low-recall cases:
      - 'section_id_exact'   -> target section_id matched directly
      - 'section_id_subtree' -> matched on descendants only
      - 'title_fallback'     -> couldn't extract a numeric id, matched by title
      - 'empty'              -> no chunks found (review needed)
    """

    recommendation_id: str
    table_id: str
    row_index: int
    target_section_id: Optional[str]
    target_section_title: str
    chunk_ids: list[str]
    chunk_section_ids: list[str]
    match_strategy: str
    linking_version: str = LINKING_VERSION
    linked_at: Optional[str] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: dict) -> "SectionLink":
        return cls(**line)


@dataclass
class SectionLinksCatalog:
    catalog: list[SectionLink] = field(default_factory=list)

    def done_ids(self, version: str) -> set[str]:
        return {l.recommendation_id for l in self.catalog
                if l.linking_version == version}

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.catalog),
            "linked": sum(1 for l in self.catalog if l.chunk_ids),
            "empty": sum(1 for l in self.catalog if not l.chunk_ids),
            "by_strategy_exact": sum(
                1 for l in self.catalog if l.match_strategy == "section_id_exact"
            ),
            "by_strategy_subtree": sum(
                1 for l in self.catalog if l.match_strategy == "section_id_subtree"
            ),
            "by_strategy_title": sum(
                1 for l in self.catalog if l.match_strategy == "title_fallback"
            ),
        }


_SECTION_ID_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")


def extract_section_id(title: str) -> Optional[str]:
    """Extract numeric section id from a title like '5.1.1. Weight reduction'."""
    if not title:
        return None
    m = _SECTION_ID_RE.match(title)
    return m.group(1) if m else None


def normalise_title(title: str) -> str:
    """For title-fallback matching: lowercase, strip leading numbers/punct."""
    return re.sub(r"^[\d\.\s]+", "", title or "").strip().lower()




class SectionLinksManager:
    """Per-document I/O + linking."""

    def __init__(
            self,
            tables_catalog,  # TablesCatalog
            chunks: list[Any],  # list[Document] from pickle
            save_path: pathlib.Path,
            linking_version: str,
    ):
        self.logger = get_logger("SectionLinksManager")
        self.tables = tables_catalog
        self.chunks = chunks
        self.save_path = save_path
        self.linking_version = linking_version
        self.catalog: SectionLinksCatalog = SectionLinksCatalog()

        # Pre-index chunks by section_id and normalised title for O(1) lookup.
        self._chunks_by_section_id: dict[str, list[tuple[str, str]]] = {}
        self._chunks_by_norm_title: dict[str, list[tuple[str, str]]] = {}
        for ch in chunks:
            meta = ChunkMetadata.model_validate(ch.metadata)
            sec_id = getattr(meta, "section_id", None)
            sec_title = getattr(meta, "section_title", None) or ""
            chunk_id = getattr(meta, "chunk_id", None) or f"chunk_{id(ch)}"
            if getattr(meta, "is_empty", False):
                continue
            if sec_id:
                self._chunks_by_section_id.setdefault(sec_id, []).append(
                    (chunk_id, sec_id)
                )
            if sec_title:
                self._chunks_by_norm_title.setdefault(
                    normalise_title(sec_title), []
                ).append((chunk_id, sec_id or ""))

    def load(self, must_exist: bool = False) -> None:
        if not self.save_path.exists():
            if must_exist:
                raise FileNotFoundError(self.save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            return
        with open(self.save_path, "r") as f:
            data = json.load(f)
            for entry in data:
                try:
                    self.catalog.catalog.append(SectionLink.from_json(entry))
                except (json.JSONDecodeError, TypeError) as exc:
                    self.logger.warning("Skipping malformed JSONL line: %s", exc)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.save_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            data = [json.loads(entry.to_jsonl()) for entry in self.catalog.catalog]
            json.dump(data, f, indent=4, ensure_ascii=False)
        tmp.replace(self.save_path)


    def _find_chunks(self, target_section_id: Optional[str],
                     target_title: str) -> tuple[list[str], list[str], str]:
        """Return (chunk_ids, chunk_section_ids, match_strategy)."""
        if target_section_id:
            # 1. Exact section id match.
            exact = self._chunks_by_section_id.get(target_section_id, [])
            # 2. Descendants (chunks whose section_id starts with target + ".").
            prefix = target_section_id + "."
            subtree = [
                pair for sid, pairs in self._chunks_by_section_id.items()
                if sid.startswith(prefix)
                for pair in pairs
            ]
            combined = exact + subtree
            if combined:
                chunk_ids = [cid for cid, _ in combined]
                sec_ids = [sid for _, sid in combined]
                strategy = "section_id_exact" if exact and not subtree else (
                    "section_id_subtree" if not exact else "section_id_exact"
                )
                return chunk_ids, sec_ids, strategy

        # 3. Title fallback.
        title_key = normalise_title(target_title)
        if title_key:
            hits = self._chunks_by_norm_title.get(title_key, [])
            if hits:
                return [c for c, _ in hits], [s for _, s in hits], "title_fallback"

        return [], [], "empty"

    def link(self) -> SectionLinksCatalog:
        done = self.catalog.done_ids(self.linking_version)
        new_count = 0
        for table in self.tables.catalog:
            attribution = getattr(table, "attribution", None)
            section_path = list(getattr(attribution, "section_path", []) or [])
            target_title = section_path[-1] if section_path else (
                    getattr(attribution, "topic_title", "") or ""
            )
            target_id = extract_section_id(target_title)

            for row in table.recommendation_rows:
                if getattr(row, "is_section_row", False):
                    continue
                rec_id = f"{table.id}::row_{row.row_index}"
                if rec_id in done:
                    continue
                if not (getattr(row, "recommendation", "") or "").strip():
                    continue

                chunk_ids, sec_ids, strategy = self._find_chunks(
                    target_id, target_title
                )
                self.catalog.catalog.append(SectionLink(
                    recommendation_id=rec_id,
                    table_id=table.id,
                    row_index=row.row_index,
                    target_section_id=target_id,
                    target_section_title=target_title,
                    chunk_ids=chunk_ids,
                    chunk_section_ids=sec_ids,
                    match_strategy=strategy,
                    linking_version=self.linking_version,
                    linked_at=datetime.now(timezone.utc).isoformat(),
                ))
                new_count += 1

        if new_count:
            self.save()
        return self.catalog


# ---------- Top-level orchestrator ------------------------------------------


class SectionLinkingManager(metaclass=Singleton):
    """Per-file orchestration of recommendation -> chunk section linking."""

    def __init__(
            self,
            config: PreprocessingConfig,
            tabs_folder: pathlib.Path,
            output_folder: pathlib.Path,
            app_id: str,
            linking_version: str = LINKING_VERSION,
    ):
        self.logger = get_logger("SectionLinker")
        self.config = config
        self.tabs_folder = tabs_folder
        self.output_folder = output_folder
        self.app_id = app_id
        self.linking_version = linking_version
        # Header level matches HierarchicalChunkingManager's convention.
        self.header_levels = (
                self.config.chunking_manager.splitter[0].header_levels or 1
        )

    def _load_chunks(self, doc_id: str) -> list[Any]:
        base = str(self.config.chunks_folder.folder)
        suffix = f"_{self.header_levels}" if self.header_levels > 0 else ""
        path = pathlib.Path(base + suffix) / f"{doc_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing chunks file for {doc_id}: {path}. "
                "Run HierarchicalChunkingManager first."
            )
        with path.open("rb") as f:
            return pickle.load(f)

    def __call__(self, filepath: pathlib.Path) -> SectionLinksCatalog:
        self.doc_id = filepath.stem

        # 1. Load upstream tables catalog.
        tm = TableManager(filepath=filepath, save_folder=self.tabs_folder)
        tm.load(must_exist=True, recommendation=True)

        if not tm.catalog.catalog:
            self.logger.info("Empty tables catalog for %s; skipping.", filepath.name)
            return SectionLinksCatalog()

        # 2. Load chunks.
        chunks = self._load_chunks(self.doc_id)

        # 3. Set up sub-manager.
        save_path = (pathlib.Path(self.output_folder) / f"{self.doc_id}_section_links.json")
        sm = SectionLinksManager(
            tables_catalog=tm.catalog,
            chunks=chunks,
            save_path=save_path,
            linking_version=self.linking_version,
        )
        sm.load(must_exist=False)

        self.logger.info(
            "Section-linking %d tables / %d chunks for %s",
            len(tm.catalog.catalog), len(chunks), filepath.name,
        )

        # 4. Run.
        sm.link()

        # 5. Summary.
        stats = sm.catalog.stats()
        self.logger.info(
            "  links: %d/%d (empty=%d, exact=%d, subtree=%d, title_fallback=%d)",
            stats["linked"], stats["total"], stats["empty"],
            stats["by_strategy_exact"], stats["by_strategy_subtree"],
            stats["by_strategy_title"],
        )

        return sm.catalog