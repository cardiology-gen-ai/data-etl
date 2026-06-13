import json
import pathlib
import re
from typing import Iterable, List, Optional

from cardiology_gen_ai import Singleton, get_logger

from config.manager import PreprocessingConfig
from managers.tables.scorer import CaptionScorer, LexicalScorer
from managers.tables.table_manager import (
    SectionAttribution,
    TablesCatalog,
    TablesCatalogEntry,
    TableManager,
)
from managers.toc_extraction.table_of_contents_manager import TOCMetadata, TOCSection


_CAPTION_TOPIC_RE = re.compile(
    r"^\s*(?:Recommendation\s+)?Table\s+\d+\s*[:—\-–]\s*(.+?)\s*$",
    re.IGNORECASE,
)


class SectionAttributor:
    def __init__(
        self,
        toc_path: pathlib.Path,
        scorer: Optional[CaptionScorer] = None,
        low_confidence_threshold: float = 0.30,
    ):
        self.scorer = scorer or LexicalScorer()
        self.threshold = low_confidence_threshold

        toc_payload = json.loads(pathlib.Path(toc_path).read_text(encoding="utf-8"))
        self._toc: List[TOCSection] = TOCMetadata(**toc_payload).flat_toc
        self._toc_by_id = {s.id: s for s in self._toc if s.id}

        # Pre-warm the scorer with all section titles (no-op for lexical).
        self.scorer.prime([s.title for s in self._toc])

    def attribute(self, entry: TablesCatalogEntry) -> Optional[SectionAttribution]:
        """SectionAttribution for one entry, or None if no caption/page."""
        topic_text = self._extract_caption_topic(entry.caption)

        if topic_text is None or entry.page is None:
            return None

        container = self._container_for_page(entry.page)
        if container is None:
            return None

        candidates = {
            s.id: s
            for s in self._ancestors_inclusive(container.id)
            if s.id
        }

        # Add nearby boundary sections and their ancestors: recommendation tables
        # often sit at the end/start of a section.
        for s in self._toc:
            if s.page_end in (entry.page, entry.page - 1) and s.id not in candidates:
                for anc in self._ancestors_inclusive(s.id):
                    if anc.id:
                        candidates.setdefault(anc.id, anc)

        cand_list = list(candidates.values())
        if not cand_list:
            return None

        scores = self.scorer.score_many(topic_text, [c.title for c in cand_list])
        scored = sorted(zip(scores, cand_list), key=lambda x: -x[0])
        best_score, topic = scored[0]

        cont_chain_ids = {
            s.id
            for s in self._ancestors_inclusive(container.id)
            if s.id
        }

        return SectionAttribution(
            container_id=container.id,
            container_title=container.title,
            topic_id=topic.id,
            topic_title=topic.title,
            topic_score=round(best_score, 3),
            section_path=[
                f"{s.id}. {s.title}"
                for s in reversed(list(self._ancestors_inclusive(container.id)))
            ],
            cross_section=topic.id not in cont_chain_ids,
            low_confidence=best_score < self.threshold,
            scorer=self.scorer.name,
        )

    @staticmethod
    def _extract_caption_topic(caption: Optional[str]) -> Optional[str]:
        if not caption:
            return None
        m = _CAPTION_TOPIC_RE.match(caption.strip())
        # If it is "Recommendation Table N — topic" or "Table N — topic",
        # use just the topic; otherwise keep the full caption for attribution.
        return m.group(1).strip() if m else caption.strip()

    def _container_for_page(self, page: int) -> Optional[TOCSection]:
        candidates = [
            s
            for s in self._toc
            if s.page_start <= page <= s.page_end
        ]
        return max(candidates, key=lambda s: s.level) if candidates else None

    def _ancestors_inclusive(self, section_id: str) -> Iterable[TOCSection]:
        if not section_id:
            return

        parts = section_id.split(".")

        for k in range(len(parts), 0, -1):
            aid = ".".join(parts[:k])
            if aid in self._toc_by_id:
                yield self._toc_by_id[aid]


class SectionAttributionManager(metaclass=Singleton):
    """Per-file orchestration of table-to-section attribution."""

    def __init__(
        self,
        config: PreprocessingConfig,
        app_id: str,
        scorer: Optional[CaptionScorer] = None,
        low_confidence_threshold: float = 0.30,
    ):
        self.logger = get_logger("SectionAttributor")
        self.config = config
        self.app_id = app_id
        self.scorer = scorer or LexicalScorer()
        self.threshold = low_confidence_threshold

    def __call__(self, filepath: pathlib.Path) -> TablesCatalog:
        """Attribute sections for the table catalog of one document."""
        self.doc_id = filepath.stem

        save_folder = (
            pathlib.Path(self.config.output_folder.folder)
            / f"{self.doc_id}_tables"
        )

        tm = TableManager(filepath=filepath, save_folder=save_folder)
        tm.load(must_exist=True)

        if not tm.catalog.catalog:
            self.logger.info("Empty catalog for %s; skipping attribution.", filepath.name)
            return tm.catalog

        self.logger.info(
            "Attributing %d tables for %s using scorer=%s",
            len(tm.catalog.catalog),
            filepath.name,
            self.scorer.name,
        )

        toc_path = pathlib.Path(self.config.tocs_folder.folder) / f"{self.doc_id}.json"

        tm.attribute_sections(
            toc_path=toc_path,
            scorer=self.scorer,
            low_confidence_threshold=self.threshold,
        )

        attributed = [e for e in tm.catalog.catalog if e.attribution]
        cross = [e for e in attributed if e.attribution.cross_section]
        low = [e for e in attributed if e.attribution.low_confidence]

        self.logger.info(
            "  attributed: %d/%d (cross_section=%d, low_confidence=%d)",
            len(attributed),
            len(tm.catalog.catalog),
            len(cross),
            len(low),
        )

        return tm.catalog
