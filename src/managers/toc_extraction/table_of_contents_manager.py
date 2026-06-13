import json
import pathlib
from enum import Enum
from typing import List, Optional, Tuple, Set, Any

import fitz
from pydantic import BaseModel, Field, model_validator

from cardiology_gen_ai.utils.logger import get_logger
from cardiology_gen_ai.utils.singleton import Singleton
from config.manager import PreprocessingConfig
from managers.parsing.parsing_manager import ParsedHeading
from managers.toc_extraction.docling_toc import extract_toc_from_docling
from managers.toc_extraction.fallback_toc import extract_toc_from_text
from managers.toc_extraction.mineru_toc import extract_toc_from_backend_headings
from managers.toc_extraction.raw_toc import extract_toc_from_fitz_outline
from managers.toc_extraction.toc_configs import TocConfigFactory
from managers.toc_extraction.utils import (
    build_tree,
    classify_sections,
    compute_fallback_ranges,
    compute_outline_ranges,
)
from utils.headers import load_headings


class TOCSectionType(str, Enum):
    body = "body"
    front_matter = "front_matter"
    back_matter = "back_matter"
    toc = "toc"


class TOCSection(BaseModel):
    id: Optional[str]
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    type: TOCSectionType = TOCSectionType.body
    children: List["TOCSection"] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError(f"Invalid page range: {self.page_start}-{self.page_end}")
        return self


class TOCMetadata(BaseModel):
    doc_id: str
    toc_source: str
    n_pages: int
    flat_toc: List[TOCSection]
    toc_tree: List[TOCSection]


TOCSection.model_rebuild()


class TOCExtractionManager(metaclass=Singleton):

    def __init__(self, config: PreprocessingConfig, app_id: str = "upper_gi_protocols"):
        self.logger = get_logger("TOCExtractionManager")
        self.config = config
        self.toc_config = TocConfigFactory.get_toc_config(app_id)
        assert self.config.tocs_folder is not None
        pathlib.Path(self.config.tocs_folder.folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        filename: str,
        headings: Optional[List[ParsedHeading]] = None,
    ) -> TOCMetadata:
        """Run the extraction.

        Parameters
        ----------
        filename : str
            Stem of the PDF file (without ``.pdf``).
        headings : list[ParsedHeading], optional
            Override the sidecar-loaded headings.  Pass an empty list to
            force-skip Path 0 even if a sidecar exists.
        """
        self.filename = filename
        self.filepath = self.config.input_folder.folder / (filename + ".pdf")
        self.doc: fitz.Document = fitz.open(self.filepath)
        self.n_pages = self.doc.page_count

        if headings is None:
            cache_dir = (
                self.config.cache_folder.folder
                if self.config.cache_folder is not None
                else self.config.output_folder.folder
            )
            headings = load_headings(cache_dir, filename)
            if headings:
                self.logger.info(
                    f"Loaded {len(headings)} headings from sidecar"
                )
        self._headings_hint = headings or []

        try:
            return self._run()
        finally:
            self.doc.close()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self) -> TOCMetadata:
        raw_toc, toc_source = self._try_extract()
        flat_toc = [
            s.model_copy(update={"children": []}, deep=True)
            for s in raw_toc
        ]
        classify_sections(raw_toc, self.toc_config)
        tree_input = [
            s.model_copy(update={"children": []}, deep=True)
            for s in raw_toc
        ]

        tree = build_tree(tree_input)
        self.logger.info(
            f"TOC done ({toc_source}): {len(raw_toc)} sections, {len(tree)} roots"
        )

        toc_metadata = TOCMetadata(
            doc_id=self.filename,
            toc_source=toc_source,
            n_pages=self.n_pages,
            flat_toc=flat_toc,
            toc_tree=tree,
        )
        toc_filepath = self.config.tocs_folder.folder / (self.filepath.stem + ".json")
        with open(toc_filepath, "w", encoding="utf-8") as f:
            json.dump(toc_metadata.model_dump(), f, indent=2, ensure_ascii=False)
        self.logger.info(f"TOC saved to {toc_filepath}")
        return toc_metadata

    def _try_extract(self) -> Tuple[List[TOCSection], str]:
        # ---- Path 0: backend-provided headings ----
        if self._headings_hint:
            try:
                sections = extract_toc_from_backend_headings(
                    self._headings_hint, self.toc_config, self.logger
                )
                if not sections:
                    raise ValueError("backend headings produced no usable sections")
                compute_fallback_ranges(sections)
                return sections, "backend_headings"
            except Exception as exc:
                self.logger.warning(
                    f"backend headings failed ({exc}) — trying fitz outline"
                )

        # ---- Path 1: fitz outline ----
        try:
            sections = extract_toc_from_fitz_outline(self.doc, self.toc_config, self.logger)
            if not sections:
                raise ValueError("fitz outline produced no usable sections")
            compute_outline_ranges(
                sections, self.n_pages, self.doc,
                self.toc_config.heading_top_y_thresholds,
            )
            return sections, "pdf_outline"
        except Exception as exc:
            self.logger.warning(f"fitz outline failed ({exc}) — trying Docling headers")

        # ---- Path 2: Docling headers ----
        cache_dir = (
            self.config.cache_folder.folder
            if self.config.cache_folder is not None
            else self.config.output_folder.folder
        )
        try:
            sections = extract_toc_from_docling(
                self.filepath, cache_dir, self.toc_config, self.logger
            )
            if not sections:
                raise ValueError("Docling headers produced no usable sections")
            compute_fallback_ranges(sections)
            return sections, "docling_headers"
        except Exception as exc:
            self.logger.warning(f"Docling headers failed ({exc}) — trying textual TOC")

        # ---- Path 3: textual TOC ----
        sections = extract_toc_from_text(self.doc, self.n_pages, self.toc_config, self.logger)
        if not sections:
            raise RuntimeError("All TOC extraction paths failed to produce sections")
        compute_fallback_ranges(sections)
        return sections, "textual_toc"

    # ------------------------------------------------------------------
    # Pretty-printing / post-processing helpers (unchanged from legacy)
    # ------------------------------------------------------------------

    def toc_to_txt(
            self,
            sections: List[TOCSection],
            indent: int = 0,
            indent_step: int = 4,
            recursive: bool = True,
            deduplicate: bool = True,
    ) -> str:
        lines: List[str] = []
        seen: Set[Tuple[str, str, int, int]] = set()

        def get(sec: Any, name: str, default=None):
            if isinstance(sec, dict):
                return sec.get(name, default)
            return getattr(sec, name, default)

        def walk(nodes: List[Any], current_indent: int) -> None:
            for sec in nodes or []:
                sec_id = str(get(sec, "id", "") or "").strip()
                title = str(get(sec, "title", "") or "").strip()
                page_start = int(get(sec, "page_start", 0) or 0)
                page_end = int(get(sec, "page_end", page_start) or page_start)

                if not title:
                    continue

                key = (sec_id, title, page_start, page_end)
                if deduplicate and key in seen:
                    continue
                seen.add(key)

                id_part = f"{sec_id} " if sec_id else ""
                lines.append(
                    f"{' ' * current_indent}{id_part}{title} ...... {page_start}-{page_end}"
                )

                if recursive:
                    children = get(sec, "children", []) or []
                    walk(children, current_indent + indent_step)

        walk(sections, indent)
        return "\n".join(lines)

    def normalize_section_ids(self, section: TOCSection) -> TOCSection:
        if section.id is None and isinstance(section.title, str) and self.toc_config.section_re is not None:
            m = self.toc_config.section_re.match(section.title)
            if m:
                section.id = m.group("id")
                section.title = m.group("title")
        section.children = [self.normalize_section_ids(c) for c in section.children or []]
        return section

    @staticmethod
    def prune_ghost_sections(section: TOCSection) -> List[TOCSection]:
        if section.id is None and not section.children:
            return []
        if section.id is None:
            return [
                pruned
                for child in section.children
                for pruned in TOCExtractionManager.prune_ghost_sections(child)
            ]
        section.children = [
            pruned
            for child in (section.children or [])
            for pruned in TOCExtractionManager.prune_ghost_sections(child)
        ]
        return [section]