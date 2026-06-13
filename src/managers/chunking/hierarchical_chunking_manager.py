"""
Hierarchical Chunker

- Anchor-narrowed start detection (robust)
- TOC-order end boundaries (for leaf sections with no children)
- Parents stop at first child (prevents parent swallowing children)
- Skips TOC dot-leader lines as headers (or else we ingest TOC entries)
- Keeps empty parents, drops empty leaves
- Does NOT split oversized chunks #TODO: implement splitting later if needed, depends on use case
"""

import json
import pathlib
import pickle
import re
from typing import List, Any, Optional, Tuple, Dict

from langchain_core.documents import Document

from utils.text_utils import title_overlap_score
from managers.toc_extraction.table_of_contents_manager import TOCExtractionManager
from config.manager import PreprocessingConfig
from managers.chunking.chunking_manager import ChunkMetadata, ChunkingManager
from managers.toc_extraction.table_of_contents_manager import TOCSection, TOCMetadata, TOCSectionType
from managers.toc_extraction.toc_configs import TocConfigFactory, BODY_START_PATTERNS, HEADER_PATTERNS

from cardiology_gen_ai.utils.logger import get_logger


class ChunkingTOCSection(TOCSection):
    parent_id: Optional[str] = None
    has_children: bool = False


class TOCChunkMetadata(ChunkMetadata):
    chunk_id: str
    section_id: str
    parent_section_id: Optional[str | None] = None
    section_title: str
    section_level: int
    section_type: TOCSectionType
    page_start: int
    page_end: int
    is_empty: bool
    embed: bool

    def model_post_init(self, context: Any, /) -> None:
        # Only set a default if headers were not explicitly provided (e.g. in the
        # header_levels=0 path that never builds breadcrumb dicts).
        if not self.headers:
            self.headers = {"Header " + str(self.section_level): [self.section_title]}


class HierarchicalChunkingManager(ChunkingManager):

    def __init__(self, config: PreprocessingConfig, header_levels: int = 2, app_id: str = "cardiology_protocols"):
        self.config = config
        self.header_levels = self.config.chunking_manager.splitter[0].header_levels or header_levels
        self.app_id = app_id
        self.toc_config = TocConfigFactory.get_toc_config(app_id)
        self.logger = get_logger("Hierarchical Chunker based on Table of Contents.")
        pathlib.Path(self.config.chunks_folder.folder).mkdir(parents=True, exist_ok=True)
        if self.header_levels > 0:
            pathlib.Path(str(self.config.chunks_folder.folder) + f"_{self.header_levels}").mkdir(parents=True, exist_ok=True)


    def __call__(self, filepath: pathlib.Path) -> List[Document]:
        self.doc_id = filepath.stem
        toc_path = pathlib.Path(self.config.tocs_folder.folder) / (self.doc_id + ".json")
        toc_extractor = TOCExtractionManager(config=self.config, app_id=self.app_id)
        if not toc_path.exists() or not toc_path.is_file():
            file_toc = toc_extractor(self.doc_id)
        else:
            file_toc = TOCMetadata(**json.loads(toc_path.read_text(encoding="utf-8")))
        toc_txt_path = pathlib.Path(self.config.tocs_folder.folder) / (self.doc_id + ".txt")
        if not toc_txt_path.exists() or not toc_txt_path.is_file():
            file_txt_toc = toc_extractor.toc_to_txt(file_toc.toc_tree, recursive=True)
            toc_txt_path.write_text(file_txt_toc, encoding="utf-8")
        self.toc_tree = file_toc.toc_tree or file_toc.flat_toc[0].children
        new_sections = []
        for section in self.toc_tree:
            new_sections.append(toc_extractor.normalize_section_ids(section))
        self.toc_tree = [child for root in new_sections for child in toc_extractor.prune_ghost_sections(root)]
        md_path = pathlib.Path(self.config.output_folder.folder) / (self.doc_id + ".md")
        self.md_text = md_path.read_text(encoding="utf-8")
        pdf_path = pathlib.Path(self.config.input_folder.folder) / (self.doc_id + ".pdf")
        self.anchors = self._build_page_offsets(pdf_path)
        chunks = self.split_text()
        chunk_folder_suffix = f"_{self.header_levels}" if self.header_levels > 0 else ""
        output_path = pathlib.Path(str(self.config.chunks_folder.folder) + chunk_folder_suffix) / (self.doc_id + ".pkl")
        with open(output_path, "wb") as f:
            pickle.dump(chunks, f)
        self.logger.info(f"Saved chunks to {output_path}")
        return chunks

    @staticmethod
    def collect_sections(toc_nodes: List[ChunkingTOCSection | TOCSection], parent_id: Optional[str] = None) -> List[ChunkingTOCSection]:
        """Find all sections in TOC tree as flat list with parent references"""
        out: List[ChunkingTOCSection] = []
        for node in toc_nodes:
            entry = ChunkingTOCSection(**node.model_dump(), parent_id=parent_id, has_children=bool(node.children))
            out.append(entry)
            for child in node.children:
                out.extend(HierarchicalChunkingManager.collect_sections(
                    toc_nodes=[child], parent_id=node.id)
                )
        return out

    @staticmethod
    def find_body_start(markdown: str) -> int:
        matches = []
        for rx in BODY_START_PATTERNS:
            m = rx.search(markdown)
            if m:
                matches.append(m.start())
        return min(matches) if matches else 0

    def is_excluded_section(self, section: ChunkingTOCSection) -> bool:
        title = (section.title or "").lower()
        if section.type in {TOCSectionType.front_matter, TOCSectionType.back_matter, TOCSectionType.toc}:
            return True
        return any(k in title for k in self.toc_config.excluded_title_keywords) \
            if self.toc_config.excluded_title_keywords is not None else False

    def _compute_search_window(self, page_start: int, page_end: int, window: int = 1) -> Tuple[int, int]:
        """
        Returns (start_offset, end_offset) in markdown for a page range.
        Falls back to (0, len(markdown)) if anchors are missing.
        """
        markdown_len = len(self.md_text)
        if not self.anchors:
            return 0, markdown_len
        pages = sorted(self.anchors.keys())
        if not pages:
            return 0, markdown_len
        start_page = max(min(page_start - window, pages[-1]), pages[0])
        end_page = min(max(page_end + window, pages[0]), pages[-1])
        start_offset = self.anchors.get(start_page, self.anchors[pages[0]])
        end_offset = self.anchors.get(end_page + 1, markdown_len)
        return start_offset, end_offset

    @staticmethod
    def word_count(text: str) -> int:
        return len(re.findall(r"\w+", text))

    @staticmethod
    def is_toc_entry_line(line: str) -> bool:
        # e.g. "1. Preamble ............ 3509"
        return bool(re.search(r"\.{5,}\s*\d+\s*$", line.strip()))

    def locate_header(self, section_id, section_title, search_start, search_end):
        all_matches = []
        for rx in HEADER_PATTERNS:  # noqa: F821 (defined in original file)
            all_matches.extend(rx.finditer(self.md_text, search_start, search_end))

        if not all_matches:
            return None

        # tie-break per posizione nel documento
        all_matches.sort(key=lambda m: m.start())

        best, best_score = None, -1.0
        for match in all_matches:
            line = match.group(0)
            if self.is_toc_entry_line(line):
                continue
            title = (match.group("title") or "").strip()
            if not title:
                tail = self.md_text[match.end():search_end]
                for ln in tail.splitlines():
                    if ln.strip():
                        title = ln.strip()
                        break
            title = self.toc_config.section_id_re.sub("", title).strip() \
                if self.toc_config.section_id_re else title.strip()
            score = title_overlap_score(title, section_title)  # noqa: F821
            if score > best_score:
                best, best_score = match, score

        if best is None:
            return None
        if len(section_id.split(".")) == 1 and section_title and best_score < 0.15:
            return None
        return best.start(), best.end()

    def extract_section_text(
            self, section: TOCSection, next_section: Optional[TOCSection], first_child: Optional[TOCSection],
            body_start: int, window: int = 1,
    ) -> str:
        """
        Content extraction with TOC boundaries.
        Extract section content using:
        - header position found within anchor window (fallback to global guarded)
        - end boundary:
            - first child header if present
            - else next section header (TOC order)
            - else window end / EOF
        """
        md_len = len(self.md_text)

        search_window_start, search_window_end = self._compute_search_window(
            page_start=section.page_start, page_end=section.page_end, window=window,
        )
        search_window_start = max(search_window_start, body_start)

        header_pos = self.locate_header(
            section_id=section.id, section_title=section.title,
            search_start=search_window_start, search_end=search_window_end,
        )
        if header_pos is None:
            header_pos = self.locate_header(
                section_id=section.id, section_title=section.title,
                search_start=body_start, search_end=md_len,
            )
        if header_pos is None:
            return ""
        _, header_end = header_pos
        content_start = header_end
        content_end: Optional[int] = None

        if first_child is not None:
            child_header = self.locate_header(
                section_id=first_child.id, section_title=first_child.title,
                search_start=content_start, search_end=search_window_end,
            )
            if child_header is None:
                child_header = self.locate_header(
                    section_id=first_child.id, section_title=first_child.title,
                    search_start=content_start, search_end=md_len,
                )
            if child_header is not None:
                content_end = child_header[0]

        if content_end is None and next_section is not None:
            next_header = self.locate_header(
                section_id=next_section.id, section_title=next_section.title,
                search_start=content_start, search_end=md_len,
            )
            if next_header is not None:
                content_end = next_header[0]

        if content_end is None:
            content_end = search_window_end if search_window_end > content_start else md_len

        return self.md_text[content_start:content_end].strip()

    @staticmethod
    def looks_like_leakage(text: str, min_words: int = 10) -> bool:
        # TODO: refine leakage detection (what should be considered maximum leakage length?)
        if not isinstance(text, str):
            return True
        if not text.strip():
            return True
        if HierarchicalChunkingManager.word_count(text) < min_words:
            return True
        first = text.strip().split("\n", 1)[0]
        # dot-leader TOC line
        if re.search(r"\.{5,}\s*\d+\s*$", first):
            return True
        return False

    @staticmethod
    def get_next_peer(section: ChunkingTOCSection, siblings: List[ChunkingTOCSection]) -> Optional[ChunkingTOCSection]:
        for idx, s in enumerate(siblings):
            if s.id == section.id and idx + 1 < len(siblings):
                return siblings[idx + 1]
        return None

    def _collect_deeper_descendant_headers(
            self, section: "ChunkingTOCSection",  # noqa: F821
    ) -> Dict[str, List]:
        """Walk ricorsivo dei figli; titoli per ogni discendente con level > header_levels."""
        headers: Dict[str, List] = {}
        for child in section.children or []:
            if child.level > self.header_levels:
                headers.setdefault(f"Header {child.level}", []).append(child.title)
            for k, v in self._collect_deeper_descendant_headers(child).items():
                headers.setdefault(k, []).extend(v)
        return headers

    def _first_chunk_level_descendant(
            self, section: "ChunkingTOCSection",  # noqa: F821
    ) -> Optional["ChunkingTOCSection"]:  # noqa: F821
        """Primo discendente (escluso self) con level <= header_levels in pre-order.

        Nessun filtro su `is_excluded_section`: serve come *bound testuale*,
        quindi anche una sezione esclusa (es. References) costituisce comunque
        il confine destro di estrazione per la sezione precedente.
        """
        for child in section.children or []:
            wrapped = ChunkingTOCSection(  # noqa: F821
                **child.model_dump(),
                parent_id=section.id,
                has_children=(len(child.children) > 0),
            )
            if wrapped.level <= self.header_levels:
                return wrapped
            found = self._first_chunk_level_descendant(wrapped)
            if found is not None:
                return found
        return None

    def _first_chunk_level_in_subtree(
            self, section: "ChunkingTOCSection",  # noqa: F821
    ) -> Optional["ChunkingTOCSection"]:  # noqa: F821
        """Prima sezione (inclusa self) a chunk-level nel sotto-albero radicato in `section`."""
        if section.level <= self.header_levels:
            return section
        return self._first_chunk_level_descendant(section)

    def collect_chunks_from_section(
            self,
            section: "ChunkingTOCSection",  # noqa: F821
            chunks: List[Document],
            parent_headers: Dict[str, List],
            siblings: List["ChunkingTOCSection"],  # noqa: F821
            body_start: int,
            min_words: int,
            next_chunk_section: Optional["ChunkingTOCSection"] = None,  # noqa: F821
    ) -> None:
        current_headers = {k: v.copy() for k, v in parent_headers.items()}
        current_headers[f"Header {section.level}"] = [section.title]
        current_headers = {
            k: v for k, v in current_headers.items()
            if int(k.split()[-1]) <= section.level
        }

        if section.level <= self.header_levels:
            chunk_headers = {k: v.copy() for k, v in current_headers.items()}
            for k, v in self._collect_deeper_descendant_headers(section).items():
                chunk_headers.setdefault(k, []).extend(v)

            nxt = self._first_chunk_level_descendant(section) or next_chunk_section

            raw_text = self.extract_section_text(
                section=section,
                next_section=nxt,
                first_child=None,
                body_start=body_start,
                window=1,
            )
            is_empty = self.looks_like_leakage(raw_text, min_words=min_words)

            if not is_empty:
                breadcrumb = self._build_breadcrumb(current_headers)
                page_content = f"{breadcrumb}\n\n{raw_text}" if breadcrumb else raw_text

                chunk_metadata = TOCChunkMetadata(  # noqa: F821
                    chunk_id=f"{self.doc_id}:{section.id}:0",
                    chunk_idx=len(chunks),
                    filename=self.doc_id,
                    section_id=section.id,
                    parent_section_id=section.parent_id,
                    section_title=section.title,
                    section_level=section.level,
                    section_type=section.type,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    is_empty=False,
                    embed=True,
                    n_tokens=0,
                    headers=chunk_headers,
                )
                chunks.append(Document(
                    page_content=page_content,
                    metadata=chunk_metadata.model_dump(mode="json"),
                ))

        section_children = [
            ChunkingTOCSection(  # noqa: F821
                **child.model_dump(),
                parent_id=section.id,
                has_children=(len(child.children) > 0),
            )
            for child in section.children
        ] if section.children else []

        for i, child in enumerate(section_children):
            child_next: Optional["ChunkingTOCSection"] = None  # noqa: F821
            for j in range(i + 1, len(section_children)):
                candidate = self._first_chunk_level_in_subtree(section_children[j])
                if candidate is not None:
                    child_next = candidate
                    break
            if child_next is None:
                child_next = next_chunk_section

            self.collect_chunks_from_section(
                section=child,
                chunks=chunks,
                parent_headers=current_headers,
                siblings=section_children,
                body_start=body_start,
                min_words=min_words,
                next_chunk_section=child_next,
            )

    @staticmethod
    def _build_breadcrumb(current_headers: Dict[str, List]) -> str:
        if not current_headers:
            return ""
        parts: List[str] = []
        levels = sorted(int(k.split()[-1]) for k in current_headers)
        for level_num in levels:
            titles = current_headers.get(f"Header {level_num}") or []
            if titles:
                parts.append(str(titles[-1]).strip())
        return " > ".join(p for p in parts if p)

    def _build_page_offsets(self, pdf_path: pathlib.Path) -> Dict[int, int]:
        """Approximate md character offset for each PDF page using fitz."""
        import fitz
        offsets: Dict[int, int] = {}
        try:
            pdf = fitz.open(pdf_path.as_posix())
            search_from = 0
            for page_no in range(1, pdf.page_count + 1):
                page: fitz.Page = pdf.load_page(page_no - 1)
                raw_text = page.get_text("text")  # type: ignore[attr-defined]
                anchor = ""
                for line in raw_text.splitlines():
                    line = line.strip()
                    if len(line) >= 40:
                        anchor = " ".join(line.split()[:6])
                        break
                if anchor:
                    pos = self.md_text.find(anchor, search_from)
                    if pos != -1:
                        offsets[page_no] = pos
                        search_from = pos
                        continue
                offsets[page_no] = search_from
            pdf.close()
        except (OSError, RuntimeError):
            self.logger.warning(f"Could not build page offsets for {pdf_path}, falling back to empty anchors")
        return offsets

    def split_text(self, min_words: int = 10) -> List[Document]:
        """Chunk builder (no splitting yet).

        Ogni sezione con `level <= self.header_levels` produce un chunk il cui
        testo va dall'header della sezione all'header della prossima sezione
        di chunk-level in pre-order (o EOF se ultima). Niente piu accumulo
        di estrazioni multiple => niente duplicazione e niente leakage a EOF.
        """
        body_start = self.find_body_start(self.md_text)
        chunks: List[Document] = []

        if self.header_levels <= 0:
            self.logger.info("Starting hierarchical chunking (anchors + TOC boundaries, no split)")
            sections = self.collect_sections(self.toc_tree)
            self.logger.info(f"TOC nodes considered: {len(sections)}")
            ordered_sections = [s for s in sections if s.id]
            for idx, section in enumerate(ordered_sections):
                section_id = section.id
                if self.is_excluded_section(section):
                    continue
                next_sec = ordered_sections[idx + 1] if idx + 1 < len(ordered_sections) else None
                first_child = None
                children = section.children or []
                if children:
                    if next_sec and next_sec.parent_id == section_id:
                        first_child = next_sec

                raw_text = self.extract_section_text(
                    section=section, next_section=next_sec, first_child=first_child,
                    body_start=body_start, window=1,
                )
                empty = self.looks_like_leakage(raw_text, min_words=min_words)
                if empty and not section.has_children:
                    continue
                chunk_metadata = TOCChunkMetadata(  # noqa: F821 (defined in original file)
                    chunk_id=f"{self.doc_id}:{section_id}:0",
                    chunk_idx=idx,
                    filename=self.doc_id,
                    section_id=section_id,
                    parent_section_id=section.parent_id,
                    section_title=section.title,
                    section_level=section.level,
                    section_type=section.type,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    is_empty=empty,
                    embed=not empty,
                    n_tokens=0,
                )
                chunk = Document(page_content="" if empty else raw_text)
                chunk.metadata = chunk_metadata.model_dump(mode="json")
                chunks.append(chunk)
        else:
            self.logger.info(
                f"Starting hierarchical chunking (anchors + TOC boundaries, "
                f"header levels to split on: {self.header_levels})"
            )
            explicit_toc_tree = [
                ChunkingTOCSection(  # noqa: F821
                    **tree_node.model_dump(),
                    has_children=(len(tree_node.children) > 0),
                )
                for tree_node in self.toc_tree
            ]
            for i, top_section in enumerate(explicit_toc_tree):
                if self.is_excluded_section(top_section):
                    continue
                top_next: Optional["ChunkingTOCSection"] = None  # noqa: F821
                for j in range(i + 1, len(explicit_toc_tree)):
                    candidate = self._first_chunk_level_in_subtree(explicit_toc_tree[j])
                    if candidate is not None:
                        top_next = candidate
                        break
                self.collect_chunks_from_section(
                    section=top_section,
                    chunks=chunks,
                    parent_headers={},
                    siblings=explicit_toc_tree,
                    body_start=body_start,
                    min_words=min_words,
                    next_chunk_section=top_next,
                )

        self.logger.info(f"Built {len(chunks)} chunks")
        return chunks