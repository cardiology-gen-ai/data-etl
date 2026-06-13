import pathlib
import re
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, Iterable, List, Optional, Tuple

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, SectionHeaderItem

from managers.toc_extraction.utils import _deduplicate_outline, normalize_toc_levels, _toc_needs_sorting, _safe_sort, \
    _set_terminal_section, _extract_document_title_from_doc, _generate_sequential_id
from managers.toc_extraction.toc_configs import BaseTocConfig
from utils.text_utils import title_overlap_score


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HeadingEntry:
    """One section heading with its inferred level."""
    text: str
    page_no: int
    font_size: float
    font_name: str
    level: int
    is_title: bool = field(default=False)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _font_family(font_name: str) -> str:
    """Strip PDF subset prefix and weight/style suffixes -> root name.

    Examples: ``"AdvTTde03b09e.B"`` -> ``"AdvTTde03b09e"``
    """
    name = re.sub(r"^[A-Z]{6}\+", "", font_name)
    name = re.split(r"[.,\-+]", name)[0]
    return name.strip()


# ---------------------------------------------------------------------------
# Heading filtering
# ---------------------------------------------------------------------------

def _is_excluded(text: str, toc_config: BaseTocConfig) -> bool:
    """Return True if this heading should be excluded from the TOC."""
    t = text.strip()
    tl = t.lower()
    if len(tl) <= 2:
        return True
    if toc_config.excluded_title_keywords and tl in set(toc_config.excluded_title_keywords):
        return True
    if toc_config.bad_section_title_starts and tl.startswith(toc_config.bad_section_title_starts):
        return True
    if toc_config.bad_section_title_ends and tl.endswith(toc_config.bad_section_title_ends):
        return True
    if toc_config.bad_title_patterns and any(p.search(tl) for p in toc_config.bad_title_patterns):
        return True
    return False


def _is_document_title(text: str, doc_title: Optional[str], threshold: float = 0.6) -> bool:
    if not doc_title:
        return False
    return title_overlap_score(text, doc_title) >= threshold


# ---------------------------------------------------------------------------
# fitz font extraction
# ---------------------------------------------------------------------------

def _extract_line_info(pdf_path: str) -> Dict[str, Tuple[float, str]]:
    """Return ``{line_text: (avg_font_size, font_name)}`` preserving original casing.

    When the same text appears on multiple pages, the largest-size entry is kept.
    """
    info: Dict[str, Tuple[float, str]] = {}
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page: fitz.Page = doc.load_page(page_num)
        for block in page.get_text("dict")["blocks"]:  # type: ignore[attr-defined]
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                avg_size = sum(s["size"] for s in spans) / len(spans)
                font_name = spans[0]["font"]
                if avg_size > info.get(text, (0.0, ""))[0]:
                    info[text] = (avg_size, font_name)
    doc.close()
    return info


# ---------------------------------------------------------------------------
# Heading -> fitz matching
# ---------------------------------------------------------------------------

def _best_match(
    heading_text: str,
    line_info: Dict[str, Tuple[float, str]],
) -> Optional[Tuple[float, str]]:
    """Match heading text to a fitz line: exact -> case-insensitive -> token overlap."""
    if heading_text in line_info:
        return line_info[heading_text]

    heading_lower = heading_text.lower()
    for fitz_text, entry in line_info.items():
        if fitz_text.lower() == heading_lower:
            return entry

    heading_tokens = set(re.findall(r"\w+", heading_lower))
    best_score, best_entry = 0.0, None
    for fitz_text, entry in line_info.items():
        fitz_tokens = set(re.findall(r"\w+", fitz_text.lower()))
        union = len(heading_tokens | fitz_tokens)
        if union == 0:
            continue
        score = len(heading_tokens & fitz_tokens) / union
        if score > 0.6 and score > best_score:
            best_score = score
            best_entry = entry
    return best_entry


# ---------------------------------------------------------------------------
# Level clustering
# ---------------------------------------------------------------------------

def _cluster_fingerprints(
    fingerprints: List[Tuple[float, str]],
    first_appearance: Dict[str, int],
    size_tolerance: float,
) -> Dict[Tuple[float, str], int]:
    """Cluster (size, font_family) pairs into discrete levels.

    Sort key: (-size, first_appearance_ordinal) — larger fonts get lower level
    numbers, first-appearance tiebreaking when sizes are equal.
    """
    unique = list(set((round(s, 1), f) for s, f in fingerprints))
    clusters: List[List[Tuple[float, str]]] = []
    for fp in unique:
        size, family = fp
        placed = False
        for cluster in clusters:
            rep_size, rep_family = cluster[0]
            if rep_family == family and abs(size - rep_size) <= size_tolerance:
                cluster.append(fp)
                placed = True
                break
        if not placed:
            clusters.append([fp])

    clusters.sort(key=lambda c: (-c[0][0], first_appearance.get(c[0][1], 10_000)))

    fp_to_level: Dict[Tuple[float, str], int] = {}
    for level, cluster in enumerate(clusters, start=1):
        for fp in cluster:
            fp_to_level[fp] = level
    return fp_to_level


# ---------------------------------------------------------------------------
# Document iteration helpers
# ---------------------------------------------------------------------------

def _iter_section_headers(doc: DoclingDocument) -> Iterable[SectionHeaderItem]:
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            yield item


def _make_docling_doc(pdf_path: str) -> DoclingDocument:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    return converter.convert(pdf_path).document


# ---------------------------------------------------------------------------
# Unmatched title heuristic (path 3 of title detection)
# ---------------------------------------------------------------------------

def _apply_unmatched_title_heuristic(
    entries: List[HeadingEntry],
) -> Tuple[Optional[str], List[HeadingEntry]]:
    """Remove sole unmatched heading at position 0 treating it as document title.

    Conditions:
    1. First heading has size=0.0 and font_name="unknown" (fitz match failed).
    2. It is the only such heading (multiple unmatched = heuristic unreliable).
    """
    unmatched = [
        (i, e) for i, e in enumerate(entries)
        if e.font_size == 0.0 and e.font_name == "unknown"
    ]
    if len(unmatched) == 1 and unmatched[0][0] == 0:
        _, title_entry = unmatched[0]
        title_entry.is_title = True
        return title_entry.text, [e for i, e in enumerate(entries) if i != 0]
    return None, entries


# ---------------------------------------------------------------------------
# Section id helpers
# ---------------------------------------------------------------------------

def _detect_section_id_in_title(
    entries: List[HeadingEntry], toc_config: BaseTocConfig
) -> bool:
    """Return True if the majority of headings start with a numeric section id."""
    if toc_config.section_id_re is None:
        return False
    sample = entries[:20]
    if not sample:
        return False
    matches = sum(1 for e in sample if toc_config.section_id_re.match(e.text))
    return matches / len(sample) >= 0.5


def _level_from_numeric_id(text: str, toc_config: BaseTocConfig) -> Optional[int]:
    """Return heading level from numeric id prefix, or None if no id found."""
    if toc_config.section_id_re is None:
        return None
    m = toc_config.section_id_re.match(text)
    if not m:
        return None
    return m.group(1).count(".") + 1


# ---------------------------------------------------------------------------
# HeadingEntry -> TOCSection conversion
# ---------------------------------------------------------------------------

def _entries_to_toc_sections(
    entries: List[HeadingEntry],
    section_id_in_title: bool,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> List:
    """Convert HeadingEntry list to TOCSection list.

    When ``section_id_in_title`` is True, level is derived deterministically
    from the numeric id depth and headings without an id are skipped.
    Otherwise, font-size cluster levels are used and sequential ids generated.
    """
    from managers.toc_extraction.table_of_contents_manager import TOCSection, TOCSectionType

    sections = []
    level_counters: Dict[int, int] = {}

    for entry in entries:
        if section_id_in_title:
            level = _level_from_numeric_id(entry.text, toc_config)
            if level is None:
                logger.debug(f"Skipping non-numbered heading: '{entry.text}'")
                continue
            m = toc_config.section_id_re.match(entry.text)
            section_id = m.group(1) if m else None
            clean_title = m.group(2).strip() if m else entry.text
        else:
            level = entry.level
            section_id = _generate_sequential_id(level, level_counters)
            clean_title = entry.text

        sections.append(TOCSection(
            id=section_id, title=clean_title, level=level,
            page_start=entry.page_no, page_end=entry.page_no,
            type=TOCSectionType.body,
        ))
    return sections


# ---------------------------------------------------------------------------
# Heading level inference (public, also usable standalone)
# ---------------------------------------------------------------------------

def infer_heading_levels(
    pdf_path: str,
    logger: Logger,
    toc_config: BaseTocConfig,
    size_tolerance: float = 0.5,
    docling_doc: Optional[DoclingDocument] = None,
    title_overlap_threshold: float = 0.6,
    infer_title_from_unmatched: bool = True,
    drop_unmatched: bool = False,
) -> List[HeadingEntry]:
    """Infer heading levels for a PDF using Docling + fitz.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    logger : Logger
        Logging utility.
    toc_config : BaseTocConfig
        Drives heading filtering and section-id detection.
    size_tolerance : float
        Font-size tolerance (pt) for same-level grouping. Default ``0.5``.
    docling_doc : DoclingDocument, optional
        Pre-converted document; skips Docling conversion when provided.
    title_overlap_threshold : float
        Minimum Jaccard overlap to treat a heading as the document title.
    infer_title_from_unmatched : bool
        Apply path-3 heuristic for the sole unmatched heading at position 0.
    drop_unmatched : bool
        Drop all remaining unmatched headings after path-3 runs.

    Returns
    -------
    List[HeadingEntry]
        Retained section headings in document order with inferred levels.
    """
    pdf_path = str(pdf_path)

    if docling_doc is None:
        logger.info(f"Converting {pdf_path} with Docling …")
        docling_doc = _make_docling_doc(pdf_path)

    logger.info(f"Extracting font info from {pdf_path} via fitz …")
    line_info = _extract_line_info(pdf_path)

    # title detection (paths 1 + 2 — path 3 below)
    doc_title = _extract_document_title_from_doc(docling_doc, pdf_path, toc_config)
    if doc_title:
        logger.info(f"Document title (path 1/2): '{doc_title[:80]}'")

    # collect headings preserving original casing for fitz matching
    raw_headers: List[Tuple[str, int]] = [
        (item.text, item.prov[0].page_no)
        for item in _iter_section_headers(docling_doc)
        if item.prov
        and not _is_excluded(item.text, toc_config)
        and not _is_document_title(item.text, doc_title, title_overlap_threshold)
    ]

    if not raw_headers:
        logger.error("No SECTION_HEADER items remain after filtering.")
        return []

    matched: List[Optional[Tuple[float, str]]] = [
        _best_match(text, line_info) for text, _ in raw_headers
    ]

    # build fingerprints and first-appearance map
    fingerprints: List[Optional[Tuple[float, str]]] = []
    first_appearance: Dict[str, int] = {}
    for ordinal, (_, fitz_entry) in enumerate(zip(raw_headers, matched)):
        if fitz_entry is None:
            fingerprints.append(None)
            continue
        size, font_name = fitz_entry
        family = _font_family(font_name)
        fingerprints.append((round(size, 1), family))
        if family not in first_appearance:
            first_appearance[family] = ordinal

    valid_fps = [fp for fp in fingerprints if fp is not None]
    if not valid_fps:
        logger.error("Could not match any heading to fitz font info.")
        return []

    fp_to_level = _cluster_fingerprints(valid_fps, first_appearance, size_tolerance)
    fallback_level = max(fp_to_level.values()) + 1

    entries: List[HeadingEntry] = []
    for (text, page_no), fp, fitz_entry in zip(raw_headers, fingerprints, matched):
        if fp is None or fitz_entry is None:
            entries.append(HeadingEntry(
                text=text, page_no=page_no,
                font_size=0.0, font_name="unknown", level=fallback_level,
            ))
        else:
            size, font_name = fitz_entry
            entries.append(HeadingEntry(
                text=text, page_no=page_no,
                font_size=size, font_name=font_name,
                level=fp_to_level.get(fp, fallback_level),
            ))

    # path 3: unmatched title heuristic
    if infer_title_from_unmatched:
        inferred_title, entries = _apply_unmatched_title_heuristic(entries)
        if inferred_title:
            logger.info(f"Document title (path 3): '{inferred_title[:80]}'")

    if drop_unmatched:
        before = len(entries)
        entries = [e for e in entries if not (e.font_size == 0.0 and e.font_name == "unknown")]
        if dropped := before - len(entries):
            logger.info(f"drop_unmatched: removed {dropped} heading(s).")

    return entries


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_toc_from_docling(
    filepath: pathlib.Path,
    cache_dir: pathlib.Path,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> List:
    """Build TOC from SECTION_HEADER items in the DoclingDocument cache.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the source PDF (used for fitz font extraction).
    cache_dir : pathlib.Path
        Directory containing the ``<stem>.json`` DoclingDocument cache.
    toc_config : BaseTocConfig
        Configuration driving filtering, clustering and id detection.
    logger : Logger
        Logging utility.

    Returns
    -------
    List[TOCSection]
        Extracted and normalised TOC sections.

    Raises
    ------
    FileNotFoundError
        If the DoclingDocument cache does not exist.
    ValueError
        If no valid sections are produced after filtering.
    """
    cache_path = cache_dir / (filepath.stem + ".json")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"DoclingDocument cache not found at {cache_path}. Run MarkdownConverter first."
        )
    logger.info(f"Loading DoclingDocument from {cache_path}")
    docling_doc = DoclingDocument.model_validate_json(cache_path.read_text(encoding="utf-8"))

    entries = infer_heading_levels(
        pdf_path=str(filepath), docling_doc=docling_doc,
        logger=logger, toc_config=toc_config,
        infer_title_from_unmatched=True, drop_unmatched=True,
    )
    if not entries:
        raise ValueError("infer_heading_levels returned no headings after filtering")

    section_id_in_title = (
        _detect_section_id_in_title(entries, toc_config)
        or toc_config.section_id_in_title
    )
    logger.info(f"section_id_in_title (dynamic): {section_id_in_title}")

    sections = _entries_to_toc_sections(entries, section_id_in_title, toc_config, logger)
    if not sections:
        raise ValueError("No valid sections — all headings lacked numeric ids")

    sections = _deduplicate_outline(sections)
    if not section_id_in_title:
        sections = normalize_toc_levels(sections)
    if _toc_needs_sorting(sections, logger):
        sections = _safe_sort(sections, logger)
    return _set_terminal_section(sections, toc_config, logger)