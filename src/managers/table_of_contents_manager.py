"""
Guideline TOC Extraction Pipeline

Extracts a Table of Contents from a PDF guideline using:
1) PDF outline (preferred, when present and not corrupted)
2) Fallback textual TOC parsed from pages containing "Table of Contents"

Outputs:
- Flat TOC with page ranges (page_start/page_end)
  Note: Some guidelines have page ranges from the outline starting from 1.
  Others have pages starting from a higher number, because they are extracted
  from the guideline textual TOC and don't necessarily follow that schema.
- Hierarchical TOC tree suitable for Graph RAG
"""

import fitz
import re
import json
import logging
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Section IDs like "1", "1.1", "3.7.4.2"
SECTION_ID_RE = re.compile(r"^(\d+(?:\.\d+)*)")

CONTROL_CHAR_RE = re.compile(
    "["
    "\u0000-\u0008"
    "\u000b-\u000c"
    "\u000e-\u001f"
    "\u007f-\u009f"
    "]"
)

TRAILING_NOTE_MARK_RE = re.compile(
    r"(?P<body>.*?[A-Za-z)][A-Za-z),;: ]*)(?P<note>\d{1,2})$"
)


def sanitize_toc_title(title: str) -> str:
    """
    Apply conservative title normalization only.

    This removes PDF/OCR artefacts that hurt heading matching while
    preserving clinically meaningful wording and numbers.
    """
    title = title or ""
    title = title.replace("\xad", "")
    title = CONTROL_CHAR_RE.sub("", title)
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"\s+", " ", title).strip(" .")

    # Remove only very likely attached footnote markers, e.g.
    # "monitoring2". Do not remove numbers after a space, so titles
    # such as "Type 2 diabetes" remain unchanged.
    match = TRAILING_NOTE_MARK_RE.match(title)

    if match:
        body = match.group("body").rstrip()

        if (
            body
            and not re.search(r"\b(type|group|class|stage|phase|part|chapter|table|figure|scenario)\s*$", body, re.IGNORECASE)
        ):
            title = body.strip(" .")

    return title



# Pydantic models

class TOCSection(BaseModel):
    id: Optional[str]
    printed_id: Optional[str] = None
    duplicate_id_index: Optional[int] = None
    toc_original_section_id: Optional[str] = None
    toc_section_id_corrected_from_body: bool = False
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    type: str = "body"
    children: List["TOCSection"] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError(f"Invalid page range: {self.page_start}–{self.page_end}")
        return self


class TOCMetadata(BaseModel):
    doc_id: str
    toc_source: str
    n_pages: int
    flat_toc: List[TOCSection]
    toc_tree: List[TOCSection]
    page_normalization: Dict[str, Any] = Field(default_factory=dict)


TOCSection.model_rebuild()


# TOC extractor
class GuidelineTOCExtractor:
    HEADING_TOP_Y_THRESHOLD = 110

    # Heuristics for filtering bad textual TOC lines
    TOC_LINE = re.compile(
        r"""
        ^\s*
        (?:(\d+(?:\.\d+)*))?     # optional numeric id: 1, 1.1, 3.7.4.2
        \s*
        (.*?)                    # title text (lazy)
        [\.\s·]*                 # dot leaders / spaces / middots
        (\d{2,6})                # page number (2-6 digits)
        \s*$
        """,
        re.VERBOSE,
    )

    BAD_PAGE_WORDS = (
        "downloaded from",
        "doi",
        "oxford",
        "rights reserved",
        "the authors",
        "permissions",
        "copyright",
        "online publish",
        "eur heart j",
        "esc guidelines",
    )
    # TODO: possibly remove this filter
    PROSE_WORDS = (
        "should",
        "may",
        "demonstrated",
        "guidelines summarize",
        "guidelines and recommendations",
        "committee",
        "task force",
        "writing committee",
        "writing group",
        "clinical practice",
        "management of",
        "impact on outcome",
        "according to esc",
    )

    BAD_TITLE_PATTERNS = (
        "guidelines summarize and evaluate",
        "a great number of guidelines",
        "the level of evidence and the strength",
        "are available on the esc website",
        "the task force received its entire financial support",
        "are now more evident and clear",
        "have been documented",
        "have become available",
        "have been proposed",
    )

    EDGE_LINE_COUNT_FOR_PAGE_LABELS = 8
    MIN_PAGE_NORMALIZATION_COVERAGE = 0.85
    MIN_BODY_OFFSET_SUPPORT = 5
    MIN_BODY_OFFSET_DOMINANCE = 0.80
    MIN_BODY_OFFSET_MATCH_COVERAGE = 0.50
    MIN_BODY_OFFSET_TITLE_SIMILARITY = 0.88
    TEXTUAL_ID_IN_TITLE_RE = re.compile(
        r"^\s*(\d+(?:\.\d+)*)(?:\.|\s+)(.+?)\s*$"
    )
    TEXTUAL_ENTRY_START_RE = re.compile(
        r"^\s*(\d+(?:\.\d+)*)(?:\.|\s+)(.*?)\s*$"
    )
    TEXTUAL_TRAILING_PAGE_RE = re.compile(
        r"^(?P<title>.*?)(?:[\.\s·]{2,}|\s+)(?P<page>\d{1,6})\s*$"
    )
    BODY_HEADING_RE = re.compile(
        r"^\s*(?P<id>\d+(?:\.\d+)*)(?:\.|\s+)(?P<title>[^\d].*?)\s*$"
    )
    BODY_HEADING_ID_ONLY_RE = re.compile(r"^\s*(?P<id>\d+(?:\.\d+)*)\.?\s*$")
    NON_STRUCTURAL_UNNUMBERED_TITLE_RE = re.compile(
        r"""
        ^(
            recommendations?\s+
            (for|regarding|on|to|about)\b
            |recommendations?\b
            |tables?\s+of\s+recommendations?\b
            |summary\s+of\s+recommendations?\b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self, pdf_path: str, doc_id: str):
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.doc = fitz.open(pdf_path)
        self.n_pages = self.doc.page_count
        self._body_heading_index: Optional[List[Dict[str, Any]]] = None
        self._printed_toc_page_range: Optional[tuple[int, int]] = None

        logger.info("Initialized TOC extractor for %s (%d pages)", pdf_path, self.n_pages)

    
    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()

    
    def _toc_needs_sorting(self, sections: List[Dict[str, Any]]) -> bool:
        """
        Returns True if the TOC order is clearly structurally broken
        (e.g. first entry not level 1, or a child level appears before its parent).
        """
        if not sections:
            return False

        # 1) TOC should start with level 1
        if sections[0]["level"] != 1:
            logger.warning("TOC sanity: first entry is not level 1")
            return True

        # 2) No level >1 should appear before its parent level
        seen_levels = {1: True}
        for s in sections:
            lvl = s["level"]
            if lvl == 1:
                seen_levels[1] = True
                continue

            parent_lvl = lvl - 1
            if parent_lvl not in seen_levels:
                logger.warning(
                    "TOC sanity: found level %d without parent (title='%s')",
                    lvl,
                    s.get("title", ""),
                )
                return True

            seen_levels[lvl] = True

        return False

    def _safe_sort(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Minimal safe sort: only sort by page_start (do not reorder by level),
        to preserve parent-before-child order on the same page.
        """
        logger.info("Applying safe fallback TOC sorting")
        sections.sort(key=lambda s: s["page_start"])
        return sections

    def _structural_level(self, sid: Optional[str], fallback_level: int) -> int:
        if sid:
            return sid.count(".") + 1
        return int(fallback_level)

    #   PDF outline mode (preferred)
    def extract_pdf_outline(self) -> List[Dict[str, Any]]:
        raw = self.doc.get_toc(simple=False)  # direct outline, if present

        if not raw:
            return []

        sections: List[Dict[str, Any]] = []

        for level, title, page, *_ in raw:
            title = sanitize_toc_title(title)

            sid = None
            clean_title = title
            m = SECTION_ID_RE.match(title)

            if m:
                cid = m.group(1)
                # ignore pure year-like ids (e.g. 2023) as section ids
                if not (cid.isdigit() and 1900 <= int(cid) <= 2100):
                    sid = cid
                    clean_title = title[len(cid):].strip(" .")

            sections.append(
                {
                    "id": sid,
                    "printed_id": sid,
                    "title": clean_title,
                    "level": self._structural_level(sid, int(level)),
                    "page_start": int(page),
                    "page_end": None,
                    "type": "body",
                }
            )

        sections = self._deduplicate_outline(sections)
        return sections

    def outline_is_reliable(self, sections: List[Dict[str, Any]]) -> tuple[bool, str]:
        if not sections:
            return False, "missing outline"

        # Heuristic check for broken outlines
        bad_count = sum(
            1
            for s in sections
            if (s["title"] or "").lower().startswith(("ehab", "tblfn", "|", "ehz", "ehy", "eh"))
        )
        if bad_count / max(1, len(sections)) > 0.3:
            return False, f"{bad_count}/{len(sections)} suspicious entries"

        return True, "accepted"

    def extract_raw_toc(self) -> List[Dict[str, Any]]:
        sections = self.extract_pdf_outline()
        reliable, reason = self.outline_is_reliable(sections)
        if not reliable:
            raise ValueError(reason)

        # Only apply sorting if the structure is clearly broken
        if self._toc_needs_sorting(sections):
            sections = self._safe_sort(sections)

        # Force 'References' / 'Bibliography' to be terminal TOC boundary
        for i, s in enumerate(sections):
            title = (s.get("title") or "").lower()
            if title.startswith(("reference", "bibliograph")):
                logger.info("Truncating outline after References section")
                sections = sections[: i + 1]
                break

        return sections

    def _deduplicate_outline(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Exact same uniqueness rule for both outline & fallback."""
        seen = set()
        out = []
        for s in sections:
            key = (s["level"], s["id"], s["title"], s["page_start"])
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _recover_textual_id_from_title(
        self,
        sid: Optional[str],
        title: str,
    ) -> tuple[Optional[str], str]:
        """
        Some textual TOCs put the section number inside the captured title,
        e.g. "5.7 Exercise recommendations ...". Recover that as the real id
        so children attach to the correct parent.
        """
        if sid:
            return sid, title

        match = self.TEXTUAL_ID_IN_TITLE_RE.match(title)
        if not match:
            return sid, title

        recovered_id, recovered_title = match.groups()
        if recovered_id.isdigit() and 1900 <= int(recovered_id) <= 2100:
            return sid, title

        return recovered_id, sanitize_toc_title(recovered_title)

    def _is_non_structural_unnumbered_title(self, title: str) -> bool:
        """
        Recommendation/table-of-recommendation rows are useful content, but they
        are not stable hierarchy nodes. Keeping them as level-1 TOC entries can
        break the parent stack for following numbered sections.
        """
        return bool(self.NON_STRUCTURAL_UNNUMBERED_TITLE_RE.match(title.strip()))

    def _normalize_textual_title_key(self, title: str) -> str:
        title = (title or "").lower().replace("\xad", "")
        title = re.sub(r"[-–—]", " ", title)
        title = re.sub(r"[^a-z0-9]+", " ", title)
        return re.sub(r"\s+", " ", title).strip()

    def _is_plausible_textual_section_id(self, sid: Optional[str]) -> bool:
        if not sid:
            return False
        if sid.isdigit() and 1900 <= int(sid) <= 2100:
            return False
        return bool(re.fullmatch(r"\d+(?:\.\d+)*", sid))

    def _split_textual_title_page(
        self,
        text: str,
    ) -> tuple[str, Optional[int]]:
        match = self.TEXTUAL_TRAILING_PAGE_RE.match((text or "").strip())
        if not match:
            return (text or "").strip(" .·"), None

        page = int(match.group("page"))
        if page < 1 or page > 20000:
            return (text or "").strip(" .·"), None

        title = match.group("title").strip(" .·")
        if not title:
            return (text or "").strip(" .·"), None

        return title, page

    def _join_wrapped_textual_title(self, lines: List[str]) -> str:
        joined = ""
        for raw in lines:
            line = re.sub(r"\s+", " ", (raw or "").strip())
            if not line:
                continue
            if joined.endswith("-") and line[:1].islower():
                joined = joined[:-1] + line
            elif joined:
                joined += " " + line
            else:
                joined = line
        return re.sub(r"\s+", " ", joined).strip()

    def _parse_textual_toc_records(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        def finalize_current() -> None:
            nonlocal current
            if not current:
                return

            title_text = self._join_wrapped_textual_title(current["title_lines"])
            title, page = self._split_textual_title_page(title_text)
            if title:
                current["title"] = title
                current["page"] = page
                records.append(current)
            current = None

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue

            low = line.lower()
            if any(p in low for p in self.BAD_PAGE_WORDS):
                continue
            if len(line) > 300:
                continue

            start = self.TEXTUAL_ENTRY_START_RE.match(line)
            if start and self._is_plausible_textual_section_id(start.group(1)):
                rest = start.group(2).strip()
                if rest and not re.search(r"[A-Za-z]", rest):
                    if current:
                        current["title_lines"].append(line)
                    continue

                finalize_current()
                current = {
                    "id": start.group(1),
                    "title_lines": [rest] if rest else [],
                    "raw_lines": [line],
                    "_source_order": len(records),
                }
                _, page = self._split_textual_title_page(rest)
                if page is not None:
                    finalize_current()
                continue

            if current:
                current["title_lines"].append(line)
                current["raw_lines"].append(line)
                title_text = self._join_wrapped_textual_title(current["title_lines"])
                _, page = self._split_textual_title_page(title_text)
                if page is not None:
                    finalize_current()
                continue

            title, page = self._split_textual_title_page(line)
            if page is not None and re.search(r"[A-Za-z]", title):
                records.append(
                    {
                        "id": None,
                        "title": title,
                        "page": page,
                        "title_lines": [title],
                        "raw_lines": [line],
                        "_source_order": len(records),
                    }
                )

        finalize_current()
        return records

    def _textual_record_to_section(
        self,
        record: Dict[str, Any],
        *,
        recovery: bool = False,
    ) -> Optional[Dict[str, Any]]:
        sid = record.get("id")
        title = (record.get("title") or "").strip(". ").strip()
        page = record.get("page")

        if page is None:
            logger.debug("Skipping textual TOC entry without page: %s", record.get("raw_lines"))
            return None
        if page < 1 or page > 20000:
            return None

        sid, title = self._recover_textual_id_from_title(sid, title)
        if sid and not self._is_plausible_textual_section_id(sid):
            return None

        if len(title) < 3:
            return None
        if len(title) > (180 if recovery else 220):
            return None
        if title.count(" ") > (28 if recovery else 35):
            return None
        if title.count(".") > 3:
            return None

        low_title = title.lower()
        if any(p in low_title for p in self.BAD_TITLE_PATTERNS):
            return None
        if not sid and any(w in low_title for w in self.PROSE_WORDS):
            return None

        return {
            "id": sid,
            "printed_id": sid,
            "title": title,
            "level": self._structural_level(sid, 1),
            "page_start": page,
            "page_end": page,
            "type": "body",
            "_source_order": record.get("_source_order", 0),
        }

    def _deduplicate_textual_sections(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        out: List[Dict[str, Any]] = []

        for section in sections:
            sid = section.get("id")
            if not sid:
                out.append(section)
                continue

            existing = by_id.get(sid)
            if not existing:
                by_id[sid] = section
                out.append(section)
                continue

            same_title = (
                self._normalize_textual_title_key(existing.get("title", ""))
                == self._normalize_textual_title_key(section.get("title", ""))
            )
            compatible_page = abs(
                int(existing.get("page_start") or 0)
                - int(section.get("page_start") or 0)
            ) <= 1
            if same_title and compatible_page:
                logger.info("Collapsed duplicate textual TOC entry for section %s", sid)
                continue

            logger.warning(
                "Conflicting duplicate textual TOC id %s kept for validation | "
                "first='%s' page=%s | second='%s' page=%s",
                sid,
                existing.get("title"),
                existing.get("page_start"),
                section.get("title"),
                section.get("page_start"),
            )
            out.append(section)

        return out

    def _recover_missing_textual_parents(
        self,
        sections: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        present = {
            section.get("id")
            for section in sections
            if section.get("id")
        }
        needed = sorted(
            {
                sid.rsplit(".", 1)[0]
                for sid in present
                if "." in sid and sid.rsplit(".", 1)[0] not in present
            },
            key=lambda sid: [int(part) for part in sid.split(".")],
        )
        if not needed:
            return sections

        records_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            sid = record.get("id")
            if sid:
                records_by_id.setdefault(sid, []).append(record)

        recovered: List[Dict[str, Any]] = []
        for parent_id in needed:
            candidates = records_by_id.get(parent_id) or []
            for record in candidates:
                section = self._textual_record_to_section(record, recovery=True)
                if section and section.get("title"):
                    recovered.append(section)
                    present.add(parent_id)
                    logger.info(
                        "Recovered missing textual TOC parent %s from parsed records",
                        parent_id,
                    )
                    break
            else:
                logger.warning(
                    "Textual TOC missing direct parent %s; preserving children without synthetic parent",
                    parent_id,
                )

        if not recovered:
            return sections

        combined = sections + recovered
        combined.sort(key=lambda section: int(section.get("_source_order", 10**9)))
        return combined

    def _validate_textual_sections(self, sections: List[Dict[str, Any]]) -> None:
        malformed = []
        missing_parent = []
        bad_level = []
        empty_title = []
        duplicate_ids = [
            sid
            for sid, count in Counter(
                section.get("id")
                for section in sections
                if section.get("id")
            ).items()
            if count > 1
        ]

        ids = {
            section.get("id")
            for section in sections
            if section.get("id")
        }
        previous_order = -1
        order_errors = 0

        for section in sections:
            sid = section.get("id")
            if not section.get("title"):
                empty_title.append(sid or section.get("title"))
            if sid and not self._is_plausible_textual_section_id(sid):
                malformed.append(sid)
            if sid:
                expected_level = sid.count(".") + 1
                if section.get("level") != expected_level:
                    bad_level.append((sid, section.get("level"), expected_level))
                if "." in sid:
                    parent = sid.rsplit(".", 1)[0]
                    if parent not in ids:
                        missing_parent.append((sid, parent))

            source_order = int(section.get("_source_order", 0))
            if source_order < previous_order:
                order_errors += 1
            previous_order = source_order

        if duplicate_ids:
            logger.warning("Textual TOC duplicate ids remain: %s", ", ".join(duplicate_ids))
        if malformed:
            logger.warning("Textual TOC malformed ids: %s", ", ".join(malformed[:20]))
        if missing_parent:
            logger.warning("Textual TOC missing parents: %s", missing_parent[:20])
        if bad_level:
            logger.warning("Textual TOC level mismatches: %s", bad_level[:20])
        if empty_title:
            logger.warning("Textual TOC empty titles: %s", empty_title[:20])
        if order_errors:
            logger.warning("Textual TOC source-order inversions detected: %d", order_errors)

    def _section_parent_id(self, sid: Optional[str]) -> Optional[str]:
        if not sid or "." not in sid:
            return None
        return sid.rsplit(".", 1)[0]

    def _normalize_heading_title(self, title: str) -> str:
        title = re.sub(r"[*_`#\[\]()>]+", " ", title or "")
        title = re.sub(r"[^0-9a-zA-Z]+", " ", title.lower())
        return re.sub(r"\s+", " ", title).strip()

    def _title_similarity(self, left: str, right: str) -> float:
        left_norm = self._normalize_heading_title(left)
        right_norm = self._normalize_heading_title(right)
        if not left_norm or not right_norm:
            return 0.0
        if left_norm == right_norm:
            return 1.0

        left_tokens = set(left_norm.split())
        right_tokens = set(right_norm.split())
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
        return max(overlap, ratio)

    def _is_year_like_id(self, sid: str) -> bool:
        return sid.isdigit() and 1900 <= int(sid) <= 2100

    def _is_likely_body_heading_title(self, title: str) -> bool:
        title = (title or "").strip()
        if len(title) < 3 or len(title) > 160:
            return False
        if not re.search(r"[A-Za-z]", title):
            return False
        if title.startswith("|") or title.endswith("|"):
            return False

        normalized = self._normalize_heading_title(title)
        if not normalized:
            return False
        if normalized in {"i", "ii", "iii", "iv", "a", "b", "c", "class", "level"}:
            return False
        if normalized.startswith(("table ", "figure ", "continued ", "recommendation table ")):
            return False
        if any(word in normalized for word in self.BAD_PAGE_WORDS):
            return False
        if re.fullmatch(r"(class|level)\s+[abc]|[ivx]+", normalized):
            return False
        if re.fullmatch(r"\d+(?:\s+\d+)*", normalized):
            return False
        return True

    def _body_heading_from_lines(
        self,
        lines: List[str],
        idx: int,
    ) -> Optional[tuple[str, str, str]]:
        raw = lines[idx].rstrip()
        stripped = raw.strip()
        if not stripped or stripped.startswith("|"):
            return None

        match = self.BODY_HEADING_RE.match(stripped)
        if match:
            sid = match.group("id")
            title = match.group("title").strip(" .")
            if self._is_year_like_id(sid) or not self._is_likely_body_heading_title(title):
                return None
            return sid, title, stripped

        id_only = self.BODY_HEADING_ID_ONLY_RE.match(stripped)
        if not id_only:
            return None

        sid = id_only.group("id")
        if self._is_year_like_id(sid):
            return None

        for next_raw in lines[idx + 1 : idx + 4]:
            next_line = next_raw.strip()
            if not next_line:
                continue
            if self.BODY_HEADING_ID_ONLY_RE.match(next_line):
                return None
            if not self._is_likely_body_heading_title(next_line):
                return None
            return sid, next_line.strip(" ."), f"{stripped} {next_line.strip()}"

        return None

    def _get_body_heading_index(self) -> List[Dict[str, Any]]:
        if self._body_heading_index is not None:
            return self._body_heading_index

        headings = self.extract_body_heading_candidates()
        self._body_heading_index = headings
        return headings

    def extract_body_heading_candidates(self) -> List[Dict[str, Any]]:
        """
        Extract numbered headings from the document body.

        When the textual-TOC strategy is active, all pages that were
        identified as printed-TOC pages are excluded. This prevents erroneous
        or collapsed numbers from the printed TOC from competing with the
        canonical headings found later in the body.
        """
        body_heading_candidates: List[Dict[str, Any]] = []
        source_order = 0

        toc_start: Optional[int] = None
        toc_end: Optional[int] = None
        if self._printed_toc_page_range is not None:
            toc_start, toc_end = self._printed_toc_page_range

        for page_idx in range(self.n_pages):
            if (
                toc_start is not None
                and toc_end is not None
                and toc_start <= page_idx <= toc_end
            ):
                continue

            page_text = self.doc.load_page(page_idx).get_text("text")
            compact = page_text.lower().replace(" ", "")

            # Retain this guard for callers that request body headings before
            # textual TOC extraction has established the complete page range.
            if "tableofcontents" in compact or (page_idx < 10 and "contents" in compact):
                continue

            lines = page_text.splitlines()
            for line_idx in range(len(lines)):
                parsed = self._body_heading_from_lines(lines, line_idx)
                if not parsed:
                    continue

                sid, title, raw = parsed
                body_heading_candidates.append(
                    {
                        "id": sid,
                        "title": sanitize_toc_title(title),
                        "page_index": page_idx,
                        "source_order": source_order,
                        "raw": raw,
                    }
                )
                source_order += 1

        return body_heading_candidates

    def _ids_hierarchically_compatible(
        self,
        toc_id: Optional[str],
        candidate_id: str,
        conflict_id: str,
    ) -> bool:
        if not toc_id:
            return False
        if candidate_id == toc_id:
            return True
        if candidate_id.startswith(f"{toc_id}."):
            return True
        if self._section_parent_id(candidate_id) == toc_id:
            return True
        if self._section_parent_id(candidate_id) == self._section_parent_id(toc_id):
            return True
        if candidate_id.startswith(f"{conflict_id}."):
            return True
        if self._section_parent_id(candidate_id) == conflict_id:
            return True
        return False

    def _page_proximity_bonus(self, section: Dict[str, Any], heading: Dict[str, Any]) -> float:
        page = section.get("page_start")
        if not isinstance(page, int) or not (1 <= page <= self.n_pages):
            return 0.0

        distance = abs((page - 1) - int(heading["page_index"]))
        if distance == 0:
            return 0.05
        if distance <= 2:
            return 0.03
        return 0.0

    def _best_body_heading_match(
        self,
        section: Dict[str, Any],
        conflict_id: str,
        min_similarity: float = 0.90,
    ) -> Optional[Dict[str, Any]]:
        toc_id = section.get("id")
        candidates: List[tuple[float, int, Dict[str, Any]]] = []

        for heading in self._get_body_heading_index():
            candidate_id = heading["id"]
            if not self._ids_hierarchically_compatible(toc_id, candidate_id, conflict_id):
                continue

            similarity = self._title_similarity(section.get("title", ""), heading["title"])
            if similarity < min_similarity:
                continue

            score = similarity + self._page_proximity_bonus(section, heading)
            candidates.append((score, int(heading["source_order"]), heading))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1]))
        best_score, _, best = candidates[0]
        if len(candidates) > 1 and best_score - candidates[1][0] < 0.03:
            return None
        return best

    def _body_heading_matches_for_section(
        self,
        section: Dict[str, Any],
        conflict_id: str,
        min_similarity: float = 0.72,
        max_candidates: int = 16,
    ) -> List[Dict[str, Any]]:
        """Return strong ordered body-heading candidates for one TOC record."""
        toc_id = section.get("id")
        matches: List[Dict[str, Any]] = []
        normalized_title = self._normalize_heading_title(section.get("title", ""))

        for heading in self._get_body_heading_index():
            candidate_id = heading["id"]
            if not self._ids_hierarchically_compatible(toc_id, candidate_id, conflict_id):
                continue
            similarity = self._title_similarity(section.get("title", ""), heading["title"])
            if similarity < min_similarity:
                continue

            exact_bonus = 0.08 if normalized_title == self._normalize_heading_title(heading["title"]) else 0.0
            score = similarity + exact_bonus + self._page_proximity_bonus(section, heading)
            matches.append({**heading, "match_score": score, "title_similarity": similarity})

        matches.sort(key=lambda item: (-float(item["match_score"]), int(item["source_order"])))
        return matches[:max_candidates]

    def _align_conflict_window_to_body(
        self,
        sections: List[Dict[str, Any]],
        affected_indices: List[int],
        conflict_id: str,
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """Monotonically align a local TOC sequence with body headings.

        A small beam search enforces one-to-one matches and source order. This
        handles collapsed printed numbering where correcting one duplicated ID
        also requires shifting adjacent section IDs.
        """
        candidate_lists = [
            self._body_heading_matches_for_section(sections[idx], conflict_id)
            for idx in affected_indices
        ]
        if any(not candidates for candidates in candidate_lists):
            return None

        # score, last source order, used body ids, mapping
        beam: List[tuple[float, int, frozenset[str], Dict[int, Dict[str, Any]]]] = [
            (0.0, -1, frozenset(), {})
        ]
        for idx, candidates in zip(affected_indices, candidate_lists):
            next_beam: List[tuple[float, int, frozenset[str], Dict[int, Dict[str, Any]]]] = []
            for score, last_order, used_ids, mapping in beam:
                for candidate in candidates:
                    source_order = int(candidate["source_order"])
                    candidate_id = candidate["id"]
                    if source_order <= last_order or candidate_id in used_ids:
                        continue
                    next_mapping = dict(mapping)
                    next_mapping[idx] = candidate
                    next_beam.append((
                        score + float(candidate["match_score"]),
                        source_order,
                        used_ids | {candidate_id},
                        next_mapping,
                    ))

            if not next_beam:
                return None
            next_beam.sort(key=lambda item: (-item[0], item[1]))
            beam = next_beam[:64]

        best = beam[0]
        if len(beam) > 1 and best[0] - beam[1][0] < 0.08:
            best_ids = [best[3][idx]["id"] for idx in affected_indices]
            second_ids = [beam[1][3][idx]["id"] for idx in affected_indices]
            if best_ids != second_ids:
                return None
        return best[3]

    def _proposed_ids_have_plausible_structure(
        self,
        sections: List[Dict[str, Any]],
        proposed_ids: Dict[int, str],
    ) -> bool:
        final_ids = {
            proposed_ids.get(idx, section.get("id"))
            for idx, section in enumerate(sections)
            if proposed_ids.get(idx, section.get("id"))
        }
        for idx, corrected_id in proposed_ids.items():
            if not self._is_plausible_textual_section_id(corrected_id):
                return False
            parent = self._section_parent_id(corrected_id)
            if parent and parent not in final_ids:
                return False
        return True

    def _conflicting_duplicate_id_groups(
        self,
        sections: List[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        grouped: Dict[str, List[int]] = {}
        for idx, section in enumerate(sections):
            sid = section.get("id")
            if not sid:
                continue
            grouped.setdefault(sid, []).append(idx)

        conflicts: Dict[str, List[int]] = {}
        for sid, indices in grouped.items():
            if len(indices) < 2:
                continue
            titles = {
                self._normalize_heading_title(sections[idx].get("title", ""))
                for idx in indices
            }
            titles.discard("")
            if len(titles) > 1:
                conflicts[sid] = indices
        return conflicts

    def _affected_conflict_window(
        self,
        sections: List[Dict[str, Any]],
        conflict_id: str,
        conflict_indices: List[int],
    ) -> List[int]:
        start = min(conflict_indices)
        end = max(conflict_indices)
        conflict_parent = self._section_parent_id(conflict_id)

        while end + 1 < len(sections):
            sid = sections[end + 1].get("id")
            if not sid:
                break
            # Include descendants whose printed numbering may also have shifted,
            # but do not absorb unrelated siblings of the conflicting section.
            if sid.startswith(f"{conflict_id}.") or self._section_parent_id(sid) == conflict_id:
                end += 1
                continue
            break

        return list(range(start, end + 1))

    def _would_create_conflicting_ids(
        self,
        sections: List[Dict[str, Any]],
        proposed_ids: Dict[int, str],
    ) -> bool:
        seen: Dict[str, str] = {}
        for idx, section in enumerate(sections):
            sid = proposed_ids.get(idx, section.get("id"))
            if not sid:
                continue
            title = self._normalize_heading_title(section.get("title", ""))
            existing_title = seen.get(sid)
            if existing_title is not None and existing_title != title:
                return True
            seen[sid] = title
        return False

    def _conflicting_title_sets(
        self,
        sections: List[Dict[str, Any]],
        proposed_ids: Optional[Dict[int, str]] = None,
    ) -> Dict[str, set[str]]:
        """
        Return canonical IDs associated with more than one distinct title.

        ``proposed_ids`` can describe an atomic future state without mutating
        the section records.
        """
        grouped: Dict[str, set[str]] = {}

        for idx, section in enumerate(sections):
            sid = (
                proposed_ids.get(idx, section.get("id"))
                if proposed_ids is not None
                else section.get("id")
            )

            if not sid:
                continue

            grouped.setdefault(sid, set()).add(
                self._normalize_heading_title(section.get("title", ""))
            )

        return {
            sid: titles
            for sid, titles in grouped.items()
            if len(titles) > 1
        }

    def _would_create_new_conflicting_ids(
        self,
        sections: List[Dict[str, Any]],
        proposed_ids: Dict[int, str],
    ) -> bool:
        """
        Reject conflicts introduced by a proposed atomic correction.

        Existing unresolved duplicate groups are tolerated while another
        independent group is being validated. This prevents separate conflict
        groups from blocking one another merely because neither has been
        applied yet.
        """
        before = self._conflicting_title_sets(sections)
        after = self._conflicting_title_sets(
            sections,
            proposed_ids=proposed_ids,
        )

        for sid, titles in after.items():
            original_titles = before.get(sid)

            if original_titles is None:
                return True

            if not titles.issubset(original_titles):
                return True

        return False

    def _align_duplicate_group_by_unique_body_titles(
        self,
        sections: List[Dict[str, Any]],
        conflict_indices: List[int],
        conflict_id: str,
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """Conservative fallback for a conflicting duplicate-ID group.

        The normal local alignment remains the preferred path. This fallback is
        used only when every duplicate title has a strong, unique body-heading
        ID match and the resulting IDs are distinct and monotonic. It handles
        source TOCs that collapse sibling numbers, for example two different
        titles both printed as ``9.4.2.1`` while the body contains
        ``9.4.2.1`` and ``9.4.2.2``.
        """
        if len(conflict_indices) < 2:
            return None

        candidate_lists: List[List[Dict[str, Any]]] = []

        for idx in conflict_indices:
            section = sections[idx]
            section_title = section.get("title", "")
            exact_title = self._normalize_heading_title(section_title)
            by_candidate_id: Dict[str, Dict[str, Any]] = {}

            for heading in self._get_body_heading_index():
                candidate_id = heading.get("id")
                if not candidate_id or not self._ids_hierarchically_compatible(
                    section.get("id"),
                    candidate_id,
                    conflict_id,
                ):
                    continue

                similarity = self._title_similarity(section_title, heading.get("title", ""))
                heading_title = self._normalize_heading_title(heading.get("title", ""))
                if similarity < 0.90 and heading_title != exact_title:
                    continue

                candidate = {
                    **heading,
                    "title_similarity": similarity,
                    "match_score": similarity + (0.08 if heading_title == exact_title else 0.0),
                }
                previous = by_candidate_id.get(candidate_id)
                if previous is None or (
                    float(candidate["match_score"]),
                    -int(candidate["source_order"]),
                ) > (
                    float(previous["match_score"]),
                    -int(previous["source_order"]),
                ):
                    by_candidate_id[candidate_id] = candidate

            candidates = sorted(
                by_candidate_id.values(),
                key=lambda item: (-float(item["match_score"]), int(item["source_order"])),
            )
            if not candidates:
                return None
            candidate_lists.append(candidates[:12])

        beam: List[tuple[float, int, frozenset[str], Dict[int, Dict[str, Any]]]] = [
            (0.0, -1, frozenset(), {})
        ]

        for idx, candidates in zip(conflict_indices, candidate_lists):
            next_beam: List[tuple[float, int, frozenset[str], Dict[int, Dict[str, Any]]]] = []
            for score, last_order, used_ids, mapping in beam:
                for candidate in candidates:
                    candidate_id = str(candidate["id"])
                    source_order = int(candidate["source_order"])
                    if candidate_id in used_ids or source_order <= last_order:
                        continue
                    next_mapping = dict(mapping)
                    next_mapping[idx] = candidate
                    next_beam.append(
                        (
                            score + float(candidate["match_score"]),
                            source_order,
                            used_ids | {candidate_id},
                            next_mapping,
                        )
                    )

            if not next_beam:
                return None
            next_beam.sort(key=lambda item: (-item[0], item[1]))
            beam = next_beam[:64]

        best = beam[0]
        best_ids = [best[3][idx]["id"] for idx in conflict_indices]
        if len(set(best_ids)) != len(best_ids):
            return None

        if len(beam) > 1:
            second_ids = [beam[1][3][idx]["id"] for idx in conflict_indices]
            if best_ids != second_ids and best[0] - beam[1][0] < 0.12:
                return None

        return best[3]

    def _reconcile_conflicting_textual_ids_with_body_headings(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reconcile all resolvable duplicate printed-ID groups atomically.

        Each local group is first aligned independently against ordered body
        headings. The resulting mappings are then combined and validated as
        one future TOC state before any section is mutated. This avoids the
        order-dependent failure where one still-unresolved duplicate group
        prevents correction of another valid group.
        """
        conflicts = self._conflicting_duplicate_id_groups(sections)

        if not conflicts:
            return sections

        group_proposals: List[
            tuple[
                str,
                List[int],
                Dict[int, Dict[str, Any]],
            ]
        ] = []

        for conflict_id, conflict_indices in conflicts.items():
            affected_indices = self._affected_conflict_window(
                sections,
                conflict_id,
                conflict_indices,
            )

            alignment = self._align_conflict_window_to_body(
                sections,
                affected_indices,
                conflict_id,
            )

            if not alignment:
                alignment = self._align_duplicate_group_by_unique_body_titles(
                    sections,
                    conflict_indices,
                    conflict_id,
                )
                if alignment:
                    affected_indices = list(conflict_indices)
                    logger.info(
                        "Recovered conflicting textual TOC id %s in %s using "
                        "unique high-confidence body-title matches",
                        conflict_id,
                        self.doc_id,
                    )

            if not alignment:
                logger.warning(
                    "Unresolved conflicting textual TOC id %s in %s; keeping original ids",
                    conflict_id,
                    self.doc_id,
                )
                continue

            group_proposals.append(
                (
                    conflict_id,
                    affected_indices,
                    alignment,
                )
            )

        if not group_proposals:
            return sections

        accepted_groups: List[
            tuple[
                str,
                List[int],
                Dict[int, Dict[str, Any]],
            ]
        ] = []
        proposed_ids: Dict[int, str] = {}
        proposed_headings: Dict[int, Dict[str, Any]] = {}

        # First try to combine every resolvable group into one atomic proposal.
        merge_conflict = False

        for conflict_id, affected_indices, alignment in group_proposals:
            for idx in affected_indices:
                corrected_id = alignment[idx]["id"]
                existing = proposed_ids.get(idx)

                if existing is not None and existing != corrected_id:
                    merge_conflict = True
                    logger.warning(
                        "Overlapping TOC reconciliation proposals disagree in %s | "
                        "section_index=%d | first=%s | second=%s | group=%s",
                        self.doc_id,
                        idx,
                        existing,
                        corrected_id,
                        conflict_id,
                    )
                    break

                proposed_ids[idx] = corrected_id
                proposed_headings[idx] = alignment[idx]

            if merge_conflict:
                break

            accepted_groups.append(
                (
                    conflict_id,
                    affected_indices,
                    alignment,
                )
            )

        combined_valid = (
            not merge_conflict
            and not self._would_create_new_conflicting_ids(
                sections,
                proposed_ids,
            )
            and self._proposed_ids_have_plausible_structure(
                sections,
                proposed_ids,
            )
        )

        if not combined_valid:
            # A malformed local group should not prevent independent valid
            # groups from being corrected. Rebuild the proposal incrementally,
            # while tolerating duplicate conflicts already present in the
            # original TOC.
            accepted_groups = []
            proposed_ids = {}
            proposed_headings = {}

            for conflict_id, affected_indices, alignment in group_proposals:
                trial_ids = dict(proposed_ids)
                trial_headings = dict(proposed_headings)
                overlap_conflict = False

                for idx in affected_indices:
                    corrected_id = alignment[idx]["id"]
                    existing = trial_ids.get(idx)

                    if existing is not None and existing != corrected_id:
                        overlap_conflict = True
                        break

                    trial_ids[idx] = corrected_id
                    trial_headings[idx] = alignment[idx]

                if (
                    overlap_conflict
                    or self._would_create_new_conflicting_ids(
                        sections,
                        trial_ids,
                    )
                    or not self._proposed_ids_have_plausible_structure(
                        sections,
                        trial_ids,
                    )
                ):
                    logger.warning(
                        "Rejected body-heading reconciliation for textual TOC id %s in %s; "
                        "proposed sequence is structurally inconsistent",
                        conflict_id,
                        self.doc_id,
                    )
                    continue

                proposed_ids = trial_ids
                proposed_headings = trial_headings
                accepted_groups.append(
                    (
                        conflict_id,
                        affected_indices,
                        alignment,
                    )
                )

        if not accepted_groups:
            return sections

        corrected_count = 0
        corrected_groups = {
            conflict_id
            for conflict_id, _, _ in accepted_groups
        }

        # Apply the complete accepted mapping only after validation.
        for idx in sorted(proposed_ids):
            section = sections[idx]
            original_id = section.get("id")
            corrected_id = proposed_ids[idx]

            if not original_id or corrected_id == original_id:
                continue

            heading = proposed_headings[idx]

            section["toc_original_section_id"] = section.get(
                "toc_original_section_id",
                original_id,
            )
            section["toc_section_id_corrected_from_body"] = True
            section["id"] = corrected_id
            section["printed_id"] = corrected_id
            section["level"] = self._structural_level(
                corrected_id,
                section.get("level", 1),
            )

            corrected_count += 1

            logger.info(
                "Corrected textual TOC id from atomic body-heading alignment | "
                "doc=%s | original=%s | corrected=%s | title='%s' | body_page=%s",
                self.doc_id,
                original_id,
                corrected_id,
                section.get("title", ""),
                int(heading["page_index"]) + 1,
            )

        logger.info(
            "Corrected %d textual TOC section ids across %d duplicate groups for %s",
            corrected_count,
            len(corrected_groups),
            self.doc_id,
        )

        return sections

    def _log_textual_toc_validation(
        self,
        sections: List[Dict[str, Any]],
        stage: str,
    ) -> None:
        ids = [section.get("id") for section in sections if section.get("id")]
        id_counts = Counter(ids)
        conflicting_duplicates = []
        for sid, count in id_counts.items():
            if count < 2:
                continue
            titles = {
                self._normalize_heading_title(section.get("title", ""))
                for section in sections
                if section.get("id") == sid
            }
            titles.discard("")
            if len(titles) > 1:
                conflicting_duplicates.append(sid)

        id_set = set(ids)
        missing_parents = []
        hierarchy_depth_mismatches = []
        empty_titles = []
        page_order_reversals = 0
        previous_page: Optional[int] = None

        for section in sections:
            sid = section.get("id")
            title = (section.get("title") or "").strip()
            if not title:
                empty_titles.append(sid or "<no-id>")

            if sid:
                expected_level = sid.count(".") + 1
                if int(section.get("level", expected_level)) != expected_level:
                    hierarchy_depth_mismatches.append(sid)

                parent = self._section_parent_id(sid)
                if parent and parent not in id_set:
                    missing_parents.append(sid)

            page = section.get("page_start")
            if isinstance(page, int):
                if previous_page is not None and page < previous_page:
                    page_order_reversals += 1
                previous_page = page

        if conflicting_duplicates:
            logger.warning(
                "Textual TOC validation (%s): unresolved conflicting duplicate ids in %s: %s",
                stage,
                self.doc_id,
                ", ".join(sorted(conflicting_duplicates)),
            )
        if missing_parents:
            logger.warning(
                "Textual TOC validation (%s): missing parents in %s: %s",
                stage,
                self.doc_id,
                ", ".join(missing_parents[:20]),
            )
        if hierarchy_depth_mismatches:
            logger.warning(
                "Textual TOC validation (%s): hierarchy depth mismatches in %s: %s",
                stage,
                self.doc_id,
                ", ".join(hierarchy_depth_mismatches[:20]),
            )
        if empty_titles:
            logger.warning(
                "Textual TOC validation (%s): empty titles in %s: %s",
                stage,
                self.doc_id,
                ", ".join(empty_titles[:20]),
            )
        if page_order_reversals:
            logger.warning(
                "Textual TOC validation (%s): %d page-order reversals in %s",
                stage,
                page_order_reversals,
                self.doc_id,
            )

    def _make_ids_unique(self, sections: List[Dict[str, Any]]) -> None:
        """
        Graph chunks need stable unique section ids. Some PDFs repeat the same
        printed number for multiple rows; keep the printed id and make the
        internal id unique with a deterministic suffix.
        """
        totals = Counter(
            section.get("id")
            for section in sections
            if section.get("id")
        )
        seen: Counter[str] = Counter()

        for section in sections:
            sid = section.get("id")
            section["printed_id"] = section.get("printed_id") or sid
            if not sid:
                continue

            seen[sid] += 1
            if totals[sid] <= 1:
                section["duplicate_id_index"] = None
                continue

            duplicate_index = seen[sid]
            section["duplicate_id_index"] = duplicate_index
            if duplicate_index > 1:
                section["id"] = f"{sid}__dup{duplicate_index}"

        duplicate_ids = sorted(section_id for section_id, count in totals.items() if count > 1)
        if duplicate_ids:
            logger.warning(
                "TOC duplicate printed section ids made unique for %s: %s",
                self.doc_id,
                ", ".join(duplicate_ids),
            )

    #   Heading search  
    def _find_heading_top_y(self, page_idx: int, sec: Dict[str, Any]):
        if page_idx < 0 or page_idx >= self.n_pages:
            return None
        page = self.doc.load_page(page_idx)
        title = sec["title"]
        sid = sec.get("id") or ""
        candidates = []
        if sid:
            candidates.append(f"{sid}. {title}")
            candidates.append(f"{sid} {title}")
        candidates.append(title)

        best_y = None
        for cand in candidates:
            rects = page.search_for(cand)
            for r in rects:
                y = float(r.y0)
                if best_y is None or y < best_y:
                    best_y = y
        return best_y

    def compute_outline_ranges(self, sections: List[Dict[str, Any]]) -> None:
        for i, sec in enumerate(sections):
            lvl = sec["level"]
            nxt = None

            for j in range(i + 1, len(sections)):
                if sections[j]["level"] <= lvl:
                    nxt = sections[j]
                    break

            if not nxt:
                sec["page_end"] = self.n_pages
                continue

            end = nxt["page_start"] - 1

            if nxt["page_start"] <= self.n_pages:
                y = self._find_heading_top_y(nxt["page_start"] - 1, nxt)
                if y and y > self.HEADING_TOP_Y_THRESHOLD:
                    end = nxt["page_start"]

            sec["page_end"] = max(sec["page_start"], end)

    #   Classification  
    def classify_sections(self, sections: List[Dict[str, Any]]) -> None:
        for s in sections:
            t = (s["title"] or "").lower()
            if s.get("id") is None and t.strip() == "preamble":
                s["type"] = "front_matter"
                continue

            # Front matter: first page with level 1 and no id
            if s["level"] == 1 and s.get("id") is None and s["page_start"] == 1:
                s["type"] = "front_matter"
                continue
            # Common front-matter titles (TOC, abbreviations, lists)
            if t.startswith(
                (
                    "table of contents",
                    "contents",
                    "acronyms",
                    "abbreviation",
                    "list of figures",
                    "list of tables",
                )
            ):
                # TODO: evaluate whether "preamble" should also be included
                s["type"] = "front_matter"
                continue
            # Common back-matter titles
            if t.startswith(
                (
                    "reference",
                    "appendix",
                    "supplementary",
                    "evidence table",
                    "acknowledge",
                    "author",
                    "bibliography",
                    "data availability",
                    "quality indicator",
                )
            ):
                s["type"] = "back_matter"


    def _expand_parent_ranges_to_descendants(
        self,
        sections: List[Dict[str, Any]],
    ) -> None:
        """
        Ensure each section page range contains all descendant ranges.

        Direct-text extraction may still stop parents at children, but
        provenance metadata should reflect the full hierarchical span.
        """
        by_id = {
            section.get("id"): section
            for section in sections
            if section.get("id")
        }

        for section in sorted(
            sections,
            key=lambda item: (
                int(item.get("level") or 1),
                str(item.get("id") or ""),
            ),
            reverse=True,
        ):
            sid = section.get("id")

            if not sid or "." not in sid:
                continue

            parent_id = sid.rsplit(".", 1)[0]
            parent = by_id.get(parent_id)

            if not parent:
                continue

            parent["page_start"] = min(
                int(parent.get("page_start") or section["page_start"]),
                int(section.get("page_start") or parent["page_start"]),
            )

            parent["page_end"] = max(
                int(parent.get("page_end") or parent["page_start"]),
                int(section.get("page_end") or section["page_start"]),
            )

    #   Build tree  
    def build_tree(self, flat: List[TOCSection]) -> List[TOCSection]:
        roots: List[TOCSection] = []
        stack: List[TOCSection] = []
        for sec in flat:
            sec.children = []
            # Pop until we find a true parent
            while stack and stack[-1].level >= sec.level:
                stack.pop()
            if stack:
                stack[-1].children.append(sec)
            else:
                roots.append(sec)
            stack.append(sec)
        return roots

    #   Fallback textual TOC
    def _find_printed_toc_start(self) -> Optional[int]:
        for i in range(min(10, self.n_pages)):
            text = self.doc.load_page(i).get_text("text")
            low = text.lower().replace(" ", "")
            if "tableofcontents" in low or "contents" in low:
                logger.info("Detected TOC starting on PDF page %d", i + 1)
                return i
        return None

    def extract_printed_toc_records(self) -> List[Dict[str, Any]]:
        toc_start = self._find_printed_toc_start()
        if toc_start is None:
            raise ValueError("NO_TOC_TEXT")

        printed_toc_records: List[Dict[str, Any]] = []
        toc_end = toc_start
        for p in range(toc_start, min(toc_start + 6, self.n_pages)):
            page = self.doc.load_page(p)
            page_records = self._parse_textual_toc_records(page.get_text("text"))
            valid_page_records = [
                record
                for record in page_records
                if self._textual_record_to_section(record) is not None
            ]

            if not valid_page_records:
                logger.info(
                    "Detected TOC ending before PDF page %d",
                    p + 1,
                )
                break

            toc_end = p
            for record in valid_page_records:
                record["_source_order"] = len(printed_toc_records)
                printed_toc_records.append(record)

        if not printed_toc_records:
            raise ValueError("FAILED_TOC")

        self._printed_toc_page_range = (toc_start, toc_end)

        # The body-heading index may have been requested before the complete
        # printed-TOC interval was known. Invalidate it so the next access
        # excludes every detected TOC page.
        self._body_heading_index = None

        logger.info(
            "Parsed %d printed TOC records from PDF pages %d-%d",
            len(printed_toc_records),
            toc_start + 1,
            toc_end + 1,
        )
        return printed_toc_records

    def _printed_toc_record_to_canonical_section(
        self,
        record: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        section = self._textual_record_to_section(record)
        if not section:
            return None

        sid = section.get("id")
        title = section.get("title") or ""

        SUPPLEMENT_PATTERNS = (
            r"^table\b",
            r"^figure\b",
            r"^list of\b",
            r"^tables\b",
            r"^figures\b",
        )

        title_lower = title.lower().strip()
        is_supplement = any(re.match(pattern, title_lower) for pattern in SUPPLEMENT_PATTERNS)

        # Exclude non-numbered sections that are likely tables/figures or
        # recommendation index rows. They are content pointers, not stable
        # hierarchy nodes, and can reset the level stack incorrectly.
        if not sid and is_supplement:
            logger.debug("Skipping supplemental TOC entry: %s", title)
            return None
        if not sid and self._is_non_structural_unnumbered_title(title):
            logger.debug("Skipping non-structural TOC entry: %s", title)
            return None

        return section

    def _printed_toc_records_to_sections(
        self,
        printed_toc_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        canonical_sections: List[Dict[str, Any]] = []
        for record in printed_toc_records:
            section = self._printed_toc_record_to_canonical_section(record)
            if not section:
                continue
            section["_source_order"] = record.get("_source_order", len(canonical_sections))
            canonical_sections.append(section)
        return canonical_sections

    def fallback_extract_toc(self) -> List[Dict[str, Any]]:
        logger.info("Using textual_reconciled TOC strategy")

        printed_toc_records = self.extract_printed_toc_records()
        body_heading_candidates = self.extract_body_heading_candidates()
        self._body_heading_index = body_heading_candidates
        logger.info(
            "Extracted %d body heading candidates for textual TOC reconciliation",
            len(body_heading_candidates),
        )

        canonical_sections = self._printed_toc_records_to_sections(printed_toc_records)
        if not canonical_sections:
            raise ValueError("FAILED_TOC")

        canonical_sections = self._deduplicate_textual_sections(canonical_sections)
        canonical_sections = self._reconcile_conflicting_textual_ids_with_body_headings(canonical_sections)
        self._log_textual_toc_validation(canonical_sections, "after_body_reconciliation")
        canonical_sections = self._deduplicate_outline(canonical_sections)
        canonical_sections = self._recover_missing_textual_parents(
            canonical_sections,
            printed_toc_records,
        )
        canonical_sections.sort(key=lambda section: int(section.get("_source_order", 10**9)))

        # Preserve textual TOC source order. Structural issues are logged for
        # review instead of being hidden by page sorting.
        if self._toc_needs_sorting(canonical_sections):
            logger.warning(
                "Textual TOC structure has validation issues; preserving source order"
            )
        self._validate_textual_sections(canonical_sections)

        # Force 'References' / 'Bibliography' to be terminal TOC boundary
        for i, s in enumerate(canonical_sections):
            title = (s.get("title") or "").lower()
            if title.startswith(("reference", "bibliograph")):
                logger.info("Truncating TOC after References section")
                canonical_sections = canonical_sections[: i + 1]
                break

        logger.info(
            "Produced %d textual_reconciled canonical TOC sections after cleanup",
            len(canonical_sections),
        )
        return canonical_sections

    def _parse_toc_page(self, text):
        """
        Parse a textual TOC page into structured section entries.
        Also filters out supplemental non-structural entries such as:
        - Table ...
        - Figure ...
        - List of Tables
        - List of Figures
        These should not appear as structural TOC sections.
        """
        records = self._parse_textual_toc_records(text)
        entries = []

        #  extract structured entries 
        for record in records:
            section = self._printed_toc_record_to_canonical_section(record)
            if not section:
                continue

            entries.append(section)

        return entries


    def compute_fallback_ranges(self, sections: List[Dict[str, Any]]) -> None:
        for i, sec in enumerate(sections):
            lvl = sec["level"]
            nxt = None
            for j in range(i + 1, len(sections)):
                if sections[j]["level"] <= lvl:
                    nxt = sections[j]
                    break

            if nxt:
                sec["page_end"] = max(sec["page_start"], nxt["page_start"] - 1)
            else:
                sec["page_end"] = sec["page_start"]

    def _edge_page_number_candidates(self, page_idx: int) -> List[int]:
        """
        Extract plausible printed page labels from page header/footer lines.

        Textual TOCs often contain journal/publication page numbers instead of
        PDF page indices. We only look at page edges to avoid counting section
        numbers, years, table values, or references from the body.
        """
        page_text = self.doc.load_page(page_idx).get_text("text")
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        edge_lines = (
            lines[: self.EDGE_LINE_COUNT_FOR_PAGE_LABELS]
            + lines[-self.EDGE_LINE_COUNT_FOR_PAGE_LABELS :]
        )

        candidates: List[int] = []
        for line in edge_lines:
            for match in re.finditer(r"\b\d{1,5}\b", line):
                value = int(match.group(0))
                if 1 <= value <= 20000:
                    candidates.append(value)
        return candidates

    def _body_heading_page_offset_evidence(
        self,
        sections: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Infer printed-page offset from strong TOC/body heading matches.

        This path is evaluated even when all textual-TOC page numbers happen
        to fall within the physical PDF page count. That is necessary for
        journal PDFs whose editorial numbering starts above page 1 but still
        remains numerically smaller than ``n_pages``.

        Each TOC section contributes at most one vote. A vote is accepted only
        when exact section ID and strong title agreement identify an
        unambiguous physical body page.
        """
        headings_by_id: Dict[str, List[Dict[str, Any]]] = {}
        for heading in self._get_body_heading_index():
            sid = heading.get("id")
            if sid:
                headings_by_id.setdefault(str(sid), []).append(heading)

        offset_votes: Counter[int] = Counter()
        evidence_rows: List[Dict[str, Any]] = []
        eligible_sections = 0

        for section in sections:
            sid = section.get("id")
            raw_page = section.get("page_start")
            if not sid or not isinstance(raw_page, int):
                continue

            candidates = []
            for heading in headings_by_id.get(str(sid), []):
                similarity = self._title_similarity(
                    section.get("title", ""),
                    heading.get("title", ""),
                )
                if similarity < self.MIN_BODY_OFFSET_TITLE_SIMILARITY:
                    continue
                candidates.append((similarity, heading))

            if not candidates:
                continue

            eligible_sections += 1
            candidates.sort(
                key=lambda item: (-item[0], int(item[1]["source_order"]))
            )
            best_similarity = candidates[0][0]
            top = [
                heading
                for similarity, heading in candidates
                if best_similarity - similarity < 0.02
            ]
            candidate_offsets = {
                int(raw_page) - (int(heading["page_index"]) + 1)
                for heading in top
            }

            # Repeated headings on different pages are ambiguous and should
            # not influence a corpus-wide page mapping.
            if len(candidate_offsets) != 1:
                continue

            offset = candidate_offsets.pop()
            if offset < 0:
                continue

            offset_votes[offset] += 1
            chosen = min(top, key=lambda item: int(item["source_order"]))
            evidence_rows.append(
                {
                    "section_id": sid,
                    "printed_page": int(raw_page),
                    "physical_page": int(chosen["page_index"]) + 1,
                    "offset": offset,
                    "title_similarity": best_similarity,
                }
            )

        if not offset_votes:
            return None

        offset, support = offset_votes.most_common(1)[0]
        total_votes = sum(offset_votes.values())
        dominance = support / max(1, total_votes)

        raw_pages = [
            int(section["page_start"])
            for section in sections
            if isinstance(section.get("page_start"), int)
        ]
        mapped_in_range = sum(
            1 for page in raw_pages if 1 <= page - offset <= self.n_pages
        )
        mapped_coverage = mapped_in_range / max(1, len(raw_pages))
        body_match_coverage = support / max(1, eligible_sections)

        status = "candidate"
        if (
            support < self.MIN_BODY_OFFSET_SUPPORT
            or dominance < self.MIN_BODY_OFFSET_DOMINANCE
            or body_match_coverage < self.MIN_BODY_OFFSET_MATCH_COVERAGE
            or mapped_coverage < self.MIN_PAGE_NORMALIZATION_COVERAGE
        ):
            status = "insufficient_body_evidence"
        elif offset == 0:
            status = "not_needed"

        examples = [
            row
            for row in evidence_rows
            if row["offset"] == offset
        ][:8]

        return {
            "status": status,
            "method": "body_heading_offset",
            "offset": offset,
            "coverage": mapped_coverage,
            "body_match_coverage": body_match_coverage,
            "support": support,
            "total_votes": total_votes,
            "dominance": dominance,
            "eligible_sections": eligible_sections,
            "examples": examples,
        }

    def _detect_printed_page_offset(
        self,
        sections: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        raw_pages = sorted(
            {
                int(sec["page_start"])
                for sec in sections
                if sec.get("page_start") is not None
            }
        )
        if not raw_pages:
            return None

        body_info = self._body_heading_page_offset_evidence(sections)
        if body_info and body_info["status"] in {"candidate", "not_needed"}:
            body_info.update(
                {
                    "out_of_range_pages": sum(
                        1 for page in raw_pages if page > self.n_pages
                    ),
                    "total_pages": len(raw_pages),
                }
            )
            return body_info

        out_of_range_pages = [page for page in raw_pages if page > self.n_pages]
        if not out_of_range_pages:
            result = {
                "status": "not_needed",
                "method": "physical_page_range",
                "offset": 0,
                "coverage": 1.0,
                "out_of_range_pages": 0,
                "total_pages": len(raw_pages),
            }
            if body_info:
                result["body_heading_evidence"] = body_info
            return result

        offsets: Counter[int] = Counter()
        for page_idx in range(self.n_pages):
            pdf_page = page_idx + 1
            for printed_page in self._edge_page_number_candidates(page_idx):
                offset = printed_page - pdf_page
                if offset <= 0:
                    continue

                mapped_count = sum(
                    1
                    for raw_page in raw_pages
                    if 1 <= raw_page - offset <= self.n_pages
                )
                if mapped_count:
                    offsets[offset] += mapped_count

        if not offsets:
            return body_info

        offset, votes = offsets.most_common(1)[0]
        mapped_pages = [
            raw_page - offset
            for raw_page in raw_pages
            if raw_page > self.n_pages
        ]
        mapped_in_range = sum(
            1 for page in mapped_pages if 1 <= page <= self.n_pages
        )
        coverage = mapped_in_range / max(1, len(out_of_range_pages))

        result = {
            "status": "candidate",
            "method": "edge_page_labels",
            "offset": offset,
            "coverage": coverage,
            "votes": votes,
            "out_of_range_pages": len(out_of_range_pages),
            "total_pages": len(raw_pages),
        }
        if body_info:
            result["body_heading_evidence"] = body_info
        return result

    def _printed_page_labels(self) -> set[int]:
        labels: set[int] = set()
        for page_idx in range(self.n_pages):
            labels.update(self._edge_page_number_candidates(page_idx))
        return labels

    def _normalize_page_with_offset(self, page: int, offset: int) -> Optional[int]:
        mapped = page - offset
        if 1 <= mapped <= self.n_pages:
            return mapped
        if 1 <= page <= self.n_pages:
            return page
        return None

    def _repair_single_digit_printed_page(
        self,
        page: int,
        offset: int,
        printed_labels: set[int],
        previous_page: Optional[int],
        next_page: Optional[int],
    ) -> Optional[int]:
        """
        Repair likely OCR mistakes in printed TOC pages.

        Example: a printed journal page "3269" may be read as "4269".
        We only accept a one-digit repair if it maps into the PDF and keeps
        the TOC page sequence monotonic around neighboring valid entries.
        """
        raw = str(page)
        candidates: List[tuple[int, int, int, int]] = []

        for idx, original_digit in enumerate(raw):
            for replacement in "0123456789":
                if replacement == original_digit:
                    continue

                candidate_raw = int(raw[:idx] + replacement + raw[idx + 1 :])
                mapped = candidate_raw - offset
                if not 1 <= mapped <= self.n_pages:
                    continue

                below_previous = previous_page is not None and mapped < previous_page
                above_next = next_page is not None and mapped > next_page
                order_penalty = int(below_previous or above_next)

                if previous_page is not None and mapped < previous_page:
                    order_distance = previous_page - mapped
                elif next_page is not None and mapped > next_page:
                    order_distance = mapped - next_page
                else:
                    order_distance = 0

                label_penalty = 0 if candidate_raw in printed_labels else 1
                candidates.append(
                    (
                        order_penalty,
                        order_distance,
                        label_penalty,
                        mapped,
                    )
                )

        if not candidates:
            return None

        candidates.sort()
        best_order_penalty, _, best_label_penalty, best_mapped = candidates[0]
        if best_order_penalty or best_label_penalty:
            return None

        return best_mapped

    def _normalize_textual_toc_page_sequence(
        self,
        sections: List[Dict[str, Any]],
        offset: int,
    ) -> Dict[str, int]:
        printed_labels = self._printed_page_labels()
        normalized_pages: List[Optional[int]] = []

        for sec in sections:
            page = int(sec["page_start"])
            normalized_pages.append(self._normalize_page_with_offset(page, offset))

        direct_values = 0
        repaired_values = 0
        unresolved_values = 0

        for idx, sec in enumerate(sections):
            raw_page = int(sec["page_start"])
            mapped = normalized_pages[idx]

            if mapped is None:
                previous_page = next(
                    (
                        normalized_pages[j]
                        for j in range(idx - 1, -1, -1)
                        if normalized_pages[j] is not None
                    ),
                    None,
                )
                next_page = next(
                    (
                        normalized_pages[j]
                        for j in range(idx + 1, len(normalized_pages))
                        if normalized_pages[j] is not None
                    ),
                    None,
                )
                mapped = self._repair_single_digit_printed_page(
                    page=raw_page,
                    offset=offset,
                    printed_labels=printed_labels,
                    previous_page=previous_page,
                    next_page=next_page,
                )
                normalized_pages[idx] = mapped
                if mapped is not None:
                    repaired_values += 2

            if mapped is None:
                unresolved_values += 2
                continue

            for key in ("page_start", "page_end"):
                if int(sec[key]) != mapped:
                    sec[key] = mapped
                    if raw_page - offset == mapped:
                        direct_values += 1
                else:
                    sec[key] = mapped

        return {
            "direct_values": direct_values,
            "repaired_values": repaired_values,
            "unresolved_values": unresolved_values,
        }

    def normalize_textual_toc_pages(
        self,
        sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map textual TOC publication pages to PDF page indices when safe.

        If the mapping is uncertain, leave pages unchanged. The chunker will
        disable anchor narrowing for out-of-range pages and fall back to guarded
        body search instead of clamping to the final PDF page.
        """
        info = self._detect_printed_page_offset(sections)
        if not info:
            logger.warning(
                "Could not infer textual TOC page normalization for %s; "
                "out-of-range pages will use global guarded header search",
                self.doc_id,
            )
            return {
                "status": "not_inferred",
                "normalized": False,
                "offset": None,
            }

        if info["status"] == "not_needed":
            info["normalized"] = False
            return info

        if info["coverage"] < self.MIN_PAGE_NORMALIZATION_COVERAGE:
            logger.warning(
                "Textual TOC page normalization rejected for %s | offset=%s | coverage=%.2f",
                self.doc_id,
                info["offset"],
                info["coverage"],
            )
            info["status"] = "rejected_low_coverage"
            info["normalized"] = False
            return info

        offset = int(info["offset"])
        normalization_counts = self._normalize_textual_toc_page_sequence(
            sections,
            offset,
        )

        logger.info(
            "Normalized textual TOC publication pages for %s | "
            "method=%s | offset=%d | coverage=%.2f | direct=%d | repaired=%d | unresolved=%d",
            self.doc_id,
            info.get("method", "unknown"),
            offset,
            info["coverage"],
            normalization_counts["direct_values"],
            normalization_counts["repaired_values"],
            normalization_counts["unresolved_values"],
        )

        info["status"] = "normalized"
        info["normalized"] = True
        info.update(normalization_counts)
        info["normalized_values"] = (
            normalization_counts["direct_values"]
            + normalization_counts["repaired_values"]
        )
        return info

    #  Pipeline 
    def run(self) -> TOCMetadata:
        page_normalization: Dict[str, Any] = {}
        outline = self.extract_pdf_outline()
        outline_reliable, outline_rejection_reason = self.outline_is_reliable(outline)

        if outline_reliable:
            extraction_mode = "outline"
            logger.info("TOC extraction mode: outline")
            raw = outline
            toc_source = "pdf_outline"
            if self._toc_needs_sorting(raw):
                raw = self._safe_sort(raw)
            for i, s in enumerate(raw):
                title = (s.get("title") or "").lower()
                if title.startswith(("reference", "bibliograph")):
                    logger.info("Truncating outline after References section")
                    raw = raw[: i + 1]
                    break
            self.compute_outline_ranges(raw)
        else:
            extraction_mode = "textual_reconciled"
            if outline:
                logger.warning(
                    "PDF outline present but rejected: %s",
                    outline_rejection_reason,
                )
            logger.info("TOC extraction mode: textual_reconciled")
            raw = self.fallback_extract_toc()
            toc_source = "textual_toc"
            page_normalization = self.normalize_textual_toc_pages(raw)
            self.compute_fallback_ranges(raw)

        self.classify_sections(raw)
        self._expand_parent_ranges_to_descendants(raw)
        self._make_ids_unique(raw)

        # Keep flat_toc and toc_tree independent. build_tree() mutates the
        # children lists, so using the same objects would make flat_toc nested
        # and duplicate descendants during recursive consumption.
        flat = [TOCSection(**s) for s in raw]
        tree_nodes = [TOCSection(**s) for s in raw]
        tree = self.build_tree(tree_nodes)

        logger.info(
            "TOC extraction completed (%s/%s): %d sections, %d roots",
            extraction_mode,
            toc_source,
            len(flat),
            len(tree),
        )

        return TOCMetadata(
            doc_id=self.doc_id,
            toc_source=toc_source,
            n_pages=self.n_pages,
            flat_toc=flat,
            toc_tree=tree,
            page_normalization=page_normalization,
        )

    def save(self, output_path: str) -> None:
        try:
            meta = self.run()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(meta.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info("TOC metadata saved to %s", output_path)
        finally:
            self.close()
