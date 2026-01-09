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
import os  # Used in the __main__ test block
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Section IDs like "1", "1.1", "3.7.4.2"
SECTION_ID_RE = re.compile(r"^(\d+(?:\.\d+)*)")


# Pydantic models

class TOCSection(BaseModel):
    id: Optional[str]
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

    def __init__(self, pdf_path: str, doc_id: str):
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.doc = fitz.open(pdf_path)
        self.n_pages = self.doc.page_count

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

    #   PDF outline mode (preferred)  
    def extract_raw_toc(self) -> List[Dict[str, Any]]:
        raw = self.doc.get_toc(simple=False)  # direct outline, if present

        if not raw:
            raise ValueError("NO_OUTLINE")

        sections: List[Dict[str, Any]] = []

        for level, title, page, *_ in raw:
            title = (title or "").strip()

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
                    "title": clean_title,
                    "level": int(level),
                    "page_start": int(page),
                    "page_end": None,
                    "type": "body",
                }
            )

        sections = self._deduplicate_outline(sections)

        # Heuristic check for broken outlines
        bad_count = sum(
            1
            for s in sections
            if (s["title"] or "").lower().startswith(("ehab", "tblfn", "|", "ehz", "ehy", "eh"))
        )
        if bad_count / max(1, len(sections)) > 0.3:
            logger.warning(
                "Outline rejected as broken: %d/%d suspicious entries",
                bad_count,
                len(sections),
            )
            raise RuntimeError("BROKEN_OUTLINE")

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
    def fallback_extract_toc(self) -> List[Dict[str, Any]]:
        logger.info("Using fallback textual TOC")

        toc_start = None
        for i in range(min(10, self.n_pages)):
            text = self.doc.load_page(i).get_text("text")
            low = text.lower().replace(" ", "")
            if "tableofcontents" in low or "contents" in low:
                toc_start = i
                logger.info("Detected TOC starting on PDF page %d", i + 1)
                break

        if toc_start is None:
            raise ValueError("NO_TOC_TEXT")

        sections: List[Dict[str, Any]] = []

        for p in range(toc_start, min(toc_start + 6, self.n_pages)):
            page = self.doc.load_page(p)
            parsed = self._parse_toc_page(page.get_text("text"))
            if not parsed:
                break
            sections.extend(parsed)

        if not sections:
            raise ValueError("FAILED_TOC")

        sections = self._deduplicate_outline(sections)

        # Smart ordering: only sort if structurally broken
        if self._toc_needs_sorting(sections):
            sections = self._safe_sort(sections)

        # Force 'References' / 'Bibliography' to be terminal TOC boundary
        for i, s in enumerate(sections):
            title = (s.get("title") or "").lower()
            if title.startswith(("reference", "bibliograph")):
                logger.info("Truncating TOC after References section")
                sections = sections[: i + 1]
                break

        logger.info("Parsed %d textual TOC entries after cleanup", len(sections))
        return sections

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

        SUPPLEMENT_PATTERNS = (
            r"^table\b",
            r"^figure\b",
            r"^list of\b",
            r"^tables\b",
            r"^figures\b",
        )

        def is_supplement(title: str) -> bool:
            t = title.lower().strip()
            return any(re.match(p, t) for p in SUPPLEMENT_PATTERNS)

        lines = text.splitlines()
        stitched = []
        buf = ""

        #  normalize broken TOC wrapped lines 
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            low = line.lower()
            if any(p in low for p in self.BAD_PAGE_WORDS):
                continue
            if len(line) > 250:
                continue

            # If line ends with number, it's a completed TOC row
            if re.search(r"\d+\s*$", line):
                buf += " " + line
                stitched.append(buf.strip())
                buf = ""
            else:
                buf += " " + line

        if buf:
            stitched.append(buf.strip())

        entries = []

        #  extract structured entries 
        for logical in stitched:
            m = self.TOC_LINE.match(logical)
            if not m:
                continue

            sid, title, page_str = m.groups()

            # Validate page
            try:
                page = int(page_str)
            except:
                continue
            if page < 1 or page > 20000:
                continue

            # Clean title
            title = title.strip(". ").strip()
            if len(title) < 3:
                continue
            if len(title) > 140:
                continue
            if title.count(" ") > 18:
                continue
            if title.count(".") > 1:
                continue

            low_title = title.lower()
            if any(p in low_title for p in self.BAD_TITLE_PATTERNS):
                continue
            if any(w in low_title for w in self.PROSE_WORDS):
                continue

            # Exclude non-numbered sections that are likely tables/figures
            if not sid and is_supplement(title):
                logger.debug("Skipping supplemental TOC entry: %s", title)
                continue

            level = sid.count(".") + 1 if sid else 1

            entries.append(
                {
                    "id": sid,
                    "title": title,
                    "level": level,
                    "page_start": page,
                    "page_end": page,
                    "type": "body",
                }
            )

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

    #  Pipeline 
    def run(self) -> TOCMetadata:
        try:
            raw = self.extract_raw_toc()
            toc_source = "pdf_outline"
            self.compute_outline_ranges(raw)
        except Exception:
            logger.warning("Outline missing or broken → using textual TOC fallback")
            raw = self.fallback_extract_toc()
            toc_source = "textual_toc"
            self.compute_fallback_ranges(raw)

        self.classify_sections(raw)

        flat = [TOCSection(**s) for s in raw]
        tree = self.build_tree(flat)

        logger.info(
            "TOC extraction completed (%s): %d sections, %d roots",
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
        )

    def save(self, output_path: str) -> None:
        try:
            meta = self.run()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(meta.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info("TOC metadata saved to %s", output_path)
        finally:
            self.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir = "../../data/pdfdocs"
    output_dir = "../../data/toc"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(input_dir, filename)
        doc_id = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{doc_id}_toc.json")

        logging.info("Processing %s", filename)

        extractor = GuidelineTOCExtractor(
            pdf_path=pdf_path,
            doc_id=doc_id,
        )
        extractor.save(output_path)

    logging.info("Finished processing all PDFs.")
