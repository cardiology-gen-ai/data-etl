"""
Guideline TOC Extraction Pipeline

Extracts the authoritative hierarchical Table of Contents
from a PDF guideline using the embedded PDF outline.

Outputs:
- Flat TOC with page ranges (page_start/page_end)
- Hierarchical TOC tree suitable for Graph RAG
"""

import fitz
import re
import json
import logging
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field, model_validator

# Configure logging
logger = logging.getLogger(__name__)

# Regex for section IDs (ESC guideline compatible)
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
    toc_source: str = "pdf_outline"
    n_pages: int
    flat_toc: List[TOCSection]
    toc_tree: List[TOCSection]


TOCSection.model_rebuild()


# TOC extractor

class GuidelineTOCExtractor:
    """
    Extracts and normalizes the Table of Contents from a PDF guideline.
    """

    #If next section heading top Y is below this threshold,
    # we assume previous section continues on that page.
    HEADING_TOP_Y_THRESHOLD = 110

    def __init__(self, pdf_path: str, doc_id: str):
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.doc = fitz.open(pdf_path)
        self.n_pages = self.doc.page_count

        logger.info(
            "Initialized TOC extractor for %s (%d pages)",
            pdf_path, self.n_pages
        )

    def close(self) -> None:
        if self.doc is not None:
            self.doc.close()
    
    # Extract raw TOC from PDF outline
    def extract_raw_toc(self) -> List[Dict[str, Any]]:
        raw_toc = self.doc.get_toc(simple=False)

        if not raw_toc:
            logger.error("No PDF outline found")
            raise ValueError("PDF does not contain a logical TOC (outline).")

        logger.info("Extracted %d TOC entries (raw)", len(raw_toc))

        sections: List[Dict[str, Any]] = []

        for level, title, page, *_ in raw_toc:
            title = (title or "").strip()

            match = SECTION_ID_RE.match(title)
            section_id = None
            clean_title = title

            if match:
                candidate_id = match.group(1)

                # Exclude year-like numeric prefixes (e.g. 2023) to avoid having the guideline title as section ID
                if not (candidate_id.isdigit() and 1900 <= int(candidate_id) <= 2100):
                    section_id = candidate_id
                    clean_title = title[len(candidate_id):].strip(" .")

            sections.append({
                "id": section_id,
                "title": clean_title.strip(),
                "level": int(level),
                "page_start": int(page),  # PyMuPDF TOC pages are 1-based
                "page_end": None,
                "type": "body",
            })

        sections = self._deduplicate_outline(sections)
        logger.info("Kept %d TOC entries", len(sections))
        return sections

    def _deduplicate_outline(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Some PDFs have duplicated outline entries (often near References).
        We drop consecutive duplicates with same (level, id, title, page_start).
        """
        deduped: List[Dict[str, Any]] = []
        prev_key: Optional[Tuple] = None

        for sec in sections:
            key = (sec["level"], sec["id"], sec["title"], sec["page_start"])
            if key == prev_key:
                logger.warning("Dropping duplicate TOC entry: %s", key)
                continue
            deduped.append(sec)
            prev_key = key

        return deduped

    
    # Helpers: find where a heading appears on a page (Y position)
  
    def _heading_candidates(self, sec: Dict[str, Any]) -> List[str]:
        """
        Build a few likely strings we can search for on the page.
        Titles in the outline may be stored without the numeric prefix, but
        the rendered heading often includes it (e.g. "15. Gaps in evidence").
        """
        title = sec["title"].strip()
        sid = (sec["id"] or "").strip()

        cands = []
        if sid:
            # Common ESC styles
            cands.append(f"{sid}. {title}")
            cands.append(f"{sid} {title}")
        cands.append(title)

        # Normalize fancy quotes that sometimes appear
        cands.extend([c.replace("’", "'") for c in cands])
        cands.extend([c.replace("'", "’") for c in cands])

        # Unique preserve order
        seen = set()
        out = []
        for c in cands:
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _find_heading_top_y(self, page_index0: int, sec: Dict[str, Any]) -> Optional[float]:
        """
        Return the smallest y0 among matches for the section heading candidates
        on a page, or None if not found.
        """
        page = self.doc.load_page(page_index0)
        best_y = None

        for cand in self._heading_candidates(sec):
            # Try exact search first
            rects = page.search_for(cand)
            # Removed fallback measure (maybe implement later)
            for r in rects:
                y0 = float(r.y0)
                if best_y is None or y0 < best_y:
                    best_y = y0

        return best_y

    
    # Compute page ranges (with mid-page heading fix)

    def compute_page_ranges(self, sections: List[Dict[str, Any]]) -> None:
        """
        Compute hierarchical page ranges.

        Default: section ends on page before the next section at same or higher level.
        Fix: if the next section heading is mid-page (not near top), then the previous
        section includes that page too.
        """
        for i, sec in enumerate(sections):
            sec_level = sec["level"]
            next_idx = None

            for j in range(i + 1, len(sections)):
                if sections[j]["level"] <= sec_level:
                    next_idx = j
                    break

            if next_idx is None:
                sec["page_end"] = self.n_pages
                continue

            next_sec = sections[next_idx]
            next_start = int(next_sec["page_start"])

            # If next section starts on a later page, decide whether current should include that page.
            end_page = next_start - 1

            if next_start > sec["page_start"]:
                # Check where the next heading appears on its start page.
                y = self._find_heading_top_y(next_start - 1, next_sec)  # convert to 0-based
                if y is not None and y > self.HEADING_TOP_Y_THRESHOLD:
                    # Heading starts mid-page -> previous section includes this page
                    end_page = next_start

            sec["page_end"] = max(int(sec["page_start"]), int(end_page))

        for sec in sections:
            if sec["level"] == 1 and sec["id"] is None:
                first_body_start = None
                for s in sections:
                    if s["level"] == 2 and s["id"] is not None:
                        first_body_start = int(s["page_start"])
                        break
                if first_body_start and first_body_start > 1:
                    sec["page_start"] = 1
                    sec["page_end"] = first_body_start - 1
                else:
                    sec["page_start"] = 1
                    sec["page_end"] = 1
                break

    
    # Classify sections as front matter, back matter, or body
    def classify_sections(self, sections: List[Dict[str, Any]]) -> None:
        for sec in sections:
            title = sec["title"].lower()

            # Document title (level 1 without ID)
            if sec["level"] == 1 and sec["id"] is None:
                sec["type"] = "front_matter"
                continue

            if title.startswith((
                "reference",
                "acknowledge",
                "author",
                "supplementary",
                "data availability",
                "appendix",
            )):
                sec["type"] = "back_matter"

            elif title.startswith((
                "abbreviation",
                "table",
                "list",
            )):
                sec["type"] = "front_matter"


    # Build hierarchical tree
    def build_toc_tree(self, flat_sections: List[TOCSection]) -> List[TOCSection]:
        stack: List[TOCSection] = []
        roots: List[TOCSection] = []

        for sec in flat_sections:
            sec.children = []

            while stack and stack[-1].level >= sec.level:
                stack.pop()

            if stack:
                stack[-1].children.append(sec)
            else:
                roots.append(sec)

            stack.append(sec)

        return roots

    # Run full pipeline
    def run(self) -> TOCMetadata:
        raw_sections = self.extract_raw_toc()
        self.compute_page_ranges(raw_sections)
        self.classify_sections(raw_sections)

        flat_sections = [TOCSection(**sec) for sec in raw_sections]
        toc_tree = self.build_toc_tree(flat_sections)

        logger.info(
            "TOC extraction completed: %d sections (%d root nodes)",
            len(flat_sections), len(toc_tree)
        )

        return TOCMetadata(
            doc_id=self.doc_id,
            n_pages=self.n_pages,
            flat_toc=flat_sections,
            toc_tree=toc_tree
        )

    
    # Save to JSON
    def save(self, output_path: str) -> None:
        try:
            metadata = self.run()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info("TOC metadata saved to %s", output_path)
        finally:
            self.close()


if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
    
    extractor = GuidelineTOCExtractor(
        pdf_path="../../test_data/pdfdocs/Cardiomyopathies_2023.pdf",
        doc_id="Cardiomyopathies_2023"
    )
    extractor.save("../../test_data/toc/Cardiomyopathies_2023_toc.json")
