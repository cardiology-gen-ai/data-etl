"""
Guideline TOC Extraction Pipeline

Extracts the authoritative hierarchical Table of Contents
from a PDF guideline using the embedded PDF outline.

Outputs:
- Flat TOC with validated page ranges
- Hierarchical TOC tree suitable for Graph RAG
"""

import fitz
import re
import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

# Logging configuration

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Regex for section IDs
SECTION_ID_RE = re.compile(r"^(\d+(?:\.\d+)*)")

class TOCSection(BaseModel):
    id: Optional[str]
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    type: str = "body"
    children: List["TOCSection"] = []

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError(
                f"Invalid page range: {self.page_start}–{self.page_end}"
            )
        return self


class TOCMetadata(BaseModel):
    doc_id: str
    toc_source: str = "pdf_outline"
    n_pages: int
    flat_toc: List[TOCSection]
    toc_tree: List[TOCSection]


# Needed for recursive models
TOCSection.model_rebuild()

class GuidelineTOCExtractor:
    """
    Extracts and normalizes the Table of Contents from a PDF guideline.
    """

    def __init__(self, pdf_path: str, doc_id: str):
        self.pdf_path = pdf_path
        self.doc_id = doc_id
        self.doc = fitz.open(pdf_path)
        self.n_pages = self.doc.page_count

        logger.info(
            "Initialized TOC extractor for %s (%d pages)",
            pdf_path, self.n_pages
        )

    # Extract raw TOC 

    def extract_raw_toc(self) -> List[dict]:
        raw_toc = self.doc.get_toc(simple=False)

        if not raw_toc:
            logger.error("No PDF outline found")
            raise ValueError("PDF does not contain a logical TOC (outline).")

        logger.info("Extracted %d TOC entries", len(raw_toc))

        sections = []

        for level, title, page, *_ in raw_toc:
            title = title.strip()
            match = SECTION_ID_RE.match(title)

            section_id = None
            clean_title = title

            if match:
                candidate_id = match.group(1)

                # Exclude year-like numeric prefixes (e.g. 2023)
                if not (candidate_id.isdigit() and 1900 <= int(candidate_id) <= 2100):
                    section_id = candidate_id
                    clean_title = title[len(candidate_id):].strip(" .")

            sections.append({
                "id": section_id,
                "title": clean_title,
                "level": level,
                "page_start": page,
                "page_end": None,
                "type": "body"
            })

        return sections

    # Compute safe page ranges

    def compute_page_ranges(self, sections: List[dict]) -> None:
        for i, sec in enumerate(sections):
            if i + 1 < len(sections):
                next_start = sections[i + 1]["page_start"]
                raw_end = next_start - 1
                sec["page_end"] = max(sec["page_start"], raw_end)

                if raw_end < sec["page_start"]:
                    logger.warning(
                        "Same-page sections detected at page %d (section %s)",
                        sec["page_start"], sec["id"]
                    )
            else:
                sec["page_end"] = self.n_pages

    # Classify sections 

    # Classify sections 

    def classify_sections(self, sections: List[dict]) -> None:
        for sec in sections:
            title = sec["title"].lower()

            # Treat document title (level 1 without section ID) as front matter
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

    def save(self, output_path: str) -> None:
        metadata = self.run()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        logger.info("TOC metadata saved to %s", output_path)


if __name__ == "__main__":
    extractor = GuidelineTOCExtractor(
        pdf_path="../../test_data/pdfdocs/Cardiomyopathies_2023.pdf",
        doc_id="Cardiomyopathies_2023"
    )
    toc_metadata = extractor.run()
    extractor.save("../../test_data/toc/Cardiomyopathies_2023_toc.json")
