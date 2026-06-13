import re
from typing import List, Tuple, Optional

from pydantic import BaseModel

HEADER_PATTERNS = [
    re.compile(r"(?m)^\s*(?P<num>#{1,6})\s+\*\*(?P<title>.+?)\*\*\s*$"),
    re.compile(r"(?m)^\s*(?P<num>#{1,6})\s+(?P<title>[^*\n].+?)\s*$"),
    re.compile(r"(?m)^\s*\*\*(?P<title>.+?)\*\*\s*$"),
]

BODY_START_PATTERNS = [
    re.compile(r"(?m)^\s*#{1,6}\s+\*\*1[\.\s].+?\*\*\s*$"),
    re.compile(r"(?m)^\s*#{1,6}\s+1[\.\s].+?\s*$"),
]

class BaseTocConfig(BaseModel):
    headings_numbered_only: bool = False
    section_id_re: Optional[re.Pattern] = None
    section_re: Optional[re.Pattern] = None
    section_id_in_title: bool = False
    toc_line: Optional[re.Pattern] = None
    bad_page_words: Optional[Tuple[str, ...]] = None
    prose_words: Optional[Tuple[str, ...]] = None
    bad_title_str_pattern: Optional[Tuple[str, ...]] = None
    bad_section_title_ends: Optional[Tuple[str, ...]] = None
    bad_title_patterns: Optional[Tuple[re.Pattern]] = None
    bad_section_title_starts: Tuple[str, ...] = None
    terminal_sections_starts: Tuple[str, ...] = None
    front_matter_section_starts: Optional[Tuple[str, ...]] = None
    back_matter_section_starts: Tuple[str, ...] = None
    supplement_patterns: Optional[Tuple[str, ...]] = None
    fallback_toc_sections: Optional[Tuple[str, ...]] = None
    heading_top_y_thresholds: int = 100
    excluded_title_keywords: Optional[Tuple[str, ...]] = None
    bad_doc_title_patterns: Optional[List[re.Pattern]] = None


class UpperGiTocConfig(BaseTocConfig):
    section_id_in_title: bool = False
    section_id_re: re.Pattern = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)")
    bad_doc_title_patterns: List[re.Pattern] = (
        re.compile(r"^\d"),                   # starts with digit
        re.compile(r"\d{2,}\.\.\d{2,}"),     # page range like "693..711"
        re.compile(r"^10\.\d{4}/"),          # DOI
        re.compile(r"^doi:", re.IGNORECASE),
        re.compile(r"^https?://"),           # URL
    )
    excluded_title_keywords: Optional[Tuple[str, ...]] = (
        "authors", "author", "institutions", "institution",
        # "bibliography", "references", "reference",
        "corresponding author", "competing interests",
        "conflict of interest", "acknowledgements", "acknowledgments",
        "abbreviations", "abbreviations:", "abbreviation",
        "funding", "disclosures", "disclosure",
        "appendix", "supplementary", "supplementary data",
        "main recommendations",
    )
    bad_title_patterns: Optional[List[re.Pattern]] = (
        re.compile(r"^\s*[!·•\-–—]\s*$"),           # lone punctuation
        re.compile(r"^Table\s*\d+", re.IGNORECASE),  # table captions
        re.compile(r"^Fig\.?\s*\d+", re.IGNORECASE), # figure captions
        re.compile(r"^\d{1,4}$"),                    # bare page numbers
    )
    terminal_sections_starts: Tuple[str, ...] = ("reference", ) # ("disclaimer", "acknowledgement", "appendi", "competing interest", "supplementary", "reference")
    back_matter_section_starts: Tuple[str, ...] = ("disclaimer", "acknowledgement", "appendi", "competing interest", "supplementary", "reference")
    front_matter_section_starts: Tuple[str, ...] = ("definition", "abbreviation")
    bad_section_title_ends: Tuple[str, ...] = (", md", ", md phd", ",md, mshs")
    bad_section_title_starts: Tuple[str, ...] = ("dr.", "published online")



class HepatologyProtocolsTocConfig(BaseTocConfig):
    bad_section_title_starts: Tuple[str, ...] = ("ehab", "tblfn", "|", "ehz", "ehy", "eh")  # TODO: update if needed
    terminal_sections_starts: Tuple[str, ...] = ("reference", "bibliograph")
    back_matter_section_starts: Tuple[str, ...] = ("acknowledg", "supplementar", "referece", "conflict of interest")
    section_id_in_title: bool = False
    excluded_title_keywords: Tuple[str, ...] = (
        "references", "acknowledgements", "conflict of interest", "supplementary data"
    )


class CardiologyProtocolsTocConfig(BaseTocConfig):
    section_id_re: re.Pattern = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
    section_re: re.Pattern = re.compile(
        r"^(?P<id>[0-9]{1,2}(?:\.\d+)*)(?:\.)?\s+(?P<title>.+)$"
    )
    section_id_in_title: bool = True
    toc_line: re.Pattern = re.compile(
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
    bad_page_words: Tuple[str, ...] = (
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
    prose_words: Tuple[str, ...] = ( # TODO: possibly remove this filter
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
    bad_title_str_pattern: Tuple[str, ...] = (
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
    bad_section_title_starts: Tuple[str, ...] = ("ehab", "tblfn", "|", "ehz", "ehy", "eh")
    terminal_sections_starts: Tuple[str, ...] = ("reference", "bibliograph")
    front_matter_section_starts: Tuple[str, ...] = (
        "table of contents",
        "contents",
        "acronyms",
        "abbreviation",
        "list of figures",
        "list of tables",
    )
    back_matter_section_starts: Tuple[str, ...] = (
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
    supplement_patterns: Tuple[str, ...] = (
        r"^table\b",
        r"^figure\b",
        r"^list of\b",
        r"^tables\b",
        r"^figures\b",
    )
    fallback_toc_sections: Tuple[str, ...] = ("tableofcontents", "contents")
    excluded_title_keywords: Tuple[str, ...] = ( # TODO: Exclusion rules (check if there are others to add)
        "table of contents", "list of figures", "list of tables",
        "references", "bibliography",
        "abbreviations", "acronyms",
        "acknowledgements", "acknowledgments",
        "appendix", "supplementary data",
        "data availability",
        "author information", "disclaimer",
        "index", "what to do and what not to", "preamble", "key messages", "gaps in evidence",
    )

class TocConfigFactory:
    toc_config_mapping = {
        "cardiology_protocols": CardiologyProtocolsTocConfig,
        "hepatology_protocols": HepatologyProtocolsTocConfig,
        "upper_gi_protocols": UpperGiTocConfig,
    }

    @classmethod
    def get_toc_config(cls, app_id: str = "cardiology_protocols") -> BaseTocConfig:
        return cls.toc_config_mapping[app_id]()