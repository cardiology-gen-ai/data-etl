"""
acronym_extractor.py

Extracts acronym-definition pairs from medical PDFs using multi-stage heuristics
to locate the acronyms section, parse definitions, and handle common extraction challenges.

Output: JSON file mapping acronym tokens to definitions, with metadata about source,
status, page range, and suspicious candidates.
"""

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

from managers.table_of_contents_manager import TOCMetadata


logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = ROOT_DIR / "test_data"

DEFAULT_PDF_DIR = TEST_DATA_DIR / "pdfdocs"
DEFAULT_TOC_DIR = TEST_DATA_DIR / "toc"
DEFAULT_ACRONYM_DIR = TEST_DATA_DIR / "acronyms"

ACRONYM_SAMPLE_SIZE = 25
PRINT_FULL_ACRONYMS = False

ACRONYM_FILENAME_SUFFIX = "_acronyms.json"
TOC_FILENAME_SUFFIX = "_toc.json"

ACRO_TITLE_LOOSE_RX = re.compile(
    r"(?i)\b(?:abbreviations?(?:\s+and\s+acronyms?)?|acronyms?)\b"
)

ACRO_TITLE_EXACT_RX = re.compile(
    r"(?i)^\s*(?:abbreviations?(?:\s+and\s+acronyms?)?|acronyms?)\s*$"
)

# Accept ASCII letters, Greek letters, digits, and common acronym symbols.
# Examples: %HRmax, 18F-FDG, 99mTc, CHA2DS2-VASc, b.p.m., P/LP, ATTRwt,
# mWHO, Lp(a), α-SMA, β-blocker, and E/e′.
SHORT_SHAPE_RX = re.compile(
    r"^(?=.*[A-Za-zΑ-Ωα-ω])"
    r"[%A-Za-z0-9Α-Ωα-ω]"
    r"[%A-Za-z0-9Α-Ωα-ω/\.\-+\(\)′']{0,31}$"
)

LINE_NOISE_RX = re.compile(
    r"(?i)(downloaded from|academic\.oup\.com|oup\.com|https?://|www\.|"
    r"\bdoi\b|permissions|copyright|eur heart j|esc guidelines|"
    r"for personal use only|by guest)"
)

FOOTER_NOISE_LINE_RX = re.compile(
    r"(?i)^(downloaded|from|by|guest|on|for personal use only|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december)$"
)

DATE_LINE_RX = re.compile(
    r"(?i)^\d{1,2}\s+[A-Z][a-z]{2,9}\s+\d{4}$"
)

BODY_HEADING_RX = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]")

PREAMBLE_HEADING_RX = re.compile(
    r"(?i)^\s*(?:\d+(?:\.\d+)*)?\.?\s*preamble\s*$"
)

TOC_TITLE_RX = re.compile(r"(?im)^\s*(contents|table of contents)\s*$")
TOC_DOTTED_LINE_RX = re.compile(r"(?m)^.*\.{2,}\s*\d+\s*$")
TOC_SECTION_LINE_RX = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z].{3,}\s+\d+\s*$"
)
TOC_DOT_LEADER_RX = re.compile(r"(?:\.\s*){2,}\s*\d+\s*$")

MONTH_NAMES_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
)

FOOTER_BLEED_START_RX = re.compile(
    r"(?i)(?:^|\s+)"
    r"(?:downloaded\s+from|academic\.oup\.com\S*|oup\.com\S*|"
    r"https?://\S+|www\.\S+|\bdoi\b|permissions\b|copyright\b|"
    r"for\s+personal\s+use\s+only\b|by\s+guest\b)"
    r".*$"
)

TRAILING_FOOTER_DATE_RX = re.compile(
    rf"(?i)\s+(?:by\s+guest\s+)?(?:on\s+)?\d{{1,2}}\s+{MONTH_NAMES_PATTERN}\s+\d{{4}}\s*$"
)

TRAILING_FOOTER_TOKENS = {
    "downloaded",
    "from",
    "by",
    "guest",
    "on",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
}

MIXED_CASE_SHORT_WHITELIST = {
    "Ach",
    "Tx",
    "Vmax",
    "mHealth",
}

CONTINUATION_CONNECTOR_ENDINGS = {
    "of", "or", "and", "for", "from", "the", "to", "with", "by", "in", "on",
    "a", "an", "via", "vs", "as", "at",
    "pulmonary", "stroke", "trial", "disease", "heart", "failure",
    "left", "right", "ventricular", "atrial",
    "reduced", "preserved", "mildly",
    "cardiology", "society", "college", "prevention", "treatment",
    "association", "american", "care", "approach", "fraction",
    "infection", "outcome", "study", "group",
    "syndrome", "classification", "volume", "ratio", "score",
}

MAX_FRONT_MATTER_SCAN_PAGES = 30
DENSITY_MIN_BEST_PAGE_SCORE = 8
DENSITY_NEIGHBOUR_SCORE = 3

# The first body page is sometimes detected too early because the acronym
# section itself can share a page with front-matter/body-like headings.
# Therefore, we search a few pages beyond the first body candidate when looking
# for the actual acronym heading.
HEADING_SEARCH_EXTRA_PAGES_AFTER_BODY_CANDIDATE = 4

# Once the acronym heading is found, do not use first_body_page as a hard cutoff
# unless it is strictly after the acronym heading. Instead, scan forward with a
# safe cap and let extract_lines_from_page_range() stop at the first real body
# heading, e.g. "1. Preamble".
MAX_ACRONYM_SECTION_PAGES_AFTER_START = 12


@dataclass
class PDFLine:
    page_no: int
    column: str
    y: float
    x0: float
    text: str


def configure_cli_logging(level: int = logging.INFO) -> None:
    """
    Configure logging only for standalone CLI execution.

    The graph pipeline should configure logging itself, so this function is not
    called at import time.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def get_default_toc_path(toc_dir: Path, doc_id: str) -> Path:
    return Path(toc_dir) / f"{doc_id}{TOC_FILENAME_SUFFIX}"


def get_default_acronym_path(acronym_dir: Path, doc_id: str) -> Path:
    return Path(acronym_dir) / f"{doc_id}{ACRONYM_FILENAME_SUFFIX}"


def load_optional_toc(toc_path: Optional[Path]) -> Optional[TOCMetadata]:
    if toc_path is None:
        return None

    toc_path = Path(toc_path)

    if not toc_path.exists():
        return None

    try:
        return TOCMetadata.model_validate(
            json.loads(toc_path.read_text(encoding="utf-8"))
        )
    except Exception as e:
        logger.warning("Failed to load/validate TOC %s: %s", toc_path.name, e)
        return None


def load_cached_acronym_payload(out_path: Path) -> Optional[Dict]:
    out_path = Path(out_path)

    if not out_path.exists():
        return None

    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read cached acronym payload %s: %s", out_path, e)
        return None

    if not isinstance(payload, dict):
        logger.warning("Cached acronym payload is not a JSON object: %s", out_path)
        return None

    return payload


def find_acronym_section_from_toc(toc_metadata: TOCMetadata):
    for sec in toc_metadata.flat_toc:
        title = (sec.title or "").strip()
        if ACRO_TITLE_LOOSE_RX.search(title):
            return sec
    return None


def toc_pages_look_usable(page_start: int, page_end: int, n_pages: int) -> bool:
    return 1 <= page_start <= n_pages and 1 <= page_end <= n_pages


def find_section_one_title(toc_metadata: TOCMetadata) -> Optional[str]:
    for sec in toc_metadata.flat_toc:
        if sec.id == "1":
            return sec.title
    return None


def normalize_unicode_text(text: str) -> str:
    """
    Normalize PDF-extracted Unicode text.

    Important fixes:
    - ligatures: ﬁ -> fi, ﬂ -> fl
    - soft hyphen removal
    - zero-width character removal
    - non-breaking spaces converted to normal spaces
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u00ad": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2007": " ",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def normalize_spaces(text: str) -> str:
    text = normalize_unicode_text(text)
    return re.sub(r"\s+", " ", text).strip()


def strip_footer_bleed(text: str) -> str:
    """
    Remove common PDF footer/header fragments when they bleed into useful lines.
    """
    text = normalize_spaces(text)

    if not text:
        return ""

    for _ in range(3):
        old = text

        text = FOOTER_BLEED_START_RX.sub("", text)
        text = TRAILING_FOOTER_DATE_RX.sub("", text)
        text = normalize_spaces(text)

        if text == old:
            break

    return text


def page_looks_like_toc(text: str) -> bool:
    text = normalize_unicode_text(text)

    dotted = len(TOC_DOTTED_LINE_RX.findall(text))
    section_like = len(TOC_SECTION_LINE_RX.findall(text))
    has_contents_title = bool(TOC_TITLE_RX.search(text))

    if has_contents_title:
        return True
    if dotted >= 3:
        return True
    if dotted >= 2 and section_like >= 4:
        return True

    return False


def is_toc_entry_line(line: str) -> bool:
    """
    Detect line-level TOC entries such as:
      Abbreviations and acronyms . . . . . . . 2923
      Preamble . . . . . . . . . . . . . . . . 2923
    """
    s = normalize_spaces(line)
    return bool(TOC_DOT_LEADER_RX.search(s))


def is_real_acronym_heading(line: str) -> bool:
    """
    True only for the actual acronym section heading, not a TOC entry.
    """
    s = normalize_spaces(line)
    if is_toc_entry_line(s):
        return False
    return bool(ACRO_TITLE_EXACT_RX.fullmatch(s))


def find_first_body_page(
    toc_metadata: Optional[TOCMetadata],
    doc: fitz.Document,
) -> Optional[int]:
    """
    Find the first real body page.

    Returns a 1-based PDF page number.

    Note:
    This is only a candidate. Some PDFs place the acronym section close to the
    first body heading, and page mapping can be imperfect. Do not use this as a
    hard acronym extraction cutoff unless it is strictly after the acronym start.
    """
    if toc_metadata is not None:
        for sec in toc_metadata.flat_toc:
            if sec.id == "1" and toc_pages_look_usable(
                sec.page_start,
                sec.page_end,
                doc.page_count,
            ):
                return sec.page_start

    sec1_title = find_section_one_title(toc_metadata) if toc_metadata is not None else None
    title_tokens = re.findall(r"\w+", sec1_title or "")
    title_pattern = None

    if title_tokens:
        title_tokens = title_tokens[:6]
        title_pattern = re.compile(
            r"\b" + r"\W+".join(re.escape(tok) for tok in title_tokens) + r"\b",
            re.IGNORECASE,
        )

    scan_limit = min(doc.page_count, MAX_FRONT_MATTER_SCAN_PAGES)

    for page_no in range(1, scan_limit + 1):
        text = doc.load_page(page_no - 1).get_text("text")
        text = normalize_unicode_text(text)

        if PREAMBLE_HEADING_RX.search(text):
            return page_no

        if page_looks_like_toc(text):
            continue

        if re.search(r"(?m)^\s*1(?:\.(?=\s)|\s+)", text):
            if title_pattern is None or title_pattern.search(text):
                return page_no

    generic_rx = re.compile(r"(?m)^\s*1(?:\.(?=\s)|\s+)[A-Z]")

    for page_no in range(1, scan_limit + 1):
        text = doc.load_page(page_no - 1).get_text("text")
        text = normalize_unicode_text(text)

        if page_looks_like_toc(text):
            continue
        if generic_rx.search(text):
            return page_no

    return None


def find_last_toc_page(doc: fitz.Document, first_body_page: Optional[int]) -> Optional[int]:
    """
    Heuristically detect the last TOC-like page directly from the PDF.
    """
    if first_body_page is not None:
        scan_limit = min(doc.page_count, max(1, first_body_page), 12)
    else:
        scan_limit = min(doc.page_count, 12)

    last_toc_page = None

    for page_no in range(1, scan_limit + 1):
        text = doc.load_page(page_no - 1).get_text("text")
        text = normalize_unicode_text(text)

        if page_looks_like_toc(text):
            last_toc_page = page_no

    return last_toc_page


def group_words_into_lines(words: List[Tuple], y_tol: float = 2.5) -> List[Tuple[float, float, str]]:
    """
    Group PyMuPDF word tuples into text lines.

    Each word tuple is typically:
      (x0, y0, x1, y1, text, block_no, line_no, word_no)
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w[1], w[0]))

    groups: List[List[Tuple]] = []
    current = [words[0]]
    current_y = float(words[0][1])

    for w in words[1:]:
        y = float(w[1])

        if abs(y - current_y) <= y_tol:
            current.append(w)
            current_y = (current_y * (len(current) - 1) + y) / len(current)
        else:
            groups.append(current)
            current = [w]
            current_y = y

    groups.append(current)

    lines: List[Tuple[float, float, str]] = []

    for group in groups:
        group = sorted(group, key=lambda w: w[0])
        y = min(float(w[1]) for w in group)
        x0 = min(float(w[0]) for w in group)
        line = normalize_spaces(" ".join(str(w[4]) for w in group))

        if line:
            lines.append((y, x0, line))

    return lines


def clean_candidate_line(line: str) -> Optional[str]:
    line = strip_footer_bleed(line)

    if not line:
        return None
    if LINE_NOISE_RX.search(line):
        return None
    if FOOTER_NOISE_LINE_RX.fullmatch(line):
        return None
    if DATE_LINE_RX.fullmatch(line):
        return None
    if re.fullmatch(r"\d{1,4}", line):
        return None
    if line.lower() in {"contents", "table of contents"}:
        return None

    return line


def extract_page_lines(
    page: fitz.Page,
    page_no: int,
    top_margin: float = 35.0,
    bottom_margin: float = 35.0,
) -> List[PDFLine]:
    """
    Extract lines from a page while preserving simple two-column reading order:
    left column top-to-bottom, then right column top-to-bottom.
    """
    page_rect = page.rect
    mid_x = page_rect.width / 2.0

    words = page.get_text("words")

    filtered = [
        w for w in words
        if top_margin <= float(w[1]) <= page_rect.height - bottom_margin
    ]

    left = [w for w in filtered if float(w[0]) < mid_x]
    right = [w for w in filtered if float(w[0]) >= mid_x]

    out: List[PDFLine] = []

    for column_name, column_words in [("left", left), ("right", right)]:
        column_lines = group_words_into_lines(column_words)

        for y, x0, raw_line in column_lines:
            line = clean_candidate_line(raw_line)
            if line:
                out.append(
                    PDFLine(
                        page_no=page_no,
                        column=column_name,
                        y=y,
                        x0=x0,
                        text=line,
                    )
                )

    return out


def is_acronym_heading(line: str) -> bool:
    return is_real_acronym_heading(line)


def is_body_heading(line: str) -> bool:
    """
    Detect true body/preamble headings, while avoiding TOC entries.
    """
    s = normalize_spaces(line)

    if is_toc_entry_line(s):
        return False

    if PREAMBLE_HEADING_RX.fullmatch(s):
        return True

    if BODY_HEADING_RX.match(s):
        return True

    return False


def clean_long_definition(long: str) -> str:
    """
    Clean long definition text and remove common trailing footer fragments.
    """
    long = strip_footer_bleed(long)
    long = long.strip(" -–:;,")

    for _ in range(8):
        parts = long.split()
        if not parts:
            break

        last = parts[-1].strip(".,;:()[]").lower()

        if last in TRAILING_FOOTER_TOKENS:
            long = " ".join(parts[:-1]).strip()
        else:
            break

    long = strip_footer_bleed(long)
    return long.strip(" -–:;")


def contains_greek_letter(text: str) -> bool:
    return any("Α" <= ch <= "Ω" or "α" <= ch <= "ω" for ch in text)


def looks_like_short(short: str) -> bool:
    """
    Accept conventional acronym-like strings, including:
    %HRmax, 18F-FDG, 99mTc, CHA2DS2-VASc, b.p.m., P/LP, ATTRwt, mWHO,
    Lp(a), α-SMA, β-blocker, and E/e′.

    This is intentionally strict with simple Titlecase or one-uppercase mixed
    words, otherwise normal PDF text or styled trial fragments can become
    false acronym keys.
    """
    s = normalize_spaces(short)

    if not SHORT_SHAPE_RX.fullmatch(s):
        return False

    if s.isdigit():
        return False

    if s in MIXED_CASE_SHORT_WHITELIST:
        return True

    n_upper = sum(ch.isupper() for ch in s)
    n_lower = sum(ch.islower() for ch in s)
    has_digit = any(ch.isdigit() for ch in s)
    has_symbol = any(ch in "/.-+()′'%" for ch in s)
    has_greek = contains_greek_letter(s)

    if n_upper >= 2:
        return True

    if n_upper >= 1 and n_lower == 0 and len(s) <= 8:
        return True

    if has_digit and (n_upper >= 1 or has_symbol or len(s) <= 8):
        return True

    if has_symbol and (n_upper >= 1 or "." in s or "%" in s):
        return True

    if has_greek and has_symbol and len(s) <= 32:
        return True

    if "." in s and len(s) <= 12:
        return True

    return False


def clean_short_token(token: str) -> str:
    """
    Clean the acronym token while preserving meaningful dots such as b.p.m.
    """
    token = normalize_spaces(token)
    token = token.strip("[],:;")
    return token


def looks_like_short_sequence(short: str) -> bool:
    """
    Accept normal single-token acronyms and conservative multi-token acronyms.

    Examples accepted:
    - SAVOR-TIMI 53
    - TIMI 50
    - PROVE IT-TIMI 22

    Multi-token forms are only accepted when at least one token after the first
    contains a digit. This prevents false parses like:
      ASCEND A Study ...
    becoming:
      short = "ASCEND A"
    """
    s = normalize_spaces(short)

    if not s:
        return False

    if " " not in s:
        return looks_like_short(s)

    parts = s.split()

    if not (2 <= len(parts) <= 4):
        return False

    if len(s) > 48:
        return False

    if not looks_like_short(parts[0]):
        return False

    has_numeric_suffix = False

    for part in parts[1:]:
        part = clean_short_token(part)

        if re.fullmatch(r"\d+[A-Za-z]?", part):
            has_numeric_suffix = True
            continue

        if looks_like_short(part):
            continue

        return False

    return has_numeric_suffix


def looks_like_standalone_short_line(line: str) -> bool:
    """
    Detect acronym tokens that appear alone on one line, with the definition
    on the next line.

    This is intentionally conservative and reuses looks_like_short().
    """
    s = clean_short_token(line)

    if not s:
        return False
    if " " in s:
        return False
    if len(s) > 32:
        return False

    return looks_like_short(s)


def parse_acronym_row(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse rows like:
      AF Atrial fibrillation
      BNP Brain natriuretic peptide
      CHA2DS2-VASc Congestive heart failure ...
      SAVOR-TIMI 53 Saxagliptin Assessment ...
      %HRmax Percentage of maximum heart rate
    """
    line = strip_footer_bleed(line)

    if not line:
        return None

    if line.lstrip().startswith("("):
        return None

    parts = line.split()

    if len(parts) < 2:
        return None

    max_short_tokens = min(4, len(parts) - 1)

    # Try the longest plausible acronym key first, then fall back to a normal
    # one-token acronym. This handles "SAVOR-TIMI 53 ..." without breaking
    # common rows like "ABC Atrial fibrillation Better Care".
    for n_tokens in range(max_short_tokens, 0, -1):
        raw_short = " ".join(parts[:n_tokens])
        raw_long = " ".join(parts[n_tokens:])

        short = clean_short_token(raw_short)
        long = clean_long_definition(raw_long)

        if not looks_like_short_sequence(short):
            continue

        if short.isdigit():
            continue

        if len(long) < 3:
            continue

        if LINE_NOISE_RX.search(long):
            continue

        return short, normalize_spaces(long)

    return None


def should_attach_continuation(
    line: PDFLine,
    current_long: str,
    prev_line: Optional[PDFLine],
) -> bool:
    """
    Heuristics for wrapped definition lines.
    """
    text = strip_footer_bleed(line.text)

    if not text or len(text) < 2:
        return False

    if LINE_NOISE_RX.search(text):
        return False

    if is_body_heading(text):
        return False

    if is_acronym_heading(text):
        return False

    parsed = parse_acronym_row(text)
    if parsed is not None:
        return False

    if looks_like_standalone_short_line(text):
        return False

    if text.lstrip().startswith("("):
        return True

    if prev_line is None:
        return False

    current_tail = current_long.rstrip()
    current_tail_words = current_tail.lower().split()
    last_prev = current_tail_words[-1].rstrip(" ,;:-–") if current_tail_words else ""

    if line.column == prev_line.column:
        if line.x0 >= prev_line.x0 + 10:
            return True

    if (
        current_tail.endswith(",")
        or current_tail.endswith("-")
        or current_tail.endswith("–")
        or current_tail.endswith("(")
    ):
        return True

    if last_prev in CONTINUATION_CONNECTOR_ENDINGS:
        return True

    if text[0].islower():
        return True

    if len(text.split()) <= 5 and len(text) <= 40:
        return True

    return False


def add_or_update_acronym(acronyms: Dict[str, str], short: str, long: str) -> None:
    """
    Keep the longer version if the same short form is seen more than once.

    In the current guideline corpus, duplicate acronym keys within the same
    document are expected to be rare. If duplicates become common, this should
    be extended to preserve variants instead of keeping only one definition.
    """
    short = clean_short_token(short)
    long = clean_long_definition(long)

    if not short or not long:
        return

    existing = acronyms.get(short)

    if existing is None or len(long) > len(existing):
        acronyms[short] = long


def extract_acronyms_from_lines(lines: List[PDFLine]) -> Dict[str, str]:
    acronyms: Dict[str, str] = {}

    current_short: Optional[str] = None
    pending_short: Optional[str] = None
    prev_line: Optional[PDFLine] = None

    for line in lines:
        text = strip_footer_bleed(line.text)

        if not text:
            prev_line = line
            continue

        parsed = parse_acronym_row(text)

        if parsed is not None:
            short, long = parsed
            add_or_update_acronym(acronyms, short, long)
            current_short = short
            pending_short = None
            prev_line = line
            continue

        if pending_short:
            if (
                not is_body_heading(text)
                and not is_acronym_heading(text)
                and not LINE_NOISE_RX.search(text)
                and not looks_like_standalone_short_line(text)
            ):
                long = clean_long_definition(text)
                if len(long) >= 3:
                    add_or_update_acronym(acronyms, pending_short, long)
                    current_short = pending_short
                    pending_short = None
                    prev_line = line
                    continue

            pending_short = None

        if looks_like_standalone_short_line(text):
            pending_short = clean_short_token(text)
            current_short = None
            prev_line = line
            continue

        if current_short and should_attach_continuation(
            line=line,
            current_long=acronyms[current_short],
            prev_line=prev_line,
        ):
            merged = clean_long_definition(acronyms[current_short] + " " + text)
            acronyms[current_short] = merged
            prev_line = line
            continue

        current_short = None
        prev_line = line

    return acronyms


def definition_looks_suspicious(short: str, long: str) -> bool:
    """
    Flag extracted definitions that may need manual inspection.

    This does not delete or modify acronyms. It only marks likely extraction
    problems, such as truncated definitions or footer fragments.
    """
    long_norm = normalize_spaces(long)

    if not long_norm:
        return True

    words = long_norm.split()
    last = words[-1].strip(".,;:()[]").lower() if words else ""

    strong_dangling_endings = {
        "of", "or", "and", "for", "from", "the", "to", "with", "without",
        "by", "in", "on", "via", "vs", "versus", "as", "at", "after",
        "following", "followed", "who", "that", "which", "ischemic",
        "ischaemic", "clinical", "patients", "patient", "eligible",
        "halt",
    }

    if last in strong_dangling_endings:
        return True

    if re.search(r"\s(?:0?[1-9]|[12][0-9]|3[01])$", long_norm):
        if any(ch.isdigit() for ch in short):
            return False

        combined = f"{short} {long_norm}"

        if re.search(
            r"(?i)\b(?:type|subunit|factor|kinase|transporter|convertase|"
            r"plakophilin|receptor|channel|protein|gene|score|trial|study|"
            r"registry|pregnancy)\s+\d+$",
            combined,
        ):
            return False

        return True

    return False


def collect_suspicious_acronyms(acronyms: Dict[str, str]) -> List[Dict[str, str]]:
    suspicious: List[Dict[str, str]] = []

    for short, long in sorted(acronyms.items()):
        if definition_looks_suspicious(short, long):
            suspicious.append(
                {
                    "short": short,
                    "definition": long,
                    "reason": "definition_may_be_truncated_or_contain_pdf_noise",
                }
            )

    return suspicious


def warn_on_suspicious_acronyms(doc_id: str, acronyms: Dict[str, str]) -> None:
    suspicious = collect_suspicious_acronyms(acronyms)

    if not suspicious:
        return

    logger.warning(
        "%s has %d acronym definitions that may need review",
        doc_id,
        len(suspicious),
    )

    for item in suspicious:
        logger.warning(
            "Suspicious acronym candidate: %s -> %s",
            item["short"],
            item["definition"],
        )


def score_page_for_acronyms(doc: fitz.Document, page_no: int) -> int:
    lines = extract_page_lines(doc.load_page(page_no - 1), page_no)
    return sum(1 for line in lines if parse_acronym_row(line.text) is not None)


def find_dense_acronym_span(
    doc: fitz.Document,
    page_start: int,
    page_end: int,
) -> Optional[Tuple[int, int]]:
    """
    Fallback when heading detection is insufficient.

    Find the page with the highest acronym-row density, then expand to
    adjacent pages that still show some acronym-row evidence.
    """
    if page_start > page_end:
        return None

    scores = {
        page_no: score_page_for_acronyms(doc, page_no)
        for page_no in range(page_start, page_end + 1)
    }

    if not scores:
        return None

    best_page, best_score = max(scores.items(), key=lambda kv: kv[1])

    logger.info("Acronym density scores in search range: %s", scores)

    if best_score < DENSITY_MIN_BEST_PAGE_SCORE:
        return None

    start_page = best_page
    end_page = best_page

    while (
        start_page - 1 >= page_start
        and scores.get(start_page - 1, 0) >= DENSITY_NEIGHBOUR_SCORE
    ):
        start_page -= 1

    while (
        end_page + 1 <= page_end
        and scores.get(end_page + 1, 0) >= DENSITY_NEIGHBOUR_SCORE
    ):
        end_page += 1

    return start_page, end_page


def get_acronym_extraction_end_page(
    doc: fitz.Document,
    acronym_start_page: int,
    first_body_page: Optional[int],
) -> int:
    """
    Return a safe provisional end page for acronym extraction.

    Important:
    - first_body_page is only trusted if it is strictly after the acronym start.
    - If first_body_page equals the acronym page, it may be a TOC/page-mapping
      mistake, so we do not use it as a hard cutoff.
    - extract_lines_from_page_range() will still stop early when it sees
      a real body heading such as "1. Preamble".
    """
    hard_end = min(
        doc.page_count,
        acronym_start_page + MAX_ACRONYM_SECTION_PAGES_AFTER_START,
    )

    if first_body_page is not None and first_body_page > acronym_start_page:
        hard_end = min(hard_end, first_body_page)

    return hard_end


def find_acronym_window(
    toc_metadata: Optional[TOCMetadata],
    doc: fitz.Document,
) -> Optional[Tuple[int, int, str]]:
    """
    Return (start_page, end_page, source), all 1-based inclusive.

    Key idea:
    - first_body_page is useful for narrowing the search for the acronym heading,
      but it must not be used as a hard extraction boundary.
    - Once the real acronym heading is found, we scan forward with a safe cap.
      extract_lines_from_page_range() then stops at the first real body heading.
    """
    first_body_page = find_first_body_page(toc_metadata, doc)

    if first_body_page is None:
        logger.warning("Could not determine first body page")
        heading_search_end = min(doc.page_count, MAX_FRONT_MATTER_SCAN_PAGES)
    else:
        logger.info("First body page candidate detected: %d", first_body_page)
        heading_search_end = min(
            doc.page_count,
            MAX_FRONT_MATTER_SCAN_PAGES,
            first_body_page + HEADING_SEARCH_EXTRA_PAGES_AFTER_BODY_CANDIDATE,
        )

    last_toc_page = find_last_toc_page(doc, first_body_page)

    if last_toc_page is not None:
        logger.info("Last TOC-like page detected: %d", last_toc_page)
        search_start = last_toc_page
    else:
        logger.info("No TOC-like PDF pages detected")
        search_start = 1

    if search_start > heading_search_end:
        logger.warning(
            "Search range collapsed (start=%d, end=%d). Falling back to 1-%d",
            search_start,
            heading_search_end,
            heading_search_end,
        )
        search_start = 1

    logger.info(
        "Searching acronym heading/window in pages %d-%d",
        search_start,
        heading_search_end,
    )

    for page_no in range(search_start, heading_search_end + 1):
        page_lines = extract_page_lines(doc.load_page(page_no - 1), page_no)

        for line in page_lines:
            if is_real_acronym_heading(line.text):
                extraction_end = get_acronym_extraction_end_page(
                    doc=doc,
                    acronym_start_page=page_no,
                    first_body_page=first_body_page,
                )

                logger.info(
                    "Found real acronym heading on page %d: %r. "
                    "Extraction window expanded to pages %d-%d",
                    page_no,
                    line.text,
                    page_no,
                    extraction_end,
                )

                return (
                    page_no,
                    extraction_end,
                    "pdf_real_heading_after_toc_scan_until_body",
                )

    if toc_metadata is not None:
        acro_sec = find_acronym_section_from_toc(toc_metadata)

        if acro_sec and toc_pages_look_usable(
            acro_sec.page_start,
            acro_sec.page_end,
            doc.page_count,
        ):
            start_page = max(search_start, acro_sec.page_start)
            end_page = get_acronym_extraction_end_page(
                doc=doc,
                acronym_start_page=start_page,
                first_body_page=first_body_page,
            )

            if start_page <= end_page:
                logger.info(
                    "Falling back to TOC acronym section, expanded to pages %d-%d",
                    start_page,
                    end_page,
                )
                return start_page, end_page, "toc_acronym_section_expanded_fallback"

    dense_span = find_dense_acronym_span(doc, search_start, heading_search_end)

    if dense_span is not None:
        return dense_span[0], dense_span[1], "density_scan_after_toc_up_to_body"

    return None


def extract_lines_from_page_range(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
) -> List[PDFLine]:
    """
    Extract candidate lines from a PDF page range, 1-based inclusive.

    If a real acronym heading is present, everything before that heading is
    dropped. Extraction stops before the first real body/preamble heading.
    """
    page_lines_by_no: Dict[int, List[PDFLine]] = {}
    heading_location: Optional[Tuple[int, int]] = None

    for page_no in range(start_page, end_page + 1):
        page = doc.load_page(page_no - 1)
        page_lines = extract_page_lines(page, page_no)
        page_lines_by_no[page_no] = page_lines

        if heading_location is None:
            for i, line in enumerate(page_lines):
                if is_real_acronym_heading(line.text):
                    heading_location = (page_no, i)
                    break

    lines: List[PDFLine] = []

    for page_no in range(start_page, end_page + 1):
        page_lines = page_lines_by_no.get(page_no, [])

        if heading_location is not None:
            heading_page, heading_idx = heading_location

            if page_no < heading_page:
                continue

            if page_no == heading_page:
                page_lines = page_lines[heading_idx + 1:]

        body_idx = None

        for i, line in enumerate(page_lines):
            if is_body_heading(line.text):
                body_idx = i
                break

        if body_idx is not None:
            lines.extend(page_lines[:body_idx])
            break

        lines.extend(page_lines)

    return lines


def build_output_payload(
    doc_id: str,
    status: str,
    source: str,
    page_start: Optional[int],
    page_end: Optional[int],
    acronyms: Dict[str, str],
) -> Dict:
    suspicious = collect_suspicious_acronyms(acronyms)

    return {
        "doc_id": doc_id,
        "status": status,
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "n_acronyms": len(acronyms),
        "n_suspicious": len(suspicious),
        "suspicious": suspicious,
        "acronyms": dict(sorted(acronyms.items())),
    }


def write_output_payload(out_path: Path, payload: Dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_acronyms(
    doc_id: str,
    acronyms: Dict[str, str],
    limit: int = ACRONYM_SAMPLE_SIZE,
    print_all: bool = PRINT_FULL_ACRONYMS,
) -> None:
    total = len(acronyms)
    sorted_items = sorted(acronyms.items())

    if print_all:
        items_to_print = sorted_items
    else:
        items_to_print = sorted_items[:max(0, limit)]

    print(f"\n=== {doc_id}: showing {len(items_to_print)} of {total} acronyms ===\n")

    for short, long in items_to_print:
        print(f"{short} -> {long}")

    remaining = total - len(items_to_print)
    if remaining > 0:
        print(f"... {remaining} more saved in JSON")

    print()


def extract_acronym_payload_from_pdf(
    pdf_path: Path,
    doc_id: Optional[str] = None,
    toc_path: Optional[Path] = None,
) -> Dict:
    """
    Extract acronym payload from a single PDF without handling cache logic.

    This is the core extraction function used by both the CLI and the graph
    preprocessing pipeline wrapper.
    """
    pdf_path = Path(pdf_path)
    doc_id = doc_id or pdf_path.stem

    logger.info("Starting acronym extraction for %s", doc_id)

    toc_metadata = load_optional_toc(toc_path)

    if toc_metadata is None:
        logger.info("Proceeding without TOC support for %s", doc_id)
    else:
        logger.info("TOC loaded successfully for %s", doc_id)

    doc = fitz.open(pdf_path)

    try:
        logger.info("Opened PDF %s with %d pages", pdf_path.name, doc.page_count)

        window = find_acronym_window(toc_metadata, doc)

        if window is None:
            logger.warning("Could not locate acronym section/window for %s", doc_id)
            return build_output_payload(
                doc_id=doc_id,
                status="not_found",
                source="not_found",
                page_start=None,
                page_end=None,
                acronyms={},
            )

        start_page, end_page, source = window

        logger.info(
            "Using acronym window for %s: pages %d-%d (source=%s)",
            doc_id,
            start_page,
            end_page,
            source,
        )

        candidate_lines = extract_lines_from_page_range(
            doc=doc,
            start_page=start_page,
            end_page=end_page,
        )

        if not candidate_lines:
            logger.warning("No candidate lines found in acronym window for %s", doc_id)
            return build_output_payload(
                doc_id=doc_id,
                status="empty_window",
                source=source,
                page_start=start_page,
                page_end=end_page,
                acronyms={},
            )

        logger.info("Collected %d candidate lines for %s", len(candidate_lines), doc_id)

        acronyms = extract_acronyms_from_lines(candidate_lines)

        if not acronyms:
            logger.warning("No acronyms extracted for %s", doc_id)
            return build_output_payload(
                doc_id=doc_id,
                status="parse_empty",
                source=source,
                page_start=start_page,
                page_end=end_page,
                acronyms={},
            )

        payload = build_output_payload(
            doc_id=doc_id,
            status="success",
            source=source,
            page_start=start_page,
            page_end=end_page,
            acronyms=acronyms,
        )

        warn_on_suspicious_acronyms(doc_id, acronyms)
        return payload

    finally:
        doc.close()


def load_or_extract_acronyms(
    pdf_path: Path,
    doc_id: Optional[str] = None,
    toc_path: Optional[Path] = None,
    acronym_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
    force: bool = False,
    write_output: bool = True,
    sample_size: int = 0,
    print_all: bool = False,
) -> Dict:
    """
    Pipeline-friendly cache wrapper.

    Intended call site in build_graph.py preprocessing:

        load_or_extract_acronyms(
            pdf_path=pdf_path,
            doc_id=doc_id,
            toc_path=config.toc_dir / f"{doc_id}_toc.json",
            acronym_dir=config.acronym_dir,
            force=config.force_acronyms,
        )

    If the output JSON already exists and force=False, the cached payload is
    returned. Otherwise, extraction is run and the JSON artifact is written.
    """
    pdf_path = Path(pdf_path)
    doc_id = doc_id or pdf_path.stem

    if out_path is None:
        if acronym_dir is None:
            acronym_dir = DEFAULT_ACRONYM_DIR
        out_path = get_default_acronym_path(Path(acronym_dir), doc_id)
    else:
        out_path = Path(out_path)

    if not force:
        cached = load_cached_acronym_payload(out_path)
        if cached is not None:
            logger.info("Using cached acronym output for %s: %s", doc_id, out_path)
            return cached

    payload = extract_acronym_payload_from_pdf(
        pdf_path=pdf_path,
        doc_id=doc_id,
        toc_path=toc_path,
    )

    if write_output:
        write_output_payload(out_path, payload)
        logger.info(
            "Saved acronym output for %s to %s (status=%s, n_acronyms=%d)",
            doc_id,
            out_path,
            payload.get("status"),
            payload.get("n_acronyms", 0),
        )

    acronyms = payload.get("acronyms", {})
    if isinstance(acronyms, dict) and (sample_size > 0 or print_all):
        print_acronyms(
            doc_id=doc_id,
            acronyms=acronyms,
            limit=sample_size,
            print_all=print_all,
        )

    return payload


def run_acronym_extraction_for_pdf(
    pdf_path: Path,
    toc_dir: Path = DEFAULT_TOC_DIR,
    acronym_dir: Path = DEFAULT_ACRONYM_DIR,
    sample_size: int = ACRONYM_SAMPLE_SIZE,
    print_all: bool = PRINT_FULL_ACRONYMS,
    force: bool = False,
) -> Dict:
    """
    Backward-compatible single-PDF runner for manual tests / CLI use.
    """
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem
    toc_path = get_default_toc_path(Path(toc_dir), doc_id)

    return load_or_extract_acronyms(
        pdf_path=pdf_path,
        doc_id=doc_id,
        toc_path=toc_path,
        acronym_dir=Path(acronym_dir),
        force=force,
        write_output=True,
        sample_size=sample_size,
        print_all=print_all,
    )


def run_acronym_extraction(
    sample_size: int = ACRONYM_SAMPLE_SIZE,
    print_all: bool = PRINT_FULL_ACRONYMS,
    pdf_filter: Optional[str] = None,
    pdf_dir: Path = DEFAULT_PDF_DIR,
    toc_dir: Path = DEFAULT_TOC_DIR,
    acronym_dir: Path = DEFAULT_ACRONYM_DIR,
    force: bool = False,
) -> None:
    pdf_dir = Path(pdf_dir)
    toc_dir = Path(toc_dir)
    acronym_dir = Path(acronym_dir)

    if not pdf_dir.exists():
        logger.error("PDF directory not found: %s", pdf_dir)
        return

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if pdf_filter:
        needle = pdf_filter.lower()
        pdf_paths = [
            path for path in pdf_paths
            if needle in path.name.lower() or needle in path.stem.lower()
        ]

    if not pdf_paths:
        if pdf_filter:
            logger.warning(
                "No PDF files matching filter %r found in %s",
                pdf_filter,
                pdf_dir,
            )
        else:
            logger.warning("No PDF files found in %s", pdf_dir)
        return

    logger.info("Found %d PDF files in %s", len(pdf_paths), pdf_dir)

    for pdf_path in pdf_paths:
        try:
            run_acronym_extraction_for_pdf(
                pdf_path=pdf_path,
                toc_dir=toc_dir,
                acronym_dir=acronym_dir,
                sample_size=sample_size,
                print_all=print_all,
                force=force,
            )
        except Exception as e:
            logger.exception("Failed on %s: %s", pdf_path.name, e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract acronym-definition pairs from guideline PDFs."
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=ACRONYM_SAMPLE_SIZE,
        help="Number of acronyms to print per PDF. Use 0 to suppress printing.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all extracted acronyms instead of a sample.",
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Optional substring filter to process only matching PDF filenames.",
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing input PDFs. Default: {DEFAULT_PDF_DIR}",
    )

    parser.add_argument(
        "--toc-dir",
        type=Path,
        default=DEFAULT_TOC_DIR,
        help=f"Directory containing cached TOC JSON files. Default: {DEFAULT_TOC_DIR}",
    )

    parser.add_argument(
        "--acronym-dir",
        type=Path,
        default=DEFAULT_ACRONYM_DIR,
        help=f"Directory where acronym JSON files are written. Default: {DEFAULT_ACRONYM_DIR}",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract acronyms even if a cached JSON output already exists.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    configure_cli_logging()
    args = parse_args()

    run_acronym_extraction(
        sample_size=args.sample,
        print_all=args.all,
        pdf_filter=args.pdf,
        pdf_dir=args.pdf_dir,
        toc_dir=args.toc_dir,
        acronym_dir=args.acronym_dir,
        force=args.force,
    )