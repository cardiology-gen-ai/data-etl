"""
acronym_extractor.py

Extracts acronym-definition pairs from medical PDFs using multi-stage heuristics
to locate the acronyms section, parse definitions, and handle common extraction challenges.

Output: JSON file mapping acronym tokens to definitions, with metadata about source,
status, page range, suspicious candidates, and non-destructive post-validation notes.

Important design choice:
- post-validation is diagnostic only. It may flag possible embedded rows or suspicious
  parser artefacts, but it does not split, delete, or shorten extracted acronym entries.
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
    r"(?i)\b(?:(?:list\s+of\s+)?abbreviations?"
    r"(?:\s+and\s+acronyms?)?|acronyms?)\b"
)

ACRO_TITLE_EXACT_RX = re.compile(
    r"(?i)^\s*(?:(?:list\s+of\s+)?abbreviations?"
    r"(?:\s+and\s+acronyms?)?|acronyms?)\s*$"
)

# Accept ASCII letters, Greek letters, digits, and common acronym symbols.
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

# Generic false-positive words that may appear inside definitions but should
# not be interpreted as short forms just because they are short words.
#
# Important:
# - all-caps forms are NOT globally blocked here, because some real acronyms
#   are identical to ordinary English words in uppercase.
# - ambiguous all-caps connector-like tokens are handled by a stricter
#   acronym-definition initial-alignment check during row splitting.
SHORT_FALSE_POSITIVE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "vs",
    "versus",
    "with",
    "without",
    "after",
    "before",
    "between",
    "during",
    "following",
    "who",
    "whom",
    "whose",
    "which",
    "that",
    "this",
    "these",
    "those",
    "have",
    "has",
    "had",
    "having",
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "care",
    "case",
    "class",
    "data",
    "drug",
    "event",
    "events",
    "group",
    "heart",
    "left",
    "level",
    "main",
    "major",
    "method",
    "model",
    "patient",
    "patients",
    "risk",
    "right",
    "score",
    "study",
    "trial",
    "type",
    "valve",
    "year",
    "years",
    "age",
    "sex",
    "male",
    "female",
    "may",
    "june",
    "july",
}

# Generic shape helpers for mixed-case biomedical acronyms.
VARIABLE_SUFFIXES = {"max", "min", "mean"}

# Ordinary English connectors that are especially dangerous when they appear
# as all-caps candidates inside a definition. They can still be accepted as
# real row-start acronyms if their definition initials align with the short form.
AMBIGUOUS_CONNECTOR_SHORT_WORDS = {
    "as",
    "at",
    "by",
    "in",
    "of",
    "on",
    "or",
    "to",
    "vs",
}

CONTINUATION_CONNECTOR_ENDINGS = {
    "of",
    "or",
    "and",
    "for",
    "from",
    "the",
    "to",
    "with",
    "without",
    "by",
    "in",
    "on",
    "a",
    "an",
    "via",
    "vs",
    "versus",
    "as",
    "at",
    "after",
    "before",
    "between",
    "during",
    "following",
    "followed",
    "who",
    "that",
    "which",
    "pulmonary",
    "stroke",
    "trial",
    "disease",
    "heart",
    "failure",
    "left",
    "right",
    "ventricular",
    "atrial",
    "reduced",
    "preserved",
    "mildly",
    "cardiology",
    "society",
    "college",
    "prevention",
    "treatment",
    "association",
    "american",
    "care",
    "approach",
    "fraction",
    "infection",
    "outcome",
    "study",
    "group",
    "syndrome",
    "classification",
    "volume",
    "ratio",
    "score",
    "ischemic",
    "ischaemic",
    "clinical",
    "patients",
    "patient",
    "eligible",
    "type",
}

# Used only in post-validation embedded-row detection.
POST_VALIDATION_SPLIT_BLOCKING_ENDINGS = {
    "of",
    "or",
    "and",
    "for",
    "from",
    "the",
    "to",
    "with",
    "without",
    "by",
    "in",
    "on",
    "a",
    "an",
    "via",
    "vs",
    "versus",
    "as",
    "at",
    "after",
    "before",
    "between",
    "during",
    "following",
    "followed",
    "who",
    "that",
    "which",
}

MAX_FRONT_MATTER_SCAN_PAGES = 30
DENSITY_MIN_BEST_PAGE_SCORE = 8
DENSITY_NEIGHBOUR_SCORE = 3
HEADING_SEARCH_EXTRA_PAGES_AFTER_BODY_CANDIDATE = 4
MAX_ACRONYM_SECTION_PAGES_AFTER_START = 12


@dataclass
class PDFLine:
    page_no: int
    column: str
    y: float
    x0: float
    text: str


def configure_cli_logging(level: int = logging.INFO) -> None:
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
    s = normalize_spaces(line)
    return bool(TOC_DOT_LEADER_RX.search(s))


def is_real_acronym_heading(line: str) -> bool:
    s = normalize_spaces(line)
    if is_toc_entry_line(s):
        return False
    return bool(ACRO_TITLE_EXACT_RX.fullmatch(s))


def find_first_body_page(
    toc_metadata: Optional[TOCMetadata],
    doc: fitz.Document,
) -> Optional[int]:
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
    s = normalize_spaces(line)

    if is_toc_entry_line(s):
        return False

    if PREAMBLE_HEADING_RX.fullmatch(s):
        return True

    if BODY_HEADING_RX.match(s):
        return True

    return False


def clean_long_definition(long: str) -> str:
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


def clean_short_token(token: str) -> str:
    token = normalize_spaces(token)
    token = token.strip("[],:;")

    while len(token) > 1 and token[-1] in {"-", "–"}:
        token = token[:-1].strip()

    return token


def normalized_false_positive_key(token: str) -> str:
    token = clean_short_token(token).lower()
    token = token.strip(".()[]{}:;,′'")
    token = token.replace("/", "")
    token = token.replace("-", "")
    token = token.replace("–", "")
    return token


def is_plain_titlecase_word(token: str) -> bool:
    token = clean_short_token(token)
    return bool(re.fullmatch(r"[A-Z][a-z]+", token))


def is_all_caps_like(token: str) -> bool:
    token = clean_short_token(token)
    if not token:
        return False

    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False

    return any(ch.isupper() for ch in letters) and not any(ch.islower() for ch in letters)


def is_false_positive_short_word(token: str) -> bool:
    token = clean_short_token(token)
    if not token:
        return True

    if is_all_caps_like(token):
        return False

    return normalized_false_positive_key(token) in SHORT_FALSE_POSITIVE_WORDS


def has_lowercase_hyphen_segment(token: str) -> bool:
    token = clean_short_token(token)
    pieces = re.split(r"\s+", token)

    for piece in pieces:
        if "-" not in piece and "–" not in piece:
            continue
        for segment in re.split(r"[-–]", piece):
            if any(ch.islower() for ch in segment):
                return True

    return False


def has_mixed_case_hyphen_acronym_marker(token: str) -> bool:
    """
    Detect biomedical mixed-case hyphenated short forms that are not ordinary
    hyphenated phrases.
    """
    token = clean_short_token(token)

    if "-" not in token and "–" not in token:
        return False

    compact = re.sub(r"[-–]", "", token)

    if not compact:
        return False

    n_upper = sum(ch.isupper() for ch in compact)
    n_lower = sum(ch.islower() for ch in compact)

    if n_upper < 2 or n_lower == 0:
        return False

    if compact[0].islower() and any(ch.isupper() for ch in compact[1:]):
        return True

    for segment in re.split(r"[-–]", token):
        if sum(ch.isupper() for ch in segment) >= 2 and any(ch.islower() for ch in segment):
            return True

    return False


def has_strong_acronym_marker(token: str) -> bool:
    token = clean_short_token(token)

    if any(ch.isdigit() for ch in token):
        return True

    if contains_greek_letter(token):
        return True

    if any(ch in "/.()′'%" for ch in token):
        return True

    if has_mixed_case_hyphen_acronym_marker(token):
        return True

    if re.search(r"[A-ZΑ-Ω]{2,}", token):
        return True

    return False


def is_slash_titlecase_phrase_fragment(token: str) -> bool:
    """
    Reject slash-separated phrase fragments while keeping compact slash acronyms.
    """
    token = clean_short_token(token)

    if "/" not in token:
        return False

    parts = [p for p in token.split("/") if p]
    if len(parts) < 2:
        return False

    for part in parts:
        part_clean = part.strip(".,;:()[]{}")

        if re.fullmatch(r"[A-Z]{2,}", part_clean):
            continue

        if re.fullmatch(r"[A-Z][a-z]?", part_clean):
            continue

        if re.fullmatch(r"[A-Z][a-z]{2,}", part_clean):
            return True

        if re.search(r"[a-z]{3,}", part_clean):
            return True

    return False


def is_ordinary_lowercase_hyphen_digit_fragment(token: str) -> bool:
    """
    Reject ordinary lowercase hyphen+digit phrase fragments.

    This prevents false splits inside definitions such as:
      <short> -> phrase-like prefix
      peptide-1 -> remaining definition

    while keeping real acronym-like digit forms.
    """
    token = clean_short_token(token)

    if not token:
        return False

    if token.startswith("e-"):
        return False

    return bool(re.fullmatch(r"[a-z]{3,}-\d+[a-z]?", token))


def is_upper_prefix_lowercase_hyphen_phrase(token: str) -> bool:
    token = clean_short_token(token)

    if not token or any(ch.isdigit() for ch in token):
        return False

    if contains_greek_letter(token):
        return False

    if has_mixed_case_hyphen_acronym_marker(token):
        return False

    pieces = re.split(r"\s+", token)
    if len(pieces) != 1:
        return False

    parts = re.split(r"[-–]", pieces[0])
    if len(parts) < 2:
        return False

    first = parts[0]
    rest = parts[1:]

    if not re.fullmatch(r"[A-Z]{2,}", first):
        return False

    return all(re.fullmatch(r"[a-z][a-z]+", part or "") for part in rest)


def is_hyphenated_phrase_fragment(token: str) -> bool:
    token = clean_short_token(token)

    if not has_lowercase_hyphen_segment(token):
        return False

    return not has_strong_acronym_marker(token)


def definition_starts_with_lowercase_word(long: str) -> bool:
    long = clean_long_definition(long)
    if not long:
        return False

    first = long.split()[0].strip(".,;:()[]{}")
    if not first:
        return False

    return first[0].islower()


def definition_starts_like_glossary_definition(long: str) -> bool:
    long = clean_long_definition(long)
    if not long:
        return False

    first = long.split()[0].strip(".,;:()[]{}")
    if not first:
        return False

    return first[0].isupper() or first[0].isdigit() or contains_greek_letter(first)


def definition_initials(long: str) -> str:
    """
    Return rough initial letters for a definition.

    Hyphens and slashes are treated as token boundaries. This is intentionally
    approximate; it is only used to disambiguate ordinary connector-like
    all-caps candidates during row splitting.
    """
    long = clean_long_definition(long)
    if not long:
        return ""

    tokens = re.findall(r"[A-Za-zΑ-Ωα-ω]+", long)
    return "".join(tok[0].upper() for tok in tokens if tok)


def short_alpha_letters(short: str) -> str:
    short = clean_short_token(short)
    return "".join(ch.upper() for ch in short if ch.isalpha())


def is_subsequence(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False

    pos = 0
    for ch in haystack:
        if pos < len(needle) and needle[pos] == ch:
            pos += 1
        if pos == len(needle):
            return True

    return False


def acronym_letters_match_definition(short: str, long: str) -> bool:
    """
    Generic acronym-definition alignment test.

    The short-form letters must appear in order among the initials of the
    definition. Subsequence matching allows cases such as:
      AS -> Aortic valve stenosis
    """
    letters = short_alpha_letters(short)
    initials = definition_initials(long)

    if len(letters) < 2 or not initials:
        return False

    return is_subsequence(letters, initials)


def is_ambiguous_connector_like_all_caps_short(token: str) -> bool:
    """
    Detect all-caps short candidates that are also common English connectors.

    These are not rejected globally because they can be real acronyms at row
    start. Instead, when they appear as an embedded split candidate, they must
    pass acronym-definition initial alignment.
    """
    token = clean_short_token(token)

    if not is_all_caps_like(token):
        return False

    if any(ch.isdigit() for ch in token):
        return False

    if any(ch in "/.()+′'%" for ch in token):
        return False

    letters = short_alpha_letters(token)
    if not (2 <= len(letters) <= 3):
        return False

    return normalized_false_positive_key(token) in AMBIGUOUS_CONNECTOR_SHORT_WORDS


def is_lowercase_hyphen_short(token: str) -> bool:
    token = clean_short_token(token)

    if is_false_positive_short_word(token):
        return False

    if len(token) > 32:
        return False

    return bool(re.fullmatch(r"e-[a-z0-9]+(?:-[a-z0-9]+)*\.?,?", token))


def is_probable_stylized_word_fragment(token: str) -> bool:
    """
    Detect stylized word/backronym fragments that often occur inside study,
    trial, or title names.

    This is conservative diagnostic logic. It should not be used to reject
    row-start acronym labels globally, because real acronyms can also be
    mixed-case.
    """
    token = clean_short_token(token)

    if not token:
        return False

    if contains_greek_letter(token):
        return False

    if "." in token or "%" in token or "′" in token or "'" in token:
        return False

    if is_slash_titlecase_phrase_fragment(token):
        return True

    if "/" in token:
        slash_parts = [p for p in re.split(r"/", token) if p]
        if any(re.search(r"[a-z]", p) and len(p) >= 6 for p in slash_parts):
            return True

    # Enzyme-like biomedical forms should not be treated as parser fragments
    # only because they contain an uppercase prefix followed by lowercase text.
    if re.fullmatch(r"[A-Z]{2,}(?:ase|ases)", token):
        return False

    if "-" in token or "–" in token:
        parts = [p for p in re.split(r"[-–]", token) if p]

        if any(re.fullmatch(r"[A-Z]{2,}[a-z]{2,}", p) for p in parts):
            return True

        if any(re.fullmatch(r"[A-Z][a-z]{2,}[A-Z][A-Za-z]*", p) for p in parts):
            return True

        if any(re.fullmatch(r"[a-z]{3,}[A-Z][A-Za-z]*", p) for p in parts):
            return True

    if re.fullmatch(r"[A-Z]{2,}[a-z]{4,}", token):
        return True

    if re.fullmatch(r"[A-Z][a-z]{2,}[A-Z][A-Za-z]*", token):
        return True

    if re.fullmatch(r"[a-z]{3,}[A-Z][A-Za-z]*", token):
        return True

    return False


def is_generic_mixed_case_short(token: str) -> bool:
    """
    Generic mixed-case acronym detector.

    No document-specific allowlist is used.
    """
    token = clean_short_token(token)

    if not token or " " in token:
        return False

    if is_false_positive_short_word(token):
        return False

    if len(token) > 24:
        return False

    n_upper = sum(ch.isupper() for ch in token)
    n_lower = sum(ch.islower() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)

    if n_upper == 0 or n_lower == 0:
        return False

    # Simple title-case abbreviations and variable-style labels.
    if is_plain_titlecase_word(token):
        suffix = token[1:].lower()
        if 2 <= len(token) <= 3:
            return True
        if suffix in VARIABLE_SUFFIXES:
            return True
        return False

    # Biomedical lowercase-prefix forms.
    if token[0].islower() and any(ch.isupper() for ch in token[1:]):
        lower_prefix_match = re.match(r"^[a-z]+", token)
        lower_prefix_len = len(lower_prefix_match.group(0)) if lower_prefix_match else 0

        if lower_prefix_len <= 3 and len(token) <= 16:
            return True

        if has_digit and len(token) <= 16:
            return True

        if n_upper >= 2 and len(token) <= 16:
            return True

    # Compact mixed-case clinical/genetic forms.
    if n_upper >= 2 and n_lower >= 1 and len(token) <= 12:
        return True

    # Enzyme/protein-like uppercase prefix + short lowercase suffix.
    if re.fullmatch(r"[A-Z]{2,}[a-z]{1,4}", token):
        return True

    # Pascal/camel-style study/platform acronyms.
    if re.fullmatch(r"(?:[A-Z][a-z]{1,6}){2,3}", token) and len(token) <= 16:
        return True

    # Prefix + all-caps suffix.
    if re.fullmatch(r"[A-Z][a-z]{2,}[A-Z]{2,}[A-Za-z]*", token):
        return True

    return False


def looks_like_short(short: str) -> bool:
    s = clean_short_token(short)

    if not SHORT_SHAPE_RX.fullmatch(s):
        return False

    if s.isdigit():
        return False

    if is_false_positive_short_word(s):
        return False

    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω]", s):
        return False

    if "/" in s and s[0].islower() and not any(ch.isdigit() for ch in s):
        return False

    if is_slash_titlecase_phrase_fragment(s):
        return False

    if is_ordinary_lowercase_hyphen_digit_fragment(s):
        return False

    # Important: lowercase hyphenated real short forms must be accepted before
    # the generic hyphenated-phrase rejection below.
    if is_lowercase_hyphen_short(s):
        return True

    if is_upper_prefix_lowercase_hyphen_phrase(s):
        return False

    if is_hyphenated_phrase_fragment(s):
        return False

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

    if is_generic_mixed_case_short(s):
        return True

    return False


def is_plain_all_caps_word(part: str) -> bool:
    part = clean_short_token(part)
    return bool(re.fullmatch(r"[A-ZΑ-Ω]{3,}", part))


def looks_like_short_sequence(short: str) -> bool:
    s = normalize_spaces(short)

    if not s:
        return False

    if " " not in s:
        return looks_like_short(s)

    parts = [clean_short_token(p) for p in s.split()]
    parts = [p for p in parts if p]

    if not (2 <= len(parts) <= 4):
        return False

    if len(s) > 64:
        return False

    if not looks_like_short(parts[0]):
        return False

    has_numeric_suffix = False

    for part in parts[1:]:
        if re.fullmatch(r"\d+[A-Za-z]?", part):
            has_numeric_suffix = True
            continue

        if looks_like_short(part):
            continue

        return False

    if has_numeric_suffix:
        return True

    if 2 <= len(parts) <= 3 and all(is_plain_all_caps_word(part) for part in parts):
        return True

    return False


def is_code_like_single_token_definition(token: str) -> bool:
    """
    Detect one-token code-like 'definitions'.

    This prevents embedded false splits such as:
      P450 -> 3A4

    It is only used when deciding whether an embedded boundary is credible.
    """
    token = clean_short_token(token)

    if not token:
        return False

    if looks_like_short_sequence(token):
        return True

    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω]*\d[A-Za-z0-9Α-Ωα-ω/\.\-+]*", token):
        return True

    return False


def embedded_after_definition_is_too_weak(after_tokens: List[str]) -> bool:
    after = clean_long_definition(" ".join(after_tokens))
    words = after.split()

    if not words:
        return True

    if len(words) == 1 and is_code_like_single_token_definition(words[0]):
        return True

    return False


def looks_like_standalone_short_line(line: str) -> bool:
    s = clean_short_token(line)

    if not s:
        return False
    if " " in s:
        return False
    if len(s) > 32:
        return False

    return looks_like_short(s)


def find_short_sequence_at(parts: List[str], start_idx: int) -> Optional[Tuple[str, int]]:
    if start_idx >= len(parts) - 1:
        return None

    max_short_tokens = min(4, len(parts) - start_idx - 1)

    for n_tokens in range(max_short_tokens, 0, -1):
        raw_short = " ".join(parts[start_idx:start_idx + n_tokens])
        short = clean_short_token(raw_short)

        if looks_like_short_sequence(short):
            return short, n_tokens

    return None


def strip_probable_trailing_page_number(short: str, long: str) -> str:
    short = clean_short_token(short)
    long = normalize_spaces(long)

    match = re.search(r"\s+(\d{1,3})$", long)
    if not match:
        return long

    number = match.group(1)
    before = long[: match.start()].rstrip()

    if not before:
        return long

    if any(ch.isdigit() for ch in short):
        return long

    previous_word = before.split()[-1].strip(".,;:()[]").lower()

    numeric_definition_cues = {
        "type",
        "factor",
        "protein",
        "receptor",
        "subunit",
        "channel",
        "transporter",
        "kinase",
        "gene",
        "score",
        "class",
        "phase",
        "stage",
        "grade",
        "trial",
        "study",
        "registry",
    }

    if previous_word in numeric_definition_cues:
        return long

    if number.startswith("0"):
        return before

    if len(number) <= 2:
        return before

    return long


def row_definition_is_plausible(short: str, long: str) -> bool:
    short = clean_short_token(short)
    long = clean_long_definition(long)
    long = strip_probable_trailing_page_number(short, long)

    if not short or not long:
        return False

    if len(long) < 3:
        return False

    if LINE_NOISE_RX.search(long):
        return False

    if is_false_positive_short_word(short):
        return False

    # General protection against connector words being split as acronyms inside
    # definitions. They are still allowed when their definition initials align.
    if is_ambiguous_connector_like_all_caps_short(short):
        if not acronym_letters_match_definition(short, long):
            return False

    if " " in short and definition_starts_with_lowercase_word(long):
        return False

    # Older ESC documents may use lowercase definitions. Do not reject strong
    # acronym-like shorts just because the definition begins lowercase.
    if (
        has_lowercase_hyphen_segment(short)
        and not has_strong_acronym_marker(short)
        and not definition_starts_like_glossary_definition(long)
    ):
        return False

    if is_lowercase_hyphen_short(short):
        if not definition_starts_like_glossary_definition(long):
            return False

    return True


def is_confident_embedded_row_boundary(
    current_long_tokens: List[str],
    candidate_short: str,
    after_tokens: List[str],
) -> bool:
    candidate_short = clean_short_token(candidate_short)

    if not current_long_tokens or not after_tokens:
        return False

    current_long = clean_long_definition(" ".join(current_long_tokens))

    if len(current_long) < 8:
        return False

    if not looks_like_short_sequence(candidate_short):
        return False

    if is_false_positive_short_word(candidate_short):
        return False

    if candidate_short[0].isdigit():
        return False

    if is_probable_stylized_word_fragment(candidate_short):
        return False

    if is_ordinary_lowercase_hyphen_digit_fragment(candidate_short):
        return False

    if embedded_after_definition_is_too_weak(after_tokens):
        return False

    after = clean_long_definition(" ".join(after_tokens))

    if is_ambiguous_connector_like_all_caps_short(candidate_short):
        if not acronym_letters_match_definition(candidate_short, after):
            return False

    last_before = current_long_tokens[-1].strip(".,;:()[]").lower()

    if last_before in CONTINUATION_CONNECTOR_ENDINGS:
        return False

    if has_lowercase_hyphen_segment(candidate_short) and not has_strong_acronym_marker(candidate_short):
        first_after = after_tokens[0].strip(".,;:()[]")
        if not first_after or not definition_starts_like_glossary_definition(first_after):
            return False

    if is_lowercase_hyphen_short(candidate_short):
        first_after = after_tokens[0].strip(".,;:()[]")
        if not first_after or not first_after[0].isupper():
            return False

    return True


def is_confident_post_validation_embedded_row_boundary(
    current_long_tokens: List[str],
    candidate_short: str,
    after_tokens: List[str],
) -> bool:
    """
    Diagnostic embedded-row detector used after extraction.

    This does not mutate the acronym dictionary. It only flags cases that may
    deserve inspection.
    """
    candidate_short = clean_short_token(candidate_short)

    if not current_long_tokens or not after_tokens:
        return False

    current_long = clean_long_definition(" ".join(current_long_tokens))

    if len(current_long) < 8:
        return False

    if not looks_like_short_sequence(candidate_short):
        return False

    if is_false_positive_short_word(candidate_short):
        return False

    if candidate_short[0].isdigit():
        return False

    if is_probable_stylized_word_fragment(candidate_short):
        return False

    if is_ordinary_lowercase_hyphen_digit_fragment(candidate_short):
        return False

    if embedded_after_definition_is_too_weak(after_tokens):
        return False

    after = clean_long_definition(" ".join(after_tokens))

    if is_ambiguous_connector_like_all_caps_short(candidate_short):
        if not acronym_letters_match_definition(candidate_short, after):
            return False

    last_before = current_long_tokens[-1].strip(".,;:()[]").lower()

    if last_before in POST_VALIDATION_SPLIT_BLOCKING_ENDINGS:
        return False

    if not row_definition_is_plausible(candidate_short, after):
        return False

    # If the candidate is a strong acronym-like form, allow lowercase definitions
    # in diagnostic mode, because older glossary sections often use lowercase.
    if definition_starts_with_lowercase_word(after) and not has_strong_acronym_marker(candidate_short):
        return False

    return True


def find_next_row_boundary(
    parts: List[str],
    long_start_idx: int,
    search_start_idx: int,
) -> Optional[Tuple[int, int, str]]:
    for idx in range(search_start_idx, len(parts) - 1):
        candidate = find_short_sequence_at(parts, idx)

        if candidate is None:
            continue

        short, n_tokens = candidate
        current_long_tokens = parts[long_start_idx:idx]
        after_tokens = parts[idx + n_tokens:]

        if is_confident_embedded_row_boundary(
            current_long_tokens=current_long_tokens,
            candidate_short=short,
            after_tokens=after_tokens,
        ):
            return idx, n_tokens, short

    return None


def parse_acronym_rows(line: str) -> List[Tuple[str, str]]:
    line = strip_footer_bleed(line)

    if not line:
        return []

    if line.lstrip().startswith("("):
        return []

    parts = line.split()

    if len(parts) < 2:
        return []

    rows: List[Tuple[str, str]] = []

    pos = 0
    first_short = find_short_sequence_at(parts, pos)

    if first_short is None:
        return []

    while pos < len(parts) - 1:
        found = find_short_sequence_at(parts, pos)

        if found is None:
            break

        short, n_short_tokens = found
        long_start = pos + n_short_tokens

        if long_start >= len(parts):
            break

        boundary = find_next_row_boundary(
            parts=parts,
            long_start_idx=long_start,
            search_start_idx=long_start + 1,
        )

        if boundary is None:
            long_tokens = parts[long_start:]
            next_pos = len(parts)
        else:
            boundary_idx, _, _ = boundary
            long_tokens = parts[long_start:boundary_idx]
            next_pos = boundary_idx

        long = clean_long_definition(" ".join(long_tokens))

        if row_definition_is_plausible(short, long):
            rows.append((clean_short_token(short), normalize_spaces(long)))

        if boundary is None:
            break

        pos = next_pos

    return rows


def parse_acronym_row(line: str) -> Optional[Tuple[str, str]]:
    rows = parse_acronym_rows(line)
    if not rows:
        return None
    return rows[0]


def find_embedded_row_in_definition(long: str) -> Optional[Tuple[str, str, str]]:
    """
    Detect one acronym definition that may still contain another acronym row.

    Returns:
      embedded_short, suggested_current_definition, suggested_embedded_definition

    The caller must treat this as diagnostic only unless explicitly choosing to
    perform an auto-fix elsewhere.
    """
    long = clean_long_definition(long)
    parts = long.split()

    if len(parts) < 4:
        return None

    for idx in range(1, len(parts) - 1):
        candidate = find_short_sequence_at(parts, idx)

        if candidate is None:
            continue

        embedded_short, n_tokens = candidate

        before_tokens = parts[:idx]
        after_tokens = parts[idx + n_tokens:]

        if not before_tokens or not after_tokens:
            continue

        if not is_confident_post_validation_embedded_row_boundary(
            current_long_tokens=before_tokens,
            candidate_short=embedded_short,
            after_tokens=after_tokens,
        ):
            continue

        before = clean_long_definition(" ".join(before_tokens))
        after = clean_long_definition(" ".join(after_tokens))

        return clean_short_token(embedded_short), before, after

    return None


def short_contains_component(parent_short: str, embedded_short: str) -> bool:
    """
    Check whether an embedded candidate is already a component of the parent
    short form.

    This suppresses false diagnostics when a parent acronym naturally contains
    a component acronym separated by punctuation.
    """
    parent_short = clean_short_token(parent_short)
    embedded_short = clean_short_token(embedded_short)

    if not parent_short or not embedded_short:
        return False

    parent_parts = [
        short_alpha_letters(p)
        for p in re.split(r"[^A-Za-zΑ-Ωα-ω0-9]+", parent_short)
        if p
    ]
    embedded_letters = short_alpha_letters(embedded_short)

    if not embedded_letters:
        return False

    return embedded_letters in parent_parts


def definition_looks_suspicious(short: str, long: str) -> bool:
    long_norm = normalize_spaces(long)

    if not long_norm:
        return True

    embedded = find_embedded_row_in_definition(long_norm)
    if embedded is not None:
        embedded_short, _, _ = embedded
        if not short_contains_component(short, embedded_short):
            return True

    words = long_norm.split()
    last = words[-1].strip(".,;:()[]").lower() if words else ""

    strong_dangling_endings = {
        "of",
        "or",
        "and",
        "for",
        "from",
        "the",
        "to",
        "with",
        "without",
        "by",
        "in",
        "on",
        "via",
        "vs",
        "versus",
        "as",
        "at",
        "after",
        "before",
        "between",
        "during",
        "following",
        "followed",
        "who",
        "that",
        "which",
        "ischemic",
        "ischaemic",
        "clinical",
        "patients",
        "patient",
        "eligible",
        "halt",
        "fractional",
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


def add_or_update_acronym(acronyms: Dict[str, str], short: str, long: str) -> None:
    short = clean_short_token(short)
    long = clean_long_definition(long)
    long = strip_probable_trailing_page_number(short, long)

    if not short or not long:
        return

    existing = acronyms.get(short)

    if existing is None:
        acronyms[short] = long
        return

    existing_norm = normalize_spaces(existing).lower()
    new_norm = normalize_spaces(long).lower()

    if existing_norm == new_norm:
        return

    existing_suspicious = definition_looks_suspicious(short, existing)
    new_suspicious = definition_looks_suspicious(short, long)

    if existing_suspicious and not new_suspicious:
        acronyms[short] = long
        return

    if not existing_suspicious and new_suspicious:
        return

    if (
        not existing_suspicious
        and definition_starts_like_glossary_definition(existing)
        and definition_starts_with_lowercase_word(long)
    ):
        return

    if new_norm.startswith(existing_norm) and not existing_suspicious:
        return

    if existing_norm.startswith(new_norm) and not new_suspicious:
        acronyms[short] = long
        return

    if not existing_suspicious and not new_suspicious:
        if len(long) < len(existing):
            acronyms[short] = long
        return

    if len(long) < len(existing):
        acronyms[short] = long


def should_attach_continuation(
    line: PDFLine,
    current_long: str,
    prev_line: Optional[PDFLine],
) -> bool:
    text = strip_footer_bleed(line.text)

    if not text or len(text) < 2:
        return False

    if LINE_NOISE_RX.search(text):
        return False

    if is_body_heading(text):
        return False

    if is_acronym_heading(text):
        return False

    parsed_rows = parse_acronym_rows(text)
    if parsed_rows:
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

        parsed_rows = parse_acronym_rows(text)

        if parsed_rows:
            for short, long in parsed_rows:
                add_or_update_acronym(acronyms, short, long)

            current_short = parsed_rows[-1][0]
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
                if len(long) >= 3 and row_definition_is_plausible(pending_short, long):
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


def is_probable_backronym_fragment(short: str, long: str) -> bool:
    """
    Diagnostic-only detector for stylized fragments inside trial/study names.

    Important:
    - This does not remove anything.
    - This avoids flagging valid mixed-case acronyms only because they are
      mixed-case.
    """
    short = clean_short_token(short)
    long = clean_long_definition(long)

    if not short or not long:
        return False

    if not is_probable_stylized_word_fragment(short):
        return False

    words = long.split()
    if not words:
        return False

    first_long = words[0].strip(".,;:()[]{}’'").lower()

    if first_long in {
        "and",
        "or",
        "with",
        "without",
        "of",
        "for",
        "to",
        "in",
        "on",
        "have",
        "has",
        "are",
        "is",
        "be",
        "being",
    }:
        return True

    if len(words) == 1:
        return True

    return False


def post_validate_acronyms(
    acronyms: Dict[str, str],
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Non-destructive post-validation pass.

    Current behavior:
    - normalizes/cleans short and long forms;
    - flags possible embedded acronym rows;
    - flags probable backronym/parser fragments;
    - does NOT split, delete, shorten, or add acronym entries.

    This means n_acronyms should normally remain equal to n_raw_acronyms, except
    when two raw keys collapse to the same cleaned short form.
    """
    cleaned: Dict[str, str] = {}
    validation_issues: List[Dict[str, str]] = []

    for short, long in acronyms.items():
        short_clean = clean_short_token(short)
        long_clean = clean_long_definition(long)
        long_clean = strip_probable_trailing_page_number(short_clean, long_clean)

        if not short_clean or not long_clean:
            validation_issues.append(
                {
                    "short": short,
                    "definition": long,
                    "reason": "empty_after_cleaning",
                }
            )
            continue

        if short_clean in cleaned and normalize_spaces(cleaned[short_clean]).lower() != normalize_spaces(long_clean).lower():
            validation_issues.append(
                {
                    "short": short_clean,
                    "definition": long_clean,
                    "reason": "duplicate_short_after_cleaning",
                    "existing_definition": cleaned[short_clean],
                }
            )
            add_or_update_acronym(cleaned, short_clean, long_clean)
        else:
            cleaned[short_clean] = long_clean

    for short, long in sorted(cleaned.items()):
        embedded = find_embedded_row_in_definition(long)

        if embedded is not None:
            embedded_short, suggested_current_definition, suggested_embedded_definition = embedded

            if embedded_short != short and not short_contains_component(short, embedded_short):
                validation_issues.append(
                    {
                        "short": short,
                        "definition": long,
                        "reason": "possible_embedded_acronym_row",
                        "suggested_current_definition": suggested_current_definition,
                        "suggested_embedded_short": embedded_short,
                        "suggested_embedded_definition": suggested_embedded_definition,
                    }
                )

        if is_probable_backronym_fragment(short, long):
            validation_issues.append(
                {
                    "short": short,
                    "definition": long,
                    "reason": "probable_backronym_or_parser_fragment",
                }
            )

    return dict(sorted(cleaned.items())), validation_issues


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
    return sum(len(parse_acronym_rows(line.text)) for line in lines)


def find_dense_acronym_span(
    doc: fitz.Document,
    page_start: int,
    page_end: int,
) -> Optional[Tuple[int, int]]:
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
    raw_n_acronyms: Optional[int] = None,
    validation_issues: Optional[List[Dict[str, str]]] = None,
) -> Dict:
    validation_issues = validation_issues or []
    suspicious = collect_suspicious_acronyms(acronyms)

    return {
        "doc_id": doc_id,
        "status": status,
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "n_raw_acronyms": raw_n_acronyms if raw_n_acronyms is not None else len(acronyms),
        "n_acronyms": len(acronyms),
        "n_suspicious": len(suspicious),
        "suspicious": suspicious,
        "n_validation_issues": len(validation_issues),
        "validation_issues": validation_issues,
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

        raw_acronyms = extract_acronyms_from_lines(candidate_lines)

        if not raw_acronyms:
            logger.warning("No acronyms extracted for %s", doc_id)
            return build_output_payload(
                doc_id=doc_id,
                status="parse_empty",
                source=source,
                page_start=start_page,
                page_end=end_page,
                acronyms={},
            )

        acronyms, validation_issues = post_validate_acronyms(raw_acronyms)

        if validation_issues:
            logger.info(
                "Post-validation flagged %d acronym candidate(s) for %s",
                len(validation_issues),
                doc_id,
            )

            for issue in validation_issues[:10]:
                logger.info("Acronym post-validation issue: %s", issue)

            if len(validation_issues) > 10:
                logger.info(
                    "... %d more acronym post-validation issues not printed",
                    len(validation_issues) - 10,
                )

        if not acronyms:
            logger.warning("All acronym candidates empty after post-validation for %s", doc_id)
            return build_output_payload(
                doc_id=doc_id,
                status="post_validation_empty",
                source=source,
                page_start=start_page,
                page_end=end_page,
                acronyms={},
                raw_n_acronyms=len(raw_acronyms),
                validation_issues=validation_issues,
            )

        payload = build_output_payload(
            doc_id=doc_id,
            status="success",
            source=source,
            page_start=start_page,
            page_end=end_page,
            acronyms=acronyms,
            raw_n_acronyms=len(raw_acronyms),
            validation_issues=validation_issues,
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