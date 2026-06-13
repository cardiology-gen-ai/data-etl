import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


logger = logging.getLogger(__name__)


ACRONYM_FILENAME_SUFFIX = "_acronyms.json"

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

_ALNUM_BOUNDARY_CHARS = r"A-Za-z0-9Α-Ωα-ω"

_PARENTHESES_PLURAL_MARKER_RX = re.compile(
    r"\(\s*(s|es)\s*\)",
    flags=re.IGNORECASE,
)

_ACRONYM_ALLOWED_CHARS_RX = re.compile(
    r"^[A-Za-z0-9Α-Ωα-ω][A-Za-z0-9Α-Ωα-ω+\-/.\s]*$"
)

_UPPER_OR_DIGIT_RX = re.compile(r"[A-ZΑ-Ω0-9]")

_LOWER_RX = re.compile(r"[a-zα-ω]")

_LETTER_RX = re.compile(r"[A-Za-zΑ-Ωα-ω]")

_MEDICAL_SINGULAR_S_EXCEPTIONS = {
    "diabetes",
    "herpes",
    "measles",
    "mumps",
}

_MEDICAL_SINGULAR_S_SUFFIX_RX = re.compile(
    r"(sis|itis|osis|esis|iasis|asis|ysis|ss|us|is)$",
    flags=re.IGNORECASE,
)


def normalize_unicode_text(text: Any) -> str:
    """
    Normalize unicode and common PDF artefacts.

    This is intentionally local to acronym utilities so this file does not
    depend on the PDF extractor implementation.
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

    text = text.translate(_DASH_TRANSLATION)
    return text


def normalize_spaces(text: Any) -> str:
    """
    Collapse whitespace to single spaces.
    """
    text = normalize_unicode_text(text)
    return re.sub(r"\s+", " ", text).strip()


def clean_acronym_short(short: Any) -> str:
    """
    Clean an acronym short form without changing its case.

    Case is preserved because acronym matching in source text should be
    conservative. For example, the acronym "AS" should not match the ordinary
    lowercase word "as".
    """
    short = normalize_spaces(short)
    short = short.strip("[],:;.")

    while len(short) > 1 and short[-1] == "-":
        short = short[:-1].strip()

    return short


def clean_acronym_definition(definition: Any) -> str:
    """
    Clean an acronym long form while preserving readable text.
    """
    definition = normalize_spaces(definition)
    return definition.strip(" -:;,")


def get_acronym_cache_path(acronym_dir: Path, doc_id: str) -> Path:
    """
    Return the expected cached acronym JSON path for a document.
    """
    return Path(acronym_dir) / f"{doc_id}{ACRONYM_FILENAME_SUFFIX}"


def load_acronym_payload(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a cached acronym JSON payload.

    Returns None if the file is missing or invalid.
    """
    path = Path(path)

    if not path.exists():
        logger.info("Acronym cache not found: %s", path)
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read acronym cache %s: %s", path, e)
        return None

    if not isinstance(payload, dict):
        logger.warning("Acronym cache is not a JSON object: %s", path)
        return None

    return payload


def extract_acronym_map_from_payload(payload: Mapping[str, Any]) -> Dict[str, str]:
    """
    Extract the acronym dictionary from a cached acronym payload.

    Expected payload shape:
        {
            "doc_id": "...",
            "status": "success",
            "acronyms": {
                "ACS": "Acute coronary syndrome(s)",
                ...
            }
        }

    Invalid entries are skipped.
    """
    raw_acronyms = payload.get("acronyms", {})

    if not isinstance(raw_acronyms, dict):
        return {}

    acronyms: Dict[str, str] = {}

    for raw_short, raw_definition in raw_acronyms.items():
        short = clean_acronym_short(raw_short)
        definition = clean_acronym_definition(raw_definition)

        if not short or not definition:
            continue

        acronyms[short] = definition

    return dict(sorted(acronyms.items()))


def load_acronyms_for_doc(
    acronym_dir: Optional[Path],
    doc_id: Optional[str],
) -> Dict[str, str]:
    """
    Load the cached acronym dictionary for one document.

    Returns an empty dictionary when:
    - acronym_dir is None;
    - doc_id is None;
    - the cache file is missing;
    - the cache file is invalid;
    - the payload has no valid acronyms.
    """
    if acronym_dir is None or doc_id is None:
        return {}

    path = get_acronym_cache_path(Path(acronym_dir), doc_id)
    payload = load_acronym_payload(path)

    if payload is None:
        return {}

    acronyms = extract_acronym_map_from_payload(payload)

    logger.info(
        "Loaded acronym cache for %s | path=%s | n_acronyms=%d",
        doc_id,
        path,
        len(acronyms),
    )

    return acronyms


def load_acronyms_by_doc_id(
    acronym_dir: Optional[Path],
    doc_ids: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    """
    Load cached acronym dictionaries for multiple documents.
    """
    if acronym_dir is None:
        return {}

    out: Dict[str, Dict[str, str]] = {}

    for doc_id in sorted(set(doc_ids)):
        if not doc_id:
            continue
        out[doc_id] = load_acronyms_for_doc(
            acronym_dir=Path(acronym_dir),
            doc_id=doc_id,
        )

    return out


def normalize_long_form_for_matching(text: Any) -> str:
    """
    Normalize concept names and acronym definitions for long-form comparison.

    This is stricter than semantic normalization:
    - lowercase;
    - normalize dashes;
    - treat hyphens and spaces as equivalent;
    - remove most punctuation;
    - normalize whitespace.

    It intentionally does NOT do semantic rewriting.
    """
    text = normalize_unicode_text(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("'", "")
    text = text.replace("’", "")
    text = text.replace("_", " ")
    text = text.replace("/", " ")

    # Keep unicode word characters, whitespace, percent, plus and hyphen.
    # Then collapse hyphen/space differences.
    text = re.sub(r"[^\w\s%+\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[\s\-]+", " ", text)
    text = normalize_spaces(text)

    return text


def pluralize_last_token(form: str) -> Optional[str]:
    """
    Conservative pluralization for the last token of a normalized long form.
    """
    form = normalize_spaces(form)
    if not form:
        return None

    parts = form.split()
    if not parts:
        return None

    token = parts[-1]

    if token.endswith(("s", "x", "z", "ch", "sh")):
        plural = token + "es"
    elif re.search(r"[^aeiou]y$", token):
        plural = token[:-1] + "ies"
    else:
        plural = token + "s"

    return " ".join(parts[:-1] + [plural])


def singularize_last_token(form: str) -> Optional[str]:
    """
    Conservative singularization for the last token of a normalized long form.

    This is used only for matching/canonicalizing acronym definitions. It avoids
    common medical singular words ending in -s, such as stenosis, fibrosis,
    myocarditis, diabetes, and mellitus.
    """
    form = normalize_spaces(form)
    if not form:
        return None

    parts = form.split()
    if not parts:
        return None

    token = parts[-1]

    if len(token) <= 3:
        return None

    if token in _MEDICAL_SINGULAR_S_EXCEPTIONS:
        return None

    if _MEDICAL_SINGULAR_S_SUFFIX_RX.search(token):
        return None

    singular: Optional[str] = None

    if token.endswith("ies") and len(token) > 4:
        singular = token[:-3] + "y"

    elif token.endswith("es") and len(token) > 4:
        base_without_es = token[:-2]

        if base_without_es.endswith(("s", "x", "z", "ch", "sh")):
            singular = base_without_es
        else:
            base_without_s = token[:-1]
            if base_without_s.endswith("e"):
                singular = base_without_s

    elif token.endswith("s") and not token.endswith("ss"):
        singular = token[:-1]

    if not singular:
        return None

    return " ".join(parts[:-1] + [singular])


def expand_parenthetical_plural_markers(text: Any) -> Set[str]:
    """
    Expand forms such as:
        syndrome(s)

    into:
        syndrome
        syndromes
    """
    text = normalize_unicode_text(text)

    variants = {text}

    if _PARENTHESES_PLURAL_MARKER_RX.search(text):
        variants.add(_PARENTHESES_PLURAL_MARKER_RX.sub("", text))
        variants.add(
            _PARENTHESES_PLURAL_MARKER_RX.sub(
                lambda match: match.group(1).lower(),
                text,
            )
        )

    return {v for v in variants if normalize_spaces(v)}


def build_long_form_variants(text: Any) -> Set[str]:
    """
    Build normalized variants for comparing acronym definitions to concept names.

    The variants are intentionally conservative:
    - base normalized form;
    - expansion/removal of parenthetical plural markers;
    - simple plural/singular last-token variants.
    """
    variants: Set[str] = set()

    for expanded in expand_parenthetical_plural_markers(text):
        base = normalize_long_form_for_matching(expanded)

        if not base:
            continue

        variants.add(base)

        plural = pluralize_last_token(base)
        if plural:
            variants.add(plural)

        singular = singularize_last_token(base)
        if singular:
            variants.add(singular)

    return {v for v in variants if v}


def long_form_matches_concept(
    concept_name: Any,
    acronym_definition: Any,
) -> bool:
    """
    Check whether an acronym definition matches an extracted concept name.

    This is deliberately equality-based over normalized variants.
    It does not accept loose substring matches, because that would make cases
    such as "heart failure" match "heart failure with reduced ejection fraction".
    """
    concept_variants = build_long_form_variants(concept_name)
    definition_variants = build_long_form_variants(acronym_definition)

    if not concept_variants or not definition_variants:
        return False

    return bool(concept_variants & definition_variants)


def canonicalize_acronym_definition_for_concept_name(definition: Any) -> str:
    """
    Convert a cached acronym definition into the canonical KG concept name
    that should be written when the LLM extracted only the acronym short form.

    """
    definition = clean_acronym_definition(definition)

    if not definition:
        return ""

    # Prefer the non-parenthetical-plural version as the canonical base:
    # syndrome(s) -> syndrome
    without_parenthetical_plural = _PARENTHESES_PLURAL_MARKER_RX.sub(
        "",
        definition,
    )
    base = normalize_long_form_for_matching(without_parenthetical_plural)

    if not base:
        base = normalize_long_form_for_matching(definition)

    if not base:
        return ""

    singular = singularize_last_token(base)
    if singular:
        return singular

    return base


def _normalize_short_for_lookup(short: Any, case_sensitive: bool) -> str:
    """
    Normalize acronym short forms for lookup comparisons.

    This helper is deliberately separate from clean_acronym_short() because
    source-text matching must preserve exact case, while cache lookup can allow
    small punctuation/spacing differences when the raw LLM value is clearly an
    acronym.
    """
    short = clean_acronym_short(short)
    short = short.replace(".", "")
    short = re.sub(r"[\s\-]+", "-", short).strip("-")

    if not case_sensitive:
        short = short.upper()

    return short


def is_likely_acronym_short_form(short: Any) -> bool:
    """
    Heuristically detect raw acronym-only LLM outputs.

    This is intentionally conservative. It should detect cases like:
        AS, ACS, LV, CMR, ECG, ICD, HF, HFpEF only when mostly acronym-shaped.

    It should NOT treat ordinary lowercase words like "as" as acronyms.
    """
    short = clean_acronym_short(short)

    if not short:
        return False

    if len(short) > 24:
        return False

    if not _ACRONYM_ALLOWED_CHARS_RX.match(short):
        return False

    compact = re.sub(r"[\s\-/.+]", "", short)

    if len(compact) < 2:
        return False

    if not _LETTER_RX.search(compact):
        return False

    if not _UPPER_OR_DIGIT_RX.search(compact):
        return False

    # Pure uppercase/digit acronym-like forms are safe to treat as short forms.
    if not _LOWER_RX.search(compact):
        return True

    # Mixed-case biomedical acronym forms are accepted only when they contain a
    # strong uppercase/digit signal and are short. This covers forms like HFpEF
    # while avoiding ordinary clinical phrases.
    uppercase_or_digit_count = len(_UPPER_OR_DIGIT_RX.findall(compact))
    lowercase_count = len(_LOWER_RX.findall(compact))

    return (
        len(compact) <= 12
        and uppercase_or_digit_count >= 2
        and uppercase_or_digit_count >= lowercase_count
    )


def find_cached_acronym_entry(
    raw_short: Any,
    acronyms: Optional[Mapping[str, str]],
) -> Optional[Tuple[str, str]]:
    """
    Find a cached acronym entry for a raw short form.

    Matching policy:
    1. Exact cleaned match first. This preserves case-safety.
    2. If the raw form itself looks like an acronym, allow a normalized
       case-insensitive lookup. This lets minor formatting differences match,
       without allowing lowercase ordinary words such as "as" to match "AS".

    Returns:
        (cached_short, cached_definition)

    Otherwise returns None.
    """
    if not raw_short or not acronyms:
        return None

    candidate = clean_acronym_short(raw_short)

    if not candidate:
        return None

    cleaned_entries: List[Tuple[str, str]] = []

    for raw_cached_short, raw_definition in sorted(acronyms.items()):
        cached_short = clean_acronym_short(raw_cached_short)
        cached_definition = clean_acronym_definition(raw_definition)

        if not cached_short or not cached_definition:
            continue

        cleaned_entries.append((cached_short, cached_definition))

    # Exact case-sensitive match first.
    for cached_short, cached_definition in cleaned_entries:
        if candidate == cached_short:
            return cached_short, cached_definition

    # Conservative normalized fallback only if the candidate itself is acronym-like.
    if not is_likely_acronym_short_form(candidate):
        return None

    candidate_lookup = _normalize_short_for_lookup(
        candidate,
        case_sensitive=False,
    )

    for cached_short, cached_definition in cleaned_entries:
        cached_lookup = _normalize_short_for_lookup(
            cached_short,
            case_sensitive=False,
        )

        if candidate_lookup == cached_lookup:
            return cached_short, cached_definition

    return None


def build_acronym_short_pattern(short: Any) -> Optional[str]:
    """
    Build a conservative regex pattern for finding an acronym short form
    in section source text.

    Matching is case-sensitive.

    Spaces and hyphens inside acronym forms are allowed to vary slightly.
    Boundaries prevent accidental substring matches inside longer words.
    """
    short = clean_acronym_short(short)

    if not short:
        return None

    pieces: List[str] = []
    previous_was_flexible_separator = False

    for ch in short:
        if ch.isspace() or ch == "-":
            if not previous_was_flexible_separator:
                pieces.append(r"[\s\-]+")
                previous_was_flexible_separator = True
            continue

        pieces.append(re.escape(ch))
        previous_was_flexible_separator = False

    if not pieces:
        return None

    body = "".join(pieces)

    return rf"(?<![{_ALNUM_BOUNDARY_CHARS}]){body}(?![{_ALNUM_BOUNDARY_CHARS}])"


def acronym_short_is_present_in_source(short: Any, source_text: Any) -> bool:
    """
    Check whether an acronym short form appears in section source text.

    This is case-sensitive by design.
    """
    source_text = normalize_unicode_text(source_text)
    pattern = build_acronym_short_pattern(short)

    if not pattern or not source_text:
        return False

    return re.search(pattern, source_text) is not None


def get_acronym_expansion_for_short(
    raw_short: Any,
    source_text: Any,
    acronyms: Optional[Mapping[str, str]],
    require_source_presence: bool = True,
) -> Optional[Dict[str, str]]:
    """
    Return expansion evidence for a raw acronym-only concept.
    """
    if not raw_short or not acronyms:
        return None

    entry = find_cached_acronym_entry(
        raw_short=raw_short,
        acronyms=acronyms,
    )

    if entry is None:
        return None

    short, definition = entry

    if require_source_presence and not acronym_short_is_present_in_source(
        short=short,
        source_text=source_text,
    ):
        return None

    expanded_name = canonicalize_acronym_definition_for_concept_name(definition)

    if not expanded_name:
        return None

    return {
        "acronym_short": short,
        "acronym_definition": definition,
        "expanded_name": expanded_name,
        "acronym_match_method": "raw_short_matches_cached_acronym",
    }


def get_acronym_support_for_concept(
    concept_name: Any,
    source_text: Any,
    acronyms: Optional[Mapping[str, str]],
) -> Optional[Dict[str, str]]:
    """
    Return acronym-based support evidence for a concept.

    A concept is supported by acronyms only when BOTH conditions hold:

    1. The section source text contains the acronym short form.
    2. The cached acronym definition matches the concept long form.

    """
    if not concept_name or not source_text or not acronyms:
        return None

    for raw_short, raw_definition in sorted(acronyms.items()):
        short = clean_acronym_short(raw_short)
        definition = clean_acronym_definition(raw_definition)

        if not short or not definition:
            continue

        if not long_form_matches_concept(
            concept_name=concept_name,
            acronym_definition=definition,
        ):
            continue

        if not acronym_short_is_present_in_source(
            short=short,
            source_text=source_text,
        ):
            continue

        return {
            "acronym_short": short,
            "acronym_definition": definition,
            "acronym_match_method": "short_in_source_and_definition_matches_concept",
        }

    return None


__all__ = [
    "ACRONYM_FILENAME_SUFFIX",
    "get_acronym_cache_path",
    "load_acronym_payload",
    "extract_acronym_map_from_payload",
    "load_acronyms_for_doc",
    "load_acronyms_by_doc_id",
    "clean_acronym_short",
    "clean_acronym_definition",
    "normalize_long_form_for_matching",
    "build_long_form_variants",
    "long_form_matches_concept",
    "canonicalize_acronym_definition_for_concept_name",
    "is_likely_acronym_short_form",
    "find_cached_acronym_entry",
    "acronym_short_is_present_in_source",
    "get_acronym_expansion_for_short",
    "get_acronym_support_for_concept",
]