"""
validate_entities.py

Deterministic pre-write validation for extracted entities.

Purpose:
- check whether each extracted concept is still acceptable after LLM extraction
- enforce simple safety rules before writing to the KG
- reject concepts that are unsupported by the section source text
- keep the validation conservative: we do NOT auto-correct typos or rewrite names
- allow acronym-based support when a concept long form is supported by a cached
  document acronym whose short form appears in the section text
- expand raw acronym-only LLM outputs when a safe cached document-level acronym
  definition is available
- keep support evidence so review exports and MENTIONS edges can explain why a
  concept was accepted

Main policy:
- accepted concepts must have a valid name and type
- accepted concepts must not be blocklisted
- accepted concepts must be explicitly supported by the section text directly,
  or indirectly through a document-level acronym definition
- raw acronym-only concepts are not written as Concept nodes when they can be
  safely expanded; the expanded long form is written instead
- raw acronym-only concepts without a safe expansion are rejected rather than
  written as ambiguous short-form nodes
- uppercase gene symbols extracted as genetic_factor are treated as canonical
  gene symbols, not clinical acronyms; exact source capitalization is preserved
- obvious care settings, organizations, and generic population labels are
  rejected because the schema has no corresponding intrinsic entity type
- medical disciplines, research designs, generic variables, isolated clinical
  adjectives, and broad therapeutic or biomarker categories are omitted
- treatment modalities cannot be written as lifestyle/environmental exposures
- phrases containing an acronym plus ordinary words, such as "12-lead ECG" or
  "HAS-BLED score", may be grounded directly
- a cached acronym may support one explicit component inside a longer phrase,
  for example "CV surveillance" supporting "cardiovascular surveillance"
- tiny lowercase surface names such as "as" are rejected by direct source
  matching, because they can be ordinary words rather than clinical concepts
- support checking is intentionally strict, with only small matching tolerance
  for whitespace, hyphenation, common singular/plural forms, spelling variants,
  and acronym expansion
"""

import logging
import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_graph.acronym_utils import (
    clean_acronym_definition,
    clean_acronym_short,
    get_acronym_expansion_for_short,
    get_acronym_support_for_concept,
    is_likely_acronym_short_form,
)
from knowledge_graph.entity_schema import (
    normalize_name,
    normalize_type,
    normalize_whitespace,
)


logger = logging.getLogger(__name__)


_DASH_TRANSLATION = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
})


# Require at least three characters so very short clinical abbreviations such
# as AS or LV are not accepted as genes solely from surface form.
_GENE_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]{2,19}$")

_CARE_SETTING_HEADS = {
    "care",
    "clinic",
    "hospital",
    "practice",
    "service",
    "services",
    "setting",
    "unit",
    "ward",
}

_HUMAN_GROUP_MARKERS = {
    "adult",
    "adults",
    "adolescent",
    "adolescents",
    "athlete",
    "athletes",
    "child",
    "children",
    "family",
    "families",
    "individual",
    "individuals",
    "infant",
    "infants",
    "mother",
    "mothers",
    "neonate",
    "neonates",
    "patient",
    "patients",
    "people",
    "person",
    "persons",
    "relative",
    "relatives",
    "survivor",
    "survivors",
    "woman",
    "women",
}

_ORGANIZATION_HEADS = {
    "association",
    "college",
    "committee",
    "consortium",
    "council",
    "federation",
    "foundation",
    "group",
    "organisation",
    "organization",
    "society",
}

_ORGANIZATION_PHRASE_MARKERS = {
    "association of",
    "college of",
    "committee on",
    "consortium of",
    "council of",
    "society of",
    "task force",
    "working group",
}

_GENERIC_POPULATION_NAMES = {
    "patient population",
    "patient populations",
    "selected patient",
    "selected patients",
    "special population",
    "special populations",
}

_NON_ENTITY_DISCIPLINE_NAMES = {
    "cardio oncology",
    "cardiology",
    "haematology",
    "hematology",
    "oncology",
}

_NON_ANATOMICAL_ADJECTIVE_NAMES = {
    "cardiac",
    "cardiovascular",
    "coronary",
    "myocardial",
    "vascular",
}

_NONCLINICAL_RESEARCH_PHRASES = {
    "genome wide association studies",
    "genome wide association study",
    "meta analyses",
    "meta analysis",
    "randomised clinical trial",
    "randomised controlled trial",
    "randomized clinical trial",
    "randomized controlled trial",
}

_NONCLINICAL_RESEARCH_HEADS = {
    "registries",
    "registry",
    "studies",
    "study",
    "trial",
    "trials",
}


# Research/document objects are outside the current intrinsic entity schema.
# Keep these rules narrow enough not to reject bona-fide diagnostic procedures
# whose established name happens to end in ``study``.
_ALWAYS_NON_ENTITY_RESEARCH_HEADS = {
    "registries",
    "registry",
    "trial",
    "trials",
}

_CLINICAL_DIAGNOSTIC_STUDY_MARKERS = {
    "electrophysiologic",
    "electrophysiological",
    "electrophysiology",
    "haemodynamic",
    "hemodynamic",
    "imaging",
    "perfusion",
    "scintigraphic",
    "sleep",
}

_SCORE_ARTIFACT_MARKERS = {
    "classification",
    "criteria",
    "model",
    "mortality",
    "predicted",
    "prediction",
    "risk",
    "score",
    "staging",
}

_PUBLICATION_TITLE_PATTERNS = (
    re.compile(r"^\d{4}\s+.+\bguidelines?\b"),
    re.compile(r"\b(?:clinical\s+practice\s+)?guidelines?\s+(?:for|of|on|regarding)\b"),
    re.compile(r"\bconsensus\s+(?:document|paper|statement)\b"),
    re.compile(r"\bposition\s+(?:paper|statement)\b"),
    re.compile(r"\bscientific\s+statement\b"),
)

_GENERIC_VARIABLE_NAMES = {
    "cancer type",
    "clinical syndrome",
    "disease type",
    "sex category",
    "tumor type",
    "tumour type",
}

_GENERIC_PROCESS_HEADS = {
    "assessment",
    "diagnosis",
    "evaluation",
    "management",
    "monitoring",
    "screening",
    "surveillance",
}

_GENERIC_PROCESS_MODIFIERS = {
    "baseline",
    "cancer",
    "clinical",
    "general",
    "initial",
    "oncological",
    "oncology",
    "patient",
    "pre",
    "pretreatment",
    "routine",
    "treatment",
}

_GENERIC_BIOMARKER_HEADS = {
    "biomarker",
    "biomarkers",
    "marker",
    "markers",
}

_GENERIC_BIOMARKER_MODIFIERS = {
    "biological",
    "blood",
    "cardiac",
    "cardiovascular",
    "circulating",
    "plasma",
    "serum",
}

_TREATMENT_MODALITY_TOKENS = {
    "chemotherapy",
    "radiotherapy",
    "therapy",
    "treatment",
}

_EXPOSURE_CONTEXT_TOKENS = {
    "abuse",
    "consumption",
    "emission",
    "emissions",
    "exposure",
    "particulate",
    "particle",
    "particles",
    "pollutant",
    "pollutants",
    "smoking",
    "use",
    "vaping",
}

_NON_DRUG_TREATMENT_MODALITIES = {
    "immunosuppression",
    "radiation therapy",
    "radiotherapy",
}

# Known cache corruption observed in the Cardio-oncology acronym artifact.
# Unsafe definitions are ignored rather than written as malformed concepts.
_UNSAFE_ACRONYM_DEFINITION_EXACT = {
    (
        "ECG",
        "electrocardiogram echo echocardiography",
    ),
}

_EMBEDDED_SOURCE_NAME_CLASS_HEADS = {
    "agonist",
    "agonists",
    "antagonist",
    "antagonists",
    "inhibitor",
    "inhibitors",
}

_EMBEDDED_ASSOCIATED_DISEASE_HEADS = {
    "arthritis",
    "colitis",
    "hepatitis",
    "myocarditis",
    "nephritis",
    "pneumonitis",
    "thyroiditis",
}

_EMBEDDED_THERAPEUTIC_DEFINITION_HEADS = {
    "agonist",
    "agonists",
    "antagonist",
    "antagonists",
    "inhibitor",
    "inhibitors",
}


# Conservative spelling equivalences useful in cardiology guideline text.
# These are NOT used to rewrite concept names, only to support source matching.


# High-precision semantic review rules.
#
# These rules do not attempt to replace the LLM classifier. They either:
# - reject a very narrow class of clearly generic causal descriptions; or
# - attach quality flags to grounded concepts that deserve later review.
_GENERIC_CAUSAL_HEADS = {
    "cause",
    "aetiology",
    "etiology",
    "mechanism",
    "origin",
    "basis",
}

_GENERIC_CAUSAL_MODIFIERS = {
    "genetic",
    "monogenic",
    "polygenic",
    "familial",
    "inherited",
    "molecular",
    "biological",
    "pathogenic",
    "pathophysiological",
    "underlying",
    "environmental",
    "multifactorial",
    "complex",
    "disease",
}

_RESULT_LIKE_HEADS = {
    "abnormality",
    "abnormalities",
    "damage",
    "defect",
    "defects",
    "dysfunction",
    "enhancement",
    "finding",
    "findings",
    "impairment",
    "measurement",
    "measurements",
    "pattern",
    "patterns",
    "result",
    "results",
    "value",
    "values",
}

_EXPOSURE_LIKE_HEADS = {
    "abuse",
    "carbonyl",
    "consumption",
    "emission",
    "emissions",
    "exposure",
    "particulate",
    "particle",
    "particles",
    "pollutant",
    "pollutants",
    "smoking",
    "use",
    "vaping",
}

_NONMEDICAL_DEVICE_TERMS = {
    "cigarette",
    "cigarettes",
    "e-cigarette",
    "e-cigarettes",
    "vape",
    "vapes",
    "vaporizer",
    "vaporizers",
}

_GENERIC_THERAPEUTIC_HEADS = {
    "drug",
    "drugs",
    "medication",
    "medications",
    "medicine",
    "medicines",
    "therapy",
    "therapies",
    "treatment",
    "treatments",
}

_GENERIC_THERAPEUTIC_MODIFIERS = {
    "adjuvant",
    "anticancer",
    "background",
    "cancer",
    "cardiac",
    "cardiovascular",
    "chronic",
    "concomitant",
    "medical",
    "neoadjuvant",
    "oncological",
    "oncology",
    "other",
    "pharmacological",
    "pharmacologic",
    "prescribed",
    "preventive",
    "prevention",
    "primary",
    "secondary",
    "standard",
    "systemic",
}

_PROGRAMMATIC_TESTING_TERMS = {
    "cascade",
    "coordinated",
    "family",
    "familial",
    "longitudinal",
    "pathway",
    "programme",
    "program",
    "strategy",
    "structured",
    "systematic",
}


_MEASUREMENT_LIKE_HEADS = {
    "diameter",
    "dimension",
    "flow",
    "fraction",
    "function",
    "index",
    "interval",
    "mass",
    "pressure",
    "ratio",
    "rate",
    "strain",
    "thickness",
    "velocity",
    "volume",
    "work",
}

_PROCESS_LIKE_HEADS = {
    "assessment",
    "counselling",
    "counseling",
    "diagnosis",
    "evaluation",
    "management",
    "monitoring",
    "screening",
    "selection",
    "surveillance",
    "testing",
    "treatment",
}

_MEDICAL_SPELLING_EQUIVALENTS: Tuple[Tuple[str, str], ...] = (
    ("ischaemic", "ischemic"),
    ("ischaemia", "ischemia"),
    ("oedema", "edema"),
    ("haemodynamic", "hemodynamic"),
    ("haemodynamics", "hemodynamics"),
    ("haemorrhage", "hemorrhage"),
    ("tumour", "tumor"),
    ("paediatric", "pediatric"),
)



_TABLE_RE = re.compile(
    r"<table\b[^>]*>.*?</table>",
    re.IGNORECASE | re.DOTALL,
)

_TABLE_CELL_TAGS = {"td", "th"}
_TABLE_ROW_TAGS = {"tr"}
_TABLE_BREAK_TAGS = {"br", "p", "div", "li"}


def _alpha_run_left(text: str, end_exclusive: int, mode: str) -> int:
    """Return the alphabetic run length immediately left of one boundary."""
    i = end_exclusive - 1
    n = 0

    while i >= 0 and text[i].isalpha():
        if mode == "lower" and not text[i].islower():
            break
        if mode == "upper" and not text[i].isupper():
            break

        n += 1
        i -= 1

    return n


def restore_likely_table_case_boundaries(text: str) -> str:
    """
    Restore likely label boundaries lost while linearising table content.

    The transformation is deliberately narrow and is used ONLY on an auxiliary
    validation view of HTML table data. It never modifies ``Section.text``.

    Examples:
        AclarubicinArsenic trioxide -> Aclarubicin Arsenic trioxide
        dystrophySarcoidosis        -> dystrophy Sarcoidosis
        CKLiver function            -> CK Liver function

    Guardrails:
    - lowercase -> uppercase requires at least three lowercase letters on the
      left, avoiding common biomedical forms such as eGFR and iFR;
    - acronym -> Titlecase requires at least two uppercase letters on the left;
    - ordinary prose outside ``<table>`` blocks is never transformed.
    """
    if not text:
        return text

    out: List[str] = []

    for i, ch in enumerate(text):
        insert_space = False

        if i > 0 and ch.isupper():
            prev = text[i - 1]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if prev.islower():
                lower_run = _alpha_run_left(text, i, mode="lower")
                if lower_run >= 3:
                    insert_space = True

            elif prev.isupper() and nxt.islower():
                upper_run = _alpha_run_left(text, i, mode="upper")
                if upper_run >= 2:
                    insert_space = True

        if insert_space and out and not out[-1].isspace():
            out.append(" ")

        out.append(ch)

    return "".join(out)


class _TableValidationProjector(HTMLParser):
    """Convert HTML tables into an auxiliary source-grounding text view."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def _separator(self, token: str = " ") -> None:
        if not self.parts or self.parts[-1] != token:
            self.parts.append(token)

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()

        if tag in _TABLE_CELL_TAGS:
            self._separator(" | ")
        elif tag in _TABLE_ROW_TAGS:
            self._separator(" \n ")
        elif tag in _TABLE_BREAK_TAGS:
            self._separator(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in _TABLE_CELL_TAGS:
            self._separator(" | ")
        elif tag in _TABLE_ROW_TAGS:
            self._separator(" \n ")
        elif tag in _TABLE_BREAK_TAGS:
            self._separator(" ")

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(
                restore_likely_table_case_boundaries(data)
            )

    def text(self) -> str:
        return "".join(self.parts)


def project_structured_tables_for_validation(source_text: str) -> str:
    """
    Build a validation-only projection of HTML tables in ``source_text``.

    The original Section text remains untouched. Cell/row boundaries are made
    explicit and likely case-boundary collisions inside table data are restored.
    """
    blocks = _TABLE_RE.findall(str(source_text or ""))

    if not blocks:
        return ""

    projected: List[str] = []

    for block in blocks:
        parser = _TableValidationProjector()

        try:
            parser.feed(block)
            parser.close()
        except Exception:
            logger.exception(
                "Unable to build structured table validation projection"
            )
            continue

        table_text = parser.text().strip()

        if table_text:
            projected.append(table_text)

    return "\n\n".join(projected)



def normalize_text_for_matching(text: Any) -> str:
    """
    Normalize text conservatively for surface-form matching.

    This function is intentionally less aggressive than semantic normalization:
    it keeps the text recognizable, but removes common PDF/source formatting
    differences that would otherwise create false rejections.
    """
    text = str(text or "").lower()
    text = text.translate(_DASH_TRANSLATION)

    # Join PDF line-break hyphenation:
    # "cardio-\nmyopathy" -> "cardiomyopathy"
    text = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", text)

    # Normalize whitespace only after fixing line-break hyphenation.
    text = normalize_whitespace(text)

    return text


def normalize_name_for_matching(raw_name: Any) -> str:
    """
    Normalize a concept name for support matching.

    This function is intentionally defensive:
    - schema-level normalize_name() is useful for canonical KG names;
    - direct source support matching should also be able to fall back to a
      simple surface-normalized version.
    """
    raw_surface = normalize_text_for_matching(raw_name)
    schema_surface = normalize_name(raw_name)

    if schema_surface:
        return normalize_text_for_matching(schema_surface)

    return raw_surface


def build_candidate_bases_for_matching(raw_name: Any) -> Set[str]:
    """
    Build base candidate strings for direct source matching.

    We include both:
    - simple surface normalization;
    - schema-level normalization.

    This prevents false negatives if normalize_name() changes the surface form
    in a way that is useful for KG storage but too aggressive for regex matching.
    """
    bases: Set[str] = set()

    raw_surface = normalize_text_for_matching(raw_name)
    if raw_surface:
        bases.add(raw_surface)

    schema_from_raw = normalize_name(raw_name)
    if schema_from_raw:
        bases.add(normalize_text_for_matching(schema_from_raw))

    schema_from_surface = normalize_name(raw_surface)
    if schema_from_surface:
        bases.add(normalize_text_for_matching(schema_from_surface))

    return {base for base in bases if base}


def pluralize_token(token: str) -> Optional[str]:
    """
    Very small pluralization helper used only to allow a singular extracted
    concept to match an explicit plural in the source text.

    This is intentionally conservative and incomplete.
    """
    if not token:
        return None

    if token.endswith(("s", "x", "z", "ch", "sh")):
        return token + "es"

    if re.search(r"[^aeiou]y$", token):
        return token[:-1] + "ies"

    return token + "s"


def singularize_token(token: str) -> Optional[str]:
    """
    Very small singularization helper used only to allow a plural extracted
    concept to match an explicit singular in the source text.

    This is intentionally conservative and incomplete.
    """
    if not token or len(token) <= 3:
        return None

    # Avoid bad singularizations:
    # diagnosis -> diagnosi
    # fibrosis -> fibrosi
    # consensus -> consensu
    if token.endswith(("ss", "us", "is")):
        return None

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("es") and len(token) > 4:
        base = token[:-2]
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base

    if token.endswith("s"):
        return token[:-1]

    return None


def build_spelling_variants(text: str) -> Set[str]:
    """
    Build conservative spelling variants for matching only.

    Example:
    - non-ischaemic myocardial scar
    - non-ischemic myocardial scar

    These variants do not change the stored concept name.
    """
    variants = {text}

    for left, right in _MEDICAL_SPELLING_EQUIVALENTS:
        current_variants = list(variants)

        for variant in current_variants:
            if left in variant:
                variants.add(variant.replace(left, right))
            if right in variant:
                variants.add(variant.replace(right, left))

    return {variant for variant in variants if variant}


def build_surface_variants(name: str) -> Set[str]:
    """
    Build a small set of acceptable surface variants for matching.

    We keep this intentionally limited:
    - original normalized forms;
    - conservative spelling variants;
    - a simple pluralized last-token variant;
    - a simple singularized last-token variant.

    This is still deterministic support checking, not fuzzy matching.
    """
    variants: Set[str] = set()

    for base in build_candidate_bases_for_matching(name):
        for spelling_variant in build_spelling_variants(base):
            variants.add(spelling_variant)

            parts = spelling_variant.split()
            if not parts:
                continue

            plural_last = pluralize_token(parts[-1])
            if plural_last:
                variants.add(" ".join(parts[:-1] + [plural_last]))

            singular_last = singularize_token(parts[-1])
            if singular_last:
                variants.add(" ".join(parts[:-1] + [singular_last]))

    return {variant for variant in variants if variant}


def concept_surface_to_pattern(surface: str) -> Optional[str]:
    """
    Convert a normalized surface form into a conservative regex pattern.

    Instead of relying on re.escape() preserving spaces in a specific way, we
    split on spaces/hyphens and join tokens with a flexible separator.

    Example:
    - "bone tracer scintigraphy" matches "bone-tracer scintigraphy"
    - "non ischaemic scar" matches "non-ischaemic scar"
    """
    surface = normalize_text_for_matching(surface)

    tokens = [
        re.escape(token)
        for token in re.split(r"[\s\-]+", surface)
        if token
    ]

    if not tokens:
        return None

    flexible = r"[\s\-]+".join(tokens)

    # Prevent matching inside larger alphanumeric strings.
    return rf"(?<![a-z0-9]){flexible}(?![a-z0-9])"


def build_support_patterns(name: str) -> List[str]:
    """
    Build conservative regex patterns for matching a concept name in source text.

    Allowed flexibility:
    - spaces and hyphens can match each other;
    - boundaries prevent accidental substring matches inside larger words.

    Important:
    This does not try to infer semantic equivalence.
    """
    patterns: List[str] = []

    for variant in build_surface_variants(name):
        pattern = concept_surface_to_pattern(variant)

        if pattern:
            patterns.append(pattern)

    # Preserve order but remove duplicates.
    seen = set()
    unique_patterns = []

    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)

    return unique_patterns


def _get_direct_source_support_from_one_view(
    name: str,
    source_text: str,
) -> Optional[Dict[str, Any]]:
    """Run the existing strict surface matcher against one source view."""
    normalized_source = normalize_text_for_matching(source_text)

    if not normalized_source:
        return None

    patterns = build_support_patterns(name)

    if not patterns:
        return None

    for pattern in patterns:
        match = re.search(pattern, normalized_source)

        if match:
            return {
                "support_method": "direct_source",
                "support_reason": "accepted_by_direct_source_match",
                "matched_text": match.group(0),
                "matched_pattern": pattern,
            }

    return None


def get_direct_source_support(name: str, source_text: str) -> Optional[Dict[str, Any]]:
    """
    Return deterministic direct source evidence for a concept name.

    The ordinary Section source is checked first with the existing strict
    matcher. If that fails, HTML tables are projected into a validation-only
    view that preserves cell/row boundaries and restores likely lost label
    boundaries. The SAME strict surface matcher is then applied to that view.

    This is intentionally monotonic:
    - all previously supported direct matches are unchanged;
    - ordinary prose is never made more permissive;
    - no fuzzy/semantic matching is introduced;
    - ``Section.text`` itself is never modified.
    """
    direct_support = _get_direct_source_support_from_one_view(
        name=name,
        source_text=source_text,
    )

    if direct_support is not None:
        return direct_support

    table_projection = project_structured_tables_for_validation(source_text)

    if not table_projection:
        return None

    table_support = _get_direct_source_support_from_one_view(
        name=name,
        source_text=table_projection,
    )

    if table_support is None:
        return None

    return {
        **table_support,
        # Keep support_method=direct_source so existing evidence priority and
        # MENTIONS logic remain unchanged. The validation_reason records the
        # structured-table provenance explicitly.
        "support_reason": "accepted_by_structured_table_source_match",
    }


def is_supported_by_source(name: str, source_text: str) -> bool:
    """
    Check whether the concept name is explicitly supported by the source text.

    Kept as a boolean convenience wrapper for older callers/tests.
    """
    return get_direct_source_support(
        name=name,
        source_text=source_text,
    ) is not None


def get_support_evidence(
    name: str,
    source_text: str,
    acronyms: Optional[Dict[str, str]] = None,
    concept_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return deterministic support evidence for a concept name.

    A concept is supported if either:
    - its surface form appears directly in the section source text;
    - an acronym short form appears in the section source text and expands to
      the concept name according to the document acronym cache.

    Returns a small metadata dictionary if support is found, otherwise None.
    """
    direct_support = get_direct_source_support(
        name=name,
        source_text=source_text,
    )

    if direct_support is not None:
        return direct_support

    if acronyms:
        acronym_support = get_acronym_support_for_concept(
            concept_name=name,
            source_text=source_text,
            acronyms=acronyms,
        )

        if acronym_support is not None:
            return {
                "support_method": "acronym",
                "support_reason": "accepted_by_acronym_support",
                **acronym_support,
            }

        embedded_support = get_embedded_acronym_support(
            name=name,
            source_text=source_text,
            acronyms=acronyms,
            concept_type=concept_type,
        )

        if embedded_support is not None:
            return embedded_support

    return None


def get_raw_name_for_acronym_check(concept: Dict[str, Any]) -> Any:
    """
    Return the original LLM surface name when available.

    This matters because add_entities.py normalizes LLM output before calling
    validation. For example:

        raw_name = "AS"
        name = "as"

    Acronym expansion must inspect "AS", not only normalized "as".
    """
    raw_name = concept.get("raw_name")

    if raw_name not in (None, ""):
        return raw_name

    return concept.get("name")


def copy_raw_fields(
    normalized_concept: Dict[str, Any],
    source_concept: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Preserve raw_name/raw_type metadata on validation records when available.
    """
    out = dict(normalized_concept)

    for field in ("raw_name", "raw_type"):
        value = source_concept.get(field)

        if value not in (None, ""):
            out[field] = str(value)

    return out


def get_exact_case_source_support(
    surface: Any,
    source_text: str,
) -> Optional[Dict[str, Any]]:
    """Return exact case-sensitive source evidence for a short symbol."""
    raw_surface = normalize_whitespace(str(surface or "")).strip()
    if not raw_surface:
        return None

    pattern = (
        rf"(?<![A-Za-z0-9]){re.escape(raw_surface)}"
        rf"(?![A-Za-z0-9])"
    )
    match = re.search(pattern, str(source_text or ""))
    if match is None:
        return None

    return {
        "support_method": "direct_source",
        "support_reason": "accepted_gene_symbol_direct_source",
        "matched_text": match.group(0),
        "matched_pattern": pattern,
    }


def try_accept_raw_gene_symbol_concept(
    concept: Dict[str, Any],
    source_text: str,
    concept_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Accept an exact uppercase gene symbol without acronym expansion.

    The intrinsic type must be genetic_factor and the exact symbol must occur
    in the Section source with the same capitalization.
    """
    if concept_type != "genetic_factor":
        return None

    raw_symbol = normalize_whitespace(
        str(get_raw_name_for_acronym_check(concept) or "")
    ).strip()

    if _GENE_SYMBOL_PATTERN.fullmatch(raw_symbol) is None:
        return None

    support_evidence = get_exact_case_source_support(
        surface=raw_symbol,
        source_text=source_text,
    )
    if support_evidence is None:
        return None

    accepted = copy_raw_fields(
        normalized_concept={
            "name": raw_symbol,
            "type": concept_type,
        },
        source_concept=concept,
    )
    return build_accepted_concept(
        normalized_concept=accepted,
        support_evidence=support_evidence,
    )


def is_non_population_care_setting(
    name: Any,
    concept_type: str,
) -> bool:
    """Detect an obvious care setting misclassified as a population."""
    if concept_type != "population_or_patient_group":
        return False

    tokens = _normalized_name_tokens(name)
    if not tokens:
        return False

    if set(tokens).intersection(_HUMAN_GROUP_MARKERS):
        return False

    return tokens[-1] in _CARE_SETTING_HEADS


def is_organization_like_name(
    name: Any,
    concept_type: Optional[str] = None,
) -> bool:
    """
    Detect organizations and scientific bodies outside the entity schema.

    Named scores and criteria may legitimately contain an organization name in
    their title, for example ``Society of Thoracic Surgeons Predicted Risk of
    Mortality`` or ``revised Task Force criteria``. Those are clinical models,
    not organization nodes, and must remain eligible when the extracted type is
    ``score_or_risk_model``.
    """
    tokens = _normalized_name_tokens(name)

    if not tokens:
        return False

    normalized = " ".join(tokens)

    if concept_type == "score_or_risk_model":
        if set(tokens).intersection(_SCORE_ARTIFACT_MARKERS):
            return False

    if tokens[-1] in _ORGANIZATION_HEADS:
        return True

    return any(
        marker in normalized
        for marker in _ORGANIZATION_PHRASE_MARKERS
    )


def is_generic_population_name(
    name: Any,
    concept_type: str,
) -> bool:
    if concept_type != "population_or_patient_group":
        return False

    normalized = " ".join(_normalized_name_tokens(name))
    return normalized in _GENERIC_POPULATION_NAMES


def is_non_entity_discipline(name: Any) -> bool:
    normalized = " ".join(_normalized_name_tokens(name))
    return normalized in _NON_ENTITY_DISCIPLINE_NAMES


def is_non_anatomical_adjective(
    name: Any,
    concept_type: str,
) -> bool:
    if concept_type != "anatomical_structure":
        return False

    normalized = " ".join(_normalized_name_tokens(name))
    return normalized in _NON_ANATOMICAL_ADJECTIVE_NAMES


def is_nonclinical_research_or_variable(
    name: Any,
    concept_type: str,
) -> bool:
    """
    Detect research objects and generic variables outside the entity schema.

    Trial and registry names are never intrinsic clinical entities in the
    current schema, regardless of the type proposed by the LLM. ``Study`` is
    more nuanced because some established diagnostic procedures use that word,
    such as ``electrophysiological study``.
    """
    tokens = _normalized_name_tokens(name)

    if not tokens:
        return False

    normalized = " ".join(tokens)

    if normalized in _NONCLINICAL_RESEARCH_PHRASES:
        return True

    if normalized in _GENERIC_VARIABLE_NAMES:
        return True

    if tokens[-1] in _ALWAYS_NON_ENTITY_RESEARCH_HEADS:
        return True

    if tokens[-1] in {"study", "studies"}:
        if (
            concept_type in {"diagnostic_test", "procedure_or_intervention"}
            and bool(set(tokens[:-1]).intersection(
                _CLINICAL_DIAGNOSTIC_STUDY_MARKERS
            ))
        ):
            return False

        return True

    return (
        concept_type in {"clinical_finding", "diagnostic_test"}
        and tokens[-1] in _NONCLINICAL_RESEARCH_HEADS
    )


def is_document_or_publication_like_name(name: Any) -> bool:
    """Detect guideline, consensus, and publication titles outside the schema."""
    normalized = " ".join(_normalized_name_tokens(name))

    if not normalized:
        return False

    return any(
        pattern.search(normalized) is not None
        for pattern in _PUBLICATION_TITLE_PATTERNS
    )


def is_generic_process_entity(
    name: Any,
    concept_type: str,
) -> bool:
    """Detect broad process labels such as ``cancer diagnosis``."""
    if concept_type not in {
        "care_strategy",
        "clinical_finding",
        "clinical_outcome",
        "diagnostic_test",
        "disease",
        "procedure_or_intervention",
    }:
        return False

    tokens = _normalized_name_tokens(name)

    if not tokens or tokens[-1] not in _GENERIC_PROCESS_HEADS:
        return False

    modifiers = tokens[:-1]

    return not modifiers or all(
        token in _GENERIC_PROCESS_MODIFIERS
        for token in modifiers
    )


def is_generic_biomarker_category(
    name: Any,
    concept_type: str,
) -> bool:
    if concept_type != "biomarker":
        return False

    tokens = _normalized_name_tokens(name)

    if not tokens or tokens[-1] not in _GENERIC_BIOMARKER_HEADS:
        return False

    modifiers = tokens[:-1]

    return not modifiers or all(
        token in _GENERIC_BIOMARKER_MODIFIERS
        for token in modifiers
    )


def is_treatment_misclassified_as_exposure(
    name: Any,
    concept_type: str,
) -> bool:
    if concept_type != "exposure_or_lifestyle_factor":
        return False

    token_set = set(_normalized_name_tokens(name))

    return (
        bool(token_set.intersection(_TREATMENT_MODALITY_TOKENS))
        and not bool(token_set.intersection(_EXPOSURE_CONTEXT_TOKENS))
    )


def is_procedure_or_modality_misclassified_as_drug(
    name: Any,
    concept_type: str,
) -> bool:
    if concept_type != "drug_or_drug_class":
        return False

    normalized = " ".join(_normalized_name_tokens(name))
    return normalized in _NON_DRUG_TREATMENT_MODALITIES


def is_generic_therapeutic_entity(
    name: Any,
    concept_type: str,
) -> bool:
    return (
        concept_type
        in {
            "care_strategy",
            "drug_or_drug_class",
            "procedure_or_intervention",
        }
        and is_generic_therapeutic_term(name)
    )


def is_unsafe_acronym_definition(
    short: Any,
    definition: Any,
) -> bool:
    """Return True for a narrowly recognized malformed cache definition."""
    clean_short = clean_acronym_short(short)
    clean_definition = normalize_text_for_matching(
        clean_acronym_definition(definition)
    )

    if (clean_short, clean_definition) in _UNSAFE_ACRONYM_DEFINITION_EXACT:
        return True

    # Defensive version of the observed corruption: ECG must not merge
    # electrocardiography and echocardiography into one definition.
    if clean_short == "ECG":
        padded = f" {clean_definition} "
        return (
            "electrocardiogram" in clean_definition
            and (
                "echocardiography" in clean_definition
                or " echocardiogram " in padded
                or " echo " in padded
            )
        )

    return False


def filter_safe_acronyms(
    acronyms: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Return a copy of the acronym map without unsafe definitions."""
    safe: Dict[str, str] = {}

    for raw_short, raw_definition in (acronyms or {}).items():
        short = clean_acronym_short(raw_short)
        definition = clean_acronym_definition(raw_definition)

        if not short or not definition:
            continue

        if is_unsafe_acronym_definition(short, definition):
            continue

        safe[short] = definition

    return safe


def raw_short_has_unsafe_cached_definition(
    raw_short: Any,
    acronyms: Optional[Dict[str, str]],
) -> bool:
    """Check whether the raw short maps to a known-unsafe definition."""
    wanted = clean_acronym_short(raw_short)

    if not wanted:
        return False

    for raw_key, raw_definition in (acronyms or {}).items():
        key = clean_acronym_short(raw_key)

        if key != wanted:
            continue

        return is_unsafe_acronym_definition(
            short=key,
            definition=raw_definition,
        )

    return False


def canonicalize_embedded_source_name(
    matched_text: str,
    acronym_short: str,
) -> str:
    """Lowercase ordinary words while preserving the acronym short form."""
    text = normalize_text_preserve_case(matched_text)
    short = clean_acronym_short(acronym_short)

    if not text or not short:
        return text

    pattern = _mixed_phrase_acronym_body(short)
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match is None:
        return text

    return (
        text[:match.start()].casefold()
        + short
        + text[match.end():].casefold()
    ).strip()


def should_preserve_embedded_source_name(
    definition: str,
    prefix: str,
    suffix: str,
    concept_type: Optional[str],
) -> bool:
    """
    Preserve the source phrase when literal expansion creates an unnatural
    classifier construction such as ``RAF inhibitor`` or ``ICI myocarditis``.
    """
    del prefix

    suffix_tokens = _normalized_name_tokens(suffix)
    definition_tokens = _normalized_name_tokens(definition)

    if not suffix_tokens:
        return False

    suffix_head = suffix_tokens[-1]
    definition_head = definition_tokens[-1] if definition_tokens else ""

    if suffix_head in _EMBEDDED_SOURCE_NAME_CLASS_HEADS:
        return True

    return (
        concept_type == "disease"
        and suffix_head in _EMBEDDED_ASSOCIATED_DISEASE_HEADS
        and definition_head in _EMBEDDED_THERAPEUTIC_DEFINITION_HEADS
    )


def normalize_text_preserve_case(text: Any) -> str:
    """Normalize PDF whitespace and dashes while retaining acronym case."""
    text = str(text or "").translate(_DASH_TRANSLATION)
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)

    return normalize_whitespace(text)


def _mixed_phrase_surface_body(surface: str) -> str:
    tokens = [
        re.escape(token)
        for token in re.split(r"[\s\-]+", surface)
        if token
    ]

    return r"[\s\-]+".join(tokens)


def _mixed_phrase_acronym_body(short: str) -> str:
    pieces: List[str] = []
    previous_separator = False

    for char in short:
        if char.isspace() or char == "-":
            if not previous_separator:
                pieces.append(r"[\s\-]+")
                previous_separator = True

            continue

        pieces.append(re.escape(char))
        previous_separator = False

    return "".join(pieces)


def get_embedded_acronym_support(
    name: str,
    source_text: str,
    acronyms: Optional[Dict[str, str]],
    concept_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Support a longer phrase when one cached acronym replaces one explicit
    long-form component in the source.

    This remains equality-based and does not use fuzzy semantic matching.
    """
    if not name or not source_text or not acronyms:
        return None

    concept_surface = normalize_text_for_matching(name)
    source_surface = normalize_text_preserve_case(source_text)

    if not concept_surface or not source_surface:
        return None

    for raw_short, raw_definition in sorted(acronyms.items()):
        short = clean_acronym_short(raw_short)
        definition = clean_acronym_definition(raw_definition)

        if not short or not definition:
            continue

        if is_unsafe_acronym_definition(short, definition):
            continue

        short_body = _mixed_phrase_acronym_body(short)

        if not short_body:
            continue

        for definition_surface in build_surface_variants(definition):
            definition_pattern = concept_surface_to_pattern(
                definition_surface
            )

            if not definition_pattern:
                continue

            match = re.search(definition_pattern, concept_surface)

            if match is None:
                continue

            prefix = concept_surface[:match.start()].strip(" -")
            suffix = concept_surface[match.end():].strip(" -")
            parts: List[str] = []

            if prefix:
                prefix_body = _mixed_phrase_surface_body(prefix)

                if prefix_body:
                    parts.append(f"(?i:{prefix_body})")

            parts.append(short_body)

            if suffix:
                suffix_body = _mixed_phrase_surface_body(suffix)

                if suffix_body:
                    parts.append(f"(?i:{suffix_body})")

            body = r"[\s\-]+".join(parts)
            pattern = (
                rf"(?<![A-Za-z0-9]){body}"
                rf"(?![A-Za-z0-9])"
            )
            source_match = re.search(pattern, source_surface)

            if source_match is None:
                continue

            matched_text = source_match.group(0)
            preferred_name: Optional[str] = None

            if should_preserve_embedded_source_name(
                definition=definition_surface,
                prefix=prefix,
                suffix=suffix,
                concept_type=concept_type,
            ):
                preferred_name = canonicalize_embedded_source_name(
                    matched_text=matched_text,
                    acronym_short=short,
                )

            return {
                "support_method": "acronym",
                "support_reason": (
                    "accepted_by_embedded_acronym_support"
                ),
                "matched_text": matched_text,
                "matched_pattern": pattern,
                "preferred_name": preferred_name,
                "acronym_short": short,
                "acronym_definition": definition,
                "acronym_match_method": (
                    "embedded_definition_replaced_by_short"
                ),
                "expanded_from_acronym": False,
            }

    return None


def get_cached_definition_for_short(
    raw_short: Any,
    acronyms: Optional[Dict[str, str]],
) -> Optional[str]:
    """Return an exact cleaned cache definition for one acronym short."""
    wanted = clean_acronym_short(raw_short)

    if not wanted:
        return None

    for raw_key, raw_definition in (acronyms or {}).items():
        key = clean_acronym_short(raw_key)
        definition = clean_acronym_definition(raw_definition)

        if key == wanted and definition:
            return definition

    return None


def try_accept_named_acronym_score_concept(
    concept: Dict[str, Any],
    source_text: str,
    concept_type: str,
    acronyms: Optional[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Accept ``<ACRONYM> score`` when the source contains the acronym short,
    even if the literal word ``score`` is absent.
    """
    if concept_type != "score_or_risk_model":
        return None

    raw_surface = normalize_whitespace(
        str(get_raw_name_for_acronym_check(concept) or "")
    ).strip()

    # Prefer ordinary direct grounding when the complete phrase is explicitly
    # present. The acronym-only fallback is used only when the source contains
    # the short form but omits the literal word "score".
    if get_direct_source_support(
        name=raw_surface,
        source_text=source_text,
    ) is not None:
        return None

    match = re.fullmatch(
        r"(?P<short>.+?)\s+score",
        raw_surface,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    short = clean_acronym_short(match.group("short"))

    if not is_likely_acronym_short_form(short):
        return None

    source_support = get_exact_case_source_support(
        surface=short,
        source_text=source_text,
    )

    if source_support is None:
        return None

    definition = get_cached_definition_for_short(
        raw_short=short,
        acronyms=acronyms,
    )

    if definition and is_unsafe_acronym_definition(short, definition):
        definition = None

    accepted = copy_raw_fields(
        normalized_concept={
            "name": f"{short} score",
            "type": concept_type,
        },
        source_concept=concept,
    )

    return build_accepted_concept(
        normalized_concept=accepted,
        support_evidence={
            **source_support,
            "support_method": "acronym",
            "support_reason": "accepted_named_score_from_acronym",
            "acronym_short": short,
            "acronym_definition": definition,
            "acronym_match_method": (
                "acronym_short_in_source_with_score_suffix"
            ),
            "expanded_from_acronym": False,
        },
    )


def try_expand_raw_acronym_concept(
    concept: Dict[str, Any],
    source_text: str,
    concept_type: str,
    acronyms: Optional[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Try to expand a raw acronym-only LLM concept into a long-form concept.

    Example:
        incoming concept:
            {"name": "as", "type": "disease", "raw_name": "AS"}

        acronym cache:
            AS -> Aortic stenosis

        returned concept:
            {"name": "aortic stenosis", "type": "disease", ...}

    Returns None if no safe expansion is available.
    """
    raw_name_for_acronym = get_raw_name_for_acronym_check(concept)

    if not raw_name_for_acronym or not acronyms:
        return None

    expansion = get_acronym_expansion_for_short(
        raw_short=raw_name_for_acronym,
        source_text=source_text,
        acronyms=acronyms,
        require_source_presence=True,
    )

    if expansion is None:
        return None

    expanded_name = normalize_name(expansion.get("expanded_name"))

    if not expanded_name:
        return None

    expanded_concept = {
        "name": expanded_name,
        "type": concept_type,
    }

    expanded_concept = copy_raw_fields(
        normalized_concept=expanded_concept,
        source_concept=concept,
    )

    support_evidence = {
        "support_method": "acronym",
        "support_reason": "accepted_by_acronym_expansion",
        "acronym_short": expansion.get("acronym_short"),
        "acronym_definition": expansion.get("acronym_definition"),
        "acronym_match_method": expansion.get("acronym_match_method"),
        "expanded_from_acronym": True,
    }

    return build_accepted_concept(
        normalized_concept=expanded_concept,
        support_evidence=support_evidence,
    )


def raw_name_is_unexpanded_acronym_short_form(concept: Dict[str, Any]) -> bool:
    """
    Return True only when the whole original surface is acronym-like.

    Phrases containing an acronym plus ordinary words, such as ``12-lead ECG``
    or ``HAS-BLED score``, are not acronym-only concepts and can be grounded by
    their complete explicit source surface.
    """
    raw_name_for_acronym = get_raw_name_for_acronym_check(concept)
    raw_surface = normalize_whitespace(
        str(raw_name_for_acronym or "")
    ).strip()

    if not raw_surface:
        return False

    tokens = raw_surface.split()

    if len(tokens) > 1 and any(
        re.search(r"[a-z]{2,}", token)
        for token in tokens
    ):
        return False

    return is_likely_acronym_short_form(raw_surface)


def is_unsafe_short_surface_name_for_direct_match(name: Any) -> bool:
    """
    Reject tiny lowercase direct-source concepts such as "as".

    These are too ambiguous to accept by surface matching. If they are real
    acronyms, they should be accepted through acronym expansion instead.

    This specifically fixes cases where:
        concept name = "as"
        source text contains the ordinary word "as"

    Without this guard, direct source matching would incorrectly accept "as".
    """
    surface = normalize_text_for_matching(name)

    if re.fullmatch(r"[a-z]{1,2}", surface):
        return True

    return False


def build_accepted_concept(
    normalized_concept: Dict[str, Any],
    support_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the accepted concept payload while preserving validation evidence.

    These fields can later be written to MENTIONS relationships and/or exported
    to review JSONL files.
    """
    accepted_concept: Dict[str, Any] = {
        **normalized_concept,
        "validation_reason": support_evidence.get("support_reason", "accepted"),
        "support_method": support_evidence.get("support_method", "unknown"),
    }

    preferred_name = support_evidence.get("preferred_name")

    if preferred_name:
        accepted_concept["name"] = str(preferred_name)

    evidence_keys = [
        "matched_text",
        "matched_pattern",
        "acronym_short",
        "acronym_definition",
        "acronym_match_method",
        "expanded_from_acronym",
    ]

    for key in evidence_keys:
        if support_evidence.get(key) is not None:
            accepted_concept[key] = support_evidence[key]

    return accepted_concept


def build_rejected_concept_record(
    normalized_concept: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    """
    Build a rejected concept record while preserving raw_name/raw_type when
    available.

    This makes review exports easier to interpret, especially for cases where:
        raw_name = "AS"
        normalized name = "as"
    """
    rejected = {
        "name": normalized_concept.get("name", ""),
        "type": normalized_concept.get("type", ""),
        "reason": reason,
    }

    for field in ("raw_name", "raw_type"):
        value = normalized_concept.get(field)

        if value not in (None, ""):
            rejected[field] = value

    quality_flags = normalized_concept.get("quality_flags")
    if isinstance(quality_flags, (list, tuple, set)):
        normalized_flags = [
            str(flag).strip()
            for flag in quality_flags
            if str(flag).strip()
        ]
        if normalized_flags:
            rejected["quality_flags"] = list(dict.fromkeys(normalized_flags))

    return rejected


_ACRONYM_EVIDENCE_FIELDS = {
    "acronym_short",
    "acronym_definition",
    "acronym_match_method",
}


def normalize_mention_evidence(concept: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return one internally coherent accepted-concept evidence record.

    A direct-source mention must not retain acronym-expansion metadata copied
    from another duplicate candidate. Conversely, a record marked as expanded
    from an acronym is valid only when its raw surface is the acronym short
    form. This defensive normalization protects both review exports and future
    MENTIONS writes.
    """
    out = dict(concept or {})
    support_method = str(out.get("support_method") or "").strip()

    if support_method == "direct_source":
        for field in _ACRONYM_EVIDENCE_FIELDS:
            out.pop(field, None)
        out["expanded_from_acronym"] = False
        return out

    if support_method == "acronym":
        expanded = bool(out.get("expanded_from_acronym", False))
        raw_name = normalize_whitespace(str(out.get("raw_name") or ""))
        acronym_short = normalize_whitespace(
            str(out.get("acronym_short") or "")
        )

        if expanded and (
            not raw_name
            or not acronym_short
            or raw_name.casefold() != acronym_short.casefold()
        ):
            # The concept long form was supported by an acronym in the source,
            # but the LLM did not actually emit the short form itself.
            out["expanded_from_acronym"] = False

    return out


def mention_evidence_priority(concept: Dict[str, Any]) -> Tuple[int, int]:
    """Return a deterministic preference score for duplicate evidence records."""
    normalized = normalize_mention_evidence(concept)
    support_method = normalized.get("support_method")

    if support_method == "direct_source":
        primary = 3
    elif support_method == "acronym" and normalized.get(
        "expanded_from_acronym"
    ):
        primary = 2
    elif support_method == "acronym":
        primary = 1
    else:
        primary = 0

    evidence_fields = (
        "matched_text",
        "matched_pattern",
        "acronym_short",
        "acronym_definition",
        "acronym_match_method",
        "raw_name",
        "raw_type",
    )
    richness = sum(
        1
        for field in evidence_fields
        if normalized.get(field) not in (None, "")
    )
    return primary, richness


def merge_validated_concept_evidence(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge duplicate accepted records without mixing incompatible provenance.

    Only one complete evidence record wins. Non-provenance review flags are
    unioned, but support_method/raw_name/acronym fields are never assembled
    piecemeal from different candidates.
    """
    left = normalize_mention_evidence(existing)
    right = normalize_mention_evidence(incoming)

    if mention_evidence_priority(right) > mention_evidence_priority(left):
        winner, loser = right, left
    else:
        winner, loser = left, right

    merged = dict(winner)

    merged_flags: List[str] = []
    for source in (winner, loser):
        for flag in source.get("quality_flags") or []:
            clean_flag = str(flag).strip()
            if clean_flag and clean_flag not in merged_flags:
                merged_flags.append(clean_flag)

    if merged_flags:
        merged["quality_flags"] = merged_flags
    else:
        merged.pop("quality_flags", None)

    return normalize_mention_evidence(merged)


def deduplicate_validated_concepts(
    concepts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deduplicate accepted concepts while preserving validation metadata.

    We avoid using the schema-level deduplicate_concepts() here because some
    schema-level helpers may normalize concepts down to only {"name", "type"}.
    At this stage we want to keep support_method, validation_reason, acronym
    evidence, raw_name/raw_type, and matched_text/matched_pattern for graph
    writing and review.
    """
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for concept in concepts or []:
        name = concept.get("name", "")
        concept_type = concept.get("type", "")
        key = (name, concept_type)

        if key not in deduped:
            deduped[key] = normalize_mention_evidence(concept)
            continue

        deduped[key] = merge_validated_concept_evidence(
            deduped[key],
            concept,
        )

    return list(deduped.values())



def collapse_validated_concepts_by_name(
    concepts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse accepted type assertions to one coherent row per Concept name.

    Validation intentionally deduplicates by ``(name, type)`` so type evidence
    is not lost. Neo4j stores one ``MENTIONS`` relationship per Section--Concept
    pair, therefore singular provenance fields must come from one complete
    evidence record while ``observed_types`` preserves every accepted type.
    """
    collapsed: Dict[str, Dict[str, Any]] = {}

    for concept in concepts or []:
        if not isinstance(concept, dict):
            continue

        name = str(concept.get("name") or "").strip()
        concept_type = str(concept.get("type") or "").strip()
        if not name or not concept_type:
            continue

        normalized = normalize_mention_evidence(concept)

        if name not in collapsed:
            row = dict(normalized)
            row["observed_types"] = [concept_type]
            collapsed[name] = row
            continue

        existing = collapsed[name]
        observed_types = list(existing.get("observed_types") or [])
        if concept_type not in observed_types:
            observed_types.append(concept_type)

        merged = merge_validated_concept_evidence(existing, normalized)
        merged["observed_types"] = observed_types
        collapsed[name] = merged

    return list(collapsed.values())



def _normalized_name_tokens(name: Any) -> List[str]:
    """
    Return conservative lowercase tokens for semantic review rules.
    """
    normalized = normalize_text_for_matching(name)
    return [
        token
        for token in re.split(r"[\s\-/]+", normalized)
        if token
    ]


def _quality_flag_token_variants(tokens: List[str]) -> Set[str]:
    """
    Build token variants for non-blocking semantic quality checks.

    Both the observed token and a conservative singular form are included.
    This lets rules recognize plural forms such as "carbonyls",
    "particulates", or "measurements" without rewriting the concept name.
    """
    variants: Set[str] = set(tokens)

    for token in tokens:
        singular = singularize_token(token)
        if singular:
            variants.add(singular)

    return variants


def is_generic_causal_description(name: Any) -> bool:
    """
    Detect a narrow class of generic causal or aetiological descriptions.

    Examples of the intended pattern are phrases whose semantic head is only
    "cause", "aetiology", "mechanism", "basis", or a similar generic term,
    and whose preceding words merely describe the kind of causation.

    This deliberately does not reject phrases that contain a named gene,
    pathway, disease, exposure, or other specific reusable entity.
    """
    tokens = _normalized_name_tokens(name)

    if not tokens or len(tokens) > 4:
        return False

    if tokens[-1] not in _GENERIC_CAUSAL_HEADS:
        return False

    modifiers = tokens[:-1]
    return not modifiers or all(token in _GENERIC_CAUSAL_MODIFIERS for token in modifiers)


def is_generic_therapeutic_term(name: Any) -> bool:
    """
    Flag broad therapeutic phrases that do not identify a reusable drug,
    drug class, named intervention, or care process.
    """
    tokens = _normalized_name_tokens(name)

    if not tokens or len(tokens) > 4:
        return False

    if tokens[-1] not in _GENERIC_THERAPEUTIC_HEADS:
        return False

    modifiers = tokens[:-1]
    return not modifiers or all(
        token in _GENERIC_THERAPEUTIC_MODIFIERS
        for token in modifiers
    )


def build_semantic_quality_flags(
    name: str,
    concept_type: str,
    source_text: str,
) -> List[str]:
    """
    Attach conservative semantic review flags to an already grounded concept.

    Quality flags do not change the concept name or type and do not reject the
    concept. They make potentially inconsistent classifications visible in
    review exports and on MENTIONS relationships when the caller preserves the
    returned metadata.
    """
    del source_text  # Reserved for future context-window checks.

    tokens = _normalized_name_tokens(name)
    token_set = _quality_flag_token_variants(tokens)
    flags: List[str] = []

    if is_generic_therapeutic_term(name):
        flags.append("generic_therapeutic_term")

    if concept_type == "diagnostic_test" and token_set.intersection(_RESULT_LIKE_HEADS):
        flags.append("possible_result_test_mismatch")

    if concept_type == "disease" and token_set.intersection(_RESULT_LIKE_HEADS):
        flags.append("possible_disease_finding_mismatch")

    if concept_type == "biomarker" and token_set.intersection(_EXPOSURE_LIKE_HEADS):
        flags.append("possible_exposure_biomarker_mismatch")

    if concept_type == "biomarker" and token_set.intersection(_MEASUREMENT_LIKE_HEADS):
        flags.append("possible_measurement_biomarker_mismatch")

    if concept_type == "clinical_outcome" and token_set.intersection(_PROCESS_LIKE_HEADS):
        flags.append("possible_process_outcome_mismatch")

    if concept_type == "device" and token_set.intersection(_NONMEDICAL_DEVICE_TERMS):
        flags.append("possible_nonmedical_device")

    if (
        concept_type == "clinical_finding"
        and token_set.intersection({"abuse", "consumption", "smoking", "use", "vaping"})
    ):
        flags.append("possible_exposure_finding_mismatch")

    if (
        concept_type == "care_strategy"
        and tokens
        and tokens[-1] == "testing"
        and not token_set.intersection(_PROGRAMMATIC_TESTING_TERMS)
    ):
        flags.append("possible_testing_strategy_mismatch")

    # Preserve stable order and avoid duplicates.
    return list(dict.fromkeys(flags))


def attach_quality_flags(
    concept: Dict[str, Any],
    source_text: str,
) -> Dict[str, Any]:
    """
    Return a copy of a concept with semantic quality flags when applicable.
    """
    out = dict(concept)
    flags = build_semantic_quality_flags(
        name=str(out.get("name") or ""),
        concept_type=str(out.get("type") or ""),
        source_text=source_text,
    )

    if flags:
        out["quality_flags"] = flags

    return out


def get_out_of_schema_rejection(
    name: str,
    concept_type: str,
    blocklist_names: Set[str],
) -> Optional[Dict[str, Any]]:
    """
    Return a deterministic rejection for document/research objects.

    This helper is intentionally applied both to ordinary concepts and to
    acronym-expanded concepts so acronym expansion cannot bypass schema-level
    exclusions.
    """
    if name in blocklist_names:
        return {
            "reason": "blocklisted_name",
            "quality_flags": [],
        }

    if is_organization_like_name(name, concept_type=concept_type):
        return {
            "reason": "organization_not_supported_entity_type",
            "quality_flags": ["organization_outside_entity_schema"],
        }

    if is_document_or_publication_like_name(name):
        return {
            "reason": "document_or_publication_not_entity",
            "quality_flags": ["document_or_publication_outside_entity_schema"],
        }

    if is_nonclinical_research_or_variable(
        name=name,
        concept_type=concept_type,
    ):
        return {
            "reason": "nonclinical_research_or_variable",
            "quality_flags": ["nonclinical_research_or_variable"],
        }

    return None


def reject_with_semantic_metadata(
    normalized_concept: Dict[str, Any],
    rejection: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a standard validate_single_concept rejection payload."""
    concept = dict(normalized_concept)
    flags = rejection.get("quality_flags") or []
    if flags:
        concept["quality_flags"] = list(dict.fromkeys(flags))

    return {
        "accepted": False,
        "concept": concept,
        "reason": str(rejection["reason"]),
    }


def validate_single_concept(
    concept: Dict[str, Any],
    source_text: str,
    allowed_types: Set[str],
    blocklist_names: Set[str],
    acronyms: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Validate one concept against deterministic pre-write rules.

    Returns:
    {
        "accepted": bool,
        "concept": {"name": ..., "type": ...},
        "reason": "<reason_code>"
    }

    If accepted, the returned concept may also contain:
    - validation_reason
    - support_method
    - matched_text
    - matched_pattern
    - acronym_short
    - acronym_definition
    - raw_name
    - raw_type
    """
    if not isinstance(concept, dict):
        return {
            "accepted": False,
            "concept": {"name": "", "type": ""},
            "reason": "invalid_payload",
        }

    raw_name = concept.get("name")
    raw_type = concept.get("type")

    name = normalize_name(raw_name)
    concept_type = normalize_type(raw_type)

    normalized_concept = {
        "name": name,
        "type": concept_type,
    }

    normalized_concept = copy_raw_fields(
        normalized_concept=normalized_concept,
        source_concept=concept,
    )

    if not name:
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "empty_name",
        }

    if not concept_type:
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "empty_type",
        }

    if concept_type not in allowed_types:
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "non_allowed_type",
        }

    safe_acronyms = filter_safe_acronyms(acronyms)

    # Gene symbols are canonical genetic entities, not clinical acronyms.
    accepted_gene_symbol = try_accept_raw_gene_symbol_concept(
        concept=concept,
        source_text=source_text,
        concept_type=concept_type,
    )
    if accepted_gene_symbol is not None:
        accepted_gene_symbol = attach_quality_flags(
            concept=accepted_gene_symbol,
            source_text=source_text,
        )
        return {
            "accepted": True,
            "concept": accepted_gene_symbol,
            "reason": accepted_gene_symbol["validation_reason"],
        }

    named_score_concept = try_accept_named_acronym_score_concept(
        concept=concept,
        source_text=source_text,
        concept_type=concept_type,
        acronyms=safe_acronyms,
    )

    if named_score_concept is not None:
        named_score_concept = attach_quality_flags(
            concept=named_score_concept,
            source_text=source_text,
        )
        return {
            "accepted": True,
            "concept": named_score_concept,
            "reason": named_score_concept["validation_reason"],
        }

    # Next, try to expand raw acronym-only LLM outputs before normal direct
    # source matching. This prevents "AS" from becoming a written Concept
    # called "as".
    expanded_acronym_concept = try_expand_raw_acronym_concept(
        concept=concept,
        source_text=source_text,
        concept_type=concept_type,
        acronyms=safe_acronyms,
    )

    if expanded_acronym_concept is not None:
        expanded_name = expanded_acronym_concept["name"]
        expanded_rejection = get_out_of_schema_rejection(
            name=expanded_name,
            concept_type=concept_type,
            blocklist_names=blocklist_names,
        )

        if expanded_rejection is not None:
            return reject_with_semantic_metadata(
                normalized_concept=expanded_acronym_concept,
                rejection=expanded_rejection,
            )

        expanded_acronym_concept = attach_quality_flags(
            concept=expanded_acronym_concept,
            source_text=source_text,
        )

        return {
            "accepted": True,
            "concept": expanded_acronym_concept,
            "reason": expanded_acronym_concept["validation_reason"],
        }

    # Surface malformed cache definitions explicitly rather than writing a
    # corrupted long-form Concept.
    if (
        raw_name_is_unexpanded_acronym_short_form(concept)
        and raw_short_has_unsafe_cached_definition(
            raw_short=get_raw_name_for_acronym_check(concept),
            acronyms=acronyms,
        )
    ):
        normalized_concept["quality_flags"] = [
            "unsafe_acronym_definition"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "unsafe_acronym_definition",
        }

    # If the LLM gave only an acronym-like short form and no safe cached
    # expansion exists, do not write the short form as a Concept node.
    if raw_name_is_unexpanded_acronym_short_form(concept):
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "acronym_short_without_valid_expansion",
        }

    # Important guard:
    # After acronym expansion fails, tiny lowercase names such as "as" should
    # not be accepted just because the ordinary word appears in the section.
    if is_unsafe_short_surface_name_for_direct_match(name):
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "ambiguous_short_surface_name",
        }

    out_of_schema_rejection = get_out_of_schema_rejection(
        name=name,
        concept_type=concept_type,
        blocklist_names=blocklist_names,
    )
    if out_of_schema_rejection is not None:
        return reject_with_semantic_metadata(
            normalized_concept=normalized_concept,
            rejection=out_of_schema_rejection,
        )

    if is_generic_causal_description(name):
        normalized_concept["quality_flags"] = ["generic_causal_phrase"]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "generic_causal_description",
        }

    if is_non_population_care_setting(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "non_population_care_setting"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "care_setting_not_population",
        }

    if is_generic_population_name(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "generic_population_reference"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "generic_population_reference",
        }

    if is_non_entity_discipline(name):
        normalized_concept["quality_flags"] = [
            "medical_discipline_not_entity"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "medical_discipline_not_entity",
        }

    if is_non_anatomical_adjective(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "isolated_anatomical_adjective"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "anatomical_adjective_not_structure",
        }

    if is_generic_process_entity(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "generic_process_term"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "generic_process_term",
        }

    if is_treatment_misclassified_as_exposure(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "treatment_exposure_type_mismatch"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "treatment_not_exposure",
        }

    if is_procedure_or_modality_misclassified_as_drug(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "procedure_drug_type_mismatch"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "treatment_modality_not_drug",
        }

    if is_generic_therapeutic_entity(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "generic_therapeutic_term"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "generic_therapeutic_term",
        }

    if is_generic_biomarker_category(
        name=name,
        concept_type=concept_type,
    ):
        normalized_concept["quality_flags"] = [
            "generic_biomarker_category"
        ]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "generic_biomarker_category",
        }

    support_evidence = get_support_evidence(
        name=name,
        source_text=source_text,
        acronyms=safe_acronyms,
        concept_type=concept_type,
    )

    if support_evidence is None:
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "not_supported_by_source_or_acronym",
        }

    accepted_concept = build_accepted_concept(
        normalized_concept=normalized_concept,
        support_evidence=support_evidence,
    )
    accepted_concept = attach_quality_flags(
        concept=accepted_concept,
        source_text=source_text,
    )

    return {
        "accepted": True,
        "concept": accepted_concept,
        "reason": accepted_concept["validation_reason"],
    }


def validate_concepts_against_source(
    concepts: List[Dict[str, Any]],
    source_text: str,
    allowed_types: Set[str],
    blocklist_names: Set[str],
    acronyms: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Validate a list of extracted concepts against the section source text.

    Returns:
    {
        "accepted": [...],
        "rejected": [
            {"name": ..., "type": ..., "reason": ...},
            ...
        ],
        "stats": {
            "input_concepts": ...,
            "accepted_concepts": ...,
            "rejected_concepts": ...
        }
    }

    Acronym support is optional.

    Two acronym cases are handled:
    1. Long-form concept extracted, acronym appears in section:
        concept = "acute coronary syndrome"
        section text contains "ACS"
        cache has ACS -> Acute coronary syndrome(s)
        result: accepted by acronym support.

    2. Raw acronym-only concept extracted:
        raw_name = "AS"
        cache has AS -> Aortic stenosis
        result: accepted as "aortic stenosis", not as "as".
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for concept in concepts or []:
        result = validate_single_concept(
            concept=concept,
            source_text=source_text,
            allowed_types=allowed_types,
            blocklist_names=blocklist_names,
            acronyms=acronyms,
        )

        normalized_concept = result["concept"]

        if result["accepted"]:
            accepted.append(normalized_concept)
        else:
            rejected.append(
                build_rejected_concept_record(
                    normalized_concept=normalized_concept,
                    reason=result["reason"],
                )
            )

    accepted = deduplicate_validated_concepts(accepted)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "stats": {
            "input_concepts": len(concepts or []),
            "accepted_concepts": len(accepted),
            "rejected_concepts": len(rejected),
        },
    }


def summarize_rejections(rejected: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count rejection reasons for logging/reporting.
    """
    counter = Counter()

    for item in rejected or []:
        reason = item.get("reason", "unknown")
        counter[reason] += 1

    return dict(counter)