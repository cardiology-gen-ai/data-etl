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
- tiny lowercase surface names such as "as" are rejected by direct source
  matching, because they can be ordinary words rather than clinical concepts
- support checking is intentionally strict, with only small matching tolerance
  for whitespace, hyphenation, common singular/plural forms, spelling variants,
  and acronym expansion
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge_graph.acronym_utils import (
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
    "background",
    "cardiac",
    "cardiovascular",
    "chronic",
    "concomitant",
    "medical",
    "other",
    "pharmacological",
    "pharmacologic",
    "prescribed",
    "standard",
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


def get_direct_source_support(name: str, source_text: str) -> Optional[Dict[str, Any]]:
    """
    Return direct deterministic source evidence for a concept name.

    Returns metadata if a direct surface-form match is found, otherwise None.

    Note:
    matched_text is from the normalized source text, not the raw PDF text.
    """
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
    Return True when the original LLM surface looks like an acronym short form.

    This is used only after acronym expansion has failed. In that case, we avoid
    writing ambiguous short-form Concept nodes such as "AS", "LV", or "CMR".
    """
    raw_name_for_acronym = get_raw_name_for_acronym_check(concept)

    return is_likely_acronym_short_form(raw_name_for_acronym)


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
            deduped[key] = dict(concept)
            continue

        # Merge non-empty evidence fields if a duplicate has extra metadata.
        existing = deduped[key]

        for field, value in concept.items():
            if existing.get(field) in (None, "") and value not in (None, ""):
                existing[field] = value

    return list(deduped.values())



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

    # First, try to expand raw acronym-only LLM outputs before any direct
    # source matching. This prevents "AS" from becoming a written Concept
    # called "as".
    expanded_acronym_concept = try_expand_raw_acronym_concept(
        concept=concept,
        source_text=source_text,
        concept_type=concept_type,
        acronyms=acronyms,
    )

    if expanded_acronym_concept is not None:
        expanded_name = expanded_acronym_concept["name"]

        if expanded_name in blocklist_names:
            return {
                "accepted": False,
                "concept": expanded_acronym_concept,
                "reason": "expanded_acronym_blocklisted_name",
            }

        expanded_acronym_concept = attach_quality_flags(
            concept=expanded_acronym_concept,
            source_text=source_text,
        )

        return {
            "accepted": True,
            "concept": expanded_acronym_concept,
            "reason": expanded_acronym_concept["validation_reason"],
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

    if name in blocklist_names:
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "blocklisted_name",
        }

    if is_generic_causal_description(name):
        normalized_concept["quality_flags"] = ["generic_causal_phrase"]
        return {
            "accepted": False,
            "concept": normalized_concept,
            "reason": "generic_causal_description",
        }

    support_evidence = get_support_evidence(
        name=name,
        source_text=source_text,
        acronyms=acronyms,
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