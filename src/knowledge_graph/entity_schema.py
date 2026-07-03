"""
entity_schema.py

Shared local entity schema for the knowledge graph entity extraction pipeline.

Purpose:
- define the local entity types accepted by the KG
- define aliases from common LLM type outputs to local canonical types
- define generic/blocklisted concept names
- provide shared normalization and deduplication helpers

This is a lightweight local entity schema, not a full ontology.
A future ontology layer can build on top of these local entity types.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


ALLOWED_TYPES = {
    "disease",
    "clinical_finding",
    "risk_factor",
    "genetic_factor",
    "biomarker",
    "diagnostic_test",
    "imaging_modality",
    "score_or_risk_model",
    "drug_or_drug_class",
    "procedure_or_intervention",
    "device",
    "complication_or_comorbidity",
    "care_strategy",
    "anatomical_structure",
    "clinical_outcome",
}


_RAW_TYPE_ALIASES = {
    "phenotype": "clinical_finding",
    "finding": "clinical_finding",
    "clinical finding": "clinical_finding",
    "sign": "clinical_finding",
    "symptom": "clinical_finding",
    "sign_or_symptom": "clinical_finding",

    "management": "care_strategy",
    "therapy": "care_strategy",
    "treatment_strategy": "care_strategy",
    "care plan": "care_strategy",
    "follow_up": "care_strategy",
    "follow-up": "care_strategy",

    "drug": "drug_or_drug_class",
    "drug class": "drug_or_drug_class",
    "drug_class": "drug_or_drug_class",
    "medication": "drug_or_drug_class",
    "medication class": "drug_or_drug_class",
    "pharmacotherapy": "drug_or_drug_class",

    "procedure": "procedure_or_intervention",
    "intervention": "procedure_or_intervention",
    "surgery": "procedure_or_intervention",
    "surgical procedure": "procedure_or_intervention",

    "test": "diagnostic_test",
    "lab test": "diagnostic_test",
    "laboratory test": "diagnostic_test",

    "imaging": "imaging_modality",
    "imaging test": "imaging_modality",
    "imaging modality": "imaging_modality",

    "biological_marker": "biomarker",
    "laboratory_marker": "biomarker",
    "lab_marker": "biomarker",
    "marker": "biomarker",
    "lab value": "biomarker",

    "score": "score_or_risk_model",
    "risk score": "score_or_risk_model",
    "risk model": "score_or_risk_model",
    "prediction rule": "score_or_risk_model",
    "clinical prediction rule": "score_or_risk_model",
    "clinical score": "score_or_risk_model",
    "calculator": "score_or_risk_model",

    "complication": "complication_or_comorbidity",
    "comorbidity": "complication_or_comorbidity",

    "gene": "genetic_factor",
    "genetic": "genetic_factor",
    "genetic marker": "genetic_factor",
    "genetic variant": "genetic_factor",
    "gene variant": "genetic_factor",
    "variant": "genetic_factor",
    "mutation": "genetic_factor",

    "anatomy": "anatomical_structure",
    "structure": "anatomical_structure",
    "anatomical structure": "anatomical_structure",

    "outcome": "clinical_outcome",
    "clinical outcome": "clinical_outcome",
    "clinical_outcome": "clinical_outcome",
    "endpoint": "clinical_outcome",
    "clinical endpoint": "clinical_outcome",
    "clinical_endpoint": "clinical_outcome",
    "event": "clinical_outcome",
    "clinical event": "clinical_outcome",
    "clinical_event": "clinical_outcome",
}


# Generic medical words that are usually too broad to be useful as Concept nodes.
_GENERIC_MEDICAL_BLOCKLIST_NAMES = {
    "diagnosis",
    "treatment",
    "management",
    "therapy",
    "follow-up",
    "follow up",
    "recommendation",
    "recommendations",
    "patient",
    "patients",
    "disease",
    "risk",
    "test",
    "tests",
    "procedure",
    "procedures",
    "drug",
    "drugs",
    "care",
    "clinical finding",
    "clinical findings",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "biomarker",
    "biomarkers",
    "diagnostic test",
    "diagnostic tests",
    "imaging modality",
    "imaging modalities",
    "score",
    "scores",
    "model",
    "models",
}


# Generic document/PDF/section words that can be included in captions, tables, or chunk metadata.
_DOCUMENT_ARTIFACT_BLOCKLIST_NAMES = {
    "title",
    "titles",
    "body",
    "bodies",
    "section",
    "sections",
    "subsection",
    "subsections",
    "chapter",
    "chapters",
    "paragraph",
    "paragraphs",
    "table",
    "tables",
    "figure",
    "figures",
    "caption",
    "captions",
    "guideline",
    "guidelines",
    "document",
    "documents",
    "text",
    "page",
    "pages",
}


BLOCKLIST_NAMES = (
    _GENERIC_MEDICAL_BLOCKLIST_NAMES
    | _DOCUMENT_ARTIFACT_BLOCKLIST_NAMES
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_type_token(raw_type: Any) -> str:
    """
    Normalize type strings into the same canonical lookup format used for:
    - incoming LLM types
    - TYPE_ALIASES keys
    """
    concept_type = str(raw_type).strip().lower()
    concept_type = re.sub(r"^[\s,;:.()\[\]{}'\"`]+", "", concept_type)
    concept_type = re.sub(r"[\s,;:.()\[\]{}'\"`]+$", "", concept_type)
    concept_type = normalize_whitespace(concept_type)
    concept_type = concept_type.replace("-", "_")
    concept_type = concept_type.replace(" ", "_")
    return concept_type


TYPE_ALIASES = {
    normalize_type_token(alias): normalize_type_token(target)
    for alias, target in _RAW_TYPE_ALIASES.items()
}


def normalize_type(raw_type: Any) -> str:
    concept_type = normalize_type_token(raw_type)

    if concept_type in TYPE_ALIASES:
        concept_type = TYPE_ALIASES[concept_type]

    return concept_type


def normalize_name(raw_name: Any) -> str:
    name = str(raw_name).strip().lower()
    name = normalize_whitespace(name)

    # Remove only enclosing punctuation while preserving relevant
    # internal characters such as hyphens, slashes, and parentheses.
    name = re.sub(r"^[\s,;:.()\[\]{}'\"`]+", "", name)
    name = re.sub(r"[\s,;:.()\[\]{}'\"`]+$", "", name)
    name = normalize_whitespace(name)

    return name


def normalize_concept(raw: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(raw, dict):
        logger.debug("Discarding non-dict concept payload: %r", raw)
        return None

    if "name" not in raw or "type" not in raw:
        logger.debug("Discarding concept without required keys: %r", raw)
        return None

    name = normalize_name(raw["name"])
    concept_type = normalize_type(raw["type"])

    if not name or not concept_type:
        logger.debug("Discarding concept with empty normalized fields: %r", raw)
        return None

    if name in BLOCKLIST_NAMES:
        logger.debug("Discarding blocklisted concept name: %s", name)
        return None

    if concept_type not in ALLOWED_TYPES:
        logger.debug(
            "Discarding concept with non-allowed type | name=%s | type=%s",
            name,
            concept_type,
        )
        return None

    return {
        "name": name,
        "type": concept_type,
    }


def deduplicate_concepts(concepts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Exact deduplication by (name, type).

    This intentionally preserves the case where the same normalized name is
    returned with different types, so type ambiguity can be preserved and
    later resolved in the disambiguation step.
    """
    seen: set[Tuple[str, str]] = set()
    deduped: List[Dict[str, str]] = []

    for concept in concepts:
        key = (concept["name"], concept["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(concept)

    return deduped
