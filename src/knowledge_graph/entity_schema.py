"""
entity_schema.py

Shared local entity schema for the knowledge graph entity-extraction pipeline.

Purpose:
- define the local entity types accepted by the KG
- define aliases from common LLM type outputs to local canonical types
- define generic or blocklisted concept names
- provide shared normalization and deduplication helpers

Design principles:
- this is a lightweight local entity schema, not a full clinical ontology
- canonical entity types describe what a concept is, not the role it plays in
  a specific sentence
- contextual roles such as risk factor, comorbidity, complication, outcome,
  or target_population are intentionally deferred to a separate enrichment
  stage; population_or_patient_group is an intrinsic entity type, whereas
  target_population is a future role that such an entity may play
- recommendation-specific qualifiers such as negation, severity, thresholds,
  duration, and frequency are also outside this module

A future ontology or contextual-assertion layer can build on top of these local
entity types without changing the identity of the Concept nodes.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


ENTITY_SCHEMA_VERSION = "2.1"


# Canonical local entity types and their shared definitions.
#
# Keep these definitions intrinsic to the concept. For example:
# - hypertension remains a disease even when used as a risk factor
# - heart failure remains a disease even when used as a comorbidity,
#   complication, or outcome
# - stroke remains a disease even when it is an outcome to prevent
TYPE_DEFINITIONS: Dict[str, str] = {
    "disease": (
        "A named disease, syndrome, disorder, or pathological condition."
    ),
    "clinical_finding": (
        "A symptom, sign, phenotype, physiological or pathological state, "
        "ECG/imaging/laboratory finding, measured result, or other clinical "
        "observation that is not itself a named disease."
    ),
    "exposure_or_lifestyle_factor": (
        "A behavioural, lifestyle, environmental, or occupational exposure "
        "such as smoking, alcohol use, physical inactivity, diet, or air "
        "pollution. Diseases and clinical findings do not change to this type "
        "merely because they act as risk factors in context."
    ),
    "genetic_factor": (
        "A gene, mutation, genetic variant, genotype, inherited molecular "
        "factor, or explicitly described genetic status."
    ),
    "biomarker": (
        "A measurable biological or laboratory analyte or marker, rather than "
        "the test used to measure it or the resulting measured value."
    ),
    "diagnostic_test": (
        "A diagnostic, monitoring, imaging, laboratory, functional, genetic, "
        "or clinical examination or investigation performed on a patient."
    ),
    "score_or_risk_model": (
        "A named clinical score, calculator, prediction rule, staging system, "
        "or risk-estimation model."
    ),
    "drug_or_drug_class": (
        "A specific medication, medicinal product, or named pharmacological "
        "class."
    ),
    "procedure_or_intervention": (
        "A therapeutic, invasive, surgical, catheter-based, "
        "electrophysiological, or other clinical procedure or intervention."
    ),
    "device": (
        "A medical device, prosthesis, lead, implant, valve system, monitor, "
        "or mechanical support system."
    ),
    "care_strategy": (
        "A specific and reusable care process or management strategy, such as "
        "family screening, genetic counselling, structured follow-up, cardiac "
        "rehabilitation, surveillance, patient education, or shared "
        "decision-making. Generic words such as management or treatment are "
        "not sufficient."
    ),
    "anatomical_structure": (
        "An anatomical body structure, organ, chamber, vessel, valve, tissue, "
        "or other anatomical site."
    ),
    "clinical_outcome": (
        "An outcome-like endpoint such as mortality, hospitalization, "
        "rehospitalization, quality of life, functional improvement, or symptom "
        "improvement. A disease does not change to this type merely because it "
        "is described as an outcome."
    ),
    "microorganism_or_pathogen": (
        "A named bacterium, virus, fungus, parasite, other microorganism, or "
        "clinically meaningful microorganism group."
    ),
    "population_or_patient_group": (
        "An explicitly named demographic, life-stage, familial, reproductive, "
        "occupational, or clinically defined group of people relevant to care, "
        "such as children, older adults, pregnant women, athletes, first-degree "
        "relatives, neonates, fetuses, or foetuses. Generic words such as patient, "
        "patients, population, or selected patients are not sufficient."
    ),
}


ALLOWED_TYPES = set(TYPE_DEFINITIONS)


# Common type labels that an LLM may produce and that can be mapped safely to
# the local canonical types.
#
# Intentionally excluded:
# - risk_factor
# - complication
# - comorbidity
#
# These describe contextual roles and must not be converted automatically into
# canonical entity types.
_RAW_TYPE_ALIASES = {
    # Disease
    "disease entity": "disease",
    "disorder": "disease",
    "syndrome": "disease",
    "pathological condition": "disease",
    "medical condition": "disease",

    # Clinical findings and observed results
    "phenotype": "clinical_finding",
    "finding": "clinical_finding",
    "clinical finding": "clinical_finding",
    "clinical observation": "clinical_finding",
    "observation": "clinical_finding",
    "sign": "clinical_finding",
    "symptom": "clinical_finding",
    "sign_or_symptom": "clinical_finding",
    "physical finding": "clinical_finding",
    "ecg finding": "clinical_finding",
    "imaging finding": "clinical_finding",
    "laboratory finding": "clinical_finding",
    "laboratory result": "clinical_finding",
    "lab result": "clinical_finding",
    "test result": "clinical_finding",
    "lab value": "clinical_finding",
    "laboratory value": "clinical_finding",

    # Exposures and lifestyle factors
    "exposure": "exposure_or_lifestyle_factor",
    "lifestyle factor": "exposure_or_lifestyle_factor",
    "lifestyle exposure": "exposure_or_lifestyle_factor",
    "behavioural factor": "exposure_or_lifestyle_factor",
    "behavioral factor": "exposure_or_lifestyle_factor",
    "behavioural exposure": "exposure_or_lifestyle_factor",
    "behavioral exposure": "exposure_or_lifestyle_factor",
    "environmental exposure": "exposure_or_lifestyle_factor",
    "occupational exposure": "exposure_or_lifestyle_factor",
    "health behaviour": "exposure_or_lifestyle_factor",
    "health behavior": "exposure_or_lifestyle_factor",

    # Care strategies
    "care strategy": "care_strategy",
    "management strategy": "care_strategy",
    "care process": "care_strategy",
    "care plan": "care_strategy",
    "treatment strategy": "care_strategy",
    "follow_up": "care_strategy",
    "follow-up": "care_strategy",
    "follow up": "care_strategy",
    "follow-up strategy": "care_strategy",
    "screening strategy": "care_strategy",
    "surveillance strategy": "care_strategy",
    "counselling strategy": "care_strategy",
    "counseling strategy": "care_strategy",
    "prevention strategy": "care_strategy",
    "rehabilitation strategy": "care_strategy",

    # Drugs
    "drug": "drug_or_drug_class",
    "drug class": "drug_or_drug_class",
    "drug_class": "drug_or_drug_class",
    "medication": "drug_or_drug_class",
    "medication class": "drug_or_drug_class",
    "medicinal product": "drug_or_drug_class",
    "pharmacological class": "drug_or_drug_class",

    # Procedures and interventions
    "procedure": "procedure_or_intervention",
    "intervention": "procedure_or_intervention",
    "therapeutic intervention": "procedure_or_intervention",
    "surgery": "procedure_or_intervention",
    "surgical procedure": "procedure_or_intervention",
    "catheter intervention": "procedure_or_intervention",
    "invasive procedure": "procedure_or_intervention",
    "therapeutic procedure": "procedure_or_intervention",

    # Diagnostic and monitoring tests, including imaging
    "test": "diagnostic_test",
    "diagnostic test": "diagnostic_test",
    "diagnostic examination": "diagnostic_test",
    "examination": "diagnostic_test",
    "investigation": "diagnostic_test",
    "clinical assessment": "diagnostic_test",
    "lab test": "diagnostic_test",
    "laboratory test": "diagnostic_test",
    "functional test": "diagnostic_test",
    "monitoring test": "diagnostic_test",
    "genetic test": "diagnostic_test",
    "imaging": "diagnostic_test",
    "imaging test": "diagnostic_test",
    "imaging modality": "diagnostic_test",
    "imaging_modality": "diagnostic_test",

    # Biomarkers
    "biological_marker": "biomarker",
    "biological marker": "biomarker",
    "laboratory_marker": "biomarker",
    "laboratory marker": "biomarker",
    "lab_marker": "biomarker",
    "lab marker": "biomarker",
    "analyte": "biomarker",

    # Scores and models
    "score": "score_or_risk_model",
    "risk score": "score_or_risk_model",
    "risk model": "score_or_risk_model",
    "prediction rule": "score_or_risk_model",
    "clinical prediction rule": "score_or_risk_model",
    "clinical score": "score_or_risk_model",
    "calculator": "score_or_risk_model",
    "staging system": "score_or_risk_model",
    "risk calculator": "score_or_risk_model",

    # Genetic factors
    "gene": "genetic_factor",
    "genetic": "genetic_factor",
    "genetic factor": "genetic_factor",
    "genetic marker": "genetic_factor",
    "genetic variant": "genetic_factor",
    "gene variant": "genetic_factor",
    "variant": "genetic_factor",
    "mutation": "genetic_factor",
    "genotype": "genetic_factor",
    "genetic status": "genetic_factor",
    "inherited factor": "genetic_factor",

    # Anatomy
    "anatomy": "anatomical_structure",
    "structure": "anatomical_structure",
    "anatomical structure": "anatomical_structure",
    "anatomical site": "anatomical_structure",
    "body structure": "anatomical_structure",

    # Clinical outcomes
    "outcome": "clinical_outcome",
    "clinical outcome": "clinical_outcome",
    "clinical_outcome": "clinical_outcome",
    "endpoint": "clinical_outcome",
    "clinical endpoint": "clinical_outcome",
    "clinical_endpoint": "clinical_outcome",
    "prognostic outcome": "clinical_outcome",

    # Populations and patient groups
    "population group": "population_or_patient_group",
    "patient group": "population_or_patient_group",
    "patient population": "population_or_patient_group",
    "demographic group": "population_or_patient_group",
    "age group": "population_or_patient_group",
    "life-stage group": "population_or_patient_group",
    "life stage group": "population_or_patient_group",
    "familial group": "population_or_patient_group",
    "family group": "population_or_patient_group",
    "target population": "population_or_patient_group",

    # Microorganisms and pathogens
    "microorganism": "microorganism_or_pathogen",
    "micro-organism": "microorganism_or_pathogen",
    "pathogen": "microorganism_or_pathogen",
    "infectious agent": "microorganism_or_pathogen",
    "microbial organism": "microorganism_or_pathogen",
    "bacterium": "microorganism_or_pathogen",
    "bacteria": "microorganism_or_pathogen",
    "virus": "microorganism_or_pathogen",
    "fungus": "microorganism_or_pathogen",
    "fungi": "microorganism_or_pathogen",
    "parasite": "microorganism_or_pathogen",
}


# Generic medical words that are usually too broad to be useful as Concept
# nodes. Exact matching is intentional: specific phrases such as
# "genetic counselling", "cardiac rehabilitation", or "Staphylococcus aureus"
# remain eligible.
_GENERIC_MEDICAL_BLOCKLIST_NAMES = {
    "diagnosis",
    "diagnoses",
    "treatment",
    "treatments",
    "management",
    "therapy",
    "therapies",
    "follow-up",
    "follow up",
    "monitoring",
    "screening",
    "surveillance",
    "counselling",
    "counseling",
    "prevention",
    "rehabilitation",
    "assessment",
    "assessments",
    "evaluation",
    "evaluations",
    "investigation",
    "investigations",
    "examination",
    "examinations",
    "recommendation",
    "recommendations",
    "patient",
    "patients",
    "disease",
    "diseases",
    "condition",
    "conditions",
    "risk",
    "risk factor",
    "risk factors",
    "test",
    "tests",
    "procedure",
    "procedures",
    "intervention",
    "interventions",
    "drug",
    "drugs",
    "medication",
    "medications",
    "care",
    "care strategy",
    "care strategies",
    "clinical finding",
    "clinical findings",
    "finding",
    "findings",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "biomarker",
    "biomarkers",
    "diagnostic test",
    "diagnostic tests",
    "imaging",
    "imaging test",
    "imaging tests",
    "imaging modality",
    "imaging modalities",
    "score",
    "scores",
    "model",
    "models",
    "outcome",
    "outcomes",
    "endpoint",
    "endpoints",
    "event",
    "events",
    "clinical outcome",
    "clinical outcomes",
    "clinical endpoint",
    "clinical endpoints",
    "prognosis",
    "prognoses",
    "exposure",
    "exposures",
    "lifestyle factor",
    "lifestyle factors",
    "pathogen",
    "pathogens",
    "microorganism",
    "microorganisms",
    "bacterium",
    "bacteria",
    "virus",
    "viruses",
    "fungus",
    "fungi",
    "parasite",
    "parasites",
    "population",
    "populations",
    "patient group",
    "patient groups",
    "patient population",
    "patient populations",
    "target population",
    "target populations",
    "age group",
    "age groups",
    "selected patient",
    "selected patients",
}


# Generic document/PDF/section words that can be included in captions, tables,
# or chunk metadata.
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
    Normalize type strings into the canonical lookup format used for:
    - incoming LLM types
    - TYPE_ALIASES keys
    - TYPE_ALIASES values

    Examples:
        "Clinical Finding" -> "clinical_finding"
        " imaging-modality " -> "imaging_modality"
    """
    concept_type = str(raw_type).strip().lower()
    concept_type = re.sub(r"^[\s,;:.()\[\]{}'\"`]+", "", concept_type)
    concept_type = re.sub(r"[\s,;:.()\[\]{}'\"`]+$", "", concept_type)
    concept_type = normalize_whitespace(concept_type)
    concept_type = concept_type.replace("-", "_")
    concept_type = concept_type.replace(" ", "_")
    concept_type = re.sub(r"_+", "_", concept_type)
    return concept_type


TYPE_ALIASES = {
    normalize_type_token(alias): normalize_type_token(target)
    for alias, target in _RAW_TYPE_ALIASES.items()
}


def _validate_schema_configuration() -> None:
    """
    Fail fast if the local schema is internally inconsistent.

    This protects the structured-output enum and downstream validation from
    silently accepting aliases that point to removed or misspelled types.
    """
    non_canonical_type_names = {
        entity_type
        for entity_type in ALLOWED_TYPES
        if normalize_type_token(entity_type) != entity_type
    }
    if non_canonical_type_names:
        raise ValueError(
            "ALLOWED_TYPES contains non-canonical type names: "
            f"{sorted(non_canonical_type_names)}"
        )

    invalid_alias_targets = set(TYPE_ALIASES.values()) - ALLOWED_TYPES
    if invalid_alias_targets:
        raise ValueError(
            "TYPE_ALIASES contains targets that are not allowed entity types: "
            f"{sorted(invalid_alias_targets)}"
        )

    empty_definitions = {
        entity_type
        for entity_type, definition in TYPE_DEFINITIONS.items()
        if not str(definition).strip()
    }
    if empty_definitions:
        raise ValueError(
            "TYPE_DEFINITIONS contains empty definitions for: "
            f"{sorted(empty_definitions)}"
        )


_validate_schema_configuration()


def normalize_type(raw_type: Any) -> str:
    """Normalize an incoming entity type and apply safe local aliases."""
    concept_type = normalize_type_token(raw_type)

    if concept_type in TYPE_ALIASES:
        concept_type = TYPE_ALIASES[concept_type]

    return concept_type


def normalize_name(raw_name: Any) -> str:
    """
    Normalize a concept name for use as the local Concept-node key.

    The normalized key is lowercased and whitespace-normalized. Only enclosing
    punctuation is removed; clinically relevant internal characters such as
    hyphens, slashes, apostrophes, plus signs, and parentheses are preserved.

    The original LLM surface form should continue to be stored separately by
    the calling extraction pipeline as raw_name/matched_text.
    """
    name = str(raw_name).strip().lower()
    name = normalize_whitespace(name)

    name = re.sub(r"^[\s,;:.()\[\]{}'\"`]+", "", name)
    name = re.sub(r"[\s,;:.()\[\]{}'\"`]+$", "", name)
    name = normalize_whitespace(name)

    return name


def normalize_concept(raw: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Normalize and validate one raw concept payload.

    Returns None when:
    - the payload is not a dictionary
    - required keys are missing
    - normalized fields are empty
    - the normalized name is blocklisted
    - the normalized type is not part of the local schema
    """
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
    Perform exact deduplication by normalized (name, type).

    The same normalized name may intentionally remain associated with different
    observed types. That ambiguity is preserved here and can be resolved later
    by the dedicated concept-type disambiguation step.
    """
    seen: set[Tuple[str, str]] = set()
    deduped: List[Dict[str, str]] = []

    for concept in concepts:
        key = (concept["name"], concept["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(concept)

    return deduped
