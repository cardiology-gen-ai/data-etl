"""
umls_normalization.py

Optional UMLS/scispaCy normalization for existing Concept nodes.

This module intentionally does not replace LLM entity extraction or the local
entity schema. It enriches already-validated Concept nodes with UMLS metadata
and records auditable duplicate evidence through SAME_AS/POSSIBLY_SAME_AS
relationships.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from neo4j import Driver
except ImportError:
    Driver = Any  # type: ignore[misc, assignment]

from knowledge_graph.acronym_utils import (
    clean_acronym_definition,
    clean_acronym_short,
    canonicalize_acronym_definition_for_concept_name,
    load_acronyms_by_doc_id,
    long_form_matches_concept,
)
from knowledge_graph.entity_review_exports import (
    DEFAULT_ENTITY_REVIEW_DIR,
    safe_filename_component,
)
from knowledge_graph.entity_schema import normalize_name, normalize_type
from knowledge_graph.relationship_metadata import (
    build_normalization_relationship_metadata,
)

try:
    import requests
except ImportError as e:
    requests = None
    OPTIONAL_REQUESTS_IMPORT_ERROR = e
else:
    OPTIONAL_REQUESTS_IMPORT_ERROR = None

try:
    from rapidfuzz import fuzz
except ImportError as e:
    fuzz = None
    OPTIONAL_FUZZ_IMPORT_ERROR = e
else:
    OPTIONAL_FUZZ_IMPORT_ERROR = None

try:
    import spacy
    import scispacy  # noqa: F401  # Registers scispaCy pipeline components.
    from scispacy.linking import EntityLinker  # noqa: F401
except ImportError as e:
    spacy = None
    OPTIONAL_SCISPACY_IMPORT_ERROR = e
else:
    OPTIONAL_SCISPACY_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "en_core_sci_sm"
DEFAULT_LINKER_NAME = "umls"
DEFAULT_UMLS_THRESHOLD = 0.85
# Exact UMLS searches are stronger candidate evidence than normalizedString
# or words searches, but exact lookup alone does not guarantee the correct
# clinical sense. Keep a separate conservative acceptance threshold.
DEFAULT_EXACT_UMLS_THRESHOLD = 0.75
DEFAULT_FUZZY_THRESHOLD = 90
DEFAULT_MAX_CANDIDATES = 3
DEFAULT_ALIAS_LIMIT = 12
DEFAULT_UMLS_ALIAS_LIMIT = 20
DEFAULT_LOCAL_FILES_ONLY = False
DEFAULT_MIN_AVAILABLE_MEMORY_GB = 8.0
DEFAULT_API_TIMEOUT = 30.0
DEFAULT_API_RATE_LIMIT_PER_SECOND = 5.0
DEFAULT_UMLS_VERSION = "current"
DEFAULT_API_PAGE_SIZE = 25
DEFAULT_ATOM_PAGE_SIZE = 100
DEFAULT_PLAUSIBLE_UMLS_THRESHOLD = 0.50
MIN_EXACT_ATOM_SOURCE_COUNT = 2
SYNONYM_AUTO_ACCEPT_CANONICAL_TYPES = {"disease"}

# Disease-only narrowing guard. These modifiers are intentionally few and
# semantically restrictive: when UMLS introduces one that is absent from the
# local alias, the candidate may represent a subtype rather than an equivalent
# concept. Do not include broad words such as "primary", "type", "left", or
# "right" here: those would create avoidable false negatives.
DISEASE_NARROWING_MODIFIER_PHRASES = (
    ("hereditary",),
    ("familial",),
    ("senile",),
    ("wild", "type"),
    ("acquired",),
)

CANDIDATE_TRACE_VERSION = "umls_candidate_trace_v2"
RANKING_POLICY_VERSION = "umls_candidate_quality_v8_acronym_provenance"
NO_PLAUSIBLE_MATCH_STATUS = "umls_no_plausible_match"
API_NO_PLAUSIBLE_MATCH_METHOD = "umls_api_no_plausible_match"
NO_PLAUSIBLE_MATCH_METHOD = "umls_no_plausible_match"
NO_PLAUSIBLE_MATCH_REASON = "no_plausible_candidate"

SCISPACY_BACKEND = "scispacy"
UMLS_API_BACKEND = "umls_api"
FUZZY_ONLY_BACKEND = "fuzzy_only"
SUPPORTED_BACKENDS = {SCISPACY_BACKEND, UMLS_API_BACKEND, FUZZY_ONLY_BACKEND}

AUTO_NORMALIZATION_METHOD = "scispacy_umls"
LOW_CONFIDENCE_METHOD = "scispacy_umls_candidate"
NO_MATCH_METHOD = "scispacy_umls_no_match"
API_NORMALIZATION_METHOD = "umls_api"
API_LOW_CONFIDENCE_METHOD = "umls_api_candidate"
API_NO_MATCH_METHOD = "umls_api_no_match"
SKIPPED_METHOD = "preserve_existing_normalization"
FUZZY_METHOD = "fuzzy_name"
SAME_CUI_METHOD = "umls_cui"

MANUAL_NORMALIZATION_METHODS = {
    "manual",
    "curated",
    "manual_curated",
    "manual_umls",
}

UMLS_VALUE_FIELDS = {
    "umls_cui",
    "umls_canonical_name",
    "umls_definition",
    "umls_aliases",
    "umls_score",
}


UMLS_API_SEARCH_TYPES = ("exact", "normalizedString", "words")

# Candidate acceptance continues to use the conservative lexical score stored in
# ``UMLSMatch.score``. Cross-strategy selection uses a separate evidence score:
# ``words`` retrieval receives a small penalty, but exact API rank alone must
# not create artificial confidence. In particular, a weak exact rank-1 result
# must not displace a much stronger compatible exact candidate. Strong atom
# synonym evidence is handled separately during cross-strategy ranking.
EXACT_RANK_ONE_SELECTION_FLOOR = 0.0
EXACT_NON_PRIMARY_SELECTION_BONUS = 0.0
NORMALIZED_STRING_SELECTION_PENALTY = 0.01
WORDS_SELECTION_PENALTY = 0.03
STRONG_COMPATIBLE_EXACT_SELECTION_THRESHOLD = DEFAULT_EXACT_UMLS_THRESHOLD

# Conservative compatibility map between the local controlled entity types and
# UMLS semantic types. Intrinsic local types still fail closed when the API
# returns an explicitly incompatible semantic type. Contextual/document-role
# types are handled separately because their UMLS semantic type may describe the
# underlying clinical entity rather than its role in the guideline.
CANONICAL_TYPE_TO_UMLS_SEMANTIC_TYPES = {
    "disease": {
        "Acquired Abnormality",
        "Congenital Abnormality",
        "Disease or Syndrome",
        "Mental or Behavioral Dysfunction",
        "Neoplastic Process",
        "Pathologic Function",
    },
    "complication_or_comorbidity": {
        "Acquired Abnormality",
        "Congenital Abnormality",
        "Disease or Syndrome",
        "Finding",
        "Mental or Behavioral Dysfunction",
        "Pathologic Function",
        "Sign or Symptom",
    },
    "clinical_finding": {
        "Clinical Attribute",
        "Disease or Syndrome",
        "Finding",
        "Laboratory or Test Result",
        "Mental or Behavioral Dysfunction",
        "Organism Function",
        "Pathologic Function",
        "Sign or Symptom",
    },
    "device": {
        "Drug Delivery Device",
        "Medical Device",
        "Research Device",
    },
    "biomarker": {
        "Amino Acid, Peptide, or Protein",
        "Biologically Active Substance",
        "Clinical Attribute",
        "Enzyme",
        "Gene or Genome",
        "Hormone",
        "Laboratory or Test Result",
        "Nucleic Acid, Nucleoside, or Nucleotide",
    },
    "genetic_factor": {
        "Gene or Genome",
        "Genetic Function",
        "Molecular Sequence",
        "Nucleotide Sequence",
        "Organism Attribute",
    },
    "risk_factor": {
        "Clinical Attribute",
        "Environmental Effect of Humans",
        "Finding",
        "Individual Behavior",
        "Organism Attribute",
        "Population Group",
        "Sign or Symptom",
        "Social Behavior",
    },
    "diagnostic_test": {
        "Clinical Attribute",
        "Diagnostic Procedure",
        "Laboratory or Test Result",
        "Laboratory Procedure",
    },
    "imaging_modality": {
        "Diagnostic Procedure",
    },
    "score_or_risk_model": {
        "Clinical Attribute",
        "Finding",
        "Intellectual Product",
        "Quantitative Concept",
    },
    "drug_or_drug_class": {
        "Antibiotic",
        "Biomedical or Dental Material",
        "Clinical Drug",
        "Pharmacologic Substance",
    },
    "procedure_or_intervention": {
        "Diagnostic Procedure",
        "Health Care Activity",
        "Laboratory Procedure",
        "Therapeutic or Preventive Procedure",
    },
    "care_strategy": {
        "Health Care Activity",
        "Therapeutic or Preventive Procedure",
    },
    "anatomical_structure": {
        "Body Location or Region",
        "Body Part, Organ, or Organ Component",
        "Body Space or Junction",
        "Cell",
        "Cell Component",
        "Embryonic Structure",
        "Fully Formed Anatomical Structure",
        "Tissue",
    },
    "clinical_outcome": {
        "Clinical Attribute",
        "Disease or Syndrome",
        "Event",
        "Finding",
        "Laboratory or Test Result",
        "Pathologic Function",
        "Sign or Symptom",
    },
    "exposure_or_lifestyle_factor": {
        "Environmental Effect of Humans",
        "Hazardous or Poisonous Substance",
        "Human-caused Phenomenon or Process",
        "Individual Behavior",
        "Natural Phenomenon or Process",
        "Social Behavior",
    },
    "microorganism_or_pathogen": {
        "Animal",
        "Archaeon",
        "Bacterium",
        "Eukaryote",
        "Fungus",
        "Organism",
        "Virus",
    },
    "population_or_patient_group": {
        "Age Group",
        "Family Group",
        "Group",
        "Patient or Disabled Group",
        "Population Group",
        "Professional or Occupational Group",
    },
    # Backward-compatible aliases for older local type names.
    "procedure": {
        "Diagnostic Procedure",
        "Health Care Activity",
        "Laboratory Procedure",
        "Therapeutic or Preventive Procedure",
    },
    "treatment": {
        "Clinical Drug",
        "Health Care Activity",
        "Medical Device",
        "Pharmacologic Substance",
        "Therapeutic or Preventive Procedure",
    },
    "drug": {
        "Antibiotic",
        "Biomedical or Dental Material",
        "Clinical Drug",
        "Pharmacologic Substance",
    },
    "anatomy": {
        "Body Location or Region",
        "Body Part, Organ, or Organ Component",
        "Body Space or Junction",
        "Cell",
        "Cell Component",
        "Embryonic Structure",
        "Fully Formed Anatomical Structure",
        "Tissue",
    },
}


# These local types often encode the role played by an entity in a guideline,
# not its intrinsic ontology class. A non-overlapping UMLS semantic type is
# therefore reviewable evidence rather than an automatic veto.
CONTEXTUAL_CANONICAL_TYPES = frozenset({
    "risk_factor",
    "clinical_outcome",
    "care_strategy",
    "exposure_or_lifestyle_factor",
})

REVIEW_REQUIRED_STATUS = "review_required"
TYPE_REVIEW_REQUIRED_METHOD = "type_resolution_required"
TYPE_REVIEW_REQUIRED_REASON = "concept_type_requires_review"

SEMANTIC_COMPATIBLE = "compatible"
SEMANTIC_CONTEXTUAL_MISMATCH = "contextual_mismatch"
SEMANTIC_INCOMPATIBLE = "incompatible"
SEMANTIC_UNKNOWN_LOCAL_TYPE = "unknown_local_type"
SEMANTIC_TYPES_MISSING = "semantic_types_missing"


@dataclass
class ConceptRecord:
    concept_id: str
    name: str
    canonical_type: Optional[str] = None
    needs_type_review: bool = False
    type_resolution_status: Optional[str] = None
    observed_types: List[str] = field(default_factory=list)
    invalid_observed_types: List[str] = field(default_factory=list)
    type_support_pairs: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    relationship_acronyms: List[Dict[str, str]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases_considered: List[str] = field(default_factory=list)
    normalization_status: str = "pending"
    normalization_method: Optional[str] = None
    best_match: Optional["UMLSMatch"] = None
    duplicate_candidates: List[Dict[str, Any]] = field(default_factory=list)
    alias_provenance: List[Dict[str, Any]] = field(default_factory=list)
    candidate_trace: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class UMLSMatch:
    alias: str
    cui: str
    canonical_name: Optional[str]
    definition: Optional[str]
    aliases: List[str]
    score: float
    semantic_types: List[str] = field(default_factory=list)
    search_type: Optional[str] = None
    type_compatible: Optional[bool] = None
    semantic_compatibility: Optional[str] = None
    api_rank: Optional[int] = None
    selection_score: Optional[float] = None
    canonical_type: Optional[str] = None
    matched_atom_name: Optional[str] = None
    matched_atom_source: Optional[str] = None
    matched_atom_term_type: Optional[str] = None
    matched_atom_score: Optional[float] = None
    matched_atom_count: int = 0
    matched_atom_source_count: int = 0
    synonym_supported: bool = False


def concept_requires_type_review(concept: ConceptRecord) -> bool:
    """Return whether UMLS candidate generation must be skipped for a concept."""
    canonical_type = normalize_type(concept.canonical_type or "")
    return bool(concept.needs_type_review) or canonical_type == "ambiguous"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_optional_dependencies_available() -> None:
    if OPTIONAL_SCISPACY_IMPORT_ERROR is None:
        return

    raise RuntimeError(
        "UMLS normalization requires optional scispaCy dependencies. "
        "Install the optional extra, for example: pip install '.[normalization]'. "
        "Then install a matching scispaCy model, for example: "
        "pip install <en_core_sci_sm model URL from the scispaCy docs>. "
        "scispaCy models must match the installed scispaCy version. "
        f"Original import error: {OPTIONAL_SCISPACY_IMPORT_ERROR}"
    ) from OPTIONAL_SCISPACY_IMPORT_ERROR


def ensure_fuzzy_dependency_available() -> None:
    if OPTIONAL_FUZZ_IMPORT_ERROR is None:
        return

    raise RuntimeError(
        "Fuzzy duplicate detection requires rapidfuzz. Install the optional "
        "normalization extra, for example: pip install '.[normalization]'. "
        f"Original import error: {OPTIONAL_FUZZ_IMPORT_ERROR}"
    ) from OPTIONAL_FUZZ_IMPORT_ERROR


def ensure_requests_dependency_available() -> None:
    if OPTIONAL_REQUESTS_IMPORT_ERROR is None:
        return

    raise RuntimeError(
        "UMLS API normalization requires requests. Install dependencies, for "
        "example: pip install '.[normalization]'. "
        f"Original import error: {OPTIONAL_REQUESTS_IMPORT_ERROR}"
    ) from OPTIONAL_REQUESTS_IMPORT_ERROR


def get_available_memory_gb() -> Optional[float]:
    """
    Return current available system memory from /proc/meminfo when available.

    Loading the full scispaCy UMLS linker can terminate small WSL instances
    before Python can raise a normal exception, so we preflight conservatively.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None

    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / (1024 * 1024)

    return None


def ensure_available_memory(min_available_memory_gb: float) -> None:
    if min_available_memory_gb <= 0:
        return

    available_gb = get_available_memory_gb()
    if available_gb is None:
        logger.warning(
            "Could not determine available memory before loading UMLS linker"
        )
        return

    if available_gb < min_available_memory_gb:
        raise RuntimeError(
            "Not enough available memory to load the scispaCy UMLS linker safely: "
            f"{available_gb:.1f} GB available, "
            f"{min_available_memory_gb:.1f} GB required by the configured guard. "
            "The full UMLS linker can crash memory-constrained WSL sessions. "
            "Increase WSL memory/swap, run on a larger machine, or set "
            "KG_ENTITY_NORMALIZATION_MIN_AVAILABLE_MEMORY_GB=0 to bypass this "
            "guard at your own risk."
        )


def find_cached_scispacy_file_without_head(
    url_or_filename: str,
    cache_dir: Optional[str],
) -> Optional[str]:
    """
    Resolve scispaCy cached URLs without making the library's mandatory HEAD call.

    scispaCy stores cache metadata as <cached-file>.json containing the source
    URL and ETag. On clusters or sandboxed environments without internet, the
    default resolver fails before checking these already-present files.
    """
    path = Path(str(url_or_filename))
    if path.exists():
        return str(path)

    url = str(url_or_filename)
    if not url.startswith(("http://", "https://")):
        return None

    if cache_dir is None:
        try:
            import scispacy.file_cache as scispacy_file_cache
        except Exception:
            return None
        cache = Path(scispacy_file_cache.DATASET_CACHE)
    else:
        cache = Path(cache_dir)
    if not cache.exists():
        return None

    for meta_path in cache.glob("*.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(metadata, dict) or metadata.get("url") != url:
            continue

        cached_path = Path(str(meta_path)[:-5])
        if cached_path.exists():
            return str(cached_path)

    return None


def install_scispacy_local_cache_resolver(local_files_only: bool) -> None:
    """
    Prefer local scispaCy cache files before the library attempts S3 requests.
    """
    import scispacy.candidate_generation as candidate_generation
    import scispacy.file_cache as scispacy_file_cache

    if not hasattr(scispacy_file_cache, "_cardioai_original_cached_path"):
        scispacy_file_cache._cardioai_original_cached_path = (  # type: ignore[attr-defined]
            scispacy_file_cache.cached_path
        )

    original_cached_path = scispacy_file_cache._cardioai_original_cached_path  # type: ignore[attr-defined]

    def cached_path_local_first(url_or_filename, cache_dir=None):
        cached_path = find_cached_scispacy_file_without_head(
            str(url_or_filename),
            cache_dir,
        )
        if cached_path is not None:
            return cached_path

        if local_files_only:
            raise FileNotFoundError(
                "scispaCy local-files-only mode could not find cached resource "
                f"{url_or_filename!r}. Pre-download the UMLS linker files or "
                "set KG_ENTITY_NORMALIZATION_LOCAL_FILES_ONLY=false to allow "
                "network downloads."
            )

        return original_cached_path(url_or_filename, cache_dir=cache_dir)

    scispacy_file_cache.cached_path = cached_path_local_first
    candidate_generation.cached_path = cached_path_local_first


def load_scispacy_pipeline(
    model_name: str,
    linker_name: str,
    max_candidates: int,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
):
    ensure_optional_dependencies_available()

    assert spacy is not None

    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load scispaCy model '{model_name}'. Install a model "
            "compatible with your scispaCy version. For example, install "
            "en_core_sci_sm from the model URL listed in the scispaCy docs."
        ) from e

    ensure_available_memory(min_available_memory_gb)
    install_scispacy_local_cache_resolver(local_files_only=local_files_only)

    try:
        if "scispacy_linker" not in nlp.pipe_names:
            nlp.add_pipe(
                "scispacy_linker",
                config={
                    "resolve_abbreviations": False,
                    "linker_name": linker_name,
                    "threshold": 0.0,
                    "max_entities_per_mention": max_candidates,
                },
            )
        linker = nlp.get_pipe("scispacy_linker")
    except Exception as e:
        raise RuntimeError(
            f"Could not load scispaCy linker '{linker_name}'. The first UMLS "
            "linker run may download a large knowledge base. Verify that "
            "scispaCy, its nmslib dependency, and the requested linker are "
            "installed and available."
        ) from e

    return nlp, linker


def setup_normalization_schema(tx) -> None:
    tx.run(
        """
        CREATE INDEX concept_umls_cui IF NOT EXISTS
        FOR (c:Concept)
        ON (c.umls_cui)
        """
    )

    tx.run(
        """
        CREATE INDEX concept_normalization_status IF NOT EXISTS
        FOR (c:Concept)
        ON (c.normalization_status)
        """
    )

    tx.run(
        """
        CREATE INDEX concept_normalized_name IF NOT EXISTS
        FOR (c:Concept)
        ON (c.normalized_name)
        """
    )


def fetch_concepts_for_normalization(tx, doc_id: Optional[str]) -> List[ConceptRecord]:
    result = tx.run(
        """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WHERE $doc_id IS NULL OR s.doc_id = $doc_id
        WITH
            c,
            properties(c) AS concept_props,
            collect(DISTINCT s.doc_id) AS doc_ids,
            collect(DISTINCT {
                short: r.acronym_short,
                definition: r.acronym_definition
            }) AS acronym_rows
        RETURN
            elementId(c) AS concept_id,
            c.name AS name,
            c.canonical_type AS canonical_type,
            coalesce(c.needs_type_review, false) AS needs_type_review,
            c.type_resolution_status AS type_resolution_status,
            coalesce(c.observed_types, []) AS observed_types,
            coalesce(c.invalid_observed_types, []) AS invalid_observed_types,
            coalesce(c.type_support_pairs, []) AS type_support_pairs,
            concept_props['umls_cui'] AS umls_cui,
            concept_props['umls_canonical_name'] AS umls_canonical_name,
            concept_props['umls_definition'] AS umls_definition,
            concept_props['umls_aliases'] AS umls_aliases,
            concept_props['umls_score'] AS umls_score,
            concept_props['umls_semantic_types'] AS umls_semantic_types,
            concept_props['umls_linker_name'] AS umls_linker_name,
            concept_props['umls_model_name'] AS umls_model_name,
            concept_props['normalized_name'] AS normalized_name,
            concept_props['normalization_status'] AS normalization_status,
            concept_props['normalization_method'] AS normalization_method,
            doc_ids,
            acronym_rows
        ORDER BY c.name
        """,
        doc_id=doc_id,
    )

    records: List[ConceptRecord] = []

    for row in result:
        properties = {
            "umls_cui": row["umls_cui"],
            "umls_canonical_name": row["umls_canonical_name"],
            "umls_definition": row["umls_definition"],
            "umls_aliases": row["umls_aliases"],
            "umls_score": row["umls_score"],
            "umls_semantic_types": row["umls_semantic_types"],
            "umls_linker_name": row["umls_linker_name"],
            "umls_model_name": row["umls_model_name"],
            "normalized_name": row["normalized_name"],
            "normalization_status": row["normalization_status"],
            "normalization_method": row["normalization_method"],
        }

        acronym_rows = [
            {
                "short": str(item.get("short") or "").strip(),
                "definition": str(item.get("definition") or "").strip(),
            }
            for item in (row["acronym_rows"] or [])
            if isinstance(item, dict)
            and (item.get("short") or item.get("definition"))
        ]

        records.append(
            ConceptRecord(
                concept_id=row["concept_id"],
                name=str(row["name"] or ""),
                canonical_type=row["canonical_type"],
                needs_type_review=bool(row["needs_type_review"]),
                type_resolution_status=row["type_resolution_status"],
                observed_types=[
                    str(value)
                    for value in (row["observed_types"] or [])
                    if value
                ],
                invalid_observed_types=[
                    str(value)
                    for value in (row["invalid_observed_types"] or [])
                    if value
                ],
                type_support_pairs=[
                    str(value)
                    for value in (row["type_support_pairs"] or [])
                    if value
                ],
                doc_ids=[doc for doc in (row["doc_ids"] or []) if doc],
                relationship_acronyms=acronym_rows,
                properties=properties,
            )
        )

    return records


def should_preserve_existing_normalization(
    concept: ConceptRecord,
    force: bool,
) -> bool:
    if force:
        return False

    method = str(concept.properties.get("normalization_method") or "").strip().lower()
    status = str(concept.properties.get("normalization_status") or "").strip().lower()

    if method in MANUAL_NORMALIZATION_METHODS or status in MANUAL_NORMALIZATION_METHODS:
        return True

    if any(concept.properties.get(field) not in (None, "", []) for field in UMLS_VALUE_FIELDS):
        return True

    return False


def append_unique(values: List[str], raw_value: Any) -> None:
    value = str(raw_value or "").strip()
    if not value:
        return

    normalized = normalize_name(value)
    if not normalized:
        return

    if normalized not in {normalize_name(existing) for existing in values}:
        values.append(normalized)


def should_include_secondary_alias(
    concept_name: str,
    candidate_alias: str,
) -> bool:
    """Reject secondary acronym aliases that merely broaden the concept.

    A secondary expansion such as ``left ventricular`` for
    ``left ventricular hypertrabeculation`` removes the discriminating head
    term and should remain provenance only, not become an independent UMLS
    query. Primary acronym expansions are not subject to this filter.
    """
    normalized_concept = normalize_name(concept_name)
    normalized_alias = normalize_name(candidate_alias)
    if not normalized_concept or not normalized_alias:
        return False
    if normalized_alias == normalized_concept:
        return True

    concept_tokens = set(re.findall(r"[a-z0-9]+", normalized_concept))
    alias_tokens = set(re.findall(r"[a-z0-9]+", normalized_alias))
    if not alias_tokens:
        return False

    # A strict token subset is always less specific than the original concept.
    if alias_tokens < concept_tokens:
        return False
    return True


def build_aliases_for_concept(
    concept: ConceptRecord,
    acronyms_by_doc_id: Dict[str, Dict[str, str]],
) -> List[str]:
    """
    Build ordered aliases for UMLS lookup.

    When a concept is an acronym, its supported long form is placed before the
    short form. This prevents ambiguous strings such as "ICD" from winning over
    an available expansion such as "implantable cardioverter defibrillator".
    """
    preferred_expansions: List[str] = []
    secondary_aliases: List[str] = []
    normalized_concept_name = normalize_name(concept.name)

    for row in concept.relationship_acronyms:
        short = clean_acronym_short(row.get("short"))
        definition = clean_acronym_definition(row.get("definition"))
        if not definition:
            continue

        is_primary = bool(
            short and normalize_name(short) == normalized_concept_name
        )
        candidates = [
            definition,
            canonicalize_acronym_definition_for_concept_name(definition),
        ]
        for candidate in candidates:
            if is_primary:
                append_unique(preferred_expansions, candidate)
            elif should_include_secondary_alias(concept.name, candidate):
                append_unique(secondary_aliases, candidate)

    for doc_id in concept.doc_ids:
        for raw_short, raw_definition in sorted(
            acronyms_by_doc_id.get(doc_id, {}).items()
        ):
            short = clean_acronym_short(raw_short)
            definition = clean_acronym_definition(raw_definition)

            if not short or not definition:
                continue

            normalized_short = normalize_name(short)

            if normalized_short == normalized_concept_name:
                append_unique(preferred_expansions, definition)
                append_unique(
                    preferred_expansions,
                    canonicalize_acronym_definition_for_concept_name(definition),
                )
                continue

            if long_form_matches_concept(concept.name, definition):
                append_unique(secondary_aliases, definition)

    aliases: List[str] = []
    for alias in preferred_expansions:
        append_unique(aliases, alias)

    append_unique(aliases, concept.name)

    for alias in secondary_aliases:
        append_unique(aliases, alias)

    return aliases[:DEFAULT_ALIAS_LIMIT]


def build_alias_provenance_for_concept(
    concept: ConceptRecord,
    acronyms_by_doc_id: Dict[str, Dict[str, str]],
    aliases: Sequence[str],
) -> List[Dict[str, Any]]:
    """Describe where each ordered alias came from without changing alias order."""
    source_map: Dict[str, Dict[str, set]] = {}

    def register(
        raw_alias: Any,
        source: str,
        *,
        doc_id: Optional[str] = None,
        acronym_short: Optional[str] = None,
    ) -> None:
        normalized = normalize_name(str(raw_alias or ""))
        if not normalized:
            return
        bucket = source_map.setdefault(
            normalized,
            {"sources": set(), "doc_ids": set(), "acronym_shorts": set()},
        )
        bucket["sources"].add(source)
        if doc_id:
            bucket["doc_ids"].add(str(doc_id))
        if acronym_short:
            bucket["acronym_shorts"].add(str(acronym_short))

    register(concept.name, "concept_name")

    normalized_concept_name = normalize_name(concept.name)
    for row in concept.relationship_acronyms:
        short = clean_acronym_short(row.get("short"))
        definition = clean_acronym_definition(row.get("definition"))
        if not definition:
            continue
        is_primary = bool(
            short and normalize_name(short) == normalized_concept_name
        )
        source = (
            "mention_acronym_expansion"
            if is_primary
            else "mention_secondary_acronym_expansion"
        )
        provenance_candidates = [
            (definition, source),
            (
                canonicalize_acronym_definition_for_concept_name(definition),
                f"{source}_canonicalized",
            ),
        ]
        for candidate, candidate_source in provenance_candidates:
            if is_primary or should_include_secondary_alias(
                concept.name, candidate
            ):
                register(
                    candidate,
                    candidate_source,
                    acronym_short=short,
                )

    for doc_id in concept.doc_ids:
        for raw_short, raw_definition in sorted(
            acronyms_by_doc_id.get(doc_id, {}).items()
        ):
            short = clean_acronym_short(raw_short)
            definition = clean_acronym_definition(raw_definition)
            if not short or not definition:
                continue

            if normalize_name(short) == normalized_concept_name:
                source = "document_acronym_expansion"
                register(
                    definition,
                    source,
                    doc_id=doc_id,
                    acronym_short=short,
                )
                register(
                    canonicalize_acronym_definition_for_concept_name(definition),
                    f"{source}_canonicalized",
                    doc_id=doc_id,
                    acronym_short=short,
                )
            elif long_form_matches_concept(concept.name, definition):
                register(
                    definition,
                    "document_long_form_match",
                    doc_id=doc_id,
                    acronym_short=short,
                )

    provenance: List[Dict[str, Any]] = []
    for alias_index, alias in enumerate(aliases):
        bucket = source_map.get(
            normalize_name(alias),
            {"sources": {"derived_alias"}, "doc_ids": set(), "acronym_shorts": set()},
        )
        provenance.append(
            {
                "alias": alias,
                "alias_index": alias_index,
                "alias_sources": sorted(bucket["sources"]),
                "alias_doc_ids": sorted(bucket["doc_ids"]),
                "acronym_shorts": sorted(bucket["acronym_shorts"]),
            }
        )
    return provenance


def link_alias_to_umls(alias: str, nlp, linker) -> Optional[UMLSMatch]:
    alias = str(alias or "").strip()
    if not alias:
        return None

    doc = nlp.make_doc(alias)
    span = doc.char_span(0, len(alias), label="ENTITY", alignment_mode="expand")
    if span is None:
        return None

    doc.ents = [span]
    linker(doc)

    if not doc.ents:
        return None

    ent = doc.ents[0]
    kb_ents = getattr(ent._, "kb_ents", [])
    if not kb_ents:
        return None

    top_cui, score = kb_ents[0]

    try:
        umls_entity = linker.kb.cui_to_entity[top_cui]
    except KeyError:
        logger.debug("CUI %s not found in linker KB for alias %r", top_cui, alias)
        return None

    aliases = list(getattr(umls_entity, "aliases", []) or [])[:DEFAULT_UMLS_ALIAS_LIMIT]

    return UMLSMatch(
        alias=alias,
        cui=str(top_cui),
        canonical_name=getattr(umls_entity, "canonical_name", None),
        definition=getattr(umls_entity, "definition", None),
        aliases=[str(item) for item in aliases if item],
        score=round(float(score), 4),
    )


def select_best_umls_match(
    aliases: Sequence[str],
    nlp,
    linker,
) -> Optional[UMLSMatch]:
    matches = [
        match
        for match in (link_alias_to_umls(alias, nlp, linker) for alias in aliases)
        if match is not None
    ]

    if not matches:
        return None

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[0]


def normalize_backend_name(backend: str) -> str:
    normalized = str(backend or SCISPACY_BACKEND).strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            "entity_normalization backend must be one of "
            f"{sorted(SUPPORTED_BACKENDS)}; got {backend!r}"
        )
    return normalized


def normalize_api_alias(value: str) -> str:
    return normalize_name(value)


def _normalize_match_token(token: str) -> str:
    token = str(token or "").casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _matching_tokens(value: str) -> List[str]:
    normalized = normalize_api_alias(value)
    return [
        _normalize_match_token(token)
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token
    ]


def classify_semantic_compatibility(
    canonical_type: Optional[str],
    semantic_types: Sequence[str],
) -> str:
    """Classify local/UMLS semantic compatibility without losing provenance.

    Contextual local types (risk factor, clinical outcome, care strategy) may
    legitimately map to a UMLS concept whose semantic type describes the
    underlying entity rather than its role in the source document. Such cases
    remain candidates, receive a small ranking penalty, and are explicitly
    marked for audit instead of being rejected before ranking.
    """
    normalized_type = normalize_type(canonical_type or "")
    allowed = CANONICAL_TYPE_TO_UMLS_SEMANTIC_TYPES.get(normalized_type)
    if not allowed:
        return SEMANTIC_UNKNOWN_LOCAL_TYPE

    observed = {
        str(value).strip().casefold()
        for value in semantic_types
        if str(value).strip()
    }
    if not observed:
        return SEMANTIC_TYPES_MISSING

    allowed_normalized = {value.casefold() for value in allowed}
    if observed & allowed_normalized:
        return SEMANTIC_COMPATIBLE

    if normalized_type in CONTEXTUAL_CANONICAL_TYPES:
        return SEMANTIC_CONTEXTUAL_MISMATCH

    return SEMANTIC_INCOMPATIBLE


def semantic_types_are_compatible(
    canonical_type: Optional[str],
    semantic_types: Sequence[str],
) -> Optional[bool]:
    """Return the legacy tri-state view of semantic compatibility.

    ``True`` means explicitly compatible, ``False`` means a strong
    incompatibility for an intrinsic local type, and ``None`` means the
    candidate must remain observable/reviewable.
    """
    status = classify_semantic_compatibility(
        canonical_type=canonical_type,
        semantic_types=semantic_types,
    )
    if status == SEMANTIC_COMPATIBLE:
        return True
    if status == SEMANTIC_INCOMPATIBLE:
        return False
    return None


def compute_umls_candidate_score(
    alias: str,
    candidate_name: str,
    search_type: str,
) -> float:
    """
    Compute a conservative lexical score for a UMLS API candidate.

    Unlike the old fixed 0.95/0.85 pseudo-scores, this penalizes candidates that
    add unsupported qualifiers, long operational phrases, or numeric subtypes.
    """
    normalized_alias = normalize_api_alias(alias)
    normalized_name = normalize_api_alias(candidate_name)

    if not normalized_alias or not normalized_name:
        return 0.0
    if normalized_alias == normalized_name:
        return 1.0

    alias_tokens = _matching_tokens(alias)
    name_tokens = _matching_tokens(candidate_name)
    alias_set = set(alias_tokens)
    name_set = set(name_tokens)

    if alias_set and alias_set == name_set:
        return 0.98

    overlap = alias_set & name_set
    recall = len(overlap) / len(alias_set) if alias_set else 0.0
    precision = len(overlap) / len(name_set) if name_set else 0.0

    raw_similarity = SequenceMatcher(
        None,
        normalized_alias,
        normalized_name,
    ).ratio()
    token_sort_similarity = SequenceMatcher(
        None,
        " ".join(sorted(alias_tokens)),
        " ".join(sorted(name_tokens)),
    ).ratio()
    character_similarity = max(raw_similarity, token_sort_similarity)

    score = (
        0.50 * character_similarity
        + 0.30 * recall
        + 0.20 * precision
    )

    if search_type == "exact":
        score += 0.03
    elif search_type == "normalizedString":
        score += 0.01

    alias_numeric = {token for token in alias_tokens if token.isdigit()}
    name_numeric = {token for token in name_tokens if token.isdigit()}
    if name_numeric - alias_numeric:
        score -= 0.12

    if precision < 0.75:
        score -= min(0.20, (0.75 - precision) * 0.35)

    return round(max(0.0, min(score, 1.0)), 4)


def compute_umls_selection_score(match: UMLSMatch) -> float:
    """Return the conservative score used to choose among candidates.

    The stored ``match.score`` remains the acceptance score. Exact API rank is
    intentionally only a tie-breaker elsewhere: it must not manufacture a
    score advantage over a lexically stronger compatible candidate. Broad
    search strategies retain their small penalties.
    """
    score = float(match.score)
    search_type = str(match.search_type or "")

    if search_type == "exact":
        score += EXACT_NON_PRIMARY_SELECTION_BONUS
    elif search_type == "normalizedString":
        score -= NORMALIZED_STRING_SELECTION_PENALTY
    elif search_type == "words":
        score -= WORDS_SELECTION_PENALTY

    return round(max(0.0, min(score, 1.0)), 4)


def _resolved_selection_score(match: UMLSMatch) -> float:
    if match.selection_score is None:
        match.selection_score = compute_umls_selection_score(match)
    return float(match.selection_score)


def _contains_token_phrase(
    tokens: Sequence[str],
    phrase: Sequence[str],
) -> bool:
    if not tokens or not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    target = tuple(phrase)
    return any(
        tuple(tokens[index : index + width]) == target
        for index in range(len(tokens) - width + 1)
    )


def has_disease_specificity_conflict(match: UMLSMatch) -> bool:
    """Return whether a disease candidate adds a restrictive subtype label.

    UMLS atoms can list a generic source term under a narrower CUI. Strong atom
    evidence is therefore insufficient by itself when the preferred UMLS name
    adds a clearly restrictive modifier absent from the local alias. The guard
    is deliberately disease-only and uses a short, conservative modifier list.
    """
    if normalize_type(match.canonical_type or "") != "disease":
        return False

    alias_tokens = _matching_tokens(match.alias)
    canonical_tokens = _matching_tokens(match.canonical_name or "")

    for phrase in DISEASE_NARROWING_MODIFIER_PHRASES:
        if (
            _contains_token_phrase(canonical_tokens, phrase)
            and not _contains_token_phrase(alias_tokens, phrase)
        ):
            return True
    return False


def is_strong_compatible_exact_match(match: UMLSMatch) -> bool:
    """Return whether exact evidence is strong enough for selection priority.

    This is deliberately narrower than "exact search": the candidate must be
    semantically compatible with the local concept type, must not introduce a
    disease-specific narrowing modifier, and its conservative lexical score
    must already meet the exact-match acceptance threshold.
    """
    return bool(
        not has_disease_specificity_conflict(match)
        and str(match.search_type or "") == "exact"
        and str(match.semantic_compatibility or "") == "compatible"
        and float(match.score) >= STRONG_COMPATIBLE_EXACT_SELECTION_THRESHOLD
    )


PRIMARY_ACRONYM_ALIAS_SOURCES = frozenset({
    "mention_acronym_expansion",
    "mention_acronym_expansion_canonicalized",
    "document_acronym_expansion",
    "document_acronym_expansion_canonicalized",
})


def has_primary_acronym_provenance(
    provenance: Optional[Dict[str, Any]],
) -> bool:
    """Return whether an alias is a validated primary acronym expansion."""
    if not provenance:
        return False
    sources = {
        str(value).strip()
        for value in (provenance.get("alias_sources") or [])
        if str(value).strip()
    }
    acronym_shorts = [
        str(value).strip()
        for value in (provenance.get("acronym_shorts") or [])
        if str(value).strip()
    ]
    return bool(acronym_shorts and (sources & PRIMARY_ACRONYM_ALIAS_SOURCES))


def is_supported_primary_acronym_match(
    match: UMLSMatch,
    provenance: Optional[Dict[str, Any]],
) -> bool:
    """Return whether document-specific acronym evidence may affect selection.

    Provenance alone never makes a candidate acceptable. The candidate must
    already be semantically compatible and independently supported either by
    atom/synonym evidence or by the existing strong-compatible-exact rule.
    """
    return bool(
        has_primary_acronym_provenance(provenance)
        and not has_disease_specificity_conflict(match)
        and str(match.semantic_compatibility or "") == SEMANTIC_COMPATIBLE
        and (
            match.synonym_supported
            or is_strong_compatible_exact_match(match)
        )
    )


def _ranked_match_tuple(
    match: UMLSMatch,
    alias_index: int,
    search_index: int,
    alias_provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int, int, float, float, int, int, int, UMLSMatch]:
    """Return the evidence-aware cross-strategy ranking key.

    A validated primary acronym long form gets precedence only when its UMLS
    candidate already has strong compatible evidence. This resolves local
    document meanings such as FTX -> frataxin without allowing an incompatible
    long form to override the local canonical type.
    """
    specificity_safe = not has_disease_specificity_conflict(match)
    acronym_supported = is_supported_primary_acronym_match(
        match,
        alias_provenance,
    )
    return (
        1 if specificity_safe else 0,
        1 if acronym_supported else 0,
        1 if (specificity_safe and match.synonym_supported) else 0,
        1 if is_strong_compatible_exact_match(match) else 0,
        _resolved_selection_score(match),
        float(match.score),
        -alias_index,
        -search_index,
        -int(match.api_rank or 10**9),
        match,
    )


def api_cache_key(
    alias: str,
    search_type: str,
    return_id_type: str,
    version: str,
    page_size: int,
) -> str:
    payload = {
        "alias": normalize_api_alias(alias),
        "page_size": page_size,
        "return_id_type": return_id_type,
        "search_type": search_type,
        "version": version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class UMLSAPIError(RuntimeError):
    pass


class UMLSAPIAuthError(UMLSAPIError):
    pass


class UMLSAPIClient:
    """
    Conservative UMLS REST search client with local JSON-file caching.

    Search results are re-ranked locally using lexical specificity and
    compatibility with the controlled local entity type.
    """

    base_url = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(
        self,
        cache_dir: Path,
        timeout: float = DEFAULT_API_TIMEOUT,
        rate_limit_per_second: float = DEFAULT_API_RATE_LIMIT_PER_SECOND,
        version: str = DEFAULT_UMLS_VERSION,
        page_size: int = DEFAULT_API_PAGE_SIZE,
        session: Optional[Any] = None,
    ) -> None:
        ensure_requests_dependency_available()

        api_key = os.getenv("UMLS_API_KEY")
        if not api_key:
            raise UMLSAPIAuthError("UMLS_API_KEY is missing/invalid")

        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.rate_limit_per_second = max(float(rate_limit_per_second), 0.0)
        self.version = str(version or DEFAULT_UMLS_VERSION)
        self.page_size = int(page_size)
        self.session = session if session is not None else requests.Session()
        self.enable_atom_enrichment = True
        self._last_request_at = 0.0
        self.stats: Dict[str, int] = {
            "api_cache_hits": 0,
            "api_cache_misses": 0,
            "api_requests": 0,
            "api_retries": 0,
            "api_errors": 0,
        }

    def cache_path(
        self,
        alias: str,
        search_type: str,
        return_id_type: str,
    ) -> Path:
        key = api_cache_key(
            alias=alias,
            search_type=search_type,
            return_id_type=return_id_type,
            version=self.version,
            page_size=self.page_size,
        )
        return self.cache_dir / f"{key}.json"

    def throttle(self) -> None:
        if self.rate_limit_per_second <= 0:
            return
        min_interval = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def get_cached_payload(
        self,
        alias: str,
        search_type: str,
        return_id_type: str,
    ) -> Optional[Dict[str, Any]]:
        path = self.cache_path(alias, search_type, return_id_type)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        self.stats["api_cache_hits"] += 1
        return payload if isinstance(payload, dict) else None

    def write_cached_payload(
        self,
        alias: str,
        search_type: str,
        return_id_type: str,
        payload: Dict[str, Any],
    ) -> None:
        path = self.cache_path(alias, search_type, return_id_type)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def request_search(
        self,
        alias: str,
        search_type: str,
        return_id_type: str = "concept",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        cached = self.get_cached_payload(alias, search_type, return_id_type)
        if cached is not None:
            return cached

        self.stats["api_cache_misses"] += 1
        params = {
            "apiKey": self.api_key,
            "string": alias,
            "searchType": search_type,
            "returnIdType": return_id_type,
            "pageSize": self.page_size,
        }
        url = f"{self.base_url}/search/{self.version}"

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            self.throttle()
            self._last_request_at = time.monotonic()
            self.stats["api_requests"] += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
            except Exception as e:
                last_error = UMLSAPIError(type(e).__name__)
                self.stats["api_errors"] += 1
            else:
                status_code = int(getattr(response, "status_code", 0))
                if status_code in {401, 403}:
                    raise UMLSAPIAuthError("UMLS_API_KEY is missing/invalid")
                if status_code == 429 or status_code >= 500:
                    self.stats["api_errors"] += 1
                    last_error = UMLSAPIError(
                        f"UMLS API temporary failure: HTTP {status_code}"
                    )
                elif status_code >= 400:
                    self.stats["api_errors"] += 1
                    raise UMLSAPIError(f"UMLS API request failed: HTTP {status_code}")
                else:
                    try:
                        payload = response.json()
                    except Exception as e:
                        self.stats["api_errors"] += 1
                        raise UMLSAPIError("Malformed UMLS API response") from e
                    if not isinstance(payload, dict):
                        raise UMLSAPIError("Malformed UMLS API response")
                    self.write_cached_payload(
                        alias=alias,
                        search_type=search_type,
                        return_id_type=return_id_type,
                        payload=payload,
                    )
                    return payload

            if attempt < max_retries:
                self.stats["api_retries"] += 1
                time.sleep(min(2**attempt, 4))

        if last_error is not None:
            raise UMLSAPIError("UMLS API request failed")
        raise UMLSAPIError("UMLS API request failed")

    def atom_cache_path(self, cui: str) -> Path:
        payload = {
            "cui": str(cui).strip(),
            "version": self.version,
            "page_size": DEFAULT_ATOM_PAGE_SIZE,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"atoms_{key}.json"

    def get_cached_atoms_payload(self, cui: str) -> Optional[Dict[str, Any]]:
        path = self.atom_cache_path(cui)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        self.stats["api_cache_hits"] += 1
        return payload if isinstance(payload, dict) else None

    def write_cached_atoms_payload(
        self,
        cui: str,
        payload: Dict[str, Any],
    ) -> None:
        self.atom_cache_path(cui).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def _request_atom_page(
        self,
        cui: str,
        page_number: int,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        params = {
            "apiKey": self.api_key,
            "pageNumber": page_number,
            "pageSize": DEFAULT_ATOM_PAGE_SIZE,
        }
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}/atoms"
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            self.throttle()
            self._last_request_at = time.monotonic()
            self.stats["api_requests"] += 1
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
            except Exception as exc:
                last_error = UMLSAPIError(type(exc).__name__)
                self.stats["api_errors"] += 1
            else:
                status_code = int(getattr(response, "status_code", 0))
                if status_code in {401, 403}:
                    raise UMLSAPIAuthError("UMLS_API_KEY is missing/invalid")
                if status_code == 429 or status_code >= 500:
                    self.stats["api_errors"] += 1
                    last_error = UMLSAPIError(
                        f"UMLS API temporary failure: HTTP {status_code}"
                    )
                elif status_code >= 400:
                    self.stats["api_errors"] += 1
                    raise UMLSAPIError(
                        f"UMLS API atom request failed: HTTP {status_code}"
                    )
                else:
                    try:
                        payload = response.json()
                    except Exception as exc:
                        self.stats["api_errors"] += 1
                        raise UMLSAPIError(
                            "Malformed UMLS API atom response"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise UMLSAPIError(
                            "Malformed UMLS API atom response"
                        )
                    return payload

            if attempt < max_retries:
                self.stats["api_retries"] += 1
                time.sleep(min(2**attempt, 4))

        if last_error is not None:
            raise UMLSAPIError("UMLS API atom request failed")
        raise UMLSAPIError("UMLS API atom request failed")

    def request_atoms(
        self,
        cui: str,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        cached = self.get_cached_atoms_payload(cui)
        if cached is not None:
            return cached

        self.stats["api_cache_misses"] += 1
        all_atoms: List[Dict[str, Any]] = []
        page_number = 1
        page_count: Optional[int] = None

        while True:
            payload = self._request_atom_page(
                cui=cui,
                page_number=page_number,
                max_retries=max_retries,
            )
            result = payload.get("result")
            if not isinstance(result, list):
                raise UMLSAPIError("Malformed UMLS API atom response")

            all_atoms.extend(
                item for item in result if isinstance(item, dict)
            )

            raw_page_count = payload.get("pageCount")
            try:
                page_count = (
                    int(raw_page_count)
                    if raw_page_count is not None
                    else page_count
                )
            except (TypeError, ValueError):
                pass

            if page_count is not None and page_number >= page_count:
                break
            if len(result) < DEFAULT_ATOM_PAGE_SIZE:
                break

            page_number += 1
            if page_number > 100:
                raise UMLSAPIError(
                    f"Unexpected UMLS atom pagination for {cui}"
                )

        combined = {
            "result": all_atoms,
            "pageCount": page_count or page_number,
        }
        self.write_cached_atoms_payload(cui, combined)
        return combined

    def find_best_atom_match(
        self,
        alias: str,
        cui: str,
    ) -> Optional[Dict[str, Any]]:
        payload = self.request_atoms(cui)
        atoms = payload.get("result")
        if not isinstance(atoms, list):
            raise UMLSAPIError("Malformed UMLS API atom response")

        normalized_alias = normalize_api_alias(alias)
        ranked: List[Tuple[int, float, int, Dict[str, Any]]] = []
        exact_atoms: List[Dict[str, Any]] = []

        for index, atom in enumerate(atoms):
            if not isinstance(atom, dict):
                continue
            name = str(atom.get("name") or "").strip()
            if not name:
                continue

            normalized_name = normalize_api_alias(name)
            exact_match = normalized_name == normalized_alias
            score = compute_umls_candidate_score(
                alias=alias,
                candidate_name=name,
                search_type="exact",
            )
            term_type = str(atom.get("termType") or "").strip()
            preferred_bonus = 1 if term_type in {"PT", "PN", "MH"} else 0

            enriched = {
                "name": name,
                "root_source": str(atom.get("rootSource") or "").strip(),
                "term_type": term_type,
                "score": round(score, 4),
                "exact_match": exact_match,
            }
            if exact_match:
                exact_atoms.append(enriched)

            ranked.append(
                (
                    1 if exact_match else 0,
                    score,
                    preferred_bonus,
                    -index,
                    enriched,
                )
            )

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[:-1], reverse=True)
        best = dict(ranked[0][-1])
        exact_sources = {
            atom["root_source"]
            for atom in exact_atoms
            if atom.get("root_source")
        }
        best["exact_atom_count"] = len(exact_atoms)
        best["exact_atom_source_count"] = len(exact_sources)
        best["exact_atom_sources"] = sorted(exact_sources)
        return best

    def enrich_match_with_atom_evidence(
        self,
        match: UMLSMatch,
    ) -> UMLSMatch:
        if not getattr(self, "enable_atom_enrichment", False):
            return match
        if match.search_type != "exact":
            return match
        if match.score >= DEFAULT_EXACT_UMLS_THRESHOLD:
            return match

        try:
            atom = self.find_best_atom_match(match.alias, match.cui)
        except UMLSAPIError as exc:
            logger.warning(
                "UMLS atom enrichment failed | cui=%s | error=%s",
                match.cui,
                type(exc).__name__,
            )
            return match

        if atom is None:
            return match

        match.matched_atom_name = atom.get("name")
        match.matched_atom_source = atom.get("root_source")
        match.matched_atom_term_type = atom.get("term_type")
        match.matched_atom_score = atom.get("score")
        match.matched_atom_count = int(atom.get("exact_atom_count") or 0)
        match.matched_atom_source_count = int(
            atom.get("exact_atom_source_count") or 0
        )

        canonical_type = normalize_type(match.canonical_type or "")
        exact_atom_sources = {
            str(source).strip()
            for source in atom.get("exact_atom_sources") or []
            if str(source).strip()
        }
        semantic_types = {
            str(value).strip()
            for value in match.semantic_types
            if str(value).strip()
        }

        disease_synonym_supported = bool(
            atom.get("exact_match")
            and match.matched_atom_source_count
            >= MIN_EXACT_ATOM_SOURCE_COUNT
            and canonical_type in SYNONYM_AUTO_ACCEPT_CANONICAL_TYPES
        )
        hgnc_gene_supported = bool(
            atom.get("exact_match")
            and canonical_type == "genetic_factor"
            and "Gene or Genome" in semantic_types
            and "HGNC" in exact_atom_sources
        )

        match.synonym_supported = bool(
            disease_synonym_supported or hgnc_gene_supported
        )
        return match

    def search_alias_candidates(
        self,
        alias: str,
        search_type: str,
        canonical_type: Optional[str] = None,
        trace_limit: int = DEFAULT_MAX_CANDIDATES,
    ) -> Tuple[Optional[UMLSMatch], List[Dict[str, Any]]]:
        """Return the selected candidate plus an auditable bounded trace.

        Selection still considers every valid result returned by the API. The
        trace retains the first ``trace_limit`` API-ranked candidates and also
        retains the selected candidate when it falls outside that window.
        Strongly incompatible candidates remain visible in the trace but are
        never eligible for selection.
        """
        if trace_limit < 1:
            raise ValueError("trace_limit must be >= 1")

        payload = self.request_search(alias=alias, search_type=search_type)
        result = payload.get("result") if isinstance(payload, dict) else None
        results = result.get("results") if isinstance(result, dict) else None
        if not isinstance(results, list):
            raise UMLSAPIError("Malformed UMLS API response")

        selectable: List[Tuple[float, float, int, UMLSMatch]] = []
        trace_rows: List[Dict[str, Any]] = []

        for zero_based_rank, item in enumerate(results):
            if not isinstance(item, dict):
                continue

            cui = str(item.get("ui") or "").strip()
            name = str(item.get("name") or "").strip()
            if not cui or cui.upper() == "NONE" or not name:
                continue

            semantic_types = item.get("semanticTypes") or item.get("semanticType") or []
            if isinstance(semantic_types, str):
                semantic_types = [semantic_types]
            if not isinstance(semantic_types, list):
                semantic_types = []
            semantic_types = [
                str(value).strip()
                for value in semantic_types
                if str(value).strip()
            ]

            semantic_compatibility = classify_semantic_compatibility(
                canonical_type=canonical_type,
                semantic_types=semantic_types,
            )
            type_compatible = semantic_types_are_compatible(
                canonical_type=canonical_type,
                semantic_types=semantic_types,
            )

            lexical_score = compute_umls_candidate_score(
                alias=alias,
                candidate_name=name,
                search_type=search_type,
            )
            adjusted_score = lexical_score
            if type_compatible is None and canonical_type:
                adjusted_score = max(0.0, adjusted_score - 0.02)

            api_rank = zero_based_rank + 1
            match = UMLSMatch(
                alias=alias,
                cui=cui,
                canonical_name=name,
                definition=None,
                aliases=[],
                score=round(adjusted_score, 4),
                semantic_types=semantic_types,
                search_type=search_type,
                type_compatible=type_compatible,
                semantic_compatibility=semantic_compatibility,
                api_rank=api_rank,
                canonical_type=canonical_type,
            )
            match.selection_score = compute_umls_selection_score(match)

            selection_eligible = type_compatible is not False
            trace_rows.append(
                {
                    "query_alias": alias,
                    "search_type": search_type,
                    "api_rank": api_rank,
                    "cui": cui,
                    "canonical_name": name,
                    "semantic_types": semantic_types,
                    "lexical_score": round(lexical_score, 4),
                    "adjusted_score": match.score,
                    "selection_score": match.selection_score,
                    "semantic_compatibility": semantic_compatibility,
                    "type_compatible": type_compatible,
                    "selection_eligible": selection_eligible,
                    "exclusion_reason": (
                        None
                        if selection_eligible
                        else "strong_semantic_incompatibility"
                    ),
                    "matched_atom_name": None,
                    "matched_atom_source": None,
                    "matched_atom_term_type": None,
                    "matched_atom_score": None,
                    "matched_atom_count": 0,
                    "matched_atom_source_count": 0,
                    "synonym_supported": False,
                    "selected_for_search_strategy": False,
                    "selected_for_alias": False,
                    "selected_final": False,
                    "retained_reason": None,
                }
            )

            if selection_eligible:
                selectable.append(
                    (
                        _resolved_selection_score(match),
                        match.score,
                        -zero_based_rank,
                        match,
                    )
                )

        selected: Optional[UMLSMatch] = None
        if selectable:
            selectable.sort(
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            selected = selectable[0][3]

        retained = [dict(row) for row in trace_rows[:trace_limit]]
        for row in retained:
            row["retained_reason"] = "top_api_rank"

        if selected is not None:
            selected = self.enrich_match_with_atom_evidence(selected)
            selected_key = (selected.cui, selected.api_rank)
            selected_row = next(
                (
                    row
                    for row in trace_rows
                    if (row["cui"], row["api_rank"]) == selected_key
                ),
                None,
            )
            retained_selected = next(
                (
                    row
                    for row in retained
                    if (row["cui"], row["api_rank"]) == selected_key
                ),
                None,
            )
            if retained_selected is None and selected_row is not None:
                retained_selected = dict(selected_row)
                retained_selected["retained_reason"] = (
                    "selected_for_search_strategy_outside_trace_limit"
                )
                retained.append(retained_selected)
            if retained_selected is not None:
                retained_selected.update(
                    {
                        "matched_atom_name": selected.matched_atom_name,
                        "matched_atom_source": selected.matched_atom_source,
                        "matched_atom_term_type": selected.matched_atom_term_type,
                        "matched_atom_score": selected.matched_atom_score,
                        "matched_atom_count": selected.matched_atom_count,
                        "matched_atom_source_count": (
                            selected.matched_atom_source_count
                        ),
                        "synonym_supported": selected.synonym_supported,
                        "selected_for_search_strategy": True,
                    }
                )

        return selected, retained

    def search_alias(
        self,
        alias: str,
        search_type: str,
        canonical_type: Optional[str] = None,
    ) -> Optional[UMLSMatch]:
        selected, _ = self.search_alias_candidates(
            alias=alias,
            search_type=search_type,
            canonical_type=canonical_type,
            trace_limit=DEFAULT_MAX_CANDIDATES,
        )
        return selected


def get_default_api_cache_dir(
    api_cache_dir: Optional[Path],
    review_output_dir: Optional[Path],
) -> Path:
    if api_cache_dir is not None:
        return Path(api_cache_dir)
    if review_output_dir is not None:
        return Path(review_output_dir).parent / "umls_api_cache"
    return DEFAULT_ENTITY_REVIEW_DIR.parent / "umls_api_cache"


def select_best_umls_api_match(
    aliases: Sequence[str],
    client: UMLSAPIClient,
    canonical_type: Optional[str] = None,
    alias_provenance: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[UMLSMatch]:
    """Select the best candidate using evidence-aware cross-strategy ranking."""
    matches: List[
        Tuple[int, int, int, int, float, float, int, int, int, UMLSMatch]
    ] = []
    provenance_rows = list(alias_provenance or [])

    for alias_index, alias in enumerate(aliases):
        alias = str(alias or "").strip()
        if not alias:
            continue

        provenance = (
            provenance_rows[alias_index]
            if alias_index < len(provenance_rows)
            else None
        )

        for search_index, search_type in enumerate(UMLS_API_SEARCH_TYPES):
            match = client.search_alias(
                alias=alias,
                search_type=search_type,
                canonical_type=canonical_type,
            )
            if match is None or match.type_compatible is False:
                continue

            matches.append(
                _ranked_match_tuple(
                    match,
                    alias_index,
                    search_index,
                    alias_provenance=provenance,
                )
            )

    if not matches:
        return None

    matches.sort(key=lambda item: item[:-1], reverse=True)
    return matches[0][-1]

def select_best_umls_api_match_with_trace(
    aliases: Sequence[str],
    client: UMLSAPIClient,
    canonical_type: Optional[str] = None,
    alias_provenance: Optional[Sequence[Dict[str, Any]]] = None,
    trace_limit: int = DEFAULT_MAX_CANDIDATES,
) -> Tuple[Optional[UMLSMatch], List[Dict[str, Any]]]:
    """Select the normalizer result while retaining every queried strategy."""
    if trace_limit < 1:
        raise ValueError("trace_limit must be >= 1")

    matches: List[Tuple[int, int, int, int, float, float, int, int, int, UMLSMatch]] = []
    trace: List[Dict[str, Any]] = []
    provenance_rows = list(alias_provenance or [])

    for alias_index, alias in enumerate(aliases):
        alias = str(alias or "").strip()
        if not alias:
            continue

        provenance = (
            provenance_rows[alias_index]
            if alias_index < len(provenance_rows)
            else {
                "alias": alias,
                "alias_index": alias_index,
                "alias_sources": ["ordered_alias"],
                "alias_doc_ids": [],
                "acronym_shorts": [],
            }
        )
        alias_matches: List[
            Tuple[int, int, int, int, float, float, int, int, int, UMLSMatch]
        ] = []

        for search_index, search_type in enumerate(UMLS_API_SEARCH_TYPES):
            match, query_trace = client.search_alias_candidates(
                alias=alias,
                search_type=search_type,
                canonical_type=canonical_type,
                trace_limit=trace_limit,
            )
            for row in query_trace:
                enriched = dict(row)
                enriched.update(
                    {
                        "alias_index": alias_index,
                        "search_index": search_index,
                        "alias_sources": list(
                            provenance.get("alias_sources") or []
                        ),
                        "alias_doc_ids": list(
                            provenance.get("alias_doc_ids") or []
                        ),
                        "acronym_shorts": list(
                            provenance.get("acronym_shorts") or []
                        ),
                    }
                )
                trace.append(enriched)

            if match is None or match.type_compatible is False:
                continue

            ranked = _ranked_match_tuple(
                match,
                alias_index,
                search_index,
                alias_provenance=provenance,
            )
            alias_matches.append(ranked)
            matches.append(ranked)

        if alias_matches:
            alias_matches.sort(key=lambda item: item[:-1], reverse=True)
            alias_selected = alias_matches[0][-1]
            for row in trace:
                if (
                    row.get("alias_index") == alias_index
                    and row.get("query_alias") == alias_selected.alias
                    and row.get("search_type") == alias_selected.search_type
                    and row.get("cui") == alias_selected.cui
                    and row.get("api_rank") == alias_selected.api_rank
                ):
                    row["selected_for_alias"] = True
                    break

    if matches:
        matches.sort(key=lambda item: item[:-1], reverse=True)
        selected = matches[0][-1]
    else:
        selected = None

    if selected is not None:
        for row in trace:
            if (
                row.get("query_alias") == selected.alias
                and row.get("search_type") == selected.search_type
                and row.get("cui") == selected.cui
                and row.get("api_rank") == selected.api_rank
            ):
                row["selected_final"] = True
                break

    return selected, trace


def build_existing_umls_match(concept: ConceptRecord) -> Optional[UMLSMatch]:
    cui = concept.properties.get("umls_cui")
    if not cui:
        return None

    raw_aliases = concept.properties.get("umls_aliases") or []
    if not isinstance(raw_aliases, list):
        raw_aliases = []

    raw_score = concept.properties.get("umls_score")
    try:
        score = float(raw_score) if raw_score is not None else 1.0
    except (TypeError, ValueError):
        score = 1.0

    raw_semantic_types = concept.properties.get("umls_semantic_types") or []
    if not isinstance(raw_semantic_types, list):
        raw_semantic_types = []
    semantic_types = [str(value) for value in raw_semantic_types if value]
    semantic_compatibility = classify_semantic_compatibility(
        canonical_type=concept.canonical_type,
        semantic_types=semantic_types,
    )

    return UMLSMatch(
        alias=concept.name,
        cui=str(cui),
        canonical_name=concept.properties.get("umls_canonical_name"),
        definition=concept.properties.get("umls_definition"),
        aliases=[str(alias) for alias in raw_aliases if alias],
        score=round(score, 4),
        semantic_types=semantic_types,
        type_compatible=semantic_types_are_compatible(
            canonical_type=concept.canonical_type,
            semantic_types=semantic_types,
        ),
        semantic_compatibility=semantic_compatibility,
        canonical_type=concept.canonical_type,
    )


def write_concept_umls_match(
    tx,
    concept_id: str,
    match: UMLSMatch,
    model_name: str,
    linker_name: str,
    normalization_method: str,
    normalized_at: str,
) -> None:
    tx.run(
        """
        MATCH (c:Concept)
        WHERE elementId(c) = $concept_id
        SET c.umls_cui = $umls_cui,
            c.umls_canonical_name = $umls_canonical_name,
            c.umls_definition = $umls_definition,
            c.umls_aliases = $umls_aliases,
            c.umls_score = $umls_score,
            c.umls_semantic_types = $umls_semantic_types,
            c.umls_linker_name = $umls_linker_name,
            c.umls_model_name = $umls_model_name,
            c.normalized_name = $normalized_name,
            c.normalization_status = 'umls_matched',
            c.normalization_method = $normalization_method,
            c.normalized_at = datetime($normalized_at),
            c.updated_at = datetime()
        """,
        concept_id=concept_id,
        umls_cui=match.cui,
        umls_canonical_name=match.canonical_name,
        umls_definition=match.definition,
        umls_aliases=match.aliases,
        umls_score=match.score,
        umls_semantic_types=match.semantic_types,
        umls_linker_name=linker_name,
        umls_model_name=model_name,
        normalized_name=normalize_name(match.canonical_name or match.alias),
        normalization_method=normalization_method,
        normalized_at=normalized_at,
    )


def write_concept_normalization_status(
    tx,
    concept_id: str,
    status: str,
    method: str,
    normalized_at: str,
    normalized_name: Optional[str] = None,
) -> None:
    """
    Write a non-matched normalization status and clear stale UMLS metadata.

    This is required when force-reprocessing a concept that previously had an
    incorrect UMLS match.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE elementId(c) = $concept_id
        SET c.normalization_status = $status,
            c.normalization_method = $method,
            c.normalized_name = coalesce($normalized_name, c.normalized_name),
            c.normalized_at = datetime($normalized_at),
            c.updated_at = datetime()
        REMOVE c.umls_cui,
               c.umls_canonical_name,
               c.umls_definition,
               c.umls_aliases,
               c.umls_score,
               c.umls_semantic_types,
               c.umls_linker_name,
               c.umls_model_name
        """,
        concept_id=concept_id,
        status=status,
        method=method,
        normalized_name=normalized_name,
        normalized_at=normalized_at,
    )


def effective_umls_acceptance_score(match: UMLSMatch) -> float:
    """Return acceptance confidence without mutating the lexical score."""
    if match.synonym_supported:
        return 1.0
    return float(match.score)


def is_confident_umls_match(
    match: Optional[UMLSMatch],
    threshold: float,
    exact_threshold: float = DEFAULT_EXACT_UMLS_THRESHOLD,
) -> bool:
    """Return whether a UMLS candidate is safe to accept automatically."""
    if match is None or match.type_compatible is False:
        return False
    if has_disease_specificity_conflict(match):
        return False

    acceptance_score = effective_umls_acceptance_score(match)

    if match.search_type == "exact":
        return acceptance_score >= exact_threshold
    if match.search_type == "words":
        return False
    return acceptance_score >= threshold


def is_plausible_umls_match(
    match: Optional[UMLSMatch],
    plausible_threshold: float = DEFAULT_PLAUSIBLE_UMLS_THRESHOLD,
) -> bool:
    """Return whether a rejected candidate is still useful for review."""
    if match is None or match.type_compatible is False:
        return False
    if match.synonym_supported:
        return True
    if (match.matched_atom_score or 0.0) >= plausible_threshold:
        return True
    return float(match.score) >= plausible_threshold


def update_concept_from_result(
    tx,
    concept: ConceptRecord,
    model_name: str,
    linker_name: str,
    threshold: float,
    force: bool,
    normalized_at: str,
    match_method: str = AUTO_NORMALIZATION_METHOD,
    low_confidence_method: str = LOW_CONFIDENCE_METHOD,
    no_match_method: str = NO_MATCH_METHOD,
    exact_threshold: float = DEFAULT_EXACT_UMLS_THRESHOLD,
) -> None:
    if should_preserve_existing_normalization(concept, force=force):
        concept.normalization_status = "skipped"
        concept.normalization_method = SKIPPED_METHOD
        concept.reason = "existing_normalization_preserved"
        return

    match = concept.best_match

    if is_confident_umls_match(
        match,
        threshold=threshold,
        exact_threshold=exact_threshold,
    ):
        concept.normalization_status = "umls_matched"
        concept.normalization_method = match_method
        concept.reason = "best_candidate_above_threshold"
        write_concept_umls_match(
            tx=tx,
            concept_id=concept.concept_id,
            match=match,
            model_name=model_name,
            linker_name=linker_name,
            normalization_method=match_method,
            normalized_at=normalized_at,
        )
        return

    if match is not None:
        if not is_plausible_umls_match(match):
            concept.normalization_status = NO_PLAUSIBLE_MATCH_STATUS
            concept.normalization_method = NO_PLAUSIBLE_MATCH_METHOD
            concept.reason = NO_PLAUSIBLE_MATCH_REASON
            write_concept_normalization_status(
                tx=tx,
                concept_id=concept.concept_id,
                status=concept.normalization_status,
                method=concept.normalization_method,
                normalized_at=normalized_at,
                normalized_name=normalize_name(concept.name),
            )
            return

        concept.normalization_status = "umls_low_confidence"
        concept.normalization_method = low_confidence_method
        concept.reason = "best_candidate_below_threshold"
        write_concept_normalization_status(
            tx=tx,
            concept_id=concept.concept_id,
            status=concept.normalization_status,
            method=concept.normalization_method,
            normalized_at=normalized_at,
            normalized_name=normalize_name(concept.name),
        )
        return

    concept.normalization_status = "umls_no_match"
    concept.normalization_method = no_match_method
    concept.reason = "no_linker_candidates"
    write_concept_normalization_status(
        tx=tx,
        concept_id=concept.concept_id,
        status=concept.normalization_status,
        method=concept.normalization_method,
        normalized_at=normalized_at,
    )


def create_same_as_edge(
    tx,
    source_id: str,
    target_id: str,
    normalized_at: str,
) -> None:
    tx.run(
        """
        MATCH (a:Concept)
        WHERE elementId(a) = $source_id
        MATCH (b:Concept)
        WHERE elementId(b) = $target_id
        MERGE (a)-[r:SAME_AS]->(b)
        ON CREATE SET r.created_at = datetime($normalized_at)
        SET r += $relationship_metadata,
            r.method = $method,
            r.score = 1.0,
            r.status = 'auto',
            r.updated_at = datetime($normalized_at)
        """,
        source_id=source_id,
        target_id=target_id,
        method=SAME_CUI_METHOD,
        normalized_at=normalized_at,
        relationship_metadata=build_normalization_relationship_metadata("SAME_AS"),
    )


def create_possibly_same_as_edge(
    tx,
    source_id: str,
    target_id: str,
    score: float,
    normalized_at: str,
) -> None:
    tx.run(
        """
        MATCH (a:Concept)
        WHERE elementId(a) = $source_id
        MATCH (b:Concept)
        WHERE elementId(b) = $target_id
        MERGE (a)-[r:POSSIBLY_SAME_AS]->(b)
        ON CREATE SET r.created_at = datetime($normalized_at)
        SET r += $relationship_metadata,
            r.method = $method,
            r.score = $score,
            r.status = 'candidate',
            r.updated_at = datetime($normalized_at)
        """,
        source_id=source_id,
        target_id=target_id,
        method=FUZZY_METHOD,
        score=score,
        normalized_at=normalized_at,
        relationship_metadata=build_normalization_relationship_metadata(
            "POSSIBLY_SAME_AS"
        ),
    )


def edge_key(left: ConceptRecord, right: ConceptRecord) -> Tuple[str, str]:
    return tuple(sorted([left.concept_id, right.concept_id]))


def ordered_edge_pair(
    left: ConceptRecord,
    right: ConceptRecord,
) -> Tuple[ConceptRecord, ConceptRecord]:
    if left.concept_id <= right.concept_id:
        return left, right
    return right, left


def compute_same_cui_pairs(
    concepts: Sequence[ConceptRecord],
) -> List[Tuple[ConceptRecord, ConceptRecord]]:
    by_cui: Dict[str, List[ConceptRecord]] = {}

    for concept in concepts:
        if concept.normalization_status not in {"umls_matched", "skipped"}:
            continue
        if not concept.best_match:
            continue
        by_cui.setdefault(concept.best_match.cui, []).append(concept)

    pairs: List[Tuple[ConceptRecord, ConceptRecord]] = []

    for grouped in by_cui.values():
        grouped = sorted(grouped, key=lambda item: item.concept_id)
        for i, left in enumerate(grouped):
            for right in grouped[i + 1:]:
                pairs.append((left, right))

    return pairs


def is_short_or_acronym_like_name(name: str) -> bool:
    raw_compact = re.sub(r"[^A-Za-z0-9]", "", str(name or ""))
    normalized_compact = raw_compact.casefold()

    if len(normalized_compact) <= 3:
        return True
    if raw_compact.isupper() and len(raw_compact) <= 12:
        return True
    return False


def has_acronym_alias_evidence(left: ConceptRecord, right: ConceptRecord) -> bool:
    left_aliases = {normalize_name(alias) for alias in left.aliases_considered}
    right_aliases = {normalize_name(alias) for alias in right.aliases_considered}

    return (
        normalize_name(left.name) in right_aliases
        or normalize_name(right.name) in left_aliases
        or bool(left_aliases & right_aliases)
    )


def names_are_comparable(left_name: str, right_name: str) -> bool:
    left = normalize_name(left_name)
    right = normalize_name(right_name)

    if not left or not right or left == right:
        return False

    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))

    if longer == 0:
        return False

    if shorter / longer < 0.75:
        return False

    if len(left.split()) < 2 and len(right.split()) < 2:
        return False

    return True


def compute_fuzzy_score(left_name: str, right_name: str) -> float:
    if fuzz is None:
        ensure_fuzzy_dependency_available()
    assert fuzz is not None

    token_sort = fuzz.token_sort_ratio(left_name, right_name)
    token_set = fuzz.token_set_ratio(left_name, right_name)
    return float(max(token_sort, token_set))


def compute_fuzzy_pairs(
    concepts: Sequence[ConceptRecord],
    fuzzy_threshold: int,
    same_as_keys: set[Tuple[str, str]],
) -> List[Tuple[ConceptRecord, ConceptRecord, float]]:
    pairs: List[Tuple[ConceptRecord, ConceptRecord, float]] = []
    sorted_concepts = sorted(concepts, key=lambda item: item.concept_id)

    for i, left in enumerate(sorted_concepts):
        for right in sorted_concepts[i + 1:]:
            if edge_key(left, right) in same_as_keys:
                continue

            left_short = is_short_or_acronym_like_name(left.name)
            right_short = is_short_or_acronym_like_name(right.name)

            if (left_short or right_short) and not has_acronym_alias_evidence(left, right):
                continue

            if not names_are_comparable(left.name, right.name):
                continue

            score = compute_fuzzy_score(left.name, right.name)
            if score >= fuzzy_threshold:
                pairs.append((left, right, round(score, 2)))

    return pairs


def attach_duplicate_review_candidate(
    concept: ConceptRecord,
    other: ConceptRecord,
    relationship: str,
    method: str,
    score: float,
    status: str,
) -> None:
    concept.duplicate_candidates.append(
        {
            "relationship": relationship,
            "method": method,
            "score": score,
            "status": status,
            "concept_id": other.concept_id,
            "name": other.name,
            "canonical_type": other.canonical_type,
            "umls_cui": other.best_match.cui if other.best_match else None,
        }
    )


def create_duplicate_evidence(
    driver: Driver,
    concepts: Sequence[ConceptRecord],
    fuzzy_threshold: int,
    normalized_at: str,
    dry_run: bool,
    create_same_as_edges: bool = False,
    create_fuzzy_candidate_edges: bool = False,
) -> Dict[str, int]:
    eligible_concepts = [
        concept
        for concept in concepts
        if not concept_requires_type_review(concept)
        and concept.normalization_status != REVIEW_REQUIRED_STATUS
    ]
    same_as_pairs = compute_same_cui_pairs(eligible_concepts)
    same_as_keys = {edge_key(left, right) for left, right in same_as_pairs}
    fuzzy_pairs = compute_fuzzy_pairs(
        concepts=eligible_concepts,
        fuzzy_threshold=fuzzy_threshold,
        same_as_keys=same_as_keys,
    )

    for left, right in same_as_pairs:
        attach_duplicate_review_candidate(
            left,
            right,
            relationship="SAME_AS",
            method=SAME_CUI_METHOD,
            score=1.0,
            status="auto",
        )
        attach_duplicate_review_candidate(
            right,
            left,
            relationship="SAME_AS",
            method=SAME_CUI_METHOD,
            score=1.0,
            status="auto",
        )

    for left, right, score in fuzzy_pairs:
        attach_duplicate_review_candidate(
            left,
            right,
            relationship="POSSIBLY_SAME_AS",
            method=FUZZY_METHOD,
            score=score,
            status="candidate",
        )
        attach_duplicate_review_candidate(
            right,
            left,
            relationship="POSSIBLY_SAME_AS",
            method=FUZZY_METHOD,
            score=score,
            status="candidate",
        )

    if dry_run or (not create_same_as_edges and not create_fuzzy_candidate_edges):
        return {
            "same_as_edges_created": 0,
            "possibly_same_as_edges_created": 0,
        }

    same_as_edges_created = 0
    possibly_same_as_edges_created = 0

    with driver.session() as session:
        if create_same_as_edges:
            for left, right in same_as_pairs:
                source, target = ordered_edge_pair(left, right)
                session.execute_write(
                    create_same_as_edge,
                    source.concept_id,
                    target.concept_id,
                    normalized_at,
                )
                same_as_edges_created += 1

        if create_fuzzy_candidate_edges:
            for left, right, score in fuzzy_pairs:
                source, target = ordered_edge_pair(left, right)
                session.execute_write(
                    create_possibly_same_as_edge,
                    source.concept_id,
                    target.concept_id,
                    score,
                    normalized_at,
                )
                possibly_same_as_edges_created += 1

    return {
        "same_as_edges_created": same_as_edges_created,
        "possibly_same_as_edges_created": possibly_same_as_edges_created,
    }


def build_review_record(
    concept: ConceptRecord,
    run_id: str,
    doc_id: Optional[str],
    model_name: str,
    linker_name: str,
    backend: str,
    threshold: float,
    fuzzy_threshold: int,
    exact_threshold: float = DEFAULT_EXACT_UMLS_THRESHOLD,
) -> Dict[str, Any]:
    match = concept.best_match
    expose_match = (
        match is not None
        and concept.normalization_status != NO_PLAUSIBLE_MATCH_STATUS
    )

    return {
        "run_id": run_id,
        "exported_at": utc_now_iso(),
        "doc_id": doc_id,
        "concept_id": concept.concept_id,
        "concept_name": concept.name,
        "canonical_type": concept.canonical_type,
        "normalization_eligible": not concept_requires_type_review(concept),
        "needs_type_review": concept.needs_type_review,
        "type_resolution_status": concept.type_resolution_status,
        "observed_types": concept.observed_types,
        "invalid_observed_types": concept.invalid_observed_types,
        "type_support_pairs": concept.type_support_pairs,
        "aliases_considered": concept.aliases_considered,
        "alias_provenance": concept.alias_provenance,
        "candidate_trace_version": CANDIDATE_TRACE_VERSION,
        "ranking_policy_version": RANKING_POLICY_VERSION,
        "candidate_trace": concept.candidate_trace,
        "normalization_status": concept.normalization_status,
        "normalization_method": concept.normalization_method,
        "reason": concept.reason,
        "umls_cui": match.cui if expose_match else None,
        "umls_canonical_name": (
            match.canonical_name if expose_match else None
        ),
        "umls_definition": match.definition if expose_match else None,
        "umls_aliases": match.aliases if expose_match else [],
        "umls_semantic_types": (
            match.semantic_types if expose_match else []
        ),
        "umls_score": match.score if expose_match else None,
        "umls_matched_alias": match.alias if expose_match else None,
        "umls_search_type": match.search_type if expose_match else None,
        "umls_type_compatible": (
            match.type_compatible if expose_match else None
        ),
        "umls_semantic_compatibility": (
            match.semantic_compatibility if expose_match else None
        ),
        "umls_matched_atom_name": (
            match.matched_atom_name if expose_match else None
        ),
        "umls_matched_atom_source": (
            match.matched_atom_source if expose_match else None
        ),
        "umls_matched_atom_term_type": (
            match.matched_atom_term_type if expose_match else None
        ),
        "umls_matched_atom_score": (
            match.matched_atom_score if expose_match else None
        ),
        "umls_matched_atom_count": (
            match.matched_atom_count if expose_match else 0
        ),
        "umls_matched_atom_source_count": (
            match.matched_atom_source_count if expose_match else 0
        ),
        "umls_synonym_supported": (
            match.synonym_supported if expose_match else False
        ),
        "rejected_umls_cui": (
            match.cui
            if match and not expose_match
            else None
        ),
        "rejected_umls_canonical_name": (
            match.canonical_name
            if match and not expose_match
            else None
        ),
        "rejected_umls_score": (
            match.score
            if match and not expose_match
            else None
        ),
        "backend": backend,
        "model_name": model_name,
        "linker_name": linker_name,
        "threshold": threshold,
        "exact_threshold": exact_threshold,
        "fuzzy_threshold": fuzzy_threshold,
        "candidate_duplicates": concept.duplicate_candidates,
    }


def get_review_output_path(
    doc_id: Optional[str],
    review_output_dir: Optional[Path],
) -> Path:
    review_dir = (
        Path(review_output_dir)
        if review_output_dir is not None
        else DEFAULT_ENTITY_REVIEW_DIR
    )
    review_dir.mkdir(parents=True, exist_ok=True)

    suffix = safe_filename_component(doc_id, fallback="all_docs")
    return review_dir / f"{suffix}_umls_normalization.jsonl"


def write_review_records(
    concepts: Sequence[ConceptRecord],
    doc_id: Optional[str],
    model_name: str,
    linker_name: str,
    backend: str,
    threshold: float,
    fuzzy_threshold: int,
    review_output_dir: Optional[Path],
    run_id: str,
    exact_threshold: float = DEFAULT_EXACT_UMLS_THRESHOLD,
) -> int:
    path = get_review_output_path(
        doc_id=doc_id,
        review_output_dir=review_output_dir,
    )

    with path.open("a", encoding="utf-8") as f:
        for concept in concepts:
            record = build_review_record(
                concept=concept,
                run_id=run_id,
                doc_id=doc_id,
                model_name=model_name,
                linker_name=linker_name,
                backend=backend,
                threshold=threshold,
                exact_threshold=exact_threshold,
                fuzzy_threshold=fuzzy_threshold,
            )
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    logger.info("UMLS normalization review written: %s", path)
    return len(concepts)


def initialize_stats() -> Dict[str, int]:
    return {
        "concepts_seen": 0,
        "concepts_normalized": 0,
        "concepts_no_match": 0,
        "concepts_no_plausible_match": 0,
        "concepts_low_confidence": 0,
        "concepts_failed": 0,
        "concepts_skipped": 0,
        "concepts_review_required": 0,
        "same_as_edges_created": 0,
        "possibly_same_as_edges_created": 0,
        "review_records_written": 0,
        "api_cache_hits": 0,
        "api_cache_misses": 0,
        "api_requests": 0,
        "api_retries": 0,
        "api_errors": 0,
        "candidate_trace_records": 0,
    }


def update_stats_for_concept(stats: Dict[str, int], concept: ConceptRecord) -> None:
    status = concept.normalization_status

    if status == "umls_matched":
        stats["concepts_normalized"] += 1
    elif status == "umls_low_confidence":
        stats["concepts_low_confidence"] += 1
    elif status == NO_PLAUSIBLE_MATCH_STATUS:
        stats["concepts_no_plausible_match"] += 1
    elif status == "umls_no_match":
        stats["concepts_no_match"] += 1
    elif status == "skipped":
        stats["concepts_skipped"] += 1
    elif status == REVIEW_REQUIRED_STATUS:
        stats["concepts_review_required"] += 1
    elif status == "failed":
        stats["concepts_failed"] += 1


def normalize_concepts_with_umls(
    driver: Driver,
    doc_id: Optional[str] = None,
    backend: str = SCISPACY_BACKEND,
    model_name: str = DEFAULT_MODEL_NAME,
    linker_name: str = DEFAULT_LINKER_NAME,
    threshold: float = DEFAULT_UMLS_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    use_acronyms: bool = True,
    acronym_dir: Optional[Path] = None,
    dry_run: bool = False,
    export_review: bool = True,
    review_output_dir: Optional[Path] = None,
    force: bool = False,
    fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD,
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    api_cache_dir: Optional[Path] = None,
    api_timeout: float = DEFAULT_API_TIMEOUT,
    api_rate_limit_per_second: float = DEFAULT_API_RATE_LIMIT_PER_SECOND,
    create_same_as_edges: bool = False,
    create_fuzzy_candidate_edges: bool = False,
    exact_threshold: float = DEFAULT_EXACT_UMLS_THRESHOLD,
    collect_candidate_trace: bool = False,
    concept_names: Optional[Sequence[str]] = None,
    concept_ids: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Normalize existing Concept nodes with UMLS and add duplicate evidence.

    This is an optional post-processing phase. It never deletes or merges
    Concept nodes.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 <= exact_threshold <= 1:
        raise ValueError("exact_threshold must be between 0 and 1")
    if not 0 <= fuzzy_threshold <= 100:
        raise ValueError("fuzzy_threshold must be between 0 and 100")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if concept_names and concept_ids:
        raise ValueError(
            "Choose either concept_names or concept_ids, not both."
        )

    backend = normalize_backend_name(backend)
    stats = initialize_stats()
    stats["backend"] = backend  # type: ignore[assignment]
    normalized_at = utc_now_iso()
    run_id = f"umls_normalization::{normalized_at}"

    with driver.session() as session:
        # A dry run must be read-only with respect to Neo4j. Creating indexes is
        # a schema write, so defer it to real normalization runs only.
        if not dry_run:
            session.execute_write(setup_normalization_schema)
        concepts = session.execute_read(fetch_concepts_for_normalization, doc_id)

    if concept_ids:
        requested_ids = {
            str(concept_id).strip()
            for concept_id in concept_ids
            if str(concept_id).strip()
        }
        concepts = [
            concept
            for concept in concepts
            if str(concept.concept_id) in requested_ids
        ]
    elif concept_names:
        requested_names = {
            normalize_name(name)
            for name in concept_names
            if normalize_name(name)
        }
        concepts = [
            concept
            for concept in concepts
            if normalize_name(concept.name) in requested_names
        ]

    stats["concepts_seen"] = len(concepts)

    if not concepts:
        return stats

    doc_ids = sorted({doc for concept in concepts for doc in concept.doc_ids if doc})
    acronyms_by_doc_id = (
        load_acronyms_by_doc_id(
            acronym_dir=Path(acronym_dir),
            doc_ids=doc_ids,
        )
        if use_acronyms and acronym_dir is not None and doc_ids
        else {}
    )

    backend_work_required = any(
        not should_preserve_existing_normalization(concept, force=force)
        and not concept_requires_type_review(concept)
        for concept in concepts
    )

    nlp = linker = None
    api_client: Optional[UMLSAPIClient] = None
    method_for_match = AUTO_NORMALIZATION_METHOD
    method_for_low_confidence = LOW_CONFIDENCE_METHOD
    method_for_no_match = NO_MATCH_METHOD
    resolved_model_name = model_name
    resolved_linker_name = linker_name

    if backend == SCISPACY_BACKEND and backend_work_required:
        nlp, linker = load_scispacy_pipeline(
            model_name=model_name,
            linker_name=linker_name,
            max_candidates=max_candidates,
            local_files_only=local_files_only,
            min_available_memory_gb=min_available_memory_gb,
        )
    elif backend == UMLS_API_BACKEND and backend_work_required:
        api_client = UMLSAPIClient(
            cache_dir=get_default_api_cache_dir(
                api_cache_dir=api_cache_dir,
                review_output_dir=review_output_dir,
            ),
            timeout=api_timeout,
            rate_limit_per_second=api_rate_limit_per_second,
        )
        method_for_match = API_NORMALIZATION_METHOD
        method_for_low_confidence = API_LOW_CONFIDENCE_METHOD
        method_for_no_match = API_NO_MATCH_METHOD
        resolved_model_name = "UMLS REST API"
        resolved_linker_name = "umls_api"
    elif backend == UMLS_API_BACKEND:
        method_for_match = API_NORMALIZATION_METHOD
        method_for_low_confidence = API_LOW_CONFIDENCE_METHOD
        method_for_no_match = API_NO_MATCH_METHOD
        resolved_model_name = "UMLS REST API"
        resolved_linker_name = "umls_api"
    elif backend == SCISPACY_BACKEND:
        # No eligible concept requires the expensive linker, but preserve the
        # configured backend metadata in the review output.
        pass
    else:
        method_for_match = "fuzzy_only"
        method_for_low_confidence = "fuzzy_only"
        method_for_no_match = "fuzzy_only"
        resolved_model_name = "none"
        resolved_linker_name = "fuzzy_only"

    with driver.session() as session:
        for concept in concepts:
            try:
                concept.aliases_considered = build_aliases_for_concept(
                    concept=concept,
                    acronyms_by_doc_id=acronyms_by_doc_id,
                )
                concept.alias_provenance = build_alias_provenance_for_concept(
                    concept=concept,
                    acronyms_by_doc_id=acronyms_by_doc_id,
                    aliases=concept.aliases_considered,
                )

                if should_preserve_existing_normalization(concept, force=force):
                    concept.best_match = build_existing_umls_match(concept)
                    concept.normalization_status = "skipped"
                    concept.normalization_method = SKIPPED_METHOD
                    concept.reason = "existing_normalization_preserved"
                    update_stats_for_concept(stats, concept)
                    continue

                if concept_requires_type_review(concept):
                    concept.best_match = None
                    concept.normalization_status = REVIEW_REQUIRED_STATUS
                    concept.normalization_method = TYPE_REVIEW_REQUIRED_METHOD
                    concept.reason = TYPE_REVIEW_REQUIRED_REASON

                    update_stats_for_concept(stats, concept)
                    continue

                if backend == SCISPACY_BACKEND:
                    concept.best_match = select_best_umls_match(
                        aliases=concept.aliases_considered,
                        nlp=nlp,
                        linker=linker,
                    )
                elif backend == UMLS_API_BACKEND:
                    assert api_client is not None
                    if collect_candidate_trace:
                        (
                            concept.best_match,
                            concept.candidate_trace,
                        ) = select_best_umls_api_match_with_trace(
                            aliases=concept.aliases_considered,
                            client=api_client,
                            canonical_type=concept.canonical_type,
                            alias_provenance=concept.alias_provenance,
                            trace_limit=max_candidates,
                        )
                    else:
                        concept.best_match = select_best_umls_api_match(
                            aliases=concept.aliases_considered,
                            client=api_client,
                            canonical_type=concept.canonical_type,
                            alias_provenance=concept.alias_provenance,
                        )
                else:
                    concept.best_match = None

                if dry_run:
                    match = concept.best_match
                    if is_confident_umls_match(
                        match,
                        threshold=threshold,
                        exact_threshold=exact_threshold,
                    ):
                        concept.normalization_status = "umls_matched"
                        concept.normalization_method = method_for_match
                        concept.reason = "dry_run_best_candidate_above_threshold"
                    elif match is not None and not is_plausible_umls_match(match):
                        concept.normalization_status = NO_PLAUSIBLE_MATCH_STATUS
                        concept.normalization_method = (
                            API_NO_PLAUSIBLE_MATCH_METHOD
                            if backend == UMLS_API_BACKEND
                            else NO_PLAUSIBLE_MATCH_METHOD
                        )
                        concept.reason = NO_PLAUSIBLE_MATCH_REASON
                    elif match is not None:
                        concept.normalization_status = "umls_low_confidence"
                        concept.normalization_method = method_for_low_confidence
                        concept.reason = "dry_run_best_candidate_below_threshold"
                    elif backend == FUZZY_ONLY_BACKEND:
                        concept.normalization_status = "fuzzy_only"
                        concept.normalization_method = "fuzzy_only"
                        concept.reason = "dry_run_umls_skipped_by_backend"
                    else:
                        concept.normalization_status = "umls_no_match"
                        concept.normalization_method = method_for_no_match
                        concept.reason = "dry_run_no_linker_candidates"
                elif backend == FUZZY_ONLY_BACKEND:
                    concept.normalization_status = "fuzzy_only"
                    concept.normalization_method = "fuzzy_only"
                    concept.reason = "umls_skipped_by_backend"
                else:
                    session.execute_write(
                        update_concept_from_result,
                        concept,
                        resolved_model_name,
                        resolved_linker_name,
                        threshold,
                        force,
                        normalized_at,
                        method_for_match,
                        method_for_low_confidence,
                        method_for_no_match,
                        exact_threshold,
                    )

            except Exception as e:
                logger.exception(
                    "UMLS normalization failed for concept %r: %s",
                    concept.name,
                    e,
                )
                concept.normalization_status = "failed"
                concept.normalization_method = method_for_match
                concept.reason = str(e)
                if not dry_run:
                    session.execute_write(
                        write_concept_normalization_status,
                        concept.concept_id,
                        "failed",
                        method_for_match,
                        normalized_at,
                        normalize_name(concept.name),
                    )

            update_stats_for_concept(stats, concept)
            stats["candidate_trace_records"] += len(concept.candidate_trace)

    duplicate_stats = create_duplicate_evidence(
        driver=driver,
        concepts=concepts,
        fuzzy_threshold=fuzzy_threshold,
        normalized_at=normalized_at,
        dry_run=dry_run,
        create_same_as_edges=create_same_as_edges,
        create_fuzzy_candidate_edges=create_fuzzy_candidate_edges,
    )
    stats.update(duplicate_stats)

    if api_client is not None:
        for key, value in api_client.stats.items():
            stats[key] = value

    if export_review:
        stats["review_records_written"] = write_review_records(
            concepts=concepts,
            doc_id=doc_id,
            model_name=resolved_model_name,
            linker_name=resolved_linker_name,
            backend=backend,
            threshold=threshold,
            exact_threshold=exact_threshold,
            fuzzy_threshold=fuzzy_threshold,
            review_output_dir=review_output_dir,
            run_id=run_id,
        )

    logger.info(
        "UMLS normalization completed | backend=%s | threshold=%.3f | exact_threshold=%.3f | seen=%d | normalized=%d | low_confidence=%d | no_plausible=%d | no_match=%d | failed=%d | skipped=%d | review_required=%d | trace_records=%d | same_as=%d | fuzzy=%d | dry_run=%s",
        backend,
        threshold,
        exact_threshold,
        stats["concepts_seen"],
        stats["concepts_normalized"],
        stats["concepts_low_confidence"],
        stats["concepts_no_plausible_match"],
        stats["concepts_no_match"],
        stats["concepts_failed"],
        stats["concepts_skipped"],
        stats["concepts_review_required"],
        stats["candidate_trace_records"],
        stats["same_as_edges_created"],
        stats["possibly_same_as_edges_created"],
        dry_run,
    )

    return stats


__all__ = [
    "normalize_concepts_with_umls",
    "UMLSAPIClient",
    "UMLSAPIError",
    "UMLSAPIAuthError",
    "UMLSMatch",
    "DEFAULT_EXACT_UMLS_THRESHOLD",
    "setup_normalization_schema",
    "fetch_concepts_for_normalization",
    "build_aliases_for_concept",
    "should_include_secondary_alias",
    "build_alias_provenance_for_concept",
    "compute_umls_candidate_score",
    "semantic_types_are_compatible",
    "classify_semantic_compatibility",
    "compute_umls_selection_score",
    "has_disease_specificity_conflict",
    "effective_umls_acceptance_score",
    "is_plausible_umls_match",
    "concept_requires_type_review",
    "CANDIDATE_TRACE_VERSION",
    "RANKING_POLICY_VERSION",
    "NO_PLAUSIBLE_MATCH_STATUS",
    "REVIEW_REQUIRED_STATUS",
    "TYPE_REVIEW_REQUIRED_METHOD",
    "TYPE_REVIEW_REQUIRED_REASON",
    "select_best_umls_api_match",
    "select_best_umls_api_match_with_trace",
    "compute_fuzzy_pairs",
    "compute_same_cui_pairs",
    "normalize_backend_name",
    "is_confident_umls_match",
]
