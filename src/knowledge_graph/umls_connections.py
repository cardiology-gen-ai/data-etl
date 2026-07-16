"""
umls_connections.py

Export and optional materialization for ontology-derived candidate connections between
already-normalized local Concept nodes.

The default behavior is read-only: generate review files from existing Concept
nodes and cached/UMLS API relation data. When explicitly enabled, the module can
materialize collapsed UMLS/SNOMED candidate relations directly between existing
Concept nodes. It never creates UMLSConcept nodes and it never adds uniqueness
constraints on Concept.umls_cui.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from neo4j import Driver
else:
    Driver = Any

from knowledge_graph.entity_review_exports import safe_filename_component
from knowledge_graph.relationship_metadata import build_ontology_relationship_metadata
try:
    from knowledge_graph.umls_normalization import (
        DEFAULT_API_RATE_LIMIT_PER_SECOND,
        DEFAULT_API_TIMEOUT,
        DEFAULT_UMLS_VERSION,
        UMLSAPIAuthError,
        UMLSAPIError,
        ensure_requests_dependency_available,
    )
except ModuleNotFoundError as e:
    if e.name not in {"neo4j", "dotenv"}:
        raise

    DEFAULT_API_TIMEOUT = 30.0
    DEFAULT_API_RATE_LIMIT_PER_SECOND = 5.0
    DEFAULT_UMLS_VERSION = "current"

    class UMLSAPIError(RuntimeError):
        pass

    class UMLSAPIAuthError(UMLSAPIError):
        pass

    def ensure_requests_dependency_available() -> None:
        if requests is None:
            raise RuntimeError("requests is required for UMLS API access")

try:
    from knowledge_graph.neo4j_utils import (
        close_driver,
        get_neo4j_driver,
        load_project_dotenv_once,
    )
except ModuleNotFoundError as e:
    if e.name not in {"neo4j", "dotenv"}:
        raise

    _MISSING_NEO4J_UTILITY_DEPENDENCY = e.name
    _DOTENV_FALLBACK_LOADED = False

    def load_project_dotenv_once() -> None:
        global _DOTENV_FALLBACK_LOADED
        if _DOTENV_FALLBACK_LOADED:
            return

        try:
            from dotenv import load_dotenv
        except ModuleNotFoundError:
            _DOTENV_FALLBACK_LOADED = True
            return

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        _DOTENV_FALLBACK_LOADED = True

    def get_neo4j_driver(verify: bool = True) -> Driver:
        raise RuntimeError(
            "neo4j is required for UMLS connection graph access; install project "
            "dependencies before running exports or materialization"
        ) from None

    def close_driver(driver: Driver) -> None:
        close = getattr(driver, "close", None)
        if callable(close):
            close()

try:
    import requests
except ImportError:
    requests = None


logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "umls_test" / "umls_connections"
DEFAULT_RELATION_CACHE_DIR = ROOT_DIR / "umls_test" / "umls_api_cache" / "relations"
DEFAULT_SOURCE_VOCAB = "SNOMEDCT_US"
DEFAULT_RELATION_PAGE_SIZE = 200
DEFAULT_MAX_RELATIONS_PER_CUI = 500
DEFAULT_MAX_SOURCE_UI_LOOKUPS_PER_CUI = 100
DEFAULT_WRITE_PARTIAL_EVERY = 25
RELATIONS_404_NEGATIVE_CACHE_THRESHOLD = 3
MATERIALIZATION_MODE_CHOICES = ("none", "safe_only")
DEFAULT_MATERIALIZATION_MODE = "none"


@dataclass(frozen=True)
class RelationSpec:
    relationship_type: str
    family: str
    validation_mode: str
    source_types: tuple[str, ...]
    target_types: tuple[str, ...]
    broad_source_types: tuple[str, ...]
    broad_target_types: tuple[str, ...]
    default_traversal_policy: str
    materialize_by_default: bool


@dataclass(frozen=True)
class CompatibilityResult:
    status: str
    reason: str


def _type_tuple(values: Sequence[str] = ()) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip().casefold() for value in values if value}))


def _umls_relationship_type(relation_name: str) -> str:
    return "UMLS_" + re.sub(r"[^A-Z0-9]+", "_", relation_name.upper()).strip("_")


def _relation_spec(
    relation_name: str,
    *,
    family: str,
    validation_mode: str = "exact",
    source_types: Sequence[str] = (),
    target_types: Sequence[str] = (),
    broad_source_types: Sequence[str] = (),
    broad_target_types: Sequence[str] = (),
    default_traversal_policy: str = "review",
    materialize_by_default: bool = False,
) -> tuple[str, RelationSpec]:
    normalized_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(relation_name or "").strip().casefold(),
    ).strip("_")
    return (
        normalized_name,
        RelationSpec(
            relationship_type=_umls_relationship_type(normalized_name),
            family=family,
            validation_mode=validation_mode,
            source_types=_type_tuple(source_types),
            target_types=_type_tuple(target_types),
            broad_source_types=_type_tuple(broad_source_types),
            broad_target_types=_type_tuple(broad_target_types),
            default_traversal_policy=default_traversal_policy,
            materialize_by_default=bool(materialize_by_default),
        ),
    )


DISEASE_FINDING_TYPES = _type_tuple(
    {
        "disease",
        "clinical_finding",
        "complication_or_comorbidity",
    }
)
DIAGNOSTIC_FAMILY_TYPES = _type_tuple({"diagnostic_test", "imaging_modality"})
PROCEDURE_TYPES = _type_tuple(
    {
        "procedure_or_intervention",
        "diagnostic_test",
        "imaging_modality",
    }
)
DIRECT_PROCEDURE_TYPES = _type_tuple(
    {
        "procedure_or_intervention",
        "diagnostic_test",
    }
)
ANATOMICAL_TYPES = _type_tuple({"anatomical_structure"})
DEVICE_TYPES = _type_tuple({"device"})
BIOMARKER_TYPES = _type_tuple({"biomarker"})

CORE_SNOMED_RELATION_NAMES = frozenset(
    {
        "isa",
        "inverse_isa",
        "has_finding_site",
        "finding_site_of",
        "has_associated_morphology",
        "associated_morphology_of",
        "has_procedure_site",
        "has_direct_procedure_site",
    }
)

FIRST_SNOMED_EXTENSION_RELATION_NAMES = frozenset(
    {
        "has_definitional_manifestation",
        "definitional_manifestation_of",
        "uses_device",
        "device_used_by",
        "has_direct_device",
        "direct_device_of",
        "has_measured_component",
        "measured_component_of",
    }
)

AUDIT_ONLY_RELATION_NAMES = frozenset(
    {
        "procedure_site_of",
        "direct_procedure_site_of",
        "has_method",
        "method_of",
        "has_causative_agent",
        "causative_agent_of",
        "has_pathological_process",
        "pathological_process_of",
        "has_associated_finding",
        "associated_finding_of",
        "interprets",
        "is_interpreted_by",
        "has_interpretation",
        "interpretation_of",
        "has_clinical_course",
        "clinical_course_of",
        "due_to",
        "cause_of",
    }
)

EXPANDED_SNOMED_RELATION_NAMES = frozenset(
    CORE_SNOMED_RELATION_NAMES
    | FIRST_SNOMED_EXTENSION_RELATION_NAMES
)
BALANCED_CORE_SNOMED_RELATION_NAMES = frozenset(
    CORE_SNOMED_RELATION_NAMES
    | {"procedure_site_of", "direct_procedure_site_of"}
)
AUDIT_ALL_SNOMED_RELATION_NAMES = frozenset(
    EXPANDED_SNOMED_RELATION_NAMES | AUDIT_ONLY_RELATION_NAMES
)

# Backward-compatible alias. "Strong" now means the expanded SNOMED profile.
STRONG_RELATION_NAMES = EXPANDED_SNOMED_RELATION_NAMES

RELATION_SPECS = dict(
    [
        _relation_spec(
            "isa",
            family="core",
            validation_mode="hierarchy",
            default_traversal_policy="hierarchy",
            materialize_by_default=True,
        ),
        _relation_spec(
            "inverse_isa",
            family="core",
            validation_mode="hierarchy",
            default_traversal_policy="hierarchy",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_finding_site",
            family="core",
            source_types=DISEASE_FINDING_TYPES,
            target_types=ANATOMICAL_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "finding_site_of",
            family="core",
            source_types=ANATOMICAL_TYPES,
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_associated_morphology",
            family="core",
            source_types=DISEASE_FINDING_TYPES,
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "associated_morphology_of",
            family="core",
            source_types=DISEASE_FINDING_TYPES,
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_procedure_site",
            family="core",
            source_types=PROCEDURE_TYPES,
            target_types=ANATOMICAL_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_direct_procedure_site",
            family="core",
            source_types=DIRECT_PROCEDURE_TYPES,
            target_types=ANATOMICAL_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_definitional_manifestation",
            family="first_extension",
            source_types={"disease", "complication_or_comorbidity"},
            target_types={"clinical_finding"},
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "definitional_manifestation_of",
            family="first_extension",
            source_types={"clinical_finding"},
            target_types={"disease", "complication_or_comorbidity"},
            default_traversal_policy="reverse_review",
            materialize_by_default=True,
        ),
        _relation_spec(
            "uses_device",
            family="first_extension",
            source_types=PROCEDURE_TYPES,
            target_types=DEVICE_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "device_used_by",
            family="first_extension",
            source_types=DEVICE_TYPES,
            target_types=PROCEDURE_TYPES,
            default_traversal_policy="reverse_review",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_direct_device",
            family="first_extension",
            source_types=DIRECT_PROCEDURE_TYPES,
            target_types=DEVICE_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "direct_device_of",
            family="first_extension",
            source_types=DEVICE_TYPES,
            target_types=DIRECT_PROCEDURE_TYPES,
            default_traversal_policy="reverse_review",
            materialize_by_default=True,
        ),
        _relation_spec(
            "has_measured_component",
            family="first_extension",
            source_types={"diagnostic_test"},
            target_types=BIOMARKER_TYPES,
            default_traversal_policy="safe",
            materialize_by_default=True,
        ),
        _relation_spec(
            "measured_component_of",
            family="first_extension",
            source_types=BIOMARKER_TYPES,
            target_types={"diagnostic_test"},
            default_traversal_policy="reverse_review",
            materialize_by_default=True,
        ),
        _relation_spec(
            "procedure_site_of",
            family="audit_candidate",
            source_types=ANATOMICAL_TYPES,
            target_types=PROCEDURE_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "direct_procedure_site_of",
            family="audit_candidate",
            source_types=ANATOMICAL_TYPES,
            target_types=DIRECT_PROCEDURE_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "has_method",
            family="audit_candidate",
            source_types=PROCEDURE_TYPES,
            target_types={"care_strategy", "procedure_or_intervention"},
            default_traversal_policy="review",
        ),
        _relation_spec(
            "method_of",
            family="audit_candidate",
            source_types={"care_strategy", "procedure_or_intervention"},
            target_types=PROCEDURE_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "has_causative_agent",
            family="audit_candidate",
            source_types=DISEASE_FINDING_TYPES,
            target_types={"drug_or_drug_class", "genetic_factor", "risk_factor"},
            broad_target_types={"biomarker", "drug_or_drug_class", "genetic_factor", "risk_factor"},
            default_traversal_policy="review",
        ),
        _relation_spec(
            "causative_agent_of",
            family="audit_candidate",
            source_types={"drug_or_drug_class", "genetic_factor", "risk_factor"},
            target_types=DISEASE_FINDING_TYPES,
            broad_source_types={"biomarker", "drug_or_drug_class", "genetic_factor", "risk_factor"},
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "has_pathological_process",
            family="audit_candidate",
            source_types=DISEASE_FINDING_TYPES,
            target_types={"clinical_finding"},
            broad_target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="review",
        ),
        _relation_spec(
            "pathological_process_of",
            family="audit_candidate",
            source_types={"clinical_finding"},
            target_types=DISEASE_FINDING_TYPES,
            broad_source_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "has_associated_finding",
            family="audit_candidate",
            source_types=DISEASE_FINDING_TYPES,
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="review",
        ),
        _relation_spec(
            "associated_finding_of",
            family="audit_candidate",
            source_types=DISEASE_FINDING_TYPES,
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "interprets",
            family="audit_candidate",
            source_types={"diagnostic_test", "imaging_modality", "score_or_risk_model"},
            target_types={"biomarker", "clinical_finding"},
            default_traversal_policy="review",
        ),
        _relation_spec(
            "is_interpreted_by",
            family="audit_candidate",
            source_types={"biomarker", "clinical_finding"},
            target_types={"diagnostic_test", "imaging_modality", "score_or_risk_model"},
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "has_interpretation",
            family="audit_candidate",
            source_types={"biomarker", "diagnostic_test", "imaging_modality"},
            target_types={"clinical_finding"},
            broad_target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="review",
        ),
        _relation_spec(
            "interpretation_of",
            family="audit_candidate",
            source_types={"clinical_finding"},
            target_types={"biomarker", "diagnostic_test", "imaging_modality"},
            broad_source_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "has_clinical_course",
            family="audit_candidate",
            source_types=DISEASE_FINDING_TYPES,
            target_types={"clinical_outcome"},
            default_traversal_policy="review",
        ),
        _relation_spec(
            "clinical_course_of",
            family="audit_candidate",
            source_types={"clinical_outcome"},
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
        ),
        _relation_spec(
            "due_to",
            family="audit_candidate",
            source_types=DISEASE_FINDING_TYPES,
            target_types={
                "disease",
                "clinical_finding",
                "complication_or_comorbidity",
                "drug_or_drug_class",
                "genetic_factor",
                "risk_factor",
            },
            default_traversal_policy="review",
        ),
        _relation_spec(
            "cause_of",
            family="audit_candidate",
            source_types={
                "disease",
                "clinical_finding",
                "complication_or_comorbidity",
                "drug_or_drug_class",
                "genetic_factor",
                "risk_factor",
            },
            target_types=DISEASE_FINDING_TYPES,
            default_traversal_policy="reverse_review",
        ),
    ]
)

RELATION_NAMES_BY_PROFILE = {
    "core": CORE_SNOMED_RELATION_NAMES,
    "first_extension": FIRST_SNOMED_EXTENSION_RELATION_NAMES,
    "expanded": EXPANDED_SNOMED_RELATION_NAMES,
    "balanced_core": BALANCED_CORE_SNOMED_RELATION_NAMES,
    "audit_all": AUDIT_ALL_SNOMED_RELATION_NAMES,
}
RELATION_PROFILE_CHOICES = tuple(RELATION_NAMES_BY_PROFILE)

RELATION_TYPE_BY_NAME = {
    relation_name: spec.relationship_type
    for relation_name, spec in sorted(RELATION_SPECS.items())
}
UMLS_CONNECTION_RELATION_TYPES = sorted(set(RELATION_TYPE_BY_NAME.values()))

SAFE_TRAVERSAL_RELATION_NAMES = frozenset(
    relation_name
    for relation_name, spec in RELATION_SPECS.items()
    if spec.default_traversal_policy == "safe"
)
HIERARCHY_RELATION_NAMES = frozenset(
    relation_name
    for relation_name, spec in RELATION_SPECS.items()
    if spec.validation_mode == "hierarchy"
)
REVERSE_REVIEW_RELATION_NAMES = frozenset(
    relation_name
    for relation_name, spec in RELATION_SPECS.items()
    if spec.default_traversal_policy == "reverse_review"
)
LOCAL_TYPE_RULES_BY_RELATION_NAME = {
    relation_name: {
        "source_types": set(spec.source_types),
        "target_types": set(spec.target_types),
    }
    for relation_name, spec in RELATION_SPECS.items()
    if spec.validation_mode == "exact" and spec.source_types and spec.target_types
}
FIRST_EXTENSION_SAFE_RELATION_NAMES = frozenset(
    relation_name
    for relation_name in FIRST_SNOMED_EXTENSION_RELATION_NAMES
    if RELATION_SPECS[relation_name].default_traversal_policy == "safe"
)
FIRST_EXTENSION_REVERSE_REVIEW_RELATION_NAMES = frozenset(
    relation_name
    for relation_name in FIRST_SNOMED_EXTENSION_RELATION_NAMES
    if RELATION_SPECS[relation_name].default_traversal_policy == "reverse_review"
)

CONNECTION_STATUS = "candidate"
CUI_PATTERN = re.compile(r"\bC\d{7}\b", re.IGNORECASE)

CSV_COLUMNS = [
    "doc_id",
    "source_concept_id",
    "source_name",
    "source_type",
    "source_cui",
    "target_concept_id",
    "target_name",
    "target_type",
    "target_cui",
    "umls_relation_label",
    "umls_additional_relation_label",
    "umls_relation_source",
    "umls_relation_ui",
    "source_vocabulary",
    "relation_raw_id",
    "connection_status",
    "source_umls_canonical_name",
    "source_umls_semantic_types",
    "source_umls_score",
    "source_observed_types",
    "source_type_support_pairs",
    "source_type_resolution_status",
    "target_umls_canonical_name",
    "target_umls_semantic_types",
    "target_umls_score",
    "target_observed_types",
    "target_type_support_pairs",
    "target_type_resolution_status",
    "relation_raw_related_id",
    "relation_raw_related_id_name",
    "relation_raw_related_from_id",
    "relation_raw_related_from_id_name",
]

EQUIVALENCE_RELATION_LABELS = {
    "SAME_AS",
    "SY",
    "SYNONYM",
    "EQUIVALENT_TO",
}

EQUIVALENCE_ADDITIONAL_RELATION_LABELS = {
    "same_as",
    "synonym",
    "source_asserted_synonymy",
    "possibly_equivalent_to",
    "mapped_to",
    "mapped_from",
}


@dataclass(frozen=True)
class LocalConcept:
    concept_id: str
    name: str
    canonical_type: str
    umls_cui: str
    umls_canonical_name: str
    umls_semantic_types: tuple[str, ...]
    umls_score: Optional[float]
    observed_types: tuple[str, ...]
    type_support_pairs: tuple[str, ...]
    type_resolution_status: str
    needs_type_review: bool


@dataclass(frozen=True)
class SourceIdentifier:
    root_source: str
    source_ui: str


@dataclass(frozen=True)
class RelationFetchResult:
    records: list[dict[str, Any]]
    fetched_records: int
    skipped_by_limit: int = 0
    truncated_by_limit: bool = False
    page_count: Optional[int] = None
    status: str = "processed"
    http_status: Optional[int] = None
    failure_count: int = 0
    from_negative_cache: bool = False


class UMLSAPIHTTPStatusError(UMLSAPIError):
    def __init__(self, status_code: int) -> None:
        self.status_code = int(status_code)
        super().__init__(f"UMLS API request failed: HTTP {self.status_code}")


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, (list, tuple, set)):
        return tuple(
            clean_text(item)
            for item in value
            if clean_text(item)
        )

    text = clean_text(value)
    return (text,) if text else ()


def serialize_string_tuple(value: Sequence[str]) -> str:
    return "; ".join(clean_text(item) for item in value if clean_text(item))


def normalize_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_raw_relation_value(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        text = clean_text(value)
        if text:
            return text
    return ""


def normalize_cui(value: Any) -> str:
    return clean_text(value).upper()


def normalize_relation_term(value: Any) -> str:
    text = clean_text(value).casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_relation_name_set(values: Optional[Sequence[str]]) -> set[str]:
    return {
        normalized
        for normalized in (
            normalize_relation_term(value)
            for value in (values or [])
        )
        if normalized
    }


def normalize_local_type(value: Any) -> str:
    return clean_text(value).casefold()


def local_type_compatible_from_status(status: str) -> bool:
    return clean_text(status) in {"compatible", "compatible_broad"}


def _missing_type_result(source_type: str, target_type: str) -> Optional[CompatibilityResult]:
    if not source_type and not target_type:
        return CompatibilityResult("missing_type", "missing_source_and_target_type")
    if not source_type:
        return CompatibilityResult("missing_type", "missing_source_type")
    if not target_type:
        return CompatibilityResult("missing_type", "missing_target_type")
    return None


def evaluate_hierarchy_compatibility(
    source_type: str,
    target_type: str,
) -> CompatibilityResult:
    normalized_source_type = normalize_local_type(source_type)
    normalized_target_type = normalize_local_type(target_type)
    missing = _missing_type_result(normalized_source_type, normalized_target_type)
    if missing is not None:
        return missing

    if normalized_source_type == normalized_target_type:
        return CompatibilityResult("compatible", "same_canonical_type")

    hierarchy_broad_families = (
        set(DISEASE_FINDING_TYPES),
        set(DIAGNOSTIC_FAMILY_TYPES),
    )
    for broad_family in hierarchy_broad_families:
        if (
            normalized_source_type in broad_family
            and normalized_target_type in broad_family
        ):
            return CompatibilityResult("compatible_broad", "broad_hierarchy_family")

    return CompatibilityResult("review", "cross_type_hierarchy_requires_review")


def evaluate_relation_compatibility(
    relation_name: str,
    source_type: str,
    target_type: str,
) -> CompatibilityResult:
    normalized_relation_name = normalize_relation_term(relation_name)
    spec = RELATION_SPECS.get(normalized_relation_name)
    if spec is None:
        return CompatibilityResult("unsupported_local_schema", "relation_not_in_catalog")

    normalized_source_type = normalize_local_type(source_type)
    normalized_target_type = normalize_local_type(target_type)
    missing = _missing_type_result(normalized_source_type, normalized_target_type)
    if missing is not None:
        return missing

    if spec.validation_mode == "hierarchy":
        return evaluate_hierarchy_compatibility(
            normalized_source_type,
            normalized_target_type,
        )

    if spec.validation_mode == "review":
        return CompatibilityResult("review", "requires_relation_review")

    if spec.validation_mode != "exact" or not spec.source_types or not spec.target_types:
        return CompatibilityResult(
            "unsupported_local_schema",
            "no_exact_type_rule",
        )

    source_ok = normalized_source_type in spec.source_types
    target_ok = normalized_target_type in spec.target_types
    if source_ok and target_ok:
        return CompatibilityResult("compatible", "compatible")

    broad_source_types = set(spec.broad_source_types or spec.source_types)
    broad_target_types = set(spec.broad_target_types or spec.target_types)
    source_broad_ok = normalized_source_type in broad_source_types
    target_broad_ok = normalized_target_type in broad_target_types
    if source_broad_ok and target_broad_ok:
        return CompatibilityResult("compatible_broad", "broad_type_match")

    if not source_ok and not target_ok:
        reason = "source_and_target_type_not_allowed"
    elif not source_ok:
        reason = "source_type_not_allowed"
    else:
        reason = "target_type_not_allowed"
    return CompatibilityResult("incompatible", reason)


def resolve_selected_relation_profile(
    relation_profile: Optional[str] = None,
    strong_relations_only: bool = False,
) -> Optional[str]:
    selected_profile = clean_text(relation_profile).casefold()
    if not selected_profile:
        selected_profile = None

    if selected_profile is not None and selected_profile not in RELATION_NAMES_BY_PROFILE:
        raise ValueError(
            f"Invalid relation profile {relation_profile!r}; "
            f"choose one of {', '.join(RELATION_PROFILE_CHOICES)}"
        )

    if (
        strong_relations_only
        and selected_profile is not None
        and selected_profile != "expanded"
    ):
        raise ValueError(
            "--strong-relations-only is equivalent to --relation-profile "
            f"expanded and conflicts with --relation-profile {selected_profile}"
        )

    if strong_relations_only:
        return "expanded"
    return selected_profile


def evaluate_local_type_compatibility(
    relation_name: str,
    source_type: str,
    target_type: str,
) -> tuple[bool | None, str]:
    result = evaluate_relation_compatibility(
        relation_name=relation_name,
        source_type=source_type,
        target_type=target_type,
    )
    if result.status == "unsupported_local_schema":
        return None, result.reason
    return local_type_compatible_from_status(result.status), result.reason


def is_short_acronym_like_name(value: Any) -> bool:
    """
    Heuristic used only as a representative-selection tie-breaker.

    Short all-caps names such as "AF" or "CHF" are useful aliases, but a more
    descriptive matched Concept name is a better single representative for a CUI.
    """
    text = clean_text(value)
    if not text:
        return False

    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if not compact or len(compact) > 6:
        return False
    if any(char.isspace() for char in text):
        return False

    letters = [char for char in compact if char.isalpha()]
    if not letters:
        return False

    return len(compact) <= 3 or all(char.isupper() for char in letters)


def descriptive_name_score(value: Any) -> tuple[int, int]:
    text = clean_text(value)
    words = re.findall(r"[A-Za-z0-9]+", text)
    return (len(words), len(text))


def representative_sort_key(concept: LocalConcept) -> tuple[Any, ...]:
    score = concept.umls_score if concept.umls_score is not None else float("-inf")
    word_count, name_length = descriptive_name_score(concept.name)
    return (
        -score,
        is_short_acronym_like_name(concept.name),
        -word_count,
        -name_length,
        concept.concept_id,
    )


def select_representative_concepts(
    concepts: Sequence[LocalConcept],
) -> dict[str, LocalConcept]:
    """
    Choose exactly one deterministic local Concept representative per CUI.

    The query that produced ``concepts`` already enforces the eligibility rules:
    accepted UMLS match, non-empty CUI, non-ambiguous canonical type,
    needs_type_review=false, and optional doc_id mention filtering.
    """
    by_cui = concepts_by_cui(concepts)
    return {
        cui: sorted(grouped_concepts, key=representative_sort_key)[0]
        for cui, grouped_concepts in by_cui.items()
        if grouped_concepts
    }


def concept_report(concept: Optional[LocalConcept]) -> Optional[dict[str, Any]]:
    if concept is None:
        return None
    return {
        "concept_id": concept.concept_id,
        "name": concept.name,
        "canonical_type": concept.canonical_type,
        "umls_cui": concept.umls_cui,
        "umls_score": concept.umls_score,
        "is_acronym_like": is_short_acronym_like_name(concept.name),
    }


def resolve_relation_name_filters(
    include_relation_names: Optional[Sequence[str]] = None,
    exclude_relation_names: Optional[Sequence[str]] = None,
    strong_relations_only: bool = False,
    relation_profile: Optional[str] = None,
) -> tuple[set[str], set[str], Optional[str]]:
    selected_profile = resolve_selected_relation_profile(
        relation_profile=relation_profile,
        strong_relations_only=strong_relations_only,
    )
    include_names = (
        set(RELATION_NAMES_BY_PROFILE[selected_profile])
        if selected_profile is not None
        else set()
    )
    include_names.update(normalize_relation_name_set(include_relation_names))
    exclude_names = normalize_relation_name_set(exclude_relation_names)

    include_names.difference_update(exclude_names)

    return include_names, exclude_names, selected_profile


def cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def relation_record_id(record: dict[str, Any]) -> str:
    raw_ui = clean_text(
        record.get("ui")
        or record.get("relationUi")
        or record.get("relationUI")
        or record.get("id")
    )
    if raw_ui:
        return raw_ui

    raw = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_cui(value: Any) -> Optional[str]:
    match = CUI_PATTERN.search(clean_text(value))
    if match is None:
        return None
    return match.group(0).upper()


def parse_source_identifier(value: Any) -> Optional[SourceIdentifier]:
    text = clean_text(value)
    if not text:
        return None

    parsed = urlparse(text)
    path = parsed.path if parsed.scheme else text
    parts = [unquote(part) for part in path.split("/") if part]

    try:
        source_index = parts.index("source")
    except ValueError:
        return None

    if len(parts) <= source_index + 2:
        return None

    root_source = clean_text(parts[source_index + 1])
    source_ui = clean_text("/".join(parts[source_index + 2:]))
    if not root_source or not source_ui:
        return None

    return SourceIdentifier(root_source=root_source, source_ui=source_ui)


def is_equivalence_relation(
    relation_label: str,
    additional_relation_label: str,
) -> bool:
    if relation_label.upper() in EQUIVALENCE_RELATION_LABELS:
        return True

    normalized_additional = normalize_relation_term(additional_relation_label)
    return normalized_additional in EQUIVALENCE_ADDITIONAL_RELATION_LABELS


def fetch_local_concepts_for_doc(tx, doc_id: Optional[str]) -> list[LocalConcept]:
    """
    Fetch only existing local concepts that already have accepted UMLS matches.

    This is intentionally a read-only query. It ignores low-confidence UMLS
    candidates and review-only duplicate evidence.
    """
    result = tx.run(
        """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
          AND c.normalization_status = 'umls_matched'
          AND c.umls_cui IS NOT NULL
          AND trim(toString(c.umls_cui)) <> ''
          AND coalesce(c.canonical_type, '') <> 'ambiguous'
          AND coalesce(c.needs_type_review, false) = false
        WITH c
        RETURN DISTINCT
            elementId(c) AS concept_id,
            c.name AS name,
            coalesce(c.canonical_type, '') AS canonical_type,
            c.umls_cui AS umls_cui,
            c.umls_canonical_name AS umls_canonical_name,
            c.umls_semantic_types AS umls_semantic_types,
            c.umls_score AS umls_score,
            c.observed_types AS observed_types,
            c.type_support_pairs AS type_support_pairs,
            c.type_resolution_status AS type_resolution_status,
            coalesce(c.needs_type_review, false) AS needs_type_review
        ORDER BY c.name, concept_id
        """,
        doc_id=doc_id,
    )

    concepts: list[LocalConcept] = []
    for row in result:
        cui = normalize_cui(row["umls_cui"])
        if not cui:
            continue

        concepts.append(
            LocalConcept(
                concept_id=clean_text(row["concept_id"]),
                name=clean_text(row["name"]),
                canonical_type=clean_text(row["canonical_type"]),
                umls_cui=cui,
                umls_canonical_name=clean_text(row["umls_canonical_name"]),
                umls_semantic_types=normalize_string_tuple(row["umls_semantic_types"]),
                umls_score=normalize_optional_float(row["umls_score"]),
                observed_types=normalize_string_tuple(row["observed_types"]),
                type_support_pairs=normalize_string_tuple(row["type_support_pairs"]),
                type_resolution_status=clean_text(row["type_resolution_status"]),
                needs_type_review=bool(row["needs_type_review"]),
            )
        )

    return concepts


class UMLSRelationsClient:
    """
    Conservative UMLS REST client for relation retrieval with local JSON cache.
    """

    base_url = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(
        self,
        cache_dir: Path,
        timeout: float = DEFAULT_API_TIMEOUT,
        rate_limit_per_second: float = DEFAULT_API_RATE_LIMIT_PER_SECOND,
        version: str = DEFAULT_UMLS_VERSION,
        page_size: int = DEFAULT_RELATION_PAGE_SIZE,
        ignore_negative_cache: bool = False,
        session: Optional[Any] = None,
    ) -> None:
        ensure_requests_dependency_available()
        if requests is None:
            raise RuntimeError("requests is required for UMLS relation retrieval")

        load_project_dotenv_once()
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
        self.ignore_negative_cache = bool(ignore_negative_cache)
        self.session = session if session is not None else requests.Session()
        self._last_request_at = 0.0
        self._source_ui_cui_cache: dict[tuple[str, str], list[str]] = {}
        self.stats: dict[str, int] = {
            "api_cache_hits": 0,
            "api_cache_misses": 0,
            "api_requests": 0,
            "api_retries": 0,
            "api_errors": 0,
            "relation_negative_cache_hits": 0,
            "relation_negative_cache_writes": 0,
        }

    def cache_path(self, namespace: str, payload: dict[str, Any]) -> Path:
        return self.cache_dir / namespace / f"{cache_key(payload)}.json"

    def get_cached_payload(
        self,
        namespace: str,
        payload: dict[str, Any],
        count_hit: bool = True,
    ) -> Optional[dict[str, Any]]:
        path = self.cache_path(namespace, payload)
        if not path.exists():
            return None

        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if count_hit:
            self.stats["api_cache_hits"] += 1
        return cached if isinstance(cached, dict) else None

    def write_cached_payload(
        self,
        namespace: str,
        payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> None:
        path = self.cache_path(namespace, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(response_payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def throttle(self) -> None:
        if self.rate_limit_per_second <= 0:
            return

        min_interval = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def request_uncached(
        self,
        url: str,
        params: dict[str, Any],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        request_params = dict(params)
        request_params["apiKey"] = self.api_key

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            self.throttle()
            self._last_request_at = time.monotonic()
            self.stats["api_requests"] += 1

            try:
                response = self.session.get(
                    url,
                    params=request_params,
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
                    last_error = UMLSAPIError(
                        f"UMLS API temporary failure: HTTP {status_code}"
                    )
                    self.stats["api_errors"] += 1
                elif status_code >= 400:
                    self.stats["api_errors"] += 1
                    raise UMLSAPIHTTPStatusError(status_code)
                else:
                    try:
                        payload = response.json()
                    except Exception as e:
                        self.stats["api_errors"] += 1
                        raise UMLSAPIError("Malformed UMLS API response") from e

                    if not isinstance(payload, dict):
                        self.stats["api_errors"] += 1
                        raise UMLSAPIError("Malformed UMLS API response")

                    return payload

            if attempt < max_retries:
                self.stats["api_retries"] += 1
                time.sleep(min(2**attempt, 4))

        if last_error is not None:
            raise UMLSAPIError(str(last_error) or "UMLS API request failed")
        raise UMLSAPIError("UMLS API request failed")

    def relation_page_params(
        self,
        cui: str,
        source_vocab: str,
        page_number: int,
        page_size: Optional[int] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "pageNumber": page_number,
            "pageSize": page_size or self.page_size,
            "includeObsolete": "false",
            "includeSuppressible": "false",
        }
        if source_vocab:
            params["sabs"] = source_vocab
        return params

    def relation_negative_cache_key(
        self,
        cui: str,
        source_vocab: str,
    ) -> dict[str, Any]:
        return {
            "endpoint": "relations",
            "cui": normalize_cui(cui),
            "source_vocab": clean_text(source_vocab),
            "version": self.version,
            "http_status": 404,
        }

    def relation_result_from_negative_cache(
        self,
        cached: dict[str, Any],
        from_negative_cache: bool,
    ) -> RelationFetchResult:
        records = cached.get("records")
        records = records if isinstance(records, list) else []
        return RelationFetchResult(
            records=records,
            fetched_records=int_or_zero(cached.get("fetched_records")),
            skipped_by_limit=int_or_zero(cached.get("skipped_by_limit")),
            truncated_by_limit=bool(cached.get("truncated_by_limit")),
            page_count=cached.get("page_count"),
            status=clean_text(cached.get("status")) or "relations_unavailable",
            http_status=int_or_zero(cached.get("http_status")) or None,
            failure_count=int_or_zero(cached.get("failure_count")),
            from_negative_cache=from_negative_cache,
        )

    def get_unavailable_relations_cache(
        self,
        cui: str,
        source_vocab: str,
        count_hit: bool = True,
    ) -> Optional[dict[str, Any]]:
        cached = self.get_cached_payload(
            "relations_negative",
            self.relation_negative_cache_key(cui, source_vocab),
            count_hit=False,
        )
        if cached is None:
            return None

        if (
            clean_text(cached.get("status")) == "relations_unavailable"
            and int_or_zero(cached.get("http_status")) == 404
        ):
            if count_hit:
                self.stats["api_cache_hits"] += 1
                self.stats["relation_negative_cache_hits"] += 1
            return cached

        return None

    def record_relations_404(
        self,
        cui: str,
        source_vocab: str,
    ) -> dict[str, Any]:
        cache_key_payload = self.relation_negative_cache_key(cui, source_vocab)
        previous = self.get_cached_payload(
            "relations_negative",
            cache_key_payload,
            count_hit=False,
        )
        previous_failure_count = (
            int_or_zero(previous.get("failure_count"))
            if isinstance(previous, dict)
            else 0
        )
        failure_count = previous_failure_count + 1
        status = (
            "relations_unavailable"
            if failure_count >= RELATIONS_404_NEGATIVE_CACHE_THRESHOLD
            else "relations_404_retryable"
        )
        negative_payload = {
            "endpoint": "relations",
            "status": status,
            "cui": normalize_cui(cui),
            "source_vocab": clean_text(source_vocab),
            "version": self.version,
            "http_status": 404,
            "failure_count": failure_count,
            "records": [],
            "record_count": 0,
            "fetched_records": 0,
            "skipped_by_limit": 0,
            "truncated_by_limit": False,
            "last_attempted_at": utc_now_iso(),
        }
        self.write_cached_payload(
            "relations_negative",
            cache_key_payload,
            negative_payload,
        )
        self.stats["relation_negative_cache_writes"] += 1
        return negative_payload

    def clear_relations_negative_cache(
        self,
        cui: str,
        source_vocab: str,
    ) -> None:
        path = self.cache_path(
            "relations_negative",
            self.relation_negative_cache_key(cui, source_vocab),
        )
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def get_relations(
        self,
        cui: str,
        source_vocab: str,
        max_records: Optional[int] = DEFAULT_MAX_RELATIONS_PER_CUI,
    ) -> RelationFetchResult:
        cui = normalize_cui(cui)
        source_vocab = clean_text(source_vocab)
        max_records = int(max_records) if max_records is not None else None
        if max_records is not None and max_records < 1:
            max_records = None
        effective_page_size = (
            self.page_size
            if max_records is None
            else max(1, min(self.page_size, max_records + 1))
        )
        cache_payload = {
            "endpoint": "relations",
            "cui": cui,
            "source_vocab": source_vocab,
            "version": self.version,
            "page_size": effective_page_size,
            "max_records": max_records,
            "include_obsolete": False,
            "include_suppressible": False,
        }

        if not self.ignore_negative_cache:
            unavailable_cache = self.get_unavailable_relations_cache(
                cui=cui,
                source_vocab=source_vocab,
            )
            if unavailable_cache is not None:
                return self.relation_result_from_negative_cache(
                    unavailable_cache,
                    from_negative_cache=True,
                )

        cached = self.get_cached_payload("relations", cache_payload)
        if cached is not None:
            records = cached.get("records")
            records = records if isinstance(records, list) else []
            skipped_by_limit = int_or_zero(cached.get("skipped_by_limit"))
            return RelationFetchResult(
                records=records,
                fetched_records=(
                    int_or_zero(cached.get("fetched_records")) or len(records)
                ),
                skipped_by_limit=skipped_by_limit,
                truncated_by_limit=bool(cached.get("truncated_by_limit")),
                page_count=cached.get("page_count"),
            )

        self.stats["api_cache_misses"] += 1
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}/relations"

        records: list[dict[str, Any]] = []
        page_number = 1
        page_count: Optional[int] = None

        while True:
            try:
                payload = self.request_uncached(
                    url=url,
                    params=self.relation_page_params(
                        cui=cui,
                        source_vocab=source_vocab,
                        page_number=page_number,
                        page_size=effective_page_size,
                    ),
                )
            except UMLSAPIHTTPStatusError as e:
                if e.status_code != 404:
                    raise

                negative_payload = self.record_relations_404(
                    cui=cui,
                    source_vocab=source_vocab,
                )
                if clean_text(negative_payload.get("status")) == "relations_unavailable":
                    logger.info(
                        "UMLS relations marked unavailable after repeated HTTP 404 | cui=%s | source_vocab=%s | failure_count=%d",
                        cui,
                        source_vocab or "ALL",
                        int_or_zero(negative_payload.get("failure_count")),
                    )
                    return self.relation_result_from_negative_cache(
                        negative_payload,
                        from_negative_cache=False,
                    )
                raise

            records.extend(parse_relation_items(payload))

            if page_count is None:
                page_count = parse_page_count(payload)
            if max_records is not None and len(records) > max_records:
                break
            if page_count is None or page_number >= page_count:
                break

            page_number += 1

        fetched_records = len(records)
        truncated_by_limit = max_records is not None and fetched_records > max_records
        skipped_by_limit = 0
        if truncated_by_limit and max_records is not None:
            skipped_by_limit = fetched_records - max_records
            if page_count is not None and page_number < page_count:
                skipped_by_limit += (page_count - page_number) * effective_page_size
            records = records[:max_records]

        response_payload = {
            "records": records,
            "record_count": len(records),
            "fetched_records": fetched_records,
            "skipped_by_limit": skipped_by_limit,
            "truncated_by_limit": truncated_by_limit,
            "page_count": page_count,
        }
        self.clear_relations_negative_cache(cui=cui, source_vocab=source_vocab)
        self.write_cached_payload("relations", cache_payload, response_payload)
        return RelationFetchResult(
            records=records,
            fetched_records=fetched_records,
            skipped_by_limit=skipped_by_limit,
            truncated_by_limit=truncated_by_limit,
            page_count=page_count,
        )

    def lookup_cuis_for_source_ui(
        self,
        root_source: str,
        source_ui: str,
    ) -> list[str]:
        root_source = clean_text(root_source)
        source_ui = clean_text(source_ui)
        memory_key = (root_source, source_ui)

        if memory_key in self._source_ui_cui_cache:
            return self._source_ui_cui_cache[memory_key]

        cache_payload = {
            "endpoint": "source_ui_to_cui",
            "root_source": root_source,
            "source_ui": source_ui,
            "version": self.version,
            "return_id_type": "concept",
        }

        cached = self.get_cached_payload("source_ui_to_cui", cache_payload)
        if cached is not None:
            cuis = normalize_cui_list(cached.get("cuis"))
            self._source_ui_cui_cache[memory_key] = cuis
            return cuis

        self.stats["api_cache_misses"] += 1
        url = f"{self.base_url}/search/{self.version}"
        params = {
            "string": source_ui,
            "inputType": "sourceUi",
            "searchType": "exact",
            "returnIdType": "concept",
            "pageSize": 25,
        }
        if root_source:
            params["sabs"] = root_source

        payload = self.request_uncached(url=url, params=params)
        cuis = parse_search_cuis(payload)

        self.write_cached_payload(
            "source_ui_to_cui",
            cache_payload,
            {
                "cuis": cuis,
                "result": payload.get("result"),
            },
        )
        self._source_ui_cui_cache[memory_key] = cuis
        return cuis


def parse_page_count(payload: dict[str, Any]) -> Optional[int]:
    raw_page_count = payload.get("pageCount")
    if raw_page_count is None and isinstance(payload.get("result"), dict):
        raw_page_count = payload["result"].get("pageCount")

    try:
        page_count = int(raw_page_count)
    except (TypeError, ValueError):
        return None

    return page_count if page_count >= 1 else None


def parse_relation_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]

    if isinstance(result, dict):
        for key in ("results", "relations", "items"):
            items = result.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]

    raise UMLSAPIError("Malformed UMLS relations response")


def parse_search_cuis(payload: dict[str, Any]) -> list[str]:
    result = payload.get("result")
    results = result.get("results") if isinstance(result, dict) else None
    if not isinstance(results, list):
        raise UMLSAPIError("Malformed UMLS source-ui search response")

    cuis: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        cui = extract_cui(item.get("ui")) or extract_cui(item.get("uri"))
        if cui and cui not in cuis:
            cuis.append(cui)

    return cuis


def normalize_cui_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    cuis: list[str] = []
    for item in value:
        cui = normalize_cui(item)
        if cui and CUI_PATTERN.fullmatch(cui) and cui not in cuis:
            cuis.append(cui)

    return cuis


def resolve_related_cuis(
    client: UMLSRelationsClient,
    record: dict[str, Any],
    source_vocab: str,
) -> tuple[list[str], Optional[str]]:
    related_id = clean_text(record.get("relatedId") or record.get("relatedID"))
    if not related_id:
        return [], "missing_related_id"

    direct_cui = extract_cui(related_id)
    if direct_cui:
        return [direct_cui], None

    source_identifier = parse_source_identifier(related_id)
    if source_identifier is None:
        return [], "unresolved_related_id"

    root_source = source_identifier.root_source or clean_text(record.get("rootSource"))
    root_source = root_source or clean_text(source_vocab)
    try:
        cuis = client.lookup_cuis_for_source_ui(
            root_source=root_source,
            source_ui=source_identifier.source_ui,
        )
    except UMLSAPIAuthError:
        raise
    except UMLSAPIError as e:
        return [], f"source_ui_lookup_failed:{e}"

    if not cuis:
        return [], "source_ui_lookup_no_cui"
    return cuis, None


def concepts_by_cui(
    concepts: Sequence[LocalConcept],
) -> dict[str, list[LocalConcept]]:
    grouped: dict[str, list[LocalConcept]] = defaultdict(list)
    for concept in concepts:
        grouped[concept.umls_cui].append(concept)

    for grouped_concepts in grouped.values():
        grouped_concepts.sort(key=lambda item: (item.name.casefold(), item.concept_id))

    return dict(grouped)


def relation_requires_source_ui_lookup(record: dict[str, Any]) -> bool:
    related_id = clean_text(record.get("relatedId") or record.get("relatedID"))
    if not related_id or extract_cui(related_id):
        return False
    return parse_source_identifier(related_id) is not None


def build_initial_relation_stats(
    local_cuis: set[str],
    source_cuis: Sequence[str],
    skipped_cuis: Sequence[dict[str, Any]],
    max_cuis: Optional[int],
    include_cuis: Sequence[str],
    skip_cuis: Sequence[str],
    max_relations_per_cui: int,
    max_source_ui_lookups_per_cui: int,
    write_partial_every: int,
    include_relation_names: set[str],
    exclude_relation_names: set[str],
    strong_relations_only: bool,
    relation_profile: Optional[str],
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    return {
        "total_local_cuis": len(local_cuis),
        "eligible_local_cuis": sorted(local_cuis),
        "eligible_local_cui_count": len(local_cuis),
        "source_cuis_selected": len(source_cuis),
        "selected_source_cuis": list(source_cuis),
        "include_cuis": list(include_cuis),
        "skip_cuis": list(skip_cuis),
        "processed_cuis": [],
        "skipped_cuis": list(skipped_cuis),
        "max_cuis": max_cuis,
        "max_relations_per_cui": max_relations_per_cui,
        "max_source_ui_lookups_per_cui": max_source_ui_lookups_per_cui,
        "write_partial_every": write_partial_every,
        "include_relation_names": sorted(include_relation_names),
        "exclude_relation_names": sorted(exclude_relation_names),
        "strong_relations_only": strong_relations_only,
        "relation_profile": relation_profile,
        "selected_relation_profile": relation_profile,
        "materialization_mode": materialization_mode,
        "partial_exports_written": 0,
        "last_partial_export_processed_cuis": 0,
        "final_export_written": False,
        "umls_relation_records_fetched": 0,
        "umls_relation_records_processed": 0,
        "relation_records_skipped_by_limit": 0,
        "relation_records_skipped_by_relation_name_filter": 0,
        "external_target_relations_skipped": 0,
        "same_cui_relations_skipped": 0,
        "equivalence_relations_skipped": 0,
        "unresolved_target_relations_skipped": 0,
        "relation_fetch_failures": 0,
        "relation_fetches_unavailable": 0,
        "relation_fetches_skipped_by_negative_cache": 0,
        "source_vocab_mismatch_relations_skipped": 0,
        "source_ui_lookups_attempted": 0,
        "source_ui_lookups_skipped_by_limit": 0,
        "cuis_truncated_by_max_relations": [],
        "cuis_truncated_by_source_ui_lookups": [],
        "failures": failures,
    }


def select_source_cuis(
    local_cuis: set[str],
    include_cuis: Optional[Sequence[str]],
    skip_cuis: Optional[Sequence[str]],
    max_cuis: Optional[int],
    by_cui: dict[str, list[LocalConcept]],
) -> tuple[list[str], list[dict[str, Any]]]:
    requested_include_cuis = sorted(
        {
            normalize_cui(cui)
            for cui in (include_cuis or [])
            if normalize_cui(cui)
        }
    )
    requested_skip_cuis = sorted(
        {
            normalize_cui(cui)
            for cui in (skip_cuis or [])
            if normalize_cui(cui)
        }
    )
    skipped: list[dict[str, Any]] = []

    for cui in requested_include_cuis:
        if cui not in local_cuis:
            skipped.append(
                {
                    "cui": cui,
                    "reason": "requested_include_not_in_local_cuis",
                    "local_concepts": 0,
                }
            )

    for cui in requested_skip_cuis:
        if cui not in local_cuis:
            skipped.append(
                {
                    "cui": cui,
                    "reason": "requested_skip_not_in_local_cuis",
                    "local_concepts": 0,
                }
            )

    source_pool = set(local_cuis)
    if requested_include_cuis:
        source_pool.intersection_update(requested_include_cuis)

    source_cuis: list[str] = []
    for cui in sorted(source_pool):
        if cui in requested_skip_cuis:
            skipped.append(
                {
                    "cui": cui,
                    "reason": "requested_skip",
                    "local_concepts": len(by_cui.get(cui, [])),
                }
            )
            continue
        source_cuis.append(cui)

    if max_cuis is not None:
        max_cuis = max(int(max_cuis), 0)
        for cui in source_cuis[max_cuis:]:
            skipped.append(
                {
                    "cui": cui,
                    "reason": "max_cuis_limit",
                    "local_concepts": len(by_cui.get(cui, [])),
                }
            )
        source_cuis = source_cuis[:max_cuis]

    return source_cuis, skipped


def build_candidate_edges(
    doc_id: Optional[str],
    concepts: Sequence[LocalConcept],
    client: UMLSRelationsClient,
    source_vocab: str,
    max_cuis: Optional[int] = None,
    include_cuis: Optional[Sequence[str]] = None,
    skip_cuis: Optional[Sequence[str]] = None,
    max_relations_per_cui: int = DEFAULT_MAX_RELATIONS_PER_CUI,
    max_source_ui_lookups_per_cui: int = DEFAULT_MAX_SOURCE_UI_LOOKUPS_PER_CUI,
    write_partial_every: int = DEFAULT_WRITE_PARTIAL_EVERY,
    partial_csv_path: Optional[Path] = None,
    partial_summary_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    dry_run: bool = True,
    include_relation_names: Optional[Sequence[str]] = None,
    exclude_relation_names: Optional[Sequence[str]] = None,
    strong_relations_only: bool = False,
    relation_profile: Optional[str] = None,
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    materialization_mode = normalize_materialization_mode(materialization_mode)
    by_cui = concepts_by_cui(concepts)
    local_cuis = set(by_cui)
    source_cuis, skipped_cuis = select_source_cuis(
        local_cuis=local_cuis,
        include_cuis=include_cuis,
        skip_cuis=skip_cuis,
        max_cuis=max_cuis,
        by_cui=by_cui,
    )
    normalized_include_cuis = sorted(
        {normalize_cui(cui) for cui in (include_cuis or []) if normalize_cui(cui)}
    )
    normalized_skip_cuis = sorted(
        {normalize_cui(cui) for cui in (skip_cuis or []) if normalize_cui(cui)}
    )
    (
        resolved_include_names,
        resolved_exclude_names,
        selected_relation_profile,
    ) = resolve_relation_name_filters(
        include_relation_names=include_relation_names,
        exclude_relation_names=exclude_relation_names,
        strong_relations_only=strong_relations_only,
        relation_profile=relation_profile,
    )
    edges: list[dict[str, Any]] = []
    stats = build_initial_relation_stats(
        local_cuis=local_cuis,
        source_cuis=source_cuis,
        skipped_cuis=skipped_cuis,
        max_cuis=max_cuis,
        include_cuis=normalized_include_cuis,
        skip_cuis=normalized_skip_cuis,
        max_relations_per_cui=max_relations_per_cui,
        max_source_ui_lookups_per_cui=max_source_ui_lookups_per_cui,
        write_partial_every=write_partial_every,
        include_relation_names=resolved_include_names,
        exclude_relation_names=resolved_exclude_names,
        strong_relations_only=strong_relations_only,
        relation_profile=selected_relation_profile,
        materialization_mode=materialization_mode,
    )
    stats["ignore_negative_cache"] = bool(getattr(client, "ignore_negative_cache", False))

    total_cuis = len(source_cuis)
    logger.info(
        "Fetching UMLS relations for %d source CUIs | local_cuis=%d | skipped_cuis=%d | source_vocab=%s",
        total_cuis,
        len(local_cuis),
        len(skipped_cuis),
        source_vocab or "ALL",
    )

    for index, source_cui in enumerate(source_cuis, start=1):
        local_concept_count = len(by_cui[source_cui])
        per_source_ui_lookups_attempted = 0
        per_source_ui_lookups_skipped = 0
        relation_records_fetched = 0
        try:
            relation_result = client.get_relations(
                cui=source_cui,
                source_vocab=source_vocab,
                max_records=max_relations_per_cui,
            )
        except UMLSAPIAuthError:
            raise
        except UMLSAPIError as e:
            stats["relation_fetch_failures"] += 1
            stats["failures"].append({"cui": source_cui, "error": str(e)})
            stats["processed_cuis"].append(
                {
                    "index": index,
                    "cui": source_cui,
                    "local_concepts": local_concept_count,
                    "relation_records_fetched": 0,
                    "relation_records_processed": 0,
                    "source_ui_lookups_attempted": 0,
                    "source_ui_lookups_skipped": 0,
                    "candidate_edges_retained": len(deduplicate_edges(edges)),
                    "status": "fetch_failed",
                }
            )
            logger.warning(
                "Skipping CUI after UMLS relation fetch failure | cui=%s | error=%s",
                source_cui,
                e,
            )
            log_cui_progress(
                index=index,
                total=total_cuis,
                source_cui=source_cui,
                local_concepts=local_concept_count,
                relation_records_fetched=0,
                source_ui_lookups_attempted=0,
                source_ui_lookups_skipped=0,
                candidate_edges_retained=len(deduplicate_edges(edges)),
            )
            if (
                write_partial_every > 0
                and index % write_partial_every == 0
                and partial_csv_path is not None
                and partial_summary_path is not None
                and cache_dir is not None
            ):
                stats["partial_exports_written"] += 1
                stats["last_partial_export_processed_cuis"] = index
                write_review_exports(
                    doc_id=doc_id,
                    source_vocab=source_vocab,
                    concepts=concepts,
                    edges=deduplicate_edges(edges),
                    stats=stats,
                    client_stats=dict(client.stats),
                    csv_path=partial_csv_path,
                    summary_path=partial_summary_path,
                    cache_dir=cache_dir,
                    dry_run=dry_run,
                    materialization_mode=materialization_mode,
                )
                logger.info(
                    "Partial UMLS connection export written | processed_cuis=%d | csv=%s | summary=%s",
                    index,
                    partial_csv_path,
                    partial_summary_path,
                )
            continue

        relation_records = relation_result.records
        relation_records_fetched = relation_result.fetched_records
        relation_status = "processed"
        if relation_result.status == "relations_unavailable":
            if relation_result.from_negative_cache:
                stats["relation_fetches_skipped_by_negative_cache"] += 1
                relation_status = "negative_cache_skipped"
                logger.info(
                    "Skipping UMLS relations fetch from negative cache | cui=%s | source_vocab=%s | failure_count=%d",
                    source_cui,
                    source_vocab or "ALL",
                    relation_result.failure_count,
                )
            else:
                stats["relation_fetches_unavailable"] += 1
                relation_status = "relations_unavailable"

        stats["umls_relation_records_fetched"] += relation_records_fetched
        stats["umls_relation_records_processed"] += len(relation_records)
        stats["relation_records_skipped_by_limit"] += relation_result.skipped_by_limit

        if relation_result.truncated_by_limit:
            stats["cuis_truncated_by_max_relations"].append(
                {
                    "cui": source_cui,
                    "fetched_records": relation_records_fetched,
                    "processed_records": len(relation_records),
                    "skipped_by_limit": relation_result.skipped_by_limit,
                    "limit": max_relations_per_cui,
                }
            )

        for record in relation_records:
            relation_label = clean_text(record.get("relationLabel"))
            additional_label = clean_text(record.get("additionalRelationLabel"))
            relation_source = clean_text(record.get("rootSource"))
            normalized_additional_label = normalize_relation_term(additional_label)

            if (
                source_vocab
                and relation_source
                and relation_source.casefold() != source_vocab.casefold()
            ):
                stats["source_vocab_mismatch_relations_skipped"] += 1
                continue

            if (
                resolved_include_names
                and normalized_additional_label not in resolved_include_names
            ):
                stats["relation_records_skipped_by_relation_name_filter"] += 1
                continue

            if (
                resolved_exclude_names
                and normalized_additional_label in resolved_exclude_names
            ):
                stats["relation_records_skipped_by_relation_name_filter"] += 1
                continue

            if is_equivalence_relation(relation_label, additional_label):
                stats["equivalence_relations_skipped"] += 1
                continue

            if relation_requires_source_ui_lookup(record):
                if (
                    per_source_ui_lookups_attempted
                    >= max_source_ui_lookups_per_cui
                ):
                    per_source_ui_lookups_skipped += 1
                    stats["source_ui_lookups_skipped_by_limit"] += 1
                    stats["unresolved_target_relations_skipped"] += 1
                    continue

                per_source_ui_lookups_attempted += 1
                stats["source_ui_lookups_attempted"] += 1

            related_cuis, resolution_error = resolve_related_cuis(
                client=client,
                record=record,
                source_vocab=source_vocab,
            )
            if resolution_error is not None:
                stats["unresolved_target_relations_skipped"] += 1
                continue

            relation_ui = clean_text(record.get("ui") or record.get("relationUi"))
            raw_id = relation_record_id(record)
            relation_raw_related_id = get_raw_relation_value(
                record,
                "relatedId",
                "relatedID",
            )
            relation_raw_related_id_name = get_raw_relation_value(
                record,
                "relatedIdName",
                "relatedIDName",
                "relatedName",
            )
            relation_raw_related_from_id = get_raw_relation_value(
                record,
                "relatedFromId",
                "relatedFromID",
            )
            relation_raw_related_from_id_name = get_raw_relation_value(
                record,
                "relatedFromIdName",
                "relatedFromIDName",
                "relatedFromName",
            )

            for target_cui in sorted(set(related_cuis)):
                target_cui = normalize_cui(target_cui)
                if source_cui == target_cui:
                    stats["same_cui_relations_skipped"] += 1
                    continue
                if target_cui not in local_cuis:
                    stats["external_target_relations_skipped"] += 1
                    continue

                for source_concept in by_cui[source_cui]:
                    for target_concept in by_cui[target_cui]:
                        edges.append(
                            {
                                "doc_id": clean_text(doc_id),
                                "source_concept_id": source_concept.concept_id,
                                "source_name": source_concept.name,
                                "source_type": source_concept.canonical_type,
                                "source_cui": source_cui,
                                "target_concept_id": target_concept.concept_id,
                                "target_name": target_concept.name,
                                "target_type": target_concept.canonical_type,
                                "target_cui": target_cui,
                                "umls_relation_label": relation_label,
                                "umls_additional_relation_label": additional_label,
                                "umls_relation_source": relation_source,
                                "umls_relation_ui": relation_ui,
                                "source_vocabulary": source_vocab,
                                "relation_raw_id": raw_id,
                                "connection_status": CONNECTION_STATUS,
                                "source_umls_canonical_name": source_concept.umls_canonical_name,
                                "source_umls_semantic_types": serialize_string_tuple(source_concept.umls_semantic_types),
                                "source_umls_score": source_concept.umls_score,
                                "source_observed_types": serialize_string_tuple(source_concept.observed_types),
                                "source_type_support_pairs": serialize_string_tuple(source_concept.type_support_pairs),
                                "source_type_resolution_status": source_concept.type_resolution_status,
                                "source_needs_type_review": source_concept.needs_type_review,
                                "target_umls_canonical_name": target_concept.umls_canonical_name,
                                "target_umls_semantic_types": serialize_string_tuple(target_concept.umls_semantic_types),
                                "target_umls_score": target_concept.umls_score,
                                "target_observed_types": serialize_string_tuple(target_concept.observed_types),
                                "target_type_support_pairs": serialize_string_tuple(target_concept.type_support_pairs),
                                "target_type_resolution_status": target_concept.type_resolution_status,
                                "target_needs_type_review": target_concept.needs_type_review,
                                "relation_raw_related_id": relation_raw_related_id,
                                "relation_raw_related_id_name": relation_raw_related_id_name,
                                "relation_raw_related_from_id": relation_raw_related_from_id,
                                "relation_raw_related_from_id_name": relation_raw_related_from_id_name,
                            }
                        )

        deduped_so_far = deduplicate_edges(edges)
        stats["internal_candidate_edges_retained"] = len(deduped_so_far)

        if per_source_ui_lookups_skipped:
            stats["cuis_truncated_by_source_ui_lookups"].append(
                {
                    "cui": source_cui,
                    "attempted": per_source_ui_lookups_attempted,
                    "skipped": per_source_ui_lookups_skipped,
                    "limit": max_source_ui_lookups_per_cui,
                }
            )

        stats["processed_cuis"].append(
            {
                "index": index,
                "cui": source_cui,
                "local_concepts": local_concept_count,
                "relation_records_fetched": relation_records_fetched,
                "relation_records_processed": len(relation_records),
                "source_ui_lookups_attempted": per_source_ui_lookups_attempted,
                "source_ui_lookups_skipped": per_source_ui_lookups_skipped,
                "candidate_edges_retained": len(deduped_so_far),
                "status": relation_status,
            }
        )
        log_cui_progress(
            index=index,
            total=total_cuis,
            source_cui=source_cui,
            local_concepts=local_concept_count,
            relation_records_fetched=relation_records_fetched,
            source_ui_lookups_attempted=per_source_ui_lookups_attempted,
            source_ui_lookups_skipped=per_source_ui_lookups_skipped,
            candidate_edges_retained=len(deduped_so_far),
        )

        if (
            write_partial_every > 0
            and index % write_partial_every == 0
            and partial_csv_path is not None
            and partial_summary_path is not None
            and cache_dir is not None
        ):
            stats["partial_exports_written"] += 1
            stats["last_partial_export_processed_cuis"] = index
            write_review_exports(
                doc_id=doc_id,
                source_vocab=source_vocab,
                concepts=concepts,
                edges=deduped_so_far,
                stats=stats,
                client_stats=dict(client.stats),
                csv_path=partial_csv_path,
                summary_path=partial_summary_path,
                cache_dir=cache_dir,
                dry_run=dry_run,
                materialization_mode=materialization_mode,
            )
            logger.info(
                "Partial UMLS connection export written | processed_cuis=%d | csv=%s | summary=%s",
                index,
                partial_csv_path,
                partial_summary_path,
            )

    deduped_edges = deduplicate_edges(edges)
    stats["internal_candidate_edges_retained"] = len(deduped_edges)
    return deduped_edges, stats


def log_cui_progress(
    index: int,
    total: int,
    source_cui: str,
    local_concepts: int,
    relation_records_fetched: int,
    source_ui_lookups_attempted: int,
    source_ui_lookups_skipped: int,
    candidate_edges_retained: int,
) -> None:
    logger.info(
        "UMLS CUI progress | %d/%d | cui=%s | local_concepts=%d | relation_records_fetched=%d | source_ui_lookups_attempted=%d | source_ui_lookups_skipped=%d | candidate_edges_retained=%d",
        index,
        total,
        source_cui,
        local_concepts,
        relation_records_fetched,
        source_ui_lookups_attempted,
        source_ui_lookups_skipped,
        candidate_edges_retained,
    )


def deduplicate_edges(edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []

    for edge in sorted(
        edges,
        key=lambda item: tuple(clean_text(item.get(column)) for column in CSV_COLUMNS),
    ):
        key = tuple(edge.get(column) for column in CSV_COLUMNS)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)

    return deduped


def resolve_run_name(
    *,
    run_name: Optional[str] = None,
    relation_profile: Optional[str] = None,
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
    max_relations_per_cui: int = DEFAULT_MAX_RELATIONS_PER_CUI,
    max_source_ui_lookups_per_cui: int = DEFAULT_MAX_SOURCE_UI_LOOKUPS_PER_CUI,
) -> str:
    explicit_run_name = clean_text(run_name)
    if explicit_run_name:
        return safe_filename_component(explicit_run_name, fallback="umls_run")

    profile_component = safe_filename_component(
        relation_profile or "custom",
        fallback="custom",
    )
    mode_component = safe_filename_component(
        normalize_materialization_mode(materialization_mode),
        fallback=DEFAULT_MATERIALIZATION_MODE,
    )
    return "__".join(
        [
            profile_component,
            mode_component,
            f"r{int(max_relations_per_cui)}",
            f"ui{int(max_source_ui_lookups_per_cui)}",
        ]
    )


def resolve_run_output_dir(output_dir: Path, run_name: str) -> Path:
    return Path(output_dir) / safe_filename_component(run_name, fallback="umls_run")


def output_paths(doc_id: Optional[str], output_dir: Path) -> tuple[Path, Path]:
    safe_doc_id = safe_filename_component(doc_id, fallback="all_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / f"{safe_doc_id}_candidate_edges.csv",
        output_dir / f"{safe_doc_id}_summary.md",
    )


def collapsed_connections_path(doc_id: Optional[str], output_dir: Path) -> Path:
    safe_doc_id = safe_filename_component(doc_id, fallback="all_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{safe_doc_id}_collapsed_connections.json"


def materialization_report_path(doc_id: Optional[str], output_dir: Path) -> Path:
    safe_doc_id = safe_filename_component(doc_id, fallback="all_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{safe_doc_id}_materialization_report.json"


def statistics_report_path(doc_id: Optional[str], output_dir: Path) -> Path:
    safe_doc_id = safe_filename_component(doc_id, fallback="all_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{safe_doc_id}_connection_statistics.json"


def write_candidate_edges_csv(path: Path, edges: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for edge in edges:
            writer.writerow({column: edge.get(column, "") for column in CSV_COLUMNS})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def markdown_escape(value: Any) -> str:
    if value is None:
        return "(none)"

    text = str(value).strip()
    if text == "":
        return "(none)"

    return text.replace("|", "\\|")


def markdown_count_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[str]:
    if not rows:
        return ["_No rows._"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(value) for value in row) + " |")
    return lines


def relation_display(edge: dict[str, Any]) -> str:
    label = clean_text(edge.get("umls_relation_label")) or "(none)"
    additional = clean_text(edge.get("umls_additional_relation_label"))
    if additional:
        return f"{label} / {additional}"
    return label


def canonical_relation_name(edge: dict[str, Any]) -> str:
    return normalize_relation_term(edge.get("umls_additional_relation_label"))


def build_collapsed_edge_key(
    doc_id: Optional[str],
    source_vocab: str,
    umls_version: str,
    source_cui: str,
    target_cui: str,
    relation_label: str,
    relation_name: str,
) -> str:
    payload = {
        "doc_id": clean_text(doc_id),
        "source_vocab": clean_text(source_vocab),
        "umls_version": clean_text(umls_version),
        "source_cui": normalize_cui(source_cui),
        "target_cui": normalize_cui(target_cui),
        "relation_label": clean_text(relation_label),
        "relation_name": normalize_relation_term(relation_name),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def local_type_compatible_count_key(value: Any) -> str:
    if value is None:
        return "null"
    return "true" if bool(value) else "false"


def is_first_extension_relation_name(relation_name: str) -> bool:
    return normalize_relation_term(relation_name) in FIRST_SNOMED_EXTENSION_RELATION_NAMES


def catalog_relationship_type_rows() -> list[dict[str, Any]]:
    return [
        {
            "relation_name": relation_name,
            "relationship_type": spec.relationship_type,
            "family": spec.family,
            "validation_mode": spec.validation_mode,
            "materialize_by_default": spec.materialize_by_default,
            "default_traversal_policy": spec.default_traversal_policy,
        }
        for relation_name, spec in sorted(RELATION_SPECS.items())
    ]


def catalog_local_type_rule_rows() -> list[dict[str, Any]]:
    return [
        {
            "relation_name": relation_name,
            "relationship_type": spec.relationship_type,
            "source_types": list(spec.source_types),
            "target_types": list(spec.target_types),
            "validation_mode": spec.validation_mode,
            "materialize_by_default": spec.materialize_by_default,
        }
        for relation_name, spec in sorted(RELATION_SPECS.items())
        if spec.validation_mode == "exact" and spec.source_types and spec.target_types
    ]


def first_extension_relationship_types() -> list[str]:
    return sorted(
        RELATION_TYPE_BY_NAME[relation_name]
        for relation_name in FIRST_SNOMED_EXTENSION_RELATION_NAMES
    )


def first_extension_local_type_rule_rows() -> list[dict[str, Any]]:
    return [
        row
        for row in catalog_local_type_rule_rows()
        if row["relation_name"] in FIRST_SNOMED_EXTENSION_RELATION_NAMES
    ]


def normalize_materialization_mode(value: Optional[str]) -> str:
    mode = clean_text(value).casefold() or DEFAULT_MATERIALIZATION_MODE
    if mode not in MATERIALIZATION_MODE_CHOICES:
        raise ValueError(
            f"Invalid materialization mode {value!r}; the only supported "
            "values are none and safe_only"
        )
    return mode


def should_materialize_relation(
    edge: dict[str, Any],
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
    *,
    require_representatives: bool = True,
) -> tuple[bool, str]:
    mode = normalize_materialization_mode(materialization_mode)
    if mode == "none":
        return False, "materialization_mode_none"

    if require_representatives and (
        not edge.get("source_representative")
        or not edge.get("target_representative")
    ):
        return False, "missing_representative"

    relation_name = normalize_relation_term(edge.get("relation_name"))
    spec = RELATION_SPECS.get(relation_name)
    relationship_type = clean_text(edge.get("relationship_type"))
    if spec is None or not relationship_type:
        return False, "unsupported_relation"

    compatibility_status = clean_text(edge.get("compatibility_status"))
    local_type_compatible = edge.get("local_type_compatible")

    if not spec.materialize_by_default:
        return False, "not_materialize_by_default"
    if compatibility_status != "compatible":
        return False, f"compatibility_status_{compatibility_status or 'missing'}"
    if local_type_compatible is not True:
        return False, "local_type_not_compatible"

    traversal_policy = clean_text(edge.get("traversal_policy"))
    if traversal_policy not in {"safe", "hierarchy"}:
        return False, f"traversal_policy_{traversal_policy or 'missing'}"
    if bool(edge.get("review_needed")):
        return False, "review_needed"

    return True, "safe_only_materialization"


def nearest_rank_percentile(values: Sequence[int], percentile: int) -> Optional[int]:
    """
    Deterministic nearest-rank percentile over sorted integer values.

    For p95, the selected value is rank ceil(0.95 * n), clamped to the
    observed range. This intentionally avoids interpolation so experiment
    reports remain directly comparable.
    """
    if not values:
        return None

    sorted_values = sorted(int(value) for value in values)
    percentile = max(0, min(int(percentile), 100))
    rank = ((percentile * len(sorted_values)) + 99) // 100
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


def median_numeric(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None

    sorted_values = sorted(int(value) for value in values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def representative_by_cui_from_edges(
    collapsed_edges: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for edge in collapsed_edges:
        for cui_key, representative_key in (
            ("source_cui", "source_representative"),
            ("target_cui", "target_representative"),
        ):
            cui = normalize_cui(edge.get(cui_key))
            representative = edge.get(representative_key)
            if cui and isinstance(representative, dict) and cui not in representatives:
                representatives[cui] = dict(representative)
    return representatives


def compute_cui_graph_statistics(
    collapsed_edges: Sequence[dict[str, Any]],
    *,
    eligible_cuis: Optional[Sequence[str]] = None,
    selected_source_cuis: Optional[Sequence[str]] = None,
    representatives_by_cui: Optional[dict[str, LocalConcept]] = None,
) -> dict[str, Any]:
    normalized_edges = [
        (normalize_cui(edge.get("source_cui")), normalize_cui(edge.get("target_cui")))
        for edge in collapsed_edges
    ]
    normalized_edges = [
        (source_cui, target_cui)
        for source_cui, target_cui in normalized_edges
        if source_cui and target_cui
    ]
    eligible_cui_set = {
        normalize_cui(cui)
        for cui in (eligible_cuis or [])
        if normalize_cui(cui)
    }
    selected_source_cui_set = {
        normalize_cui(cui)
        for cui in (selected_source_cuis or [])
        if normalize_cui(cui)
    }
    if not eligible_cui_set:
        eligible_cui_set = {
            cui
            for edge in normalized_edges
            for cui in edge
            if cui
        }

    source_cuis = {source_cui for source_cui, _target_cui in normalized_edges}
    target_cuis = {target_cui for _source_cui, target_cui in normalized_edges}
    connected_cuis = source_cuis | target_cuis

    undirected_pairs = {
        tuple(sorted((source_cui, target_cui)))
        for source_cui, target_cui in normalized_edges
    }
    self_cui_edges = sum(
        1 for source_cui, target_cui in normalized_edges if source_cui == target_cui
    )

    adjacency: dict[str, set[str]] = defaultdict(set)
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    for source_cui, target_cui in normalized_edges:
        adjacency[source_cui].add(target_cui)
        adjacency[target_cui].add(source_cui)
        out_degree[source_cui] += 1
        in_degree[target_cui] += 1

    graph_nodes = set(eligible_cui_set) | connected_cuis
    for cui in graph_nodes:
        adjacency.setdefault(cui, set())

    degree_by_cui = {
        cui: len(neighbors)
        for cui, neighbors in adjacency.items()
        if cui in graph_nodes
    }
    degree_values = [degree_by_cui.get(cui, 0) for cui in sorted(graph_nodes)]
    max_degree = max(degree_values) if degree_values else 0
    maximum_degree_cuis = sorted(
        cui for cui, degree in degree_by_cui.items() if degree == max_degree
    ) if degree_values else []

    visited: set[str] = set()
    component_sizes: list[int] = []
    for start in sorted(graph_nodes):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    largest_component_size = max(component_sizes) if component_sizes else 0
    eligible_count = len(eligible_cui_set)
    selected_count = len(selected_source_cui_set)
    isolated_eligible_cuis = sorted(
        cui for cui in eligible_cui_set if degree_by_cui.get(cui, 0) == 0
    )
    isolated_selected_source_cuis = sorted(
        cui for cui in selected_source_cui_set if degree_by_cui.get(cui, 0) == 0
    )

    edge_representatives = representative_by_cui_from_edges(collapsed_edges)
    top_degree_rows: list[dict[str, Any]] = []
    for cui in sorted(degree_by_cui, key=lambda item: (-degree_by_cui[item], item))[:10]:
        local_representative = (
            representatives_by_cui.get(cui)
            if representatives_by_cui is not None
            else None
        )
        edge_representative = edge_representatives.get(cui) or {}
        top_degree_rows.append(
            {
                "cui": cui,
                "representative_name": (
                    local_representative.name
                    if local_representative is not None
                    else edge_representative.get("name")
                ),
                "canonical_type": (
                    local_representative.canonical_type
                    if local_representative is not None
                    else edge_representative.get("canonical_type")
                ),
                "degree": degree_by_cui.get(cui, 0),
                "in_degree": int(in_degree.get(cui, 0)),
                "out_degree": int(out_degree.get(cui, 0)),
            }
        )

    return {
        "eligible_local_cuis": eligible_count,
        "selected_source_cuis": selected_count,
        "unique_source_cuis": len(source_cuis),
        "unique_target_cuis": len(target_cuis),
        "unique_connected_cuis": len(connected_cuis),
        "isolated_eligible_local_cuis": isolated_eligible_cuis,
        "isolated_eligible_local_cui_count": len(isolated_eligible_cuis),
        "isolated_selected_source_cuis": isolated_selected_source_cuis,
        "isolated_selected_source_cui_count": len(isolated_selected_source_cuis),
        "cui_coverage_ratio": (
            len(connected_cuis & eligible_cui_set) / eligible_count
            if eligible_count
            else 0.0
        ),
        "selected_source_cui_coverage_ratio": (
            len(connected_cuis & selected_source_cui_set) / selected_count
            if selected_count
            else 0.0
        ),
        "directed_edges": len(normalized_edges),
        "unique_undirected_cui_pairs": len(undirected_pairs),
        "self_cui_edges": self_cui_edges,
        "minimum_degree": min(degree_values) if degree_values else 0,
        "median_degree": median_numeric(degree_values),
        "mean_degree": (
            sum(degree_values) / len(degree_values)
            if degree_values
            else 0.0
        ),
        "p95_degree": nearest_rank_percentile(degree_values, 95),
        "maximum_degree": max_degree,
        "maximum_degree_cuis": maximum_degree_cuis,
        "weakly_connected_components": len(component_sizes),
        "largest_component_size": largest_component_size,
        "largest_component_coverage_ratio": (
            largest_component_size / eligible_count if eligible_count else 0.0
        ),
        "maximum_in_degree": max(in_degree.values()) if in_degree else 0,
        "maximum_in_degree_cuis": sorted(
            cui
            for cui, degree in in_degree.items()
            if degree == (max(in_degree.values()) if in_degree else 0)
        ) if in_degree else [],
        "maximum_out_degree": max(out_degree.values()) if out_degree else 0,
        "maximum_out_degree_cuis": sorted(
            cui
            for cui, degree in out_degree.items()
            if degree == (max(out_degree.values()) if out_degree else 0)
        ) if out_degree else [],
        "top_degree_cui_representatives": top_degree_rows,
        "degree_percentile_method": "nearest_rank",
    }


def build_relation_profile_statistics(
    collapsed_edges: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in collapsed_edges:
        by_relation[clean_text(edge.get("relation_name")) or "(none)"].append(edge)

    summaries: list[dict[str, Any]] = []
    for relation_name in sorted(by_relation):
        edges_for_relation = by_relation[relation_name]
        compatibility_counts = Counter(
            clean_text(edge.get("compatibility_status")) or "(none)"
            for edge in edges_for_relation
        )
        traversal_counts = Counter(
            clean_text(edge.get("traversal_policy")) or "(none)"
            for edge in edges_for_relation
        )
        type_counts = Counter(
            (
                clean_text(edge.get("representative_source_type")) or "(none)",
                clean_text(edge.get("representative_target_type")) or "(none)",
            )
            for edge in edges_for_relation
        )
        examples = [
            {
                "source_cui": edge.get("source_cui"),
                "source_type": edge.get("representative_source_type"),
                "target_cui": edge.get("target_cui"),
                "target_type": edge.get("representative_target_type"),
                "compatibility_status": edge.get("compatibility_status"),
                "traversal_policy": edge.get("traversal_policy"),
                "materialization_decision": edge.get("materialization_decision"),
                "source_names": list(edge.get("source_names") or [])[:3],
                "target_names": list(edge.get("target_names") or [])[:3],
            }
            for edge in edges_for_relation[:5]
        ]
        summaries.append(
            {
                "relation_name": relation_name,
                "candidate_count": len(edges_for_relation),
                "materializable_count": sum(
                    1
                    for edge in edges_for_relation
                    if edge.get("materialization_decision") is True
                ),
                "compatibility_status_counts": dict(
                    sorted(compatibility_counts.items())
                ),
                "traversal_policy_counts": dict(sorted(traversal_counts.items())),
                "source_type_to_target_type_counts": [
                    {
                        "source_type": source_type,
                        "target_type": target_type,
                        "count": count,
                    }
                    for (source_type, target_type), count in type_counts.most_common()
                ],
                "examples": examples,
            }
        )
    return summaries


def build_collapsed_connection_statistics(
    collapsed_edges: Sequence[dict[str, Any]],
    *,
    eligible_cuis: Optional[Sequence[str]] = None,
    selected_source_cuis: Optional[Sequence[str]] = None,
    representatives_by_cui: Optional[dict[str, LocalConcept]] = None,
) -> dict[str, Any]:
    counts_by_relation_name = Counter(
        clean_text(edge.get("relation_name")) or "(none)"
        for edge in collapsed_edges
    )
    counts_by_traversal_policy = Counter(
        clean_text(edge.get("traversal_policy")) or "(none)"
        for edge in collapsed_edges
    )
    counts_by_compatibility_status = Counter(
        clean_text(edge.get("compatibility_status")) or "(none)"
        for edge in collapsed_edges
    )
    counts_by_compatibility_reason = Counter(
        clean_text(edge.get("compatibility_reason")) or "(none)"
        for edge in collapsed_edges
    )
    counts_by_local_type_compatible = Counter(
        local_type_compatible_count_key(edge.get("local_type_compatible"))
        for edge in collapsed_edges
    )
    counts_by_materialize_by_default = Counter(
        local_type_compatible_count_key(edge.get("materialize_by_default"))
        for edge in collapsed_edges
    )
    counts_by_materialization_decision = Counter(
        local_type_compatible_count_key(edge.get("materialization_decision"))
        for edge in collapsed_edges
    )
    counts_by_materialization_reason = Counter(
        clean_text(edge.get("materialization_decision_reason")) or "(none)"
        for edge in collapsed_edges
    )
    representative_type_counts = Counter(
        (
            clean_text(edge.get("representative_source_type")) or "(none)",
            clean_text(edge.get("relation_name")) or "(none)",
            clean_text(edge.get("representative_target_type")) or "(none)",
        )
        for edge in collapsed_edges
    )
    examples_by_relation_status: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for edge in collapsed_edges:
        key = (
            clean_text(edge.get("relation_name")) or "(none)",
            clean_text(edge.get("compatibility_status")) or "(none)",
        )
        examples = examples_by_relation_status.setdefault(key, [])
        if len(examples) < 3:
            examples.append(
                {
                    "source_cui": edge.get("source_cui"),
                    "source_type": edge.get("representative_source_type"),
                    "relation_name": edge.get("relation_name"),
                    "target_type": edge.get("representative_target_type"),
                    "target_cui": edge.get("target_cui"),
                    "compatibility_status": edge.get("compatibility_status"),
                    "compatibility_reason": edge.get("compatibility_reason"),
                    "traversal_policy": edge.get("traversal_policy"),
                    "materialization_decision": edge.get("materialization_decision"),
                    "materialization_decision_reason": edge.get(
                        "materialization_decision_reason"
                    ),
                    "source_names": list(edge.get("source_names") or [])[:3],
                    "target_names": list(edge.get("target_names") or [])[:3],
                }
            )

    first_extension_edges = [
        edge
        for edge in collapsed_edges
        if is_first_extension_relation_name(clean_text(edge.get("relation_name")))
    ]
    incompatible_examples = [
        {
            "source_cui": edge.get("source_cui"),
            "source_type": edge.get("representative_source_type"),
            "relation_name": edge.get("relation_name"),
            "target_type": edge.get("representative_target_type"),
            "target_cui": edge.get("target_cui"),
            "reason": edge.get("local_type_compatibility_reason"),
            "source_names": list(edge.get("source_names") or [])[:3],
            "target_names": list(edge.get("target_names") or [])[:3],
        }
        for edge in first_extension_edges
        if edge.get("local_type_compatible") is False
    ][:10]
    materializable_edges = [
        edge
        for edge in collapsed_edges
        if edge.get("materialization_decision") is True
    ]
    candidate_graph_statistics = compute_cui_graph_statistics(
        collapsed_edges,
        eligible_cuis=eligible_cuis,
        selected_source_cuis=selected_source_cuis,
        representatives_by_cui=representatives_by_cui,
    )
    materializable_graph_statistics = compute_cui_graph_statistics(
        materializable_edges,
        eligible_cuis=eligible_cuis,
        selected_source_cuis=selected_source_cuis,
        representatives_by_cui=representatives_by_cui,
    )

    return {
        "collapsed_first_extension_connections": len(first_extension_edges),
        "compatible_first_extension_connections": sum(
            1
            for edge in first_extension_edges
            if edge.get("local_type_compatible") is True
        ),
        "incompatible_first_extension_connections": sum(
            1
            for edge in first_extension_edges
            if edge.get("local_type_compatible") is False
        ),
        "counts_by_relation_name": dict(sorted(counts_by_relation_name.items())),
        "counts_by_traversal_policy": dict(
            sorted(counts_by_traversal_policy.items())
        ),
        "counts_by_compatibility_status": dict(
            sorted(counts_by_compatibility_status.items())
        ),
        "counts_by_compatibility_reason": dict(
            sorted(counts_by_compatibility_reason.items())
        ),
        "counts_by_local_type_compatible": dict(
            sorted(counts_by_local_type_compatible.items())
        ),
        "counts_by_materialize_by_default": dict(
            sorted(counts_by_materialize_by_default.items())
        ),
        "counts_by_materialization_decision": dict(
            sorted(counts_by_materialization_decision.items())
        ),
        "counts_by_materialization_reason": dict(
            sorted(counts_by_materialization_reason.items())
        ),
        "counts_by_source_representative_type_relation_target_representative_type": [
            {
                "source_type": source_type,
                "relation_name": relation_name,
                "target_type": target_type,
                "count": count,
            }
            for (
                source_type,
                relation_name,
                target_type,
            ), count in representative_type_counts.most_common()
        ],
        "examples_by_relation_and_compatibility_status": [
            {
                "relation_name": relation_name,
                "compatibility_status": compatibility_status,
                "examples": examples,
            }
            for (
                relation_name,
                compatibility_status,
            ), examples in sorted(examples_by_relation_status.items())
        ],
        "incompatible_local_type_examples": incompatible_examples,
        "candidate_graph_statistics": candidate_graph_statistics,
        "materializable_graph_statistics": materializable_graph_statistics,
        "relation_summaries": build_relation_profile_statistics(collapsed_edges),
    }


def collapsed_connection_key(
    edge: dict[str, Any],
    doc_id: Optional[str],
    source_vocab: str,
    umls_version: str,
) -> tuple[str, str, str, str, str, str, str]:
    source_cui = normalize_cui(edge.get("source_cui"))
    target_cui = normalize_cui(edge.get("target_cui"))
    relation_label = clean_text(edge.get("umls_relation_label"))
    relation_name = canonical_relation_name(edge)
    return (
        clean_text(doc_id),
        clean_text(source_vocab),
        clean_text(umls_version),
        source_cui,
        target_cui,
        relation_label,
        relation_name,
    )


def traversal_policy_for_relation(
    relation_name: str,
    source_types: Sequence[str] = (),
    target_types: Sequence[str] = (),
    local_type_compatible: bool | None = None,
    compatibility_status: Optional[str] = None,
) -> str:
    relation_name = normalize_relation_term(relation_name)
    spec = RELATION_SPECS.get(relation_name)
    if spec is None:
        return "type_review"

    source_type_set = {clean_text(item) for item in source_types if clean_text(item)}
    target_type_set = {clean_text(item) for item in target_types if clean_text(item)}

    if compatibility_status is None:
        if local_type_compatible is True:
            compatibility_status = "compatible"
        elif local_type_compatible is False:
            compatibility_status = "incompatible"
        elif len(source_type_set) == 1 and len(target_type_set) == 1:
            compatibility_status = evaluate_relation_compatibility(
                relation_name,
                next(iter(source_type_set)),
                next(iter(target_type_set)),
            ).status
        else:
            compatibility_status = "missing_type"

    compatibility_status = clean_text(compatibility_status)
    if compatibility_status in {
        "incompatible",
        "missing_type",
        "unsupported_local_schema",
    }:
        return "type_review"
    if compatibility_status == "review":
        return "review"

    if spec.validation_mode == "hierarchy":
        if compatibility_status == "compatible":
            return "hierarchy"
        if compatibility_status == "compatible_broad":
            return "hierarchy_review"
        return "review"

    return spec.default_traversal_policy


def review_needed_for_policy(traversal_policy: str) -> bool:
    return traversal_policy in {
        "hierarchy_review",
        "reverse_review",
        "review",
        "type_review",
    }


def build_collapsed_connections(
    edges: Sequence[dict[str, Any]],
    doc_id: Optional[str],
    source_vocab: str,
    umls_version: str,
    representatives_by_cui: Optional[dict[str, LocalConcept]] = None,
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}

    for edge in edges:
        key = collapsed_connection_key(
            edge=edge,
            doc_id=doc_id,
            source_vocab=source_vocab,
            umls_version=umls_version,
        )
        row = grouped.setdefault(
            key,
            {
                "doc_id": key[0],
                "source_vocabulary": key[1],
                "umls_version": key[2],
                "source_cui": key[3],
                "target_cui": key[4],
                "relation_label": key[5],
                "relation_name": key[6],
                "raw_rows": 0,
                "source_names": set(),
                "target_names": set(),
                "source_types": set(),
                "target_types": set(),
                "relation_ids": set(),
            },
        )
        row["raw_rows"] += 1
        source_name = clean_text(edge.get("source_name"))
        target_name = clean_text(edge.get("target_name"))
        source_type = clean_text(edge.get("source_type"))
        target_type = clean_text(edge.get("target_type"))
        relation_id = clean_text(edge.get("umls_relation_ui") or edge.get("relation_raw_id"))
        if source_name:
            row["source_names"].add(source_name)
        if target_name:
            row["target_names"].add(target_name)
        if source_type:
            row["source_types"].add(source_type)
        if target_type:
            row["target_types"].add(target_type)
        if relation_id:
            row["relation_ids"].add(relation_id)

    collapsed_edges: list[dict[str, Any]] = []
    representatives_by_cui = representatives_by_cui or {}
    for row in grouped.values():
        source_types = sorted(row["source_types"])
        target_types = sorted(row["target_types"])
        relation_name = row["relation_name"]
        relationship_metadata = build_ontology_relationship_metadata(
            row["source_vocabulary"]
        )
        source_representative = representatives_by_cui.get(row["source_cui"])
        target_representative = representatives_by_cui.get(row["target_cui"])
        representative_source_type = (
            clean_text(source_representative.canonical_type)
            if source_representative is not None
            else ""
        )
        representative_target_type = (
            clean_text(target_representative.canonical_type)
            if target_representative is not None
            else ""
        )
        compatibility = evaluate_relation_compatibility(
            relation_name=relation_name,
            source_type=representative_source_type,
            target_type=representative_target_type,
        )
        local_type_compatible = local_type_compatible_from_status(
            compatibility.status
        )
        traversal_policy = traversal_policy_for_relation(
            relation_name=relation_name,
            source_types=[representative_source_type],
            target_types=[representative_target_type],
            compatibility_status=compatibility.status,
        )
        spec = RELATION_SPECS.get(relation_name)
        collapsed_edge = {
            "edge_key": build_collapsed_edge_key(
                doc_id=row["doc_id"],
                source_vocab=row["source_vocabulary"],
                umls_version=row["umls_version"],
                source_cui=row["source_cui"],
                target_cui=row["target_cui"],
                relation_label=row["relation_label"],
                relation_name=relation_name,
            ),
            "doc_id": row["doc_id"],
            "source_cui": row["source_cui"],
            "target_cui": row["target_cui"],
            "relation_label": row["relation_label"],
            "relation_name": relation_name,
            "relationship_type": (
                spec.relationship_type if spec is not None else None
            ),
            "source_vocabulary": row["source_vocabulary"],
            "umls_version": row["umls_version"],
            "raw_rows": row["raw_rows"],
            "relation_ids": sorted(row["relation_ids"]),
            "source_names": sorted(row["source_names"]),
            "target_names": sorted(row["target_names"]),
            "source_types": source_types,
            "target_types": target_types,
            "status": CONNECTION_STATUS,
            **relationship_metadata,
            "compatibility_status": compatibility.status,
            "compatibility_reason": compatibility.reason,
            "relation_family": spec.family if spec is not None else None,
            "materialize_by_default": (
                spec.materialize_by_default if spec is not None else False
            ),
            "local_type_compatible": local_type_compatible,
            "local_type_compatibility_reason": compatibility.reason,
            "representative_source_type": representative_source_type or None,
            "representative_target_type": representative_target_type or None,
            "traversal_policy": traversal_policy,
            "review_needed": review_needed_for_policy(traversal_policy),
            "source_representative": concept_report(source_representative),
            "target_representative": concept_report(target_representative),
        }
        (
            materialization_decision,
            materialization_decision_reason,
        ) = should_materialize_relation(
            collapsed_edge,
            materialization_mode=materialization_mode,
        )
        collapsed_edge.update(
            {
                "materialization_mode": normalize_materialization_mode(
                    materialization_mode
                ),
                "materialization_decision": materialization_decision,
                "materialization_decision_reason": materialization_decision_reason,
            }
        )
        collapsed_edges.append(collapsed_edge)

    collapsed_edges.sort(
        key=lambda item: (
            -int(item["raw_rows"]),
            item["source_cui"],
            item["target_cui"],
            item["relation_label"],
            item["relation_name"],
        )
    )
    return collapsed_edges


def build_collapsed_connection_rows(
    edges: Sequence[dict[str, Any]] = (),
    doc_id: Optional[str] = None,
    source_vocab: str = DEFAULT_SOURCE_VOCAB,
    umls_version: str = DEFAULT_UMLS_VERSION,
    collapsed_edges: Optional[Sequence[dict[str, Any]]] = None,
) -> list[tuple[Any, ...]]:
    if collapsed_edges is None:
        collapsed_edges = build_collapsed_connections(
            edges=edges,
            doc_id=doc_id,
            source_vocab=source_vocab,
            umls_version=umls_version,
        )
    rows = [
        (
            edge["source_cui"],
            edge["target_cui"],
            edge["relation_label"],
            edge["relation_name"],
            edge.get("representative_source_type"),
            edge.get("representative_target_type"),
            local_type_compatible_count_key(edge.get("local_type_compatible")),
            edge.get("local_type_compatibility_reason"),
            edge.get("compatibility_status"),
            edge.get("compatibility_reason"),
            edge.get("relation_family"),
            edge["traversal_policy"],
            local_type_compatible_count_key(edge.get("materialize_by_default")),
            local_type_compatible_count_key(edge.get("materialization_decision")),
            edge.get("materialization_decision_reason"),
            edge["raw_rows"],
            len(edge["relation_ids"]),
            "; ".join(edge["source_names"][:3]),
            "; ".join(edge["target_names"][:3]),
        )
        for edge in collapsed_edges
    ]
    return rows


def build_summary_markdown(
    doc_id: Optional[str],
    source_vocab: str,
    concepts: Sequence[LocalConcept],
    edges: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    client_stats: dict[str, int],
    csv_path: Path,
    collapsed_json_path: Optional[Path],
    cache_dir: Path,
    dry_run: bool,
    umls_version: str = DEFAULT_UMLS_VERSION,
    collapsed_edges: Optional[Sequence[dict[str, Any]]] = None,
    materialization_report: Optional[dict[str, Any]] = None,
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
) -> str:
    materialization_mode = normalize_materialization_mode(materialization_mode)
    unique_cuis = {concept.umls_cui for concept in concepts}

    relation_counts = Counter(
        clean_text(edge.get("umls_relation_label")) or "(none)"
        for edge in edges
    )
    relation_rows = [
        (label, count)
        for label, count in relation_counts.most_common()
    ]

    type_relation_counts = Counter(
        (
            clean_text(edge.get("source_type")) or "(none)",
            relation_display(edge),
            clean_text(edge.get("target_type")) or "(none)",
        )
        for edge in edges
    )
    type_relation_rows = [
        (source_type, relation, target_type, count)
        for (source_type, relation, target_type), count in type_relation_counts.most_common()
    ]
    if collapsed_edges is None:
        representatives = select_representative_concepts(concepts)
        collapsed_edges = build_collapsed_connections(
            edges=edges,
            doc_id=doc_id,
            source_vocab=source_vocab,
            umls_version=umls_version,
            representatives_by_cui=representatives,
            materialization_mode=materialization_mode,
        )
    else:
        representatives = select_representative_concepts(concepts)
    collapsed_connection_rows = build_collapsed_connection_rows(
        collapsed_edges=collapsed_edges,
    )
    collapsed_connection_count = len(collapsed_connection_rows)
    collapsed_stats = build_collapsed_connection_statistics(
        collapsed_edges,
        eligible_cuis=stats.get("eligible_local_cuis") or [],
        selected_source_cuis=stats.get("selected_source_cuis") or [],
        representatives_by_cui=representatives,
    )
    duplicate_raw_rows_collapsed = max(0, len(edges) - collapsed_connection_count)
    strict_materializable_connections = sum(
        1 for edge in collapsed_edges if edge.get("materialization_decision") is True
    )
    cross_type_isa_edges = sum(
        1
        for edge in edges
        if normalize_relation_term(edge.get("umls_additional_relation_label"))
        in {"isa", "inverse_isa"}
        and clean_text(edge.get("source_type")) != clean_text(edge.get("target_type"))
    )
    missing_umls_semantic_type_edges = sum(
        1
        for edge in edges
        if not clean_text(edge.get("source_umls_semantic_types"))
        or not clean_text(edge.get("target_umls_semantic_types"))
    )
    needs_type_review_edges = sum(
        1
        for edge in edges
        if bool(edge.get("source_needs_type_review"))
        or bool(edge.get("target_needs_type_review"))
    )

    lines = [
        "# UMLS Connections Summary",
        "",
        f"- selected doc_id: `{doc_id or 'ALL'}`",
        f"- source vocabulary: `{source_vocab or 'ALL'}`",
        f"- UMLS version: `{umls_version}`",
        f"- run name: `{stats.get('run_name') or '(none)'}`",
        f"- output path: `{stats.get('output_dir') or csv_path.parent}`",
        f"- dry-run/read-only: `{str(dry_run).lower()}`",
        f"- selected relation profile: `{stats.get('selected_relation_profile') or '(none)'}`",
        f"- materialization mode: `{materialization_mode}`",
        f"- write_neo4j: `{str(bool(stats.get('write_neo4j'))).lower()}`",
        f"- replace_existing_connections: `{str(bool(stats.get('replace_existing_connections'))).lower()}`",
        f"- candidate CSV: `{csv_path}`",
        f"- collapsed JSON: `{collapsed_json_path or '(not written)'}`",
        f"- statistics JSON: `{stats.get('statistics_json_path') or '(not written)'}`",
        f"- relation cache directory: `{cache_dir}`",
        "",
        "## Counts",
        "",
        f"- local concepts considered: {len(concepts)}",
        f"- unique local CUIs considered: {len(unique_cuis)}",
        f"- processed CUIs: {len(stats.get('processed_cuis') or [])}",
        f"- skipped CUIs: {len(stats.get('skipped_cuis') or [])}",
        f"- UMLS relation records fetched: {stats.get('umls_relation_records_fetched', 0)}",
        f"- UMLS relation records processed: {stats.get('umls_relation_records_processed', 0)}",
        f"- UMLS relation records skipped by per-CUI limit: {stats.get('relation_records_skipped_by_limit', 0)}",
        f"- UMLS relation records skipped by relation-name filter: {stats.get('relation_records_skipped_by_relation_name_filter', 0)}",
        f"- internal candidate edges retained: {len(edges)}",
        f"- raw candidate edge rows: {len(edges)}",
        f"- collapsed candidate connections: {collapsed_connection_count}",
        f"- strict materializable connections: {strict_materializable_connections}",
        f"- collapsed first-extension connections: {collapsed_stats['collapsed_first_extension_connections']}",
        f"- compatible first-extension connections: {collapsed_stats['compatible_first_extension_connections']}",
        f"- incompatible first-extension connections: {collapsed_stats['incompatible_first_extension_connections']}",
        f"- duplicate raw rows collapsed: {duplicate_raw_rows_collapsed}",
        f"- real relation fetch failures: {stats.get('relation_fetch_failures', 0)}",
        f"- relation fetches unavailable after repeated 404: {stats.get('relation_fetches_unavailable', 0)}",
        f"- relation fetches skipped by negative cache: {stats.get('relation_fetches_skipped_by_negative_cache', 0)}",
        f"- unresolved target relations skipped: {stats.get('unresolved_target_relations_skipped', 0)}",
        f"- external target relations skipped: {stats.get('external_target_relations_skipped', 0)}",
        f"- same-CUI/equivalence relations skipped: {stats.get('same_cui_relations_skipped', 0) + stats.get('equivalence_relations_skipped', 0)}",
        f"- sourceUi target lookups attempted: {stats.get('source_ui_lookups_attempted', 0)}",
        f"- sourceUi target lookups skipped by per-CUI limit: {stats.get('source_ui_lookups_skipped_by_limit', 0)}",
        "",
        "## Review Notes",
        "",
        f"- retained candidate edges with cross-local-type isa/inverse_isa: {cross_type_isa_edges}",
        f"- retained candidate edges missing source or target UMLS semantic types: {missing_umls_semantic_type_edges}",
        f"- retained candidate edges with source or target needs_type_review=true: {needs_type_review_edges}",
        "",
        "## Safety Controls",
        "",
        f"- max_cuis: {stats.get('max_cuis') if stats.get('max_cuis') is not None else '(none)'}",
        f"- max_relations_per_cui: {stats.get('max_relations_per_cui')}",
        f"- max_source_ui_lookups_per_cui: {stats.get('max_source_ui_lookups_per_cui')}",
        f"- write_partial_every: {stats.get('write_partial_every')}",
        f"- strong_relations_only: `{str(bool(stats.get('strong_relations_only'))).lower()}`",
        f"- relation_profile: `{stats.get('selected_relation_profile') or '(none)'}`",
        f"- included CUIs: `{', '.join(stats.get('include_cuis') or []) or '(none)'}`",
        f"- skipped CUIs: `{', '.join(stats.get('skip_cuis') or []) or '(none)'}`",
        f"- ignore_negative_cache: `{str(bool(stats.get('ignore_negative_cache'))).lower()}`",
        f"- Neo4j writes enabled: `{str(bool((materialization_report or {}).get('write_neo4j'))).lower()}`",
        f"- included relation names: `{', '.join(stats.get('include_relation_names') or []) or '(none)'}`",
        f"- excluded relation names: `{', '.join(stats.get('exclude_relation_names') or []) or '(none)'}`",
        "",
        "## Partial Export Status",
        "",
        f"- partial exports written: {stats.get('partial_exports_written', 0)}",
        f"- last partial export processed CUIs: {stats.get('last_partial_export_processed_cuis', 0)}",
        f"- final export written: `{str(bool(stats.get('final_export_written'))).lower()}`",
        "",
        "## API Cache",
        "",
        f"- cache hits: {client_stats.get('api_cache_hits', 0)}",
        f"- cache misses: {client_stats.get('api_cache_misses', 0)}",
        f"- API requests: {client_stats.get('api_requests', 0)}",
        f"- API retries: {client_stats.get('api_retries', 0)}",
        f"- API errors: {client_stats.get('api_errors', 0)}",
        f"- relation negative-cache hits: {client_stats.get('relation_negative_cache_hits', 0)}",
        f"- relation negative-cache writes: {client_stats.get('relation_negative_cache_writes', 0)}",
        "",
        "## Processed CUIs",
        "",
    ]

    processed_rows = [
        (
            int_or_zero(item.get("index")),
            item.get("cui"),
            int_or_zero(item.get("local_concepts")),
            int_or_zero(item.get("relation_records_fetched")),
            int_or_zero(item.get("relation_records_processed")),
            int_or_zero(item.get("source_ui_lookups_attempted")),
            int_or_zero(item.get("source_ui_lookups_skipped")),
            int_or_zero(item.get("candidate_edges_retained")),
            item.get("status"),
        )
        for item in (stats.get("processed_cuis") or [])
        if isinstance(item, dict)
    ]
    lines.extend(
        markdown_count_table(
            [
                "index",
                "cui",
                "local_concepts",
                "relations_fetched",
                "relations_processed",
                "sourceUi_attempted",
                "sourceUi_skipped",
                "candidate_edges_so_far",
                "status",
            ],
            processed_rows,
        )
    )
    lines.extend(["", "## Skipped CUIs", ""])
    skipped_rows = [
        (item.get("cui"), item.get("reason"), item.get("local_concepts"))
        for item in (stats.get("skipped_cuis") or [])
        if isinstance(item, dict)
    ]
    lines.extend(
        markdown_count_table(["cui", "reason", "local_concepts"], skipped_rows)
    )
    lines.extend(["", "## CUIs Truncated By max_relations_per_cui", ""])
    relation_truncated_rows = [
        (
            item.get("cui"),
            item.get("limit"),
            item.get("fetched_records"),
            item.get("processed_records"),
            item.get("skipped_by_limit"),
        )
        for item in (stats.get("cuis_truncated_by_max_relations") or [])
        if isinstance(item, dict)
    ]
    lines.extend(
        markdown_count_table(
            ["cui", "limit", "fetched_records", "processed_records", "skipped_by_limit"],
            relation_truncated_rows,
        )
    )
    lines.extend(["", "## CUIs Truncated By max_source_ui_lookups_per_cui", ""])
    lookup_truncated_rows = [
        (
            item.get("cui"),
            item.get("limit"),
            item.get("attempted"),
            item.get("skipped"),
        )
        for item in (stats.get("cuis_truncated_by_source_ui_lookups") or [])
        if isinstance(item, dict)
    ]
    lines.extend(
        markdown_count_table(
            ["cui", "limit", "sourceUi_attempted", "sourceUi_skipped"],
            lookup_truncated_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Candidate Edges By Relation Label",
            "",
        ]
    )

    lines.extend(markdown_count_table(["relation_label", "candidate_edges"], relation_rows))
    lines.extend(
        [
            "",
            "## Candidate Edges By Source Type -> Relation -> Target Type",
            "",
        ]
    )
    lines.extend(
        markdown_count_table(
            ["source_type", "relation", "target_type", "candidate_edges"],
            type_relation_rows,
        )
    )
    lines.extend(["", "## Collapsed Connections By Relation Name", ""])
    lines.extend(
        markdown_count_table(
            ["relation_name", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_relation_name"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Traversal Policy", ""])
    lines.extend(
        markdown_count_table(
            ["traversal_policy", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_traversal_policy"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Compatibility Status", ""])
    lines.extend(
        markdown_count_table(
            ["compatibility_status", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_compatibility_status"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Compatibility Reason", ""])
    lines.extend(
        markdown_count_table(
            ["compatibility_reason", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_compatibility_reason"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Local Type Compatibility", ""])
    lines.extend(
        markdown_count_table(
            ["local_type_compatible", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_local_type_compatible"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Materialize-By-Default", ""])
    lines.extend(
        markdown_count_table(
            ["materialize_by_default", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_materialize_by_default"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Materialization Decision", ""])
    lines.extend(
        markdown_count_table(
            ["materialization_decision", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_materialization_decision"].items()),
        )
    )
    lines.extend(["", "## Collapsed Connections By Materialization Reason", ""])
    lines.extend(
        markdown_count_table(
            ["materialization_reason", "collapsed_connections"],
            sorted(collapsed_stats["counts_by_materialization_reason"].items()),
        )
    )
    graph_stat_rows = []
    for label, graph_stats in (
        ("candidate", collapsed_stats["candidate_graph_statistics"]),
        ("materializable", collapsed_stats["materializable_graph_statistics"]),
    ):
        graph_stat_rows.append(
            (
                label,
                graph_stats.get("unique_source_cuis"),
                graph_stats.get("unique_target_cuis"),
                graph_stats.get("unique_connected_cuis"),
                graph_stats.get("isolated_eligible_local_cui_count"),
                f"{graph_stats.get('cui_coverage_ratio', 0.0):.4f}",
                graph_stats.get("directed_edges"),
                graph_stats.get("unique_undirected_cui_pairs"),
                graph_stats.get("self_cui_edges"),
                graph_stats.get("minimum_degree"),
                graph_stats.get("median_degree"),
                f"{graph_stats.get('mean_degree', 0.0):.4f}",
                graph_stats.get("p95_degree"),
                graph_stats.get("maximum_degree"),
                graph_stats.get("weakly_connected_components"),
                graph_stats.get("largest_component_size"),
                f"{graph_stats.get('largest_component_coverage_ratio', 0.0):.4f}",
            )
        )
    lines.extend(["", "## Graph Coverage And Degree Statistics", ""])
    lines.extend(
        markdown_count_table(
            [
                "subset",
                "source_cuis",
                "target_cuis",
                "connected_cuis",
                "isolated_eligible",
                "coverage",
                "directed_edges",
                "undirected_pairs",
                "self_edges",
                "min_degree",
                "median_degree",
                "mean_degree",
                "p95_degree",
                "max_degree",
                "components",
                "largest_component",
                "largest_component_coverage",
            ],
            graph_stat_rows,
        )
    )
    top_degree_rows = [
        (
            item.get("cui"),
            item.get("representative_name"),
            item.get("canonical_type"),
            item.get("degree"),
            item.get("in_degree"),
            item.get("out_degree"),
        )
        for item in collapsed_stats["candidate_graph_statistics"].get(
            "top_degree_cui_representatives",
            [],
        )
    ]
    lines.extend(["", "## Top Candidate CUI Degrees", ""])
    lines.extend(
        markdown_count_table(
            ["cui", "representative_name", "canonical_type", "degree", "in_degree", "out_degree"],
            top_degree_rows,
        )
    )
    representative_type_rows = [
        (
            row["source_type"],
            row["relation_name"],
            row["target_type"],
            row["count"],
        )
        for row in collapsed_stats[
            "counts_by_source_representative_type_relation_target_representative_type"
        ]
    ]
    lines.extend(
        [
            "",
            "## Collapsed Connections By Representative Type -> Relation -> Representative Type",
            "",
        ]
    )
    lines.extend(
        markdown_count_table(
            [
                "source_representative_type",
                "relation_name",
                "target_representative_type",
                "collapsed_connections",
            ],
            representative_type_rows,
        )
    )
    relation_status_example_rows: list[tuple[Any, ...]] = []
    for group in collapsed_stats["examples_by_relation_and_compatibility_status"]:
        examples = group.get("examples") or []
        for item in examples:
            relation_status_example_rows.append(
                (
                    group.get("relation_name"),
                    group.get("compatibility_status"),
                    item.get("source_cui"),
                    item.get("source_type"),
                    item.get("target_type"),
                    item.get("target_cui"),
                    item.get("compatibility_reason"),
                    item.get("traversal_policy"),
                    item.get("materialization_decision"),
                    item.get("materialization_decision_reason"),
                    "; ".join(item.get("source_names") or []),
                    "; ".join(item.get("target_names") or []),
                )
            )
    lines.extend(
        [
            "",
            "## Representative Examples By Relation And Compatibility Status",
            "",
        ]
    )
    lines.extend(
        markdown_count_table(
            [
                "relation_name",
                "compatibility_status",
                "source_cui",
                "source_type",
                "target_type",
                "target_cui",
                "compatibility_reason",
                "traversal_policy",
                "materialization_decision",
                "materialization_reason",
                "example_source_names",
                "example_target_names",
            ],
            relation_status_example_rows,
        )
    )
    incompatible_example_rows = [
        (
            item.get("source_cui"),
            item.get("source_type"),
            item.get("relation_name"),
            item.get("target_type"),
            item.get("target_cui"),
            item.get("reason"),
            "; ".join(item.get("source_names") or []),
            "; ".join(item.get("target_names") or []),
        )
        for item in collapsed_stats["incompatible_local_type_examples"]
    ]
    lines.extend(["", "## Incompatible First-Extension Examples", ""])
    lines.extend(
        markdown_count_table(
            [
                "source_cui",
                "source_type",
                "relation_name",
                "target_type",
                "target_cui",
                "reason",
                "example_source_names",
                "example_target_names",
            ],
            incompatible_example_rows,
        )
    )
    lines.extend(["", "## Collapsed Candidate Connections", ""])
    lines.extend(
        markdown_count_table(
            [
                "source_cui",
                "target_cui",
                "relation_label",
                "relation_name",
                "source_representative_type",
                "target_representative_type",
                "local_type_compatible",
                "local_type_reason",
                "compatibility_status",
                "compatibility_reason",
                "relation_family",
                "traversal_policy",
                "materialize_by_default",
                "materialization_decision",
                "materialization_reason",
                "raw_rows",
                "relation_ids",
                "example_source_names",
                "example_target_names",
            ],
            collapsed_connection_rows[:50],
        )
    )

    if materialization_report is not None:
        lines.extend(["", "## Neo4j Materialization", ""])
        lines.extend(
            [
                f"- write_neo4j: `{str(bool(materialization_report.get('write_neo4j'))).lower()}`",
                f"- materialization_mode: `{materialization_report.get('materialization_mode', DEFAULT_MATERIALIZATION_MODE)}`",
                f"- replace_existing_connections: `{str(bool(materialization_report.get('replace_existing_connections'))).lower()}`",
                f"- relationships deleted before replacement write: {materialization_report.get('relationships_deleted_before_write', 0)}",
                f"- eligible collapsed edges: {materialization_report.get('eligible_collapsed_edges', 0)}",
                f"- relationships written/merged: {materialization_report.get('relationships_written', 0)}",
                f"- skipped missing representative: {materialization_report.get('skipped_missing_representative', 0)}",
                f"- skipped not whitelisted: {materialization_report.get('skipped_not_whitelisted', 0)}",
                f"- skipped incompatible local types: {materialization_report.get('skipped_incompatible_local_types', 0)}",
                f"- skipped not materialize-by-default: {materialization_report.get('skipped_not_materialize_by_default', 0)}",
                f"- duplicate edge_key findings after write: {materialization_report.get('duplicate_edge_key_findings', 0)}",
                f"- CUI mismatch findings after write: {materialization_report.get('cui_mismatch_findings', 0)}",
                f"- strict policy violation findings after write: {materialization_report.get('strict_policy_violation_findings', 0)}",
            ]
        )
        skipped_by_reason = (
            materialization_report.get("skipped_by_materialization_reason") or {}
        )
        lines.extend(["", "### Skipped By Materialization Reason", ""])
        lines.extend(
            markdown_count_table(
                ["materialization_reason", "collapsed_connections"],
                sorted(skipped_by_reason.items()),
            )
        )
        counts_by_type = materialization_report.get("counts_by_relationship_type") or {}
        lines.extend(["", "### Counts By UMLS Relationship Type", ""])
        lines.extend(
            markdown_count_table(
                ["relationship_type", "relationships"],
                sorted(counts_by_type.items()),
            )
        )

    lines.extend(["", "## Top Example Candidate Edges", ""])

    if not edges:
        lines.append("_No candidate edges retained._")
    else:
        for edge in edges[:20]:
            lines.append(
                "- "
                f"{edge['source_name']} [{edge['source_type']}, {edge['source_cui']}] "
                "-> "
                f"{edge['target_name']} [{edge['target_type']}, {edge['target_cui']}] "
                f"via {relation_display(edge)}"
            )

    failures = stats.get("failures") or []
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("_No CUI-level relation fetch failures._")
    else:
        lines.append(f"Total failures: {len(failures)}")
        lines.append("")
        for failure in failures[:20]:
            lines.append(
                f"- `{failure.get('cui', 'unknown')}`: {failure.get('error', '')}"
            )
        if len(failures) > 20:
            lines.append(f"- ... {len(failures) - 20} additional failures omitted")

    lines.append("")
    return "\n".join(lines)


def write_summary(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_review_exports(
    doc_id: Optional[str],
    source_vocab: str,
    concepts: Sequence[LocalConcept],
    edges: Sequence[dict[str, Any]],
    stats: dict[str, Any],
    client_stats: dict[str, int],
    csv_path: Path,
    summary_path: Path,
    cache_dir: Path,
    dry_run: bool,
    umls_version: str = DEFAULT_UMLS_VERSION,
    collapsed_json_path: Optional[Path] = None,
    collapsed_edges: Optional[Sequence[dict[str, Any]]] = None,
    materialization_report: Optional[dict[str, Any]] = None,
    materialization_json_path: Optional[Path] = None,
    statistics_json_path: Optional[Path] = None,
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
) -> None:
    materialization_mode = normalize_materialization_mode(materialization_mode)
    if collapsed_edges is None:
        representatives = select_representative_concepts(concepts)
        collapsed_edges = build_collapsed_connections(
            edges=edges,
            doc_id=doc_id,
            source_vocab=source_vocab,
            umls_version=umls_version,
            representatives_by_cui=representatives,
            materialization_mode=materialization_mode,
        )

    write_candidate_edges_csv(csv_path, edges)
    if collapsed_json_path is not None:
        write_json(collapsed_json_path, list(collapsed_edges))
    if materialization_report is not None and materialization_json_path is not None:
        write_json(materialization_json_path, materialization_report)
    if statistics_json_path is not None:
        write_json(
            statistics_json_path,
            {
                "doc_id": doc_id,
                "source_vocab": source_vocab,
                "umls_version": umls_version,
                "dry_run": dry_run,
                "materialization_mode": materialization_mode,
                "stats": stats,
                "client_stats": client_stats,
                "candidate_edges": len(edges),
                "collapsed_connections": len(collapsed_edges),
                "materialization_report": materialization_report,
            },
        )

    summary = build_summary_markdown(
        doc_id=doc_id,
        source_vocab=source_vocab,
        concepts=concepts,
        edges=edges,
        stats=stats,
        client_stats=client_stats,
        csv_path=csv_path,
        collapsed_json_path=collapsed_json_path,
        cache_dir=cache_dir,
        dry_run=dry_run,
        umls_version=umls_version,
        collapsed_edges=collapsed_edges,
        materialization_report=materialization_report,
        materialization_mode=materialization_mode,
    )
    write_summary(summary_path, summary)


def fetch_local_concepts(
    driver: Driver,
    doc_id: Optional[str],
) -> list[LocalConcept]:
    with driver.session() as session:
        return session.execute_read(fetch_local_concepts_for_doc, doc_id)


def delete_existing_umls_connections_for_doc(tx, doc_id: str) -> int:
    result = tx.run(
        """
        MATCH (:Concept)-[r]->(:Concept)
        WHERE type(r) IN $relationship_types
          AND properties(r)['provenance'] = 'umls_connections'
          AND properties(r)['doc_id'] = $doc_id
        WITH collect(r) AS relationships, count(r) AS deleted_count
        FOREACH (relationship IN relationships | DELETE relationship)
        RETURN deleted_count
        """,
        relationship_types=UMLS_CONNECTION_RELATION_TYPES,
        doc_id=clean_text(doc_id),
    ).single()
    if result is None:
        return 0
    return int_or_zero(result.get("deleted_count"))


def materialize_collapsed_edge(tx, edge: dict[str, Any], now: str) -> Optional[str]:
    relation_name = normalize_relation_term(edge.get("relation_name"))
    relationship_type = (
        clean_text(edge.get("relationship_type"))
        or RELATION_TYPE_BY_NAME.get(relation_name)
    )
    if relationship_type is None:
        return None

    source_representative = edge.get("source_representative") or {}
    target_representative = edge.get("target_representative") or {}
    source_concept_id = clean_text(source_representative.get("concept_id"))
    target_concept_id = clean_text(target_representative.get("concept_id"))
    if not source_concept_id or not target_concept_id:
        return None

    relationship_metadata = build_ontology_relationship_metadata(
        clean_text(edge.get("source_vocabulary"))
    )

    query = f"""
        MATCH (source:Concept)
        WHERE elementId(source) = $source_concept_id
        MATCH (target:Concept)
        WHERE elementId(target) = $target_concept_id
        MERGE (source)-[r:{relationship_type} {{edge_key: $edge_key}}]->(target)
        ON CREATE SET r.created_at = datetime($now)
        SET r.doc_id = $doc_id,
            r.source_cui = $source_cui,
            r.target_cui = $target_cui,
            r.relation_label = $relation_label,
            r.relation_name = $relation_name,
            r.source_vocabulary = $source_vocabulary,
            r.umls_version = $umls_version,
            r.raw_rows = $raw_rows,
            r.relation_ids = $relation_ids,
            r.source_names = $source_names,
            r.target_names = $target_names,
            r.source_types = $source_types,
            r.target_types = $target_types,
            r.status = $status,
            r.provenance = $provenance,
            r.relationship_family = $relationship_family,
            r.provenance_source = $provenance_source,
            r.provenance_method = $provenance_method,
            r.compatibility_status = $compatibility_status,
            r.compatibility_reason = $compatibility_reason,
            r.local_type_compatible = $local_type_compatible,
            r.local_type_compatibility_reason = $local_type_compatibility_reason,
            r.relation_family = $relation_family,
            r.materialize_by_default = $materialize_by_default,
            r.traversal_policy = $traversal_policy,
            r.review_needed = $review_needed,
            r.materialization_mode = $materialization_mode,
            r.materialization_decision = $materialization_decision,
            r.materialization_decision_reason = $materialization_decision_reason,
            r.updated_at = datetime($now)
        RETURN elementId(r) AS relationship_id
    """
    record = tx.run(
        query,
        edge_key=edge["edge_key"],
        doc_id=clean_text(edge.get("doc_id")),
        source_cui=normalize_cui(edge.get("source_cui")),
        target_cui=normalize_cui(edge.get("target_cui")),
        relation_label=clean_text(edge.get("relation_label")),
        relation_name=relation_name,
        source_vocabulary=clean_text(edge.get("source_vocabulary")),
        umls_version=clean_text(edge.get("umls_version")),
        raw_rows=int_or_zero(edge.get("raw_rows")),
        relation_ids=list(edge.get("relation_ids") or []),
        source_names=list(edge.get("source_names") or []),
        target_names=list(edge.get("target_names") or []),
        source_types=list(edge.get("source_types") or []),
        target_types=list(edge.get("target_types") or []),
        status=CONNECTION_STATUS,
        provenance=clean_text(relationship_metadata["provenance"]),
        relationship_family=clean_text(relationship_metadata["relationship_family"]),
        provenance_source=clean_text(relationship_metadata["provenance_source"]),
        provenance_method=clean_text(relationship_metadata["provenance_method"]),
        compatibility_status=clean_text(edge.get("compatibility_status")),
        compatibility_reason=clean_text(edge.get("compatibility_reason")),
        local_type_compatible=bool(edge.get("local_type_compatible")),
        local_type_compatibility_reason=clean_text(
            edge.get("local_type_compatibility_reason")
            or edge.get("compatibility_reason")
        ),
        relation_family=clean_text(edge.get("relation_family")),
        materialize_by_default=bool(edge.get("materialize_by_default")),
        traversal_policy=clean_text(edge.get("traversal_policy")),
        review_needed=bool(edge.get("review_needed")),
        materialization_mode=clean_text(edge.get("materialization_mode")),
        materialization_decision=bool(edge.get("materialization_decision")),
        materialization_decision_reason=clean_text(
            edge.get("materialization_decision_reason")
        ),
        now=now,
        source_concept_id=source_concept_id,
        target_concept_id=target_concept_id,
    ).single()
    if record is None:
        return None
    return clean_text(record["relationship_id"])


def fetch_umls_connection_write_sanity(tx) -> dict[str, Any]:
    relationship_types = UMLS_CONNECTION_RELATION_TYPES

    duplicate_rows = [
        dict(row)
        for row in tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH properties(r) AS rel_props,
                 type(r) AS relationship_type
            WITH rel_props['edge_key'] AS edge_key,
                 count(*) AS n,
                 collect(DISTINCT relationship_type) AS relationship_types
            WHERE edge_key IS NULL OR trim(toString(edge_key)) = '' OR n > 1
            RETURN edge_key, n, relationship_types
            ORDER BY n DESC, edge_key
            """,
            relationship_types=relationship_types,
        )
    ]

    mismatch_rows = [
        dict(row)
        for row in tx.run(
            """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target,
                 properties(source) AS source_props,
                 properties(target) AS target_props,
                 properties(r) AS rel_props
            WHERE toUpper(coalesce(toString(source_props['umls_cui']), '')) <>
                  toUpper(coalesce(toString(rel_props['source_cui']), ''))
               OR toUpper(coalesce(toString(target_props['umls_cui']), '')) <>
                  toUpper(coalesce(toString(rel_props['target_cui']), ''))
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   source.name AS source_name,
                   target.name AS target_name,
                   source_props['umls_cui'] AS source_node_cui,
                   rel_props['source_cui'] AS relationship_source_cui,
                   target_props['umls_cui'] AS target_node_cui,
                   rel_props['target_cui'] AS relationship_target_cui
            ORDER BY relationship_type, edge_key
            """,
            relationship_types=relationship_types,
        )
    ]

    strict_policy_violation_rows = [
        dict(row)
        for row in tx.run(
            """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target, properties(r) AS rel_props
            WHERE coalesce(toString(rel_props['materialization_mode']), '') <> 'safe_only'
               OR coalesce(rel_props['materialization_decision'], false) <> true
               OR coalesce(toString(rel_props['compatibility_status']), '') <> 'compatible'
               OR coalesce(rel_props['local_type_compatible'], false) <> true
               OR coalesce(rel_props['review_needed'], true) <> false
               OR NOT (coalesce(toString(rel_props['traversal_policy']), '') IN ['safe', 'hierarchy'])
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   source.name AS source_name,
                   target.name AS target_name,
                   rel_props['materialization_mode'] AS materialization_mode,
                   rel_props['materialization_decision'] AS materialization_decision,
                   rel_props['compatibility_status'] AS compatibility_status,
                   rel_props['local_type_compatible'] AS local_type_compatible,
                   rel_props['review_needed'] AS review_needed,
                   rel_props['traversal_policy'] AS traversal_policy
            ORDER BY relationship_type, edge_key
            """,
            relationship_types=relationship_types,
        )
    ]

    count_rows = [
        dict(row)
        for row in tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            RETURN type(r) AS relationship_type, count(*) AS n
            ORDER BY relationship_type
            """,
            relationship_types=relationship_types,
        )
    ]

    return {
        "duplicate_edge_keys": duplicate_rows,
        "cui_mismatches": mismatch_rows,
        "strict_policy_violations": strict_policy_violation_rows,
        "counts_by_relationship_type": {
            clean_text(row.get("relationship_type")): int_or_zero(row.get("n"))
            for row in count_rows
        },
    }


def materialize_collapsed_connections(
    driver: Driver,
    collapsed_edges: Sequence[dict[str, Any]],
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
    *,
    doc_id: Optional[str] = None,
    replace_existing_connections: bool = False,
) -> dict[str, Any]:
    materialization_mode = normalize_materialization_mode(materialization_mode)
    if replace_existing_connections and materialization_mode != "safe_only":
        raise ValueError(
            "replace_existing_connections=true requires materialization_mode=safe_only"
        )
    if replace_existing_connections and not clean_text(doc_id):
        raise ValueError("replace_existing_connections=true requires a selected doc_id")

    now = utc_now_iso()
    report_edges: list[dict[str, Any]] = []
    representatives: dict[str, dict[str, Any]] = {}
    skipped_not_whitelisted = 0
    skipped_missing_representative = 0
    skipped_incompatible_local_types = 0
    skipped_not_materialize_by_default = 0
    skipped_by_materialization_reason: Counter[str] = Counter()
    eligible_collapsed_edges = 0
    relationships_written = 0
    relationships_deleted_before_write = 0

    if materialization_mode == "none":
        for edge in collapsed_edges:
            decision, reason = should_materialize_relation(
                edge,
                materialization_mode=materialization_mode,
            )
            skipped_by_materialization_reason[reason] += 1
            report_edges.append(
                {
                    **edge,
                    "materialization_mode": materialization_mode,
                    "materialization_decision": decision,
                    "materialization_decision_reason": reason,
                    "materialization_status": f"skipped_{reason}",
                }
            )
        collapsed_connection_statistics = build_collapsed_connection_statistics(
            report_edges
        )
        return {
            "write_neo4j": False,
            "written_at": now,
            "materialization_mode": materialization_mode,
            "replace_existing_connections": False,
            "relationships_deleted_before_write": 0,
            "collapsed_edges": len(collapsed_edges),
            "eligible_collapsed_edges": 0,
            "relationships_written": 0,
            "skipped_not_whitelisted": 0,
            "skipped_missing_representative": 0,
            "skipped_incompatible_local_types": 0,
            "skipped_not_materialize_by_default": 0,
            "skipped_by_materialization_reason": dict(
                sorted(skipped_by_materialization_reason.items())
            ),
            "duplicate_edge_key_findings": 0,
            "cui_mismatch_findings": 0,
            "counts_by_relationship_type": {},
            **collapsed_connection_statistics,
            "collapsed_connection_statistics": collapsed_connection_statistics,
            "representatives": representatives,
            "relationships": report_edges,
            "sanity": {},
        }

    eligible_report_indexes: list[int] = []
    eligible_edges: list[dict[str, Any]] = []
    for edge in collapsed_edges:
        source_representative = edge.get("source_representative")
        target_representative = edge.get("target_representative")
        decision, reason = should_materialize_relation(
            edge,
            materialization_mode=materialization_mode,
        )
        edge_for_report = {
            **edge,
            "materialization_mode": materialization_mode,
            "materialization_decision": decision,
            "materialization_decision_reason": reason,
        }
        if not decision:
            skipped_by_materialization_reason[reason] += 1
            if reason == "unsupported_relation":
                skipped_not_whitelisted += 1
            elif reason == "missing_representative":
                skipped_missing_representative += 1
            elif reason == "not_materialize_by_default":
                skipped_not_materialize_by_default += 1
            elif (
                reason.startswith("compatibility_status_")
                or reason == "local_type_not_compatible"
                or reason.startswith("traversal_policy_")
                or reason == "review_needed"
            ):
                skipped_incompatible_local_types += 1
            report_edges.append(
                {
                    **edge_for_report,
                    "materialization_status": f"skipped_{reason}",
                }
            )
            continue

        if not source_representative or not target_representative:
            skipped_missing_representative += 1
            report_edges.append(
                {
                    **edge_for_report,
                    "materialization_status": "skipped_missing_representative",
                }
            )
            continue

        eligible_collapsed_edges += 1
        representatives[edge["source_cui"]] = source_representative
        representatives[edge["target_cui"]] = target_representative
        eligible_edges.append(edge_for_report)
        eligible_report_indexes.append(len(report_edges))
        report_edges.append(
            {
                **edge_for_report,
                "materialization_status": "pending_write",
            }
        )

    with driver.session() as session:
        if replace_existing_connections:
            relationships_deleted_before_write = session.execute_write(
                delete_existing_umls_connections_for_doc,
                clean_text(doc_id),
            )

        for edge_for_report, report_index in zip(eligible_edges, eligible_report_indexes):
            relationship_id = session.execute_write(
                materialize_collapsed_edge,
                edge_for_report,
                now,
            )
            if relationship_id:
                relationships_written += 1
                status = "merged"
            else:
                status = "skipped_missing_concept"
                skipped_missing_representative += 1

            report_edges[report_index] = (
                {
                    **report_edges[report_index],
                    "materialization_status": status,
                    "materialized_relationship_id": relationship_id,
                }
            )

        sanity = session.execute_read(fetch_umls_connection_write_sanity)

    duplicate_findings = len(sanity.get("duplicate_edge_keys") or [])
    mismatch_findings = len(sanity.get("cui_mismatches") or [])
    collapsed_connection_statistics = build_collapsed_connection_statistics(
        report_edges
    )

    return {
        "write_neo4j": True,
        "written_at": now,
        "materialization_mode": materialization_mode,
        "replace_existing_connections": bool(replace_existing_connections),
        "relationships_deleted_before_write": relationships_deleted_before_write,
        "collapsed_edges": len(collapsed_edges),
        "eligible_collapsed_edges": eligible_collapsed_edges,
        "relationships_written": relationships_written,
        "skipped_not_whitelisted": skipped_not_whitelisted,
        "skipped_missing_representative": skipped_missing_representative,
        "skipped_incompatible_local_types": skipped_incompatible_local_types,
        "skipped_not_materialize_by_default": skipped_not_materialize_by_default,
        "skipped_by_materialization_reason": dict(
            sorted(skipped_by_materialization_reason.items())
        ),
        "duplicate_edge_key_findings": duplicate_findings,
        "cui_mismatch_findings": mismatch_findings,
        "strict_policy_violation_findings": len(
            sanity.get("strict_policy_violations") or []
        ),
        "counts_by_relationship_type": sanity.get("counts_by_relationship_type", {}),
        **collapsed_connection_statistics,
        "collapsed_connection_statistics": collapsed_connection_statistics,
        "representatives": representatives,
        "relationships": report_edges,
        "sanity": sanity,
    }


def run_umls_connections(
    doc_id: Optional[str],
    source_vocab: str = DEFAULT_SOURCE_VOCAB,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_dir: Path = DEFAULT_RELATION_CACHE_DIR,
    run_name: Optional[str] = None,
    dry_run: bool = True,
    write_neo4j: bool = False,
    replace_existing_connections: bool = False,
    materialization_mode: str = DEFAULT_MATERIALIZATION_MODE,
    driver: Optional[Driver] = None,
    api_timeout: float = DEFAULT_API_TIMEOUT,
    api_rate_limit_per_second: float = DEFAULT_API_RATE_LIMIT_PER_SECOND,
    umls_version: str = DEFAULT_UMLS_VERSION,
    api_page_size: int = DEFAULT_RELATION_PAGE_SIZE,
    max_cuis: Optional[int] = None,
    include_cuis: Optional[Sequence[str]] = None,
    skip_cuis: Optional[Sequence[str]] = None,
    max_relations_per_cui: int = DEFAULT_MAX_RELATIONS_PER_CUI,
    max_source_ui_lookups_per_cui: int = DEFAULT_MAX_SOURCE_UI_LOOKUPS_PER_CUI,
    write_partial_every: int = DEFAULT_WRITE_PARTIAL_EVERY,
    include_relation_names: Optional[Sequence[str]] = None,
    exclude_relation_names: Optional[Sequence[str]] = None,
    strong_relations_only: bool = False,
    relation_profile: Optional[str] = None,
    ignore_negative_cache: bool = False,
) -> dict[str, Any]:
    materialization_mode = normalize_materialization_mode(materialization_mode)
    if max_cuis is not None and max_cuis < 0:
        raise ValueError("max_cuis must be >= 0")
    if max_relations_per_cui < 1:
        raise ValueError("max_relations_per_cui must be >= 1")
    if max_source_ui_lookups_per_cui < 0:
        raise ValueError("max_source_ui_lookups_per_cui must be >= 0")
    if write_partial_every < 0:
        raise ValueError("write_partial_every must be >= 0")
    (
        resolved_include_names,
        resolved_exclude_names,
        selected_relation_profile,
    ) = resolve_relation_name_filters(
        include_relation_names=include_relation_names,
        exclude_relation_names=exclude_relation_names,
        strong_relations_only=strong_relations_only,
        relation_profile=relation_profile,
    )
    write_neo4j = bool(write_neo4j)
    replace_existing_connections = bool(replace_existing_connections)
    if write_neo4j and selected_relation_profile == "audit_all":
        raise ValueError(
            "relation_profile=audit_all is export-only and cannot write to Neo4j"
        )
    if write_neo4j and materialization_mode != "safe_only":
        raise ValueError(
            "write_neo4j=true requires materialization_mode=safe_only; "
            "the only supported write policy is safe_only"
        )
    if replace_existing_connections and not write_neo4j:
        raise ValueError(
            "replace_existing_connections=true requires write_neo4j=true "
            "and materialization_mode=safe_only"
        )
    if replace_existing_connections and materialization_mode != "safe_only":
        raise ValueError(
            "replace_existing_connections=true requires materialization_mode=safe_only"
        )
    if replace_existing_connections and not clean_text(doc_id):
        raise ValueError("replace_existing_connections=true requires a selected doc_id")

    dry_run = not write_neo4j
    normalized_include_cuis = sorted(
        {normalize_cui(cui) for cui in (include_cuis or []) if normalize_cui(cui)}
    )
    normalized_skip_cuis = sorted(
        {normalize_cui(cui) for cui in (skip_cuis or []) if normalize_cui(cui)}
    )
    resolved_run_name = resolve_run_name(
        run_name=run_name,
        relation_profile=selected_relation_profile,
        materialization_mode=materialization_mode,
        max_relations_per_cui=max_relations_per_cui,
        max_source_ui_lookups_per_cui=max_source_ui_lookups_per_cui,
    )
    run_output_dir = resolve_run_output_dir(Path(output_dir), resolved_run_name)
    if clean_text(run_name) and safe_filename_component(run_name) != clean_text(run_name):
        logger.info(
            "Sanitized UMLS connection run_name | requested=%s | resolved=%s",
            run_name,
            resolved_run_name,
        )

    owns_driver = driver is None
    try:
        if driver is None:
            driver = get_neo4j_driver(verify=True)
        concepts = fetch_local_concepts(driver, doc_id=doc_id)
    finally:
        if owns_driver and driver is not None:
            close_driver(driver)
            driver = None

    logger.info(
        "Local UMLS-matched concepts selected | doc_id=%s | concepts=%d | unique_cuis=%d",
        doc_id or "ALL",
        len(concepts),
        len({concept.umls_cui for concept in concepts}),
    )

    csv_path, summary_path = output_paths(doc_id=doc_id, output_dir=run_output_dir)
    collapsed_json_path = collapsed_connections_path(
        doc_id=doc_id,
        output_dir=run_output_dir,
    )
    materialization_json_path = materialization_report_path(
        doc_id=doc_id,
        output_dir=run_output_dir,
    )
    statistics_json_path = statistics_report_path(
        doc_id=doc_id,
        output_dir=run_output_dir,
    )
    edges: list[dict[str, Any]] = []
    relation_stats: dict[str, Any] = {
        "total_local_cuis": len({concept.umls_cui for concept in concepts}),
        "eligible_local_cuis": sorted({concept.umls_cui for concept in concepts}),
        "eligible_local_cui_count": len({concept.umls_cui for concept in concepts}),
        "source_cuis_selected": 0,
        "selected_source_cuis": [],
        "include_cuis": normalized_include_cuis,
        "skip_cuis": normalized_skip_cuis,
        "processed_cuis": [],
        "skipped_cuis": [],
        "max_cuis": max_cuis,
        "max_relations_per_cui": max_relations_per_cui,
        "max_source_ui_lookups_per_cui": max_source_ui_lookups_per_cui,
        "write_partial_every": write_partial_every,
        "include_relation_names": sorted(resolved_include_names),
        "exclude_relation_names": sorted(resolved_exclude_names),
        "strong_relations_only": strong_relations_only,
        "relation_profile": selected_relation_profile,
        "selected_relation_profile": selected_relation_profile,
        "run_name": resolved_run_name,
        "output_dir": str(run_output_dir),
        "statistics_json_path": str(statistics_json_path),
        "write_neo4j": write_neo4j,
        "replace_existing_connections": replace_existing_connections,
        "materialization_mode": materialization_mode,
        "ignore_negative_cache": ignore_negative_cache,
        "partial_exports_written": 0,
        "last_partial_export_processed_cuis": 0,
        "final_export_written": False,
        "umls_relation_records_fetched": 0,
        "umls_relation_records_processed": 0,
        "relation_records_skipped_by_limit": 0,
        "relation_records_skipped_by_relation_name_filter": 0,
        "external_target_relations_skipped": 0,
        "same_cui_relations_skipped": 0,
        "equivalence_relations_skipped": 0,
        "unresolved_target_relations_skipped": 0,
        "relation_fetch_failures": 0,
        "relation_fetches_unavailable": 0,
        "relation_fetches_skipped_by_negative_cache": 0,
        "source_vocab_mismatch_relations_skipped": 0,
        "internal_candidate_edges_retained": 0,
        "source_ui_lookups_attempted": 0,
        "source_ui_lookups_skipped_by_limit": 0,
        "cuis_truncated_by_max_relations": [],
        "cuis_truncated_by_source_ui_lookups": [],
        "failures": [],
    }
    client_stats: dict[str, int] = {
        "api_cache_hits": 0,
        "api_cache_misses": 0,
        "api_requests": 0,
        "api_retries": 0,
        "api_errors": 0,
        "relation_negative_cache_hits": 0,
        "relation_negative_cache_writes": 0,
    }

    if concepts:
        client = UMLSRelationsClient(
            cache_dir=cache_dir,
            timeout=api_timeout,
            rate_limit_per_second=api_rate_limit_per_second,
            version=umls_version,
            page_size=api_page_size,
            ignore_negative_cache=ignore_negative_cache,
        )
        edges, relation_stats = build_candidate_edges(
            doc_id=doc_id,
            concepts=concepts,
            client=client,
            source_vocab=source_vocab,
            max_cuis=max_cuis,
            include_cuis=include_cuis,
            skip_cuis=skip_cuis,
            max_relations_per_cui=max_relations_per_cui,
            max_source_ui_lookups_per_cui=max_source_ui_lookups_per_cui,
            write_partial_every=write_partial_every,
            partial_csv_path=csv_path,
            partial_summary_path=summary_path,
            cache_dir=Path(cache_dir),
            dry_run=dry_run,
            include_relation_names=include_relation_names,
            exclude_relation_names=exclude_relation_names,
            strong_relations_only=strong_relations_only,
            relation_profile=relation_profile,
            materialization_mode=materialization_mode,
        )
        client_stats = dict(client.stats)
    else:
        logger.warning("No eligible local UMLS-matched concepts found for doc_id=%s", doc_id)

    relation_stats.update(
        {
            "run_name": resolved_run_name,
            "output_dir": str(run_output_dir),
            "statistics_json_path": str(statistics_json_path),
            "write_neo4j": write_neo4j,
            "replace_existing_connections": replace_existing_connections,
            "include_cuis": normalized_include_cuis,
            "skip_cuis": normalized_skip_cuis,
        }
    )

    representatives = select_representative_concepts(concepts)
    collapsed_edges = build_collapsed_connections(
        edges=edges,
        doc_id=doc_id,
        source_vocab=source_vocab,
        umls_version=umls_version,
        representatives_by_cui=representatives,
        materialization_mode=materialization_mode,
    )
    collapsed_connection_statistics = build_collapsed_connection_statistics(
        collapsed_edges,
        eligible_cuis=relation_stats.get("eligible_local_cuis") or [],
        selected_source_cuis=relation_stats.get("selected_source_cuis") or [],
        representatives_by_cui=representatives,
    )
    relation_stats.update(collapsed_connection_statistics)
    relation_stats["collapsed_connection_statistics"] = collapsed_connection_statistics
    materialization_report: Optional[dict[str, Any]] = None

    try:
        if write_neo4j:
            if driver is None:
                driver = get_neo4j_driver(verify=True)
                owns_driver = True
            logger.info(
                "Materializing collapsed UMLS connections to Neo4j | collapsed_edges=%d",
                len(collapsed_edges),
            )
            materialization_report = materialize_collapsed_connections(
                driver=driver,
                collapsed_edges=collapsed_edges,
                materialization_mode=materialization_mode,
                doc_id=doc_id,
                replace_existing_connections=replace_existing_connections,
            )
            materialization_report["selected_relation_profile"] = (
                selected_relation_profile
            )
            materialization_report["relation_profile"] = selected_relation_profile
            materialization_report["materialization_mode"] = materialization_mode
            materialization_report["replace_existing_connections"] = (
                replace_existing_connections
            )
            materialization_report["relation_stats"] = relation_stats

        relation_stats["final_export_written"] = True
        write_review_exports(
            doc_id=doc_id,
            source_vocab=source_vocab,
            concepts=concepts,
            edges=edges,
            stats=relation_stats,
            client_stats=client_stats,
            csv_path=csv_path,
            summary_path=summary_path,
            cache_dir=cache_dir,
            dry_run=dry_run,
            umls_version=umls_version,
            collapsed_json_path=collapsed_json_path,
            collapsed_edges=collapsed_edges,
            materialization_report=materialization_report,
            materialization_json_path=(
                materialization_json_path if materialization_report is not None else None
            ),
            statistics_json_path=statistics_json_path,
            materialization_mode=materialization_mode,
        )
    finally:
        if owns_driver and driver is not None:
            close_driver(driver)

    logger.info(
        "UMLS connection review exports written | edges=%d | collapsed_edges=%d | csv=%s | summary=%s",
        len(edges),
        len(collapsed_edges),
        csv_path,
        summary_path,
    )

    return {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "collapsed_json_path": collapsed_json_path,
        "statistics_json_path": statistics_json_path,
        "run_name": resolved_run_name,
        "output_dir": run_output_dir,
        "materialization_report_path": (
            materialization_json_path if materialization_report is not None else None
        ),
        "cache_dir": Path(cache_dir),
        "concepts_considered": len(concepts),
        "unique_cuis_considered": len({concept.umls_cui for concept in concepts}),
        "candidate_edges": len(edges),
        "collapsed_connections": len(collapsed_edges),
        "relation_stats": relation_stats,
        "client_stats": client_stats,
        "materialization_report": materialization_report,
        "writes_to_neo4j": bool(write_neo4j),
        "materialization_mode": materialization_mode,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "UMLS/SNOMED candidate connection export for local UMLS-matched "
            "Concept nodes. Defaults to read-only export."
        )
    )
    parser.add_argument("--doc-id", required=True, help="Document id to inspect")
    parser.add_argument(
        "--source-vocab",
        default=DEFAULT_SOURCE_VOCAB,
        help=f"UMLS source vocabulary for relation lookup (default: {DEFAULT_SOURCE_VOCAB})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/Markdown exports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_RELATION_CACHE_DIR,
        help=f"Directory for cached UMLS relation responses (default: {DEFAULT_RELATION_CACHE_DIR})",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Experiment run name. When omitted, a deterministic name is derived "
            "from profile, materialization mode, and relation lookup limits."
        ),
    )
    parser.add_argument(
        "--ignore-negative-cache",
        action="store_true",
        help=(
            "Bypass cached UMLS relation 404-unavailable markers and attempt "
            "the /relations API call anyway."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Accepted for clarity. Read-only export is the default.",
    )
    parser.add_argument(
        "--write-neo4j",
        action="store_true",
        help=(
            "Materialize strict safe_only collapsed candidate relations directly "
            "between representative Concept nodes. Defaults to off."
        ),
    )
    parser.add_argument(
        "--replace-existing-connections",
        action="store_true",
        help=(
            "Before a safe_only write, delete existing UMLS connection edges for "
            "this doc_id that were produced by this module."
        ),
    )
    parser.add_argument(
        "--materialization-mode",
        choices=MATERIALIZATION_MODE_CHOICES,
        default=DEFAULT_MATERIALIZATION_MODE,
        help=(
            "Materialization decision mode: none or safe_only "
            f"(default: {DEFAULT_MATERIALIZATION_MODE})."
        ),
    )
    parser.add_argument(
        "--umls-version",
        default=DEFAULT_UMLS_VERSION,
        help=f"UMLS API version (default: {DEFAULT_UMLS_VERSION})",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=DEFAULT_API_TIMEOUT,
        help=f"UMLS API request timeout in seconds (default: {DEFAULT_API_TIMEOUT})",
    )
    parser.add_argument(
        "--api-rate-limit-per-second",
        type=float,
        default=DEFAULT_API_RATE_LIMIT_PER_SECOND,
        help=(
            "Maximum UMLS API requests per second "
            f"(default: {DEFAULT_API_RATE_LIMIT_PER_SECOND})"
        ),
    )
    parser.add_argument(
        "--api-page-size",
        type=int,
        default=DEFAULT_RELATION_PAGE_SIZE,
        help=f"UMLS relation page size (default: {DEFAULT_RELATION_PAGE_SIZE})",
    )
    parser.add_argument(
        "--max-cuis",
        type=int,
        default=None,
        help="Process only the first N sorted local CUIs after explicit skips.",
    )
    parser.add_argument(
        "--skip-cui",
        action="append",
        default=[],
        help="Skip a source CUI. May be repeated, for example: --skip-cui C0013520.",
    )
    parser.add_argument(
        "--include-cui",
        action="append",
        default=[],
        help=(
            "Restrict source processing to this CUI before skips and max_cuis. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--max-relations-per-cui",
        type=int,
        default=DEFAULT_MAX_RELATIONS_PER_CUI,
        help=(
            "Maximum UMLS relation records to process for one source CUI "
            f"(default: {DEFAULT_MAX_RELATIONS_PER_CUI})"
        ),
    )
    parser.add_argument(
        "--max-source-ui-lookups-per-cui",
        type=int,
        default=DEFAULT_MAX_SOURCE_UI_LOOKUPS_PER_CUI,
        help=(
            "Maximum sourceUi target lookups to attempt for one source CUI "
            f"(default: {DEFAULT_MAX_SOURCE_UI_LOOKUPS_PER_CUI})"
        ),
    )
    parser.add_argument(
        "--include-relation-name",
        action="append",
        default=[],
        help=(
            "Keep only records whose normalized additionalRelationLabel matches "
            "this relation name. May be repeated."
        ),
    )
    parser.add_argument(
        "--exclude-relation-name",
        action="append",
        default=[],
        help=(
            "Exclude records whose normalized additionalRelationLabel matches "
            "this relation name. May be repeated."
        ),
    )
    parser.add_argument(
        "--relation-profile",
        choices=RELATION_PROFILE_CHOICES,
        default=None,
        help=(
            "Initial SNOMED relation include profile. Explicit include names "
            "are added and excludes are then removed."
        ),
    )
    parser.add_argument(
        "--strong-relations-only",
        action="store_true",
        help=(
            "Backward-compatible shorthand for --relation-profile expanded."
        ),
    )
    parser.add_argument(
        "--write-partial-every",
        type=int,
        default=DEFAULT_WRITE_PARTIAL_EVERY,
        help=(
            "Write partial CSV and summary every N processed CUIs; use 0 to disable "
            f"(default: {DEFAULT_WRITE_PARTIAL_EVERY})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        result = run_umls_connections(
            doc_id=args.doc_id,
            source_vocab=args.source_vocab,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            run_name=args.run_name,
            dry_run=not args.write_neo4j,
            write_neo4j=args.write_neo4j,
            replace_existing_connections=args.replace_existing_connections,
            materialization_mode=args.materialization_mode,
            api_timeout=args.api_timeout,
            api_rate_limit_per_second=args.api_rate_limit_per_second,
            umls_version=args.umls_version,
            api_page_size=args.api_page_size,
            max_cuis=args.max_cuis,
            include_cuis=args.include_cui,
            skip_cuis=args.skip_cui,
            max_relations_per_cui=args.max_relations_per_cui,
            max_source_ui_lookups_per_cui=args.max_source_ui_lookups_per_cui,
            write_partial_every=args.write_partial_every,
            include_relation_names=args.include_relation_name,
            exclude_relation_names=args.exclude_relation_name,
            strong_relations_only=args.strong_relations_only,
            relation_profile=args.relation_profile,
            ignore_negative_cache=args.ignore_negative_cache,
        )
    except ValueError as e:
        parser.error(str(e))

    logger.info(
        "Completed UMLS connection export | candidate_edges=%d | collapsed_connections=%d | writes_to_neo4j=%s",
        result["candidate_edges"],
        result["collapsed_connections"],
        result["writes_to_neo4j"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LocalConcept",
    "RelationSpec",
    "CompatibilityResult",
    "RELATION_SPECS",
    "MATERIALIZATION_MODE_CHOICES",
    "DEFAULT_MATERIALIZATION_MODE",
    "CORE_SNOMED_RELATION_NAMES",
    "FIRST_SNOMED_EXTENSION_RELATION_NAMES",
    "EXPANDED_SNOMED_RELATION_NAMES",
    "BALANCED_CORE_SNOMED_RELATION_NAMES",
    "AUDIT_ONLY_RELATION_NAMES",
    "AUDIT_ALL_SNOMED_RELATION_NAMES",
    "STRONG_RELATION_NAMES",
    "RELATION_NAMES_BY_PROFILE",
    "RELATION_TYPE_BY_NAME",
    "UMLS_CONNECTION_RELATION_TYPES",
    "LOCAL_TYPE_RULES_BY_RELATION_NAME",
    "FIRST_EXTENSION_SAFE_RELATION_NAMES",
    "FIRST_EXTENSION_REVERSE_REVIEW_RELATION_NAMES",
    "UMLSRelationsClient",
    "build_collapsed_connections",
    "build_collapsed_connection_statistics",
    "compute_cui_graph_statistics",
    "catalog_local_type_rule_rows",
    "catalog_relationship_type_rows",
    "delete_existing_umls_connections_for_doc",
    "evaluate_relation_compatibility",
    "evaluate_local_type_compatibility",
    "fetch_local_concepts_for_doc",
    "first_extension_local_type_rule_rows",
    "first_extension_relationship_types",
    "materialize_collapsed_connections",
    "normalize_materialization_mode",
    "resolve_run_name",
    "resolve_relation_name_filters",
    "should_materialize_relation",
    "traversal_policy_for_relation",
    "run_umls_connections",
    "build_candidate_edges",
    "select_representative_concepts",
]
