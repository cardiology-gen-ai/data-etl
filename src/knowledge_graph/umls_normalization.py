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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import Driver

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
from knowledge_graph.entity_schema import normalize_name

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


@dataclass
class ConceptRecord:
    concept_id: str
    name: str
    canonical_type: Optional[str] = None
    doc_ids: List[str] = field(default_factory=list)
    relationship_acronyms: List[Dict[str, str]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases_considered: List[str] = field(default_factory=list)
    normalization_status: str = "pending"
    normalization_method: Optional[str] = None
    best_match: Optional["UMLSMatch"] = None
    duplicate_candidates: List[Dict[str, Any]] = field(default_factory=list)
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

    try:
        import scispacy.file_cache as scispacy_file_cache
    except Exception:
        return None

    cache = Path(cache_dir or scispacy_file_cache.DATASET_CACHE)
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
            collect(DISTINCT s.doc_id) AS doc_ids,
            collect(DISTINCT {
                short: r.acronym_short,
                definition: r.acronym_definition
            }) AS acronym_rows
        RETURN
            elementId(c) AS concept_id,
            c.name AS name,
            c.canonical_type AS canonical_type,
            c.umls_cui AS umls_cui,
            c.umls_canonical_name AS umls_canonical_name,
            c.umls_definition AS umls_definition,
            c.umls_aliases AS umls_aliases,
            c.umls_score AS umls_score,
            c.umls_semantic_types AS umls_semantic_types,
            c.umls_linker_name AS umls_linker_name,
            c.umls_model_name AS umls_model_name,
            c.normalized_name AS normalized_name,
            c.normalization_status AS normalization_status,
            c.normalization_method AS normalization_method,
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


def build_aliases_for_concept(
    concept: ConceptRecord,
    acronyms_by_doc_id: Dict[str, Dict[str, str]],
) -> List[str]:
    aliases: List[str] = []
    append_unique(aliases, concept.name)

    for row in concept.relationship_acronyms:
        definition = clean_acronym_definition(row.get("definition"))
        if definition:
            append_unique(aliases, definition)
            append_unique(
                aliases,
                canonicalize_acronym_definition_for_concept_name(definition),
            )

    for doc_id in concept.doc_ids:
        for raw_short, raw_definition in sorted(
            acronyms_by_doc_id.get(doc_id, {}).items()
        ):
            short = clean_acronym_short(raw_short)
            definition = clean_acronym_definition(raw_definition)

            if not short or not definition:
                continue

            normalized_short = normalize_name(short)

            if normalized_short == concept.name:
                append_unique(aliases, definition)
                append_unique(
                    aliases,
                    canonicalize_acronym_definition_for_concept_name(definition),
                )
                continue

            if long_form_matches_concept(concept.name, definition):
                append_unique(aliases, definition)

    return aliases[:DEFAULT_ALIAS_LIMIT]


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

    The API does not expose scispaCy-style similarity scores. We therefore
    assign deterministic pseudo-scores: 1.0 for exact normalized name matches,
    0.95 for top normalizedString hits, and 0.85 for top words-search hits.
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

    def search_alias(
        self,
        alias: str,
        search_type: str,
    ) -> Optional[UMLSMatch]:
        payload = self.request_search(alias=alias, search_type=search_type)
        result = payload.get("result") if isinstance(payload, dict) else None
        results = result.get("results") if isinstance(result, dict) else None
        if not isinstance(results, list):
            raise UMLSAPIError("Malformed UMLS API response")

        for item in results:
            if not isinstance(item, dict):
                continue
            cui = str(item.get("ui") or "").strip()
            name = str(item.get("name") or "").strip()
            if not cui or cui.upper() == "NONE" or not name:
                continue

            normalized_alias = normalize_api_alias(alias)
            normalized_name = normalize_api_alias(name)
            if normalized_name == normalized_alias:
                score = 1.0
            elif search_type == "normalizedString":
                score = 0.95
            elif search_type == "words":
                score = 0.85
            else:
                score = 0.75

            semantic_types = item.get("semanticTypes") or item.get("semanticType") or []
            if isinstance(semantic_types, str):
                semantic_types = [semantic_types]
            if not isinstance(semantic_types, list):
                semantic_types = []

            return UMLSMatch(
                alias=alias,
                cui=cui,
                canonical_name=name,
                definition=None,
                aliases=[],
                score=round(score, 4),
                semantic_types=[str(value) for value in semantic_types if value],
            )

        return None


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
) -> Optional[UMLSMatch]:
    matches: List[UMLSMatch] = []

    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue

        match = client.search_alias(alias=alias, search_type="normalizedString")
        if match is None:
            match = client.search_alias(alias=alias, search_type="words")
        if match is not None:
            matches.append(match)

    if not matches:
        return None

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[0]


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

    return UMLSMatch(
        alias=concept.name,
        cui=str(cui),
        canonical_name=concept.properties.get("umls_canonical_name"),
        definition=concept.properties.get("umls_definition"),
        aliases=[str(alias) for alias in raw_aliases if alias],
        score=round(score, 4),
        semantic_types=[str(value) for value in raw_semantic_types if value],
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
) -> None:
    tx.run(
        """
        MATCH (c:Concept)
        WHERE elementId(c) = $concept_id
        SET c.normalization_status = $status,
            c.normalization_method = $method,
            c.normalized_at = datetime($normalized_at),
            c.updated_at = datetime()
        """,
        concept_id=concept_id,
        status=status,
        method=method,
        normalized_at=normalized_at,
    )


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
) -> None:
    if should_preserve_existing_normalization(concept, force=force):
        concept.normalization_status = "skipped"
        concept.normalization_method = SKIPPED_METHOD
        concept.reason = "existing_normalization_preserved"
        return

    match = concept.best_match

    if match is not None and match.score >= threshold:
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
        concept.normalization_status = "umls_low_confidence"
        concept.normalization_method = low_confidence_method
        concept.reason = "best_candidate_below_threshold"
        write_concept_normalization_status(
            tx=tx,
            concept_id=concept.concept_id,
            status=concept.normalization_status,
            method=concept.normalization_method,
            normalized_at=normalized_at,
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
        MATCH (a:Concept), (b:Concept)
        WHERE elementId(a) = $source_id AND elementId(b) = $target_id
        MERGE (a)-[r:SAME_AS]->(b)
        ON CREATE SET r.created_at = datetime($normalized_at)
        SET r.method = $method,
            r.score = 1.0,
            r.status = 'auto',
            r.updated_at = datetime($normalized_at)
        """,
        source_id=source_id,
        target_id=target_id,
        method=SAME_CUI_METHOD,
        normalized_at=normalized_at,
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
        MATCH (a:Concept), (b:Concept)
        WHERE elementId(a) = $source_id AND elementId(b) = $target_id
        MERGE (a)-[r:POSSIBLY_SAME_AS]->(b)
        ON CREATE SET r.created_at = datetime($normalized_at)
        SET r.method = $method,
            r.score = $score,
            r.status = 'candidate',
            r.updated_at = datetime($normalized_at)
        """,
        source_id=source_id,
        target_id=target_id,
        method=FUZZY_METHOD,
        score=score,
        normalized_at=normalized_at,
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
    compact = re.sub(r"[^a-z0-9]", "", normalize_name(name))
    if len(compact) <= 3:
        return True
    if compact.isupper() and len(compact) <= 12:
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
) -> Dict[str, int]:
    same_as_pairs = compute_same_cui_pairs(concepts)
    same_as_keys = {edge_key(left, right) for left, right in same_as_pairs}
    fuzzy_pairs = compute_fuzzy_pairs(
        concepts=concepts,
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

    if dry_run:
        return {
            "same_as_edges_created": 0,
            "possibly_same_as_edges_created": 0,
        }

    same_as_edges_created = 0
    possibly_same_as_edges_created = 0

    with driver.session() as session:
        for left, right in same_as_pairs:
            source, target = ordered_edge_pair(left, right)
            session.execute_write(
                create_same_as_edge,
                source.concept_id,
                target.concept_id,
                normalized_at,
            )
            same_as_edges_created += 1

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
) -> Dict[str, Any]:
    match = concept.best_match

    return {
        "run_id": run_id,
        "exported_at": utc_now_iso(),
        "doc_id": doc_id,
        "concept_id": concept.concept_id,
        "concept_name": concept.name,
        "canonical_type": concept.canonical_type,
        "aliases_considered": concept.aliases_considered,
        "normalization_status": concept.normalization_status,
        "normalization_method": concept.normalization_method,
        "reason": concept.reason,
        "umls_cui": match.cui if match else None,
        "umls_canonical_name": match.canonical_name if match else None,
        "umls_definition": match.definition if match else None,
        "umls_aliases": match.aliases if match else [],
        "umls_semantic_types": match.semantic_types if match else [],
        "umls_score": match.score if match else None,
        "umls_matched_alias": match.alias if match else None,
        "backend": backend,
        "model_name": model_name,
        "linker_name": linker_name,
        "threshold": threshold,
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
        "concepts_low_confidence": 0,
        "concepts_failed": 0,
        "concepts_skipped": 0,
        "same_as_edges_created": 0,
        "possibly_same_as_edges_created": 0,
        "review_records_written": 0,
        "api_cache_hits": 0,
        "api_cache_misses": 0,
        "api_requests": 0,
        "api_retries": 0,
        "api_errors": 0,
    }


def update_stats_for_concept(stats: Dict[str, int], concept: ConceptRecord) -> None:
    status = concept.normalization_status

    if status == "umls_matched":
        stats["concepts_normalized"] += 1
    elif status == "umls_low_confidence":
        stats["concepts_low_confidence"] += 1
    elif status == "umls_no_match":
        stats["concepts_no_match"] += 1
    elif status == "skipped":
        stats["concepts_skipped"] += 1
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
) -> Dict[str, int]:
    """
    Normalize existing Concept nodes with UMLS and add duplicate evidence.

    This is an optional post-processing phase. It never deletes or merges
    Concept nodes.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 <= fuzzy_threshold <= 100:
        raise ValueError("fuzzy_threshold must be between 0 and 100")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    backend = normalize_backend_name(backend)
    stats = initialize_stats()
    stats["backend"] = backend  # type: ignore[assignment]
    normalized_at = utc_now_iso()
    run_id = f"umls_normalization::{normalized_at}"

    with driver.session() as session:
        session.execute_write(setup_normalization_schema)
        concepts = session.execute_read(fetch_concepts_for_normalization, doc_id)

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

    nlp = linker = None
    api_client: Optional[UMLSAPIClient] = None
    method_for_match = AUTO_NORMALIZATION_METHOD
    method_for_low_confidence = LOW_CONFIDENCE_METHOD
    method_for_no_match = NO_MATCH_METHOD
    resolved_model_name = model_name
    resolved_linker_name = linker_name

    if backend == SCISPACY_BACKEND:
        nlp, linker = load_scispacy_pipeline(
            model_name=model_name,
            linker_name=linker_name,
            max_candidates=max_candidates,
            local_files_only=local_files_only,
            min_available_memory_gb=min_available_memory_gb,
        )
    elif backend == UMLS_API_BACKEND:
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

                if should_preserve_existing_normalization(concept, force=force):
                    concept.best_match = build_existing_umls_match(concept)
                    concept.normalization_status = "skipped"
                    concept.normalization_method = SKIPPED_METHOD
                    concept.reason = "existing_normalization_preserved"
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
                    concept.best_match = select_best_umls_api_match(
                        aliases=concept.aliases_considered,
                        client=api_client,
                    )
                else:
                    concept.best_match = None

                if dry_run:
                    match = concept.best_match
                    if match is not None and match.score >= threshold:
                        concept.normalization_status = "umls_matched"
                        concept.normalization_method = method_for_match
                        concept.reason = "dry_run_best_candidate_above_threshold"
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
                    )

            update_stats_for_concept(stats, concept)

    duplicate_stats = create_duplicate_evidence(
        driver=driver,
        concepts=concepts,
        fuzzy_threshold=fuzzy_threshold,
        normalized_at=normalized_at,
        dry_run=dry_run,
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
            fuzzy_threshold=fuzzy_threshold,
            review_output_dir=review_output_dir,
            run_id=run_id,
        )

    logger.info(
        "UMLS normalization completed | backend=%s | seen=%d | normalized=%d | low_confidence=%d | no_match=%d | failed=%d | skipped=%d | same_as=%d | fuzzy=%d | dry_run=%s",
        backend,
        stats["concepts_seen"],
        stats["concepts_normalized"],
        stats["concepts_low_confidence"],
        stats["concepts_no_match"],
        stats["concepts_failed"],
        stats["concepts_skipped"],
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
    "setup_normalization_schema",
    "fetch_concepts_for_normalization",
    "build_aliases_for_concept",
    "compute_fuzzy_pairs",
    "compute_same_cui_pairs",
    "normalize_backend_name",
]
