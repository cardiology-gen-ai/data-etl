from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

LOCAL_UMLS_SCOPE_SCHEMA_VERSION = "local_umls_scope_v1"
HISTORICAL_REGRESSION_SCHEMA_VERSION = "umls_relation_historical_regression_v1"
DELTA_PLAN_SCHEMA_VERSION = "umls_relation_delta_plan_v1"
DELTA_DISCOVERY_SCHEMA_VERSION = "umls_relation_delta_discovery_v1"
CURRENT_PRELABEL_SCHEMA_VERSION = "umls_relation_current_prelabel_v1"
CURRENT_FINAL_SCHEMA_VERSION = "umls_relation_current_final_v1"
GENERALIZED_V2_SCHEMA_VERSION = "umls_relation_generalized_v2_v1"
EXTERNAL_LABEL_PLAN_SCHEMA_VERSION = "external_cui_label_plan_v1"
EXTERNAL_LABEL_MAP_SCHEMA_VERSION = "external_cui_labels_v1"
EXTERNAL_LABEL_RESOLUTION_SCHEMA_VERSION = (
    "umls_relation_external_label_resolution_v1"
)
DEFAULT_RELATION_PAGE_SIZE = 200



def _clean(value: Any) -> str:
    return str(value or "").strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_key = _clean(row.get(key))
            if not row_key:
                raise ValueError(f"Missing {key!r} in {path}:{line_no}")
            if row_key in rows:
                raise ValueError(f"Duplicate {key}={row_key!r} in {path}")
            rows[row_key] = row
    return rows


def _normalise_doc_ids(values: Iterable[Any] | None) -> list[str]:
    return sorted({_clean(value) for value in (values or []) if _clean(value)})


def build_local_umls_scope(
    driver,
    output_path: Path,
    *,
    scope_name: str = "current_corpus",
    document_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export the current trusted local UMLS scope from Neo4j.

    The output deliberately retains the legacy top-level ``document_id`` field
    because the frozen census builders read it, while the user-facing name is
    ``scope_name``.  No Neo4j writes are performed.
    """

    requested_docs = _normalise_doc_ids(document_ids)
    scope_name = _clean(scope_name) or "current_corpus"

    query = """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        WHERE c.normalization_status = 'umls_matched'
          AND c.umls_cui IS NOT NULL
          AND trim(toString(c.umls_cui)) <> ''
          AND coalesce(c.needs_type_review, false) = false
          AND coalesce(c.canonical_type, '') <> 'ambiguous'
          AND ($document_ids = [] OR s.doc_id IN $document_ids)
        WITH
            trim(toString(c.umls_cui)) AS cui,
            collect(DISTINCT c.name) AS names,
            collect(DISTINCT c.canonical_type) AS canonical_types,
            collect(DISTINCT c.normalization_status) AS normalization_statuses,
            collect(DISTINCT s.doc_id) AS doc_ids,
            count(DISTINCT c) AS concept_count,
            count(DISTINCT s) AS section_count
        RETURN
            cui,
            names,
            canonical_types,
            normalization_statuses,
            doc_ids,
            concept_count,
            section_count
        ORDER BY cui
    """

    with driver.session() as session:
        records = [
            dict(record)
            for record in session.run(query, document_ids=requested_docs)
        ]

    rows: list[dict[str, Any]] = []
    observed_docs: set[str] = set()
    status_cui_counts: Counter[str] = Counter()

    for record in records:
        cui = _clean(record.get("cui"))
        if not cui:
            continue
        names = sorted({_clean(x) for x in record.get("names", []) if _clean(x)})
        canonical_types = sorted(
            {_clean(x) for x in record.get("canonical_types", []) if _clean(x)}
        )
        statuses = sorted(
            {_clean(x) for x in record.get("normalization_statuses", []) if _clean(x)}
        )
        docs = _normalise_doc_ids(record.get("doc_ids"))
        observed_docs.update(docs)
        for status in statuses:
            status_cui_counts[status] += 1

        rows.append(
            {
                "cui": cui,
                "names": names,
                "canonical_types": canonical_types,
                "normalization_statuses": statuses,
                "concept_count": int(record.get("concept_count") or 0),
                "section_count": int(record.get("section_count") or 0),
                "document_ids": docs,
            }
        )

    effective_docs = requested_docs or sorted(observed_docs)
    payload = {
        "schema_version": LOCAL_UMLS_SCOPE_SCHEMA_VERSION,
        "scope_name": scope_name,
        # Backward-compatibility alias consumed by the historical census tools.
        "document_id": scope_name,
        "document_ids": effective_docs,
        "document_count": len(effective_docs),
        "unique_cui_count": len(rows),
        "normalization_status_cui_counts": dict(sorted(status_cui_counts.items())),
        "eligibility": {
            "normalization_status": "umls_matched",
            "requires_umls_cui": True,
            "exclude_ambiguous_canonical_type": True,
            "exclude_needs_type_review": True,
        },
        "cuis": rows,
        "neo4j_reads": True,
        "neo4j_writes": False,
    }
    _write_json(output_path, payload)

    logger.info(
        "Exported local UMLS scope | scope=%s | documents=%d | cuis=%d | path=%s",
        scope_name,
        len(effective_docs),
        len(rows),
        output_path,
    )
    return payload


def compare_local_umls_scopes(
    current_scope_path: Path,
    previous_scope_path: Path,
) -> dict[str, Any]:
    """Compare two scope files by CUI without modifying either artifact."""

    current = _read_json(current_scope_path)
    previous = _read_json(previous_scope_path)
    current_cuis = {
        _clean(row.get("cui"))
        for row in current.get("cuis", [])
        if _clean(row.get("cui"))
    }
    previous_cuis = {
        _clean(row.get("cui"))
        for row in previous.get("cuis", [])
        if _clean(row.get("cui"))
    }
    return {
        "current_cui_count": len(current_cuis),
        "previous_cui_count": len(previous_cuis),
        "shared_cui_count": len(current_cuis & previous_cuis),
        "new_cui_count": len(current_cuis - previous_cuis),
        "retired_cui_count": len(previous_cuis - current_cuis),
        "new_cuis": sorted(current_cuis - previous_cuis),
        "retired_cuis": sorted(previous_cuis - current_cuis),
    }


@dataclass(frozen=True)
class HistoricalRegressionPaths:
    direct_census_dir: Path
    direct_policy: Path
    expected_direct_artifact_dir: Path
    bridge_root: Path
    historical_scope: Path
    bridge_policy: Path
    expected_bridge_artifact_dir: Path
    external_label_map: Path | None
    direct_builder: Path
    bridge_builder: Path


@dataclass(frozen=True)
class DeltaDiscoveryPaths:
    source_profile: Path
    relation_cache_dir: Path
    bridge_census_script: Path


@dataclass(frozen=True)
class CurrentArtifactBuildPaths:
    """Reusable builders for the current-scope artifact stage."""

    direct_census_script: Path
    bridge_prelabel_builder: Path
    bridge_prelabel_policy: Path
    direct_unmapped_relation_mode: str = "error"


@dataclass(frozen=True)
class ExternalLabelResolutionPaths:
    """Inputs and persistent cache for the C2 external-label stage."""

    historical_label_map: Path
    cache_dir: Path


@dataclass(frozen=True)
class CurrentFinalBuildPaths:
    """Frozen label-aware builder inputs for the C3 final artifact stage."""

    bridge_final_builder: Path
    bridge_final_policy: Path


@dataclass(frozen=True)
class GeneralizedBridgeBuildPaths:
    """Inputs for the post-C3 semantic/structural v2 policy build."""

    bridge_builder: Path
    bridge_policy: Path
    frozen_v1_1_policy_sha256: str = ""
    frozen_c3_manifest_sha256: str = ""


def _scope_cui_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("cuis", []) if isinstance(payload, Mapping) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cui = _clean(row.get("cui"))
        if not cui:
            continue
        if cui in out:
            raise ValueError(f"Duplicate CUI {cui!r} in {path}")
        out[cui] = dict(row)
    return out


def _relation_cache_payload(
    *,
    cui: str,
    source: str,
    umls_version: str,
    max_relations_per_cui: int,
    page_size: int = DEFAULT_RELATION_PAGE_SIZE,
) -> dict[str, Any]:
    max_records = int(max_relations_per_cui) if max_relations_per_cui > 0 else None
    effective_page_size = (
        int(page_size)
        if max_records is None
        else max(1, min(int(page_size), max_records + 1))
    )
    return {
        "endpoint": "relations",
        "cui": _clean(cui).upper(),
        "source_vocab": _clean(source),
        "version": str(umls_version),
        "page_size": effective_page_size,
        "max_records": max_records,
        "include_additional_relation_labels": [],
        "include_obsolete": False,
        "include_suppressible": False,
    }


def _relation_negative_cache_payload(
    *,
    cui: str,
    source: str,
    umls_version: str,
) -> dict[str, Any]:
    return {
        "endpoint": "relations",
        "cui": _clean(cui).upper(),
        "source_vocab": _clean(source),
        "version": str(umls_version),
        "include_additional_relation_labels": [],
        "http_status": 404,
    }


def _relation_cache_status(
    cache_dir: Path,
    *,
    cui: str,
    source: str,
    umls_version: str,
    max_relations_per_cui: int,
) -> str:
    """Return positive, negative, invalid_negative, or missing.

    This mirrors the frozen relation-cache key used by the historical B0/direct
    census tools.  It does not instantiate an API client and cannot call UMLS.
    """

    def local_cache_key(payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    positive_payload = _relation_cache_payload(
        cui=cui,
        source=source,
        umls_version=umls_version,
        max_relations_per_cui=max_relations_per_cui,
    )
    positive = cache_dir / "relations" / f"{local_cache_key(positive_payload)}.json"
    if positive.is_file():
        return "positive"

    negative_payload = _relation_negative_cache_payload(
        cui=cui,
        source=source,
        umls_version=umls_version,
    )
    negative = cache_dir / "relations_negative" / f"{local_cache_key(negative_payload)}.json"
    if not negative.is_file():
        return "missing"

    try:
        value = _read_json(negative)
    except Exception:
        return "invalid_negative"
    status = _clean(value.get("status")) if isinstance(value, Mapping) else ""
    if status in {
        "relations_unavailable",
        "source_vocab_relations_absent",
        "filtered_relations_absent",
    }:
        return "negative"
    return "invalid_negative"


def _membership_label(document_ids: Sequence[str] | None) -> str:
    docs = set(_normalise_doc_ids(document_ids))
    if len(docs) == 2:
        return "shared"
    if len(docs) == 1:
        return next(iter(docs))
    if not docs:
        return "none"
    return "+".join(sorted(docs))


def _require_historical_regression_pass(
    report_path: Path,
    *,
    previous_scope_path: Path,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Historical regression report not found: {report_path}. "
            "Run action=scope_and_regression first."
        )
    report = _read_json(report_path)
    if not bool(report.get("pass")):
        raise RuntimeError(f"Historical regression is not PASS: {report_path}")
    safety = report.get("safety") or {}
    if safety.get("umls_api_calls") or safety.get("neo4j_writes"):
        raise RuntimeError(
            "Historical regression report has unexpected unsafe flags: "
            f"{report_path}"
        )
    expected_scope_hash = _clean(
        ((report.get("bridge") or {}).get("historical_scope_sha256"))
    )
    actual_scope_hash = _sha256_file(previous_scope_path)
    if expected_scope_hash and expected_scope_hash != actual_scope_hash:
        raise RuntimeError(
            "Historical scope hash changed since regression: "
            f"expected={expected_scope_hash}, actual={actual_scope_hash}"
        )
    return report


def build_delta_discovery_plan(
    *,
    current_scope_path: Path,
    previous_scope_path: Path,
    regression_report_path: Path,
    historical_paths: HistoricalRegressionPaths,
    discovery_paths: DeltaDiscoveryPaths,
    sources: Sequence[str],
    umls_version: str,
    max_relations_per_cui: int,
    output_path: Path,
) -> dict[str, Any]:
    """Build a read-only, cache-aware plan for the new-CUI relation census."""

    _require_historical_regression_pass(
        regression_report_path, previous_scope_path=previous_scope_path
    )

    current = _scope_cui_index(current_scope_path)
    previous = _scope_cui_index(previous_scope_path)
    current_cuis = set(current)
    previous_cuis = set(previous)
    new_cuis = sorted(current_cuis - previous_cuis)
    retired_cuis = sorted(previous_cuis - current_cuis)
    shared_cuis = sorted(current_cuis & previous_cuis)

    profile = _read_json(discovery_paths.source_profile)
    profile_version = _clean(profile.get("umls_version"))
    if profile_version and profile_version != _clean(umls_version):
        raise RuntimeError(
            "UMLS version mismatch between frozen source profile and artifact config: "
            f"profile={profile_version}, configured={umls_version}"
        )

    configured_sources = [str(x).strip() for x in sources if str(x).strip()]
    if not configured_sources:
        raise ValueError("At least one UMLS relation source is required")
    source_cfg = profile.get("sources") or {}
    missing_or_disabled = [
        source
        for source in configured_sources
        if not isinstance(source_cfg.get(source), Mapping)
        or not bool((source_cfg.get(source) or {}).get("enabled", False))
    ]
    if missing_or_disabled:
        raise RuntimeError(
            "Configured UMLS sources are absent/disabled in frozen source profile: "
            + ", ".join(missing_or_disabled)
        )

    historical_direct_manifest = _read_json(
        historical_paths.direct_census_dir / "manifest.json"
    )
    frozen_profile_hash = _clean(
        historical_direct_manifest.get("source_profile_sha256")
    )
    current_profile_hash = _sha256_file(discovery_paths.source_profile)
    if frozen_profile_hash and frozen_profile_hash != current_profile_hash:
        raise RuntimeError(
            "Frozen source-profile hash mismatch: "
            f"historical={frozen_profile_hash}, current={current_profile_hash}"
        )

    historical_source_checks: dict[str, Any] = {}
    for source in configured_sources:
        summary_path = historical_paths.bridge_root / source / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = _read_json(summary_path)
        hist_max = ((summary.get("inputs") or {}).get("max_relations_per_cui"))
        hist_version = _clean(summary.get("umls_version"))
        hist_profile_hash = _clean(
            ((summary.get("inputs") or {}).get("source_profile_sha256"))
        )
        ok = (
            int(hist_max) == int(max_relations_per_cui)
            and (not hist_version or hist_version == _clean(umls_version))
            and (not hist_profile_hash or hist_profile_hash == current_profile_hash)
        )
        historical_source_checks[source] = {
            "historical_max_relations_per_cui": hist_max,
            "historical_umls_version": hist_version,
            "historical_source_profile_sha256": hist_profile_hash,
            "matches_frozen_parameters": ok,
        }
        if not ok:
            raise RuntimeError(
                f"Frozen discovery parameter mismatch for source {source}: "
                f"{historical_source_checks[source]}"
            )

    cache_dir = discovery_paths.relation_cache_dir
    cache_coverage: dict[str, Any] = {}
    historical_cache_complete = True
    total_new_missing = 0
    for source in configured_sources:
        old_counts: Counter[str] = Counter()
        new_counts: Counter[str] = Counter()
        missing_new: list[str] = []
        invalid_old: list[str] = []
        for cui in sorted(previous_cuis):
            status = _relation_cache_status(
                cache_dir,
                cui=cui,
                source=source,
                umls_version=umls_version,
                max_relations_per_cui=max_relations_per_cui,
            )
            old_counts[status] += 1
            if status not in {"positive", "negative"}:
                invalid_old.append(cui)
        for cui in new_cuis:
            status = _relation_cache_status(
                cache_dir,
                cui=cui,
                source=source,
                umls_version=umls_version,
                max_relations_per_cui=max_relations_per_cui,
            )
            new_counts[status] += 1
            if status not in {"positive", "negative"}:
                missing_new.append(cui)
        if invalid_old:
            historical_cache_complete = False
        total_new_missing += len(missing_new)
        cache_coverage[source] = {
            "historical_scope": dict(sorted(old_counts.items())),
            "historical_missing_or_invalid_count": len(invalid_old),
            "historical_missing_or_invalid_cuis": invalid_old[:50],
            "new_scope_delta": dict(sorted(new_counts.items())),
            "new_missing_or_invalid_count": len(missing_new),
            "new_missing_or_invalid_cuis": missing_new,
        }

    membership_counts: Counter[str] = Counter(
        _membership_label(current[cui].get("document_ids")) for cui in new_cuis
    )

    plan = {
        "schema_version": DELTA_PLAN_SCHEMA_VERSION,
        "ready_for_discovery": bool(historical_cache_complete),
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "retrieval_metrics_used": False,
        },
        "scope": {
            "current_cui_count": len(current_cuis),
            "historical_cui_count": len(previous_cuis),
            "shared_cui_count": len(shared_cuis),
            "new_cui_count": len(new_cuis),
            "retired_cui_count": len(retired_cuis),
            "new_cui_membership_counts": dict(sorted(membership_counts.items())),
            "new_cuis": new_cuis,
            "retired_cuis": retired_cuis,
            "current_scope_path": str(current_scope_path.resolve()),
            "current_scope_sha256": _sha256_file(current_scope_path),
            "historical_scope_path": str(previous_scope_path.resolve()),
            "historical_scope_sha256": _sha256_file(previous_scope_path),
        },
        "frozen_parameters": {
            "umls_version": umls_version,
            "sources": configured_sources,
            "max_relations_per_cui": int(max_relations_per_cui),
            "source_profile": str(discovery_paths.source_profile.resolve()),
            "source_profile_sha256": current_profile_hash,
            "bridge_census_script": str(discovery_paths.bridge_census_script.resolve()),
            "bridge_census_script_sha256": _sha256_file(
                discovery_paths.bridge_census_script
            ),
            "historical_source_checks": historical_source_checks,
        },
        "relation_cache": {
            "path": str(cache_dir.resolve()),
            "historical_cache_complete": historical_cache_complete,
            "coverage_by_source": cache_coverage,
            "new_top_level_relation_fetches_estimated": total_new_missing,
            "note": (
                "Estimate covers top-level CUI relation-cache misses only; "
                "AUI/source-UI endpoint resolution can require additional API requests."
            ),
        },
        "historical_regression_report": str(regression_report_path.resolve()),
        "historical_regression_report_sha256": _sha256_file(regression_report_path),
    }
    _write_json(output_path, plan)

    if not historical_cache_complete:
        raise RuntimeError(
            "Historical relation cache is incomplete for the frozen 584-CUI scope; "
            f"see {output_path}. Refusing delta discovery to avoid old-CUI API drift."
        )

    logger.info(
        "UMLS relation delta plan ready | current=%d | historical=%d | new=%d | "
        "retired=%d | estimated_top_level_fetches=%d",
        len(current_cuis),
        len(previous_cuis),
        len(new_cuis),
        len(retired_cuis),
        total_new_missing,
    )
    return plan


def _validate_delta_source_summary(
    summary: Mapping[str, Any],
    *,
    source: str,
    expected_scope_count: int,
    expected_delta_count: int,
    current_scope_sha256: str,
    source_profile_sha256: str,
    umls_version: str,
    max_relations_per_cui: int,
) -> list[str]:
    issues: list[str] = []
    if _clean(summary.get("source_vocabulary")) != source:
        issues.append("source_vocabulary")
    if int(summary.get("local_universe_count") or -1) != int(expected_scope_count):
        issues.append("local_universe_count")
    if int(summary.get("processed_local_cui_count") or -1) != int(expected_delta_count):
        issues.append("processed_local_cui_count")
    if _clean(summary.get("umls_version")) != _clean(umls_version):
        issues.append("umls_version")
    inputs = summary.get("inputs") or {}
    if _clean(inputs.get("local_universe_sha256")) != current_scope_sha256:
        issues.append("local_universe_sha256")
    if _clean(inputs.get("source_profile_sha256")) != source_profile_sha256:
        issues.append("source_profile_sha256")
    if int(inputs.get("max_relations_per_cui") or -1) != int(max_relations_per_cui):
        issues.append("max_relations_per_cui")
    if int(summary.get("fetch_failure_count") or 0) != 0:
        issues.append("fetch_failure_count")
    client_stats = summary.get("client_stats") or {}
    if int(client_stats.get("api_errors") or 0) != 0:
        issues.append("api_errors")
    safety = summary.get("safety") or {}
    if safety.get("neo4j_writes") or safety.get("second_hop_requests"):
        issues.append("unsafe_safety_flags")
    return issues


def run_delta_relation_discovery(
    *,
    project_root: Path,
    output_dir: Path,
    plan: Mapping[str, Any],
    discovery_paths: DeltaDiscoveryPaths,
    api_timeout: float,
    api_rate_limit_per_second: float,
    progress_every: int,
    resume_completed_sources: bool,
) -> dict[str, Any]:
    """Run first-hop B0 discovery only for the new CUI delta.

    The historical CUI relation cache is never refreshed.  The bridge census
    script is given the *current* local scope but only the delta CUIs as query
    seeds, so local-vs-external classification uses the full current corpus.
    No Neo4j writes or second-hop requests are performed.
    """

    if not bool(plan.get("ready_for_discovery")):
        raise RuntimeError("Delta plan is not ready_for_discovery")

    scope = plan.get("scope") or {}
    frozen = plan.get("frozen_parameters") or {}
    current_scope_path = Path(str(scope.get("current_scope_path"))).resolve()
    current_scope_sha = _clean(scope.get("current_scope_sha256"))
    if _sha256_file(current_scope_path) != current_scope_sha:
        raise RuntimeError("Current local UMLS scope changed after delta plan")

    new_cuis = [str(x) for x in scope.get("new_cuis", [])]
    sources = [str(x) for x in frozen.get("sources", [])]
    umls_version = _clean(frozen.get("umls_version"))
    max_relations_per_cui = int(frozen.get("max_relations_per_cui") or 0)
    source_profile_sha = _clean(frozen.get("source_profile_sha256"))

    output_dir.mkdir(parents=True, exist_ok=True)
    source_results: dict[str, Any] = {}
    totals: Counter[str] = Counter()

    for source in sources:
        source_out = output_dir / source
        summary_path = source_out / "summary.json"
        reused = False

        if resume_completed_sources and summary_path.is_file():
            try:
                existing = _read_json(summary_path)
                issues = _validate_delta_source_summary(
                    existing,
                    source=source,
                    expected_scope_count=int(scope.get("current_cui_count") or 0),
                    expected_delta_count=len(new_cuis),
                    current_scope_sha256=current_scope_sha,
                    source_profile_sha256=source_profile_sha,
                    umls_version=umls_version,
                    max_relations_per_cui=max_relations_per_cui,
                )
                if not issues:
                    summary = existing
                    reused = True
                    logger.info(
                        "Reusing completed UMLS delta source=%s | path=%s",
                        source,
                        summary_path,
                    )
                else:
                    shutil.rmtree(source_out, ignore_errors=True)
            except Exception:
                shutil.rmtree(source_out, ignore_errors=True)

        if not reused:
            source_out.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(discovery_paths.bridge_census_script.resolve()),
                "--local-universe",
                str(current_scope_path),
                "--source-profile",
                str(discovery_paths.source_profile.resolve()),
                "--source-vocab",
                source,
                "--output-dir",
                str(source_out),
                "--umls-version",
                umls_version,
                "--cache-dir",
                str(discovery_paths.relation_cache_dir.resolve()),
                "--timeout",
                str(float(api_timeout)),
                "--rate-limit",
                str(float(api_rate_limit_per_second)),
                "--max-relations-per-cui",
                str(max_relations_per_cui),
                "--progress-every",
                str(int(progress_every)),
            ]
            for cui in new_cuis:
                command.extend(["--include-cui", cui])
            _run_builder(command, cwd=project_root)
            summary = _read_json(summary_path)

        issues = _validate_delta_source_summary(
            summary,
            source=source,
            expected_scope_count=int(scope.get("current_cui_count") or 0),
            expected_delta_count=len(new_cuis),
            current_scope_sha256=current_scope_sha,
            source_profile_sha256=source_profile_sha,
            umls_version=umls_version,
            max_relations_per_cui=max_relations_per_cui,
        )
        if issues:
            raise RuntimeError(
                f"UMLS delta discovery validation failed for {source}: {issues}"
            )

        stats = summary.get("client_stats") or {}
        for key in (
            "api_cache_hits",
            "api_cache_misses",
            "api_requests",
            "api_retries",
            "api_errors",
            "relation_negative_cache_hits",
            "relation_negative_cache_writes",
        ):
            totals[key] += int(stats.get(key) or 0)
        totals["raw_relation_record_count"] += int(
            summary.get("raw_relation_record_count") or 0
        )
        totals["candidate_external_raw_row_count"] += int(
            summary.get("candidate_external_raw_row_count") or 0
        )
        totals["collapsed_external_assertion_count"] += int(
            summary.get("collapsed_external_assertion_count") or 0
        )
        totals["local_target_rows_excluded"] += int(
            summary.get("local_target_rows_excluded") or 0
        )
        totals["unresolved_rows"] += int(summary.get("unresolved_rows") or 0)
        totals["truncated_local_cui_count"] += int(
            summary.get("truncated_local_cui_count") or 0
        )

        source_results[source] = {
            "reused_completed_output": reused,
            "summary_path": str(summary_path),
            # Completion-critical fields are copied into the aggregate manifest
            # for convenient inspection.  The per-source summary.json remains
            # authoritative when validating previously completed discovery runs.
            "source_vocabulary": summary.get("source_vocabulary"),
            "local_universe_count": summary.get("local_universe_count"),
            "processed_local_cui_count": summary.get("processed_local_cui_count"),
            "fetch_failure_count": summary.get("fetch_failure_count"),
            "umls_version": summary.get("umls_version"),
            "raw_relation_record_count": summary.get("raw_relation_record_count"),
            "candidate_external_raw_row_count": summary.get(
                "candidate_external_raw_row_count"
            ),
            "collapsed_external_assertion_count": summary.get(
                "collapsed_external_assertion_count"
            ),
            "unique_external_cui_count": summary.get("unique_external_cui_count"),
            "local_target_rows_excluded": summary.get("local_target_rows_excluded"),
            "unresolved_rows": summary.get("unresolved_rows"),
            "truncated_local_cui_count": summary.get("truncated_local_cui_count"),
            "client_stats": stats,
        }

    manifest = {
        "schema_version": DELTA_DISCOVERY_SCHEMA_VERSION,
        "scope": dict(scope),
        "frozen_parameters": dict(frozen),
        "source_results": source_results,
        "totals": dict(sorted(totals.items())),
        "safety": {
            "umls_api_calls": True,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "second_hop_requests": False,
            "bridge_edges_materialized": False,
            "retrieval_metrics_used": False,
        },
        "note": (
            "This phase discovers first-hop evidence for delta CUIs only. "
            "It does not yet build or materialize the global CM+CO DIRECT/BRIDGE artifacts."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    logger.info(
        "UMLS relation delta discovery complete | sources=%d | new_cuis=%d | "
        "api_requests=%d | collapsed_assertions=%d",
        len(sources),
        len(new_cuis),
        totals.get("api_requests", 0),
        totals.get("collapsed_external_assertion_count", 0),
    )
    return manifest



def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    value = _read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [row for row in value if isinstance(row, dict)]


def _merge_collapsed_assertion_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically merge already-collapsed first-hop assertions."""

    list_fields = (
        "relation_labels",
        "root_sources",
        "relation_ids",
        "raw_subject_identifier_kinds",
        "raw_object_identifier_kinds",
    )
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for raw in rows:
        local_cui = _clean(raw.get("local_cui") or raw.get("local_source_cui")).upper()
        external_cui = _clean(raw.get("external_cui")).upper()
        relation_name = _clean(raw.get("relation_name")).casefold()
        local_endpoint_role = _clean(
            raw.get("local_endpoint_role") or "unknown"
        ).casefold()
        if not local_cui or not external_cui or not relation_name:
            continue

        key = (local_cui, external_cui, relation_name, local_endpoint_role)
        if key not in merged:
            merged[key] = {
                "local_cui": local_cui,
                "local_source_cui": local_cui,
                "external_cui": external_cui,
                "relation_name": relation_name,
                "local_endpoint_role": local_endpoint_role,
                "relation_labels": [],
                "root_sources": [],
                "relation_ids": [],
                "raw_rows": 0,
                "max_counterpart_fanout": 0,
                "raw_subject_identifier_kinds": [],
                "raw_object_identifier_kinds": [],
            }

        dst = merged[key]
        dst["raw_rows"] += int(raw.get("raw_rows") or 0)
        dst["max_counterpart_fanout"] = max(
            int(dst["max_counterpart_fanout"]),
            int(raw.get("max_counterpart_fanout") or 0),
        )
        for field in list_fields:
            dst[field] = sorted(
                {
                    _clean(value)
                    for value in list(dst.get(field) or []) + list(raw.get(field) or [])
                    if _clean(value)
                }
            )

    return [merged[key] for key in sorted(merged)]


def build_current_bridge_evidence_root(
    *,
    current_scope_path: Path,
    historical_bridge_root: Path,
    delta_discovery_root: Path,
    sources: Sequence[str],
    output_root: Path,
) -> dict[str, Any]:
    """Merge historical + delta first-hop evidence for the current local scope.

    Historical seed CUIs that are no longer local are removed.  An endpoint that
    was historically external but is now in the local scope is *not* retained as
    a bridge hub; it is reported as ``promoted_to_local`` and is expected to be
    recovered by the cache-only DIRECT census.
    """

    current_index = _scope_cui_index(current_scope_path)
    current_cuis = set(current_index)
    output_root.mkdir(parents=True, exist_ok=True)

    source_reports: dict[str, Any] = {}
    totals = Counter()

    for source in sources:
        historical_rows = _load_json_list(
            historical_bridge_root / source / "collapsed_first_hop_assertions.json"
        )
        delta_rows = _load_json_list(
            delta_discovery_root / source / "collapsed_first_hop_assertions.json"
        )

        kept: list[dict[str, Any]] = []
        retired_seed = 0
        promoted_to_local = 0
        invalid = 0
        promoted_pairs: set[tuple[str, str]] = set()

        for row in historical_rows + delta_rows:
            local_cui = _clean(
                row.get("local_cui") or row.get("local_source_cui")
            ).upper()
            external_cui = _clean(row.get("external_cui")).upper()
            if not local_cui or not external_cui:
                invalid += 1
                continue
            if local_cui not in current_cuis:
                retired_seed += 1
                continue
            if external_cui in current_cuis:
                promoted_to_local += 1
                promoted_pairs.add(tuple(sorted((local_cui, external_cui))))
                continue
            kept.append(dict(row))

        collapsed = _merge_collapsed_assertion_rows(kept)
        source_dir = output_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        _write_json(source_dir / "collapsed_first_hop_assertions.json", collapsed)

        report = {
            "historical_assertion_count": len(historical_rows),
            "delta_assertion_count": len(delta_rows),
            "input_assertion_count": len(historical_rows) + len(delta_rows),
            "retired_seed_assertion_count": retired_seed,
            "promoted_to_local_assertion_count": promoted_to_local,
            "promoted_to_local_distinct_pair_count": len(promoted_pairs),
            "invalid_assertion_count": invalid,
            "current_external_assertion_count": len(collapsed),
        }
        _write_json(source_dir / "summary.json", report)
        source_reports[source] = report
        totals.update(report)

    manifest = {
        "schema_version": "current_bridge_first_hop_evidence_v1",
        "current_scope": str(current_scope_path.resolve()),
        "current_scope_sha256": _sha256_file(current_scope_path),
        "current_cui_count": len(current_cuis),
        "historical_bridge_root": str(historical_bridge_root.resolve()),
        "delta_discovery_root": str(delta_discovery_root.resolve()),
        "sources": list(sources),
        "source_reports": source_reports,
        "totals": dict(totals),
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "retrieval_metrics_used": False,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def build_external_label_plan(
    *,
    retained_external_cuis_path: Path,
    historical_label_map_path: Path | None,
    umls_version: str,
    output_dir: Path,
) -> dict[str, Any]:
    retained = sorted(
        {
            line.strip().upper()
            for line in retained_external_cuis_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
    )

    labels: dict[str, Any] = {}
    historical_failures: dict[str, Any] = {}
    if historical_label_map_path is not None and historical_label_map_path.exists():
        payload = _read_json(historical_label_map_path)
        if isinstance(payload, Mapping):
            raw_labels = payload.get("labels") or {}
            raw_failures = payload.get("failures") or {}
            if isinstance(raw_labels, Mapping):
                labels = {
                    _clean(cui).upper(): value
                    for cui, value in raw_labels.items()
                    if _clean(cui)
                }
            if isinstance(raw_failures, Mapping):
                historical_failures = {
                    _clean(cui).upper(): value
                    for cui, value in raw_failures.items()
                    if _clean(cui)
                }

    already_resolved = sorted(cui for cui in retained if cui in labels)
    to_resolve = sorted(cui for cui in retained if cui not in labels)
    retry_historical_failures = sorted(
        cui for cui in to_resolve if cui in historical_failures
    )
    completely_new = sorted(set(to_resolve) - set(retry_historical_failures))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "external_cuis_to_resolve.txt").write_text(
        "\n".join(to_resolve) + ("\n" if to_resolve else ""),
        encoding="utf-8",
    )
    plan = {
        "schema_version": "external_cui_label_plan_v1",
        "umls_version": umls_version,
        "retained_external_cui_count": len(retained),
        "already_resolved_from_historical_map_count": len(already_resolved),
        "to_resolve_count": len(to_resolve),
        "retry_historical_failure_count": len(retry_historical_failures),
        "completely_new_to_resolve_count": len(completely_new),
        "historical_label_map": (
            str(historical_label_map_path.resolve())
            if historical_label_map_path is not None
            else None
        ),
        "external_cuis_to_resolve": to_resolve,
        "retry_historical_failures": retry_historical_failures,
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
        },
    }
    _write_json(output_dir / "external_label_plan_v1.json", plan)
    return plan


class _ExternalLabelNotFound(RuntimeError):
    pass


class _ExternalLabelTransientError(RuntimeError):
    pass


class _ExternalLabelPermanentError(RuntimeError):
    pass


def _redact_api_key(text: Any, api_key: str | None = None) -> str:
    """Keep UMLS credentials out of logs, manifests, and failure artifacts."""

    value = _clean(text)
    if api_key:
        value = value.replace(api_key, "<redacted>")
    return re.sub(
        r"(?i)(apiKey=)[^&\s\"']+",
        r"\1<redacted>",
        value,
    )


def _normalise_external_label_record(
    cui: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert the UMLS concept endpoint response to the frozen v1 label shape."""

    result = payload.get("result")
    if not isinstance(result, Mapping):
        # Tests/caches may pass the concept body directly.
        result = payload

    name = _clean(result.get("name"))
    ui = _clean(result.get("ui")) or cui
    class_type = _clean(result.get("classType") or result.get("class_type"))
    raw_semantic_types = result.get("semanticTypes") or result.get("semantic_types") or []

    semantic_types: list[dict[str, str]] = []
    if isinstance(raw_semantic_types, Sequence) and not isinstance(
        raw_semantic_types, (str, bytes)
    ):
        for row in raw_semantic_types:
            if not isinstance(row, Mapping):
                continue
            semantic_name = _clean(row.get("name"))
            semantic_uri = _clean(row.get("uri"))
            if semantic_name or semantic_uri:
                semantic_types.append(
                    {"name": semantic_name, "uri": semantic_uri}
                )

    if not name:
        raise _ExternalLabelPermanentError(
            f"UMLS concept payload for {cui} does not contain a preferred name"
        )

    return {
        "name": name,
        "semantic_types": semantic_types,
        "class_type": class_type,
        "ui": ui,
    }


class UMLSExternalLabelClient:
    """Small cache-aware client for the UMLS concept endpoint used by C2.

    Successful concept responses are cached locally. Failures are deliberately
    not negative-cached here: C2 is resumable and must retry transient failures
    on the next run instead of fossilising a temporary network outage.
    """

    base_url = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(
        self,
        *,
        cache_dir: Path,
        version: str,
        timeout: float = 30.0,
        rate_limit_per_second: float = 5.0,
        max_retries: int = 2,
        api_key: str | None = None,
        session: Any | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = _clean(version)
        self.timeout = float(timeout)
        self.rate_limit_per_second = max(float(rate_limit_per_second), 0.0)
        self.max_retries = max(int(max_retries), 0)
        self.api_key = _clean(api_key or os.getenv("UMLS_API_KEY"))
        if not self.api_key:
            raise RuntimeError("UMLS_API_KEY is missing/invalid")

        if session is None:
            try:
                import requests  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment guard
                raise RuntimeError(
                    "requests is required for external UMLS label resolution"
                ) from exc
            session = requests.Session()
        self.session = session
        self._last_request_at = 0.0
        self.stats: dict[str, int] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_requests": 0,
            "api_retries": 0,
            "api_errors": 0,
            "not_found": 0,
        }

    def _cache_path(self, cui: str) -> Path:
        return self.cache_dir / self.version / f"{cui}.json"

    def _throttle(self) -> None:
        if self.rate_limit_per_second <= 0:
            return
        minimum_interval = 1.0 / self.rate_limit_per_second
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

    def _load_cache(self, cui: str) -> dict[str, Any] | None:
        path = self._cache_path(cui)
        if not path.exists():
            self.stats["cache_misses"] += 1
            return None
        try:
            payload = _read_json(path)
        except Exception:
            self.stats["cache_misses"] += 1
            return None
        if not isinstance(payload, Mapping):
            self.stats["cache_misses"] += 1
            return None
        if payload.get("umls_version") != self.version:
            self.stats["cache_misses"] += 1
            return None
        label = payload.get("label")
        if not isinstance(label, Mapping):
            self.stats["cache_misses"] += 1
            return None
        self.stats["cache_hits"] += 1
        return dict(label)

    def _write_cache(self, cui: str, label: Mapping[str, Any]) -> None:
        _write_json(
            self._cache_path(cui),
            {
                "schema_version": "umls_external_concept_cache_v1",
                "umls_version": self.version,
                "cui": cui,
                "label": dict(label),
            },
        )

    def get_concept_label(self, cui: str) -> dict[str, Any]:
        cui = _clean(cui).upper()
        cached = self._load_cache(cui)
        if cached is not None:
            return cached

        url = f"{self.base_url}/content/{self.version}/CUI/{cui}"
        transient_statuses = {429, 500, 502, 503, 504}
        last_error = "unknown UMLS API error"

        for attempt in range(self.max_retries + 1):
            if attempt:
                self.stats["api_retries"] += 1
                time.sleep(min(2 ** (attempt - 1), 4))
            self._throttle()
            try:
                self.stats["api_requests"] += 1
                response = self.session.get(
                    url,
                    params={"apiKey": self.api_key},
                    timeout=self.timeout,
                )
                self._last_request_at = time.monotonic()
            except Exception as exc:
                self.stats["api_errors"] += 1
                last_error = _redact_api_key(exc, self.api_key)
                if attempt < self.max_retries:
                    continue
                raise _ExternalLabelTransientError(last_error) from exc

            status = int(getattr(response, "status_code", 0) or 0)
            if status == 200:
                try:
                    payload = response.json()
                except Exception as exc:
                    self.stats["api_errors"] += 1
                    raise _ExternalLabelPermanentError(
                        f"Invalid JSON response for {cui}: "
                        f"{_redact_api_key(exc, self.api_key)}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    self.stats["api_errors"] += 1
                    raise _ExternalLabelPermanentError(
                        f"Invalid UMLS concept payload type for {cui}"
                    )
                label = _normalise_external_label_record(cui, payload)
                self._write_cache(cui, label)
                return label

            if status in {401, 403}:
                self.stats["api_errors"] += 1
                raise RuntimeError(
                    f"UMLS API authentication/authorization failed (HTTP {status})"
                )
            if status == 404:
                self.stats["not_found"] += 1
                raise _ExternalLabelNotFound("HTTP 404")

            self.stats["api_errors"] += 1
            last_error = f"HTTP {status or 'unknown'}"
            if status in transient_statuses and attempt < self.max_retries:
                continue
            if status in transient_statuses:
                raise _ExternalLabelTransientError(last_error)
            raise _ExternalLabelPermanentError(last_error)

        raise _ExternalLabelTransientError(last_error)  # pragma: no cover


def resolve_external_labels(
    *,
    label_plan_path: Path,
    retained_external_cuis_path: Path,
    historical_label_map_path: Path,
    output_dir: Path,
    umls_version: str,
    cache_dir: Path,
    api_timeout: float = 30.0,
    api_rate_limit_per_second: float = 5.0,
    progress_every: int = 25,
    max_retries: int = 2,
    max_consecutive_transient_failures: int = 10,
    resume: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Resolve the C1 external-label plan without touching Neo4j.

    The emitted ``external_cui_labels_v1.json`` is backwards-compatible with
    the historical frozen label artifact while its manifest records C2-specific
    provenance, resume statistics, and explicit unresolved statuses.
    """

    if not label_plan_path.exists():
        raise FileNotFoundError(
            f"External label plan not found: {label_plan_path}. "
            "Run action=build_current_prelabel first."
        )
    if not retained_external_cuis_path.exists():
        raise FileNotFoundError(retained_external_cuis_path)
    if not historical_label_map_path.exists():
        raise FileNotFoundError(historical_label_map_path)

    plan = _read_json(label_plan_path)
    if plan.get("schema_version") != EXTERNAL_LABEL_PLAN_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unexpected external label plan schema: {plan.get('schema_version')}"
        )
    if _clean(plan.get("umls_version")) != _clean(umls_version):
        raise RuntimeError(
            "External label plan UMLS version does not match the configured version"
        )

    retained = sorted(
        {
            line.strip().upper()
            for line in retained_external_cuis_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
    )
    retained_set = set(retained)
    planned = sorted(
        {
            _clean(cui).upper()
            for cui in (plan.get("external_cuis_to_resolve") or [])
            if _clean(cui)
        }
    )
    planned_set = set(planned)

    if len(retained) != int(plan.get("retained_external_cui_count") or -1):
        raise RuntimeError(
            "Retained external CUI count does not match the C1 label plan"
        )
    if len(planned) != int(plan.get("to_resolve_count") or -1):
        raise RuntimeError("C2 target count does not match the C1 label plan")
    if not planned_set.issubset(retained_set):
        raise RuntimeError("C2 label plan contains CUI outside the retained bridge set")

    historical = _read_json(historical_label_map_path)
    historical_version = _clean(historical.get("umls_version"))
    if historical_version and historical_version != _clean(umls_version):
        raise RuntimeError(
            "Historical external label map uses a different UMLS version"
        )
    raw_historical_labels = historical.get("labels") or {}
    if not isinstance(raw_historical_labels, Mapping):
        raise RuntimeError("Historical external label map has invalid labels")
    historical_labels = {
        _clean(cui).upper(): dict(value)
        for cui, value in raw_historical_labels.items()
        if _clean(cui) and isinstance(value, Mapping)
        and _clean(cui).upper() in retained_set
    }

    expected_historical = int(
        plan.get("already_resolved_from_historical_map_count") or 0
    )
    if len(historical_labels) != expected_historical:
        raise RuntimeError(
            "Historical reusable label count does not match the C1 label plan: "
            f"expected={expected_historical}, actual={len(historical_labels)}"
        )
    if set(historical_labels) & planned_set:
        raise RuntimeError(
            "C1 label plan is inconsistent: a planned CUI is already resolved historically"
        )
    if set(historical_labels) | planned_set != retained_set:
        raise RuntimeError(
            "C1 label plan does not partition the retained external CUI set"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = output_dir / "external_cui_labels_v1.json"
    manifest_path = output_dir / "manifest.json"
    unresolved_path = output_dir / "unresolved_cuis.txt"

    combined_labels: dict[str, Any] = dict(historical_labels)
    failures: dict[str, str] = {}
    resumed_labels: set[str] = set()

    if resume and label_map_path.exists():
        prior = _read_json(label_map_path)
        if (
            prior.get("schema_version") != EXTERNAL_LABEL_MAP_SCHEMA_VERSION
            or _clean(prior.get("umls_version")) != _clean(umls_version)
        ):
            raise RuntimeError(
                "Existing C2 label artifact has incompatible schema/version; "
                "move it aside before rerunning"
            )
        raw_prior_labels = prior.get("labels") or {}
        if isinstance(raw_prior_labels, Mapping):
            for cui, value in raw_prior_labels.items():
                norm = _clean(cui).upper()
                if (
                    norm in retained_set
                    and norm not in combined_labels
                    and isinstance(value, Mapping)
                ):
                    combined_labels[norm] = dict(value)
                    resumed_labels.add(norm)
        raw_prior_failures = prior.get("failures") or {}
        if isinstance(raw_prior_failures, Mapping):
            failures = {
                _clean(cui).upper(): _redact_api_key(error)
                for cui, error in raw_prior_failures.items()
                if _clean(cui).upper() in retained_set
            }

    if client is None:
        client = UMLSExternalLabelClient(
            cache_dir=cache_dir,
            version=umls_version,
            timeout=api_timeout,
            rate_limit_per_second=api_rate_limit_per_second,
            max_retries=max_retries,
        )

    target_this_run = [cui for cui in planned if cui not in combined_labels]
    resolved_this_run: set[str] = set()
    attempted_this_run = 0
    consecutive_transient_failures = 0
    status_by_cui: dict[str, str] = {
        cui: "historical_reuse" for cui in historical_labels
    }
    status_by_cui.update({cui: "resumed_reuse" for cui in resumed_labels})

    def write_checkpoint(*, complete: bool) -> dict[str, Any]:
        unresolved = sorted(retained_set - set(combined_labels))
        active_failures = {
            cui: failures[cui]
            for cui in unresolved
            if cui in failures
        }
        label_payload = {
            "schema_version": EXTERNAL_LABEL_MAP_SCHEMA_VERSION,
            "umls_version": umls_version,
            "labels": dict(sorted(combined_labels.items())),
            "failures": dict(sorted(active_failures.items())),
            "requested_cui_count": len(retained),
            "resolved_cui_count": len(combined_labels),
            "failure_count": len(active_failures),
        }
        _write_json(label_map_path, label_payload)
        unresolved_path.write_text(
            "\n".join(unresolved) + ("\n" if unresolved else ""),
            encoding="utf-8",
        )

        client_stats = dict(getattr(client, "stats", {}) or {})
        status_counts = Counter(status_by_cui.values())
        status_counts["unresolved"] = len(unresolved)
        failure_reason_counts = Counter(
            error.split(":", 1)[0] for error in active_failures.values()
        )
        manifest = {
            "schema_version": EXTERNAL_LABEL_RESOLUTION_SCHEMA_VERSION,
            "umls_version": umls_version,
            "complete": bool(complete),
            "inputs": {
                "label_plan": str(label_plan_path.resolve()),
                "label_plan_sha256": _sha256_file(label_plan_path),
                "retained_external_cuis": str(
                    retained_external_cuis_path.resolve()
                ),
                "retained_external_cuis_sha256": _sha256_file(
                    retained_external_cuis_path
                ),
                "historical_label_map": str(
                    historical_label_map_path.resolve()
                ),
                "historical_label_map_sha256": _sha256_file(
                    historical_label_map_path
                ),
            },
            "counts": {
                "retained_external_cui_count": len(retained),
                "historical_reused_count": len(historical_labels),
                "planned_resolution_count": len(planned),
                "resumed_reused_count": len(resumed_labels),
                "attempted_this_run_count": attempted_this_run,
                "resolved_this_run_count": len(resolved_this_run),
                "resolved_total_count": len(combined_labels),
                "unresolved_count": len(unresolved),
                "failure_record_count": len(active_failures),
            },
            "status_counts": dict(sorted(status_counts.items())),
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            "client_stats": client_stats,
            "outputs": {
                "external_label_map": str(label_map_path.resolve()),
                "unresolved_cuis": str(unresolved_path.resolve()),
            },
            "safety": {
                "neo4j_reads": False,
                "neo4j_writes": False,
                "retrieval_metrics_used": False,
            },
        }
        _write_json(manifest_path, manifest)
        manifest["outputs"]["external_label_map_sha256"] = _sha256_file(
            label_map_path
        )
        _write_json(manifest_path, manifest)
        return manifest

    total = len(target_this_run)
    for index, cui in enumerate(target_this_run, start=1):
        attempted_this_run += 1
        try:
            label = client.get_concept_label(cui)
        except _ExternalLabelNotFound as exc:
            failures[cui] = f"not_found: {_redact_api_key(exc)}"
            status_by_cui[cui] = "not_found"
            consecutive_transient_failures = 0
        except _ExternalLabelTransientError as exc:
            failures[cui] = f"transient_error: {_redact_api_key(exc)}"
            status_by_cui[cui] = "transient_error"
            consecutive_transient_failures += 1
            if (
                max_consecutive_transient_failures > 0
                and consecutive_transient_failures
                >= max_consecutive_transient_failures
            ):
                write_checkpoint(complete=False)
                raise RuntimeError(
                    "Aborting C2 after sustained transient UMLS failures; "
                    "checkpoint written and rerun will resume"
                ) from exc
        except _ExternalLabelPermanentError as exc:
            failures[cui] = f"permanent_error: {_redact_api_key(exc)}"
            status_by_cui[cui] = "permanent_error"
            consecutive_transient_failures = 0
        except Exception:
            # Authentication/configuration errors are fatal. Preserve completed
            # work, but do not convert them into thousands of per-CUI failures.
            write_checkpoint(complete=False)
            raise
        else:
            combined_labels[cui] = dict(label)
            failures.pop(cui, None)
            resolved_this_run.add(cui)
            status_by_cui[cui] = "resolved_this_run"
            consecutive_transient_failures = 0

        if progress_every > 0 and (
            index % progress_every == 0 or index == total
        ):
            write_checkpoint(complete=False)
            logger.info(
                "External UMLS label progress | processed=%d/%d | "
                "resolved_total=%d | unresolved=%d",
                index,
                total,
                len(combined_labels),
                len(retained_set - set(combined_labels)),
            )

    final_manifest = write_checkpoint(complete=True)
    logger.info(
        "External UMLS label resolution complete | retained=%d | historical=%d | "
        "planned=%d | resolved_total=%d | unresolved=%d",
        len(retained),
        len(historical_labels),
        len(planned),
        len(combined_labels),
        final_manifest["counts"]["unresolved_count"],
    )
    return final_manifest



def _read_cui_lines(path: Path) -> list[str]:
    return sorted(
        {
            _clean(line).upper()
            for line in path.read_text(encoding="utf-8").splitlines()
            if _clean(line)
        }
    )


def _require_external_label_resolution_complete(
    *,
    manifest_path: Path,
    retained_external_cuis_path: Path,
    external_label_map_path: Path,
    umls_version: str,
) -> dict[str, Any]:
    """Validate the C2 artifact before the label-aware C3 build."""

    for path in (
        manifest_path,
        retained_external_cuis_path,
        external_label_map_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EXTERNAL_LABEL_RESOLUTION_SCHEMA_VERSION:
        raise RuntimeError(
            "Unexpected C2 manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if not bool(manifest.get("complete")):
        raise RuntimeError("C2 external-label resolution is not complete")
    if _clean(manifest.get("umls_version")) != _clean(umls_version):
        raise RuntimeError(
            "C2 UMLS version mismatch: "
            f"expected={umls_version}, actual={manifest.get('umls_version')}"
        )

    counts = manifest.get("counts") or {}
    if int(counts.get("unresolved_count") or 0) != 0:
        raise RuntimeError(
            "C2 still has unresolved external CUIs: "
            f"{counts.get('unresolved_count')}"
        )
    if int(counts.get("failure_record_count") or 0) != 0:
        raise RuntimeError(
            "C2 still has failure records: "
            f"{counts.get('failure_record_count')}"
        )

    expected_label_sha = _clean(
        (manifest.get("outputs") or {}).get("external_label_map_sha256")
    )
    actual_label_sha = _sha256_file(external_label_map_path)
    if expected_label_sha and actual_label_sha != expected_label_sha:
        raise RuntimeError("C2 external-label map SHA-256 mismatch")

    expected_retained_sha = _clean(
        (manifest.get("inputs") or {}).get("retained_external_cuis_sha256")
    )
    actual_retained_sha = _sha256_file(retained_external_cuis_path)
    if expected_retained_sha and actual_retained_sha != expected_retained_sha:
        raise RuntimeError("C2 retained-external-CUI SHA-256 mismatch")

    label_payload = _read_json(external_label_map_path)
    if label_payload.get("schema_version") != EXTERNAL_LABEL_MAP_SCHEMA_VERSION:
        raise RuntimeError(
            "Unexpected external-label map schema: "
            f"{label_payload.get('schema_version')!r}"
        )
    labels = label_payload.get("labels") or {}
    failures = label_payload.get("failures") or {}
    if not isinstance(labels, Mapping):
        raise RuntimeError("C2 external-label map has invalid labels payload")
    if failures:
        raise RuntimeError(
            "C2 external-label map still contains failure records"
        )

    retained = _read_cui_lines(retained_external_cuis_path)
    retained_set = set(retained)
    label_set = {
        _clean(cui).upper()
        for cui in labels
        if _clean(cui)
    }
    missing = sorted(retained_set - label_set)
    extra = sorted(label_set - retained_set)
    if missing or extra:
        raise RuntimeError(
            "C2 label coverage does not exactly match the retained external "
            f"CUI set: missing={len(missing)}, extra={len(extra)}"
        )

    for cui in retained:
        row = labels.get(cui)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"Invalid external-label record for {cui}")
        if _clean(row.get("ui")).upper() != cui:
            raise RuntimeError(f"External-label CUI mismatch for {cui}")
        if not _clean(row.get("name")):
            raise RuntimeError(f"External-label preferred name missing for {cui}")
        semantic_types = row.get("semantic_types") or []
        if not semantic_types:
            raise RuntimeError(f"External-label semantic types missing for {cui}")

    if int(counts.get("retained_external_cui_count") or 0) != len(retained):
        raise RuntimeError(
            "C2 retained external CUI count mismatch: "
            f"manifest={counts.get('retained_external_cui_count')}, "
            f"actual={len(retained)}"
        )
    if int(counts.get("resolved_total_count") or 0) != len(retained):
        raise RuntimeError(
            "C2 resolved total count mismatch: "
            f"manifest={counts.get('resolved_total_count')}, actual={len(retained)}"
        )

    safety = manifest.get("safety") or {}
    if bool(safety.get("neo4j_writes")) or bool(
        safety.get("retrieval_metrics_used")
    ):
        raise RuntimeError("C2 safety invariants are not satisfied")

    return manifest


def _final_bridge_audit_summary(
    *,
    final_artifact_dir: Path,
    current_scope_path: Path,
) -> dict[str, Any]:
    """Produce corpus-generic structural audit counters for C3.

    This intentionally uses only artifact semantics and provenance.  No
    retrieval benchmark data are consumed.
    """

    scope = _read_json(current_scope_path)
    cui_docs = {
        _clean(row.get("cui")): tuple(
            _normalise_doc_ids(row.get("document_ids"))
        )
        for row in (scope.get("cuis") or [])
        if isinstance(row, Mapping) and _clean(row.get("cui"))
    }

    def endpoint_membership(cui: str) -> str:
        docs = cui_docs.get(cui, ())
        if not docs:
            return "unassigned"
        if len(docs) == 1:
            return docs[0]
        return "shared"

    def pair_membership(a: str, b: str) -> str:
        return " <-> ".join(sorted((endpoint_membership(a), endpoint_membership(b))))

    pair_counts: Counter[str] = Counter()
    retained_pair_counts: Counter[str] = Counter()
    default_only_retained_counts: Counter[str] = Counter()
    tier_counts: dict[str, Counter[str]] = {}

    with (final_artifact_dir / "pair_policy_audit.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            membership = pair_membership(
                _clean(row.get("local_cui_a")),
                _clean(row.get("local_cui_b")),
            )
            pair_counts[membership] += 1
            tier_counts.setdefault(membership, Counter())[
                _clean(row.get("overall_tier")) or "UNKNOWN"
            ] += 1
            retained = _clean(row.get("retained_for_retrieval")).lower() == "true"
            if not retained:
                continue
            retained_pair_counts[membership] += 1

    hub_stats: dict[tuple[str, str], dict[str, Any]] = {}
    pair_evidence_path = final_artifact_dir / "pair_evidence.jsonl"
    with pair_evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)

            # ``default-only`` is a property of the evidence that actually
            # retains a pair in the balanced profile.  Audit CSV rule counts
            # may also contain WEAK/REJECT rules, which must not hide the fact
            # that all MEDIUM/STRONG support came from source defaults.
            external_paths = row.get("external_path_evidence") or []
            pair_is_retained = bool(row.get("retained_for_retrieval")) or any(
                bool(evidence.get("retained_for_retrieval"))
                for evidence in external_paths
            )
            if pair_is_retained:
                retained_rule_ids: set[str] = set()
                for evidence in external_paths:
                    for relation_evidence in (
                        evidence.get("relation_pair_evidence") or []
                    ):
                        if _clean(relation_evidence.get("tier")) not in {
                            "MEDIUM",
                            "STRONG",
                        }:
                            continue
                        rule_id = _clean(relation_evidence.get("policy_rule_id"))
                        if rule_id:
                            retained_rule_ids.add(rule_id)
                if retained_rule_ids and all(
                    rule_id.endswith("_default") for rule_id in retained_rule_ids
                ):
                    membership = pair_membership(
                        _clean(row.get("local_cui_a")),
                        _clean(row.get("local_cui_b")),
                    )
                    default_only_retained_counts[membership] += 1

            for evidence in external_paths:
                if not bool(evidence.get("retained_for_retrieval")):
                    continue
                source = _clean(evidence.get("source_vocabulary"))
                cui = _clean(evidence.get("external_cui"))
                if not source or not cui:
                    continue
                key = (source, cui)
                item = hub_stats.setdefault(
                    key,
                    {
                        "source_vocabulary": source,
                        "external_cui": cui,
                        "external_preferred_name": evidence.get(
                            "external_preferred_name"
                        ),
                        "external_semantic_type_names": evidence.get(
                            "external_semantic_type_names"
                        )
                        or [],
                        "external_hub_degree": int(
                            evidence.get("external_hub_degree") or 0
                        ),
                        "retained_local_pair_count": 0,
                        "policy_rule_counts": Counter(),
                    },
                )
                item["retained_local_pair_count"] += 1
                for sig in evidence.get("relation_pair_evidence") or []:
                    if _clean(sig.get("tier")) not in {"STRONG", "MEDIUM"}:
                        continue
                    item["policy_rule_counts"][
                        _clean(sig.get("policy_rule_id")) or "unknown"
                    ] += 1

    top_hubs = []
    for item in sorted(
        hub_stats.values(),
        key=lambda row: (
            -int(row["retained_local_pair_count"]),
            row["source_vocabulary"],
            row["external_cui"],
        ),
    )[:25]:
        top_hubs.append(
            {
                **item,
                "policy_rule_counts": dict(
                    sorted(item["policy_rule_counts"].items())
                ),
            }
        )

    total_retained = sum(retained_pair_counts.values())
    total_default_only = sum(default_only_retained_counts.values())
    return {
        "pair_counts": dict(sorted(pair_counts.items())),
        "retained_pair_counts": dict(sorted(retained_pair_counts.items())),
        "tier_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(tier_counts.items())
        },
        "default_only_retained": {
            "count": total_default_only,
            "retained_pair_count": total_retained,
            "fraction": (
                total_default_only / total_retained if total_retained else 0.0
            ),
            "counts_by_membership": dict(
                sorted(default_only_retained_counts.items())
            ),
            "fraction_by_membership": {
                key: (
                    default_only_retained_counts.get(key, 0) / count
                    if count
                    else 0.0
                )
                for key, count in sorted(retained_pair_counts.items())
            },
        },
        "top_retained_external_hubs": top_hubs,
        "safety": {
            "neo4j_reads": False,
            "neo4j_writes": False,
            "umls_api_calls": False,
            "retrieval_metrics_used": False,
        },
    }


def run_current_final_build(
    *,
    project_root: Path,
    out_root: Path,
    current_scope_path: Path,
    build_paths: CurrentFinalBuildPaths,
    sources: Sequence[str],
    umls_version: str,
) -> dict[str, Any]:
    """Build and validate the frozen label-aware C3 bridge artifact."""

    current_build_root = out_root / "current_build_v1"
    prelabel_dir = current_build_root / "bridge_prelabel_artifact"
    evidence_root = current_build_root / "bridge_evidence"
    direct_manifest_path = current_build_root / "direct_artifact" / "manifest.json"
    prelabel_manifest_path = prelabel_dir / "manifest.json"
    prelabel_report_path = current_build_root / "prelabel_build_report.json"
    c2_manifest_path = current_build_root / "external_labels" / "manifest.json"
    external_label_map_path = (
        current_build_root / "external_labels" / "external_cui_labels_v1.json"
    )
    retained_external_cuis_path = prelabel_dir / "retained_external_cuis.txt"

    for path in (
        current_scope_path,
        evidence_root / "manifest.json",
        direct_manifest_path,
        prelabel_manifest_path,
        prelabel_report_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    c2_manifest = _require_external_label_resolution_complete(
        manifest_path=c2_manifest_path,
        retained_external_cuis_path=retained_external_cuis_path,
        external_label_map_path=external_label_map_path,
        umls_version=umls_version,
    )

    prelabel_manifest = _read_json(prelabel_manifest_path)
    prelabel_report = _read_json(prelabel_report_path)
    if prelabel_manifest.get("schema_version") != "ontology_bridge_artifact_v1":
        raise RuntimeError("Unexpected C1 pre-label bridge schema")
    if prelabel_report.get("schema_version") != CURRENT_PRELABEL_SCHEMA_VERSION:
        raise RuntimeError("Unexpected C1 pre-label build-report schema")

    current_scope_sha = _sha256_file(current_scope_path)
    if _clean(prelabel_manifest.get("local_scope_sha256")) != current_scope_sha:
        raise RuntimeError("C1 pre-label scope fingerprint no longer matches")
    if _clean(prelabel_report.get("current_scope_sha256")) != current_scope_sha:
        raise RuntimeError("C1 pre-label report scope fingerprint no longer matches")

    direct_manifest_sha_before = _sha256_file(direct_manifest_path)
    prelabel_manifest_sha_before = _sha256_file(prelabel_manifest_path)
    c2_manifest_sha_before = _sha256_file(c2_manifest_path)
    label_map_sha_before = _sha256_file(external_label_map_path)

    final_dir = current_build_root / "bridge_final_artifact"
    if final_dir.exists():
        shutil.rmtree(final_dir)

    command = [
        sys.executable,
        str(build_paths.bridge_final_builder),
        "--bridge-root",
        str(evidence_root),
        "--local-scope",
        str(current_scope_path),
        "--policy",
        str(build_paths.bridge_final_policy),
        "--external-label-map",
        str(external_label_map_path),
        "--output-dir",
        str(final_dir),
        "--sources",
        *sources,
    ]
    _run_builder(command, cwd=project_root)

    final_manifest_path = final_dir / "manifest.json"
    if not final_manifest_path.exists():
        raise RuntimeError("C3 builder did not produce a final manifest")
    final_manifest = _read_json(final_manifest_path)

    if final_manifest.get("schema_version") != "ontology_bridge_artifact_v1_1":
        raise RuntimeError("Unexpected C3 final bridge schema")
    if _clean(final_manifest.get("local_scope_sha256")) != current_scope_sha:
        raise RuntimeError("C3 final bridge scope fingerprint mismatch")
    if _clean(final_manifest.get("policy_sha256")) != _sha256_file(
        build_paths.bridge_final_policy
    ):
        raise RuntimeError("C3 final bridge policy fingerprint mismatch")
    if _clean(final_manifest.get("external_label_map_sha256")) != label_map_sha_before:
        raise RuntimeError("C3 final bridge external-label fingerprint mismatch")

    expected_sources = list(sources)
    if list(final_manifest.get("sources_requested") or []) != expected_sources:
        raise RuntimeError(
            "C3 final bridge source list mismatch: "
            f"expected={expected_sources}, "
            f"actual={final_manifest.get('sources_requested')}"
        )

    prelabel_candidates = int(
        prelabel_manifest.get("all_distinct_local_pair_count") or 0
    )
    final_candidates = int(final_manifest.get("all_distinct_local_pair_count") or 0)
    if final_candidates != prelabel_candidates:
        raise RuntimeError(
            "C3 candidate universe changed during label-aware classification: "
            f"prelabel={prelabel_candidates}, final={final_candidates}"
        )

    final_safety = final_manifest.get("safety") or {}
    forbidden = {
        "umls_api_calls": bool(final_safety.get("umls_api_calls")),
        "neo4j_writes": bool(final_safety.get("neo4j_writes")),
        "second_hop_requests": bool(final_safety.get("second_hop_requests")),
        "retrieval_metrics_used": bool(final_safety.get("retrieval_metrics_used")),
        "benchmark_tuned": bool(final_safety.get("benchmark_tuned")),
    }
    if any(forbidden.values()):
        raise RuntimeError(f"C3 safety invariant violated: {forbidden}")

    if _sha256_file(direct_manifest_path) != direct_manifest_sha_before:
        raise RuntimeError("C3 unexpectedly modified the DIRECT artifact")
    if _sha256_file(prelabel_manifest_path) != prelabel_manifest_sha_before:
        raise RuntimeError("C3 unexpectedly modified the C1 pre-label artifact")
    if _sha256_file(c2_manifest_path) != c2_manifest_sha_before:
        raise RuntimeError("C3 unexpectedly modified the C2 manifest")
    if _sha256_file(external_label_map_path) != label_map_sha_before:
        raise RuntimeError("C3 unexpectedly modified the C2 external-label map")

    audit = _final_bridge_audit_summary(
        final_artifact_dir=final_dir,
        current_scope_path=current_scope_path,
    )
    report = {
        "schema_version": CURRENT_FINAL_SCHEMA_VERSION,
        "scope_name": _clean(prelabel_report.get("scope_name")) or "current_corpus",
        "document_ids": _normalise_doc_ids(prelabel_report.get("document_ids")),
        "umls_version": umls_version,
        "inputs": {
            "current_scope": str(current_scope_path.resolve()),
            "current_scope_sha256": current_scope_sha,
            "bridge_evidence_manifest_sha256": _sha256_file(
                evidence_root / "manifest.json"
            ),
            "direct_manifest_sha256": direct_manifest_sha_before,
            "prelabel_manifest_sha256": prelabel_manifest_sha_before,
            "c2_manifest_sha256": c2_manifest_sha_before,
            "external_label_map_sha256": label_map_sha_before,
            "bridge_final_builder": str(
                build_paths.bridge_final_builder.resolve()
            ),
            "bridge_final_builder_sha256": _sha256_file(
                build_paths.bridge_final_builder
            ),
            "bridge_final_policy": str(build_paths.bridge_final_policy.resolve()),
            "bridge_final_policy_sha256": _sha256_file(
                build_paths.bridge_final_policy
            ),
        },
        "c2": {
            "complete": bool(c2_manifest.get("complete")),
            "counts": c2_manifest.get("counts") or {},
        },
        "prelabel": {
            "all_distinct_local_pair_count": prelabel_candidates,
            "retained_distinct_local_pair_count": int(
                prelabel_manifest.get("retained_distinct_local_pair_count") or 0
            ),
            "retained_local_cui_count": int(
                prelabel_manifest.get("retained_local_cui_count") or 0
            ),
            "retained_external_cui_count": int(
                prelabel_manifest.get("retained_external_cui_count") or 0
            ),
            "overall_tier_counts": prelabel_manifest.get("overall_tier_counts") or {},
        },
        "final": {
            "artifact_dir": str(final_dir.resolve()),
            "manifest_sha256": _sha256_file(final_manifest_path),
            "all_distinct_local_pair_count": final_candidates,
            "retained_distinct_local_pair_count": int(
                final_manifest.get("retained_distinct_local_pair_count") or 0
            ),
            "retained_local_cui_count": int(
                final_manifest.get("retained_local_cui_count") or 0
            ),
            "retained_external_cui_count": int(
                final_manifest.get("retained_external_cui_count") or 0
            ),
            "overall_tier_counts": final_manifest.get("overall_tier_counts") or {},
            "retrieval_profiles": final_manifest.get("retrieval_profiles") or {},
        },
        "delta_final_minus_prelabel": {
            "retained_distinct_local_pair_count": int(
                final_manifest.get("retained_distinct_local_pair_count") or 0
            )
            - int(prelabel_manifest.get("retained_distinct_local_pair_count") or 0),
            "retained_local_cui_count": int(
                final_manifest.get("retained_local_cui_count") or 0
            )
            - int(prelabel_manifest.get("retained_local_cui_count") or 0),
            "retained_external_cui_count": int(
                final_manifest.get("retained_external_cui_count") or 0
            )
            - int(prelabel_manifest.get("retained_external_cui_count") or 0),
            "tier_counts": {
                tier: int((final_manifest.get("overall_tier_counts") or {}).get(tier, 0))
                - int((prelabel_manifest.get("overall_tier_counts") or {}).get(tier, 0))
                for tier in ("STRONG", "MEDIUM", "WEAK", "REJECT")
            },
        },
        "audit": audit,
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "retrieval_metrics_used": False,
            "benchmark_tuned": False,
        },
    }
    report_path = current_build_root / "current_final_report.json"
    _write_json(report_path, report)
    logger.info(
        "Current final UMLS bridge build complete | candidates=%d | retained=%d | "
        "default_only=%d/%d | output=%s",
        final_candidates,
        report["final"]["retained_distinct_local_pair_count"],
        audit["default_only_retained"]["count"],
        audit["default_only_retained"]["retained_pair_count"],
        final_dir,
    )
    return report


def run_current_generalized_v2_build(
    *,
    project_root: Path,
    out_root: Path,
    current_scope_path: Path,
    build_paths: GeneralizedBridgeBuildPaths,
    sources: Sequence[str],
    umls_version: str,
) -> dict[str, Any]:
    """Build v2_general beside, never over, the frozen C3/v1.1 baseline.

    The stage is deliberately artifact-only: it reuses the exact C1 evidence and
    C2 external labels, requires the frozen C3 fingerprints to match, performs
    no UMLS calls or Neo4j access, and never consumes retrieval metrics.
    """

    current_build_root = out_root / "current_build_v1"
    evidence_root = current_build_root / "bridge_evidence"
    direct_manifest_path = current_build_root / "direct_artifact" / "manifest.json"
    prelabel_manifest_path = (
        current_build_root / "bridge_prelabel_artifact" / "manifest.json"
    )
    c2_manifest_path = current_build_root / "external_labels" / "manifest.json"
    external_label_map_path = (
        current_build_root / "external_labels" / "external_cui_labels_v1.json"
    )
    frozen_report_path = current_build_root / "current_final_report.json"
    frozen_dir = current_build_root / "bridge_final_artifact"
    frozen_manifest_path = frozen_dir / "manifest.json"

    for path in (
        current_scope_path,
        evidence_root / "manifest.json",
        direct_manifest_path,
        prelabel_manifest_path,
        c2_manifest_path,
        external_label_map_path,
        frozen_report_path,
        frozen_manifest_path,
        build_paths.bridge_builder,
        build_paths.bridge_policy,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    frozen_report = _read_json(frozen_report_path)
    frozen_manifest = _read_json(frozen_manifest_path)
    if frozen_report.get("schema_version") != CURRENT_FINAL_SCHEMA_VERSION:
        raise RuntimeError("Frozen C3 report schema mismatch")
    if frozen_manifest.get("schema_version") != "ontology_bridge_artifact_v1_1":
        raise RuntimeError("Frozen C3 bridge manifest schema mismatch")

    frozen_manifest_sha = _sha256_file(frozen_manifest_path)
    frozen_policy_sha = _clean(
        (frozen_report.get("inputs") or {}).get("bridge_final_policy_sha256")
    )
    expected_manifest_sha = _clean(build_paths.frozen_c3_manifest_sha256)
    expected_policy_sha = _clean(build_paths.frozen_v1_1_policy_sha256)
    if expected_manifest_sha and frozen_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            "Frozen C3 manifest fingerprint changed: "
            f"expected={expected_manifest_sha}, actual={frozen_manifest_sha}"
        )
    if expected_policy_sha and frozen_policy_sha != expected_policy_sha:
        raise RuntimeError(
            "Frozen v1.1 policy fingerprint changed: "
            f"expected={expected_policy_sha}, actual={frozen_policy_sha}"
        )

    generalized_policy_sha = _sha256_file(build_paths.bridge_policy)
    if generalized_policy_sha == frozen_policy_sha:
        raise RuntimeError(
            "Generalized policy is byte-identical to the frozen v1.1 policy"
        )

    current_scope_sha = _sha256_file(current_scope_path)
    if _clean(frozen_manifest.get("local_scope_sha256")) != current_scope_sha:
        raise RuntimeError("Frozen C3 scope fingerprint no longer matches")
    label_map_sha = _sha256_file(external_label_map_path)
    if _clean(frozen_manifest.get("external_label_map_sha256")) != label_map_sha:
        raise RuntimeError("Frozen C3 label-map fingerprint no longer matches")

    protected_paths = {
        "direct_manifest": direct_manifest_path,
        "prelabel_manifest": prelabel_manifest_path,
        "c2_manifest": c2_manifest_path,
        "external_label_map": external_label_map_path,
        "frozen_c3_report": frozen_report_path,
        "frozen_c3_manifest": frozen_manifest_path,
    }
    protected_before = {name: _sha256_file(path) for name, path in protected_paths.items()}

    generalized_dir = current_build_root / "bridge_generalized_v2_artifact"
    if generalized_dir.exists():
        shutil.rmtree(generalized_dir)

    command = [
        sys.executable,
        str(build_paths.bridge_builder.resolve()),
        "--bridge-root",
        str(evidence_root.resolve()),
        "--local-scope",
        str(current_scope_path.resolve()),
        "--policy",
        str(build_paths.bridge_policy.resolve()),
        "--external-label-map",
        str(external_label_map_path.resolve()),
        "--output-dir",
        str(generalized_dir.resolve()),
        "--sources",
        *sources,
    ]
    _run_builder(command, cwd=project_root)

    generalized_manifest_path = generalized_dir / "manifest.json"
    if not generalized_manifest_path.exists():
        raise RuntimeError("v2_general builder did not produce a manifest")
    generalized_manifest = _read_json(generalized_manifest_path)
    if generalized_manifest.get("schema_version") != "ontology_bridge_artifact_v1_1":
        raise RuntimeError("Unexpected v2_general bridge artifact schema")
    if _clean(generalized_manifest.get("local_scope_sha256")) != current_scope_sha:
        raise RuntimeError("v2_general scope fingerprint mismatch")
    if _clean(generalized_manifest.get("policy_sha256")) != generalized_policy_sha:
        raise RuntimeError("v2_general policy fingerprint mismatch")
    if _clean(generalized_manifest.get("external_label_map_sha256")) != label_map_sha:
        raise RuntimeError("v2_general label-map fingerprint mismatch")
    if list(generalized_manifest.get("sources_requested") or []) != list(sources):
        raise RuntimeError("v2_general source list mismatch")

    frozen_candidates = int(frozen_manifest.get("all_distinct_local_pair_count") or 0)
    generalized_candidates = int(
        generalized_manifest.get("all_distinct_local_pair_count") or 0
    )
    if generalized_candidates != frozen_candidates:
        raise RuntimeError(
            "v2_general candidate universe changed relative to frozen C3: "
            f"frozen={frozen_candidates}, generalized={generalized_candidates}"
        )

    safety = generalized_manifest.get("safety") or {}
    forbidden = {
        "umls_api_calls": bool(safety.get("umls_api_calls")),
        "neo4j_writes": bool(safety.get("neo4j_writes")),
        "second_hop_requests": bool(safety.get("second_hop_requests")),
        "retrieval_metrics_used": bool(safety.get("retrieval_metrics_used")),
        "benchmark_tuned": bool(safety.get("benchmark_tuned")),
    }
    if any(forbidden.values()):
        raise RuntimeError(f"v2_general safety invariant violated: {forbidden}")

    for name, path in protected_paths.items():
        after = _sha256_file(path)
        if after != protected_before[name]:
            raise RuntimeError(f"v2_general unexpectedly modified protected input: {name}")

    frozen_audit = frozen_report.get("audit") or _final_bridge_audit_summary(
        final_artifact_dir=frozen_dir, current_scope_path=current_scope_path
    )
    generalized_audit = _final_bridge_audit_summary(
        final_artifact_dir=generalized_dir,
        current_scope_path=current_scope_path,
    )

    frozen_retained = int(frozen_manifest.get("retained_distinct_local_pair_count") or 0)
    generalized_retained = int(
        generalized_manifest.get("retained_distinct_local_pair_count") or 0
    )
    frozen_tiers = frozen_manifest.get("overall_tier_counts") or {}
    generalized_tiers = generalized_manifest.get("overall_tier_counts") or {}

    policy_payload = _read_json(build_paths.bridge_policy)
    report = {
        "schema_version": GENERALIZED_V2_SCHEMA_VERSION,
        "scope_name": _clean(frozen_report.get("scope_name")) or "current_corpus",
        "document_ids": _normalise_doc_ids(frozen_report.get("document_ids")),
        "umls_version": umls_version,
        "methodology": {
            "design_basis": "semantic_and_structural_audit_only",
            "retrieval_metrics_used": False,
            "benchmark_tuned": False,
            "frozen_baseline_policy_sha256": frozen_policy_sha,
            "frozen_baseline_manifest_sha256": frozen_manifest_sha,
            "policy_design_constraints": policy_payload.get("design_constraints") or {},
        },
        "inputs": {
            "current_scope_sha256": current_scope_sha,
            "bridge_evidence_manifest_sha256": _sha256_file(evidence_root / "manifest.json"),
            "external_label_map_sha256": label_map_sha,
            "bridge_builder": str(build_paths.bridge_builder.resolve()),
            "bridge_builder_sha256": _sha256_file(build_paths.bridge_builder),
            "generalized_policy": str(build_paths.bridge_policy.resolve()),
            "generalized_policy_sha256": generalized_policy_sha,
        },
        "frozen_v1_1": {
            "artifact_dir": str(frozen_dir.resolve()),
            "manifest_sha256": frozen_manifest_sha,
            "all_distinct_local_pair_count": frozen_candidates,
            "retained_distinct_local_pair_count": frozen_retained,
            "retained_local_cui_count": int(frozen_manifest.get("retained_local_cui_count") or 0),
            "retained_external_cui_count": int(frozen_manifest.get("retained_external_cui_count") or 0),
            "overall_tier_counts": frozen_tiers,
            "retrieval_profiles": frozen_manifest.get("retrieval_profiles") or {},
            "audit": frozen_audit,
        },
        "generalized_v2": {
            "artifact_dir": str(generalized_dir.resolve()),
            "manifest_sha256": _sha256_file(generalized_manifest_path),
            "all_distinct_local_pair_count": generalized_candidates,
            "retained_distinct_local_pair_count": generalized_retained,
            "retained_local_cui_count": int(generalized_manifest.get("retained_local_cui_count") or 0),
            "retained_external_cui_count": int(generalized_manifest.get("retained_external_cui_count") or 0),
            "overall_tier_counts": generalized_tiers,
            "retrieval_profiles": generalized_manifest.get("retrieval_profiles") or {},
            "audit": generalized_audit,
        },
        "delta_v2_minus_frozen_v1_1": {
            "retained_distinct_local_pair_count": generalized_retained - frozen_retained,
            "retained_local_cui_count": int(generalized_manifest.get("retained_local_cui_count") or 0)
            - int(frozen_manifest.get("retained_local_cui_count") or 0),
            "retained_external_cui_count": int(generalized_manifest.get("retained_external_cui_count") or 0)
            - int(frozen_manifest.get("retained_external_cui_count") or 0),
            "tier_counts": {
                tier: int(generalized_tiers.get(tier, 0)) - int(frozen_tiers.get(tier, 0))
                for tier in ("STRONG", "MEDIUM", "WEAK", "REJECT")
            },
            "default_only_retained_count": int(
                (generalized_audit.get("default_only_retained") or {}).get("count") or 0
            )
            - int((frozen_audit.get("default_only_retained") or {}).get("count") or 0),
        },
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "retrieval_metrics_used": False,
            "benchmark_tuned": False,
            "frozen_v1_1_modified": False,
        },
    }
    report_path = current_build_root / "current_generalized_v2_report.json"
    _write_json(report_path, report)
    logger.info(
        "Generalized v2 UMLS bridge build complete | candidates=%d | retained=%d | "
        "default_only=%d/%d | output=%s",
        generalized_candidates,
        generalized_retained,
        generalized_audit["default_only_retained"]["count"],
        generalized_audit["default_only_retained"]["retained_pair_count"],
        generalized_dir,
    )
    return report


def _require_delta_discovery_complete(
    *,
    manifest_path: Path,
    current_scope_path: Path,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Require a complete, safe delta discovery for the current scope.

    Compatibility note:
    Phase-B v1 manifests did not copy ``processed_local_cui_count`` and
    ``fetch_failure_count`` into ``manifest["source_results"]``.  The
    authoritative completion record is the per-source ``summary.json``.
    Validate that file directly so already-complete discovery evidence is not
    invalidated by an older aggregate-manifest shape.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Delta discovery manifest not found: {manifest_path}. "
            "Run action=delta_discovery first."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != DELTA_DISCOVERY_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unexpected delta discovery schema: {manifest.get('schema_version')}"
        )

    scope = manifest.get("scope") or {}
    expected_sha = _sha256_file(current_scope_path)
    if scope.get("current_scope_sha256") != expected_sha:
        raise RuntimeError(
            "Delta discovery was built for a different current local UMLS scope."
        )

    expected_scope_count = int(scope.get("current_cui_count") or 0)
    expected_delta_count = int(scope.get("new_cui_count") or 0)
    frozen = manifest.get("frozen_parameters") or {}
    source_profile_sha = _clean(frozen.get("source_profile_sha256"))
    umls_version = _clean(frozen.get("umls_version"))
    max_relations_per_cui = int(frozen.get("max_relations_per_cui") or 0)
    source_results = manifest.get("source_results") or {}

    missing_sources: list[str] = []
    incomplete_sources: dict[str, list[str]] = {}

    for source in sources:
        aggregate = source_results.get(source) or {}

        # Prefer the canonical directory layout used by delta_discovery_v1.
        # Fall back to summary_path for compatibility with relocated/custom
        # output directories recorded in the aggregate manifest.
        canonical_summary = manifest_path.parent / source / "summary.json"
        recorded_summary_raw = _clean(aggregate.get("summary_path"))
        recorded_summary = (
            Path(recorded_summary_raw).expanduser()
            if recorded_summary_raw
            else None
        )
        if recorded_summary is not None and not recorded_summary.is_absolute():
            recorded_summary = manifest_path.parent / recorded_summary

        if canonical_summary.is_file():
            summary_path = canonical_summary
        elif recorded_summary is not None and recorded_summary.is_file():
            summary_path = recorded_summary
        else:
            missing_sources.append(source)
            continue

        try:
            summary = _read_json(summary_path)
        except Exception as exc:
            incomplete_sources[source] = [
                f"summary_read_error:{type(exc).__name__}"
            ]
            continue

        issues = _validate_delta_source_summary(
            summary,
            source=source,
            expected_scope_count=expected_scope_count,
            expected_delta_count=expected_delta_count,
            current_scope_sha256=expected_sha,
            source_profile_sha256=source_profile_sha,
            umls_version=umls_version,
            max_relations_per_cui=max_relations_per_cui,
        )
        if issues:
            incomplete_sources[source] = issues

    if missing_sources or incomplete_sources:
        raise RuntimeError(
            "Delta discovery is incomplete: "
            f"missing_sources={missing_sources}, "
            f"incomplete_sources={sorted(incomplete_sources)}, "
            f"issues={incomplete_sources}"
        )

    return manifest



def _pair_membership_audit(
    *,
    pair_evidence_path: Path,
    current_scope_path: Path,
) -> dict[str, Any]:
    scope = _scope_cui_index(current_scope_path)
    pair_counts: Counter[str] = Counter()
    tier_counts: dict[str, Counter[str]] = {}

    with pair_evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            a = _clean(row.get("local_cui_a")).upper()
            b = _clean(row.get("local_cui_b")).upper()
            label_a = _membership_label((scope.get(a) or {}).get("document_ids"))
            label_b = _membership_label((scope.get(b) or {}).get("document_ids"))
            pair_label = " <-> ".join(sorted((label_a, label_b)))
            pair_counts[pair_label] += 1
            tier = _clean(row.get("overall_tier")).upper() or "UNKNOWN"
            tier_counts.setdefault(pair_label, Counter())[tier] += 1

    return {
        "pair_counts": dict(sorted(pair_counts.items())),
        "tier_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(tier_counts.items())
        },
    }


def run_current_prelabel_build(
    *,
    project_root: Path,
    out_root: Path,
    current_scope_path: Path,
    historical_paths: HistoricalRegressionPaths,
    discovery_paths: DeltaDiscoveryPaths,
    build_paths: CurrentArtifactBuildPaths,
    sources: Sequence[str],
    umls_version: str,
    max_relations_per_cui: int,
) -> dict[str, Any]:
    """Build current DIRECT + pre-label BRIDGE artifacts without API or Neo4j writes."""

    _require_historical_regression_pass(
        out_root / "historical_regression_v1" / "regression_report.json",
        previous_scope_path=historical_paths.historical_scope,
    )
    delta_root = out_root / "delta_discovery_v1"
    delta_manifest = _require_delta_discovery_complete(
        manifest_path=delta_root / "manifest.json",
        current_scope_path=current_scope_path,
        sources=sources,
    )

    build_root = out_root / "current_build_v1"
    direct_census_dir = build_root / "direct_census"
    direct_artifact_dir = build_root / "direct_artifact"
    bridge_evidence_root = build_root / "bridge_evidence"
    bridge_prelabel_dir = build_root / "bridge_prelabel_artifact"
    label_plan_dir = build_root / "external_label_plan"

    for path in (
        direct_census_dir,
        direct_artifact_dir,
        bridge_evidence_root,
        bridge_prelabel_dir,
        label_plan_dir,
    ):
        shutil.rmtree(path, ignore_errors=True)

    evidence_manifest = build_current_bridge_evidence_root(
        current_scope_path=current_scope_path,
        historical_bridge_root=historical_paths.bridge_root,
        delta_discovery_root=delta_root,
        sources=sources,
        output_root=bridge_evidence_root,
    )

    direct_cmd = [
        sys.executable,
        str(build_paths.direct_census_script.resolve()),
        "--local-scope",
        str(current_scope_path.resolve()),
        "--source-profile",
        str(discovery_paths.source_profile.resolve()),
        "--cache-dir",
        str(discovery_paths.relation_cache_dir.resolve()),
        "--output-dir",
        str(direct_census_dir),
        "--max-relations-per-cui",
        str(max_relations_per_cui),
        "--require-complete-cache",
    ]
    for source in sources:
        direct_cmd.extend(["--source", source])
    _run_builder(direct_cmd, cwd=project_root)

    direct_artifact_cmd = [
        sys.executable,
        str(historical_paths.direct_builder.resolve()),
        "--census-dir",
        str(direct_census_dir),
        "--policy",
        str(historical_paths.direct_policy.resolve()),
        "--output-dir",
        str(direct_artifact_dir),
        "--unmapped-relation-mode",
        str(build_paths.direct_unmapped_relation_mode),
    ]
    _run_builder(direct_artifact_cmd, cwd=project_root)

    bridge_cmd = [
        sys.executable,
        str(build_paths.bridge_prelabel_builder.resolve()),
        "--bridge-root",
        str(bridge_evidence_root),
        "--local-scope",
        str(current_scope_path.resolve()),
        "--policy",
        str(build_paths.bridge_prelabel_policy.resolve()),
        "--output-dir",
        str(bridge_prelabel_dir),
    ]
    bridge_cmd.extend(["--sources", *sources])
    _run_builder(bridge_cmd, cwd=project_root)

    label_plan = build_external_label_plan(
        retained_external_cuis_path=bridge_prelabel_dir
        / "retained_external_cuis.txt",
        historical_label_map_path=historical_paths.external_label_map,
        umls_version=umls_version,
        output_dir=label_plan_dir,
    )

    direct_manifest = _read_json(direct_artifact_dir / "manifest.json")
    bridge_manifest = _read_json(bridge_prelabel_dir / "manifest.json")
    report = {
        "schema_version": CURRENT_PRELABEL_SCHEMA_VERSION,
        "scope_name": (_read_json(current_scope_path) or {}).get("scope_name"),
        "document_ids": (_read_json(current_scope_path) or {}).get("document_ids", []),
        "current_scope_path": str(current_scope_path.resolve()),
        "current_scope_sha256": _sha256_file(current_scope_path),
        "current_cui_count": int(
            (_read_json(current_scope_path) or {}).get("unique_cui_count") or 0
        ),
        "delta_discovery_manifest_sha256": _sha256_file(
            delta_root / "manifest.json"
        ),
        "direct": {
            "pair_count": direct_manifest.get("pair_count"),
            "overall_tier_counts": direct_manifest.get("overall_tier_counts"),
            "profile_pair_counts": direct_manifest.get("profile_pair_counts"),
            "unmapped_relation_mode": direct_manifest.get("unmapped_relation_mode"),
            "unmapped_relation_name_count": direct_manifest.get(
                "unmapped_relation_name_count"
            ),
            "unmapped_relations_are_retrieval_eligible": direct_manifest.get(
                "unmapped_relations_are_retrieval_eligible"
            ),
            "membership_audit": _pair_membership_audit(
                pair_evidence_path=direct_artifact_dir / "pair_evidence.jsonl",
                current_scope_path=current_scope_path,
            ),
            "artifact_dir": str(direct_artifact_dir),
        },
        "bridge_prelabel": {
            "all_distinct_local_pair_count": bridge_manifest.get(
                "all_distinct_local_pair_count"
            ),
            "retained_distinct_local_pair_count": bridge_manifest.get(
                "retained_distinct_local_pair_count"
            ),
            "retained_local_cui_count": bridge_manifest.get(
                "retained_local_cui_count"
            ),
            "retained_external_cui_count": bridge_manifest.get(
                "retained_external_cui_count"
            ),
            "overall_tier_counts": bridge_manifest.get("overall_tier_counts"),
            "membership_audit": _pair_membership_audit(
                pair_evidence_path=bridge_prelabel_dir / "pair_evidence.jsonl",
                current_scope_path=current_scope_path,
            ),
            "artifact_dir": str(bridge_prelabel_dir),
        },
        "bridge_evidence": evidence_manifest.get("totals", {}),
        "external_label_plan": {
            k: v
            for k, v in label_plan.items()
            if k
            not in {
                "external_cuis_to_resolve",
                "retry_historical_failures",
            }
        },
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "retrieval_metrics_used": False,
        },
    }
    _write_json(build_root / "prelabel_build_report.json", report)
    logger.info(
        "Current pre-label artifact build PASS | direct=%s | bridge_retained=%s | labels_to_resolve=%s",
        report["direct"]["pair_count"],
        report["bridge_prelabel"]["retained_distinct_local_pair_count"],
        label_plan["to_resolve_count"],
    )
    return report


def _run_builder(command: list[str], *, cwd: Path) -> None:
    logger.info("Running frozen artifact builder: %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _compare_jsonl_artifact(
    actual_path: Path,
    expected_path: Path,
    *,
    key: str,
) -> dict[str, Any]:
    actual = _read_jsonl_index(actual_path, key)
    expected = _read_jsonl_index(expected_path, key)

    actual_keys = set(actual)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    changed = sorted(k for k in actual_keys & expected_keys if actual[k] != expected[k])

    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "changed_count": len(changed),
        "missing_ids": missing[:50],
        "extra_ids": extra[:50],
        "changed_ids": changed[:50],
        "exact_match": not missing and not extra and not changed,
    }


def _manifest_counts(path: Path, kind: str) -> dict[str, Any]:
    manifest = _read_json(path)
    if kind == "direct":
        return {
            "pair_count": manifest.get("pair_count"),
            "overall_tier_counts": manifest.get("overall_tier_counts"),
            "profile_pair_counts": manifest.get("profile_pair_counts"),
        }
    if kind == "bridge":
        return {
            "local_scope_count": manifest.get(
                "local_scope_count", manifest.get("local_universe_count")
            ),
            "all_distinct_local_pair_count": manifest.get("all_distinct_local_pair_count"),
            "retained_distinct_local_pair_count": manifest.get(
                "retained_distinct_local_pair_count"
            ),
            "overall_tier_counts": manifest.get("overall_tier_counts"),
        }
    raise ValueError(kind)


def run_historical_relation_artifact_regression(
    *,
    project_root: Path,
    output_dir: Path,
    paths: HistoricalRegressionPaths,
) -> dict[str, Any]:
    """Rebuild frozen CM artifacts offline and require pair-for-pair equality.

    This function performs no UMLS API calls and no Neo4j writes.  It reuses
    the historical raw census/evidence and the historical policy files.
    """

    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    direct_out = output_dir / "direct_rebuilt"
    bridge_out = output_dir / "bridge_rebuilt"
    shutil.rmtree(direct_out, ignore_errors=True)
    shutil.rmtree(bridge_out, ignore_errors=True)
    direct_out.mkdir(parents=True, exist_ok=True)
    bridge_out.mkdir(parents=True, exist_ok=True)

    direct_cmd = [
        sys.executable,
        str(paths.direct_builder.resolve()),
        "--census-dir",
        str(paths.direct_census_dir.resolve()),
        "--policy",
        str(paths.direct_policy.resolve()),
        "--output-dir",
        str(direct_out),
        "--bridge-artifact-dir",
        str(paths.expected_bridge_artifact_dir.resolve()),
    ]
    _run_builder(direct_cmd, cwd=project_root)

    bridge_cmd = [
        sys.executable,
        str(paths.bridge_builder.resolve()),
        "--bridge-root",
        str(paths.bridge_root.resolve()),
        "--local-universe",  # legacy CLI name retained by frozen builder
        str(paths.historical_scope.resolve()),
        "--policy",
        str(paths.bridge_policy.resolve()),
        "--output-dir",
        str(bridge_out),
    ]
    if paths.external_label_map is not None and paths.external_label_map.exists():
        bridge_cmd.extend(["--external-label-map", str(paths.external_label_map.resolve())])
    _run_builder(bridge_cmd, cwd=project_root)

    direct_comparison = _compare_jsonl_artifact(
        direct_out / "pair_evidence.jsonl",
        paths.expected_direct_artifact_dir / "pair_evidence.jsonl",
        key="direct_id",
    )
    bridge_comparison = _compare_jsonl_artifact(
        bridge_out / "pair_evidence.jsonl",
        paths.expected_bridge_artifact_dir / "pair_evidence.jsonl",
        key="bridge_id",
    )

    report = {
        "schema_version": HISTORICAL_REGRESSION_SCHEMA_VERSION,
        "safety": {
            "umls_api_calls": False,
            "neo4j_reads": False,
            "neo4j_writes": False,
            "retrieval_metrics_used": False,
        },
        "direct": {
            "comparison": direct_comparison,
            "expected_manifest_counts": _manifest_counts(
                paths.expected_direct_artifact_dir / "manifest.json", "direct"
            ),
            "rebuilt_manifest_counts": _manifest_counts(
                direct_out / "manifest.json", "direct"
            ),
            "input_census_manifest_sha256": _sha256_file(
                paths.direct_census_dir / "manifest.json"
            ),
            "policy_sha256": _sha256_file(paths.direct_policy),
        },
        "bridge": {
            "comparison": bridge_comparison,
            "expected_manifest_counts": _manifest_counts(
                paths.expected_bridge_artifact_dir / "manifest.json", "bridge"
            ),
            "rebuilt_manifest_counts": _manifest_counts(
                bridge_out / "manifest.json", "bridge"
            ),
            "historical_scope_sha256": _sha256_file(paths.historical_scope),
            "policy_sha256": _sha256_file(paths.bridge_policy),
            "external_label_map_sha256": (
                _sha256_file(paths.external_label_map)
                if paths.external_label_map is not None and paths.external_label_map.exists()
                else None
            ),
        },
    }
    report["pass"] = bool(
        direct_comparison["exact_match"] and bridge_comparison["exact_match"]
    )

    _write_json(output_dir / "regression_report.json", report)
    if not report["pass"]:
        raise RuntimeError(
            "Historical UMLS relation artifact regression failed; "
            f"see {output_dir / 'regression_report.json'}"
        )

    logger.info(
        "Historical UMLS relation artifact regression PASS | direct=%d | bridge=%d",
        direct_comparison["actual_count"],
        bridge_comparison["actual_count"],
    )
    return report


def run_umls_relation_artifact_workflow(
    driver,
    *,
    project_root: Path,
    work_root: Path,
    action: str,
    scope_name: str = "current_corpus",
    document_ids: Sequence[str] | None = None,
    previous_scope_path: Path | None = None,
    historical_regression_paths: HistoricalRegressionPaths | None = None,
    delta_discovery_paths: DeltaDiscoveryPaths | None = None,
    current_artifact_build_paths: CurrentArtifactBuildPaths | None = None,
    external_label_resolution_paths: ExternalLabelResolutionPaths | None = None,
    current_final_build_paths: CurrentFinalBuildPaths | None = None,
    generalized_bridge_build_paths: GeneralizedBridgeBuildPaths | None = None,
    sources: Sequence[str] = (),
    umls_version: str = "2026AA",
    max_relations_per_cui: int = 1000,
    api_timeout: float = 30.0,
    api_rate_limit_per_second: float = 5.0,
    progress_every: int = 25,
    resume_completed_sources: bool = True,
    external_label_max_retries: int = 2,
    external_label_max_consecutive_transient_failures: int = 10,
    external_label_resume: bool = True,
) -> dict[str, Any]:
    """Pipeline entry point for the frozen relation-artifact workflow.

    ``delta_plan`` is fully read-only and API-free. ``delta_discovery`` performs
    UMLS API/cache work for new CUIs only, remains read-only with respect to
    Neo4j, and intentionally stops before global artifact construction.
    """

    action = _clean(action).lower() or "scope"
    allowed = {
        "scope",
        "historical_regression",
        "scope_and_regression",
        "delta_plan",
        "delta_discovery",
        "build_current_prelabel",
        "resolve_external_labels",
        "build_current_final",
        "build_current_generalized",
    }
    if action not in allowed:
        raise ValueError(
            f"Unsupported UMLS relation artifact action {action!r}. "
            f"Use one of: {sorted(allowed)}"
        )

    out_root = (work_root / "umls_relation_artifacts").resolve()
    result: dict[str, Any] = {
        "action": action,
        "output_root": str(out_root),
        "umls_api_calls": False,
        "neo4j_writes": False,
    }

    if action in {"scope", "scope_and_regression", "delta_plan", "delta_discovery", "build_current_prelabel"}:
        scope_path = out_root / "inputs" / "local_umls_scope_v1.json"
        scope = build_local_umls_scope(
            driver,
            scope_path,
            scope_name=scope_name,
            document_ids=document_ids,
        )
        result["scope_path"] = str(scope_path)
        result["scope_cui_count"] = scope["unique_cui_count"]
        result["document_ids"] = scope["document_ids"]

        if previous_scope_path is not None:
            delta = compare_local_umls_scopes(scope_path, previous_scope_path)
            delta_path = out_root / "inputs" / "scope_delta_vs_previous.json"
            _write_json(delta_path, delta)
            result["scope_delta"] = delta
            result["scope_delta_path"] = str(delta_path)

    if action in {"historical_regression", "scope_and_regression"}:
        if historical_regression_paths is None:
            raise ValueError(
                "historical_regression_paths are required for historical regression"
            )
        regression = run_historical_relation_artifact_regression(
            project_root=project_root,
            output_dir=out_root / "historical_regression_v1",
            paths=historical_regression_paths,
        )
        result["historical_regression_pass"] = regression["pass"]
        result["historical_regression_report"] = str(
            out_root / "historical_regression_v1" / "regression_report.json"
        )

    if action in {"delta_plan", "delta_discovery"}:
        if previous_scope_path is None:
            raise ValueError("previous_scope_path is required for delta actions")
        if historical_regression_paths is None:
            raise ValueError(
                "historical_regression_paths are required for delta actions"
            )
        if delta_discovery_paths is None:
            raise ValueError("delta_discovery_paths are required for delta actions")

        scope_path = out_root / "inputs" / "local_umls_scope_v1.json"
        plan_path = out_root / "delta_plan_v1.json"
        plan = build_delta_discovery_plan(
            current_scope_path=scope_path,
            previous_scope_path=previous_scope_path,
            regression_report_path=(
                out_root / "historical_regression_v1" / "regression_report.json"
            ),
            historical_paths=historical_regression_paths,
            discovery_paths=delta_discovery_paths,
            sources=sources,
            umls_version=umls_version,
            max_relations_per_cui=max_relations_per_cui,
            output_path=plan_path,
        )
        result["delta_plan_path"] = str(plan_path)
        result["delta_plan_ready"] = bool(plan.get("ready_for_discovery"))
        result["estimated_top_level_relation_fetches"] = (
            (plan.get("relation_cache") or {}).get(
                "new_top_level_relation_fetches_estimated"
            )
        )

        if action == "delta_discovery":
            discovery = run_delta_relation_discovery(
                project_root=project_root,
                output_dir=out_root / "delta_discovery_v1",
                plan=plan,
                discovery_paths=delta_discovery_paths,
                api_timeout=api_timeout,
                api_rate_limit_per_second=api_rate_limit_per_second,
                progress_every=progress_every,
                resume_completed_sources=resume_completed_sources,
            )
            result["umls_api_calls"] = True
            result["delta_discovery_manifest"] = str(
                out_root / "delta_discovery_v1" / "manifest.json"
            )
            result["delta_discovery_totals"] = discovery.get("totals", {})

    if action == "build_current_prelabel":
        if historical_regression_paths is None:
            raise ValueError(
                "historical_regression_paths are required for build_current_prelabel"
            )
        if delta_discovery_paths is None:
            raise ValueError(
                "delta_discovery_paths are required for build_current_prelabel"
            )
        if current_artifact_build_paths is None:
            raise ValueError(
                "current_artifact_build_paths are required for build_current_prelabel"
            )
        scope_path = out_root / "inputs" / "local_umls_scope_v1.json"
        prelabel = run_current_prelabel_build(
            project_root=project_root,
            out_root=out_root,
            current_scope_path=scope_path,
            historical_paths=historical_regression_paths,
            discovery_paths=delta_discovery_paths,
            build_paths=current_artifact_build_paths,
            sources=sources,
            umls_version=umls_version,
            max_relations_per_cui=max_relations_per_cui,
        )
        result["current_prelabel_report"] = str(
            out_root / "current_build_v1" / "prelabel_build_report.json"
        )
        result["current_prelabel"] = prelabel

    if action == "resolve_external_labels":
        if external_label_resolution_paths is None:
            raise ValueError(
                "external_label_resolution_paths are required for "
                "resolve_external_labels"
            )
        current_build_root = out_root / "current_build_v1"
        resolution = resolve_external_labels(
            label_plan_path=(
                current_build_root
                / "external_label_plan"
                / "external_label_plan_v1.json"
            ),
            retained_external_cuis_path=(
                current_build_root
                / "bridge_prelabel_artifact"
                / "retained_external_cuis.txt"
            ),
            historical_label_map_path=(
                external_label_resolution_paths.historical_label_map
            ),
            output_dir=current_build_root / "external_labels",
            umls_version=umls_version,
            cache_dir=external_label_resolution_paths.cache_dir,
            api_timeout=api_timeout,
            api_rate_limit_per_second=api_rate_limit_per_second,
            progress_every=progress_every,
            max_retries=external_label_max_retries,
            max_consecutive_transient_failures=(
                external_label_max_consecutive_transient_failures
            ),
            resume=external_label_resume,
        )
        result["umls_api_calls"] = bool(
            (resolution.get("client_stats") or {}).get("api_requests", 0)
        )
        result["external_label_resolution_manifest"] = str(
            current_build_root / "external_labels" / "manifest.json"
        )
        result["external_label_resolution"] = resolution

    if action == "build_current_final":
        if current_final_build_paths is None:
            raise ValueError(
                "current_final_build_paths are required for build_current_final"
            )
        scope_path = out_root / "inputs" / "local_umls_scope_v1.json"
        final = run_current_final_build(
            project_root=project_root,
            out_root=out_root,
            current_scope_path=scope_path,
            build_paths=current_final_build_paths,
            sources=sources,
            umls_version=umls_version,
        )
        result["current_final_report"] = str(
            out_root / "current_build_v1" / "current_final_report.json"
        )
        result["current_final"] = final

    if action == "build_current_generalized":
        if generalized_bridge_build_paths is None:
            raise ValueError(
                "generalized_bridge_build_paths are required for "
                "build_current_generalized"
            )
        scope_path = out_root / "inputs" / "local_umls_scope_v1.json"
        generalized = run_current_generalized_v2_build(
            project_root=project_root,
            out_root=out_root,
            current_scope_path=scope_path,
            build_paths=generalized_bridge_build_paths,
            sources=sources,
            umls_version=umls_version,
        )
        result["current_generalized_v2_report"] = str(
            out_root / "current_build_v1" / "current_generalized_v2_report.json"
        )
        result["current_generalized_v2"] = generalized

    return result


def _resolve_project_path(
    value: Any,
    *,
    project_root: Path,
    required: bool = False,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError("Missing required UMLS relation artifact path")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def run_umls_relation_artifact_workflow_from_config(
    driver,
    *,
    project_root: Path,
    work_root: Path,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the config-owned Phase-A artifact workflow and execute it."""

    cfg = dict(config or {})
    action = _clean(cfg.get("action")) or "scope"
    scope_name = _clean(cfg.get("scope_name")) or "current_corpus"
    document_ids = cfg.get("document_ids") or None

    previous_scope_path = _resolve_project_path(
        cfg.get("previous_scope_path"), project_root=project_root
    )

    regression_paths: HistoricalRegressionPaths | None = None
    if action.lower() in {
        "historical_regression",
        "scope_and_regression",
        "delta_plan",
        "delta_discovery",
        "build_current_prelabel",
    }:
        historical = cfg.get("historical_regression")
        if not isinstance(historical, Mapping):
            raise ValueError(
                "umls_connections.artifact_workflow.historical_regression "
                "must be configured for regression actions"
            )

        def req(name: str) -> Path:
            path = _resolve_project_path(
                historical.get(name), project_root=project_root, required=True
            )
            assert path is not None
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        label_map = _resolve_project_path(
            historical.get("external_label_map"), project_root=project_root
        )
        regression_paths = HistoricalRegressionPaths(
            direct_census_dir=req("direct_census_dir"),
            direct_policy=req("direct_policy"),
            expected_direct_artifact_dir=req("expected_direct_artifact_dir"),
            bridge_root=req("bridge_root"),
            historical_scope=req("historical_scope"),
            bridge_policy=req("bridge_policy"),
            expected_bridge_artifact_dir=req("expected_bridge_artifact_dir"),
            external_label_map=label_map,
            direct_builder=req("direct_builder"),
            bridge_builder=req("bridge_builder"),
        )

    delta_paths: DeltaDiscoveryPaths | None = None
    discovery_cfg = cfg.get("discovery") or {}
    if action.lower() in {"delta_plan", "delta_discovery", "build_current_prelabel"}:
        if not isinstance(discovery_cfg, Mapping):
            raise ValueError(
                "umls_connections.artifact_workflow.discovery must be configured "
                "for delta actions"
            )

        def discovery_req(name: str) -> Path:
            path = _resolve_project_path(
                discovery_cfg.get(name), project_root=project_root, required=True
            )
            assert path is not None
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        delta_paths = DeltaDiscoveryPaths(
            source_profile=discovery_req("source_profile"),
            relation_cache_dir=discovery_req("relation_cache_dir"),
            bridge_census_script=discovery_req("bridge_census_script"),
        )

    build_paths: CurrentArtifactBuildPaths | None = None
    final_build_paths: CurrentFinalBuildPaths | None = None
    generalized_build_paths: GeneralizedBridgeBuildPaths | None = None
    if action.lower() in {
        "build_current_prelabel",
        "build_current_final",
        "build_current_generalized",
    }:
        build_cfg = cfg.get("artifact_build")
        if not isinstance(build_cfg, Mapping):
            raise ValueError(
                "umls_connections.artifact_workflow.artifact_build must be "
                "configured for build_current_prelabel/build_current_final"
            )

        def build_req(name: str) -> Path:
            path = _resolve_project_path(
                build_cfg.get(name), project_root=project_root, required=True
            )
            assert path is not None
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        if action.lower() == "build_current_prelabel":
            direct_unmapped_relation_mode = (
                _clean(build_cfg.get("direct_unmapped_relation_mode")) or "error"
            ).lower()
            if direct_unmapped_relation_mode not in {"error", "reject"}:
                raise ValueError(
                    "umls_connections.artifact_workflow.artifact_build."
                    "direct_unmapped_relation_mode must be 'error' or 'reject'"
                )
            build_paths = CurrentArtifactBuildPaths(
                direct_census_script=build_req("direct_census_script"),
                bridge_prelabel_builder=build_req("bridge_prelabel_builder"),
                bridge_prelabel_policy=build_req("bridge_prelabel_policy"),
                direct_unmapped_relation_mode=direct_unmapped_relation_mode,
            )
        elif action.lower() == "build_current_final":
            final_build_paths = CurrentFinalBuildPaths(
                bridge_final_builder=build_req("bridge_final_builder"),
                bridge_final_policy=build_req("bridge_final_policy"),
            )
        else:
            generalized_build_paths = GeneralizedBridgeBuildPaths(
                bridge_builder=build_req("bridge_final_builder"),
                bridge_policy=build_req("bridge_generalized_policy"),
                frozen_v1_1_policy_sha256=_clean(
                    build_cfg.get("frozen_v1_1_policy_sha256")
                ),
                frozen_c3_manifest_sha256=_clean(
                    build_cfg.get("frozen_c3_manifest_sha256")
                ),
            )

    external_label_paths: ExternalLabelResolutionPaths | None = None
    external_label_cfg = cfg.get("external_labels") or {}
    if action.lower() == "resolve_external_labels":
        if not isinstance(external_label_cfg, Mapping):
            raise ValueError(
                "umls_connections.artifact_workflow.external_labels must be "
                "configured for resolve_external_labels"
            )

        configured_historical_label_map = external_label_cfg.get(
            "historical_label_map"
        )
        if configured_historical_label_map in (None, ""):
            historical_cfg = cfg.get("historical_regression") or {}
            if isinstance(historical_cfg, Mapping):
                configured_historical_label_map = historical_cfg.get(
                    "external_label_map"
                )
        historical_label_map = _resolve_project_path(
            configured_historical_label_map,
            project_root=project_root,
            required=True,
        )
        assert historical_label_map is not None
        if not historical_label_map.exists():
            raise FileNotFoundError(historical_label_map)

        cache_dir = _resolve_project_path(
            external_label_cfg.get("cache_dir"),
            project_root=project_root,
            required=True,
        )
        assert cache_dir is not None
        external_label_paths = ExternalLabelResolutionPaths(
            historical_label_map=historical_label_map,
            cache_dir=cache_dir,
        )

    api_cfg = (
        external_label_cfg
        if action.lower() == "resolve_external_labels"
        else discovery_cfg
    )

    result = run_umls_relation_artifact_workflow(
        driver,
        project_root=project_root,
        work_root=work_root,
        action=action,
        scope_name=scope_name,
        document_ids=document_ids,
        previous_scope_path=previous_scope_path,
        historical_regression_paths=regression_paths,
        delta_discovery_paths=delta_paths,
        current_artifact_build_paths=build_paths,
        external_label_resolution_paths=external_label_paths,
        current_final_build_paths=final_build_paths,
        generalized_bridge_build_paths=generalized_build_paths,
        sources=cfg.get("sources") or (),
        umls_version=_clean(cfg.get("umls_version")) or "2026AA",
        max_relations_per_cui=int(cfg.get("max_relations_per_cui") or 1000),
        api_timeout=float(api_cfg.get("api_timeout") or 30.0),
        api_rate_limit_per_second=float(
            api_cfg.get("api_rate_limit_per_second") or 5.0
        ),
        progress_every=int(api_cfg.get("progress_every") or 25),
        resume_completed_sources=bool(
            discovery_cfg.get("resume_completed_sources", True)
        ),
        external_label_max_retries=int(
            external_label_cfg.get("max_retries") or 2
        ),
        external_label_max_consecutive_transient_failures=int(
            external_label_cfg.get("max_consecutive_transient_failures") or 10
        ),
        external_label_resume=bool(external_label_cfg.get("resume", True)),
    )

    expected_cui_count = cfg.get("expected_cui_count")
    if (
        expected_cui_count is not None
        and "scope_cui_count" in result
        and result.get("scope_cui_count") != int(expected_cui_count)
    ):
        raise RuntimeError(
            "Local UMLS scope count mismatch: "
            f"expected={int(expected_cui_count)}, actual={result.get('scope_cui_count')}"
        )

    expected_new = cfg.get("expected_new_cui_count")
    expected_retired = cfg.get("expected_retired_cui_count")
    delta = result.get("scope_delta") or {}
    if (
        expected_new is not None
        and "scope_delta" in result
        and delta.get("new_cui_count") != int(expected_new)
    ):
        raise RuntimeError(
            "Local UMLS scope new-CUI count mismatch: "
            f"expected={int(expected_new)}, actual={delta.get('new_cui_count')}"
        )
    if (
        expected_retired is not None
        and "scope_delta" in result
        and delta.get("retired_cui_count") != int(expected_retired)
    ):
        raise RuntimeError(
            "Local UMLS scope retired-CUI count mismatch: "
            f"expected={int(expected_retired)}, actual={delta.get('retired_cui_count')}"
        )

    return result
