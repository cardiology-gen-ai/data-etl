"""Materialize an immutable UMLS relation freeze into Neo4j.

The frozen relation artifacts are CUI-level undirected pair projections.  The
Neo4j representation therefore connects existing ``UMLSConcept`` nodes rather
than local ``Concept`` nodes.  No UMLSConcept node is created here.

This module is intentionally independent from the exploratory/generic
``umls_connections`` materializer.  The latter works with relation-specific
Concept->Concept edges; this module writes only the approved pair-level
``UMLS_DIRECT`` and ``UMLS_BRIDGE`` layer selected by a freeze manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FROZEN_MATERIALIZATION_SCHEMA_VERSION = "umls_relation_materialization_freeze_v1"
MATERIALIZED_RELATION_SCHEMA_VERSION = "umls_relation_materialized_v1"
FROZEN_RELATION_TYPES = ("UMLS_DIRECT", "UMLS_BRIDGE")
ALLOWED_MATERIALIZED_TIERS = {"STRONG", "MEDIUM"}
PROVENANCE = "umls_relation_artifact"
PROVENANCE_SOURCE = "UMLS"
PROVENANCE_METHOD = "frozen_policy_projection"
PAIR_DIRECTION_POLICY = "lexicographic_cui_pair"


@dataclass(frozen=True)
class FrozenMaterializationContract:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    scope_path: Path
    scope_cuis: tuple[str, ...]
    document_ids: tuple[str, ...]
    direct_payload_path: Path
    bridge_payload_path: Path
    direct_rows: tuple[dict[str, Any], ...]
    bridge_rows: tuple[dict[str, Any], ...]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_project_path(project_root: Path, value: Any) -> Path:
    if value in (None, ""):
        raise ValueError("Missing frozen materialization path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _normalize_cui(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("Empty CUI in frozen relation payload")
    return text


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str]:
    a = _normalize_cui(row.get("local_cui_a"))
    b = _normalize_cui(row.get("local_cui_b"))
    if a == b:
        raise ValueError(f"Self-loop in frozen relation payload: {a}")
    if a > b:
        raise ValueError(
            "Frozen relation pair is not in canonical lexicographic order: "
            f"{a} > {b}"
        )
    return a, b


def _validate_unique_payload_rows(
    rows: Sequence[Mapping[str, Any]], *, kind: str
) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = _pair_key(row)
        if key in seen:
            raise ValueError(f"Duplicate {kind} pair in frozen payload: {key}")
        seen.add(key)
        tier = str(row.get("overall_tier") or "").strip().upper()
        if tier not in ALLOWED_MATERIALIZED_TIERS:
            raise ValueError(
                f"{kind} payload contains non-materializable tier {tier!r}: {key}"
            )


def _verify_file_hash(project_root: Path, spec: Mapping[str, Any], key: str) -> Path:
    raw = spec.get(key)
    path = _resolve_project_path(project_root, raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    hash_key = "sha256" if key == "path" else key.replace("_path", "_sha256")
    expected = str(spec.get(hash_key) or "").strip().lower()
    if expected and _sha256_file(path) != expected:
        raise ValueError(f"SHA256 mismatch for {key}: {path}")
    return path


def _validate_manifest_safety(manifest: Mapping[str, Any]) -> None:
    safety = manifest.get("safety") or {}
    if not isinstance(safety, Mapping):
        raise ValueError("Freeze manifest safety block must be an object")
    forbidden_true = (
        "umls_api_calls",
        "neo4j_reads",
        "neo4j_writes",
        "retrieval_metrics_used",
        "benchmark_tuned",
        "weak_edges_materialized",
        "reject_edges_materialized",
    )
    bad = {key: safety.get(key) for key in forbidden_true if safety.get(key)}
    if bad:
        raise ValueError(f"Unsafe freeze manifest flags: {bad}")


def _scope_cuis(scope: Mapping[str, Any]) -> tuple[str, ...]:
    """Return trusted CUIs from current or legacy scope schemas.

    Current ``local_umls_scope_v1`` artifacts store ``cuis`` as records such
    as ``{"cui": "C0001206", ...}``. Older/testing artifacts may store a
    simple list of CUI strings. Materialization accepts both forms while
    remaining strict about malformed or missing identifiers.
    """

    candidates = (
        scope.get("cuis")
        or scope.get("unique_cuis")
        or scope.get("local_cuis")
        or scope.get("trusted_cuis")
        or []
    )
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("Scope CUI list is missing or malformed")

    normalized: set[str] = set()
    for item in candidates:
        if isinstance(item, Mapping):
            raw_cui = item.get("cui")
            if not raw_cui:
                raise ValueError("Scope CUI record is missing required 'cui'")
        else:
            raw_cui = item
        normalized.add(_normalize_cui(raw_cui))

    cuis = tuple(sorted(normalized))
    expected = int(scope.get("unique_cui_count") or len(cuis))
    if len(cuis) != expected:
        raise ValueError(
            f"Scope CUI list/count mismatch: list={len(cuis)} expected={expected}"
        )
    return cuis


def _payload_subset_gate(
    payload_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> None:
    evidence = {_pair_key(row): row for row in evidence_rows}
    for row in payload_rows:
        key = _pair_key(row)
        source = evidence.get(key)
        if source is None:
            raise ValueError(f"{kind} payload pair absent from frozen evidence: {key}")
        if str(source.get("overall_tier") or "").upper() != str(
            row.get("overall_tier") or ""
        ).upper():
            raise ValueError(f"{kind} payload/evidence tier mismatch for {key}")
        if kind == "BRIDGE" and source.get("retained_for_retrieval") is not True:
            raise ValueError(f"BRIDGE payload contains non-retained pair: {key}")


def load_and_validate_frozen_materialization_contract(
    *,
    project_root: Path,
    config: Mapping[str, Any],
) -> FrozenMaterializationContract:
    """Load and fully validate the immutable materialization contract offline."""

    manifest_path = _resolve_project_path(project_root, config.get("manifest_path"))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    manifest_sha = _sha256_file(manifest_path)

    if manifest.get("schema_version") != FROZEN_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported freeze schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "FROZEN":
        raise ValueError(f"Freeze status must be FROZEN, got {manifest.get('status')!r}")

    # The config is a reference/guard, while the on-disk manifest is authoritative.
    for key in ("schema_version", "status"):
        if config.get(key) not in (None, manifest.get(key)):
            raise ValueError(f"Config freeze {key} disagrees with frozen manifest")
    for key in ("scope", "direct", "bridge"):
        if config.get(key) is not None and config.get(key) != manifest.get(key):
            raise ValueError(f"Config freeze block {key!r} disagrees with manifest")

    _validate_manifest_safety(manifest)

    scope_spec = manifest.get("scope") or {}
    direct_spec = manifest.get("direct") or {}
    bridge_spec = manifest.get("bridge") or {}
    if not all(isinstance(x, Mapping) for x in (scope_spec, direct_spec, bridge_spec)):
        raise ValueError("Freeze manifest scope/direct/bridge blocks are required")

    scope_path = _verify_file_hash(project_root, scope_spec, "path")
    scope = _load_json(scope_path)
    cuis = _scope_cuis(scope)
    expected_scope = int(scope_spec.get("expected_cui_count") or -1)
    if len(cuis) != expected_scope:
        raise ValueError(
            f"Frozen scope count mismatch: expected={expected_scope} actual={len(cuis)}"
        )
    document_ids = tuple(sorted(str(x) for x in (scope_spec.get("document_ids") or [])))

    # Verify every frozen dependency fingerprint before reading payloads.
    for spec in (direct_spec, bridge_spec):
        for key in (
            "manifest_path",
            "pair_evidence_path",
            "policy_path",
            "policy_snapshot_path",
            "payload_path",
        ):
            _verify_file_hash(project_root, spec, key)

    direct_manifest = _load_json(
        _resolve_project_path(project_root, direct_spec["manifest_path"])
    )
    bridge_manifest = _load_json(
        _resolve_project_path(project_root, bridge_spec["manifest_path"])
    )

    if direct_manifest.get("policy_sha256") != direct_spec.get("policy_sha256"):
        raise ValueError("DIRECT artifact policy SHA does not match freeze")
    if bridge_manifest.get("policy_sha256") != bridge_spec.get("policy_sha256"):
        raise ValueError("BRIDGE artifact policy SHA does not match freeze")

    if int(direct_manifest.get("pair_count") or -1) != int(
        direct_spec.get("expected_all_pair_count") or -1
    ):
        raise ValueError("DIRECT frozen candidate count mismatch")
    if int((direct_manifest.get("profile_pair_counts") or {}).get("balanced") or -1) != int(
        direct_spec.get("expected_materialized_pair_count") or -1
    ):
        raise ValueError("DIRECT frozen balanced count mismatch")
    if dict(direct_manifest.get("overall_tier_counts") or {}) != dict(
        direct_spec.get("expected_tier_counts") or {}
    ):
        raise ValueError("DIRECT frozen tier distribution mismatch")

    if int(bridge_manifest.get("all_distinct_local_pair_count") or -1) != int(
        bridge_spec.get("expected_all_pair_count") or -1
    ):
        raise ValueError("BRIDGE frozen candidate count mismatch")
    if int(bridge_manifest.get("retained_distinct_local_pair_count") or -1) != int(
        bridge_spec.get("expected_materialized_pair_count") or -1
    ):
        raise ValueError("BRIDGE frozen retained count mismatch")
    if dict(bridge_manifest.get("overall_tier_counts") or {}) != dict(
        bridge_spec.get("expected_tier_counts") or {}
    ):
        raise ValueError("BRIDGE frozen tier distribution mismatch")

    direct_payload_path = _resolve_project_path(project_root, direct_spec["payload_path"])
    bridge_payload_path = _resolve_project_path(project_root, bridge_spec["payload_path"])
    direct_rows = _read_jsonl(direct_payload_path)
    bridge_rows = _read_jsonl(bridge_payload_path)
    _validate_unique_payload_rows(direct_rows, kind="DIRECT")
    _validate_unique_payload_rows(bridge_rows, kind="BRIDGE")

    if len(direct_rows) != int(direct_spec.get("expected_materialized_pair_count") or -1):
        raise ValueError("DIRECT payload count mismatch")
    if len(bridge_rows) != int(bridge_spec.get("expected_materialized_pair_count") or -1):
        raise ValueError("BRIDGE payload count mismatch")

    direct_evidence = _read_jsonl(
        _resolve_project_path(project_root, direct_spec["pair_evidence_path"])
    )
    bridge_evidence = _read_jsonl(
        _resolve_project_path(project_root, bridge_spec["pair_evidence_path"])
    )
    _payload_subset_gate(direct_rows, direct_evidence, kind="DIRECT")
    _payload_subset_gate(bridge_rows, bridge_evidence, kind="BRIDGE")

    payload_cuis = {
        cui
        for row in (*direct_rows, *bridge_rows)
        for cui in _pair_key(row)
    }
    outside = sorted(payload_cuis.difference(cuis))
    if outside:
        raise ValueError(
            "Frozen payload contains endpoints outside the trusted local CUI scope: "
            f"{outside[:20]}"
        )

    return FrozenMaterializationContract(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        scope_path=scope_path,
        scope_cuis=cuis,
        document_ids=document_ids,
        direct_payload_path=direct_payload_path,
        bridge_payload_path=bridge_payload_path,
        direct_rows=tuple(direct_rows),
        bridge_rows=tuple(bridge_rows),
    )


def _direct_materialization_row(
    row: Mapping[str, Any], *, contract: FrozenMaterializationContract
) -> dict[str, Any]:
    a, b = _pair_key(row)
    relation_names: set[str] = set()
    relation_families: set[str] = set()
    for source in row.get("source_evidence") or []:
        relation_names.update(str(x) for x in (source.get("relation_names") or []) if x)
        for family in source.get("relation_families") or []:
            name = family.get("relation_family")
            if name:
                relation_families.add(str(name))

    spec = contract.manifest["direct"]
    return {
        "cui_a": a,
        "cui_b": b,
        "edge_key": f"UMLS_DIRECT::{a}::{b}",
        "properties": {
            "edge_key": f"UMLS_DIRECT::{a}::{b}",
            "local_cui_a": a,
            "local_cui_b": b,
            "tier": str(row.get("overall_tier")),
            "profile": "balanced",
            "direct_id": str(row.get("direct_id") or ""),
            "sources": sorted(str(x) for x in (row.get("sources") or []) if x),
            "relation_names": sorted(relation_names),
            "relation_families": sorted(relation_families),
            "projection_ambiguity": bool(row.get("projection_ambiguity")),
            "policy_sha256": str(spec["policy_sha256"]),
            "artifact_manifest_sha256": str(spec["manifest_sha256"]),
            "pair_evidence_sha256": str(spec["pair_evidence_sha256"]),
            "freeze_manifest_sha256": contract.manifest_sha256,
            "evidence_artifact_path": str(spec["pair_evidence_path"]),
            "relationship_family": "umls_direct",
            "provenance": PROVENANCE,
            "provenance_source": PROVENANCE_SOURCE,
            "provenance_method": PROVENANCE_METHOD,
            "materialization_schema_version": MATERIALIZED_RELATION_SCHEMA_VERSION,
            "pair_direction_policy": PAIR_DIRECTION_POLICY,
        },
    }


def _bridge_policy_rule_ids(row: Mapping[str, Any]) -> list[str]:
    rules: set[str] = set()
    for evidence in row.get("external_path_evidence") or []:
        if evidence.get("retained_for_retrieval") is not True:
            continue
        for pair in evidence.get("relation_pair_evidence") or []:
            tier = str(pair.get("tier") or "").upper()
            rule = pair.get("policy_rule_id")
            if rule and tier in ALLOWED_MATERIALIZED_TIERS:
                rules.add(str(rule))
    return sorted(rules)


def _bridge_materialization_row(
    row: Mapping[str, Any], *, contract: FrozenMaterializationContract
) -> dict[str, Any]:
    a, b = _pair_key(row)
    balanced = (row.get("profile_scores_v1_1") or {}).get("balanced") or {}
    score = float(balanced.get("score") or row.get("bridge_score_v1_1") or 0.0)
    sources = balanced.get("sources") or row.get("retrieval_sources") or []
    spec = contract.manifest["bridge"]
    return {
        "cui_a": a,
        "cui_b": b,
        "edge_key": f"UMLS_BRIDGE::{a}::{b}",
        "properties": {
            "edge_key": f"UMLS_BRIDGE::{a}::{b}",
            "local_cui_a": a,
            "local_cui_b": b,
            "tier": str(row.get("overall_tier")),
            "profile": "balanced",
            "bridge_id": str(row.get("bridge_id") or ""),
            "score": score,
            "bridge_score_v1_1": float(row.get("bridge_score_v1_1") or score),
            "score_top_k": int(row.get("score_top_k") or balanced.get("top_k") or 0),
            "retained_external_hub_count": int(row.get("retained_external_hub_count") or 0),
            "all_external_hub_count": int(row.get("all_external_hub_count") or 0),
            "retrieval_sources": sorted(str(x) for x in sources if x),
            "policy_rule_ids": _bridge_policy_rule_ids(row),
            "policy_sha256": str(spec["policy_sha256"]),
            "artifact_manifest_sha256": str(spec["manifest_sha256"]),
            "pair_evidence_sha256": str(spec["pair_evidence_sha256"]),
            "freeze_manifest_sha256": contract.manifest_sha256,
            "evidence_artifact_path": str(spec["pair_evidence_path"]),
            "relationship_family": "umls_bridge",
            "provenance": PROVENANCE,
            "provenance_source": PROVENANCE_SOURCE,
            "provenance_method": PROVENANCE_METHOD,
            "materialization_schema_version": MATERIALIZED_RELATION_SCHEMA_VERSION,
            "pair_direction_policy": PAIR_DIRECTION_POLICY,
        },
    }


def build_frozen_materialization_rows(
    contract: FrozenMaterializationContract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [_direct_materialization_row(row, contract=contract) for row in contract.direct_rows],
        [_bridge_materialization_row(row, contract=contract) for row in contract.bridge_rows],
    )


def _graph_preflight(tx, scope_cuis: Sequence[str]) -> dict[str, Any]:
    documents = [
        str(row["doc_id"])
        for row in tx.run(
            "MATCH (d:Document) RETURN d.doc_id AS doc_id ORDER BY doc_id"
        )
    ]
    endpoint_rows = [
        dict(row)
        for row in tx.run(
            """
            UNWIND $cuis AS cui
            OPTIONAL MATCH (u:UMLSConcept)
            WHERE toUpper(coalesce(toString(u.cui), '')) = cui
            WITH cui, count(u) AS umls_node_count,
                 collect(elementId(u)) AS umls_node_ids
            OPTIONAL MATCH (:Concept)-[:NORMALIZED_TO]->(u2:UMLSConcept)
            WHERE toUpper(coalesce(toString(u2.cui), '')) = cui
            RETURN cui,
                   umls_node_count,
                   size(umls_node_ids) AS umls_node_id_count,
                   count(DISTINCT u2) AS normalized_umls_node_count
            ORDER BY cui
            """,
            cuis=list(scope_cuis),
        )
    ]
    existing = {
        str(row["relationship_type"]): int(row["n"])
        for row in tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types
            RETURN type(r) AS relationship_type, count(r) AS n
            ORDER BY relationship_type
            """,
            types=list(FROZEN_RELATION_TYPES),
        )
    }
    return {
        "document_ids": documents,
        "scope_endpoint_rows": endpoint_rows,
        "existing_relation_counts": existing,
    }


def _validate_graph_preflight(
    preflight: Mapping[str, Any], *, contract: FrozenMaterializationContract
) -> None:
    actual_docs = tuple(sorted(str(x) for x in (preflight.get("document_ids") or [])))
    if contract.document_ids and actual_docs != contract.document_ids:
        raise RuntimeError(
            "Connected Neo4j database document set does not match frozen scope: "
            f"expected={list(contract.document_ids)} actual={list(actual_docs)}"
        )

    bad = []
    for row in preflight.get("scope_endpoint_rows") or []:
        if int(row.get("umls_node_count") or 0) != 1:
            bad.append((row.get("cui"), "umls_node_count", row.get("umls_node_count")))
        elif int(row.get("normalized_umls_node_count") or 0) != 1:
            bad.append(
                (
                    row.get("cui"),
                    "normalized_umls_node_count",
                    row.get("normalized_umls_node_count"),
                )
            )
    if bad:
        raise RuntimeError(
            "Frozen scope is not represented by exactly one locally normalized "
            f"UMLSConcept per CUI; examples={bad[:20]}"
        )


def _replace_frozen_layer_tx(
    tx,
    direct_rows: Sequence[Mapping[str, Any]],
    bridge_rows: Sequence[Mapping[str, Any]],
    *,
    replace_existing_connections: bool,
    now: str,
    expected_direct: int,
    expected_bridge: int,
    direct_policy_sha256: str,
    bridge_policy_sha256: str,
) -> dict[str, Any]:
    existing = {
        str(row["relationship_type"]): int(row["n"])
        for row in tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types
            RETURN type(r) AS relationship_type, count(r) AS n
            """,
            types=list(FROZEN_RELATION_TYPES),
        )
    }
    existing_total = sum(existing.values())
    if existing_total and not replace_existing_connections:
        raise RuntimeError(
            "UMLS_DIRECT/UMLS_BRIDGE already exist; refusing a non-replacing "
            "frozen materialization"
        )

    deleted = 0
    if replace_existing_connections:
        record = tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types
            WITH collect(r) AS rels, count(r) AS n
            FOREACH (rel IN rels | DELETE rel)
            RETURN n
            """,
            types=list(FROZEN_RELATION_TYPES),
        ).single()
        deleted = int(record["n"] if record else 0)

    direct_written = int(
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:UMLSConcept {cui: row.cui_a})
            MATCH (b:UMLSConcept {cui: row.cui_b})
            MERGE (a)-[r:UMLS_DIRECT {edge_key: row.edge_key}]->(b)
            ON CREATE SET r.created_at = datetime($now)
            SET r += row.properties,
                r.updated_at = datetime($now)
            RETURN count(r) AS n
            """,
            rows=list(direct_rows),
            now=now,
        ).single()["n"]
    )
    bridge_written = int(
        tx.run(
            """
            UNWIND $rows AS row
            MATCH (a:UMLSConcept {cui: row.cui_a})
            MATCH (b:UMLSConcept {cui: row.cui_b})
            MERGE (a)-[r:UMLS_BRIDGE {edge_key: row.edge_key}]->(b)
            ON CREATE SET r.created_at = datetime($now)
            SET r += row.properties,
                r.updated_at = datetime($now)
            RETURN count(r) AS n
            """,
            rows=list(bridge_rows),
            now=now,
        ).single()["n"]
    )
    if direct_written != expected_direct or bridge_written != expected_bridge:
        raise RuntimeError(
            "Frozen relationship write count mismatch before commit: "
            f"DIRECT={direct_written}/{expected_direct}, "
            f"BRIDGE={bridge_written}/{expected_bridge}"
        )

    sanity = _post_write_sanity_tx(
        tx,
        expected_direct=expected_direct,
        expected_bridge=expected_bridge,
        direct_policy_sha256=direct_policy_sha256,
        bridge_policy_sha256=bridge_policy_sha256,
    )
    if not sanity["pass"]:
        raise RuntimeError(f"Frozen materialization postconditions failed: {sanity}")

    return {
        "relationships_deleted_before_write": deleted,
        "direct_written": direct_written,
        "bridge_written": bridge_written,
        "sanity": sanity,
    }


def _post_write_sanity_tx(
    tx,
    *,
    expected_direct: int,
    expected_bridge: int,
    direct_policy_sha256: str,
    bridge_policy_sha256: str,
) -> dict[str, Any]:
    counts = {
        str(row["relationship_type"]): int(row["n"])
        for row in tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types
            RETURN type(r) AS relationship_type, count(r) AS n
            ORDER BY relationship_type
            """,
            types=list(FROZEN_RELATION_TYPES),
        )
    }
    duplicate_count = int(
        tx.run(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types
            WITH type(r) AS t, r.edge_key AS edge_key, count(*) AS n
            WHERE edge_key IS NULL OR trim(toString(edge_key)) = '' OR n <> 1
            RETURN count(*) AS n
            """,
            types=list(FROZEN_RELATION_TYPES),
        ).single()["n"]
    )
    invalid_count = int(
        tx.run(
            """
            MATCH (a:UMLSConcept)-[r]->(b:UMLSConcept)
            WHERE type(r) IN $types
              AND (
                NOT r.tier IN ['STRONG','MEDIUM']
                OR r.profile <> 'balanced'
                OR r.provenance <> $provenance
                OR r.materialization_schema_version <> $schema_version
                OR toUpper(toString(a.cui)) <> r.local_cui_a
                OR toUpper(toString(b.cui)) <> r.local_cui_b
                OR r.local_cui_a >= r.local_cui_b
                OR (type(r) = 'UMLS_DIRECT' AND r.policy_sha256 <> $direct_sha)
                OR (type(r) = 'UMLS_BRIDGE' AND r.policy_sha256 <> $bridge_sha)
              )
            RETURN count(r) AS n
            """,
            types=list(FROZEN_RELATION_TYPES),
            provenance=PROVENANCE,
            schema_version=MATERIALIZED_RELATION_SCHEMA_VERSION,
            direct_sha=direct_policy_sha256,
            bridge_sha=bridge_policy_sha256,
        ).single()["n"]
    )
    self_loops = int(
        tx.run(
            """
            MATCH (a:UMLSConcept)-[r]->(a)
            WHERE type(r) IN $types
            RETURN count(r) AS n
            """,
            types=list(FROZEN_RELATION_TYPES),
        ).single()["n"]
    )
    pass_flag = bool(
        counts.get("UMLS_DIRECT", 0) == expected_direct
        and counts.get("UMLS_BRIDGE", 0) == expected_bridge
        and duplicate_count == 0
        and invalid_count == 0
        and self_loops == 0
    )
    return {
        "pass": pass_flag,
        "counts_by_relationship_type": counts,
        "duplicate_edge_key_findings": duplicate_count,
        "invalid_property_or_endpoint_findings": invalid_count,
        "self_loop_findings": self_loops,
    }


def run_frozen_relation_materialization(
    driver,
    *,
    project_root: Path,
    config: Mapping[str, Any],
    write_neo4j: bool,
    replace_existing_connections: bool,
) -> dict[str, Any]:
    """Preflight or atomically materialize a frozen DIRECT+BRIDGE relation layer."""

    if driver is None:
        raise ValueError("materialize_frozen requires an active Neo4j driver")

    contract = load_and_validate_frozen_materialization_contract(
        project_root=project_root,
        config=config,
    )
    direct_rows, bridge_rows = build_frozen_materialization_rows(contract)

    with driver.session() as session:
        preflight = session.execute_read(_graph_preflight, contract.scope_cuis)
    _validate_graph_preflight(preflight, contract=contract)

    direct_spec = contract.manifest["direct"]
    bridge_spec = contract.manifest["bridge"]
    expected_direct = int(direct_spec["expected_materialized_pair_count"])
    expected_bridge = int(bridge_spec["expected_materialized_pair_count"])

    report: dict[str, Any] = {
        "schema_version": MATERIALIZED_RELATION_SCHEMA_VERSION,
        "action": "materialize_frozen",
        "freeze_manifest": str(contract.manifest_path),
        "freeze_manifest_sha256": contract.manifest_sha256,
        "endpoint_label": "UMLSConcept",
        "relationship_types": list(FROZEN_RELATION_TYPES),
        "pair_direction_policy": PAIR_DIRECTION_POLICY,
        "scope_cui_count": len(contract.scope_cuis),
        "document_ids": list(contract.document_ids),
        "direct_expected": expected_direct,
        "bridge_expected": expected_bridge,
        "write_neo4j": bool(write_neo4j),
        "replace_existing_connections": bool(replace_existing_connections),
        "preflight": preflight,
        "umls_api_calls": False,
        "retrieval_metrics_used": False,
    }

    output_dir = contract.manifest_path.parent
    if not write_neo4j:
        report.update(
            {
                "neo4j_reads": True,
                "neo4j_writes": False,
                "status": "PREFLIGHT_PASS",
            }
        )
        _write_json(output_dir / "neo4j_preflight_report.json", report)
        return report

    now = _utc_now_iso()
    with driver.session() as session:
        write_report = session.execute_write(
            _replace_frozen_layer_tx,
            direct_rows,
            bridge_rows,
            replace_existing_connections=bool(replace_existing_connections),
            now=now,
            expected_direct=expected_direct,
            expected_bridge=expected_bridge,
            direct_policy_sha256=str(direct_spec["policy_sha256"]),
            bridge_policy_sha256=str(bridge_spec["policy_sha256"]),
        )
        committed_sanity = session.execute_read(
            _post_write_sanity_tx,
            expected_direct=expected_direct,
            expected_bridge=expected_bridge,
            direct_policy_sha256=str(direct_spec["policy_sha256"]),
            bridge_policy_sha256=str(bridge_spec["policy_sha256"]),
        )
    if not committed_sanity["pass"]:
        raise RuntimeError(
            "Post-commit frozen relation sanity failed unexpectedly: "
            f"{committed_sanity}"
        )

    report.update(
        {
            "neo4j_reads": True,
            "neo4j_writes": True,
            "status": "MATERIALIZED_PASS",
            "written_at": now,
            **write_report,
            "post_commit_sanity": committed_sanity,
        }
    )
    _write_json(output_dir / "neo4j_materialization_report.json", report)
    return report
