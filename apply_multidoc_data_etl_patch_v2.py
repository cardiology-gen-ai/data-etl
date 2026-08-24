#!/usr/bin/env python3
"""Apply the multi-document prebuilt indexing patch to cardiology-gen-ai/data-etl (v2, idempotent).

Run from the data-etl repository root on branch `test`.
The script is deliberately fail-fast: if the expected current code no longer matches,
it stops instead of making a partial/unsafe edit.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
MAIN = ROOT / "src/main.py"
CHUNKING = ROOT / "src/managers/chunking_manager.py"
INDEX = ROOT / "src/managers/index_manager.py"
VALIDATION_DIR = ROOT / "experiments/validation"

for required in (MAIN, CHUNKING, INDEX):
    if not required.is_file():
        raise SystemExit(
            f"Missing {required}. Run this script from the data-etl repository root."
        )


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    """Replace one exact block, but allow an already-applied patch."""
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 1:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"patched: {path} [{label}]")
        return
    if count == 0 and new in text:
        print(f"already patched: {path} [{label}]")
        return
    raise SystemExit(
        f"[{label}] expected one old block or an already-patched new block in "
        f"{path}; old_matches={count}. Inspect before continuing."
    )


def replace_function_between(
    path: Path,
    *,
    start_marker: str,
    end_marker: str,
    new_block: str,
    label: str,
    patched_sentinel: str,
) -> None:
    """Replace a function by stable boundary markers rather than exact body text."""
    text = path.read_text(encoding="utf-8")
    if patched_sentinel in text:
        print(f"already patched: {path} [{label}]")
        return
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"[{label}] start marker not found in {path}")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"[{label}] end marker not found in {path}")
    path.write_text(text[:start] + new_block + text[end:], encoding="utf-8")
    print(f"patched: {path} [{label}]")


# 1) Document-scope every retrieval unit identity.
replace_once(
    CHUNKING,
    '''            source_key = f"{doc_id}::{normalized_source_type.value}"
            metadata: Dict[str, Any] = {
                # Keep filename for provenance and legacy compatibility. New
                # prebuilt replacement uses the location-independent source_key.
                "filename": str(path),
                "source_key": source_key,
                "chunk_idx": len(documents),
''',
    '''            source_key = f"{doc_id}::{normalized_source_type.value}"
            retrieval_unit_key = f"{source_key}::{record_id}"
            metadata: Dict[str, Any] = {
                # Keep filename for provenance and legacy compatibility. New
                # prebuilt replacement uses the location-independent source_key.
                "filename": str(path),
                "source_key": source_key,
                # Globally unique across documents and retrieval representations.
                "retrieval_unit_key": retrieval_unit_key,
                "chunk_idx": len(documents),
''',
    "document-scoped retrieval unit key",
)

# 2) Make deterministic vector IDs global-corpus safe.
replace_function_between(
    INDEX,
    start_marker="    @staticmethod\n    def _stable_vector_ids(",
    end_marker="    def create_index(self) -> None:",
    new_block='    @staticmethod\n    def _stable_vector_ids(documents: List[Document]) -> Optional[List[str]]:\n        """Return deterministic backend-safe IDs for prebuilt records.\n\n        Legacy Markdown chunks have no ``record_id`` and retain LangChain-generated\n        IDs. Prebuilt records are scoped by ``retrieval_unit_key`` when available,\n        otherwise by ``source_key`` + ``record_id``. This prevents collisions when\n        two guidelines contain the same local record/section identifier.\n        """\n        record_ids = [\n            str(document.metadata.get("record_id") or "").strip()\n            for document in documents\n        ]\n        has_record_ids = [bool(record_id) for record_id in record_ids]\n        if any(has_record_ids) and not all(has_record_ids):\n            raise ValueError(\n                "A document batch cannot mix records with and without metadata[\'record_id\']"\n            )\n        if not any(has_record_ids):\n            return None\n\n        identities: List[str] = []\n        for document, record_id in zip(documents, record_ids):\n            retrieval_unit_key = str(\n                document.metadata.get("retrieval_unit_key") or ""\n            ).strip()\n            if retrieval_unit_key:\n                identities.append(retrieval_unit_key)\n                continue\n\n            source_key = str(document.metadata.get("source_key") or "").strip()\n            identities.append(\n                f"{source_key}::{record_id}" if source_key else record_id\n            )\n\n        if len(identities) != len(set(identities)):\n            raise ValueError(\n                "Duplicate document-scoped retrieval identities in indexing batch"\n            )\n        return [\n            str(uuid5(NAMESPACE_URL, f"cardiology-rag:{identity}"))\n            for identity in identities\n        ]\n',
    label="global-safe stable vector IDs",
    patched_sentinel="Duplicate document-scoped retrieval identities in indexing batch",
)

replace_once(
    INDEX,
    '''        Prebuilt records are replaced by a stable ``metadata['source_key']``
        independent of the local filesystem path. Legacy Markdown chunks keep
        the historical filename-based replacement. Deterministic vector-store
        IDs continue to derive from ``metadata['record_id']``.
''',
    '''        Prebuilt records are replaced by a stable ``metadata['source_key']``
        independent of the local filesystem path. Legacy Markdown chunks keep
        the historical filename-based replacement. Deterministic vector-store
        IDs derive from the document-scoped retrieval identity.
''',
    "stable ID documentation",
)

# 3) Keep root .env usable when reproducible configs live under experiments/.
replace_once(
    MAIN,
    '''        load_dotenv(dotenv_path=config_dir / ".env")
        os.environ["CONFIG_PATH"] = str(config_path)
''',
    '''        env_path = config_dir / ".env"
        if not env_path.is_file():
            repo_env = Path(__file__).resolve().parents[1] / ".env"
            env_path = repo_env if repo_env.is_file() else env_path
        load_dotenv(dotenv_path=env_path)
        os.environ["CONFIG_PATH"] = str(config_path)
''',
    "config-local env with repository fallback",
)

# 4) Add multi-source config normalization after artifact metadata extraction.
marker = '''    return metadata

def _index_output_files(processor: Any, base_dir: Path) -> dict[str, Any]:
'''
insert = '''    return metadata


def _resolve_prebuilt_sources(
    *,
    config_path: Path,
    run_config: dict[str, Any],
    mode: str,
    source_type_factory: Any,
) -> list[dict[str, Any]]:
    """Normalize one or many prebuilt sources into a strict corpus definition."""
    if mode == "prebuilt":
        source_value = run_config.get("source_path")
        source_type_value = run_config.get("source_type")
        if not source_value or not source_type_value:
            raise ValueError(
                "Prebuilt mode requires run.source_path and run.source_type"
            )
        raw_sources: list[dict[str, Any]] = [
            {
                "source_path": source_value,
                "source_type": source_type_value,
                "expected_chunk_count": run_config.get("expected_chunk_count"),
            }
        ]
    elif mode == "prebuilt_multi":
        configured = run_config.get("sources")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "prebuilt_multi mode requires a non-empty run.sources list"
            )
        if not all(isinstance(item, dict) for item in configured):
            raise TypeError("Every run.sources item must be a JSON object")
        raw_sources = [dict(item) for item in configured]
    else:
        raise ValueError(f"Unsupported prebuilt mode: {mode!r}")

    resolved: list[dict[str, Any]] = []
    seen_source_keys: set[str] = set()
    for source_index, item in enumerate(raw_sources):
        source_value = item.get("source_path")
        source_type_value = item.get("source_type")
        if not source_value or not source_type_value:
            raise ValueError(
                f"run.sources[{source_index}] requires source_path and source_type"
            )
        source_path = _resolve_from_config(config_path, source_value)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_type = source_type_factory(source_type_value)
        artifact = _artifact_metadata(source_path)
        doc_id = str(artifact.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(
                f"Cannot determine doc_id from prebuilt artifact: {source_path}"
            )
        source_key = f"{doc_id}::{source_type.value}"
        if source_key in seen_source_keys:
            raise ValueError(
                "Duplicate document/source_type in configured corpus: "
                f"{source_key}"
            )
        seen_source_keys.add(source_key)

        expected_count = item.get("expected_chunk_count")
        if expected_count is not None:
            expected_count = int(expected_count)
            if expected_count < 1:
                raise ValueError(
                    f"expected_chunk_count must be >= 1 for {source_key}"
                )

        resolved.append(
            {
                "source_path": source_path,
                "source_type": source_type.value,
                "source_key": source_key,
                "artifact": artifact,
                "expected_chunk_count": expected_count,
            }
        )
    return resolved


def _index_output_files(processor: Any, base_dir: Path) -> dict[str, Any]:
'''
replace_once(MAIN, marker, insert, "multi-source config normalization")

# 5) Replace the single-source manifest with a corpus manifest (while keeping the old
#    `source` field for single-source backwards compatibility).
old_manifest = '''def _write_manifest(
    *,
    processor: Any,
    config_path: Path,
    source_path: Path,
    source_type: str,
    indexed_count: int,
) -> Path:
    base_dir = config_path.parent.resolve()
    artifact = _artifact_metadata(source_path)
    source_key = None
    if artifact.get("doc_id"):
        source_key = f"{artifact['doc_id']}::{source_type}"
    manifest = {
        "schema_version": "rag_index_build_v2",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_id": processor.app_id,
        "mode": "prebuilt",
        "config": {
            "path": _display_path(config_path, base_dir),
            "sha256": _sha256(config_path),
        },
        "source": {
            "path": _display_path(source_path, base_dir),
            "type": source_type,
            "source_key": source_key,
            "sha256": _sha256(source_path),
            "artifact": artifact,
        },
        "indexed_chunk_count": indexed_count,
        "stored_vector_count": (
            processor.index_manager.get_n_documents_in_vectorstore()
        ),
        "index": {
            "name": processor.config.indexing.name,
            "folder": str(processor.config.indexing.folder),
            "type": processor.config.indexing.type.value,
            "distance": processor.config.indexing.distance.value,
            "output_files": _index_output_files(processor, base_dir),
        },
        "embeddings": {
            "model": processor.config.embeddings.model_name,
            "dimensions": processor.config.embeddings.dim,
            "input_template": "title_body_v1",
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "git": _git_provenance(base_dir),
    }
    output_path = Path(processor.config.indexing.folder) / "build_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )
    return output_path
'''
new_manifest = '''def _write_manifest(
    *,
    processor: Any,
    config_path: Path,
    mode: str,
    sources: list[dict[str, Any]],
) -> Path:
    base_dir = config_path.parent.resolve()
    manifest_sources: list[dict[str, Any]] = []
    for source in sources:
        source_path = Path(source["source_path"])
        item = {
            "path": _display_path(source_path, base_dir),
            "type": source["source_type"],
            "source_key": source["source_key"],
            "sha256": _sha256(source_path),
            "artifact": source["artifact"],
            "indexed_chunk_count": int(source["indexed_chunk_count"]),
        }
        if source.get("expected_chunk_count") is not None:
            item["expected_chunk_count"] = int(source["expected_chunk_count"])
        manifest_sources.append(item)

    total_indexed = sum(item["indexed_chunk_count"] for item in manifest_sources)
    manifest = {
        "schema_version": "rag_index_build_v3",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_id": processor.app_id,
        "mode": mode,
        "config": {
            "path": _display_path(config_path, base_dir),
            "sha256": _sha256(config_path),
        },
        "sources": manifest_sources,
        "indexed_chunk_count": total_indexed,
        "stored_vector_count": (
            processor.index_manager.get_n_documents_in_vectorstore()
        ),
        "index": {
            "name": processor.config.indexing.name,
            "folder": str(processor.config.indexing.folder),
            "type": processor.config.indexing.type.value,
            "distance": processor.config.indexing.distance.value,
            "output_files": _index_output_files(processor, base_dir),
        },
        "embeddings": {
            "model": processor.config.embeddings.model_name,
            "dimensions": processor.config.embeddings.dim,
            "input_template": "title_body_v1",
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "git": _git_provenance(base_dir),
    }
    if mode == "prebuilt" and len(manifest_sources) == 1:
        # Backwards compatibility for tools that still read the v2 single-source key.
        manifest["source"] = manifest_sources[0]

    output_path = Path(processor.config.indexing.folder) / "build_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )
    return output_path
'''
replace_once(MAIN, old_manifest, new_manifest, "multi-source build manifest")

# 6) Replace the single-source execution block with prebuilt/prebuilt_multi execution.
old_run = '''        if mode != "prebuilt":
            raise ValueError(
                f"Unsupported run mode for {args.app_id}: {mode!r}"
            )
        source_value = run_config.get("source_path")
        source_type_value = run_config.get("source_type")
        if not source_value or not source_type_value:
            raise ValueError(
                "Prebuilt mode requires run.source_path and run.source_type"
            )

        source_path = _resolve_from_config(config_path, source_value)
        source_type = PrebuiltChunkSource(source_type_value)
        # Avoid leaving a stale success manifest when a rebuild fails midway.
        manifest_path = Path(processor.config.indexing.folder) / "build_manifest.json"
        manifest_path.unlink(missing_ok=True)

        indexed_count = processor.process_prebuilt_chunks(
            source_path=source_path,
            source_type=source_type,
        )
        stored_count = processor.index_manager.get_n_documents_in_vectorstore()
        expected_count = run_config.get("expected_chunk_count")
        if expected_count is not None and indexed_count != int(expected_count):
            raise RuntimeError(
                "Indexed chunk count differs from config expectation: "
                f"expected={expected_count}, actual={indexed_count}"
            )
        if expected_count is not None and stored_count != int(expected_count):
            raise RuntimeError(
                "Stored vector count differs from config expectation: "
                f"expected={expected_count}, actual={stored_count}"
            )
        manifest_path = _write_manifest(
            processor=processor,
            config_path=config_path,
            source_path=source_path,
            source_type=source_type.value,
            indexed_count=indexed_count,
        )
        print(f"app_id: {args.app_id}")
        print(f"source: {_display_path(source_path, config_dir)}")
        print(f"source_type: {source_type.value}")
        print(f"indexed_chunks: {indexed_count}")
        print(f"stored_vectors: {stored_count}")
        print(f"index_folder: {processor.config.indexing.folder}")
        print(f"manifest: {manifest_path}")
        print("OK: configured RAG indexing run completed.")
        return 0
'''
new_run = '''        if mode not in {"prebuilt", "prebuilt_multi"}:
            raise ValueError(
                f"Unsupported run mode for {args.app_id}: {mode!r}"
            )

        sources = _resolve_prebuilt_sources(
            config_path=config_path,
            run_config=run_config,
            mode=mode,
            source_type_factory=PrebuiltChunkSource,
        )
        # Avoid leaving a stale success manifest when a rebuild fails midway.
        manifest_path = Path(processor.config.indexing.folder) / "build_manifest.json"
        manifest_path.unlink(missing_ok=True)

        indexed_sources: list[dict[str, Any]] = []
        for source in sources:
            source_path = Path(source["source_path"])
            source_type = PrebuiltChunkSource(source["source_type"])
            indexed_count = processor.process_prebuilt_chunks(
                source_path=source_path,
                source_type=source_type,
            )
            expected_count = source.get("expected_chunk_count")
            if expected_count is not None and indexed_count != int(expected_count):
                raise RuntimeError(
                    "Indexed chunk count differs from source expectation: "
                    f"source={source['source_key']}, expected={expected_count}, "
                    f"actual={indexed_count}"
                )
            indexed_sources.append({**source, "indexed_chunk_count": indexed_count})

        total_indexed = sum(
            int(source["indexed_chunk_count"]) for source in indexed_sources
        )
        stored_count = processor.index_manager.get_n_documents_in_vectorstore()

        expected_total = run_config.get("expected_total_chunk_count")
        if expected_total is not None and total_indexed != int(expected_total):
            raise RuntimeError(
                "Indexed corpus size differs from config expectation: "
                f"expected={expected_total}, actual={total_indexed}"
            )
        if expected_total is not None and stored_count != int(expected_total):
            raise RuntimeError(
                "Stored vector count differs from config expectation: "
                f"expected={expected_total}, actual={stored_count}"
            )

        if mode == "prebuilt_multi" and stored_count != total_indexed:
            raise RuntimeError(
                "Multi-document index contains vectors outside the configured corpus "
                "or is missing configured vectors: "
                f"configured/indexed={total_indexed}, stored={stored_count}. "
                "Re-run with --recreate-index for a clean corpus build."
            )
        if mode == "prebuilt" and sources[0].get("expected_chunk_count") is not None:
            expected_count = int(sources[0]["expected_chunk_count"])
            if stored_count != expected_count:
                raise RuntimeError(
                    "Stored vector count differs from config expectation: "
                    f"expected={expected_count}, actual={stored_count}"
                )

        manifest_path = _write_manifest(
            processor=processor,
            config_path=config_path,
            mode=mode,
            sources=indexed_sources,
        )
        print(f"app_id: {args.app_id}")
        for source in indexed_sources:
            print(
                "source: "
                f"{_display_path(Path(source['source_path']), config_dir)} | "
                f"source_type={source['source_type']} | "
                f"source_key={source['source_key']} | "
                f"indexed_chunks={source['indexed_chunk_count']}"
            )
        print(f"indexed_chunks_total: {total_indexed}")
        print(f"stored_vectors: {stored_count}")
        print(f"index_folder: {processor.config.indexing.folder}")
        print(f"manifest: {manifest_path}")
        print("OK: configured RAG indexing run completed.")
        return 0
'''
replace_once(MAIN, old_run, new_run, "prebuilt_multi execution")

# 7) Add tracked, API-free contract tests under experiments/ (tests/ is ignored in this repo).
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
contract_test = VALIDATION_DIR / "test_multidoc_prebuilt_contract.py"
contract_test.write_text(
    '''from __future__ import annotations

import json

import pytest

from src.managers.chunking_manager import ChunkingManager, PrebuiltChunkSource
from src.managers.index_manager import IndexManager


def _record(doc_id: str, record_id: str, section_id: str = "7.2") -> dict:
    return {
        "doc_id": doc_id,
        "retrieval_unit_id": record_id,
        "section_view_role": "retrieval",
        "retrieval_strategy": "hierarchical_sections",
        "section_id": section_id,
        "section_title": "Synthetic section",
        "text": f"Evidence from {doc_id}",
        "source_section_ids": [section_id],
        "source_chunk_ids": [f"{record_id}::chunk"],
        "represented_section_ids": [section_id],
        "embed": True,
        "excluded": False,
        "is_empty": False,
    }


def _write_artifact(path, *records: dict) -> None:
    path.write_text(
        json.dumps({"chunks": list(records)}, indent=2) + "\\n",
        encoding="utf-8",
    )


def test_same_local_record_id_in_two_documents_gets_distinct_global_ids(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_artifact(a, _record("doc-A", "section-7.2"))
    _write_artifact(b, _record("doc-B", "section-7.2"))

    manager = ChunkingManager([])
    docs_a = manager.load_prebuilt_chunks(
        a, PrebuiltChunkSource.hierarchical_section_view
    )
    docs_b = manager.load_prebuilt_chunks(
        b, PrebuiltChunkSource.hierarchical_section_view
    )

    assert docs_a[0].metadata["source_key"] != docs_b[0].metadata["source_key"]
    assert (
        docs_a[0].metadata["retrieval_unit_key"]
        != docs_b[0].metadata["retrieval_unit_key"]
    )
    ids_a = IndexManager._stable_vector_ids(docs_a)
    ids_b = IndexManager._stable_vector_ids(docs_b)
    assert ids_a is not None and ids_b is not None
    assert ids_a[0] != ids_b[0]

    # Deterministic when the same artifact is loaded again.
    docs_a_again = manager.load_prebuilt_chunks(
        a, PrebuiltChunkSource.hierarchical_section_view
    )
    assert IndexManager._stable_vector_ids(docs_a_again) == ids_a


def test_one_prebuilt_artifact_cannot_mix_documents(tmp_path):
    mixed = tmp_path / "mixed.json"
    _write_artifact(
        mixed,
        _record("doc-A", "u1", "1"),
        _record("doc-B", "u2", "2"),
    )
    manager = ChunkingManager([])
    with pytest.raises(ValueError, match="exactly one doc_id"):
        manager.load_prebuilt_chunks(
            mixed, PrebuiltChunkSource.hierarchical_section_view
        )
''',
    encoding="utf-8",
)
print(f"created: {contract_test}")

validator = VALIDATION_DIR / "validate_multidoc_index.py"
validator.write_text(
    '''#!/usr/bin/env python3
"""Validate a locally-built prebuilt_multi FAISS corpus without API calls."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import faiss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-docs", required=True, type=int)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "prebuilt_multi":
        raise SystemExit(f"Expected prebuilt_multi manifest, got {manifest.get('mode')!r}")

    sources = manifest.get("sources") or []
    if len(sources) != args.expected_docs:
        raise SystemExit(
            f"Expected {args.expected_docs} configured documents, got {len(sources)}"
        )
    source_keys = [str(item.get("source_key") or "") for item in sources]
    if len(source_keys) != len(set(source_keys)) or not all(source_keys):
        raise SystemExit("Manifest source_key values are missing or duplicated")

    expected_doc_ids = {
        str(item.get("artifact", {}).get("doc_id") or "").strip()
        for item in sources
    }
    if "" in expected_doc_ids or len(expected_doc_ids) != args.expected_docs:
        raise SystemExit(f"Bad document IDs in manifest: {sorted(expected_doc_ids)}")

    total_indexed = int(manifest["indexed_chunk_count"])
    stored = int(manifest["stored_vector_count"])
    source_total = sum(int(item["indexed_chunk_count"]) for item in sources)
    if not (total_indexed == stored == source_total):
        raise SystemExit(
            "Count mismatch: "
            f"manifest_total={total_indexed}, stored={stored}, source_sum={source_total}"
        )

    index_name = manifest["index"]["name"]
    index_dir = manifest_path.parent
    faiss_path = index_dir / f"{index_name}.faiss"
    pkl_path = index_dir / f"{index_name}.pkl"
    if not faiss_path.is_file() or not pkl_path.is_file():
        raise SystemExit(
            f"Missing FAISS/docstore files next to manifest: {faiss_path}, {pkl_path}"
        )

    index = faiss.read_index(str(faiss_path))
    if index.ntotal != stored:
        raise SystemExit(f"FAISS ntotal={index.ntotal} but manifest stored={stored}")

    # Trusted local LangChain artifact produced by this project.
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise SystemExit("Unexpected LangChain FAISS pickle structure")
    docstore, index_to_docstore_id = payload
    documents = [docstore.search(doc_id) for doc_id in index_to_docstore_id.values()]
    documents = [doc for doc in documents if doc is not None]
    if len(documents) != stored:
        raise SystemExit(
            f"Docstore contains {len(documents)} resolved documents, expected {stored}"
        )

    doc_ids = [str(doc.metadata.get("doc_id") or "").strip() for doc in documents]
    actual_doc_ids = set(doc_ids)
    if actual_doc_ids != expected_doc_ids:
        raise SystemExit(
            f"Index document set mismatch: expected={sorted(expected_doc_ids)}, "
            f"actual={sorted(actual_doc_ids)}"
        )

    retrieval_keys = [
        str(doc.metadata.get("retrieval_unit_key") or "").strip()
        for doc in documents
    ]
    if not all(retrieval_keys):
        raise SystemExit("Some indexed records have no retrieval_unit_key")
    if len(retrieval_keys) != len(set(retrieval_keys)):
        raise SystemExit("Duplicate retrieval_unit_key values found in corpus")

    counts = Counter(doc_ids)
    print("OK: multi-document FAISS corpus contract satisfied")
    print(f"manifest: {manifest_path}")
    print(f"vectors: {stored}")
    print(f"documents: {sorted(actual_doc_ids)}")
    print("vectors_per_document:")
    for doc_id, count in sorted(counts.items()):
        print(f"  {doc_id}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    encoding="utf-8",
)
validator.chmod(0o755)
print(f"created: {validator}")

print("\nPatch complete. Suggested next commands:")
print("  python -m py_compile src/main.py src/managers/chunking_manager.py src/managers/index_manager.py")
print("  PYTHONPATH=. pytest -q experiments/validation/test_multidoc_prebuilt_contract.py")
