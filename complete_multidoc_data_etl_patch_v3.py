#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
MAIN = ROOT / 'src/main.py'
CHUNKING = ROOT / 'src/managers/chunking_manager.py'
INDEX = ROOT / 'src/managers/index_manager.py'
VALIDATION_DIR = ROOT / 'experiments/validation'

for required in (MAIN, CHUNKING, INDEX):
    if not required.is_file():
        raise SystemExit(f'Missing {required}. Run from data-etl repository root.')


def insert_before(path: Path, marker: str, block: str, sentinel: str, label: str) -> None:
    text = path.read_text(encoding='utf-8')
    if sentinel in text:
        print(f'already patched: {path} [{label}]')
        return
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit(f'[{label}] marker not found in {path}')
    path.write_text(text[:pos] + block + text[pos:], encoding='utf-8')
    print(f'patched: {path} [{label}]')


def replace_between(path: Path, start_marker: str, end_marker: str, new_block: str, sentinel: str, label: str) -> None:
    text = path.read_text(encoding='utf-8')
    if sentinel in text:
        print(f'already patched: {path} [{label}]')
        return
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f'[{label}] start marker not found in {path}')
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f'[{label}] end marker not found in {path}')
    path.write_text(text[:start] + new_block + text[end:], encoding='utf-8')
    print(f'patched: {path} [{label}]')


# Verify the prior partial patch is present.
chunk_text = CHUNKING.read_text(encoding='utf-8')
if '"retrieval_unit_key": retrieval_unit_key' not in chunk_text:
    raise SystemExit('Expected retrieval_unit_key patch is missing from chunking_manager.py')
print(f'verified: {CHUNKING} [document-scoped retrieval unit key]')

index_text = INDEX.read_text(encoding='utf-8')
if 'Duplicate document-scoped retrieval identities in indexing batch' not in index_text:
    raise SystemExit('Expected document-scoped stable-ID patch is missing from index_manager.py')
print(f'verified: {INDEX} [global-safe stable vector IDs]')

# 1) Multi-source source normalization helper.
helper = '''def _resolve_prebuilt_sources(\n    *,\n    config_path: Path,\n    run_config: dict[str, Any],\n    mode: str,\n    source_type_factory: Any,\n) -> list[dict[str, Any]]:\n    """Normalize one or many prebuilt sources into a strict corpus definition."""\n    if mode == "prebuilt":\n        source_value = run_config.get("source_path")\n        source_type_value = run_config.get("source_type")\n        if not source_value or not source_type_value:\n            raise ValueError(\n                "Prebuilt mode requires run.source_path and run.source_type"\n            )\n        raw_sources: list[dict[str, Any]] = [\n            {\n                "source_path": source_value,\n                "source_type": source_type_value,\n                "expected_chunk_count": run_config.get("expected_chunk_count"),\n            }\n        ]\n    elif mode == "prebuilt_multi":\n        configured = run_config.get("sources")\n        if not isinstance(configured, list) or not configured:\n            raise ValueError(\n                "prebuilt_multi mode requires a non-empty run.sources list"\n            )\n        if not all(isinstance(item, dict) for item in configured):\n            raise TypeError("Every run.sources item must be a JSON object")\n        raw_sources = [dict(item) for item in configured]\n    else:\n        raise ValueError(f"Unsupported prebuilt mode: {mode!r}")\n\n    resolved: list[dict[str, Any]] = []\n    seen_source_keys: set[str] = set()\n    for source_index, item in enumerate(raw_sources):\n        source_value = item.get("source_path")\n        source_type_value = item.get("source_type")\n        if not source_value or not source_type_value:\n            raise ValueError(\n                f"run.sources[{source_index}] requires source_path and source_type"\n            )\n\n        source_path = _resolve_from_config(config_path, source_value)\n        if not source_path.is_file():\n            raise FileNotFoundError(source_path)\n\n        source_type = source_type_factory(source_type_value)\n        artifact = _artifact_metadata(source_path)\n        doc_id = str(artifact.get("doc_id") or "").strip()\n        if not doc_id:\n            raise ValueError(\n                f"Cannot determine doc_id from prebuilt artifact: {source_path}"\n            )\n\n        source_key = f"{doc_id}::{source_type.value}"\n        if source_key in seen_source_keys:\n            raise ValueError(\n                "Duplicate document/source_type in configured corpus: "\n                f"{source_key}"\n            )\n        seen_source_keys.add(source_key)\n\n        expected_count = item.get("expected_chunk_count")\n        if expected_count is not None:\n            expected_count = int(expected_count)\n            if expected_count < 1:\n                raise ValueError(\n                    f"expected_chunk_count must be >= 1 for {source_key}"\n                )\n\n        resolved.append(\n            {\n                "source_path": source_path,\n                "source_type": source_type.value,\n                "source_key": source_key,\n                "artifact": artifact,\n                "expected_chunk_count": expected_count,\n            }\n        )\n    return resolved\n\n\n'''
insert_before(
    MAIN,
    'def _index_output_files(processor: Any, base_dir: Path) -> dict[str, Any]:',
    helper,
    'def _resolve_prebuilt_sources(',
    'multi-source config normalization',
)

# 2) Corpus-aware manifest writer.
manifest_fn = '''def _write_manifest(\n    *,\n    processor: Any,\n    config_path: Path,\n    mode: str,\n    sources: list[dict[str, Any]],\n) -> Path:\n    base_dir = config_path.parent.resolve()\n    manifest_sources: list[dict[str, Any]] = []\n\n    for source in sources:\n        source_path = Path(source["source_path"])\n        item = {\n            "path": _display_path(source_path, base_dir),\n            "type": source["source_type"],\n            "source_key": source["source_key"],\n            "sha256": _sha256(source_path),\n            "artifact": source["artifact"],\n            "indexed_chunk_count": int(source["indexed_chunk_count"]),\n        }\n        if source.get("expected_chunk_count") is not None:\n            item["expected_chunk_count"] = int(source["expected_chunk_count"])\n        manifest_sources.append(item)\n\n    total_indexed = sum(item["indexed_chunk_count"] for item in manifest_sources)\n    manifest = {\n        "schema_version": "rag_index_build_v3",\n        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),\n        "app_id": processor.app_id,\n        "mode": mode,\n        "config": {\n            "path": _display_path(config_path, base_dir),\n            "sha256": _sha256(config_path),\n        },\n        "sources": manifest_sources,\n        "indexed_chunk_count": total_indexed,\n        "stored_vector_count": processor.index_manager.get_n_documents_in_vectorstore(),\n        "index": {\n            "name": processor.config.indexing.name,\n            "folder": str(processor.config.indexing.folder),\n            "type": processor.config.indexing.type.value,\n            "distance": processor.config.indexing.distance.value,\n            "output_files": _index_output_files(processor, base_dir),\n        },\n        "embeddings": {\n            "model": processor.config.embeddings.model_name,\n            "dimensions": processor.config.embeddings.dim,\n            "input_template": "title_body_v1",\n        },\n        "runtime": {\n            "python": platform.python_version(),\n            "packages": _package_versions(),\n        },\n        "git": _git_provenance(base_dir),\n    }\n\n    if mode == "prebuilt" and len(manifest_sources) == 1:\n        manifest["source"] = manifest_sources[0]\n\n    output_path = Path(processor.config.indexing.folder) / "build_manifest.json"\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    output_path.write_text(\n        json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n",\n        encoding="utf-8",\n    )\n    return output_path\n\n\n'''
replace_between(
    MAIN,
    'def _write_manifest(',
    'def main(argv: list[str] | None = None) -> int:',
    manifest_fn,
    '"schema_version": "rag_index_build_v3"',
    'multi-source build manifest',
)

# 3) Replace single-source prebuilt execution with corpus execution.
run_block = '''        if mode not in {"prebuilt", "prebuilt_multi"}:\n            raise ValueError(\n                f"Unsupported run mode for {args.app_id}: {mode!r}"\n            )\n\n        sources = _resolve_prebuilt_sources(\n            config_path=config_path,\n            run_config=run_config,\n            mode=mode,\n            source_type_factory=PrebuiltChunkSource,\n        )\n\n        manifest_path = Path(processor.config.indexing.folder) / "build_manifest.json"\n        manifest_path.unlink(missing_ok=True)\n\n        indexed_sources: list[dict[str, Any]] = []\n        for source in sources:\n            source_path = Path(source["source_path"])\n            source_type = PrebuiltChunkSource(source["source_type"])\n            indexed_count = processor.process_prebuilt_chunks(\n                source_path=source_path,\n                source_type=source_type,\n            )\n\n            expected_count = source.get("expected_chunk_count")\n            if expected_count is not None and indexed_count != int(expected_count):\n                raise RuntimeError(\n                    "Indexed chunk count differs from source expectation: "\n                    f"source={source['source_key']}, expected={expected_count}, "\n                    f"actual={indexed_count}"\n                )\n\n            indexed_sources.append(\n                {**source, "indexed_chunk_count": indexed_count}\n            )\n\n        total_indexed = sum(\n            int(source["indexed_chunk_count"]) for source in indexed_sources\n        )\n        stored_count = processor.index_manager.get_n_documents_in_vectorstore()\n\n        expected_total = run_config.get("expected_total_chunk_count")\n        if expected_total is not None and total_indexed != int(expected_total):\n            raise RuntimeError(\n                "Indexed corpus size differs from config expectation: "\n                f"expected={expected_total}, actual={total_indexed}"\n            )\n        if expected_total is not None and stored_count != int(expected_total):\n            raise RuntimeError(\n                "Stored vector count differs from config expectation: "\n                f"expected={expected_total}, actual={stored_count}"\n            )\n\n        if mode == "prebuilt_multi" and stored_count != total_indexed:\n            raise RuntimeError(\n                "Multi-document index contains vectors outside the configured corpus "\n                "or is missing configured vectors: "\n                f"configured/indexed={total_indexed}, stored={stored_count}. "\n                "Re-run with --recreate-index for a clean corpus build."\n            )\n\n        if mode == "prebuilt" and sources[0].get("expected_chunk_count") is not None:\n            expected_count = int(sources[0]["expected_chunk_count"])\n            if stored_count != expected_count:\n                raise RuntimeError(\n                    "Stored vector count differs from config expectation: "\n                    f"expected={expected_count}, actual={stored_count}"\n                )\n\n        manifest_path = _write_manifest(\n            processor=processor,\n            config_path=config_path,\n            mode=mode,\n            sources=indexed_sources,\n        )\n\n        print(f"app_id: {args.app_id}")\n        for source in indexed_sources:\n            print(\n                "source: "\n                f"{_display_path(Path(source['source_path']), config_dir)} | "\n                f"source_type={source['source_type']} | "\n                f"source_key={source['source_key']} | "\n                f"indexed_chunks={source['indexed_chunk_count']}"\n            )\n        print(f"indexed_chunks_total: {total_indexed}")\n        print(f"stored_vectors: {stored_count}")\n        print(f"index_folder: {processor.config.indexing.folder}")\n        print(f"manifest: {manifest_path}")\n        print("OK: configured RAG indexing run completed.")\n        return 0\n'''
replace_between(
    MAIN,
    '        if mode != "prebuilt":',
    '    finally:',
    run_block,
    'if mode not in {"prebuilt", "prebuilt_multi"}:',
    'prebuilt_multi execution',
)

# 4) API-free validation tests/utilities.
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
contract_test = VALIDATION_DIR / 'test_multidoc_prebuilt_contract.py'
contract_test.write_text('''from __future__ import annotations\n\nimport json\n\nimport pytest\n\nfrom src.managers.chunking_manager import ChunkingManager, PrebuiltChunkSource\nfrom src.managers.index_manager import IndexManager\n\n\ndef _record(doc_id: str, record_id: str, section_id: str = "7.2") -> dict:\n    return {\n        "doc_id": doc_id,\n        "retrieval_unit_id": record_id,\n        "section_view_role": "retrieval",\n        "retrieval_strategy": "hierarchical_sections",\n        "section_id": section_id,\n        "section_title": "Synthetic section",\n        "text": f"Evidence from {doc_id}",\n        "source_section_ids": [section_id],\n        "source_chunk_ids": [f"{record_id}::chunk"],\n        "represented_section_ids": [section_id],\n        "embed": True,\n        "excluded": False,\n        "is_empty": False,\n    }\n\n\ndef _write_artifact(path, *records: dict) -> None:\n    path.write_text(\n        json.dumps({"chunks": list(records)}, indent=2) + "\\n",\n        encoding="utf-8",\n    )\n\n\ndef test_same_local_record_id_in_two_documents_gets_distinct_global_ids(tmp_path):\n    a = tmp_path / "a.json"\n    b = tmp_path / "b.json"\n    _write_artifact(a, _record("doc-A", "section-7.2"))\n    _write_artifact(b, _record("doc-B", "section-7.2"))\n\n    manager = ChunkingManager([])\n    docs_a = manager.load_prebuilt_chunks(\n        a, PrebuiltChunkSource.hierarchical_section_view\n    )\n    docs_b = manager.load_prebuilt_chunks(\n        b, PrebuiltChunkSource.hierarchical_section_view\n    )\n\n    assert docs_a[0].metadata["source_key"] != docs_b[0].metadata["source_key"]\n    assert docs_a[0].metadata["retrieval_unit_key"] != docs_b[0].metadata["retrieval_unit_key"]\n\n    ids_a = IndexManager._stable_vector_ids(docs_a)\n    ids_b = IndexManager._stable_vector_ids(docs_b)\n    assert ids_a is not None and ids_b is not None\n    assert ids_a[0] != ids_b[0]\n\n    docs_a_again = manager.load_prebuilt_chunks(\n        a, PrebuiltChunkSource.hierarchical_section_view\n    )\n    assert IndexManager._stable_vector_ids(docs_a_again) == ids_a\n\n\ndef test_one_prebuilt_artifact_cannot_mix_documents(tmp_path):\n    mixed = tmp_path / "mixed.json"\n    _write_artifact(\n        mixed,\n        _record("doc-A", "u1", "1"),\n        _record("doc-B", "u2", "2"),\n    )\n    manager = ChunkingManager([])\n    with pytest.raises(ValueError, match="exactly one doc_id"):\n        manager.load_prebuilt_chunks(\n            mixed, PrebuiltChunkSource.hierarchical_section_view\n        )\n''', encoding='utf-8')
print(f'created: {contract_test}')

validator = VALIDATION_DIR / 'validate_multidoc_index.py'
validator.write_text('''#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport pickle\nfrom collections import Counter\nfrom pathlib import Path\n\nimport faiss\n\n\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", required=True, type=Path)\n    parser.add_argument("--expected-docs", required=True, type=int)\n    args = parser.parse_args()\n\n    manifest_path = args.manifest.resolve()\n    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))\n    if manifest.get("mode") != "prebuilt_multi":\n        raise SystemExit(f"Expected prebuilt_multi manifest, got {manifest.get('mode')!r}")\n\n    sources = manifest.get("sources") or []\n    if len(sources) != args.expected_docs:\n        raise SystemExit(f"Expected {args.expected_docs} configured documents, got {len(sources)}")\n\n    expected_doc_ids = {\n        str(item.get("artifact", {}).get("doc_id") or "").strip()\n        for item in sources\n    }\n    if "" in expected_doc_ids or len(expected_doc_ids) != args.expected_docs:\n        raise SystemExit(f"Bad document IDs in manifest: {sorted(expected_doc_ids)}")\n\n    total_indexed = int(manifest["indexed_chunk_count"])\n    stored = int(manifest["stored_vector_count"])\n    source_total = sum(int(item["indexed_chunk_count"]) for item in sources)\n    if not (total_indexed == stored == source_total):\n        raise SystemExit(\n            f"Count mismatch: manifest_total={total_indexed}, stored={stored}, source_sum={source_total}"\n        )\n\n    index_name = manifest["index"]["name"]\n    index_dir = manifest_path.parent\n    faiss_path = index_dir / f"{index_name}.faiss"\n    pkl_path = index_dir / f"{index_name}.pkl"\n    if not faiss_path.is_file() or not pkl_path.is_file():\n        raise SystemExit(f"Missing FAISS/docstore files: {faiss_path}, {pkl_path}")\n\n    index = faiss.read_index(str(faiss_path))\n    if index.ntotal != stored:\n        raise SystemExit(f"FAISS ntotal={index.ntotal} but manifest stored={stored}")\n\n    with pkl_path.open("rb") as handle:\n        payload = pickle.load(handle)\n    if not isinstance(payload, tuple) or len(payload) != 2:\n        raise SystemExit("Unexpected LangChain FAISS pickle structure")\n\n    docstore, index_to_docstore_id = payload\n    documents = [docstore.search(doc_id) for doc_id in index_to_docstore_id.values()]\n    documents = [doc for doc in documents if doc is not None]\n    if len(documents) != stored:\n        raise SystemExit(f"Docstore contains {len(documents)} documents, expected {stored}")\n\n    doc_ids = [str(doc.metadata.get("doc_id") or "").strip() for doc in documents]\n    if set(doc_ids) != expected_doc_ids:\n        raise SystemExit(\n            f"Index document set mismatch: expected={sorted(expected_doc_ids)}, actual={sorted(set(doc_ids))}"\n        )\n\n    retrieval_keys = [\n        str(doc.metadata.get("retrieval_unit_key") or "").strip()\n        for doc in documents\n    ]\n    if not all(retrieval_keys):\n        raise SystemExit("Some indexed records have no retrieval_unit_key")\n    if len(retrieval_keys) != len(set(retrieval_keys)):\n        raise SystemExit("Duplicate retrieval_unit_key values found in corpus")\n\n    counts = Counter(doc_ids)\n    print("OK: multi-document FAISS corpus contract satisfied")\n    print(f"vectors: {stored}")\n    print(f"documents: {sorted(set(doc_ids))}")\n    for doc_id, count in sorted(counts.items()):\n        print(f"  {doc_id}: {count}")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n''', encoding='utf-8')
validator.chmod(0o755)
print(f'created: {validator}')

print('\nPatch complete.')
print('Next: git diff --check')
print('Next: python -m py_compile src/main.py src/managers/chunking_manager.py src/managers/index_manager.py')
print('Next: PYTHONPATH=. pytest -q experiments/validation/test_multidoc_prebuilt_contract.py')
