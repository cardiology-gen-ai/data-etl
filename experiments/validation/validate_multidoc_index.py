#!/usr/bin/env python3
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
        raise SystemExit(f"Expected {args.expected_docs} configured documents, got {len(sources)}")

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
            f"Count mismatch: manifest_total={total_indexed}, stored={stored}, source_sum={source_total}"
        )

    index_name = manifest["index"]["name"]
    index_dir = manifest_path.parent
    faiss_path = index_dir / f"{index_name}.faiss"
    pkl_path = index_dir / f"{index_name}.pkl"
    if not faiss_path.is_file() or not pkl_path.is_file():
        raise SystemExit(f"Missing FAISS/docstore files: {faiss_path}, {pkl_path}")

    index = faiss.read_index(str(faiss_path))
    if index.ntotal != stored:
        raise SystemExit(f"FAISS ntotal={index.ntotal} but manifest stored={stored}")

    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise SystemExit("Unexpected LangChain FAISS pickle structure")

    docstore, index_to_docstore_id = payload
    documents = [docstore.search(doc_id) for doc_id in index_to_docstore_id.values()]
    documents = [doc for doc in documents if doc is not None]
    if len(documents) != stored:
        raise SystemExit(f"Docstore contains {len(documents)} documents, expected {stored}")

    doc_ids = [str(doc.metadata.get("doc_id") or "").strip() for doc in documents]
    if set(doc_ids) != expected_doc_ids:
        raise SystemExit(
            f"Index document set mismatch: expected={sorted(expected_doc_ids)}, actual={sorted(set(doc_ids))}"
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
    print(f"vectors: {stored}")
    print(f"documents: {sorted(set(doc_ids))}")
    for doc_id, count in sorted(counts.items()):
        print(f"  {doc_id}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
