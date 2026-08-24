from __future__ import annotations

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
        json.dumps({"chunks": list(records)}, indent=2) + "\n",
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
    assert docs_a[0].metadata["retrieval_unit_key"] != docs_b[0].metadata["retrieval_unit_key"]

    ids_a = IndexManager._stable_vector_ids(docs_a)
    ids_b = IndexManager._stable_vector_ids(docs_b)
    assert ids_a is not None and ids_b is not None
    assert ids_a[0] != ids_b[0]

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
