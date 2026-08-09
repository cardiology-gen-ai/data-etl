"""Resolve the chunk artifact used to build retrieval Section views.

The canonical hierarchical chunks remain immutable. When text cleaning is
active, this module builds or validates the clean-chunk sidecar and returns it
as the only text source for Section-view construction.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from managers.text_cleaning_manager import (
    VERSION as TEXT_CLEANING_VERSION,
    load_or_build_clean_chunks,
)

CANONICAL_SOURCE_KIND = "canonical_chunks"
CLEAN_SOURCE_KIND = "clean_chunks"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_cleaning_enabled(config: Any) -> bool:
    configured = getattr(config, "run_text_cleaning", None)
    if configured is not None:
        return bool(configured)
    return _env_bool("KG_RUN_TEXT_CLEANING", True)

def force_text_cleaning(config: Any) -> bool:
    configured = getattr(config, "force_text_cleaning", None)
    if configured is not None:
        return bool(configured)
    return _env_bool("KG_FORCE_TEXT_CLEANING", False)

def get_clean_chunk_dir(config: Any) -> Path:
    configured = getattr(config, "clean_chunk_dir", None)
    if configured is not None:
        return Path(configured)
    return Path(config.chunk_dir).parent / "clean_chunks"


def get_text_cleaning_audit_dir(config: Any) -> Path:
    configured = getattr(config, "text_cleaning_audit_dir", None)
    if configured is not None:
        return Path(configured)
    profile = TEXT_CLEANING_VERSION.removeprefix("canonical_text_")
    return Path(config.chunk_dir).parent / "text_cleaning_audit" / profile


def ensure_clean_chunk_dirs(config: Any) -> None:
    if not text_cleaning_enabled(config):
        return
    get_clean_chunk_dir(config).mkdir(parents=True, exist_ok=True)
    get_text_cleaning_audit_dir(config).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SectionViewChunkSource:
    canonical_path: Path
    source_path: Path
    source_kind: str
    canonical_sha256: str
    source_sha256: str
    text_cleaning_enabled: bool
    text_cleaning_version: str | None
    text_cleaning_audit_path: Path | None
    text_cleaning_cache_status: str | None

    def cache_metadata(self) -> Dict[str, Any]:
        # Only stable provenance belongs in the Section-view cache key.
        # `text_cleaning_cache_status` changes from rebuilt to reused and must
        # therefore remain a log-only value.
        return {
            "source_chunk_kind": self.source_kind,
            "source_chunk_file": self.source_path.name,
            "source_chunk_sha256": self.source_sha256,
            "canonical_chunk_file": self.canonical_path.name,
            "canonical_chunk_sha256": self.canonical_sha256,
            "text_cleaning_enabled": self.text_cleaning_enabled,
            "text_cleaning_version": self.text_cleaning_version,
            "text_cleaning_audit_file": (
                self.text_cleaning_audit_path.name
                if self.text_cleaning_audit_path is not None
                else None
            ),
        }


def resolve_section_view_chunk_source(
    config: Any,
    canonical_chunk_path: Path,
) -> SectionViewChunkSource:
    """Return the validated artifact that must feed the Section-view builder."""
    canonical_chunk_path = Path(canonical_chunk_path).resolve()
    if not canonical_chunk_path.exists():
        raise FileNotFoundError(
            f"Canonical chunk file not found: {canonical_chunk_path}"
        )

    canonical_hash = sha256_file(canonical_chunk_path)
    if not text_cleaning_enabled(config):
        return SectionViewChunkSource(
            canonical_path=canonical_chunk_path,
            source_path=canonical_chunk_path,
            source_kind=CANONICAL_SOURCE_KIND,
            canonical_sha256=canonical_hash,
            source_sha256=canonical_hash,
            text_cleaning_enabled=False,
            text_cleaning_version=None,
            text_cleaning_audit_path=None,
            text_cleaning_cache_status=None,
        )

    ensure_clean_chunk_dirs(config)
    clean_path, report = load_or_build_clean_chunks(
        input_path=canonical_chunk_path,
        output_dir=get_clean_chunk_dir(config),
        audit_dir=get_text_cleaning_audit_dir(config),
        force=force_text_cleaning(config),
    )
    clean_path = Path(clean_path).resolve()

    cleaning_version = report.get("cleaning_version")
    if cleaning_version != TEXT_CLEANING_VERSION:
        raise ValueError(
            "Unexpected text-cleaning version for Section-view source: "
            f"expected {TEXT_CLEANING_VERSION!r}, got {cleaning_version!r}"
        )
    if report.get("source_sha256") != canonical_hash:
        raise ValueError(
            "Clean-chunk audit source hash does not match the canonical chunk "
            f"file: {canonical_chunk_path}"
        )

    clean_hash = sha256_file(clean_path)
    if report.get("output_sha256") != clean_hash:
        raise ValueError(
            "Clean-chunk audit output hash does not match the clean artifact: "
            f"{clean_path}"
        )

    audit_path_raw = report.get("audit_path")
    audit_path = Path(audit_path_raw).resolve() if audit_path_raw else None
    if audit_path is None or not audit_path.exists():
        raise FileNotFoundError(
            f"Text-cleaning audit is missing for {canonical_chunk_path.name}"
        )

    return SectionViewChunkSource(
        canonical_path=canonical_chunk_path,
        source_path=clean_path,
        source_kind=CLEAN_SOURCE_KIND,
        canonical_sha256=canonical_hash,
        source_sha256=clean_hash,
        text_cleaning_enabled=True,
        text_cleaning_version=str(cleaning_version),
        text_cleaning_audit_path=audit_path,
        text_cleaning_cache_status=str(report.get("cache_status") or "unknown"),
    )
