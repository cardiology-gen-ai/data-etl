from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import huggingface_hub
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "False"


_MANIFEST_PACKAGES = (
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-openai",
    "faiss-cpu",
    "openai",
    "cardiologygenai-coordo",
)

_ARTIFACT_METADATA_KEYS = (
    "version",
    "strategy",
    "doc_id",
    "chunk_size_unit",
    "chunk_size",
    "chunk_overlap",
    "overlap_policy",
    "source_record_count",
    "eligible_source_record_count",
    "output_chunk_count",
    "retrieval_max_level",
    "aggregation_max_level",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one configured ETL or prebuilt RAG indexing application. "
            "Run fixed and hierarchical applications in separate invocations."
        )
    )
    parser.add_argument(
        "--config",
        default=os.getenv("CONFIG_PATH", "config.json"),
        help=(
            "Path to the JSON configuration file. Relative source and index "
            "paths are resolved from this file's directory."
        ),
    )
    parser.add_argument(
        "--app-id",
        default=os.getenv("APP_ID", "cardiology_protocols"),
        help="Application key inside the configuration file.",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the selected index before ingestion.",
    )
    parser.add_argument(
        "--force-md-conv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy mode only: force PDF-to-Markdown conversion.",
    )
    return parser


def _load_run_config(config_path: Path, app_id: str) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if app_id not in payload:
        raise KeyError(f"Application ID not found in {config_path}: {app_id}")
    run_config = payload[app_id].get("run", {"mode": "legacy"})
    if not isinstance(run_config, dict):
        raise TypeError(f"{app_id}.run must be a JSON object")
    return run_config


def _resolve_from_config(config_path: Path, value: str | os.PathLike[str]) -> Path:
    """Resolve a configured path against the configuration directory."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def _display_path(path: Path, base_dir: Path) -> str:
    """Return a portable relative path when it is inside ``base_dir``."""
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in _MANIFEST_PACKAGES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain") if commit else None
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _artifact_metadata(source_path: Path) -> dict[str, Any]:
    """Extract concise, human-readable metadata from a prebuilt artifact."""
    payload = json.loads(source_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        metadata = {
            key: payload[key]
            for key in _ARTIFACT_METADATA_KEYS
            if key in payload
        }
        records = payload.get("chunks") or payload.get("sections") or []
    elif isinstance(payload, list):
        metadata = {"record_count": len(payload)}
        records = payload
    else:
        return {}

    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if not metadata.get("doc_id") and record.get("doc_id"):
                metadata["doc_id"] = str(record["doc_id"])
            if (
                "retrieval_strategy" not in metadata
                and record.get("retrieval_strategy")
            ):
                metadata["retrieval_strategy"] = record["retrieval_strategy"]
            if (
                "aggregation_max_level" not in metadata
                and record.get("aggregation_max_level") is not None
            ):
                metadata["aggregation_max_level"] = record[
                    "aggregation_max_level"
                ]
            if metadata.get("doc_id") and metadata.get("retrieval_strategy"):
                break
    return metadata


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
    folder = Path(processor.config.indexing.folder)
    if not folder.is_absolute():
        folder = base_dir / folder
    folder = folder.resolve()

    output: dict[str, Any] = {}
    candidates = {
        "faiss": folder / f"{processor.config.indexing.name}.faiss",
        "docstore": folder / f"{processor.config.indexing.name}.pkl",
        "index_config": folder / "config.json",
    }
    for label, path in candidates.items():
        if path.is_file():
            output[label] = {
                "path": _display_path(path, base_dir),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return output


def _write_manifest(
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
        "stored_vector_count": processor.index_manager.get_n_documents_in_vectorstore(),
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
        manifest["source"] = manifest_sources[0]

    output_path = Path(processor.config.indexing.folder) / "build_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent.resolve()
    previous_cwd = Path.cwd()
    hf_token: str | None = None

    # All relative paths in the selected config are interpreted from the
    # config directory, regardless of where the CLI command was launched.
    os.chdir(config_dir)
    try:
        env_path = config_dir / ".env"
        if not env_path.is_file():
            repo_env = Path(__file__).resolve().parents[1] / ".env"
            env_path = repo_env if repo_env.is_file() else env_path
        load_dotenv(dotenv_path=env_path)
        os.environ["CONFIG_PATH"] = str(config_path)

        run_config = _load_run_config(config_path, args.app_id)
        mode = str(run_config.get("mode", "legacy")).strip().lower()

        # Delayed imports are intentional: ETLConfigManager historically reads
        # CONFIG_PATH while its module is imported.
        from src.etl_processor import ETLProcessor
        from src.managers.chunking_manager import PrebuiltChunkSource

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            huggingface_hub.login(hf_token)

        processor = ETLProcessor(app_id=args.app_id)

        if args.recreate_index:
            if processor.index_manager.vectorstore.vectorstore_exists():
                processor.index_manager.delete_index()
            processor.index_manager.create_index()

        if mode == "legacy":
            processor.perform_etl(force_md_conv=args.force_md_conv)
            return 0

        if mode not in {"prebuilt", "prebuilt_multi"}:
            raise ValueError(
                f"Unsupported run mode for {args.app_id}: {mode!r}"
            )

        sources = _resolve_prebuilt_sources(
            config_path=config_path,
            run_config=run_config,
            mode=mode,
            source_type_factory=PrebuiltChunkSource,
        )

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

            indexed_sources.append(
                {**source, "indexed_chunk_count": indexed_count}
            )

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
    finally:
        if hf_token:
            try:
                huggingface_hub.logout()
            except OSError:
                pass
        os.chdir(previous_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
