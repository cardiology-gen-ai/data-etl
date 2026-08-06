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
        load_dotenv(dotenv_path=config_dir / ".env")
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

        if mode != "prebuilt":
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
    finally:
        if hf_token:
            try:
                huggingface_hub.logout()
            except OSError:
                pass
        os.chdir(previous_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
