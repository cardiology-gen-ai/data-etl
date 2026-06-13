import json
import pathlib
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Tuple, Type

from cardiology_gen_ai import Singleton
from cardiology_gen_ai.utils.logger import get_logger

from knowledge_graph.structured_recommendation import build_extraction_chain, validate_extraction
from managers.tables.table_manager import TableManager


PROMPT_VERSION = "v2.0"

def _get_retryable_exception_types() -> Tuple[Type[BaseException], ...]:
    """Discover OpenAI client exception classes worth retrying."""
    excs: list[Type[BaseException]] = []
    try:
        from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
        excs.extend([RateLimitError, APITimeoutError, APIConnectionError, InternalServerError])
    except ImportError:
        pass
    return tuple(excs)


_RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = _get_retryable_exception_types()


def _extract_retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Best-effort extraction of a ``Retry-After`` hint from an exception."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            headers = getattr(resp, "headers", None) or {}
            for key in ("retry-after", "Retry-After"):
                value = None
                try:
                    value = headers.get(key)
                except AttributeError:
                    pass
                if value:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        except Exception:  # noqa: BLE001
            pass
    for attr in ("retry_after", "_retry_after"):
        val = getattr(exc, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class ExtractionEntry:
    """One row of the extractions catalog."""
    recommendation_id: str            # f"{table_id}::row_{row_index}"
    doc_id: str
    table_id: str
    row_index: int
    table_caption: str
    section_path: list[str]
    container_id: Optional[str]
    source_text: str
    group_header: Optional[str]
    effective_source: str
    catalog_class: Optional[str]
    catalog_level: Optional[str]
    prompt_version: str
    model: str
    extracted_at: str
    ok: bool
    extraction: Optional[dict[str, Any]]
    validation_flags: dict[str, str]
    error: Optional[str]
    retry_attempts: int = 0
    acronyms_snapshot: Optional[dict[str, str]] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str | dict) -> "ExtractionEntry":
        data = json.loads(line) if isinstance(line, str) else line
        defaults = {
            "doc_id": "",
            "container_id": None,
            "group_header": None,
            "effective_source": data.get("source_text", ""),
            "acronyms_snapshot": None,
            "retry_attempts": 0,
        }
        for k, v in defaults.items():
            data.setdefault(k, v)
        return cls(**data)


@dataclass
class ExtractionsCatalog:
    catalog: list[ExtractionEntry] = field(default_factory=list)

    def done_ids(self, prompt_version: str) -> set[str]:
        return {
            e.recommendation_id
            for e in self.catalog
            if e.ok and e.prompt_version == prompt_version
        }

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.catalog),
            "ok": sum(1 for e in self.catalog if e.ok),
            "errors": sum(1 for e in self.catalog if not e.ok),
            "flagged": sum(1 for e in self.catalog if e.ok and e.validation_flags),
            "modality_mismatch": sum(
                1 for e in self.catalog
                if e.ok and "modality_mismatch" in e.validation_flags
            ),
            "non_verbatim": sum(
                1 for e in self.catalog
                if e.ok and any(
                    k.startswith("non_verbatim_") for k in e.validation_flags
                )
            ),
            "bad_type": sum(
                1 for e in self.catalog
                if e.ok and any(
                    k.startswith("bad_type_") for k in e.validation_flags
                )
            ),
            "rows_with_retries": sum(
                1 for e in self.catalog if e.retry_attempts > 0
            ),
            "total_retry_attempts": sum(e.retry_attempts for e in self.catalog),
        }


class ExtractionsManager:
    """Per-document I/O + extraction loop with rate-limit-aware retries."""

    def __init__(
        self,
        input_tables_catalog,
        save_path: pathlib.Path,
        doc_id: str,
        model: str,
        prompt_version: str,
        chain: Any,
        acronyms: Optional[dict[str, str]] = None,
        max_retries: int = 6,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
        inter_request_delay: float = 0.0,
    ):
        self.logger = get_logger("ExtractionsManager")
        self.input_catalog = input_tables_catalog
        self.save_path = save_path
        self.doc_id = doc_id
        self.model = model
        self.prompt_version = prompt_version
        self.chain = chain
        self.acronyms = acronyms or None

        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if initial_backoff <= 0 or max_backoff <= 0:
            raise ValueError("backoff values must be positive")
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.inter_request_delay = max(0.0, inter_request_delay)

        self.catalog: ExtractionsCatalog = ExtractionsCatalog()

    def load(self, must_exist: bool = False) -> None:
        if not self.save_path.exists():
            if must_exist:
                raise FileNotFoundError(self.save_path)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            return
        with open(self.save_path, "r") as f:
            data = json.load(f)
            for entry in data:
                try:
                    self.catalog.catalog.append(ExtractionEntry.from_json(entry))
                except (json.JSONDecodeError, TypeError) as exc:
                    self.logger.warning("Skipping malformed JSONL line: %s", exc)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.save_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            data = [json.loads(entry.to_jsonl()) for entry in self.catalog.catalog]
            json.dump(data, f, indent=4, ensure_ascii=False)
        tmp.replace(self.save_path)

    def _iter_extractable_rows(self) -> Iterator[dict[str, Any]]:
        for table in self.input_catalog:
            attr = getattr(table, "attribution", None)
            section_path = (
                list(getattr(attr, "section_path", None) or []) if attr else []
            )
            container_id = getattr(attr, "container_id", None) if attr else None

            for row in table.recommendation_rows:
                if getattr(row, "is_section_row", False):
                    continue
                text = (getattr(row, "recommendation", "") or "").strip()
                if not text:
                    continue
                yield {
                    "table_id": table.id,
                    "table_caption": getattr(table, "caption", "") or "",
                    "section_path": section_path,
                    "container_id": container_id,
                    "row_index": row.row_index,
                    "recommendation": text,
                    "group_header": (getattr(row, "group_header", None) or None),
                    "class_": getattr(row, "class_", None)
                              or getattr(row, "raw_class", None) or None,
                    "level": getattr(row, "level", None)
                             or getattr(row, "raw_level", None) or None,
                }

    @staticmethod
    def _build_effective_source(
        group_header: Optional[str], recommendation: str
    ) -> str:
        if not group_header:
            return recommendation
        header = group_header.rstrip(". ").strip()
        return f"{header}. {recommendation}" if header else recommendation

    def _invoke_chain_with_retry(
        self, chain_input: dict[str, Any], row_id: str,
    ) -> Tuple[Any, int]:
        if not _RETRYABLE_EXCEPTIONS:
            return self.chain.invoke(chain_input), 0

        delay = self.initial_backoff
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.chain.invoke(chain_input)
                return result, attempt - 1
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    self.logger.error(
                        "[%s] chain.invoke failed after %d attempts (%s): %s",
                        row_id, attempt, type(exc).__name__, exc,
                    )
                    raise

                retry_after = _extract_retry_after_seconds(exc)
                base_wait = retry_after if retry_after is not None else delay
                # Full jitter in [0.5*base, 1.0*base]
                wait_for = base_wait * (0.5 + random.random() * 0.5)
                wait_for = min(wait_for, self.max_backoff)

                self.logger.warning(
                    "[%s][attempt %d/%d] %s: %s -- waiting %.1fs%s",
                    row_id, attempt, self.max_retries,
                    type(exc).__name__, str(exc)[:200], wait_for,
                    f" (Retry-After={retry_after}s)" if retry_after else "",
                )
                time.sleep(wait_for)
                delay = min(delay * 2.0, self.max_backoff)

        assert last_exc is not None
        raise last_exc

    def _extract_one(self, payload: dict[str, Any]) -> ExtractionEntry:
        rec_id = f"{payload['table_id']}::row_{payload['row_index']}"
        effective_source = self._build_effective_source(
            payload.get("group_header"), payload["recommendation"]
        )
        base = dict(
            recommendation_id=rec_id,
            doc_id=self.doc_id,
            table_id=payload["table_id"],
            row_index=payload["row_index"],
            table_caption=payload["table_caption"],
            section_path=payload["section_path"],
            container_id=payload["container_id"],
            source_text=payload["recommendation"],
            group_header=payload.get("group_header"),
            effective_source=effective_source,
            catalog_class=payload["class_"],
            catalog_level=payload["level"],
            prompt_version=self.prompt_version,
            model=self.model,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            acronyms_snapshot=self.acronyms,
        )
        try:
            result, attempts = self._invoke_chain_with_retry(
                chain_input={
                    "class_": payload["class_"] or "(not specified)",
                    "level": payload["level"] or "(not specified)",
                    "recommendation": effective_source,
                },
                row_id=rec_id,
            )
            flags = validate_extraction(
                result, payload["class_"], effective_source
            )
            return ExtractionEntry(
                **base,
                ok=True,
                extraction=result.model_dump(),
                validation_flags=flags,
                error=None,
                retry_attempts=attempts,
            )
        except Exception as exc:  # noqa: BLE001
            return ExtractionEntry(
                **base,
                ok=False,
                extraction=None,
                validation_flags={},
                error=f"{type(exc).__name__}: {exc}",
                retry_attempts=self.max_retries if isinstance(exc, _RETRYABLE_EXCEPTIONS) else 0,
            )

    def extract(self, limit: Optional[int] = None) -> ExtractionsCatalog:
        done = self.catalog.done_ids(self.prompt_version)
        todo = [
            p for p in self._iter_extractable_rows()
            if f"{p['table_id']}::row_{p['row_index']}" not in done
        ]
        if limit is not None:
            todo = todo[:limit]

        self.logger.info(
            "Extracting %d rows (resume: %d already done at %s, "
            "max_retries=%d, inter_request_delay=%.2fs)",
            len(todo), len(done), self.prompt_version,
            self.max_retries, self.inter_request_delay,
        )

        for i, payload in enumerate(todo, 1):
            if i > 1 and self.inter_request_delay > 0:
                time.sleep(self.inter_request_delay)

            entry = self._extract_one(payload)
            self.catalog.catalog.append(entry)
            self.save()
            self.logger.debug(
                "[%d/%d] %s ok=%s attempts=%d flags=%s",
                i, len(todo), entry.recommendation_id, entry.ok,
                entry.retry_attempts, list(entry.validation_flags),
            )
        return self.catalog

    @staticmethod
    def _error_is_retryable(error: Optional[str]) -> bool:
        if not error:
            return False
        retryable_names = {t.__name__ for t in _RETRYABLE_EXCEPTIONS}
        # Strict prefix match: "RateLimitError: ..."
        head = error.split(":", 1)[0].strip()
        if head in retryable_names:
            return True
        # Loose fallback (covers wrapped/renamed providers).
        lower = error.lower()
        return any(needle in lower for needle in (
            "ratelimit", "rate_limit", "rate limit",
            "timeout", "timed out",
            "connection", "connect ",
            "internal server", "service unavailable", "bad gateway",
            "503", "502", "504",
        ))

    def _rebuild_payload_for_entry(
        self, entry: ExtractionEntry,
    ) -> Optional[dict[str, Any]]:
        for table in self.input_catalog:
            if getattr(table, "id", None) != entry.table_id:
                continue
            for row in table.recommendation_rows:
                if getattr(row, "row_index", None) != entry.row_index:
                    continue
                if getattr(row, "is_section_row", False):
                    return None
                text = (getattr(row, "recommendation", "") or "").strip()
                if not text:
                    return None
                attr = getattr(table, "attribution", None)
                section_path = (
                    list(getattr(attr, "section_path", None) or [])
                    if attr else []
                )
                container_id = getattr(attr, "container_id", None) if attr else None
                return {
                    "table_id": table.id,
                    "table_caption": getattr(table, "caption", "") or "",
                    "section_path": section_path,
                    "container_id": container_id,
                    "row_index": row.row_index,
                    "recommendation": text,
                    "group_header": (
                        getattr(row, "group_header", None) or entry.group_header
                    ),
                    "class_": getattr(row, "class_", None)
                              or getattr(row, "raw_class", None)
                              or entry.catalog_class,
                    "level": getattr(row, "level", None)
                             or getattr(row, "raw_level", None)
                             or entry.catalog_level,
                }
        return None

    def retry_failed(
        self,
        cooldown: float = 30.0,
        max_retries_per_row: Optional[int] = None,
        inter_request_delay: Optional[float] = None,
    ) -> dict[str, int]:
        """Re-process entries that previously failed with a retryable error."""
        original_max_retries = self.max_retries
        original_inter_delay = self.inter_request_delay
        if max_retries_per_row is not None:
            if max_retries_per_row < 1:
                raise ValueError("max_retries_per_row must be >= 1")
            self.max_retries = max_retries_per_row
        if inter_request_delay is not None:
            self.inter_request_delay = max(0.0, inter_request_delay)

        try:
            # Identify candidates: ok=False AND retryable error class.
            candidates: list[tuple[int, ExtractionEntry]] = [
                (i, e) for i, e in enumerate(self.catalog.catalog)
                if not e.ok and self._error_is_retryable(e.error)
            ]
            stats = {
                "candidates": len(candidates),
                "recovered": 0,
                "still_failing": 0,
                "unrecoverable": 0,
            }
            if not candidates:
                self.logger.info("retry_failed: no retryable failures.")
                return stats

            self.logger.info(
                "retry_failed: %d candidate(s) -- cooldown=%.1fs, "
                "max_retries=%d, inter_request_delay=%.2fs",
                len(candidates), cooldown, self.max_retries,
                self.inter_request_delay,
            )
            if cooldown > 0:
                time.sleep(cooldown)

            for i, (idx, old_entry) in enumerate(candidates, 1):
                if i > 1 and self.inter_request_delay > 0:
                    time.sleep(self.inter_request_delay)

                payload = self._rebuild_payload_for_entry(old_entry)
                if payload is None:
                    stats["unrecoverable"] += 1
                    self.logger.warning(
                        "[%d/%d] %s: source row no longer in tables catalog; "
                        "leaving failure as-is.",
                        i, len(candidates), old_entry.recommendation_id,
                    )
                    continue

                new_entry = self._extract_one(payload)
                # Replace in-place so order and indices stay stable.
                self.catalog.catalog[idx] = new_entry
                self.save()

                if new_entry.ok:
                    stats["recovered"] += 1
                    self.logger.info(
                        "[%d/%d] %s recovered (attempts=%d)",
                        i, len(candidates), new_entry.recommendation_id,
                        new_entry.retry_attempts,
                    )
                else:
                    stats["still_failing"] += 1
                    self.logger.warning(
                        "[%d/%d] %s still failing: %s",
                        i, len(candidates), new_entry.recommendation_id,
                        new_entry.error,
                    )

            self.logger.info(
                "retry_failed done: recovered=%d / candidates=%d "
                "(still_failing=%d, unrecoverable=%d)",
                stats["recovered"], stats["candidates"],
                stats["still_failing"], stats["unrecoverable"],
            )
            return stats
        finally:
            self.max_retries = original_max_retries
            self.inter_request_delay = original_inter_delay

class RecommendationExtractionManager(metaclass=Singleton):
    """Per-file orchestration of LLM-based recommendation extraction."""

    def __init__(
        self,
        output_folder: pathlib.Path,
        tabs_folder: pathlib.Path,
        app_id: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        prompt_version: str = PROMPT_VERSION,
        acronym_folder: Optional[pathlib.Path] = None,
        request_timeout: float = 60.0,
        max_retries: int = 6,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
        inter_request_delay: float = 0.0,
        # Final cleanup pass on rate-limited failures.
        retry_failed_pass: bool = True,
        retry_failed_cooldown: float = 30.0,
        retry_failed_max_retries: Optional[int] = None,
        retry_failed_inter_request_delay: Optional[float] = 1.0,
    ):
        self.logger = get_logger("RecommendationExtractor")
        self.output_folder = output_folder
        self.tabs_folder = tabs_folder
        self.acronym_folder = acronym_folder
        self.app_id = app_id
        self.model = model
        self.temperature = temperature
        self.prompt_version = prompt_version

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.inter_request_delay = inter_request_delay

        self.retry_failed_pass = retry_failed_pass
        self.retry_failed_cooldown = retry_failed_cooldown
        self.retry_failed_max_retries = retry_failed_max_retries
        self.retry_failed_inter_request_delay = retry_failed_inter_request_delay

        self.chain = build_extraction_chain(
            model=model,
            temperature=temperature,
            timeout=request_timeout,
            max_retries=0,
        )

    def _load_acronyms(self, doc_id: str) -> Optional[dict[str, str]]:
        if not self.acronym_folder:
            return None
        path = pathlib.Path(self.acronym_folder) / f"{doc_id}.json"
        if not path.exists():
            self.logger.info(
                "No acronym file for %s at %s; proceeding without.", doc_id, path
            )
            return None
        try:
            with path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning(
                "Failed to load acronyms for %s: %s; proceeding without.",
                doc_id, exc,
            )
            return None
        if not isinstance(data, dict):
            self.logger.warning(
                "Acronym file for %s is not a JSON object; proceeding without.",
                doc_id,
            )
            return None
        return data

    def _build_manager(
        self, filepath: pathlib.Path,
    ) -> Tuple["ExtractionsManager", int, int]:
        """Build an ExtractionsManager for one document, loading the tables catalog, the acronym dict, and any existing extractions JSON from disk."""
        self.doc_id = filepath.stem

        tables_folder = self.tabs_folder / f"{self.doc_id}_tables"
        tm = TableManager(filepath=filepath, save_folder=tables_folder)
        tm.load(must_exist=True, recommendation=True)

        acronyms = self._load_acronyms(self.doc_id)

        save_folder = pathlib.Path(self.output_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        em = ExtractionsManager(
            input_tables_catalog=tm.catalog,
            save_path=save_folder / f"{self.doc_id}_recommendations.json",
            doc_id=self.doc_id,
            model=self.model,
            prompt_version=self.prompt_version,
            chain=self.chain,
            acronyms=acronyms,
            max_retries=self.max_retries,
            initial_backoff=self.initial_backoff,
            max_backoff=self.max_backoff,
            inter_request_delay=self.inter_request_delay,
        )
        em.load(must_exist=False)
        return em, (len(tm.catalog) if tm.catalog else 0), (len(acronyms) if acronyms else 0)

    def __call__(self, filepath: pathlib.Path) -> ExtractionsCatalog:
        em, n_tables, n_acronyms = self._build_manager(filepath)

        if n_tables == 0:
            self.logger.info(
                "Empty tables catalog for %s; skipping extraction.", filepath.name,
            )
            return ExtractionsCatalog()

        self.logger.info(
            "Extracting recommendations from %d tables for %s using model=%s prompt=%s acronyms=%d timeout=%.1fs max_retries=%d",
            n_tables, filepath.name, self.model, self.prompt_version,
            n_acronyms, self.request_timeout, self.max_retries,
        )

        em.extract()

        # Optional final pass on rate-limit / transient failures.
        if self.retry_failed_pass:
            retry_stats = em.retry_failed(
                cooldown=self.retry_failed_cooldown,
                max_retries_per_row=self.retry_failed_max_retries,
                inter_request_delay=self.retry_failed_inter_request_delay,
            )
            self.logger.info(
                "  retry_failed: recovered=%d / candidates=%d "
                "(still_failing=%d, unrecoverable=%d)",
                retry_stats["recovered"], retry_stats["candidates"],
                retry_stats["still_failing"], retry_stats["unrecoverable"],
            )

        stats = em.catalog.stats()
        self.logger.info(
            "  extracted: %d/%d (errors=%d, flagged=%d, "
            "modality_mismatch=%d, non_verbatim=%d, bad_type=%d, "
            "rows_with_retries=%d, total_retry_attempts=%d)",
            stats["ok"], stats["total"], stats["errors"], stats["flagged"],
            stats["modality_mismatch"], stats["non_verbatim"], stats["bad_type"],
            stats["rows_with_retries"], stats["total_retry_attempts"],
        )
        return em.catalog

    def retry_failed_file(
        self,
        filepath: pathlib.Path,
        cooldown: float = 0.0,
        max_retries_per_row: Optional[int] = None,
        inter_request_delay: Optional[float] = None,
    ) -> dict[str, int]:
        """Re-run the retry pass on an existing on-disk extractions catalog."""
        em, n_tables, _ = self._build_manager(filepath)

        if not em.catalog.catalog:
            raise FileNotFoundError(
                f"No extractions catalog on disk for {filepath.stem}. "
                "Run RecommendationExtractionManager(filepath) first."
            )
        if n_tables == 0:
            self.logger.warning(
                "Empty tables catalog for %s; cannot rebuild payloads, "
                "no retry possible.", filepath.name,
            )
            return {
                "candidates": 0, "recovered": 0,
                "still_failing": 0, "unrecoverable": 0,
            }

        self.logger.info(
            "Retrying failed extractions for %s (catalog size=%d, model=%s, "
            "max_retries=%d, cooldown=%.1fs)",
            filepath.name, len(em.catalog.catalog), self.model,
            max_retries_per_row or self.max_retries, cooldown,
        )
        stats = em.retry_failed(
            cooldown=cooldown,
            max_retries_per_row=max_retries_per_row,
            inter_request_delay=inter_request_delay,
        )
        self.logger.info(
            "  retry_failed: recovered=%d / candidates=%d "
            "(still_failing=%d, unrecoverable=%d)",
            stats["recovered"], stats["candidates"],
            stats["still_failing"], stats["unrecoverable"],
        )
        return stats