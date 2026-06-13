import json
import pathlib
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from managers.parsing.parsing_manager import ParsedTable
from managers.tables.scorer import LexicalScorer


class SectionAttribution(BaseModel):
    """Section attribution derived from a flat TOC + caption matching."""

    # Page-based: deepest TOC section whose [page_start, page_end] contains the table page.
    container_id: str
    container_title: str

    # Caption-match: best-scoring ancestor or nearby section boundary candidate.
    topic_id: str
    topic_title: str
    topic_score: float

    # Full path from document root to container, as "id. title" strings.
    section_path: List[str]

    # True when the topic is not an ancestor of the container.
    cross_section: bool = False

    # True when topic_score < threshold.
    low_confidence: bool = False

    # Which scorer produced this: "lexical" | "embedding:<model>"
    scorer: str = "lexical"


class RecommendationRow(BaseModel):
    """One structured row from a recommendation table.

    `class_` is exposed as JSON key `"class"` because `class` is a Python keyword.
    """

    row_index: int
    recommendation: str

    raw_class: Optional[str] = None
    class_: Optional[str] = Field(default=None, alias="class")

    raw_level: Optional[str] = None
    level: Optional[str] = None

    # True for internal section/header rows that span all columns or behave like
    # table-section labels. These rows keep `recommendation` but have no class/level.
    is_section_row: bool = False

    # The verbatim text of the most recent section/header row that precedes
    # this row (or None for rows before any header). Populated only on
    # non-section rows: for `is_section_row=True` rows this stays None to
    # avoid self-references. Inherited by downstream LLM extraction so that
    # population scoping declared at the group level (e.g. "Patients with
    # HFrEF") is visible when extracting each individual recommendation.
    group_header: Optional[str] = None

    model_config = {
        "populate_by_name": True,
    }


class RecommendationTablesCatalogEntry(BaseModel):
    """Reduced catalog entry for the separate recommendation-only JSON file."""

    id: str
    caption: Optional[str] = None
    footnote: Optional[str] = None
    attribution: Optional[SectionAttribution] = None
    recommendation_rows: List[RecommendationRow] = []


class TablesCatalogEntry(BaseModel):
    id: str
    filepath: pathlib.Path
    page: int
    bbox: Optional[List[float]] = None

    # Raw table content, if persisted in the catalog.
    # Detection prefers html over markdown.
    html: Optional[str] = None
    markdown: Optional[str] = None

    # Optional paths, for backward compatibility with catalogs that store files.
    html_path: Optional[pathlib.Path] = None
    markdown_path: Optional[pathlib.Path] = None

    imagepath: Optional[pathlib.Path] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None

    # Recommendation-table metadata.
    recommendation: bool = False
    recommendation_columns: Optional[Dict[str, str]] = None
    recommendation_rows: Optional[List[RecommendationRow]] = None
    recommendation_invalid_values: Optional[Dict[str, List[str]]] = None

    # Filled by TableManager.attribute_sections().
    attribution: Optional[SectionAttribution] = None

    def make_alt_text(self) -> str:
        """Sanitized one-liner suitable as Markdown alt text."""
        alt = (self.caption or self.id).strip()
        alt = re.sub(r"\s+", " ", alt).strip()
        alt = re.sub(r"[\[\]\(\)]", "", alt)
        return alt or "table"

    def is_recommendation_table(self) -> bool:
        return self.recommendation

    @classmethod
    def from_parsed(cls, table: ParsedTable, source_pdf: pathlib.Path) -> "TablesCatalogEntry":
        return cls(
            id=table.id,
            filepath=source_pdf,
            page=table.page,
            bbox=list(table.bbox) if table.bbox else None,
            html=getattr(table, "html", None),
            markdown=getattr(table, "markdown", None),
            imagepath=table.imagepath,
            caption=table.caption,
            footnote=table.footnote,
        )


class TablesCatalog(BaseModel):
    name: str = "tables_catalog.json"
    catalog: Optional[List[TablesCatalogEntry]] = None


# ============================================================================
# HTML / Markdown table parsing helpers
# ============================================================================


class _HTMLTableParser(HTMLParser):
    """Small HTML table parser that preserves cell text and colspan."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[Dict[str, Any]]] = []
        self._current_row: Optional[List[Dict[str, Any]]] = None
        self._current_cell: Optional[List[str]] = None
        self._current_colspan: int = 1

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_dict = dict(attrs)

        if tag == "tr":
            self._current_row = []

        elif tag in {"th", "td"}:
            self._current_cell = []
            try:
                self._current_colspan = int(attrs_dict.get("colspan", "1"))
            except ValueError:
                self._current_colspan = 1

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in {"th", "td"} and self._current_cell is not None:
            text = " ".join(self._current_cell)
            text = re.sub(r"\s+", " ", unescape(text)).strip()

            if self._current_row is not None:
                self._current_row.append({
                    "text": text,
                    "colspan": self._current_colspan,
                })

            self._current_cell = None
            self._current_colspan = 1

        elif tag == "tr" and self._current_row is not None:
            if any(cell["text"].strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def _rows_from_html(html: Optional[str]) -> List[List[Dict[str, Any]]]:
    if not html:
        return []
    parser = _HTMLTableParser()
    parser.feed(html)
    return parser.rows


def _rows_from_markdown(markdown: Optional[str]) -> List[List[Dict[str, Any]]]:
    if not markdown:
        return []

    rows: List[List[Dict[str, Any]]] = []

    for line in markdown.splitlines():
        if "|" not in line:
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]

        # Skip Markdown separator rows: | --- | :---: | --- |
        if cells and all(re.match(r"^:?-{3,}:?$", c) for c in cells):
            continue

        if any(cells):
            rows.append([
                {"text": cell, "colspan": 1}
                for cell in cells
            ])

    return rows


def _clean_cell(text: Optional[str]) -> str:
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\*\_`]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _norm_header(text: str) -> str:
    text = _clean_cell(text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _norm_value_key(text: Optional[str]) -> str:
    text = _clean_cell(text)

    text = (
        text.replace("Ⅰ", "I")
            .replace("Ⅱ", "II")
            .replace("Ⅲ", "III")
    )

    text = re.sub(
        r"\b(class|level|of|recommendation|recommendations|evidence)\b",
        " ",
        text,
        flags=re.I,
    )

    text = re.sub(r"[^A-Za-z0-9]+", "", text).upper()

    # OCR often reads roman I as lowercase l.
    text = text.replace("L", "I")

    return text


# ============================================================================
# Recommendation-table detector
# ============================================================================


class RecommendationTableDetector:
    """Detect and normalize recommendation tables from HTML/Markdown.

    A table is considered a recommendation table if its header row contains the
    required canonical columns, by default:
        - recommendation
        - class
        - level

    The detector keeps "section rows" that span all columns, assigning them
    `is_section_row=True` and preserving the section text as `recommendation`.
    """

    def __init__(
        self,
        required_columns: Sequence[str] = ("recommendation", "class", "level"),
        allowed_values: Optional[Dict[str, Sequence[str]]] = None,
        header_aliases: Optional[Dict[str, Sequence[str]]] = None,
    ):
        self.required_columns = tuple(required_columns)

        self.allowed_values = allowed_values or {
            "class": ("I", "II", "IIa", "IIb", "III"),
            "level": ("A", "B", "C"),
        }

        self.header_aliases = header_aliases or {
            "recommendation": (
                "recommendation",
                "recommendations",
                "recommendation text",
            ),
            "class": (
                "class",
                "class of recommendation",
                "class of recommendations",
                "cor",
            ),
            "level": (
                "level",
                "level of evidence",
                "loe",
            ),
        }

        self._alias_lookup = {
            _norm_header(alias): canonical
            for canonical, aliases in self.header_aliases.items()
            for alias in aliases
        }

        self._allowed_lookup = {
            key: {
                _norm_value_key(value): value
                for value in values
            }
            for key, values in self.allowed_values.items()
        }

    def detect(
        self,
        html: Optional[str] = None,
        markdown: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Precedence: HTML first, then Markdown.
        rows = _rows_from_html(html) or _rows_from_markdown(markdown)

        if not rows:
            return self._empty_result()

        header_idx, headers, column_map = self._find_header_row(rows)

        recommendation = all(
            col in column_map
            for col in self.required_columns
        )

        if not recommendation:
            return self._empty_result()

        recommendation_rows, invalid = self._extract_recommendation_rows(
            rows=rows[header_idx + 1:],
            headers=headers,
            column_map=column_map,
        )

        return {
            "recommendation": True,
            "recommendation_columns": {
                canonical: _clean_cell(headers[idx].get("text"))
                for canonical, idx in column_map.items()
            },
            "recommendation_rows": recommendation_rows,
            "recommendation_invalid_values": invalid or None,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "recommendation": False,
            "recommendation_columns": None,
            "recommendation_rows": None,
            "recommendation_invalid_values": None,
        }

    def _find_header_row(
        self,
        rows: List[List[Dict[str, Any]]],
    ) -> tuple[int, List[Dict[str, Any]], Dict[str, int]]:
        best_idx = 0
        best_headers = rows[0]
        best_map: Dict[str, int] = {}

        # MinerU may output caption-like rows before the header, so inspect a few.
        for i, row in enumerate(rows[:5]):
            current_map: Dict[str, int] = {}

            for j, cell in enumerate(row):
                canonical = self._canonical_header(cell.get("text", ""))
                if canonical and canonical not in current_map:
                    current_map[canonical] = j

            if len(current_map) > len(best_map):
                best_idx = i
                best_headers = row
                best_map = current_map

            if all(col in current_map for col in self.required_columns):
                return i, row, current_map

        return best_idx, best_headers, best_map

    def _canonical_header(self, header: str) -> Optional[str]:
        norm = _norm_header(header)

        if norm in self._alias_lookup:
            return self._alias_lookup[norm]

        # Recommendation(s)
        if norm in {"recommendation", "recommendations"}:
            return "recommendation"

        # Class, Classa, Class a, Class of recommendation, Class of recommendationa
        if re.match(r"^class(?:\s*of\s*recommendations?)?\s*[a-z]?$", norm):
            return "class"

        # Level, Levelb, Level b, Level of evidence, Level of evidenceb
        if re.match(r"^level(?:\s*of\s*evidence)?\s*[a-z]?$", norm):
            return "level"

        return None

    def _cell_text(
        self,
        row: List[Dict[str, Any]],
        idx: Optional[int],
    ) -> Optional[str]:
        if idx is None or idx >= len(row):
            return None
        return _clean_cell(row[idx].get("text"))

    def _normalize_allowed_value(
        self,
        key: str,
        raw: Optional[str],
    ) -> Optional[str]:
        norm = _norm_value_key(raw)
        if not norm:
            return None
        return self._allowed_lookup.get(key, {}).get(norm)

    def _is_spanning_section_row(
        self,
        row: List[Dict[str, Any]],
        n_header_cols: int,
        column_map: Dict[str, int],
    ) -> bool:
        non_empty_cells = [
            cell
            for cell in row
            if _clean_cell(cell.get("text"))
        ]

        if not non_empty_cells:
            return False

        # HTML: one cell with colspan covering all or almost all columns.
        if len(non_empty_cells) == 1:
            colspan = int(non_empty_cells[0].get("colspan", 1) or 1)
            if colspan >= max(2, n_header_cols - 1):
                return True

        # Markdown/OCR: text in recommendation column, no class/level.
        rec_idx = column_map.get("recommendation")
        class_idx = column_map.get("class")
        level_idx = column_map.get("level")

        rec_text = self._cell_text(row, rec_idx)
        raw_class = self._cell_text(row, class_idx)
        raw_level = self._cell_text(row, level_idx)

        if rec_text and not raw_class and not raw_level:
            return True

        return False

    def _extract_recommendation_rows(
        self,
        rows: List[List[Dict[str, Any]]],
        headers: List[Dict[str, Any]],
        column_map: Dict[str, int],
    ) -> tuple[List[RecommendationRow], Dict[str, List[str]]]:
        recommendation_rows: List[RecommendationRow] = []

        invalid: Dict[str, List[str]] = {
            key: []
            for key in self.allowed_values
        }

        rec_idx = column_map.get("recommendation")
        class_idx = column_map.get("class")
        level_idx = column_map.get("level")
        n_header_cols = len(headers)

        # Cursor: the verbatim text of the most recent section row. Reset to
        # None at the top of the table and updated every time we encounter a
        # row classified as a section header. Non-section rows inherit it.
        current_group_header: Optional[str] = None

        for row_idx, row in enumerate(rows, start=1):
            is_section_row = self._is_spanning_section_row(
                row=row,
                n_header_cols=n_header_cols,
                column_map=column_map,
            )

            if is_section_row:
                recommendation_text = next(
                    (
                        _clean_cell(cell.get("text"))
                        for cell in row
                        if _clean_cell(cell.get("text"))
                    ),
                    "",
                )

                if recommendation_text:
                    # Update the cursor for downstream rows. The header row
                    # itself does NOT inherit a group_header (avoids
                    # self-loops and the case of nested headers reading
                    # ambiguously).
                    current_group_header = recommendation_text
                    recommendation_rows.append(
                        RecommendationRow(
                            row_index=row_idx,
                            recommendation=recommendation_text,
                            raw_class=None,
                            class_=None,
                            raw_level=None,
                            level=None,
                            is_section_row=True,
                            group_header=None,
                        )
                    )

                continue

            recommendation_text = self._cell_text(row, rec_idx) or ""
            raw_class = self._cell_text(row, class_idx)
            raw_level = self._cell_text(row, level_idx)

            norm_class = self._normalize_allowed_value("class", raw_class)
            norm_level = self._normalize_allowed_value("level", raw_level)

            if raw_class and norm_class is None:
                invalid["class"].append(raw_class)

            if raw_level and norm_level is None:
                invalid["level"].append(raw_level)

            if recommendation_text or raw_class or raw_level:
                recommendation_rows.append(
                    RecommendationRow(
                        row_index=row_idx,
                        recommendation=recommendation_text,
                        raw_class=raw_class,
                        class_=norm_class,
                        raw_level=raw_level,
                        level=norm_level,
                        is_section_row=False,
                        group_header=current_group_header,
                    )
                )

        invalid = {
            key: sorted(set(vals))
            for key, vals in invalid.items()
            if vals
        }

        return recommendation_rows, invalid


class TableManager:
    """Build, persist, load and enrich table catalogs.

    Supports:
      - section attribution via TOC + caption matching;
      - recommendation-table detection from HTML/Markdown headers;
      - separate recommendation-only catalog export.
    """

    def __init__(self, filepath: pathlib.Path, save_folder: pathlib.Path):
        self.filepath = filepath
        self.save_folder = save_folder
        self.catalog = TablesCatalog(catalog=[])

    def get_catalog_path(self) -> pathlib.Path:
        return self.save_folder / self.catalog.name

    def get_recommendation_catalog_path(self) -> pathlib.Path:
        catalog_path = self.get_catalog_path()
        return catalog_path.with_name(f"recommendation_{catalog_path.name}")

    def build_from_parsed(self, parsed_tables: List[ParsedTable]) -> TablesCatalog:
        """Populate the catalog from backend-provided tables and persist it."""
        self.save_folder.mkdir(parents=True, exist_ok=True)

        entries = [
            TablesCatalogEntry(
                id=table.id,
                filepath=self.filepath,
                page=table.page,
                bbox=table.bbox,
                html=getattr(table, "html", None),
                markdown=getattr(table, "markdown", None),
                html_path=None,
                markdown_path=None,
                imagepath=table.imagepath,
                caption=table.caption,
                footnote=table.footnote,
            )
            for table in parsed_tables
        ]

        self.catalog = TablesCatalog(catalog=entries)
        self.enrich_recommendation_metadata(persist=False)
        self._persist()
        return self.catalog

    def enrich_recommendation_metadata(
        self,
        detector: Optional[RecommendationTableDetector] = None,
        persist: bool = True,
    ) -> TablesCatalog:
        """Add recommendation-table metadata to every catalog entry.

        Detection uses `entry.html` first, then `entry.markdown`; if those are
        missing it tries `html_path` and `markdown_path`.
        """
        detector = detector or RecommendationTableDetector(
            allowed_values={
                "class": ("I", "II", "IIa", "IIb", "III"),
                "level": ("A", "B", "C"),
            }
        )

        for entry in self.catalog.catalog or []:
            html = getattr(entry, "html", None)
            markdown = getattr(entry, "markdown", None)

            if not html and getattr(entry, "html_path", None):
                path = pathlib.Path(entry.html_path)
                if path.exists():
                    html = path.read_text(encoding="utf-8")

            if not markdown and getattr(entry, "markdown_path", None):
                path = pathlib.Path(entry.markdown_path)
                if path.exists():
                    markdown = path.read_text(encoding="utf-8")

            rec_meta = detector.detect(
                html=html,
                markdown=markdown,
            )

            entry.recommendation = rec_meta["recommendation"]
            entry.recommendation_columns = rec_meta["recommendation_columns"]
            entry.recommendation_rows = rec_meta["recommendation_rows"]
            entry.recommendation_invalid_values = rec_meta["recommendation_invalid_values"]

        if persist:
            self._persist()

        return self.catalog

    def build_recommendation_catalog(self) -> List[RecommendationTablesCatalogEntry]:
        return [
            RecommendationTablesCatalogEntry(
                id=entry.id,
                caption=entry.caption,
                footnote=entry.footnote,
                attribution=entry.attribution,
                recommendation_rows=entry.recommendation_rows or [],
            )
            for entry in self.catalog.catalog or []
            if entry.recommendation
        ]

    def persist_recommendation_catalog(self) -> None:
        entries = self.build_recommendation_catalog()

        payload = [
            entry.model_dump(
                mode="json",
                exclude_none=True,
                by_alias=True,
            )
            for entry in entries
        ]

        path = self.get_recommendation_catalog_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def attribute_sections(
        self,
        toc_path: pathlib.Path,
        scorer: Optional["CaptionScorer"] = None,
        low_confidence_threshold: float = 0.30,
    ) -> TablesCatalog:
        """Enrich each entry with a SectionAttribution.

        The TOC file is expected to have a top-level `flat_toc` key containing
        {id, title, level, page_start, page_end}.
        """
        from managers.tables.section_attribution import SectionAttributor

        attributor = SectionAttributor(
            toc_path=toc_path,
            scorer=scorer or LexicalScorer(),
            low_confidence_threshold=low_confidence_threshold,
        )

        for entry in self.catalog.catalog or []:
            entry.attribution = attributor.attribute(entry)

        self._persist()
        return self.catalog

    def _persist(self) -> None:
        payload = self.catalog.model_dump(
            mode="json",
            exclude_none=True,
            by_alias=True,
        )

        with open(self.get_catalog_path(), "w", encoding="utf-8") as f:
            json.dump(payload.get("catalog") or [], f, ensure_ascii=False, indent=2)

        self.persist_recommendation_catalog()

    def load(self, must_exist: bool = False, recommendation: bool = False) -> TablesCatalog | List[RecommendationTablesCatalogEntry]:
        path = self.get_catalog_path() if not recommendation else self.get_recommendation_catalog_path()

        if not path.exists():
            if must_exist:
                raise FileNotFoundError(f"Tables catalog not found at {path}")
            return TablesCatalog(catalog=[])

        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list) and not recommendation:
            data = {"catalog": data}

        try:
            self.catalog = TablesCatalog.model_validate(data) if not recommendation \
                else [RecommendationTablesCatalogEntry.model_validate(d) for d  in data]
            return self.catalog
        except ValidationError as e:
            raise ValueError(f"Tables catalog validation error: {e}") from e