
import pathlib
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types import DoclingDocument
from docling_core.types.doc import TableItem

from cardiology_gen_ai.utils.logger import get_logger

from managers.parsing.parsing_manager import ParsingBackend, ParsedDocument, ParsedImage, ParsedTable
from managers.toc_extraction.toc_configs import BaseTocConfig
from managers.toc_extraction.utils import _extract_document_title_from_doc
from utils.text_utils import post_process_markdown


class DoclingBackend(ParsingBackend):
    """Parsing backend that uses Docling for layout and structure."""

    name = "docling"

    def __init__(self, toc_config: BaseTocConfig):
        self.logger = get_logger("DoclingBackend")
        self.toc_config = toc_config
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(
        self,
        pdf_path: pathlib.Path,
        output_dir: pathlib.Path,
        cache_dir: Optional[pathlib.Path] = None,
        images_dir: Optional[pathlib.Path] = None,
    ) -> ParsedDocument:
        result = self._converter.convert(pdf_path.as_posix())
        doc: DoclingDocument = result.document

        title = _extract_document_title_from_doc(
            docling_doc=doc,
            pdf_path=pdf_path.as_posix(),
            toc_config=self.toc_config,
        )

        cache_path: Optional[pathlib.Path] = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{pdf_path.stem}.json"
            cache_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            self.logger.info(f"DoclingDocument cached to {cache_path}")

        markdown = post_process_markdown(doc.export_to_markdown())
        images: List[ParsedImage] = []   # legacy fitz-based ImageManager handles this
        tables = self._extract_tables(doc)

        return ParsedDocument(
            backend_name=self.name,
            n_pages=len(doc.pages),
            markdown=markdown,
            images=images,
            tables=tables,
            headings=[],  # legacy Path 2 reads from the cached JSON
            doc_title=title,
            cache_path=cache_path,
        )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _extract_tables(self, doc: DoclingDocument) -> List[ParsedTable]:
        out: List[ParsedTable] = []
        idx = 0
        for item, _level in doc.iterate_items():
            if not isinstance(item, TableItem):
                continue
            idx += 1
            page, bbox = self._table_page_and_bbox(item)
            html = self._safe_export(item, "export_to_html", doc)
            md = self._safe_export(item, "export_to_markdown", doc)
            n_rows, n_cols = self._table_shape(item)
            out.append(ParsedTable(
                id=f"TBL_{page:03d}_{idx:02d}",
                page=page,
                bbox=bbox,
                html=html,
                markdown=md,
                caption=self._table_caption(item),
                footnote=None,
                n_rows=n_rows,
                n_cols=n_cols,
            ))
        return out

    @staticmethod
    def _safe_export(item: TableItem, method: str, doc: DoclingDocument) -> Optional[str]:
        fn = getattr(item, method, None)
        if fn is None:
            return None
        try:
            # Some Docling versions need the parent doc, some don't
            try:
                return fn(doc=doc)
            except TypeError:
                return fn()
        except Exception:
            return None

    @staticmethod
    def _table_page_and_bbox(item: TableItem) -> tuple:
        """Best-effort page (1-based) + bbox extraction across Docling versions."""
        page = 1
        bbox = [0.0, 0.0, 0.0, 0.0]
        prov = getattr(item, "prov", None)
        if prov:
            p0 = prov[0]
            page = int(getattr(p0, "page_no", 1) or 1)
            b = getattr(p0, "bbox", None)
            if b is not None:
                bbox = [
                    float(getattr(b, "l", 0.0)),
                    float(getattr(b, "t", 0.0)),
                    float(getattr(b, "r", 0.0)),
                    float(getattr(b, "b", 0.0)),
                ]
        return page, bbox

    @staticmethod
    def _table_shape(item: TableItem):
        data = getattr(item, "data", None)
        if data is None:
            return None, None
        n_rows = getattr(data, "num_rows", None)
        n_cols = getattr(data, "num_cols", None)
        return n_rows, n_cols

    @staticmethod
    def _table_caption(item: TableItem) -> Optional[str]:
        captions = getattr(item, "captions", None) or []
        for c in captions:
            text = getattr(c, "text", None) or getattr(c, "value", None)
            if text:
                return str(text).strip()
        return None