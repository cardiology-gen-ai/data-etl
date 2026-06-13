import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import fitz
from pydantic import BaseModel

from cardiology_gen_ai.utils.logger import get_logger
from cardiology_gen_ai.utils.singleton import Singleton
from config.manager import PreprocessingConfig
from managers.image_manager import ImageManager, ImagesCatalog, ImagesCatalogEntry
from managers.parsing.factory import ParsingBackendFactory
from managers.parsing.parsing_manager import ParsedDocument
from managers.tables.table_manager import TableManager
from managers.toc_extraction.toc_configs import TocConfigFactory
from managers.toc_extraction.utils import _is_valid_title
from utils.headers import dump_headings
from utils.text_utils import post_process_markdown


class DocumentMetadata(BaseModel):
    file_title: Optional[str] = None
    filename: str
    filepath: str
    file_extension: Optional[str] = None
    md_filepath: str
    cache_filepath: Optional[str] = None
    headings_sidecar: Optional[str] = None
    tables_catalog: Optional[str] = None
    n_pages: int
    image_folder: str
    tables_folder: Optional[str] = None
    n_chunks: Optional[int] = None
    backend: str = "docling"


class MarkdownConverter(metaclass=Singleton):
    """Convert a PDF to Markdown with a pluggable parsing backend."""

    def __init__(self, config: PreprocessingConfig, app_id: str = "cardiology_protocols"):
        self.logger = get_logger("MarkdownConverter")
        self.config = config
        self.app_id = app_id
        self.toc_config = TocConfigFactory.get_toc_config(app_id)

        parsing_cfg = getattr(config, "parsing", None)
        backend_name = getattr(parsing_cfg, "backend", "docling")
        self.backend = ParsingBackendFactory.get(
            name=backend_name,
            toc_config=self.toc_config,
            mineru_force_ocr=getattr(parsing_cfg, "mineru_force_ocr", False),
            mineru_language=getattr(parsing_cfg, "mineru_language", "en"),
            mineru_backend=getattr(parsing_cfg, "mineru_backend", "pipeline"),
            mineru_formula_enable=getattr(parsing_cfg, "mineru_formula_enable", True),
            mineru_table_enable=getattr(parsing_cfg, "mineru_table_enable", True),
            mineru_server_url=getattr(parsing_cfg, "mineru_server_url", None),
            mineru_runtime=getattr(parsing_cfg, "mineru_runtime", "local"),
            mineru_artifacts_root=getattr(parsing_cfg, "mineru_artifacts_root", None),
        )
        self.logger.info(f"Parsing backend: {self.backend.name}")

        pathlib.Path(self.config.output_folder.folder).mkdir(parents=True, exist_ok=True)
        if self.config.cache_folder:
            pathlib.Path(self.config.cache_folder.folder).mkdir(parents=True, exist_ok=True)


    def __call__(self, filename: str, attribute_tables: bool = False) -> Tuple[bool, Optional[DocumentMetadata]]:
        self.filename = filename
        self.filepath = self.config.input_folder.folder / self.filename
        base_name = os.path.splitext(self.filename)[0]
        self.images_dir = (
            pathlib.Path(self.config.output_folder.folder) / f"{base_name}_images"
        )
        self.tables_dir = (
            pathlib.Path(self.config.output_folder.folder) / f"{base_name}_tables"
        )
        return self.process_single_file(base_name)

    def process_single_file(self, base_name: str):
        try:
            cache_dir = pathlib.Path(
                self.config.cache_folder.folder
                if self.config.cache_folder is not None
                else self.config.output_folder.folder
            )

            parsed: ParsedDocument = self.backend.parse(
                pdf_path=self.filepath,
                output_dir=pathlib.Path(self.config.output_folder.folder),
                cache_dir=cache_dir,
                images_dir=self.images_dir,
            )

            sidecar_path = dump_headings(cache_dir, base_name, parsed.headings)
            self.logger.info(
                f"Headings sidecar written to {sidecar_path} "
                f"({len(parsed.headings)} entries)"
            )

            tables_catalog_path: Optional[pathlib.Path] = None
            if parsed.tables:
                tm = TableManager(filepath=self.filepath, save_folder=self.tables_dir)
                tm.build_from_parsed(parsed.tables)
                tables_catalog_path = tm.get_catalog_path()
                self.logger.info(
                    f"Tables catalog written to {tables_catalog_path} "
                    f"({len(parsed.tables)} tables)"
                )

            file_title = parsed.doc_title
            if file_title and not _is_valid_title(file_title, self.toc_config):
                file_title = None

            md_text = post_process_markdown(parsed.markdown)
            updated_markdown_text = self.place_images_in_markdown(md_text, parsed)

            md_path = self.config.output_folder.folder / f"{base_name}.md"
            md_path.write_text(updated_markdown_text, encoding="utf-8")

            doc_metadata = DocumentMetadata(
                file_title=file_title,
                filename=self.filename,
                filepath=self.filepath.as_posix(),
                md_filepath=md_path.as_posix(),
                cache_filepath=str(parsed.cache_path) if parsed.cache_path else None,
                headings_sidecar=str(sidecar_path),
                tables_catalog=str(tables_catalog_path) if tables_catalog_path else None,
                n_pages=parsed.n_pages,
                image_folder=self.images_dir.as_posix(),
                tables_folder=self.tables_dir.as_posix() if parsed.tables else None,
                backend=parsed.backend_name,
            )
            self.logger.info(
                f"Successfully processed {self.filename} via {parsed.backend_name}"
            )
            return True, doc_metadata
        except Exception as e:
            self.logger.error(f"Error processing {self.filename}: {e}")
            return False, None

    def place_images_in_markdown(self, md_text: str, parsed: ParsedDocument) -> str:
        catalog = self._build_or_load_catalog(parsed)
        if not catalog.catalog:
            return md_text

        by_page: Dict[int, List[ImagesCatalogEntry]] = {}
        for image in catalog.catalog:
            by_page.setdefault(int(image.page), []).append(image)
        for page in by_page:
            by_page[page].sort(key=lambda x: float(x.bbox[1]))

        page_offsets = self._build_page_offsets(md_text=md_text, pages=sorted(by_page.keys()))
        insertions: List[Tuple[int, str]] = []
        md_len = len(md_text)

        for page_n in sorted(by_page.keys()):
            entries = by_page[page_n]
            start = page_offsets.get(page_n, md_len)
            next_pages = [p for p in sorted(by_page.keys()) if p > page_n]
            end = page_offsets.get(next_pages[0], md_len) if next_pages else md_len

            caption_spots = self._find_caption_spots(md_text=md_text, start=start, end=end)
            caption_used = [False] * len(caption_spots)

            for image in entries:
                alt = image.make_alt_text()
                line = f"\n\n![{alt}]({image.imagepath}) <!-- {image.id} -->\n"
                placed = False
                for i, spot in enumerate(caption_spots):
                    if not caption_used[i]:
                        caption_used[i] = True
                        insertions.append((spot, line))
                        placed = True
                        break
                if not placed:
                    offset = self.get_next_content_break(md_text, start)
                    insertions.append((offset, line))

        insertions.sort(key=lambda x: x[0], reverse=True)
        out = md_text
        for offset, line in insertions:
            out = out[:offset] + line + out[offset:]
        return out

    def _build_or_load_catalog(self, parsed: ParsedDocument) -> ImagesCatalog:
        if parsed.images:
            entries = [
                ImagesCatalogEntry(
                    id=img.id,
                    imagepath=img.imagepath or self.images_dir / f"{img.id}.png",
                    filepath=self.filepath,
                    page=img.page,
                    bbox=img.bbox,
                    caption=img.caption,
                )
                for img in parsed.images
            ]
            return ImagesCatalog(catalog=entries)

        image_manager = ImageManager(
            filepath=self.filepath,
            save_folder=self.images_dir,
            config=self.config.image_manager,
        )
        return image_manager.load_images_catalog(create_if_missing=True)

    # ------------------------------------------------------------------
    # Helpers (unchanged from legacy)
    # ------------------------------------------------------------------

    def _build_page_offsets(self, md_text: str, pages: List[int]) -> Dict[int, int]:
        offsets: Dict[int, int] = {}
        try:
            pdf = fitz.open(self.filepath.as_posix())
            search_from = 0
            for page_no in pages:
                if page_no < 1 or page_no > pdf.page_count:
                    offsets[page_no] = search_from
                    continue
                page: fitz.Page = pdf.load_page(page_no - 1)
                raw_text = page.get_text("text")
                anchor = ""
                for line in raw_text.splitlines():
                    line = line.strip()
                    if len(line) >= 40:
                        anchor = " ".join(line.split()[:6])
                        break
                if anchor:
                    pos = md_text.find(anchor, search_from)
                    if pos != -1:
                        offsets[page_no] = pos
                        search_from = pos
                        continue
                offsets[page_no] = search_from
            pdf.close()
        except (OSError, RuntimeError):
            md_len = len(md_text)
            for i, page_no in enumerate(pages):
                offsets[page_no] = int(i / max(len(pages), 1) * md_len)
        return offsets

    def _find_caption_spots(self, md_text: str, start: int, end: int) -> List[int]:
        keywords = list(self.config.image_manager.caption_keywords)
        kw_pattern = "|".join(re.escape(k) for k in keywords)
        rx = re.compile(rf"(?m)^[ \t]*\n^({kw_pattern}\b.*)$")
        spots = []
        for m in rx.finditer(md_text, pos=start, endpos=end):
            line_start = md_text.rfind("\n", 0, m.start()) + 1
            spots.append(line_start)
        return spots

    @staticmethod
    def get_next_content_break(candidate_text: str, start_idx: int) -> int:
        s = candidate_text[start_idx:]

        def first_outside(pattern: str, ret: str = "start"):
            for m in re.finditer(pattern, s):
                if (len(re.findall(r"(?m)^```", s[:m.start()])) % 2) == 0:
                    return start_idx + (m.start() if ret == "start" else m.end())
            return None

        c_before_header = first_outside(r"\n\s*\n(?=\s{0,3}#{1,6}\s)", "start")
        c_after_header = first_outside(r"\n\s{0,3}#{1,6}.*?(?:\n|$)'", "end")
        c_hrule = first_outside(r"\n(?:-{3,}|\*{3,}|_{3,})\s*(?:\n|$)", "end")
        c_blank = first_outside(r"\n\s*\n", "start")
        candidates = [c for c in (c_before_header, c_after_header, c_hrule, c_blank)
                      if c is not None]
        return min(candidates) if candidates else start_idx
