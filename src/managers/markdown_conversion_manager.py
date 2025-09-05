"""
PDF to Markdown Converter with Image Extraction
Converts PDFs to markdown and extracts images,
then references the images in the markdown file.
"""

import os
import pathlib
from typing import Tuple, Optional

import fitz
import pymupdf4llm
from pydantic import BaseModel

from src.managers.image_manager import ImageManager
from src.config.manager import PreprocessingConfig
from src.managers.markdown_manager import MarkdownManager
from cardiology_gen_ai.utils.singleton import Singleton
from cardiology_gen_ai.utils.logger import get_logger


# TODO: at the moment saving and loading functions assume local environment


class DocumentMetadata(BaseModel):
    filename: str
    filepath: str
    file_extension: Optional[str] = None
    md_filepath: str
    n_pages: int
    image_folder: str
    n_chunks: Optional[int] = None


class MarkdownConverter(metaclass=Singleton):
    def __init__(self, config: PreprocessingConfig):
        self.logger = get_logger("Markdown converter based on PyMuPDF")
        self.config = config
        pathlib.Path(self.config.output_folder.folder).mkdir(parents=True, exist_ok=True)

    def __call__(self, filename: str):
        self.filename = filename
        self.filepath = self.config.input_folder.folder / self.filename
        images_folder_name = f"{os.path.splitext(self.filename)[0]}_images"
        self.images_dir = pathlib.Path(self.config.output_folder.folder) / images_folder_name
        return self.process_single_file()

    def process_single_file(self) -> Tuple[bool, DocumentMetadata | None]:
        """Processa single PDF: convert to markdown and extract images."""
        base_name = os.path.splitext(self.filename)[0]
        self.logger.info(f"Processing: {self.filename}...")
        try:
            document = fitz.open(self.filepath.as_posix())
            md_text = pymupdf4llm.to_markdown(
                document,
                write_images=False,
                image_path=self.images_dir,
                image_format="png"
            )
            self.logger.info("Markdown conversion done.")
            updated_markdown_text = self.place_images_in_markdown(md_text)
            self.logger.info("Images extracted and saved.")
            # save markdown file
            md_filename = f"{base_name}.md"
            md_path = self.config.output_folder.folder / md_filename
            md_path.write_text(updated_markdown_text, encoding="utf-8")
            doc_metadata = DocumentMetadata(
                filename = self.filename,
                filepath = self.filepath.as_posix(),
                md_filepath = md_path.as_posix(),
                n_pages = document.page_count,
                image_folder = self.images_dir.as_posix()
            )
            self.logger.info(f"Successfully parsed {self.filename}.")
            return True, doc_metadata

        except Exception as e:
            self.logger.info(f"Error processing {self.filename}: {str(e)}")
            return False, None

    def place_images_in_markdown(self, md_text: str):
        """
        Insert '![alt](path) <!-- FIG_* -->' above the caption (if detected),
        else at the first content break in the right page.
        """
        image_manager = ImageManager(
            filepath=self.filepath,
            save_folder=self.images_dir,
            config=self.config.image_manager
        )
        markdown_manager = MarkdownManager(
            filepath=self.filepath,
            text=md_text
        )
        page_anchor = markdown_manager.find_page_anchors_in_markdown()
        # group figures by page, and sort according to y0
        catalog_entries = image_manager.load_images_catalog()
        by_page = {}
        for image in catalog_entries.catalog:
            by_page.setdefault(int(image.page), []).append(image)
        for page in by_page:
            by_page[page].sort(key=lambda x: float(x.bbox[1]))
        insertions = []
        md_len = len(markdown_manager.text)
        pages = sorted(page_anchor)
        bounds = {}
        for page_n, page in enumerate(pages):
            start = page_anchor[page]
            end = page_anchor[pages[page_n+1]] if page_n + 1 < len(pages) else md_len
            bounds[page] = (start, end)
        for page_n, page in enumerate(pages):
            page_images = by_page.get(page, [])
            if not page_images:
                continue
            start, end = bounds[page]
            caption_spots = markdown_manager.get_keywords_matches_in_slice(
                start, end, list(image_manager.config.caption_keywords))
            caption_used = [False]*len(caption_spots)
            for image in page_images:
                # caption, if found, has the precedence
                placed = False
                for i, pos in enumerate(caption_spots):
                    if not caption_used[i]:
                        caption_used[i] = True
                        idx = pos
                        alt = image.make_alt_text()
                        line = f'\n\n![{alt}]({image.imagepath}) <!-- {image.id} -->\n'
                        insertions.append((idx, line))
                        placed = True
                        break
                if placed:
                    continue
                # fallback: content break in the page
                idx = markdown_manager.get_next_content_break(markdown_manager.text, start)
                alt = image.make_alt_text()
                line = f'\n\n![{alt}]({image.imagepath}) <!-- {image.id} -->\n'
                insertions.append((idx, line))
                start = idx + len(line)
        insertions.sort(key=lambda t: t[0], reverse=True)
        out = markdown_manager.text
        for idx, txt in insertions:
            out = out[:idx] + txt + out[idx:]
        return out
