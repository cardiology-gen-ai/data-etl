import pathlib
from pathlib import Path
from typing import List, Optional
import json
import re

import fitz
from pydantic import BaseModel, ValidationError

from src.config.manager import ImageManagerConfig


class ImagesCatalogEntry(BaseModel):
    id: str
    imagepath: Path
    filepath: Path
    page: int
    bbox: List[float]
    caption: Optional[str] = None

    def make_alt_text(self) -> str:
        """Create alt text to display in the markdown reference to the image"""
        alt = (self.caption or self.id).strip()
        alt = re.sub(r"\s+", " ", alt).strip()
        alt = re.sub(r"[\[\]\(\)]", "", alt)
        return alt or "image"


class ImagesCatalog(BaseModel):
    name: str = "images_catalog.json"
    catalog: List[ImagesCatalogEntry] | None = None


class ImageManager:
    def __init__(self, filepath: pathlib.Path, save_folder: pathlib.Path, config: ImageManagerConfig):
        self.filepath = filepath
        self.save_folder = save_folder
        self.config = config
        self.catalog = ImagesCatalog(catalog=[])

    @staticmethod
    def rect_union(rect_a: fitz.Rect, rect_b: fitz.Rect) -> fitz.Rect:
        return fitz.Rect(min(rect_a.x0, rect_b.x0), min(rect_a.y0, rect_b.y0), max(rect_a.x1, rect_b.x1), max(rect_a.y1, rect_b.y1))

    def rects_overlap_or_touch(self, rect_a: fitz.Rect, rect_b: fitz.Rect) -> bool:
        """Uses tol to determine whether two rectangles overlap or are adjacent (touch)"""
        return not (rect_a.x1 + self.config.tol < rect_b.x0 or rect_b.x1 + self.config.tol < rect_a.x0 or
                    rect_a.y1 + self.config.tol < rect_b.y0 or rect_b.y1 + self.config.tol < rect_a.y0)

    def merge_rects(self, rects: List[fitz.Rect]) -> List[fitz.Rect]:
        rects = rects[:]
        merged_rect = True
        while merged_rect and len(rects) > 1:
            merged_rect = False
            updated_rects = []
            used_rect = [False] * len(rects)
            for i in range(len(rects)):
                if used_rect[i]:
                    continue
                current_rect = rects[i]
                used_rect[i] = True
                change = True
                while change:
                    change = False
                    for j in range(len(rects)):
                        if used_rect[j]:
                            continue
                        if self.rects_overlap_or_touch(current_rect, rects[j]):
                            current_rect = self.rect_union(current_rect, rects[j])
                            used_rect[j] = True
                            change = True
                            merged_rect = True
                updated_rects.append(current_rect)
            rects = updated_rects
        return rects

    def expand_rect(self, rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
        """Expand rect without exceeding page limits, in all directions"""
        expanded_rect = fitz.Rect(rect.x0 - self.config.pad, rect.y0 - self.config.pad,
                                  rect.x1 + self.config.pad, rect.y1 + self.config.pad)
        return expanded_rect & page_rect

    def get_catalog_path(self):
        return self.save_folder / self.catalog.name

    def extract_visual_images(self):
        self.save_folder.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(self.filepath)
        zoom = self.config.dpi / 72.0  # TODO: check if 72 is specific for pdf
        matrix = fitz.Matrix(zoom, zoom)
        for page_number in range(doc.page_count):
            current_page = doc[page_number]
            raw_page_dict = current_page.get_text("rawdict")
            rects = []
            for b in raw_page_dict.get("blocks", []):
                if b.get("type") == 1 and "bbox" in b:  # raw image (type 1) blocks
                    x0, y0, x1, y1 = b["bbox"]
                    rects.append(fitz.Rect(x0, y0, x1, y1))
            fused_rects = self.merge_rects(rects)  # fuse rectangles to recompose entire figures
            fused_rects.sort(key=lambda rect: (rect.y0, rect.x0))
            for idx, rect_idx in enumerate(fused_rects, start=1):
                padded_rect = self.expand_rect(rect_idx, current_page.rect) if self.config.pad > 0 else rect_idx
                pix = current_page.get_pixmap(matrix=matrix, clip=padded_rect, alpha=False)

                fig_id = f"FIG_{page_number + 1:03d}_{idx:02d}"
                fig_filename = f"{fig_id}.png"
                fig_filepath = (self.save_folder / fig_filename)
                pix.save(fig_filepath.as_posix())

                catalog_entry = ImagesCatalogEntry(
                    id = fig_id,
                    imagepath = fig_filepath,
                    filepath = self.filepath,
                    page = page_number + 1,
                    bbox = [float(padded_rect.x0), float(padded_rect.y0), float(padded_rect.x1), float(padded_rect.y1)],
                )
                current_catalog = self.catalog
                self.catalog = current_catalog.model_copy(update={"catalog": current_catalog.catalog + [catalog_entry]})

        doc.close()
        catalogue_dict = self.catalog.model_dump(mode="json", exclude_none=True)
        with open(self.get_catalog_path(), "w", encoding="utf-8") as f:
            json.dump(catalogue_dict.get("catalog"), f, ensure_ascii=False, indent=2)
        return self.catalog

    def load_images_catalog(self, create_if_missing: bool = True):
        catalog_path = self.get_catalog_path()
        if not catalog_path.exists():
            if create_if_missing:
                return self.extract_visual_images()
            raise FileNotFoundError(f"Catalog not found at {catalog_path}")
        catalog_data = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(catalog_data, list):
            catalog_data = {"catalog": catalog_data}
        try:
            return ImagesCatalog.model_validate(catalog_data)
        except ValidationError as e:
            raise ValueError(f"Catalog validation error: {e}") from e
