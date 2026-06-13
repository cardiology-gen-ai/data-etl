import pathlib
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field


class ParsedImage(BaseModel):
    """A single figure extracted from the PDF by the backend."""
    id: str
    page: int  # 1-based
    bbox: List[float]  # [x0, y0, x1, y1]
    imagepath: Optional[pathlib.Path] = None
    caption: Optional[str] = None


class ParsedTable(BaseModel):
    id: str
    page: int
    bbox: Optional[List[float]] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    imagepath: Optional[pathlib.Path] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None


class ParsedHeading(BaseModel):
    """A heading detected by the backend with its hierarchical level."""
    title: str
    level: int = Field(ge=1)  # 1 = top-level
    page: int = Field(ge=1)   # 1-based PDF page where the heading appears
    bbox: Optional[List[float]] = None


class ParsedDocument(BaseModel):
    """Common output of every parsing backend."""
    backend_name: str
    n_pages: int
    markdown: str
    images: List[ParsedImage] = Field(default_factory=list)
    tables: List[ParsedTable] = Field(default_factory=list)
    headings: List[ParsedHeading] = Field(default_factory=list)
    doc_title: Optional[str] = None
    cache_path: Optional[pathlib.Path] = None

    class Config:
        arbitrary_types_allowed = True


class ParsingBackend(ABC):
    """Abstract base class for PDF parsing backends."""
    name: str = "abstract"

    @abstractmethod
    def parse(
        self,
        pdf_path: pathlib.Path,
        output_dir: pathlib.Path,
        cache_dir: Optional[pathlib.Path] = None,
        images_dir: Optional[pathlib.Path] = None,
    ) -> ParsedDocument:
        raise NotImplementedError