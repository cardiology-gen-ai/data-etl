from enum import Enum
from typing import Optional

from managers.parsing.parsing_manager import ParsingBackend
from managers.toc_extraction.toc_configs import BaseTocConfig


class ParsingBackendName(str, Enum):
    docling = "docling"
    mineru = "mineru"


class ParsingBackendFactory:
    """Build a parsing backend from its name."""

    @staticmethod
    def get(
        name: str,
        toc_config: BaseTocConfig,
        mineru_force_ocr: bool = False,
        mineru_language: str = "en",
        mineru_backend: str = "pipeline",
        mineru_formula_enable: bool = True,
        mineru_table_enable: bool = True,
        mineru_server_url: Optional[str] = None,
        mineru_runtime: str = "local",
        mineru_artifacts_root: Optional[str] = None,
    ) -> ParsingBackend:
        backend = ParsingBackendName(name)
        if backend == ParsingBackendName.docling:
            from managers.parsing.docling_backend import DoclingBackend
            return DoclingBackend(toc_config=toc_config)
        if backend == ParsingBackendName.mineru:
            from managers.parsing.mineru_backend import MinerUBackend
            return MinerUBackend(
                force_ocr=mineru_force_ocr,
                language=mineru_language,
                mineru_backend=mineru_backend,
                formula_enable=mineru_formula_enable,
                table_enable=mineru_table_enable,
                server_url=mineru_server_url,
                runtime=mineru_runtime,
                artifacts_dir=mineru_artifacts_root,
            )
        raise ValueError(f"Unknown parsing backend: {name!r}")