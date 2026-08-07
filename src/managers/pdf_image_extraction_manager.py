"""
Optional PDF-native image extraction for preprocessing.

This module keeps image extraction independent from MinerU Markdown and
canonical chunk generation. Exported images are stored under:

    <image_root>/<doc_id>/

The existing ImageManager owns image detection/rendering; this module only
controls whether extraction runs and where its sidecar artifacts are stored.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from config.manager import ImageManagerConfig
from managers.image_manager import ImageManager


logger = logging.getLogger(__name__)


def extract_document_images_if_enabled(
    pdf_path: Path,
    image_root: Path,
    image_config: ImageManagerConfig,
) -> Optional[Dict[str, Any]]:
    """Load or extract a document image catalog when enabled.

    Parameters
    ----------
    pdf_path:
        Source PDF.
    image_root:
        Root directory reserved for image artifacts, normally
        ``work_root / "images"``.
    image_config:
        Image extraction configuration. When ``enabled`` is false this
        function is a no-op.

    Returns
    -------
    Optional[Dict[str, Any]]
        ``None`` when disabled, otherwise basic catalog metadata.
    """
    if not image_config.enabled:
        return None

    pdf_path = Path(pdf_path)
    image_root = Path(image_root)
    document_image_dir = image_root / pdf_path.stem

    image_manager = ImageManager(
        filepath=pdf_path,
        save_folder=document_image_dir,
        config=image_config,
    )
    catalog = image_manager.load_images_catalog(create_if_missing=True)
    entries = catalog.catalog or []

    logger.info(
        "Image extraction ready for %s | images=%d | dir=%s",
        pdf_path.stem,
        len(entries),
        document_image_dir,
    )

    return {
        "doc_id": pdf_path.stem,
        "image_dir": str(document_image_dir),
        "catalog_path": str(image_manager.get_catalog_path()),
        "n_images": len(entries),
    }
