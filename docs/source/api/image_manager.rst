Image Manager
=============

Image extraction and cataloging utilities for PDF documents.

This module provides a lightweight :class:`~src.managers.image_manager.ImageManager` that scans a PDF with
:pymupdf:`PyMuPDF <index.html>` (a.k.a. ``fitz``), detects image blocks, merges adjacent regions to
recompose whole figures, exports each figure as a PNG at a chosen DPI, and
maintains a JSON catalog validated via :pydantic:`Pydantic BaseModel <base_model>`.

.. rubric:: Notes

- :class:`~src.config.manager.ImageManagerConfig` is expected to expose at least ``dpi`` (int), ``pad`` (float/int), and ``tol`` (float/int) attributes.
- Exported images are named as default as ``FIG_<page>_<idx>.png`` and a JSON catalog is written to ``save_folder / images_catalog.json`` by default.

.. automodule:: src.managers.image_manager
   :members:
   :exclude-members:
   :member-order: bysource
