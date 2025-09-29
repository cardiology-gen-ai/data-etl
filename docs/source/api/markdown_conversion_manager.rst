Markdown Conversion Manager
===========================

File → Markdown conversion with image placement and cataloging.

This module defines a :class:`~src.managers.markdown_conversion_manager.MarkdownConverter` that converts a file
(supported extensions are pdf, xps, epub, mobi, fb2, cbz, svg, txt) to Markdown
via :pymupdf4llm:`PyMuPDF4LLM <index.html>`, extracts and saves figures through :class:`~src.managers.image_manager.ImageManager`, and
inlines the exported images into the Markdown near their captions or at sensible
content breaks. It also returns a validated :class:`~src.managers.markdown_conversion_manager.DocumentMetadata` record.


.. rubric:: Notes

- :class:`~src.config.manager.PreprocessingConfig` is expected to expose:

* ``input_folder.folder`` (pathlib.Path), where PDFs are read from,
* ``output_folder.folder`` (pathlib.Path), where Markdown & images are written,
* ``image_manager`` (:class:`~src.config.manager.ImageManagerConfig`), parameters for the image stage,
* ``image_manager.caption_keywords``, iterable of keywords used to detect caption lines.

- Exported images are inserted as ``![alt](path) <!-- FIG_* -->`` comments to keep a stable anchor for subsequent processing.

.. automodule:: src.managers.markdown_conversion_manager
   :members:
   :exclude-members:
   :member-order: bysource

See also
--------

- :mod:`src.config.manager`
- :mod:`src.managers.image_manager.ImageManager`