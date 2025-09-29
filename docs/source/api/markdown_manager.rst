Markdown Manager
================

Markdown post‑processing and page anchor detection utilities.

This module defines :class:`~src.managers.markdown_manager.MarkdownManager`, a helper for normalizing Markdown
(text cleanup, Unicode normalization, newline compression, hyphenation fixes),
and for heuristically mapping PDF pages to offsets within a large Markdown text
via content anchors extracted with :pymupdf4llm:`PyMuPDF4LLM <index.html>`.

.. autoclass:: src.managers.markdown_manager.MarkdownManager
   :members:
   :member-order: bysource
   :show-inheritance: