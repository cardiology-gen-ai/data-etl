import pathlib
import re
import unicodedata
from typing import List, Dict

import fitz
import pymupdf4llm


class MarkdownManager:
    """Normalize Markdown and locate page anchors w.r.t. a source PDF.

    Parameters
    ----------
    filepath : :class:`pathlib.Path`
        Path to the source PDF.
    text : str
        Raw Markdown input; it is immediately normalized via
        :meth:`~src.managers.markdown_manager.MarkdownManager.post_process_markdown` and stored in :py:attr:`~src.managers.markdown_manager.MarkdownManager.text`.
    """

    text: str #: :class:`pathlib.Path` : Path to the source PDF.
    filepath: pathlib.Path #: str : Normalized Markdown text.

    def __init__(self, filepath: pathlib.Path, text: str):
        self.filepath = filepath
        self.text = self.post_process_markdown(text)

    @staticmethod
    def post_process_markdown(text) -> str:
        """
        Clean and normalize Markdown text.

        .. rubric:: Steps:

        - Remove lines containing bracketed ellipses like ``[....]``.
        - Normalize Unicode using ``NFKC``.
        - Normalize line endings to ``\\n``.
        - Compress consecutive spaces/tabs to a single space.
        - Limit consecutive blank lines to at most two.
        - Fix hyphenation: remove soft hyphen (U+00AD) and join words split by
          hyphen variants at end of line.

        Parameters
        ----------
        text : str
            Raw Markdown text.

        Returns
        -------
        str
            Normalized Markdown string.
        """
        # remove [....]
        markdown_text = "\n".join([line for line in text.split("\n") if not re.search(r"\[\.+\]", line)])
        # normalize unicode
        markdown_text = unicodedata.normalize("NFKC", markdown_text)
        # normalize end-of-line
        markdown_text = markdown_text.replace("\r\n", "\n")
        # compress whitespaces
        markdown_text = re.sub(r"[ \t]+", " ", markdown_text)
        # limit consecutive newlines
        max_new_lines = 2
        markdown_text = re.sub(r"\n{%d,}" % (max_new_lines + 1), "\n" * max_new_lines, markdown_text)
        # fix hyphenation
        markdown_text = markdown_text.replace("\u00AD", "")  # remove soft hyphen
        markdown_text = re.sub(r"(\w)[-\u2010\u2011\u2212]\n(\w)", r"\1\2",
                               markdown_text)  # unify segmented words at the enf line
        return markdown_text.strip()

    @staticmethod
    def get_first_long_line(candidate_text: str, min_chars: int = 50) -> str:
        """Return the first non-empty line with at least ``min_chars`` characters.

        If no such line exists, return the first ``200`` characters of the
        input (trimmed).

        Parameters
        ----------
        candidate_text : str
            Text to scan.
        min_chars : int
            Minimum line length to qualify as "long".

        Returns
        -------
        str
            A representative snippet (truncated to 200 chars).
        """
        for line in candidate_text.splitlines():
            line = line.strip()
            if len(line) >= min_chars:
                return line[:200]
        return candidate_text[:200].strip()

    @staticmethod
    def get_next_content_break(candidate_text: str, start_idx: int) -> int:
        """Heuristically find the next content break.

        The search considers several candidate boundaries after ``start_idx`` and
        returns the earliest among them:

        - before header: a blank line followed by a Markdown header (``#``..``######``),
        - after header: the end of a header line,
        - horizontal rule: a line of ``---``, ``***`` or ``___``,
        - blank paragraph: a blank line.

        Matches that fall inside fenced code blocks (````` … ````` ) are ignored by
        counting the number of opening fences before each match and requiring it to be even.

        Parameters
        ----------
        candidate_text : str
            Text to search.
        start_idx : int
            Starting offset (0-based).

        Returns
        -------
        int
            Offset of the next break or ``start_idx`` if none is found.
        """
        s = candidate_text[start_idx:]

        def first_outside(pattern: str, ret: str = "start"):
            for m in re.finditer(pattern, s):
                if (len(re.findall(r"(?m)^```", s[:m.start()])) % 2) == 0:
                    return start_idx + (m.start() if ret == "start" else m.end())
            return None

        candidate_before_header = first_outside(r"\n\s*\n(?=\s{0,3}#{1,6}\s)", "start")  # before header
        candidate_after_header = first_outside(r"\n\s{0,3}#{1,6}.*?(?:\n|$)'", "end")  # after header
        candidate_hrule = first_outside(r"\n(?:-{3,}|\*{3,}|_{3,})\s*(?:\n|$)", "end")  # after hrule
        candidate_blank = first_outside(r"\n\s*\n", "start")  # blank line
        candidates = [c for c in (candidate_before_header, candidate_after_header, candidate_hrule, candidate_blank)
                      if c is not None]
        return min(candidates) if candidates else start_idx

    @staticmethod
    def build_anchor_regex(snippet: str, max_tokens: int = 10) -> re.Pattern | None:
        """Build a loose regex to locate a textual snippet in the full Markdown.

        The pattern is composed by the first ``max_tokens`` alphanumeric tokens of
        ``snippet`` joined by ``\W+``. This allows matching across whitespace and
        punctuation variations.

        Parameters
        ----------
        snippet : str
            Text snippet extracted from a single page.
        max_tokens : int
            Maximum tokens from the snippet to include in the pattern.

        Returns
        -------
        re.Pattern | None
            Compiled regex pattern or ``None`` if no alphanumeric tokens are found.
        """
        words = re.findall(r"\w+", snippet)
        if not words:
            return None
        words = words[:max_tokens]
        pattern = r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"
        return re.compile(pattern)

    def get_keywords_matches_in_slice(self, start: int, end: int, keywords: List[str]) -> List[int]:
        """Return line start offsets for lines beginning with any keywords.

        The search is limited to the slice ``self.text[start:end]`` and uses
        multiline mode. The returned offsets refer to the absolute positions
        in :py:attr:`~src.managers.markdown_manager.MarkdownManager.text`.

        Parameters
        ----------
        start : int
            Start offset (inclusive).
        end : int
            End offset (exclusive).
        keywords : List[str]
            List of keywords (plain strings) to match at line start.

        Returns
        -------
        List[int]
            List of character offsets corresponding to the starts of matching lines.
        """
        kw = keywords if isinstance(keywords, str) else "|".join(keywords)
        rx = re.compile(rf"(?m)^[ \t]*$\r?\n^({kw}\b.*)$")
        matches = []
        for m in rx.finditer(self.text, pos=start, endpos=end):
            # above row starting with keyword
            line_start = self.text.rfind("\n", 0, m.start()) + 1
            matches.append(line_start)
        return matches

    def find_page_anchors_in_markdown(self) -> Dict[int, int]:
        """Map each PDF page to an approximate offset in the Markdown text.

        For each page in the source PDF, the method renders Markdown with
        :pymupdf4llm:`PyMuPDF4LLLM <index.html>`, normalizes it with :meth:`post_process_markdown`,
        extracts a representative snippet via :meth:`get_first_long_line`,
        builds a fuzzy anchor pattern with :meth:`build_anchor_regex`, and searches
        the main Markdown text from the previous anchor onward. If no match is
        found, the previous position is reused.

        The result dictionary is made monotonic non-decreasing across pages
        to ensure consistent ordering.

        Returns
        -------
        Dict[int, int]
            Dictionary mapping ``{page_index (1-based): offset_in_markdown}``.

        Raises
        ------
        RuntimeError
            Propagated from PyMuPDF/PyMuPDF4LLLM if the PDF cannot be opened or processed.
        OSError
            If reading the PDF fails due to I/O issues.
        """
        doc = fitz.open(self.filepath.as_posix())
        anchors = {}
        search_from = 0
        for page_n in range(doc.page_count):
            page_text = pymupdf4llm.to_markdown(
                doc,
                pages=[page_n],
                write_images=False,
            )
            page_text = self.post_process_markdown(page_text)
            page_snippet = self.get_first_long_line(page_text)
            rx = self.build_anchor_regex(page_snippet)
            idx = -1
            if rx:
                m = rx.search(self.text, pos=search_from)
                if m:
                    idx = m.start()
                    search_from = idx
            if idx == -1:
                # fallback: keep previous position
                idx = search_from
            anchors[page_n + 1] = idx - 1
        doc.close()
        # get strict monotony in page anchor numbers
        prev = 0
        for page_n in sorted(anchors):
            if anchors.get(page_n) < prev:
                anchors[page_n] = prev
            prev = anchors[page_n]
        return anchors