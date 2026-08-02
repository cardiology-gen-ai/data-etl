import json
import logging
import pathlib
import re
import unicodedata
from typing import Any, Dict, List

import fitz
import pymupdf4llm


logger = logging.getLogger(__name__)


class MarkdownManager:
    """Normalize Markdown and locate page anchors w.r.t. a source PDF.

    Parameters
    ----------
    filepath : :class:`pathlib.Path`
        Path to the source PDF.
    text : str
        Raw Markdown input; it is immediately normalized via
        :meth:`post_process_markdown` and stored in :attr:`text`.
    """

    text: str
    filepath: pathlib.Path

    def __init__(self, filepath: pathlib.Path, text: str):
        self.filepath = filepath
        self.text = self.post_process_markdown(text)

    @staticmethod
    def post_process_markdown(text: str) -> str:
        """Clean and normalize Markdown text."""
        markdown_text = "\n".join(
            line
            for line in text.split("\n")
            if not re.search(r"\[\.+\]", line)
        )

        markdown_text = unicodedata.normalize(
            "NFKC",
            markdown_text,
        )

        markdown_text = markdown_text.replace(
            "\r\n",
            "\n",
        )

        markdown_text = re.sub(
            r"[ \t]+",
            " ",
            markdown_text,
        )

        max_new_lines = 2

        markdown_text = re.sub(
            r"\n{%d,}" % (max_new_lines + 1),
            "\n" * max_new_lines,
            markdown_text,
        )

        markdown_text = markdown_text.replace(
            "\u00AD",
            "",
        )

        markdown_text = re.sub(
            r"(\w)[-\u2010\u2011\u2212]\n(\w)",
            r"\1\2",
            markdown_text,
        )

        return markdown_text.strip()

    @staticmethod
    def get_first_long_line(
        candidate_text: str,
        min_chars: int = 50,
    ) -> str:
        """Return the first non-empty line with at least ``min_chars`` characters."""
        for line in candidate_text.splitlines():
            line = line.strip()

            if len(line) >= min_chars:
                return line[:200]

        return candidate_text[:200].strip()

    @staticmethod
    def get_next_content_break(
        candidate_text: str,
        start_idx: int,
    ) -> int:
        """Heuristically find the next content break."""
        s = candidate_text[start_idx:]

        def first_outside(
            pattern: str,
            ret: str = "start",
        ) -> int | None:
            for match in re.finditer(pattern, s):
                fences_before = len(
                    re.findall(
                        r"(?m)^```",
                        s[: match.start()],
                    )
                )

                if fences_before % 2 == 0:
                    relative = (
                        match.start()
                        if ret == "start"
                        else match.end()
                    )

                    return start_idx + relative

            return None

        candidate_before_header = first_outside(
            r"\n\s*\n(?=\s{0,3}#{1,6}\s)",
            "start",
        )

        candidate_after_header = first_outside(
            r"\n\s{0,3}#{1,6}.*?(?:\n|$)",
            "end",
        )

        candidate_hrule = first_outside(
            r"\n(?:-{3,}|\*{3,}|_{3,})\s*(?:\n|$)",
            "end",
        )

        candidate_blank = first_outside(
            r"\n\s*\n",
            "start",
        )

        candidates = [
            candidate
            for candidate in (
                candidate_before_header,
                candidate_after_header,
                candidate_hrule,
                candidate_blank,
            )
            if candidate is not None
        ]

        return min(candidates) if candidates else start_idx

    @staticmethod
    def build_anchor_regex(
        snippet: str,
        max_tokens: int = 10,
    ) -> re.Pattern | None:
        """Build a loose regex to locate a textual snippet in the full Markdown."""
        words = re.findall(
            r"\w+",
            snippet,
        )

        if not words:
            return None

        words = words[:max_tokens]

        pattern = (
            r"\b"
            + r"\W+".join(
                re.escape(word)
                for word in words
            )
            + r"\b"
        )

        return re.compile(pattern)

    def get_keywords_matches_in_slice(
        self,
        start: int,
        end: int,
        keywords: List[str],
    ) -> List[int]:
        """Return line start offsets for lines beginning with any keyword."""
        keyword_pattern = (
            keywords
            if isinstance(keywords, str)
            else "|".join(keywords)
        )

        regex = re.compile(
            rf"(?m)^[ \t]*$\r?\n^({keyword_pattern}\b.*)$"
        )

        matches: List[int] = []

        for match in regex.finditer(
            self.text,
            pos=start,
            endpos=end,
        ):
            line_start = self.text.rfind(
                "\n",
                0,
                match.start(),
            ) + 1

            matches.append(line_start)

        return matches

    @staticmethod
    def _find_anchor_plateaus(
        anchors: Dict[int, int],
        min_pages: int = 3,
    ) -> List[tuple[int, int, int]]:
        """Return runs of pages sharing the same anchor offset."""
        items = sorted(
            anchors.items()
        )

        if not items:
            return []

        runs: List[
            tuple[int, int, int]
        ] = []

        start_page, current_offset = items[0]
        end_page = start_page

        for page, offset in items[1:]:
            if offset == current_offset:
                end_page = page
                continue

            if (
                end_page
                - start_page
                + 1
                >= min_pages
            ):
                runs.append(
                    (
                        start_page,
                        end_page,
                        current_offset,
                    )
                )

            start_page = page
            end_page = page
            current_offset = offset

        if (
            end_page
            - start_page
            + 1
            >= min_pages
        ):
            runs.append(
                (
                    start_page,
                    end_page,
                    current_offset,
                )
            )

        return runs

    def _validate_cached_anchors(
        self,
        anchors: Dict[int, int],
    ) -> tuple[bool, List[str]]:
        """Validate cached page anchors before reusing them."""
        reasons: List[str] = []

        try:
            with fitz.open(
                self.filepath.as_posix()
            ) as document:
                page_count = document.page_count

        except (RuntimeError, OSError) as exc:
            logger.warning(
                "Could not validate anchor cache for %s: %s",
                self.filepath,
                exc,
            )

            return False, [
                "could not read source PDF"
            ]

        expected_pages = list(
            range(
                1,
                page_count + 1,
            )
        )

        actual_pages = sorted(
            anchors
        )

        if actual_pages != expected_pages:
            reasons.append(
                "cache does not contain exactly "
                "one anchor per PDF page"
            )

        values = [
            anchors[page]
            for page in actual_pages
        ]

        if any(
            offset < 0
            or offset > len(self.text)
            for offset in values
        ):
            reasons.append(
                "one or more offsets are outside "
                "Markdown bounds"
            )

        if any(
            next_offset < current_offset
            for current_offset, next_offset
            in zip(
                values,
                values[1:],
            )
        ):
            reasons.append(
                "anchor offsets are not monotonic"
            )

        plateaus = self._find_anchor_plateaus(
            anchors,
            min_pages=3,
        )

        if plateaus:
            formatted = ", ".join(
                f"{start}-{end}@{offset}"
                for start, end, offset
                in plateaus
            )

            reasons.append(
                "suspicious plateau across at least "
                f"3 pages: {formatted}"
            )

        return not reasons, reasons

    def find_page_anchors_in_markdown(
        self,
    ) -> Dict[int, int]:
        """Map each physical PDF page to a Markdown offset.

        Each page is searched after the previous real textual match.
        This prevents consecutive pages from repeatedly matching the
        same occurrence of a repeated header or table line.

        If no match is found, the most recent usable anchor is kept.
        """
        anchors: Dict[int, int] = {}

        previous_match = -1
        previous_anchor = 0

        document = fitz.open(
            self.filepath.as_posix()
        )

        try:
            for page_index in range(
                document.page_count
            ):
                page_text = (
                    pymupdf4llm.to_markdown(
                        document,
                        pages=[page_index],
                        write_images=False,
                    )
                )

                page_text = (
                    self.post_process_markdown(
                        page_text
                    )
                )

                page_snippet = (
                    self.get_first_long_line(
                        page_text
                    )
                )

                regex = (
                    self.build_anchor_regex(
                        page_snippet
                    )
                )

                anchor = previous_anchor

                if regex is not None:
                    search_start = max(
                        0,
                        previous_match + 1,
                    )

                    match = regex.search(
                        self.text,
                        pos=search_start,
                    )

                    if match is not None:
                        anchor = match.start()
                        previous_match = anchor
                        previous_anchor = anchor

                anchors[
                    page_index + 1
                ] = anchor

        finally:
            document.close()

        previous_offset = 0

        for page_number in sorted(
            anchors
        ):
            if (
                anchors[page_number]
                < previous_offset
            ):
                anchors[
                    page_number
                ] = previous_offset

            previous_offset = (
                anchors[page_number]
            )

        plateaus = self._find_anchor_plateaus(
            anchors,
            min_pages=3,
        )

        if plateaus:
            logger.warning(
                "Generated page anchors for %s "
                "contain suspicious plateaus: %s",
                self.filepath.name,
                ", ".join(
                    f"{start}-{end}@{offset}"
                    for start, end, offset
                    in plateaus
                ),
            )

        return anchors

    def get_page_anchors(
        self,
        cache_path: pathlib.Path | None = None,
        cache_metadata: Dict[str, Any] | None = None,
    ) -> Dict[int, int]:
        """Load valid cached anchors or compute a fresh sequence."""
        if (
            cache_path
            and cache_path.exists()
        ):
            try:
                data = json.loads(
                    cache_path.read_text(
                        encoding="utf-8"
                    )
                )

                anchor_dict = data.get(
                    "anchors",
                    data,
                )

                if cache_metadata is not None:
                    metadata_valid = (
                        isinstance(data, dict)
                        and "anchors" in data
                        and all(
                            data.get(key) == value
                            for key, value in cache_metadata.items()
                        )
                    )
                    if not metadata_valid:
                        logger.warning(
                            "Ignoring stale page-anchor cache %s: "
                            "metadata does not match current Markdown source",
                            cache_path,
                        )
                        raise ValueError("stale page-anchor metadata")

                cached_anchors = {
                    int(key): int(value)
                    for key, value
                    in anchor_dict.items()
                }

                valid, reasons = (
                    self._validate_cached_anchors(
                        cached_anchors
                    )
                )

                if valid:
                    return cached_anchors

                logger.warning(
                    "Ignoring invalid page-anchor "
                    "cache %s: %s",
                    cache_path,
                    "; ".join(reasons),
                )

            except (
                json.JSONDecodeError,
                OSError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Could not load page-anchor "
                    "cache %s: %s",
                    cache_path,
                    exc,
                )

        anchors = (
            self.find_page_anchors_in_markdown()
        )

        if cache_path:
            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cache_path.write_text(
                json.dumps(
                    {
                        "doc_id": self.filepath.stem,
                        **(cache_metadata or {}),
                        "anchors": anchors,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        return anchors
