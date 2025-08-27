import pathlib
import re
import unicodedata
from typing import List, Dict

import fitz
import pymupdf4llm


class MarkdownManager:
    def __init__(self, filepath: pathlib.Path, text: str):
        self.filepath = filepath
        self.text = self.post_process_markdown(text)

    @staticmethod
    def post_process_markdown(text):
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
        for line in candidate_text.splitlines():
            line = line.strip()
            if len(line) >= min_chars:
                return line[:200]
        return candidate_text[:200].strip()

    @staticmethod
    def get_next_content_break(candidate_text: str, start_idx: int) -> int:
        """Find candidate content break after start_idx"""
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
        words = re.findall(r"\w+", snippet)
        if not words:
            return None
        words = words[:max_tokens]
        pattern = r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"
        return re.compile(pattern)

    def get_keywords_matches_in_slice(self, start: int, end: int, keywords: List[str]) -> List[int]:
        """Get row indexes starting with one of the required keywords"""
        kw = keywords if isinstance(keywords, str) else "|".join(keywords)
        rx = re.compile(rf"(?m)^[ \t]*$\r?\n^({kw}\b.*)$")
        matches = []
        for m in rx.finditer(self.text, pos=start, endpos=end):
            # above row starting with keyword
            line_start = self.text.rfind("\n", 0, m.start()) + 1
            matches.append(line_start)
        return matches

    def find_page_anchors_in_markdown(self) -> Dict[int, int]:
        """Build dict {page_index(1-based): offset_in_md} with approximate start for each page"""
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