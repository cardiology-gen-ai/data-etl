import json
import pathlib
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# from mineru.cli.common import do_parse, read_fn

from cardiology_gen_ai.utils.logger import get_logger

from managers.parsing.parsing_manager import ParsingBackend, ParsedDocument, ParsedImage, ParsedTable, ParsedHeading


class MinerUBackend(ParsingBackend):
    """Parsing backend that uses MinerU 3.x."""

    name = "mineru"

    def __init__(
        self,
        force_ocr: bool = False,
        language: str = "en",
        mineru_backend: str = "pipeline",
        formula_enable: bool = True,
        table_enable: bool = True,
        server_url: Optional[str] = None, # TODO: remove
        runtime: str = "local",  # "local" oppure "artifacts"
        artifacts_dir: Optional[pathlib.Path] = None,
    ):
        """
        Parameters
        ----------
        force_ocr : bool
            Force OCR instead of MinerU's auto-classification.
        language : str
            OCR language hint (``en``, ``ch``, ...).
        mineru_backend : str
            One of ``pipeline`` (CPU/GPU), ``vlm-transformers``,
            ``vlm-sglang-engine``, ``vlm-sglang-client``,
            ``hybrid-auto-engine``.  ``pipeline`` is CPU-friendly.
        formula_enable, table_enable : bool
            Toggle formula / table recognition (pipeline backend).
        server_url : str, optional
            Required when ``mineru_backend`` ends in ``-client``.
        """
        self.logger = get_logger("MinerUBackend")
        self.force_ocr = force_ocr
        self.language = language
        self.mineru_backend = mineru_backend
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.server_url = server_url
        self.logger = get_logger("MinerUBackend")
        self.force_ocr = force_ocr
        self.language = language
        self.mineru_backend = mineru_backend
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.server_url = server_url
        self.runtime = runtime
        self.artifacts_dir = pathlib.Path(artifacts_dir) if artifacts_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        pdf_path: pathlib.Path,
        output_dir: pathlib.Path,
        cache_dir: Optional[pathlib.Path] = None,
        images_dir: Optional[pathlib.Path] = None,
    ) -> ParsedDocument:
        stem = pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        if images_dir is None:
            images_dir = output_dir / f"{stem}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = cache_dir or output_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        md_text, content_list = self._invoke_mineru(pdf_path, stem, images_dir)

        cache_path = cache_dir / f"{stem}.mineru.json"
        cache_path.write_text(
            json.dumps(content_list, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        images = self._extract_images(content_list, images_dir)
        tables = self._extract_tables(content_list)
        headings = self._extract_headings(content_list)
        n_pages = self._infer_n_pages(content_list)
        doc_title = self._guess_title(content_list)

        self.logger.info(
            f"MinerU parsed {pdf_path.name}: {n_pages} pages, "
            f"{len(images)} images, {len(tables)} tables, {len(headings)} headings"
        )
        return ParsedDocument(
            backend_name=self.name,
            n_pages=n_pages,
            markdown=md_text,
            images=images,
            tables=tables,
            headings=headings,
            doc_title=doc_title,
            cache_path=cache_path,
        )


    def _invoke_mineru_local(
        self,
        pdf_path: pathlib.Path,
        stem: str,
        images_dir: pathlib.Path,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Call ``do_parse`` into a scratch dir and collect outputs."""
        # parse_method = "ocr" if self.force_ocr else "auto"

        # with tempfile.TemporaryDirectory(prefix="mineru_") as scratch:
        #     scratch_path = pathlib.Path(scratch)

        #     pdf_bytes = read_fn(pdf_path)

        #     do_parse(
        #         output_dir=str(scratch_path),
        #         pdf_file_names=[stem],
        #         pdf_bytes_list=[pdf_bytes],
        #         p_lang_list=[self.language],
        #         backend=self.mineru_backend,
        #         parse_method=parse_method,
        #         formula_enable=self.formula_enable,
        #         table_enable=self.table_enable,
        #         server_url=self.server_url,
        #         start_page_id=0,
        #         end_page_id=None,
        #         # output toggles: we only need .md + content_list.json
        #         f_draw_layout_bbox=False,
        #         f_draw_span_bbox=False,
        #         f_dump_md=True,
        #         f_dump_middle_json=False,
        #         f_dump_model_output=False,
        #         f_dump_orig_pdf=False,
        #         f_dump_content_list=True,
        #         f_make_md_mode="mm_markdown",
        #     )

        #     return self._collect_outputs(scratch_path, stem, images_dir)
        pass

    def _invoke_mineru(
            self,
            pdf_path: pathlib.Path,
            stem: str,
            images_dir: pathlib.Path,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if self.runtime == "artifacts":
            if self.artifacts_dir is None:
                raise ValueError(
                    "artifacts_root is required when MinerUBackend runtime='artifacts'"
                )

            root = pathlib.Path(self.artifacts_dir)

            if not root.is_absolute():
                data_root = pdf_path.parent.parent
                root = data_root / root

            artifacts_dir = root / stem

            if not artifacts_dir.is_dir():
                raise FileNotFoundError(
                    f"MinerU artifacts folder not found for {stem}: {artifacts_dir}"
                )

            self.logger.info(f"Reading MinerU artifacts from: {artifacts_dir}")

            return self._collect_outputs(
                scratch_path=artifacts_dir,
                stem=stem,
                images_dir=images_dir,
            )

        if self.runtime == "local":
            return self._invoke_mineru_local(pdf_path, stem, images_dir)

        raise ValueError(f"Unknown MinerU runtime: {self.runtime!r}")

    @staticmethod
    def _choose_markdown(root: pathlib.Path, stem: str) -> pathlib.Path:
        candidates = [
            root / "full.md",
            root / f"{stem}.md",
        ]

        for p in candidates:
            if p.is_file():
                return p

        raise RuntimeError(
            f"No MinerU markdown found in {root}. "
            f"Expected {root / 'full.md'} or {root / (stem + '.md')}"
        )

    @staticmethod
    def _choose_content_list(root: pathlib.Path, stem: str) -> pathlib.Path:
        candidates = [
            root / f"{stem}_content_list.json",
            root / "content_list.json",
        ]

        for p in candidates:
            if p.is_file():
                return p

        # fallback: allow UUID-style MinerU Desktop names, but only inside this doc folder
        matches = [
            p for p in root.glob("*_content_list.json")
            if "content_list_v2" not in p.name
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple MinerU content_list files found in {root}: "
                f"{[p.name for p in matches]}"
            )

        raise RuntimeError(
            f"No MinerU v1 content_list.json found in {root}"
        )

    def _collect_outputs(
        self,
        scratch_path: pathlib.Path,
        stem: str,
        images_dir: pathlib.Path,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Locate ``.md`` and ``_content_list.json`` produced by do_parse,
        copy images to ``images_dir`` and rewrite paths to be absolute.

        ``do_parse`` writes to ``scratch/<stem>/<backend>/<stem>.md`` (and
        siblings).  We glob to stay robust against minor layout shifts.
        """
        md_path = MinerUBackend._choose_markdown(scratch_path, stem)
        cl_path = MinerUBackend._choose_content_list(scratch_path, stem)

        self.logger.info(f"MinerU markdown: {md_path}")
        self.logger.info(f"MinerU content_list: {cl_path}")
        # copy images out of the scratch dir into our target images_dir
        produced_images_dir = md_path.parent / "images"
        if produced_images_dir.is_dir():
            for src in produced_images_dir.iterdir():
                if src.is_file():
                    shutil.copy2(src, images_dir / src.name)

        md_text = md_path.read_text(encoding="utf-8")
        content_list = json.loads(cl_path.read_text(encoding="utf-8"))

        # rewrite img_path entries to absolute paths in our images_dir
        for block in content_list:
            if block.get("type") in {"image", "table"}:
                img_rel = block.get("img_path")
                if img_rel:
                    block["img_path"] = str(images_dir / pathlib.Path(img_rel).name)

        # rewrite md "images/xxx.jpg" references too
        md_text = md_text.replace("images/", images_dir.as_posix() + "/")

        return md_text, content_list

    # ------------------------------------------------------------------
    # content_list -> typed objects
    # ------------------------------------------------------------------

    @staticmethod
    def _to_page_1based(page_idx: Optional[int]) -> int:
        return 1 if page_idx is None else int(page_idx) + 1

    @staticmethod
    def _join_strs(parts) -> Optional[str]:
        if not parts:
            return None
        out = " ".join(p for p in parts if p).strip()
        return out or None

    def _extract_images(
        self,
        content_list: List[Dict[str, Any]],
        images_dir: pathlib.Path,
    ) -> List[ParsedImage]:
        out: List[ParsedImage] = []
        for i, block in enumerate(content_list, start=1):
            if block.get("type") != "image":
                continue
            page = self._to_page_1based(block.get("page_idx"))
            bbox = list(block.get("bbox") or [0.0, 0.0, 0.0, 0.0])
            img_p = block.get("img_path")
            imagepath: Optional[pathlib.Path] = None
            if img_p:
                p = pathlib.Path(img_p)
                imagepath = p if p.is_absolute() else images_dir / p.name
            out.append(ParsedImage(
                id=f"FIG_{page:03d}_{i:04d}",
                page=page,
                bbox=[float(v) for v in bbox],
                imagepath=imagepath,
                caption=self._join_strs(
                    block.get("image_caption")
                    or block.get("img_caption")
                    or block.get("caption")
                ),
            ))
        return out

    def _extract_tables(self, content_list: List[Dict[str, Any]]) -> List[ParsedTable]:
        tables: List[ParsedTable] = []
        for idx, block in enumerate(content_list, start=1):
            if block.get("type") != "table":
                continue
            table_id = f"TAB_{idx:04d}"
            tables.append(
                ParsedTable(
                    id=table_id,
                    page=int(block.get("page_idx", 0)) + 1,
                    bbox=block.get("bbox"),
                    html=block.get("table_body"),
                    markdown=block.get("table_body"),
                    imagepath=(
                        pathlib.Path(block["img_path"])
                        if block.get("img_path")
                        else None
                    ),
                    caption=self._join_strs(
                        block.get("table_caption")
                        or block.get("caption")
                    ),
                    footnote=self._join_strs(
                        block.get("table_footnote")
                        or block.get("footnote")
                    ),
                )
            )
        return tables

    def _extract_headings(
        self,
        content_list: List[Dict[str, Any]],
    ) -> List[ParsedHeading]:
        out: List[ParsedHeading] = []
        for block in content_list:
            if block.get("type") != "text":
                continue
            level = block.get("text_level")
            if not level:
                continue
            title = (block.get("text") or "").strip()
            if not title:
                continue
            out.append(ParsedHeading(
                title=title,
                level=int(level),
                page=self._to_page_1based(block.get("page_idx")),
                bbox=list(block.get("bbox")) if block.get("bbox") else None,
            ))
        return out

    @staticmethod
    def _infer_n_pages(content_list: List[Dict[str, Any]]) -> int:
        max_idx = -1
        for b in content_list:
            pi = b.get("page_idx")
            if pi is not None and int(pi) > max_idx:
                max_idx = int(pi)
        return max_idx + 1 if max_idx >= 0 else 0

    @staticmethod
    def _guess_title(content_list: List[Dict[str, Any]]) -> Optional[str]:
        for b in content_list:
            if b.get("type") != "text":
                continue
            if b.get("text_level") == 1 and int(b.get("page_idx", 1)) == 0:
                t = (b.get("text") or "").strip()
                if t:
                    return t
        return None