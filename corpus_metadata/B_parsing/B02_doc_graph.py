# corpus_metadata/corpus_metadata/B_parsing/B02_doc_graph.py
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# [OK] IMPORTANT: use local package-style imports (no "corpus_metadata....")
from A_core.A01_domain_models import BoundingBox


class ContentRole(str, Enum):
    BODY_TEXT = "BODY_TEXT"
    SECTION_HEADER = "SECTION_HEADER"
    PAGE_HEADER = "PAGE_HEADER"
    PAGE_FOOTER = "PAGE_FOOTER"
    TABLE_CAPTION = "TABLE_CAPTION"
    TABLE_CELL = "TABLE_CELL"


class TableType(str, Enum):
    UNKNOWN = "UNKNOWN"
    DATA_GRID = "DATA_GRID"
    GLOSSARY = "GLOSSARY"  # critical for abbreviation pipeline
    LAYOUT_GRID = "LAYOUT_GRID"


class ImageType(str, Enum):
    """Type of image/figure in document."""
    UNKNOWN = "UNKNOWN"
    FIGURE = "FIGURE"  # General figure
    FLOWCHART = "FLOWCHART"  # Patient flow, CONSORT diagrams
    CHART = "CHART"  # Bar/line/pie charts
    DIAGRAM = "DIAGRAM"  # Schematic diagrams
    PHOTO = "PHOTO"  # Photographs
    LOGO = "LOGO"  # Company/journal logos


class TextBlock(BaseModel):
    """
    Atomic unit of text.
    'role' lets generators skip headers/footers.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str

    page_num: int = Field(..., ge=1)  # 1-based
    reading_order_index: int = Field(..., ge=0)  # 0-based per page
    role: ContentRole = ContentRole.BODY_TEXT

    bbox: BoundingBox

    model_config = ConfigDict(frozen=True, extra="forbid")


class ImageBlock(BaseModel):
    """
    Extracted image/figure from document.

    Contains:
    - base64 encoded image data
    - OCR'd text from the image (if available)
    - Caption from nearby FigureCaption elements
    - Bounding box for highlighting
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    page_num: int = Field(..., ge=1)
    reading_order_index: int = Field(..., ge=0)

    # Image data
    image_base64: Optional[str] = None  # Base64-encoded image
    image_format: str = "png"  # Format hint (png, jpg)

    # Content extracted from image
    ocr_text: Optional[str] = None  # Text OCR'd from within the image
    caption: Optional[str] = None  # Figure caption (from FigureCaption element)

    # Classification
    image_type: ImageType = ImageType.UNKNOWN

    # Bounding box
    bbox: BoundingBox

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class TableCell(BaseModel):
    """
    Physical cell. Used for highlighting evidence.
    """

    text: str
    row_index: int = Field(..., ge=0)
    col_index: int = Field(..., ge=0)
    row_span: int = Field(1, ge=1)
    col_span: int = Field(1, ge=1)

    is_header: bool = False
    bbox: BoundingBox

    model_config = ConfigDict(frozen=True, extra="forbid")


class Table(BaseModel):
    """
    Dual-view table:
      - cells: geometric view (audit/highlight)
      - logical_rows: semantic view (generator-friendly)
      - logical_cell_refs: bridge logical -> physical (row/col coords)
      - image_base64: rendered image for vision analysis
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_num: int = Field(..., ge=1)
    reading_order_index: int = Field(..., ge=0)

    caption: Optional[str] = None
    table_type: TableType = TableType.UNKNOWN

    # View 1: physical
    cells: List[TableCell] = Field(default_factory=list)

    # View 2: logical
    # e.g. [{"Abbrev": "AE", "Term": "Adverse Event"}]
    logical_rows: List[Dict[str, str]] = Field(default_factory=list)

    # View 3: audit bridge (parallel to logical_rows)
    # e.g. [{"Abbrev": {"rc": (0,0)}, "Term": {"rc": (0,1)}}]
    logical_cell_refs: List[Dict[str, Dict[str, Tuple[int, int]]]] = Field(
        default_factory=list
    )

    # View 4: rendered image (for vision LLM analysis)
    image_base64: Optional[str] = None  # Base64-encoded table image
    image_format: str = "png"  # Format hint (png, jpg)

    # Multi-page table support
    page_nums: List[int] = Field(default_factory=list)  # All pages this table spans
    is_multipage: bool = False  # True if table spans multiple pages

    bbox: BoundingBox
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    _cell_index: Dict[Tuple[int, int], TableCell] = PrivateAttr(default_factory=dict)

    def _ensure_index(self) -> None:
        if not self._cell_index and self.cells:
            self._cell_index = {(c.row_index, c.col_index): c for c in self.cells}

    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """O(1) after first call (lazy index)."""
        self._ensure_index()
        return self._cell_index.get((row, col))

    def get_cell_from_ref(self, ref: Dict[str, Tuple[int, int]]) -> Optional[TableCell]:
        """ref format: {"rc": (row_idx, col_idx)}"""
        if not ref or "rc" not in ref:
            return None
        r, c = ref["rc"]
        return self.get_cell(r, c)

    def get_bbox_for_logical(
        self, logical_row_idx: int, key: str
    ) -> Optional[BoundingBox]:
        """Given logical row index + column key, return the exact cell bbox (if available)."""
        if logical_row_idx < 0 or logical_row_idx >= len(self.logical_cell_refs):
            return None
        ref = self.logical_cell_refs[logical_row_idx].get(key)
        cell = self.get_cell_from_ref(ref) if ref else None
        return cell.bbox if cell else None

    def to_markdown(self, max_rows: int = 10) -> str:
        """Simple Markdown view for LLM/context/debug."""
        rows = self.logical_rows[:max_rows]
        if not rows:
            return f"**Table**: {self.caption or 'Untitled'} (empty)\n"

        headers_map = self.metadata.get("headers")  # {col_idx: header_name}
        ordered_cols = self.metadata.get("ordered_cols")  # [col_idx,...]
        if (
            isinstance(headers_map, dict)
            and isinstance(ordered_cols, list)
            and ordered_cols
        ):
            cols = [headers_map[c] for c in ordered_cols if c in headers_map]
        else:
            cols = list(rows[0].keys())

        out = []
        out.append(f"**Table**: {self.caption or 'Untitled'}")
        out.append("| " + " | ".join(cols) + " |")
        out.append("| " + " | ".join(["---"] * len(cols)) + " |")

        for r in rows:
            out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")

        return "\n".join(out) + "\n"

    def iter_glossary_pairs(
        self,
    ) -> Iterator[Tuple[str, str, Optional[TableCell], Optional[TableCell]]]:
        """
        If table_type is GLOSSARY and metadata contains glossary_cols + headers mapping,
        yields (sf, lf, sf_cell, lf_cell) for each logical row.
        """
        if self.table_type != TableType.GLOSSARY:
            return iter(())  # type: ignore

        meta = self.metadata.get("glossary_cols") or {}
        headers = self.metadata.get("headers") or {}

        sf_col = meta.get("sf_col_idx")
        lf_col = meta.get("lf_col_idx")
        if sf_col is None or lf_col is None:
            return iter(())  # type: ignore
        if sf_col not in headers or lf_col not in headers:
            return iter(())  # type: ignore

        sf_key = headers[sf_col]
        lf_key = headers[lf_col]

        def _gen():
            for i, row in enumerate(self.logical_rows):
                sf = row.get(sf_key, "").strip()
                lf = row.get(lf_key, "").strip()
                if not sf or not lf:
                    continue
                sf_cell = (
                    self.get_cell_from_ref(self.logical_cell_refs[i].get(sf_key, {}))
                    if i < len(self.logical_cell_refs)
                    else None
                )
                lf_cell = (
                    self.get_cell_from_ref(self.logical_cell_refs[i].get(lf_key, {}))
                    if i < len(self.logical_cell_refs)
                    else None
                )
                yield (sf, lf, sf_cell, lf_cell)

        return _gen()


class Page(BaseModel):
    number: int = Field(..., ge=1)
    width: float
    height: float

    blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    images: List[ImageBlock] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def iter_content(self) -> Iterator[Tuple[str, str]]:
        """
        Deterministic stream by (reading_order, type_priority, y0, x0).
        """
        items = []

        for b in self.blocks:
            x0, y0, *_ = b.bbox.coords
            items.append((b.reading_order_index, 0, y0, x0, "TEXT", b.text))

        for t in self.tables:
            x0, y0, *_ = t.bbox.coords
            label = f"[Table: {t.caption or 'Untitled'}]"
            items.append((t.reading_order_index, 1, y0, x0, "TABLE", label))

        for img in self.images:
            x0, y0, *_ = img.bbox.coords
            label = f"[Image: {img.caption or 'Figure'}]"
            items.append((img.reading_order_index, 2, y0, x0, "IMAGE", label))

        items.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for *_sort, typ, txt in items:
            yield (typ, txt)


class DocumentGraph(BaseModel):
    doc_id: str
    pages: Dict[int, Page] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    _linear_text: Optional[str] = PrivateAttr(default=None)

    def get_page(self, page_num: int) -> Page:
        if page_num not in self.pages:
            raise KeyError(f"Page {page_num} not found")
        return self.pages[page_num]

    def iter_images(self) -> Iterator[ImageBlock]:
        """
        Iterate all images across all pages.
        """
        for pnum in sorted(self.pages.keys()):
            page = self.pages[pnum]
            for img in sorted(page.images, key=lambda i: i.reading_order_index):
                yield img

    def iter_linear_blocks(
        self, skip_header_footer: bool = True
    ) -> Iterator[TextBlock]:
        """
        Primary feeder for generators that read linearly.
        """
        for pnum in sorted(self.pages.keys()):
            page = self.pages[pnum]
            for block in sorted(page.blocks, key=lambda b: b.reading_order_index):
                if skip_header_footer and block.role in (
                    ContentRole.PAGE_HEADER,
                    ContentRole.PAGE_FOOTER,
                ):
                    continue
                yield block

    def iter_tables(self, table_type: Optional[TableType] = None) -> Iterator[Table]:
        for pnum in sorted(self.pages.keys()):
            for t in sorted(
                self.pages[pnum].tables, key=lambda x: x.reading_order_index
            ):
                if table_type is None or t.table_type == table_type:
                    yield t
