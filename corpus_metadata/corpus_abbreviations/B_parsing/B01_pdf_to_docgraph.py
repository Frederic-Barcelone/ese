# corpus_metadata/corpus_abbreviations/B_parsing/B01_pdf_to_docgraph.py
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf

from A_core.A02_interfaces import BaseParser
from A_core.A01_domain_models import BoundingBox
from B_parsing.B02_doc_graph import DocumentGraph, Page, TextBlock, ContentRole


# -----------------------------
# Text normalization helpers
# -----------------------------

def normalize_repeated_text(text: str) -> str:
    """
    Normaliza texto para detectar headers/footers repetidos:
    - lowercase
    - colapsa espacios
    - sustituye dígitos por '#'
    """
    t = " ".join((text or "").split()).lower()
    t = re.sub(r"\d+", "#", t)
    return t.strip()


# --- glue hyphens in abbreviation patterns only (MG- ADL -> MG-ADL) ---
ABBREV_HYPHEN_RE = re.compile(
    r"(?P<a>[A-Za-z0-9][A-Za-z0-9+\./]{0,12})\s*-\s*(?P<b>[A-Za-z0-9][A-Za-z0-9+\./]{0,12})"
)


def _looks_like_abbrev_token(tok: str) -> bool:
    if not tok:
        return False
    has_upper = any(c.isupper() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    has_plus = "+" in tok
    shortish = len(tok) <= 12
    return shortish and (has_upper or has_digit or has_plus)


def normalize_abbrev_hyphens(text: str) -> str:
    """
    Convierte 'MG- ADL' -> 'MG-ADL' si ambos lados parecen abreviaturas.
    Evita tocar palabras normales tipo 'long-term' (minúsculas).
    """
    def repl(m: re.Match) -> str:
        a = m.group("a")
        b = m.group("b")
        if _looks_like_abbrev_token(a) and _looks_like_abbrev_token(b):
            return f"{a}-{b}"
        return m.group(0)

    return ABBREV_HYPHEN_RE.sub(repl, text)


# -----------------------------
# Noise patterns (typical journal footers/headers)
# -----------------------------
KNOWN_FOOTER_PATTERNS = [
    r"\bdownloaded from\b",
    r"\bwiley online library\b",
    r"\bterms and conditions\b",
    r"\boa articles?\b",
    r"\bcreative commons\b",
    r"\bdoi:\s*10\.\d{4,9}/",
    r"\bhttps?://onlinelibrary\.wiley\.com\b",
    r"\bsee the terms and conditions\b",
    r"\beuropean journal of neurology\b",
    r"\bdovepress\b",
    r"\btaylor\s*&\s*francis\s*group\b",
    r"\bpublish your work in this journal\b",
    r"\bjournal of inflammation research\b",
    r"\bsubmit your manuscript\b",
    r"\bopen access full text article\b",
]
KNOWN_FOOTER_RE = re.compile("|".join(KNOWN_FOOTER_PATTERNS), flags=re.IGNORECASE)

# Running header patterns (author names like "Liao et al", "Smith et al.")
RUNNING_HEADER_RE = re.compile(r"^[A-Z][a-z]+\s+et\s+al\.?$", flags=re.IGNORECASE)

# Numbered reference pattern (e.g., "7. Author A, ..." or "7. DCVAS Study Group, ...")
NUMBERED_REFERENCE_RE = re.compile(
    r"^\d{1,3}\.\s+[A-Z]",  # Starts with number, period, then capital letter
    flags=re.MULTILINE
)

# OCR garbage patterns to remove
OCR_GARBAGE_PATTERNS = [
    r"\b\d+\s*[bBpP»«]+\s*$",      # "18194 bP»" -> page number garbage
    r'[""]\s*[*>]\s*$',             # ""*", ">" at end
    r"\bfo\)\s*$",                  # "fo)" misread lock icon
    r"^\s*[+*~>]{2,}\s*$",          # lines of just symbols
    r"\s+[»«]+\s*$",                # trailing » or «
]
OCR_GARBAGE_RE = re.compile("|".join(OCR_GARBAGE_PATTERNS))

# Pattern to detect garbled flowchart/diagram content
# Matches text with numbered steps separated by symbols like + * | ~
FLOWCHART_PATTERN = re.compile(
    r"^\d+[a-z]?\.\s+.{10,}\s+[+*|~>-]\s+.{10,}\s+[+*|~>-]\s+",
    re.IGNORECASE
)


def _is_garbled_flowchart(text: str) -> bool:
    """
    Detect if text block is a garbled flowchart/diagram.
    These typically have numbered steps with symbols like + * | ~ separating them.
    """
    if not text or len(text) < 100:
        return False

    # Count flowchart-like symbols (including various dash types)
    symbol_count = sum(1 for c in text if c in "+*|~>-")
    word_count = len(text.split())

    # Check for numbered step pattern with symbols (e.g., "1. ... + ... | 2. ...")
    has_numbered_steps = bool(re.search(
        r"\d+[a-z]?\.\s+[^|+*]+[+*|]\s+.*\d+[a-z]?\.\s+",
        text
    ))

    # High symbol-to-word ratio suggests garbled diagram
    if word_count > 0:
        ratio = symbol_count / word_count
        # Lower threshold if we detect numbered step pattern
        if has_numbered_steps and ratio > 0.08:
            return True
        if ratio > 0.12:
            if FLOWCHART_PATTERN.search(text):
                return True
            if has_numbered_steps:
                return True

    return False


def _extract_figure_reference(text: str) -> Optional[str]:
    """Extract figure number from garbled flowchart text."""
    # Look for patterns like "Figure 5" or "Fig. 5"
    match = re.search(r"(?:Figure|Fig\.?)\s*(\d+)", text, re.IGNORECASE)
    if match:
        return f"Figure {match.group(1)}"
    return None


def _bbox_overlaps(bbox1: BoundingBox, bbox2: BoundingBox, threshold: float = 0.5) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_0, y1_0, x1_1, y1_1 = bbox1.coords
    x2_0, y2_0, x2_1, y2_1 = bbox2.coords

    # Calculate intersection
    ix0 = max(x1_0, x2_0)
    iy0 = max(y1_0, y2_0)
    ix1 = min(x1_1, x2_1)
    iy1 = min(y1_1, y2_1)

    if ix1 <= ix0 or iy1 <= iy0:
        return False

    intersection = (ix1 - ix0) * (iy1 - iy0)
    area1 = (x1_1 - x1_0) * (y1_1 - y1_0)

    if area1 <= 0:
        return False

    return (intersection / area1) >= threshold


def document_to_markdown(
    doc: DocumentGraph,
    include_tables: bool = True,
    skip_header_footer: bool = True,
) -> str:
    """
    Convert DocumentGraph to Markdown.
    - Renders tables as proper markdown tables (not flattened text)
    - Skips text blocks that overlap with table regions
    """
    lines: List[str] = []

    for pnum in sorted(doc.pages.keys()):
        page = doc.pages[pnum]
        tables = getattr(page, "tables", []) or []

        # Collect all content items (blocks + tables) for proper ordering
        content_items = []

        for b in page.blocks:
            if skip_header_footer and b.role in (ContentRole.PAGE_HEADER, ContentRole.PAGE_FOOTER):
                continue

            # Skip blocks that overlap with any table region
            if tables:
                overlaps_table = any(
                    _bbox_overlaps(b.bbox, t.bbox, threshold=0.3)
                    for t in tables
                )
                if overlaps_table:
                    continue

            y0 = b.bbox.coords[1] if b.bbox else 0
            content_items.append((b.reading_order_index, y0, "block", b))

        if include_tables:
            for t in tables:
                y0 = t.bbox.coords[1] if t.bbox else 0
                content_items.append((t.reading_order_index, y0, "table", t))

        # Sort by reading order, then y position
        content_items.sort(key=lambda x: (x[0], x[1]))

        for _, _, item_type, item in content_items:
            if item_type == "block":
                text = item.text

                # Detect and replace garbled flowcharts/diagrams
                if _is_garbled_flowchart(text):
                    fig_ref = _extract_figure_reference(text)
                    if fig_ref:
                        lines.append(f"[{fig_ref}: Flowchart/Diagram - see PDF for visual]")
                    else:
                        lines.append("[Flowchart/Diagram - see PDF for visual]")
                    continue

                if item.role == ContentRole.SECTION_HEADER:
                    lines.append(f"## {text}")
                else:
                    lines.append(text)
            elif item_type == "table":
                # Render as markdown table
                lines.append("")
                lines.append(_table_to_markdown(item))

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _table_to_markdown(table) -> str:
    """Convert Table object to markdown table format."""
    if not table.logical_rows:
        return f"[Table: {table.caption or 'Empty'}]"

    # Get headers
    headers_map = table.metadata.get("headers", {})
    if headers_map:
        headers = [headers_map.get(i, f"Col{i}") for i in sorted(headers_map.keys())]
    elif table.logical_rows:
        headers = list(table.logical_rows[0].keys())
    else:
        return f"[Table: {table.caption or 'No data'}]"

    out = []
    if table.caption:
        out.append(f"**{table.caption}**")
        out.append("")

    # Header row
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for row in table.logical_rows:
        cells = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        out.append("| " + " | ".join(cells) + " |")

    return "\n".join(out)


# -----------------------------
# Header detection helpers
# -----------------------------

SECTION_NUM_RE = re.compile(r"^\s*\d+(\.\d+)*\s*\|\s*\S+")
SECTION_PLAIN_RE = re.compile(
    r"^(abstract|introduction|methods|materials and methods|results|discussion|conclusion|conclusions|references)$",
    flags=re.IGNORECASE,
)

META_PREFIX_RE = re.compile(
    r"^(correspondence|received|revised|accepted|funding|keywords)\s*:",
    flags=re.IGNORECASE,
)

# very common affiliation tokens in biomedical PDFs
AFFIL_TOKENS_RE = re.compile(
    r"\b(university|hospital|college|centre|center|institute|department|school|foundation|clinic|irccs|charité)\b",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(r"\b\S+@\S+\b")

# many author blocks have pipes separating names
PIPEY_RE = re.compile(r"\s\|\s")


class PDFToDocGraphParser(BaseParser):
    """
    PDF -> DocumentGraph usando Unstructured.

    Objetivos:
    - Orden de lectura determinista (2 columnas)
    - Detectar headers/footers repetidos (robusto)
    - Detectar SECTION_HEADER evitando falsos positivos (autores/afiliaciones)
    - Reparar guiones de abreviaturas: 'MG- ADL' -> 'MG-ADL'
    - Limpieza de pipes iniciales: '| Andreas ...' -> 'Andreas ...'

    Config options:
    - unstructured_strategy: "hi_res" (default), "fast", "ocr_only", "auto"
    - hi_res_model_name: "yolox" (default), "detectron2_onnx"
    - infer_table_structure: True (default) - extract table structure in hi_res
    - extract_images_in_pdf: False (default) - extract images from PDF
    - languages: ["eng"] (default) - OCR language packs for better accuracy
    - include_page_breaks: True (default) - track page boundaries
    - force_fast: False (default) - force fast strategy regardless of config
    - force_hf_offline: False (default) - block HuggingFace model downloads
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Offline HF - disabled by default to allow model downloads for hi_res
        self.force_hf_offline = bool(self.config.get("force_hf_offline", False))
        if self.force_hf_offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Unstructured knobs - default to hi_res for better extraction
        self.unstructured_strategy = str(self.config.get("unstructured_strategy", "hi_res")).strip() or "hi_res"
        self.force_fast = bool(self.config.get("force_fast", False))
        if self.force_fast:
            self.unstructured_strategy = "fast"

        # hi_res specific options
        self.hi_res_model_name = self.config.get("hi_res_model_name", "yolox")
        self.infer_table_structure = bool(self.config.get("infer_table_structure", True))
        self.extract_images_in_pdf = bool(self.config.get("extract_images_in_pdf", False))

        # OCR and structure options
        self.languages = self.config.get("languages", ["eng"])  # improves OCR accuracy
        self.include_page_breaks = bool(self.config.get("include_page_breaks", True))  # track page boundaries

        # categorías a descartar (case-insensitive)
        self.drop_categories = {c.strip().lower() for c in (self.config.get("drop_categories") or []) if isinstance(c, str)}

        # Orden / líneas
        self.y_tolerance = float(self.config.get("y_tolerance", 3.0))

        # Header/Footer por posición
        self.header_top_pct = float(self.config.get("header_top_pct", 0.07))        # top 7%
        self.footer_bottom_pct = float(self.config.get("footer_bottom_pct", 0.90)) # bottom 10%

        # Header/Footer por repetición
        self.min_repeat_count = int(self.config.get("min_repeat_count", 3))
        self.min_repeat_pages = int(self.config.get("min_repeat_pages", 3))
        self.repeat_zone_majority = float(self.config.get("repeat_zone_majority", 0.60))

        # 2 columnas
        self.two_col_min_side_blocks = int(self.config.get("two_col_min_side_blocks", 6))

    # -----------------------
    # Public
    # -----------------------

    def parse(self, file_path: str) -> DocumentGraph:
        page_dims = self._get_page_dimensions(file_path)
        elements = self._call_partition_pdf(file_path)

        # Pass 1: raw blocks + repetition stats
        raw_pages: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        norm_count: Counter[str] = Counter()
        norm_pages: Dict[str, set] = defaultdict(set)
        norm_zone_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        norm_sample_text: Dict[str, str] = {}

        for el in elements:
            # drop categories if requested
            cat = getattr(el, "category", None)
            cat_norm = (cat or "").strip().lower() if isinstance(cat, str) else ""
            cls_norm = el.__class__.__name__.strip().lower()

            if self.drop_categories and (cat_norm in self.drop_categories or cls_norm in self.drop_categories):
                continue

            page_num = self._get_element_page_num(el)
            if page_num is None:
                continue

            page_w, page_h = page_dims.get(page_num, (0.0, 0.0))

            text_raw = getattr(el, "text", "") or ""
            text = self._clean_text(text_raw)
            if not text:
                continue

            bbox = self._bbox_from_element(el, page_w=page_w, page_h=page_h)
            zone = self._zone_from_bbox(bbox, page_h=page_h)

            is_section_header = self._is_section_header(el, zone, text)

            rb = {
                "text": text,
                "bbox": bbox,
                "x0": float(bbox.coords[0]),
                "y0": float(bbox.coords[1]),
                "zone": zone,
                "is_section_header": is_section_header,
            }
            raw_pages[page_num].append(rb)

            norm = normalize_repeated_text(text)
            if norm:
                norm_count[norm] += 1
                norm_pages[norm].add(page_num)
                if zone in ("HEADER", "FOOTER"):
                    norm_zone_votes[norm][zone] += 1
                norm_sample_text.setdefault(norm, text)

        repeated_headers, repeated_footers = self._infer_repeated_headers_footers(
            norm_count=norm_count,
            norm_pages=norm_pages,
            norm_zone_votes=norm_zone_votes,
            norm_sample_text=norm_sample_text,
        )

        # Pass 2: build DocumentGraph
        graph = DocumentGraph(doc_id=file_path)

        for page_num in sorted(raw_pages.keys()):
            page_w, page_h = page_dims.get(page_num, (0.0, 0.0))
            page_obj = Page(number=page_num, width=page_w, height=page_h)

            ordered = self._order_blocks_deterministically(raw_pages[page_num], page_w=page_w)

            blocks: List[TextBlock] = []
            for idx, rb in enumerate(ordered):
                txt = rb["text"]
                if not txt:
                    continue

                role = ContentRole.BODY_TEXT
                norm = normalize_repeated_text(txt)

                # Check if this looks like a numbered reference (preserve as body text)
                is_reference = bool(NUMBERED_REFERENCE_RE.match(txt))

                # Hard override: known footer noise (but not references)
                if self._looks_like_known_footer(txt) and not is_reference:
                    role = ContentRole.PAGE_FOOTER
                # Running header pattern: "Author et al" or "Author et al."
                elif self._is_running_header(txt, rb.get("zone")):
                    role = ContentRole.PAGE_HEADER
                elif rb.get("zone") == "HEADER" and norm in repeated_headers:
                    role = ContentRole.PAGE_HEADER
                elif rb.get("zone") == "FOOTER" and norm in repeated_footers and not is_reference:
                    role = ContentRole.PAGE_FOOTER
                # fallback repetition even if zone couldn't be trusted
                elif norm in repeated_footers and self._is_short_repeated_noise(txt) and not is_reference:
                    role = ContentRole.PAGE_FOOTER
                elif rb.get("is_section_header"):
                    role = ContentRole.SECTION_HEADER

                blocks.append(
                    TextBlock(
                        text=txt,
                        page_num=page_num,
                        reading_order_index=idx,
                        role=role,
                        bbox=rb["bbox"],
                    )
                )

            page_obj.blocks = blocks
            graph.pages[page_num] = page_obj

        return graph

    # -----------------------
    # Unstructured call
    # -----------------------

    def _call_partition_pdf(self, file_path: str):
        """
        Partition PDF using unstructured. Supports hi_res with inference.
        """
        kwargs = {
            "filename": file_path,
            "strategy": self.unstructured_strategy,
            "languages": self.languages,
            "include_page_breaks": self.include_page_breaks,
        }

        # Add hi_res specific options when using hi_res strategy
        if self.unstructured_strategy == "hi_res":
            kwargs["hi_res_model_name"] = self.hi_res_model_name
            kwargs["infer_table_structure"] = self.infer_table_structure
            kwargs["extract_images_in_pdf"] = self.extract_images_in_pdf

        return partition_pdf(**kwargs)

    # -----------------------
    # Repetition inference
    # -----------------------

    def _infer_repeated_headers_footers(
        self,
        norm_count: Counter[str],
        norm_pages: Dict[str, set],
        norm_zone_votes: Dict[str, Counter[str]],
        norm_sample_text: Dict[str, str],
    ) -> Tuple[set, set]:
        repeated_headers: set = set()
        repeated_footers: set = set()

        for norm, total_c in norm_count.items():
            pages = norm_pages.get(norm, set())
            if total_c < self.min_repeat_count:
                continue
            if len(pages) < self.min_repeat_pages:
                continue

            sample = norm_sample_text.get(norm, "")
            if self._looks_like_known_footer(sample):
                repeated_footers.add(norm)
                continue

            # page-number-only lines are usually footers
            if sample.strip().isdigit():
                repeated_footers.add(norm)
                continue

            zone_votes = norm_zone_votes.get(norm)
            if not zone_votes:
                continue

            top_zone, top_votes = zone_votes.most_common(1)[0]
            frac = float(top_votes) / float(total_c) if total_c else 0.0
            if top_zone == "HEADER" and frac >= self.repeat_zone_majority:
                repeated_headers.add(norm)
            elif top_zone == "FOOTER" and frac >= self.repeat_zone_majority:
                repeated_footers.add(norm)

        return repeated_headers, repeated_footers

    def _looks_like_known_footer(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return bool(KNOWN_FOOTER_RE.search(t))

    def _is_short_repeated_noise(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if len(t) <= 3 and t.isdigit():
            return True
        if self._looks_like_known_footer(t):
            return True
        return False

    def _is_running_header(self, text: str, zone: Optional[str]) -> bool:
        """Detect running headers like 'Liao et al' or 'Smith et al.'"""
        t = (text or "").strip()
        if not t:
            return False
        # Must be short (author name + et al)
        if len(t) > 30:
            return False
        # Match "Author et al" pattern
        if RUNNING_HEADER_RE.match(t):
            return True
        # Also catch in header zone with page numbers like "Liao et al 18194"
        if zone == "HEADER" and re.match(r"^[A-Z][a-z]+\s+et\s+al\.?\s*\d*$", t, re.IGNORECASE):
            return True
        return False

    # -----------------------
    # Element -> bbox, page, roles
    # -----------------------

    def _get_element_page_num(self, el) -> Optional[int]:
        md = getattr(el, "metadata", None)
        if md is None:
            return None
        pn = getattr(md, "page_number", None)
        if pn is None:
            pn = getattr(md, "page", None)
        try:
            return int(pn) if pn is not None else None
        except Exception:
            return None

    def _bbox_from_element(self, el, page_w: float, page_h: float) -> BoundingBox:
        md = getattr(el, "metadata", None)
        coords_meta = getattr(md, "coordinates", None) if md else None

        points = None
        if coords_meta is not None and hasattr(coords_meta, "points"):
            try:
                points = list(coords_meta.points)
            except Exception:
                points = None

        if points is None and isinstance(coords_meta, dict) and "points" in coords_meta:
            points = coords_meta.get("points")

        if points:
            xs = [float(p[0]) for p in points if p and len(p) >= 2]
            ys = [float(p[1]) for p in points if p and len(p) >= 2]
            if xs and ys:
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                return BoundingBox(
                    coords=(max(0.0, x0), max(0.0, y0), max(0.0, x1), max(0.0, y1)),
                    page_width=page_w or None,
                    page_height=page_h or None,
                    is_normalized=False,
                )

        return BoundingBox(
            coords=(0.0, 0.0, 0.0, 0.0),
            page_width=page_w or None,
            page_height=page_h or None,
            is_normalized=False,
        )

    def _zone_from_bbox(self, bbox: BoundingBox, page_h: float) -> str:
        if page_h <= 0:
            return "BODY"
        x0, y0, x1, y1 = bbox.coords
        if (x0, y0, x1, y1) == (0.0, 0.0, 0.0, 0.0):
            return "BODY"
        header_y = page_h * self.header_top_pct
        footer_y = page_h * self.footer_bottom_pct
        if y1 <= header_y:
            return "HEADER"
        if y0 >= footer_y:
            return "FOOTER"
        return "BODY"

    # -----------------------
    # SECTION_HEADER detection (stricter)
    # -----------------------

    def _looks_like_author_or_affiliation(self, t: str) -> bool:
        if not t:
            return False
        tl = t.strip().lower()

        # easy metadata exclusions
        if META_PREFIX_RE.search(t):
            return True
        if EMAIL_RE.search(t):
            return True

        # author blocks often have pipes
        if PIPEY_RE.search(t):
            return True

        # affiliations: university/hospital etc
        if AFFIL_TOKENS_RE.search(t):
            return True

        # lots of commas + digits (affiliations / footnotes)
        digits = sum(ch.isdigit() for ch in t)
        commas = t.count(",")
        if digits >= 2 and commas >= 1:
            return True

        return False

    def _is_section_header(self, el, zone: str, cleaned_text: str) -> bool:
        """
        Goal: mark true paper section headings, not author/affiliation/title fragments.
        """
        if zone != "BODY":
            return False

        t = (cleaned_text or "").strip()
        if not t:
            return False

        tl = t.lower()

        # anti-ruido típico
        if tl.startswith(("figure", "fig.", "table", "supplementary", "appendix")):
            return False
        if "downloaded from" in tl or "wiley online library" in tl:
            return False
        if tl.startswith("doi") or "https://doi.org" in tl:
            return False

        # kill obvious metadata / author/affiliation lines
        if self._looks_like_author_or_affiliation(t):
            return False

        # section numbering style: "2 | Methods"
        if SECTION_NUM_RE.match(t):
            return True

        # plain section headings (rare but happens): "Methods"
        if SECTION_PLAIN_RE.match(t.strip()):
            return True

        # avoid sentence-like things
        if ". " in t or t.endswith("."):
            return False

        # avoid too long
        if len(t.split()) > 18:
            return False

        # category hints: only accept Title/Header if it *still* looks like a header
        cat = getattr(el, "category", None)
        if isinstance(cat, str):
            cat_norm = cat.strip().lower()
            if cat_norm in {"title", "header"}:
                # must be short and "header-like"
                if t[0].isupper() or t[0].isdigit():
                    return True
                return False

        cls = el.__class__.__name__.lower()
        if cls in {"title", "header"}:
            if t[0].isupper() or t[0].isdigit():
                return True
            return False

        return False

    # -----------------------
    # Ordering (2 columns)
    # -----------------------

    def _is_two_column_page(self, raw_blocks: List[Dict[str, Any]], page_w: float) -> bool:
        if not raw_blocks or page_w <= 0:
            return False

        xs = []
        for rb in raw_blocks:
            x0, _, x1, _ = rb["bbox"].coords
            xc = (x0 + x1) / 2.0
            xs.append(xc)

        left = sum(1 for x in xs if x < page_w * 0.45)
        right = sum(1 for x in xs if x > page_w * 0.55)
        mid = sum(1 for x in xs if page_w * 0.45 <= x <= page_w * 0.55)

        min_side = self.two_col_min_side_blocks
        return (left >= min_side and right >= min_side and mid <= max(2, int(0.08 * len(xs))))

    def _order_single_column(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = sorted(items, key=lambda r: (r["y0"], r["x0"]))

        ordered: List[Dict[str, Any]] = []
        current_line: List[Dict[str, Any]] = []
        current_y = None

        for it in items:
            if current_y is None:
                current_y = it["y0"]
                current_line = [it]
                continue

            if abs(it["y0"] - current_y) <= self.y_tolerance:
                current_line.append(it)
            else:
                current_line.sort(key=lambda r: r["x0"])
                ordered.extend(current_line)
                current_y = it["y0"]
                current_line = [it]

        if current_line:
            current_line.sort(key=lambda r: r["x0"])
            ordered.extend(current_line)

        return ordered

    def _order_blocks_deterministically(self, raw_blocks: List[Dict[str, Any]], page_w: float) -> List[Dict[str, Any]]:
        if not raw_blocks:
            return []

        two_cols = self._is_two_column_page(raw_blocks, page_w=page_w)
        if not two_cols:
            return self._order_single_column(raw_blocks)

        full_width: List[Dict[str, Any]] = []
        left_items: List[Dict[str, Any]] = []
        right_items: List[Dict[str, Any]] = []

        for rb in raw_blocks:
            x0, _, x1, _ = rb["bbox"].coords
            width = (x1 - x0)

            if page_w > 0 and width >= page_w * 0.75:
                full_width.append(rb)
                continue

            xc = (x0 + x1) / 2.0
            if xc < page_w / 2.0:
                left_items.append(rb)
            else:
                right_items.append(rb)

        return (
            self._order_single_column(full_width)
            + self._order_single_column(left_items)
            + self._order_single_column(right_items)
        )

    # -----------------------
    # Utils
    # -----------------------

    _KEEP_HYPHEN_PREFIXES = {
        "anti", "non", "pre", "post", "co", "multi", "bi", "tri",
        "long", "short", "open", "double", "single", "high", "low",
        "well", "self", "cross", "small", "medium", "large",
    }
    _COMMON_SUFFIX_FRAGS = {
        "ment", "tion", "tions", "sion", "sions", "tive", "tives",
        "ness", "less", "able", "ible", "ated", "ation", "ations",
        "ing", "ed", "ive", "ous", "ally", "ity", "ies", "es", "ly",
        "bulin", "blast", "cytes", "rhage", "lines", "tology",  # medical suffixes
        "alities", "ressive", "pressive", "globulin",
        "itis", "osis", "emia", "pathy", "plasty",  # more medical suffixes
    }
    # Common hyphenated compound words to preserve
    _KEEP_HYPHENATED = {
        "anca-associated", "medium-sized", "end-stage",
    }
    # Patterns that need space preserved: "small- and" not "small-and"
    _HYPHEN_SPACE_PRESERVE = {
        "small", "medium", "large", "short", "long",
    }

    def _clean_text(self, text: str) -> str:
        """
        Text cleaning:
        - Remove OCR garbage artifacts
        - Rejoin hyphenated words split across lines
        - Collapse whitespace
        - Fix abbreviation hyphens: 'MG- ADL' -> 'MG-ADL'
        """
        t = text or ""
        t = t.replace("\r", "\n")

        # 0) Remove OCR garbage patterns
        t = OCR_GARBAGE_RE.sub("", t)

        # 1) Join hyphenated line-break words: word-\n word -> wordword
        t = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", t)

        # 2) Handle "immuno- globulin" style (hyphen + space, broken word)
        def _dehyphen_repl(m: re.Match) -> str:
            a = m.group(1)
            b = m.group(2)
            a_l = a.lower()
            b_l = b.lower()
            combined = f"{a_l}-{b_l}"

            # Keep known hyphenated compounds
            if combined in self._KEEP_HYPHENATED:
                return f"{a}-{b}"

            # Preserve "small- and", "medium- and" patterns (hyphen + space + and)
            if a_l in self._HYPHEN_SPACE_PRESERVE and b_l == "and":
                return f"{a}- {b}"

            # Keep hyphen for known prefixes
            if a_l in self._KEEP_HYPHEN_PREFIXES:
                return f"{a}-{b}"

            # Remove hyphen when right side looks like a suffix fragment
            if b_l in self._COMMON_SUFFIX_FRAGS:
                return f"{a}{b}"

            # Remove hyphen for short left parts (likely broken words)
            if len(a) <= 5 and len(b) >= 3:
                return f"{a}{b}"

            return f"{a}-{b}"

        t = re.sub(r"\b([A-Za-z]{2,})-\s+([A-Za-z]{2,})\b", _dehyphen_repl, t)

        # 3) collapse whitespace
        t = " ".join(t.split()).strip()

        # 4) strip leading pipe artefact ("| Andreas..." -> "Andreas...")
        if re.match(r"^\|\s*[A-Za-z]", t):
            t = re.sub(r"^\|\s*", "", t)

        # 5) normalize abbrev hyphen spacing only for abbrev-like tokens
        t = normalize_abbrev_hyphens(t)

        return t.strip()

    def _get_page_dimensions(self, file_path: str) -> Dict[int, Tuple[float, float]]:
        dims: Dict[int, Tuple[float, float]] = {}
        doc = fitz.open(file_path)
        try:
            for i in range(doc.page_count):
                page = doc[i]
                dims[i + 1] = (float(page.rect.width), float(page.rect.height))
        finally:
            doc.close()
        return dims