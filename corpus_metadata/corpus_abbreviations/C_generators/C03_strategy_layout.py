# corpus_metadata/corpus_abbreviations/C_generators/C03_strategy_layout.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import (
    ContentRole,
    DocumentGraph,
    TableType,
    TextBlock,
)


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _bbox_xy(bbox) -> Tuple[float, float, float, float]:
    # BoundingBox(coords=(x0,y0,x1,y1))
    x0, y0, x1, y1 = bbox.coords
    return float(x0), float(y0), float(x1), float(y1)


def _center_y(bbox) -> float:
    _, y0, _, y1 = _bbox_xy(bbox)
    return (y0 + y1) / 2.0


class LayoutCandidateGenerator(BaseCandidateGenerator):
    """
    Abbreviation Glossary Extractor (layout/table driven).

    Priority order:
      1) Structured glossary tables (TableType.GLOSSARY) -> highest precision
      2) Fallback: two-column alignment inside "abbreviations/glossary" sections

    Output:
      - FieldType.GLOSSARY_ENTRY
      - GeneratorType.GLOSSARY_TABLE or GeneratorType.TABLE_LAYOUT
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        self.glossary_headers = {
            (s or "").strip().lower()
            for s in cfg.get(
                "headers",
                [
                    "list of abbreviations",
                    "abbreviations",
                    "glossary",
                    "definition of terms",
                    "acronyms",
                ],
            )
        }

        # Vertical tolerance (same "row") in PDF coordinate units (points/pixels)
        self.row_tolerance = float(cfg.get("row_tolerance", 6.0))

        # Filters
        self.max_sf_len = int(cfg.get("max_sf_len", 15))
        self.require_caps = bool(cfg.get("require_caps", True))
        self.min_pairs_per_page = int(cfg.get("min_pairs_per_page", 2))

        # Dedup across doc
        self.dedupe = bool(cfg.get("dedupe", True))

        # Optional suppression list (if you already extracted DEFINITION_PAIR)
        self.known_short_forms: Set[str] = {
            str(x).strip().upper()
            for x in cfg.get("known_short_forms", [])
            if str(x).strip()
        }

        # Provenance defaults
        self.pipeline_version = str(cfg.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(cfg.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(cfg.get("doc_fingerprint") or "unknown-doc-fingerprint")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.TABLE_LAYOUT

    # -------------------------
    # Public API
    # -------------------------

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure
        out: List[Candidate] = []
        seen: Set[Tuple[str, str]] = set()

        # 1) BEST PATH: use structured glossary tables (if present)
        for t in doc.iter_tables(table_type=TableType.GLOSSARY):
            for sf, lf, sf_cell, lf_cell in t.iter_glossary_pairs():
                sf_u = sf.strip().upper()
                lf_c = _clean_ws(lf)
                if not self._is_valid_sf(sf_u) or not lf_c:
                    continue
                if sf_u in self.known_short_forms:
                    continue

                key = (sf_u, lf_c)
                if self.dedupe and key in seen:
                    continue
                seen.add(key)

                # Anchor bbox: prefer SF cell bbox; fallback to table bbox
                anchor_bbox = sf_cell.bbox if sf_cell else t.bbox

                # Build Coordinate
                loc = Coordinate(
                    page_num=int(t.page_num),
                    table_id=str(t.id),
                    cell_row=int(sf_cell.row_index) if sf_cell else None,
                    cell_col=int(sf_cell.col_index) if sf_cell else None,
                    bbox=anchor_bbox,
                )

                # Build ProvenanceMetadata
                prov = ProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=self.doc_fingerprint_default,
                    generator_name=GeneratorType.GLOSSARY_TABLE,
                    rule_version="layout::table_glossary::v1",
                )

                # Context: small markdown table for LLM/debug
                ctx = t.to_markdown(max_rows=20)

                out.append(
                    Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.GLOSSARY_ENTRY,
                        generator_type=GeneratorType.GLOSSARY_TABLE,
                        short_form=sf_u,
                        long_form=lf_c,
                        context_text=ctx,
                        context_location=loc,
                        initial_confidence=0.97,
                        provenance=prov,
                    )
                )

        # 2) FALLBACK: two-column alignment in glossary sections (if tables missing or incomplete)
        for pnum in sorted(doc.pages.keys()):
            page = doc.pages[pnum]
            if not self._page_in_glossary_section(page.blocks):
                continue

            body_blocks = [b for b in page.blocks if b.role == ContentRole.BODY_TEXT]
            if len(body_blocks) < 4:
                continue

            pairs = self._pairs_from_two_columns(body_blocks)
            if len(pairs) < self.min_pairs_per_page:
                continue

            for sf_block, lf_block in pairs:
                sf_u = _clean_ws(sf_block.text).upper()
                lf_c = _clean_ws(lf_block.text)

                if not self._is_valid_sf(sf_u) or not lf_c:
                    continue
                if sf_u in self.known_short_forms:
                    continue

                key = (sf_u, lf_c)
                if self.dedupe and key in seen:
                    continue
                seen.add(key)

                # Build Coordinate
                loc = Coordinate(
                    page_num=int(sf_block.page_num),
                    block_id=str(sf_block.id),
                    bbox=sf_block.bbox,
                )

                # Build ProvenanceMetadata
                prov = ProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=self.doc_fingerprint_default,
                    generator_name=GeneratorType.TABLE_LAYOUT,
                    rule_version="layout::column_alignment::v1",
                )

                # Context: the two blocks together
                ctx = f"SF: {sf_u}\nLF: {lf_c}"

                out.append(
                    Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.GLOSSARY_ENTRY,
                        generator_type=GeneratorType.TABLE_LAYOUT,
                        short_form=sf_u,
                        long_form=lf_c,
                        context_text=ctx,
                        context_location=loc,
                        initial_confidence=0.88,
                        provenance=prov,
                    )
                )

        return out

    # -------------------------
    # Helpers
    # -------------------------

    def _page_in_glossary_section(self, blocks: List[TextBlock]) -> bool:
        """
        Simple: if the page contains a section header matching glossary keywords -> true.
        """
        for b in blocks:
            if b.role != ContentRole.SECTION_HEADER:
                continue
            title = _clean_ws(b.text).lower()
            if not title:
                continue
            if any(h in title for h in self.glossary_headers):
                return True
        return False

    def _is_valid_sf(self, sf: str) -> bool:
        if not sf:
            return False
        if len(sf) > self.max_sf_len:
            return False
        if self.require_caps and not any(ch.isupper() for ch in sf):
            return False
        # Avoid junk like "TABLE", "FIG" as SF in left column
        if sf.isdigit():
            return False
        return True

    def _pairs_from_two_columns(self, blocks: List[TextBlock]) -> List[Tuple[TextBlock, TextBlock]]:
        """
        Column split by x0 midpoint, then greedy y-alignment matching.
        Works for implicit 2-column "AE   Adverse Event" style lists.
        """
        # Compute x0s
        x0s = []
        for b in blocks:
            x0, _, _, _ = _bbox_xy(b.bbox)
            x0s.append(x0)

        if not x0s:
            return []

        min_x = min(x0s)
        max_x = max(x0s)
        mid = (min_x + max_x) / 2.0

        left: List[TextBlock] = []
        right: List[TextBlock] = []

        for b in blocks:
            x0, _, _, _ = _bbox_xy(b.bbox)
            if x0 < mid:
                left.append(b)
            else:
                right.append(b)

        if not left or not right:
            return []

        # Sort by Y (top->bottom)
        left.sort(key=lambda b: _bbox_xy(b.bbox)[1])
        right.sort(key=lambda b: _bbox_xy(b.bbox)[1])

        used_right: Set[int] = set()
        pairs: List[Tuple[TextBlock, TextBlock]] = []

        # Greedy match: for each left SF block, find nearest right LF block by centerY
        for lb in left:
            sf_txt = _clean_ws(lb.text)
            if not self._looks_like_left_sf(sf_txt):
                continue

            lcy = _center_y(lb.bbox)

            best_idx = -1
            best_dist = float("inf")

            for i, rb in enumerate(right):
                if i in used_right:
                    continue
                rcy = _center_y(rb.bbox)
                dist = abs(lcy - rcy)
                if dist <= self.row_tolerance and dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx != -1:
                used_right.add(best_idx)
                pairs.append((lb, right[best_idx]))

        return pairs

    def _looks_like_left_sf(self, text: str) -> bool:
        """
        Left column must be acronym-ish.
        Keep it simple to avoid overfitting.
        """
        t = text.strip()
        if not t:
            return False
        if len(t) > self.max_sf_len:
            return False
        if self.require_caps and not any(ch.isupper() for ch in t):
            return False
        # Avoid obvious labels
        tl = t.lower()
        if tl in {"abbreviation", "abbrev", "acronym", "term", "definition", "meaning"}:
            return False
        return True