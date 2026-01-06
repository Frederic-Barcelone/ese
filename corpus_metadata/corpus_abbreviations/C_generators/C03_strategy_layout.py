# corpus_metadata/corpus_abbreviations/C_generators/C03_strategy_layout.py
"""
El Geógrafo - Spatial Extraction.

Extracts data based on WHERE it is, not WHAT it says.
Uses document graph coordinates to find data in specific zones.

Strategies:
  - Zone extraction: Top-right corner (protocol ID), headers, footers
  - Label-value pairs: "Protocol ID:" followed by the actual value
  - Column alignment: Two-column glossary-style layouts

Analogy: A mail carrier. Doesn't read the letters, just looks at
the address on the envelope to know where to deliver.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from enum import Enum
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


def _bbox_coords(bbox) -> Tuple[float, float, float, float]:
    """Extract (x0, y0, x1, y1) from BoundingBox."""
    if bbox is None:
        return (0.0, 0.0, 0.0, 0.0)
    coords = bbox.coords
    return (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))


def _center_y(bbox) -> float:
    _, y0, _, y1 = _bbox_coords(bbox)
    return (y0 + y1) / 2.0


class PageZone(str, Enum):
    """Predefined page zones for spatial extraction."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class LabelPattern:
    """Definition of a label-value extraction pattern."""

    name: str
    label_regex: re.Pattern
    entity_type: str
    confidence: float = 0.90


# Common label patterns for pharma/clinical documents
LABEL_PATTERNS: List[LabelPattern] = [
    LabelPattern(
        name="protocol_id",
        label_regex=re.compile(r"Protocol\s*(?:ID|No\.?|Number)?[:\s]*", re.IGNORECASE),
        entity_type="PROTOCOL_ID",
        confidence=0.95,
    ),
    LabelPattern(
        name="study_id",
        label_regex=re.compile(r"Study\s*(?:ID|No\.?|Number)?[:\s]*", re.IGNORECASE),
        entity_type="STUDY_ID",
        confidence=0.95,
    ),
    LabelPattern(
        name="sponsor",
        label_regex=re.compile(r"Sponsor[:\s]*", re.IGNORECASE),
        entity_type="SPONSOR",
        confidence=0.90,
    ),
    LabelPattern(
        name="investigator",
        label_regex=re.compile(r"(?:Principal\s+)?Investigator[:\s]*", re.IGNORECASE),
        entity_type="INVESTIGATOR",
        confidence=0.90,
    ),
    LabelPattern(
        name="version",
        label_regex=re.compile(r"(?:Document\s+)?Version[:\s]*", re.IGNORECASE),
        entity_type="VERSION",
        confidence=0.85,
    ),
    LabelPattern(
        name="date",
        label_regex=re.compile(r"(?:Effective\s+)?Date[:\s]*", re.IGNORECASE),
        entity_type="DATE",
        confidence=0.85,
    ),
    LabelPattern(
        name="indication",
        label_regex=re.compile(r"Indication[:\s]*", re.IGNORECASE),
        entity_type="INDICATION",
        confidence=0.90,
    ),
    LabelPattern(
        name="drug_name",
        label_regex=re.compile(r"(?:Study\s+)?Drug[:\s]*", re.IGNORECASE),
        entity_type="DRUG_NAME",
        confidence=0.90,
    ),
    LabelPattern(
        name="phase",
        label_regex=re.compile(r"Phase[:\s]*", re.IGNORECASE),
        entity_type="PHASE",
        confidence=0.90,
    ),
]


class LayoutCandidateGenerator(BaseCandidateGenerator):
    """
    Spatial Extraction - finds data by location, not content.

    Strategies:
      1. Zone extraction: specific page regions (top-right, headers, etc.)
      2. Label-value: "Protocol ID:" followed by the actual value
      3. Column alignment: two-column glossary-style layouts
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Zone detection thresholds (percentage of page dimensions)
        self.zone_margin = float(self.config.get("zone_margin", 0.15))  # 15% from edges

        # Label-value extraction
        self.max_value_distance = float(
            self.config.get("max_value_distance", 200)
        )  # pixels
        self.max_value_length = int(self.config.get("max_value_length", 100))

        # Glossary section keywords
        self.glossary_headers = {
            s.strip().lower()
            for s in self.config.get(
                "glossary_headers",
                [
                    "list of abbreviations",
                    "abbreviations",
                    "glossary",
                    "definition of terms",
                    "acronyms",
                ],
            )
        }

        # Column alignment tolerance
        self.row_tolerance = float(self.config.get("row_tolerance", 6.0))
        self.max_sf_len = int(self.config.get("max_sf_len", 15))

        # Deduplication
        self.dedupe = bool(self.config.get("dedupe", True))

        # Provenance
        self.pipeline_version = str(
            self.config.get("pipeline_version") or get_git_revision_hash()
        )
        self.run_id = str(self.config.get("run_id") or generate_run_id("LAYOUT"))
        self.doc_fingerprint_default = str(
            self.config.get("doc_fingerprint") or "unknown-doc-fingerprint"
        )

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.TABLE_LAYOUT

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        """Extract candidates using spatial strategies."""
        doc = doc_structure
        candidates: List[Candidate] = []
        seen: Set[str] = set()

        # Strategy 1: Label-value extraction (forms)
        candidates.extend(self._extract_label_values(doc, seen))

        # Strategy 2: Zone extraction (first page corners)
        candidates.extend(self._extract_from_zones(doc, seen))

        # Strategy 3: Glossary table extraction
        candidates.extend(self._extract_glossary_tables(doc, seen))

        # Strategy 4: Two-column alignment (fallback for glossaries)
        candidates.extend(self._extract_column_pairs(doc, seen))

        return candidates

    # -------------------------------------------------------------------------
    # Strategy 1: Label-Value Extraction
    # -------------------------------------------------------------------------

    def _extract_label_values(
        self, doc: DocumentGraph, seen: Set[str]
    ) -> List[Candidate]:
        """Extract values that follow known label patterns."""
        candidates = []

        for block in doc.iter_linear_blocks(skip_header_footer=False):
            text = block.text or ""
            if not text.strip():
                continue

            for pattern in LABEL_PATTERNS:
                match = pattern.label_regex.search(text)
                if not match:
                    continue

                # Extract value after the label
                value_start = match.end()
                remaining = text[value_start:].strip()

                # Take first line or up to max length
                value = remaining.split("\n")[0][: self.max_value_length].strip()

                # STRICT VALIDATION to prevent hallucinations
                if not self._is_valid_extracted_value(value, pattern.entity_type):
                    continue

                # Dedupe
                key = f"{pattern.entity_type}:{value}"
                if self.dedupe and key in seen:
                    continue
                seen.add(key)

                candidates.append(
                    self._make_candidate(
                        doc=doc,
                        block=block,
                        value=value,
                        entity_type=pattern.entity_type,
                        rule_name=f"label_value::{pattern.name}",
                        confidence=pattern.confidence,
                        context=text[:200],
                    )
                )

        return candidates

    def _is_valid_extracted_value(self, value: str, entity_type: str) -> bool:
        """
        Strict validation to filter out noise/hallucinations.
        IDs should be short, specific tokens - not sentence fragments.
        """
        if not value or len(value) < 2:
            return False

        # Max word count (IDs are rarely > 5 words)
        word_count = len(value.split())
        if word_count > 6:
            return False

        # Max character length
        if len(value) > 60:
            return False

        # Reject if starts with lowercase (likely sentence fragment)
        if value[0].islower():
            return False

        # Reject common sentence starters
        sentence_starters = [
            "the ",
            "a ",
            "an ",
            "this ",
            "that ",
            "these ",
            "those ",
            "it ",
            "we ",
            "they ",
            "he ",
            "she ",
            "was ",
            "were ",
            "is ",
            "are ",
            "has ",
            "have ",
            "had ",
            "will ",
            "would ",
            "could ",
            "should ",
            "may ",
            "might ",
            "must ",
            "can ",
            "documented ",
            "reported ",
            "showed ",
            "demonstrated ",
            "indicated ",
            "found ",
        ]
        value_lower = value.lower()
        if any(value_lower.startswith(s) for s in sentence_starters):
            return False

        # Reject if contains too much punctuation (likely prose)
        punct_count = sum(1 for c in value if c in ".,;:!?")
        if punct_count > 2:
            return False

        # Entity-specific validation
        if entity_type in ["PROTOCOL_ID", "STUDY_ID"]:
            # Should contain alphanumeric pattern
            import re

            if not re.search(r"[A-Z]{2,}[-_]?\d+|^\d{4,}", value):
                # Doesn't look like an ID pattern
                if word_count > 3:
                    return False

        return True

    # -------------------------------------------------------------------------
    # Strategy 2: Zone Extraction
    # -------------------------------------------------------------------------

    def _extract_from_zones(
        self, doc: DocumentGraph, seen: Set[str]
    ) -> List[Candidate]:
        """Extract data from specific page zones (first page only for now)."""
        candidates = []

        # Only process first page for zone extraction
        if 1 not in doc.pages:
            return candidates

        page = doc.pages[1]
        page_blocks = [b for b in page.blocks if b.bbox]

        if not page_blocks:
            return candidates

        # Estimate page dimensions from block positions
        all_x1 = [_bbox_coords(b.bbox)[2] for b in page_blocks]
        all_y1 = [_bbox_coords(b.bbox)[3] for b in page_blocks]
        page_width = max(all_x1) if all_x1 else 612  # Default letter width
        page_height = max(all_y1) if all_y1 else 792  # Default letter height

        # Define zone boundaries
        margin_x = page_width * self.zone_margin
        margin_y = page_height * self.zone_margin

        # Check blocks in top-right zone (often contains protocol ID, version)
        for block in page_blocks:
            x0, y0, x1, y1 = _bbox_coords(block.bbox)

            # Top-right zone
            if x0 > (page_width - margin_x * 2) and y0 < margin_y * 2:
                text = _clean_ws(block.text)
                if text and len(text) > 2:
                    key = f"ZONE_TOP_RIGHT:{text}"
                    if key not in seen:
                        seen.add(key)
                        candidates.append(
                            self._make_candidate(
                                doc=doc,
                                block=block,
                                value=text,
                                entity_type="HEADER_INFO",
                                rule_name="zone::top_right",
                                confidence=0.75,
                                context=text,
                            )
                        )

        return candidates

    # -------------------------------------------------------------------------
    # Strategy 3: Glossary Tables
    # -------------------------------------------------------------------------

    def _extract_glossary_tables(
        self, doc: DocumentGraph, seen: Set[str]
    ) -> List[Candidate]:
        """Extract from structured glossary tables."""
        candidates = []

        for table in doc.iter_tables(table_type=TableType.GLOSSARY):
            for sf, lf, sf_cell, lf_cell in table.iter_glossary_pairs():
                sf_clean = _clean_ws(sf).upper()
                lf_clean = _clean_ws(lf)

                if not sf_clean or not lf_clean:
                    continue
                if len(sf_clean) > self.max_sf_len:
                    continue

                key = f"GLOSSARY:{sf_clean}:{lf_clean}"
                if self.dedupe and key in seen:
                    continue
                seen.add(key)

                loc = Coordinate(
                    page_num=int(table.page_num),
                    table_id=str(table.id),
                    cell_row=int(sf_cell.row_index) if sf_cell else None,
                    cell_col=int(sf_cell.col_index) if sf_cell else None,
                    bbox=sf_cell.bbox if sf_cell else table.bbox,
                )

                prov = ProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=self.doc_fingerprint_default,
                    generator_name=GeneratorType.GLOSSARY_TABLE,
                    rule_version="layout::glossary_table",
                )

                candidates.append(
                    Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.GLOSSARY_ENTRY,
                        generator_type=GeneratorType.GLOSSARY_TABLE,
                        short_form=sf_clean,
                        long_form=lf_clean,
                        context_text=table.to_markdown(max_rows=20),
                        context_location=loc,
                        initial_confidence=0.97,
                        provenance=prov,
                    )
                )

        return candidates

    # -------------------------------------------------------------------------
    # Strategy 4: Two-Column Alignment
    # -------------------------------------------------------------------------

    def _extract_column_pairs(
        self, doc: DocumentGraph, seen: Set[str]
    ) -> List[Candidate]:
        """Extract SF-LF pairs from two-column layouts in glossary sections."""
        candidates = []

        for pnum in sorted(doc.pages.keys()):
            page = doc.pages[pnum]

            # Check if page is in glossary section
            if not self._is_glossary_section(page.blocks):
                continue

            body_blocks = [
                b for b in page.blocks if b.role == ContentRole.BODY_TEXT and b.bbox
            ]
            if len(body_blocks) < 4:
                continue

            pairs = self._find_column_pairs(body_blocks)

            for sf_block, lf_block in pairs:
                sf_clean = _clean_ws(sf_block.text).upper()
                lf_clean = _clean_ws(lf_block.text)

                if not sf_clean or not lf_clean:
                    continue
                if len(sf_clean) > self.max_sf_len:
                    continue
                if not any(c.isupper() for c in sf_clean):
                    continue

                key = f"COLUMN:{sf_clean}:{lf_clean}"
                if self.dedupe and key in seen:
                    continue
                seen.add(key)

                loc = Coordinate(
                    page_num=int(sf_block.page_num),
                    block_id=str(sf_block.id),
                    bbox=sf_block.bbox,
                )

                prov = ProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=self.doc_fingerprint_default,
                    generator_name=GeneratorType.TABLE_LAYOUT,
                    rule_version="layout::column_alignment",
                )

                candidates.append(
                    Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.GLOSSARY_ENTRY,
                        generator_type=GeneratorType.TABLE_LAYOUT,
                        short_form=sf_clean,
                        long_form=lf_clean,
                        context_text=f"{sf_clean}: {lf_clean}",
                        context_location=loc,
                        initial_confidence=0.88,
                        provenance=prov,
                    )
                )

        return candidates

    def _is_glossary_section(self, blocks: List[TextBlock]) -> bool:
        """Check if page contains a glossary section header."""
        for block in blocks:
            if block.role != ContentRole.SECTION_HEADER:
                continue
            title = _clean_ws(block.text).lower()
            if any(h in title for h in self.glossary_headers):
                return True
        return False

    def _find_column_pairs(
        self, blocks: List[TextBlock]
    ) -> List[Tuple[TextBlock, TextBlock]]:
        """Find left-right pairs based on column alignment."""
        if not blocks:
            return []

        # Split by X midpoint
        x_coords = [_bbox_coords(b.bbox)[0] for b in blocks]
        mid_x = (min(x_coords) + max(x_coords)) / 2

        left = [b for b in blocks if _bbox_coords(b.bbox)[0] < mid_x]
        right = [b for b in blocks if _bbox_coords(b.bbox)[0] >= mid_x]

        if not left or not right:
            return []

        # Sort by Y
        left.sort(key=lambda b: _bbox_coords(b.bbox)[1])
        right.sort(key=lambda b: _bbox_coords(b.bbox)[1])

        # Pre-compute center Y values for right blocks (for binary search)
        right_centers = [_center_y(rb.bbox) for rb in right]

        # Greedy match by Y alignment using binary search (O(n log n) vs O(n²))
        pairs = []
        used: Set[int] = set()

        for lb in left:
            lcy = _center_y(lb.bbox)

            # Binary search to find range of candidates within tolerance
            lo = bisect.bisect_left(right_centers, lcy - self.row_tolerance)
            hi = bisect.bisect_right(right_centers, lcy + self.row_tolerance)

            # Find best unused match in the narrow range
            best_idx = -1
            best_dist = float("inf")

            for i in range(lo, hi):
                if i in used:
                    continue
                dist = abs(lcy - right_centers[i])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                used.add(best_idx)
                pairs.append((lb, right[best_idx]))

        return pairs

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block: TextBlock,
        value: str,
        entity_type: str,
        rule_name: str,
        confidence: float,
        context: str,
    ) -> Candidate:
        loc = Coordinate(
            page_num=int(block.page_num),
            block_id=str(block.id),
            bbox=block.bbox,
        )

        prov = ProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=self.doc_fingerprint_default,
            generator_name=self.generator_type,
            rule_version=rule_name,
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.SHORT_FORM_ONLY,
            generator_type=self.generator_type,
            short_form=value,
            long_form=entity_type,  # Store entity type for downstream
            context_text=context,
            context_location=loc,
            initial_confidence=confidence,
            provenance=prov,
        )
