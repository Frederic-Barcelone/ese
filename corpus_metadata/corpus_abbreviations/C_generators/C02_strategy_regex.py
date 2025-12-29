# corpus_metadata/corpus_abbreviations/C_generators/C02_strategy_regex.py
"""
El Cazador de Patrones - The Pattern Hunter.

Rigid pattern matching for structured data with predictable formats.
Context-immune: extracts patterns regardless of surrounding text.

Targets:
  - Trial IDs: NCT01234567, EudraCT 2020-001234-56, ISRCTN12345678
  - Doses: 10 mg, 500mg, 2.5 mL
  - Dates: 2024-01-15, 15/01/2024, January 15, 2024
  - References: DOI, PMID, PMCID, URLs

Analogy: A barcode scanner. If it sees the right pattern, it beeps.
It doesn't care what product the barcode is on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern

from A_core.A01_domain_models import (
    Candidate,
    Coordinate,
    FieldType,
    GeneratorType,
    ProvenanceMetadata,
)
from A_core.A02_interfaces import BaseCandidateGenerator
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from B_parsing.B02_doc_graph import DocumentGraph


class ReferenceType(str, Enum):
    """Classification of reference sources."""
    UNIVERSAL = "universal"
    LITERATURE = "literature"
    PREPRINT = "preprint"
    INDEX = "index"
    REPOSITORY = "repository"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY = "regulatory"
    PATENT = "patent"
    GUIDELINE = "guideline"
    DATABASE = "database"
    PUBLISHER = "publisher"
    UNKNOWN = "unknown"


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


@dataclass
class PatternDef:
    """Definition of a rigid pattern to extract."""
    name: str
    pattern: Pattern
    entity_type: str  # e.g., "TRIAL_ID", "DOSE", "DATE"
    confidence: float = 0.95


# Pre-compiled patterns for pharma/clinical data
RIGID_PATTERNS: List[PatternDef] = [
    # Clinical Trial IDs
    PatternDef(
        name="nct_id",
        pattern=re.compile(r'\bNCT\d{8}\b', re.IGNORECASE),
        entity_type="TRIAL_ID",
        confidence=0.99,
    ),
    PatternDef(
        name="eudract_id",
        pattern=re.compile(r'\b\d{4}-\d{6}-\d{2}\b'),  # EudraCT format
        entity_type="TRIAL_ID",
        confidence=0.95,
    ),
    PatternDef(
        name="isrctn_id",
        pattern=re.compile(r'\bISRCTN\d{8}\b', re.IGNORECASE),
        entity_type="TRIAL_ID",
        confidence=0.99,
    ),
    PatternDef(
        name="ctis_id",
        pattern=re.compile(r'\b\d{4}-\d{6}-\d{2}-\d{2}\b'),  # CTIS format
        entity_type="TRIAL_ID",
        confidence=0.95,
    ),

    # Doses with compound units (e.g., mg/L, g/dL, mg/mmol)
    # Compound units FIRST to avoid partial matches
    PatternDef(
        name="dose_compound",
        pattern=re.compile(
            r'\b(\d+(?:\.\d+)?)\s*'
            r'(mg/(?:L|dL|mL|mmol|mol|kg|m²)|'
            r'g/(?:L|dL|mL|mol|kg)|'
            r'μg/(?:L|dL|mL|kg)|'
            r'mmol/(?:L|mol)|'
            r'μmol/(?:L|mol)|'
            r'IU/(?:L|mL)|'
            r'U/(?:L|mL)|'
            r'×\s*10[\^]?\d+/L)\b',
            re.IGNORECASE
        ),
        entity_type="CONCENTRATION",
        confidence=0.95,
    ),
    # Simple doses (number + single unit)
    PatternDef(
        name="dose_mg",
        pattern=re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|g|μg|mcg|ug)(?![/])\b', re.IGNORECASE),
        entity_type="DOSE",
        confidence=0.90,
    ),
    PatternDef(
        name="dose_ml",
        pattern=re.compile(r'\b(\d+(?:\.\d+)?)\s*(mL|L|μL|uL)(?![/])\b', re.IGNORECASE),
        entity_type="DOSE",
        confidence=0.90,
    ),
    PatternDef(
        name="dose_iu",
        pattern=re.compile(r'\b(\d+(?:\.\d+)?)\s*(IU|U)(?![/])\b'),
        entity_type="DOSE",
        confidence=0.90,
    ),
    PatternDef(
        name="dose_percent",
        pattern=re.compile(r'\b(\d+(?:\.\d+)?)\s*%\b'),
        entity_type="PERCENTAGE",
        confidence=0.85,
    ),

    # Dates (ISO format)
    PatternDef(
        name="date_iso",
        pattern=re.compile(r'\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'),
        entity_type="DATE",
        confidence=0.95,
    ),
    PatternDef(
        name="date_euro",
        pattern=re.compile(r'\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(20\d{2})\b'),
        entity_type="DATE",
        confidence=0.90,
    ),
    PatternDef(
        name="date_us",
        pattern=re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(20\d{2})\b'),
        entity_type="DATE",
        confidence=0.85,  # Ambiguous with euro format
    ),

    # Drug codes / compound IDs
    PatternDef(
        name="compound_id",
        pattern=re.compile(r'\b[A-Z]{2,4}-?\d{3,6}\b'),  # e.g., ABC-12345
        entity_type="COMPOUND_ID",
        confidence=0.80,
    ),

    # Reference identifiers
    PatternDef(
        name="doi",
        pattern=re.compile(r'\b10\.\d{4,}/[^\s\]>]+'),  # DOI: 10.1234/xxxxx
        entity_type="DOI",
        confidence=0.98,
    ),
    PatternDef(
        name="pmid",
        pattern=re.compile(r'\bPMID[:\s]*(\d{7,8})\b', re.IGNORECASE),
        entity_type="PMID",
        confidence=0.99,
    ),
    PatternDef(
        name="pmcid",
        pattern=re.compile(r'\bPMC\d{7,8}\b', re.IGNORECASE),
        entity_type="PMCID",
        confidence=0.99,
    ),
    PatternDef(
        name="url_https",
        pattern=re.compile(r'https?://[^\s\])<>"]+'),
        entity_type="URL",
        confidence=0.95,
    ),

    # Year extraction (for references)
    PatternDef(
        name="year_parens",
        pattern=re.compile(r'\((\d{4})\)'),  # (2024)
        entity_type="YEAR",
        confidence=0.80,
    ),
]


class RegexCandidateGenerator(BaseCandidateGenerator):
    """
    Rigid Pattern Matcher - extracts structured data by form, not meaning.

    Context-immune: "Do not administer 10mg" still extracts "10mg".
    The validation layer decides if it's negated.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Context window around match (chars)
        self.ctx_window = int(self.config.get("ctx_window", 60))

        # Which pattern types to extract (default: all)
        self.enabled_types = set(
            self.config.get("enabled_types", [
                "TRIAL_ID", "DOSE", "CONCENTRATION", "PERCENTAGE",
                "DATE", "COMPOUND_ID", "DOI", "PMID", "PMCID", "URL", "YEAR"
            ])
        )

        # Deduplicate by value (emit each unique value once)
        self.dedupe = bool(self.config.get("dedupe", True))

        # Provenance
        self.pipeline_version = str(self.config.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(self.config.get("run_id") or generate_run_id("REGEX"))
        self.doc_fingerprint_default = str(self.config.get("doc_fingerprint") or "unknown-doc-fingerprint")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.LEXICON_MATCH

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure
        candidates: List[Candidate] = []
        seen: set = set()

        # Filter patterns by enabled types
        active_patterns = [p for p in RIGID_PATTERNS if p.entity_type in self.enabled_types]

        for block in doc.iter_linear_blocks(skip_header_footer=True):
            text = block.text or ""
            if not text.strip():
                continue

            for pdef in active_patterns:
                for match in pdef.pattern.finditer(text):
                    value = match.group(0).strip()

                    # Dedupe
                    key = (pdef.entity_type, value.upper())
                    if self.dedupe and key in seen:
                        continue
                    seen.add(key)

                    # Context snippet
                    start, end = match.start(), match.end()
                    ctx_start = max(0, start - self.ctx_window)
                    ctx_end = min(len(text), end + self.ctx_window)
                    context = _clean_ws(text[ctx_start:ctx_end])

                    candidates.append(self._make_candidate(
                        doc=doc,
                        block=block,
                        value=value,
                        entity_type=pdef.entity_type,
                        pattern_name=pdef.name,
                        confidence=pdef.confidence,
                        context=context,
                    ))

        return candidates

    def _make_candidate(
        self,
        doc: DocumentGraph,
        block,
        value: str,
        entity_type: str,
        pattern_name: str,
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
            doc_fingerprint=str(self.config.get("doc_fingerprint") or self.doc_fingerprint_default),
            generator_name=self.generator_type,
            rule_version=f"regex::{pattern_name}",
        )

        return Candidate(
            doc_id=doc.doc_id,
            field_type=FieldType.SHORT_FORM_ONLY,  # No definition, just the value
            generator_type=self.generator_type,
            short_form=value,
            long_form=entity_type,  # Store entity type as "long_form" for downstream use
            context_text=context,
            context_location=loc,
            initial_confidence=confidence,
            provenance=prov,
        )
