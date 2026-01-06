# corpus_metadata/corpus_abbreviations/C_generators/C05_strategy_glossary.py
"""
Glossary Table Extractor - Extract SF/LF pairs from glossary tables.

Target: Tables with columns like "Abbreviation | Definition" or "Term | Meaning".

High confidence extractions since glossary tables are authoritative sources.
"""
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
from A_core.A03_provenance import (
    generate_run_id,
    get_git_revision_hash,
)
from B_parsing.B02_doc_graph import (
    DocumentGraph,
    TableType,
)


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


class GlossaryTableCandidateGenerator(BaseCandidateGenerator):
    """
    Extract SF/LF pairs from tables classified as GLOSSARY.
    Uses the Table dual-view (logical_rows + physical cells).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_rows_per_table = int(self.config.get("max_rows_per_table", 500))

        self.pipeline_version = str(self.config.get("pipeline_version") or get_git_revision_hash())
        self.run_id = str(self.config.get("run_id") or generate_run_id("ABBR"))
        self.doc_fingerprint_default = str(self.config.get("doc_fingerprint") or "unknown-doc-fingerprint")

    @property
    def generator_type(self) -> GeneratorType:
        return GeneratorType.GLOSSARY_TABLE

    def extract(self, doc_structure: DocumentGraph) -> List[Candidate]:
        doc = doc_structure
        out: List[Candidate] = []
        seen: Set[Tuple[str, str, str]] = set()

        for table in doc.iter_tables(table_type=TableType.GLOSSARY):
            count = 0
            for sf, lf, sf_cell, lf_cell in table.iter_glossary_pairs():
                if count >= self.max_rows_per_table:
                    break

                sf = _clean_ws(sf)
                lf = _clean_ws(lf)
                if not sf or not lf:
                    continue

                key = (doc.doc_id, sf.upper(), lf.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Prefer SF cell for pinpointing; fallback to table bbox
                bbox = (sf_cell.bbox if sf_cell else table.bbox)
                cell_row = int(sf_cell.row_index) if sf_cell else None
                cell_col = int(sf_cell.col_index) if sf_cell else None

                loc = Coordinate(
                    page_num=int(table.page_num),
                    table_id=str(table.id),
                    cell_row=cell_row,
                    cell_col=cell_col,
                    bbox=bbox,
                )

                prov = ProvenanceMetadata(
                    pipeline_version=self.pipeline_version,
                    run_id=self.run_id,
                    doc_fingerprint=str(self.config.get("doc_fingerprint") or self.doc_fingerprint_default),
                    generator_name=self.generator_type,
                    rule_version="glossary_table::v1",
                )

                # Context: small markdown table for LLM/debug
                ctx = table.to_markdown(max_rows=20)

                out.append(
                    Candidate(
                        doc_id=doc.doc_id,
                        field_type=FieldType.GLOSSARY_ENTRY,
                        generator_type=self.generator_type,
                        short_form=sf,
                        long_form=lf,
                        context_text=ctx,
                        context_location=loc,
                        initial_confidence=0.99,
                        provenance=prov,
                    )
                )

                count += 1

        return out