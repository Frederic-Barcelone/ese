# corpus_metadata/C_generators/C18_strategy_pharma.py
"""
Pharmaceutical company mention detection.

Uses FlashText lexicon matching against pharma_companies_lexicon.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from flashtext import KeywordProcessor

from A_core.A01_domain_models import Coordinate, EvidenceSpan, ValidationStatus
from A_core.A03_provenance import generate_run_id, get_git_revision_hash
from A_core.A09_pharma_models import (
    ExtractedPharma,
    PharmaCandidate,
    PharmaExportDocument,
    PharmaExportEntry,
    PharmaGeneratorType,
    PharmaProvenanceMetadata,
)
from B_parsing.B02_doc_graph import DocumentGraph


class PharmaCompanyDetector:
    """
    Detect pharmaceutical company mentions using lexicon matching.

    Uses FlashText for fast keyword matching against pharma_companies_lexicon.json.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("PHARMA"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )

        # Lexicon paths
        self.lexicon_base_path = Path(
            self.config.get("lexicon_base_path", "ouput_datasources")
        )

        # FlashText processor
        self.pharma_processor: Optional[KeywordProcessor] = None
        self.pharma_data: Dict[str, Dict[str, Any]] = {}

        # Track loaded stats
        self._loaded_count = 0

        # Initialize
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        """Load pharma companies lexicon."""
        path = self.lexicon_base_path / "pharma_companies_lexicon.json"
        if not path.exists():
            logger.warning("Pharma lexicon not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            processor = KeywordProcessor(case_sensitive=False)
            self.pharma_processor = processor

            for key, info in data.items():
                if not isinstance(info, dict):
                    continue

                canonical = info.get("canonical_name", key)
                full_name = info.get("full_name", "")
                variants = info.get("variants", [])

                # Store company data
                self.pharma_data[key.lower()] = {
                    "key": key,
                    "canonical_name": canonical,
                    "full_name": full_name,
                    "headquarters": info.get("headquarters"),
                    "parent": info.get("parent"),
                    "subsidiaries": info.get("subsidiaries", []),
                    "source": info.get("source"),
                }

                # Add all variants to FlashText
                for variant in variants:
                    if variant and len(variant) >= 2:
                        processor.add_keyword(variant, key.lower())

                # Also add canonical and full name
                if canonical and len(canonical) >= 2:
                    processor.add_keyword(canonical, key.lower())
                if full_name and len(full_name) >= 3:
                    processor.add_keyword(full_name, key.lower())

                self._loaded_count += 1

        except Exception as e:
            logger.warning("Failed to load pharma lexicon: %s", e)

    def detect(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[PharmaCandidate]:
        """
        Detect pharma company mentions in document.

        Args:
            doc_graph: Document graph
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Optional pre-built full text

        Returns:
            List of PharmaCandidate objects
        """
        if not self.pharma_processor:
            return []

        candidates: List[PharmaCandidate] = []
        seen_companies: Set[str] = set()

        # Build full text if not provided
        if not full_text:
            full_text = " ".join(
                block.text
                for block in doc_graph.iter_linear_blocks()
                if block.text
            )

        if not full_text:
            return candidates

        # Find all pharma company mentions
        matches = self.pharma_processor.extract_keywords(full_text, span_info=True)

        for match_key, start, end in matches:
            # Deduplicate by company key
            if match_key in seen_companies:
                continue
            seen_companies.add(match_key)

            company_info = self.pharma_data.get(match_key, {})
            if not company_info:
                continue

            matched_text = full_text[start:end]
            context_start = max(0, start - 100)
            context_end = min(len(full_text), end + 100)
            context_text = full_text[context_start:context_end]

            # Create provenance
            provenance = PharmaProvenanceMetadata(
                pipeline_version=self.pipeline_version,
                run_id=self.run_id,
                doc_fingerprint=doc_fingerprint,
                generator_name=PharmaGeneratorType.LEXICON_MATCH,
                lexicon_source="pharma_companies_lexicon.json",
            )

            # Create candidate
            candidate = PharmaCandidate(
                doc_id=doc_id,
                matched_text=matched_text,
                canonical_name=company_info.get("canonical_name", match_key),
                full_name=company_info.get("full_name"),
                headquarters=company_info.get("headquarters"),
                parent_company=company_info.get("parent"),
                subsidiaries=company_info.get("subsidiaries", []),
                generator_type=PharmaGeneratorType.LEXICON_MATCH,
                context_text=context_text,
                context_location=Coordinate(page_num=1, block_id="unknown"),
                initial_confidence=0.95,
                provenance=provenance,
            )

            candidates.append(candidate)

        return candidates

    def validate_candidates(
        self, candidates: List[PharmaCandidate]
    ) -> List[ExtractedPharma]:
        """
        Validate pharma candidates (simple pass-through for now).

        Since these are lexicon matches, we auto-validate them.
        """
        validated: List[ExtractedPharma] = []

        for candidate in candidates:
            evidence = EvidenceSpan(
                text=candidate.matched_text,
                location=candidate.context_location,
                scope_ref="pharma_detection",
                start_char_offset=0,
                end_char_offset=len(candidate.matched_text),
            )

            extracted = ExtractedPharma(
                candidate_id=candidate.id,
                doc_id=candidate.doc_id,
                matched_text=candidate.matched_text,
                canonical_name=candidate.canonical_name,
                full_name=candidate.full_name,
                headquarters=candidate.headquarters,
                parent_company=candidate.parent_company,
                subsidiaries=candidate.subsidiaries,
                primary_evidence=evidence,
                status=ValidationStatus.VALIDATED,
                confidence_score=candidate.initial_confidence,
                validation_flags=["lexicon_match"],
                provenance=candidate.provenance,
            )

            validated.append(extracted)

        return validated

    def export_to_json(
        self,
        validated: List[ExtractedPharma],
        doc_id: str,
        doc_path: Optional[str] = None,
    ) -> PharmaExportDocument:
        """
        Create export document for pharma companies.
        """
        entries = []
        unique_companies = set()

        for pharma in validated:
            unique_companies.add(pharma.canonical_name)

            entry = PharmaExportEntry(
                matched_text=pharma.matched_text,
                canonical_name=pharma.canonical_name,
                full_name=pharma.full_name,
                headquarters=pharma.headquarters,
                parent_company=pharma.parent_company,
                subsidiaries=pharma.subsidiaries,
                confidence=pharma.confidence_score,
                context=pharma.primary_evidence.text,
                page=pharma.primary_evidence.location.page_num,
                lexicon_source="pharma_companies_lexicon.json",
            )
            entries.append(entry)

        return PharmaExportDocument(
            run_id=self.run_id,
            timestamp=self.config.get("timestamp", ""),
            document=doc_id,
            document_path=doc_path,
            pipeline_version=self.pipeline_version,
            total_detected=len(validated),
            unique_companies=len(unique_companies),
            companies=entries,
        )

    def get_loaded_count(self) -> int:
        """Return number of loaded pharma companies."""
        return self._loaded_count

    def print_summary(self) -> None:
        """Print loading summary."""
        logger.info("Pharma companies lexicon: %d companies loaded", self._loaded_count)
