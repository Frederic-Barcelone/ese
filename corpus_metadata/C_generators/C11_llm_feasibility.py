"""
LLM-based clinical trial feasibility extraction using Claude.

This module uses Claude to extract structured feasibility data from clinical
trial documents, replacing noisy regex-based extraction with high-quality
structured output. Features section-targeted prompting and quote verification.

Key Components:
    - LLMFeasibilityExtractor: Main extractor using Claude API
    - FeasibilityResponseParserMixin: Parses LLM JSON responses (C30)
    - Extraction categories:
        1. Study design (phase, sample size, randomization, arms)
        2. Eligibility criteria (structured inclusion/exclusion)
        3. Endpoints (primary, secondary with measures/timepoints)
        4. Sites/Countries (full country list)
        5. Operational burden (procedures, vaccinations, visit schedule)
        6. Screening flow (CONSORT data, screen fail reasons)

Example:
    >>> from C_generators.C11_llm_feasibility import LLMFeasibilityExtractor
    >>> extractor = LLMFeasibilityExtractor(config={"model": "claude-sonnet-4-5-20250929"})
    >>> candidates = extractor.extract(doc_graph, "doc_123", "fingerprint")
    >>> for c in candidates:
    ...     print(f"{c.field_type}: {c.value}")
    STUDY_PHASE: Phase 3
    SAMPLE_SIZE: 500

Dependencies:
    - A_core.A03_provenance: Provenance tracking utilities
    - A_core.A07_feasibility_models: FeasibilityCandidate, FeasibilityGeneratorType
    - B_parsing.B02_doc_graph: DocumentGraph
    - B_parsing.B05_section_detector: Section classification
    - D_validation.D04_quote_verifier: Quote and numerical verification
    - C_generators.C29_feasibility_prompts: LLM prompt templates
    - C_generators.C30_feasibility_response_parser: Response parsing mixin
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from A_core.A03_provenance import generate_run_id, get_git_revision_hash

logger = logging.getLogger(__name__)
from A_core.A07_feasibility_models import (
    FeasibilityCandidate,
    FeasibilityGeneratorType,
    FeasibilityProvenanceMetadata,
)
from A_core.A23_doc_graph_models import DocumentGraph
from B_parsing.B05_section_detector import SectionDetector
from Z_utils.Z14_quote_verifier import (
    ExtractionVerifier,
    QuoteVerifier,
    NumericalVerifier,
)

from .C29_feasibility_prompts import (
    SECTION_TARGETS,
    MAX_SECTION_CHARS,
    MAX_TOTAL_CHARS,
)
from .C30_feasibility_response_parser import FeasibilityResponseParserMixin


class LLMFeasibilityExtractor(FeasibilityResponseParserMixin):
    """
    LLM-based feasibility information extractor.

    Uses Claude to extract structured feasibility data with high precision.
    Runs all 9 extraction types in parallel for efficiency.
    """

    def __init__(
        self,
        llm_client: Any,
        llm_model: str = "claude-sonnet-4-5-20250929",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.config = config or {}
        self.run_id = str(self.config.get("run_id") or generate_run_id("FEAS_LLM"))
        self.pipeline_version = (
            self.config.get("pipeline_version") or get_git_revision_hash()
        )

        # Section detector for targeted extraction
        self.section_detector = SectionDetector()

        # Anti-hallucination verification
        self.quote_verifier = QuoteVerifier(fuzzy_threshold=0.90)
        self.numerical_verifier = NumericalVerifier()
        self.extraction_verifier = ExtractionVerifier(
            fuzzy_threshold=0.90,
            numerical_tolerance=0.0,
        )

        # Stats
        self._extraction_stats: Dict[str, int] = {}
        self._verification_stats: Dict[str, int] = {
            "quotes_verified": 0,
            "quotes_failed": 0,
            "numbers_verified": 0,
            "numbers_failed": 0,
            "missing_fields": 0,
        }

    def extract(
        self,
        doc_graph: DocumentGraph,
        doc_id: str,
        doc_fingerprint: str,
        full_text: Optional[str] = None,
    ) -> List[FeasibilityCandidate]:
        """
        Extract feasibility information using LLM with section-targeted extraction.

        Runs all 9 extraction types in parallel for efficiency.

        Args:
            doc_graph: Document graph for section-aware extraction
            doc_id: Document identifier
            doc_fingerprint: Document fingerprint
            full_text: Pre-built full text (fallback if doc_graph unavailable)

        Returns list of FeasibilityCandidate with structured data.
        """
        candidates: List[FeasibilityCandidate] = []

        # Build section map from doc_graph
        section_map = self._build_section_map(doc_graph)

        # Define extraction tasks: (method, extraction_type)
        extraction_tasks = [
            (self._extract_study_design, "study_design"),
            (self._extract_eligibility, "eligibility"),
            (self._extract_endpoints, "endpoints"),
            (self._extract_sites, "sites"),
            (self._extract_operational_burden, "operational_burden"),
            (self._extract_screening_flow, "screening_flow"),
            (self._extract_epidemiology, "epidemiology"),
            (self._extract_patient_population, "patient_population"),
            (self._extract_local_guidelines, "local_guidelines"),
            (self._extract_patient_journey, "patient_journey"),
        ]

        # Run all extractions in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for method, extraction_type in extraction_tasks:
                content = self._get_targeted_content(extraction_type, section_map, full_text)
                future = executor.submit(method, content, doc_id, doc_fingerprint)
                futures[future] = extraction_type

            # Collect results as they complete
            for future in as_completed(futures):
                extraction_type = futures[future]
                try:
                    result = future.result()
                    candidates.extend(result)
                except Exception as e:
                    logger.warning("LLM extraction failed for %s: %s", extraction_type, e)

        return candidates

    def _build_section_map(self, doc_graph: DocumentGraph) -> Dict[str, str]:
        """
        Build a map of section names to their text content with page markers.

        Returns dict like {"abstract": "[PAGE 1] text...", "methods": "[PAGE 2] text...", ...}
        Page markers are included so LLM can report page numbers with evidence quotes.
        """
        section_map: Dict[str, List[str]] = {}
        current_section = "preamble"
        current_page: Optional[int] = None

        for block in doc_graph.iter_linear_blocks():
            if not block.text:
                continue

            # Check if this block is a section header
            section_info = self.section_detector.detect(block.text, block)
            if section_info:
                current_section = section_info.name

            # Add text to current section
            if current_section not in section_map:
                section_map[current_section] = []
                current_page = None  # Reset page tracking for new section

            # Add page marker when page changes
            block_page = getattr(block, "page_num", None)
            if block_page and block_page != current_page:
                section_map[current_section].append(f"[PAGE {block_page}]")
                current_page = block_page

            section_map[current_section].append(block.text)

        # Join text for each section
        return {
            section: " ".join(texts)[:MAX_SECTION_CHARS]
            for section, texts in section_map.items()
        }

    def _get_targeted_content(
        self,
        extraction_type: str,
        section_map: Dict[str, str],
        fallback_text: Optional[str] = None,
    ) -> str:
        """
        Get content targeted for a specific extraction type.

        Combines relevant sections up to MAX_TOTAL_CHARS.
        """
        target_sections = SECTION_TARGETS.get(extraction_type, ["abstract", "methods"])
        content_parts = []
        total_chars = 0

        for section in target_sections:
            if section in section_map:
                section_text = section_map[section]
                if total_chars + len(section_text) <= MAX_TOTAL_CHARS:
                    content_parts.append(f"[{section.upper()}]\n{section_text}")
                    total_chars += len(section_text)
                else:
                    # Add partial section to fill remaining space
                    remaining = MAX_TOTAL_CHARS - total_chars
                    if remaining > 500:  # Only add if meaningful amount left
                        content_parts.append(f"[{section.upper()}]\n{section_text[:remaining]}")
                    break

        if content_parts:
            return "\n\n".join(content_parts)

        # Fallback to full text if no sections matched
        if fallback_text:
            return fallback_text[:MAX_TOTAL_CHARS]

        return ""

    def _call_llm(self, system_prompt: str, content: str) -> Optional[Dict[str, Any]]:
        """Call LLM and parse JSON response."""
        if not self.llm_client:
            return None

        user_prompt = f"Document content:\n\n{content}\n\n---\nExtract the requested information. Return JSON only."

        try:
            if hasattr(self.llm_client, "complete_json_any"):
                response = self.llm_client.complete_json_any(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=2000,
                    call_type="feasibility_extraction",
                )
            else:
                response = self.llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.llm_model,
                    temperature=0.0,
                    max_tokens=2000,
                    call_type="feasibility_extraction",
                )

            if isinstance(response, list) and len(response) > 0:
                response = response[0]

            return response if isinstance(response, dict) else None
        except Exception as e:
            logger.warning("LLM feasibility extraction failed: %s", e)
            return None

    def _make_provenance(self, doc_fingerprint: str) -> FeasibilityProvenanceMetadata:
        """Create provenance metadata."""
        return FeasibilityProvenanceMetadata(
            pipeline_version=self.pipeline_version,
            run_id=self.run_id,
            doc_fingerprint=doc_fingerprint,
            generator_name=FeasibilityGeneratorType.LLM_EXTRACTION,
        )

    def _verify_quote(self, quote: Optional[str], context: str) -> bool:
        """
        Verify that a quote exists in the context.

        Returns True if verified, False otherwise.
        Updates verification stats.
        """
        if not quote or not context:
            return False

        result = self.quote_verifier.verify(quote, context)
        if result.verified:
            self._verification_stats["quotes_verified"] += 1
        else:
            self._verification_stats["quotes_failed"] += 1
        return result.verified

    def _verify_number(self, value: Optional[int | float | str], context: str) -> bool:
        """
        Verify that a numerical value exists in the context.

        Returns True if verified, False otherwise.
        Updates verification stats.
        """
        if value is None or not context:
            return False

        # Handle case where value is not a number (e.g., dict, list)
        if not isinstance(value, (int, float, str)):
            self._verification_stats["numbers_failed"] += 1
            return False

        result = self.numerical_verifier.verify(value, context)
        if result.verified:
            self._verification_stats["numbers_verified"] += 1
        else:
            self._verification_stats["numbers_failed"] += 1
        return result.verified

    def _apply_verification_penalty(
        self,
        base_confidence: float,
        quote: Optional[str],
        context: str,
        numerical_values: Optional[Dict[str, int | float]] = None,
    ) -> float:
        """
        Apply confidence penalties based on verification results.

        NO FALLBACKS: Missing or unverified evidence results in penalties.

        Args:
            base_confidence: Starting confidence score (0.0 - 1.0).
            quote: Quote to verify. If None/empty, penalty applies.
            context: Source document text.
            numerical_values: Dict of field -> value to verify.

        Returns:
            Adjusted confidence score with penalties applied.
            - Missing quote: confidence x 0.5
            - Unverified quote: confidence x 0.5
            - Unverified number: confidence x 0.7 (per unverified number)
        """
        confidence = base_confidence

        # Quote verification: missing or unverified quote = penalty
        if not quote or not quote.strip():
            # No quote provided - apply penalty
            self._verification_stats["quotes_failed"] += 1
            confidence *= 0.5
        elif not self._verify_quote(quote, context):
            # Quote provided but not found in context
            confidence *= 0.5

        # Verify numerical values if provided
        if numerical_values:
            for field_name, value in numerical_values.items():
                if value is not None and not self._verify_number(value, context):
                    confidence *= 0.7

        return round(confidence, 3)

    @staticmethod
    def _get_bool(data: Dict[str, Any], key: str, default: bool = False) -> bool:
        """Get boolean value, handling None from LLM responses."""
        val = data.get(key)
        return val if val is not None else default

    @staticmethod
    def _get_list(data: Dict[str, Any], key: str) -> list:
        """Get list value, handling None from LLM responses."""
        val = data.get(key)
        return val if val is not None else []

    def _check_required_fields(
        self,
        data: Dict[str, Any],
        required_fields: List[str],
        context_name: str,
    ) -> int:
        """
        Check for missing required fields and track them.

        Args:
            data: The data dict to check
            required_fields: List of field names that should be present
            context_name: Name for logging (e.g., "eligibility_criterion")

        Returns:
            Count of missing required fields
        """
        missing_count = 0
        for field in required_fields:
            value = data.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_count += 1
                self._verification_stats["missing_fields"] += 1

        return missing_count

    def _apply_completeness_penalty(
        self,
        base_confidence: float,
        missing_field_count: int,
        penalty_per_field: float = 0.05,
    ) -> float:
        """
        Apply confidence penalty for missing fields.

        Args:
            base_confidence: Starting confidence
            missing_field_count: Number of missing required fields
            penalty_per_field: Penalty per missing field (default 5%)

        Returns:
            Adjusted confidence (minimum 0.1)
        """
        if missing_field_count <= 0:
            return base_confidence

        penalty = missing_field_count * penalty_per_field
        return max(0.1, base_confidence - penalty)

    def get_stats(self) -> Dict[str, int]:
        """Return extraction statistics."""
        return self._extraction_stats.copy()

    def print_summary(self) -> None:
        """Print extraction summary."""
        if not self._extraction_stats:
            logger.info("LLM Feasibility extraction: No items found")
            return

        total = sum(self._extraction_stats.values())
        logger.info("LLM Feasibility extraction: %d items found", total)
        for field_type, count in sorted(self._extraction_stats.items()):
            logger.info("  %-40s %5d", field_type, count)

        # Log verification stats
        if any(self._verification_stats.values()):
            logger.info("Verification statistics:")
            quotes_verified = self._verification_stats.get("quotes_verified", 0)
            quotes_failed = self._verification_stats.get("quotes_failed", 0)
            numbers_verified = self._verification_stats.get("numbers_verified", 0)
            numbers_failed = self._verification_stats.get("numbers_failed", 0)

            total_quotes = quotes_verified + quotes_failed
            total_numbers = numbers_verified + numbers_failed

            if total_quotes > 0:
                quote_rate = quotes_verified / total_quotes * 100
                logger.info("  Quotes verified: %d/%d (%.1f%%)", quotes_verified, total_quotes, quote_rate)
            if total_numbers > 0:
                number_rate = numbers_verified / total_numbers * 100
                logger.info("  Numbers verified: %d/%d (%.1f%%)", numbers_verified, total_numbers, number_rate)

            missing_fields = self._verification_stats.get("missing_fields", 0)
            if missing_fields > 0:
                logger.info("  Missing required fields: %d (items skipped)", missing_fields)
