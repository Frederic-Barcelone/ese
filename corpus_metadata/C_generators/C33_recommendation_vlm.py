"""
VLM-based extraction mixin for guideline recommendations.

This module provides mixin methods for vision-based extraction of evidence
levels and recommendation strengths from guideline table images. Handles
PDF rendering, image analysis, and correlation with text recommendations.

Key Components:
    - VLMExtractionMixin: Mixin class for VLM-based extraction
    - Methods:
        - extract_loe_sor_with_vlm: Extract LoE/SoR from table images
        - _render_pdf_page: Render PDF pages to images for VLM
        - _match_recommendations: Correlate VLM results with text
    - resize_image_for_vlm: Image preprocessing (from C15)

Example:
    >>> class MyExtractor(VLMExtractionMixin):
    ...     def __init__(self):
    ...         self.llm_client = anthropic.Anthropic()
    ...         self._vlm_loe_sor_cache = {}
    >>> extractor = MyExtractor()
    >>> loe_sor = extractor.extract_loe_sor_with_vlm(pdf_path, page_num)

Dependencies:
    - A_core.A18_recommendation_models: EvidenceLevel, RecommendationStrength
    - C_generators.C31_recommendation_patterns: VLM_LOE_SOR_EXTRACTION_PROMPT
    - C_generators.C15_vlm_table_extractor: resize_image_for_vlm
    - fitz (PyMuPDF): PDF page rendering (optional)
"""

from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

from A_core.A18_recommendation_models import (
    EvidenceLevel,
    RecommendationSet,
    RecommendationStrength,
)

from .C31_recommendation_patterns import VLM_LOE_SOR_EXTRACTION_PROMPT
from .C15_vlm_table_extractor import resize_image_for_vlm


# Optional PyMuPDF for page image rendering
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


class VLMExtractionMixin:
    """
    Mixin class providing VLM-based extraction methods.

    Designed to be mixed into GuidelineRecommendationExtractor to provide
    vision-based extraction of LoE/SoR codes from table images.

    Assumes the host class has:
    - self.llm_client: LLM API client
    - self.llm_model: Model identifier string
    - self._vlm_loe_sor_cache: Dict for VLM cache
    - self.pdf_path: Optional PDF path
    - self._clean_json_response(): JSON cleaning method
    - self._loe_code_to_level(): LoE conversion method
    - self._sor_code_to_strength(): SoR conversion method
    """

    # Runtime attribute with type annotation (used as cache)
    _vlm_loe_sor_cache: Dict[str, Tuple[str, str, str, List[str]]]

    # Declare expected attributes from host class for type checking
    if TYPE_CHECKING:
        pdf_path: Optional[str]
        llm_client: Any
        llm_model: str
        _clean_json_response: Any
        _sor_code_to_strength: Any
        _loe_code_to_level: Any

    def extract_loe_sor_with_vlm(self, text: str) -> Dict[str, Tuple[str, str, str, List[str]]]:
        """
        Extract LoE/SoR codes from recommendation table using VLM.

        When text-based extraction fails due to garbled PDF parsing,
        this method renders the relevant pages as images and uses
        vision LLM to extract the table structure.

        Args:
            text: Document text (used to identify table pages)

        Returns:
            Dict mapping recommendation number to (loe_code, sor_code, text_snippet, keywords) tuple
        """
        if not self.pdf_path or not self.llm_client:
            return {}

        if not PYMUPDF_AVAILABLE or fitz is None:
            return {}

        # Return cached results if available
        if self._vlm_loe_sor_cache:
            return self._vlm_loe_sor_cache

        # Find pages containing recommendation tables
        table_pages = self._find_recommendation_table_pages(text)
        if not table_pages:
            return {}

        codes: Dict[str, Tuple[str, str, str, List[str]]] = {}

        for page_num in table_pages:
            # Render page as image
            img_base64 = self._render_page_as_image(page_num)
            if not img_base64:
                continue

            # Extract LoE/SoR using VLM
            page_codes = self._extract_loe_sor_from_image(img_base64)
            for rec_num, (loe, sor, text_snippet, keywords) in page_codes.items():
                if rec_num not in codes:
                    codes[rec_num] = (loe, sor, text_snippet, keywords)

        self._vlm_loe_sor_cache = codes
        if codes:
            logger.info("VLM: Extracted %d LoE/SoR codes with text snippets", len(codes))
            for rec_num, (loe, sor, text_snippet, keywords) in sorted(codes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                snippet_preview = text_snippet[:50] + "..." if text_snippet else "(no text)"
                kw_str = ", ".join(keywords[:3]) if keywords else "(no keywords)"
                logger.debug("Rec %s: LoE=%s, SoR=%s, text=%r, kw=[%s]", rec_num, loe, sor, snippet_preview, kw_str)
        return codes

    def _find_recommendation_table_pages(self, text: str) -> List[int]:
        """
        Find page numbers containing recommendation tables with LoE/SoR.

        Args:
            text: Document text with page markers

        Returns:
            List of 1-indexed page numbers
        """
        pages = []

        # Split text by page markers
        page_pattern = re.compile(r'---\s*Page\s+(\d+)\s*---', re.IGNORECASE)
        page_matches = list(page_pattern.finditer(text))

        for i, match in enumerate(page_matches):
            page_num = int(match.group(1))
            start = match.end()
            end = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(text)
            page_text = text[start:end]

            # Check if this page has recommendation table content
            # Match: "Table X recommendation", "Table X ... LoE SoR", "Table X Continued",
            # or pages with numbered recommendations that have LoE/SoR codes
            if re.search(r'Table\s+\d+.*(?:recommendation|LoE\s+SoR|Continued)', page_text, re.IGNORECASE):
                pages.append(page_num)
            # Also check for numbered recommendations with embedded LoE/SoR codes
            elif re.search(r'\d{1,2}\s+(?:In\s+patients|For\s+patients|We\s+recommend).*?[1-5][ab]?\s+[ABCD]\s+\d{2,3}', page_text, re.IGNORECASE):
                pages.append(page_num)

        return pages

    def _render_page_as_image(self, page_num: int, dpi: int = 150) -> Optional[str]:
        """
        Render a PDF page as base64-encoded PNG image.

        Args:
            page_num: 1-indexed page number
            dpi: Resolution for rendering

        Returns:
            Base64-encoded PNG string, or None if fails
        """
        if not PYMUPDF_AVAILABLE or fitz is None or not self.pdf_path:
            return None

        try:
            doc = fitz.open(self.pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None

            page = doc[page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            doc.close()
            return img_base64

        except Exception as e:
            logger.warning("Failed to render page %d: %s", page_num, e)
            return None

    def _extract_loe_sor_from_image(self, img_base64: str) -> Dict[str, Tuple[str, str, str, List[str]]]:
        """
        Extract LoE/SoR codes, text snippets, and keywords from a table image using VLM.

        Args:
            img_base64: Base64-encoded PNG image

        Returns:
            Dict mapping recommendation number to (loe_code, sor_code, text_snippet, keywords) tuple
        """
        if not self.llm_client or not img_base64:
            return {}

        try:
            # Resize image if needed to fit Claude Vision limits
            resized_image, _ = resize_image_for_vlm(img_base64)

            # Call VLM with the image
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": resized_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": VLM_LOE_SOR_EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
            )

            if not response.content:
                return {}

            # Parse JSON response
            import json
            text_response = response.content[0].text.strip()
            text_response = self._clean_json_response(text_response)

            # Handle empty response
            if not text_response or not text_response.strip():
                logger.debug("VLM returned empty text response for LoE/SoR extraction")
                return {}

            # Parse JSON with error handling
            try:
                data = json.loads(text_response)
            except json.JSONDecodeError as je:
                logger.debug("VLM returned invalid JSON for LoE/SoR: %s", je)
                return {}

            # Ensure data is a dict
            if not isinstance(data, dict):
                logger.debug("VLM returned non-dict for LoE/SoR: %s", type(data).__name__)
                return {}

            codes: Dict[str, Tuple[str, str, str, List[str]]] = {}
            for rec in data.get("recommendations", []):
                rec_num = str(rec.get("rec_num", ""))
                loe = rec.get("loe", "").lower()
                sor = rec.get("sor", "").upper()
                text_snippet = rec.get("text", "")
                keywords = rec.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                if rec_num and (loe or sor):
                    codes[rec_num] = (loe, sor, text_snippet, keywords)

            return codes

        except Exception as e:
            logger.warning("VLM LoE/SoR extraction failed: %s", e)
            return {}

    def _apply_vlm_codes_to_recommendations(
        self,
        rec_set: RecommendationSet,
    ) -> None:
        """
        Apply VLM-extracted LoE/SoR codes to recommendations using text and keyword matching.

        This post-processing step matches recommendations by text similarity
        and keyword overlap, handling cases where LLM extraction creates
        different recommendation groupings than the PDF table.

        Args:
            rec_set: RecommendationSet to update (modified in place)
        """
        if not self._vlm_loe_sor_cache:
            return

        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r'[\s\-–—]+', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()

        def get_content_words(text: str) -> set:
            """Extract meaningful content words, filtering out common words."""
            if not text:
                return set()
            # Common words to ignore in matching
            stopwords = {
                'the', 'a', 'an', 'in', 'of', 'for', 'with', 'and', 'or', 'to',
                'is', 'are', 'be', 'we', 'recommend', 'should', 'may', 'can',
                'patients', 'patient', 'treatment', 'disease', 'therapy',
                'new', 'onset', 'relapsing', 'that', 'this', 'as', 'by', 'on'
            }
            words = set(normalize_text(text).split())
            return words - stopwords

        def text_similarity(text1: str, text2: str) -> float:
            """Calculate word overlap similarity using content words."""
            words1 = get_content_words(text1)
            words2 = get_content_words(text2)
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            return intersection / min(len(words1), len(words2))

        def keyword_match_score(keywords: List[str], text: str) -> float:
            """Calculate how many keywords appear in the text."""
            if not keywords or not text:
                return 0.0
            text_lower = text.lower()
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            return matches / len(keywords) if keywords else 0.0

        # Build list of VLM entries for matching
        vlm_entries = []
        for rec_num, (loe_code, sor_code, text_snippet, keywords) in self._vlm_loe_sor_cache.items():
            vlm_entries.append({
                'rec_num': rec_num,
                'loe_code': loe_code,
                'sor_code': sor_code,
                'text_snippet': text_snippet,
                'keywords': keywords or [],
                'matched': False,
            })

        # Match each recommendation to best VLM entry by combined text + keyword similarity
        for rec in rec_set.recommendations:
            rec_text = rec.source_text or rec.action or ""
            if not rec_text:
                continue

            best_match = None
            best_score = 0.0
            best_text_score = 0.0
            best_kw_score = 0.0

            for entry in vlm_entries:
                if entry['matched']:
                    continue

                snippet = str(entry['text_snippet'])
                keywords = cast(List[str], entry['keywords'])

                # Calculate text similarity (using more of the text)
                text_score = text_similarity(snippet, rec_text[:150]) if snippet else 0.0

                # Calculate keyword match score
                kw_score = keyword_match_score(keywords, rec_text)

                # Combined score: weighted average favoring keywords when available
                if keywords:
                    combined_score = 0.4 * text_score + 0.6 * kw_score
                else:
                    combined_score = text_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_text_score = text_score
                    best_kw_score = kw_score
                    best_match = entry

            # Apply match if score is above threshold
            # Lower threshold (0.25) since we now use keyword matching
            if best_match and best_score >= 0.25:
                best_match['matched'] = True
                loe_code = str(best_match['loe_code'])
                sor_code = str(best_match['sor_code'])

                # Debug output
                match_detail = f"txt={best_text_score:.2f}"
                if best_match['keywords']:
                    match_detail += f", kw={best_kw_score:.2f}"
                logger.debug("VLM-MATCH: Rec '%s...' -> PDF rec %s (score=%.2f, %s)", rec_text[:40], best_match['rec_num'], best_score, match_detail)

                # Apply LoE from VLM - VLM reads actual PDF table, so it's authoritative
                # For matches above 0.5, always apply VLM codes
                # For lower scores (0.25-0.5), only apply if current level is uncertain
                new_evidence = self._loe_code_to_level(loe_code)
                should_update_evidence = (
                    new_evidence != EvidenceLevel.UNKNOWN and (
                        best_score >= 0.5 or  # Good confidence - trust VLM
                        rec.evidence_level in (EvidenceLevel.UNKNOWN, EvidenceLevel.EXPERT_OPINION)
                    )
                )
                if should_update_evidence and new_evidence != rec.evidence_level:
                    logger.debug("Updated evidence: %s -> %s (from LoE=%s)", rec.evidence_level.value, new_evidence.value, loe_code)
                    rec.evidence_level = new_evidence

                # Apply SoR from VLM with same logic
                new_strength = self._sor_code_to_strength(sor_code)
                should_update_strength = (
                    new_strength != RecommendationStrength.UNKNOWN and (
                        best_score >= 0.5 or
                        rec.strength == RecommendationStrength.UNKNOWN
                    )
                )
                if should_update_strength and new_strength != rec.strength:
                    logger.debug("Updated strength: %s -> %s (from SoR=%s)", rec.strength.value, new_strength.value, sor_code)
                    rec.strength = new_strength


__all__ = ["VLMExtractionMixin", "PYMUPDF_AVAILABLE"]
