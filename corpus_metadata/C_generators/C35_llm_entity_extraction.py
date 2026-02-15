"""
LLM-based gap-fill entity extraction.

Uses Claude Haiku to find diseases and drugs not captured by lexicon/NER
detection layers. Given the document text and already-detected entities,
asks the LLM to identify additional mentions.

Key Components:
    - LLMEntityGapFiller: Main class for LLM gap-fill extraction
    - find_missing_diseases(): Find disease mentions not yet captured
    - find_missing_drugs(): Find drug mentions not yet captured

Dependencies:
    - D_validation.D02_llm_engine: ClaudeClient
    - Z_utils.Z13_llm_tracking: CallType
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, TYPE_CHECKING

from A_core.A00_logging import get_logger
from Z_utils.Z13_llm_tracking import CallType

if TYPE_CHECKING:
    from D_validation.D02_llm_engine import ClaudeClient

logger = get_logger(__name__)

_DISEASE_SYSTEM_PROMPT = """\
You are a biomedical entity extraction expert. Your task is to find disease \
mentions in clinical/medical text that are NOT already in the provided list.

Rules:
- Only return diseases that appear VERBATIM in the text
- Do not return diseases already in the "known" list
- Include the disease name exactly as it appears in text
- Include rare diseases, syndromes, conditions, and disorders
- Do NOT include symptoms, signs, or lab findings (e.g., "fever", "elevated CRP")
- Do NOT include gene names or drug names
- Do NOT include generic terms like "disease", "disorder", "condition"
- Return a JSON array of objects: [{"name": "disease name as in text"}]
- If no additional diseases found, return an empty array: []
"""

_DRUG_SYSTEM_PROMPT = """\
You are a biomedical entity extraction expert. Your task is to find drug \
and chemical compound mentions in clinical/medical text that are NOT already \
in the provided list.

Rules:
- Only return drugs/chemicals that appear VERBATIM in the text
- Do not return drugs already in the "known" list
- Include brand names, generic names, and chemical compound names
- Include abbreviations of drug names (e.g., "5-FU", "MTX")
- Do NOT include drug classes (e.g., "antibiotics", "NSAIDs")
- Do NOT include biological entities (e.g., "insulin receptor", "TNF-alpha")
- Do NOT include disease names or gene names
- Return a JSON array of objects: [{"name": "drug name as in text"}]
- If no additional drugs found, return an empty array: []
"""


class LLMEntityGapFiller:
    """Find entities missed by lexicon/NER detection using LLM."""

    def __init__(self, claude_client: "ClaudeClient", max_text_chars: int = 4000) -> None:
        self.client = claude_client
        self.max_text_chars = max_text_chars

    def find_missing_diseases(
        self,
        text: str,
        known_diseases: List[str],
    ) -> List[Dict[str, str]]:
        """Ask LLM to find disease mentions not in the known list.

        Returns list of {"name": "disease name"} dicts.
        """
        truncated = text[:self.max_text_chars]
        known_str = ", ".join(sorted(set(known_diseases))) if known_diseases else "(none)"

        user_prompt = (
            f"Already detected diseases: {known_str}\n\n"
            f"Text:\n{truncated}\n\n"
            f"List additional disease mentions NOT in the above list."
        )

        try:
            result = self.client.complete_json_any(
                system_prompt=_DISEASE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.0,
                call_type=CallType.ENTITY_GAP_FILL,
            )
        except Exception as e:
            logger.warning("LLM gap-fill disease extraction failed: %s", e)
            return []

        return self._parse_result(result, text)

    def find_missing_drugs(
        self,
        text: str,
        known_drugs: List[str],
    ) -> List[Dict[str, str]]:
        """Ask LLM to find drug mentions not in the known list.

        Returns list of {"name": "drug name"} dicts.
        """
        truncated = text[:self.max_text_chars]
        known_str = ", ".join(sorted(set(known_drugs))) if known_drugs else "(none)"

        user_prompt = (
            f"Already detected drugs: {known_str}\n\n"
            f"Text:\n{truncated}\n\n"
            f"List additional drug/chemical mentions NOT in the above list."
        )

        try:
            result = self.client.complete_json_any(
                system_prompt=_DRUG_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.0,
                call_type=CallType.ENTITY_GAP_FILL,
            )
        except Exception as e:
            logger.warning("LLM gap-fill drug extraction failed: %s", e)
            return []

        return self._parse_result(result, text)

    def _parse_result(
        self, result: Any, text: str,
    ) -> List[Dict[str, str]]:
        """Parse and validate LLM gap-fill results."""
        if not isinstance(result, list):
            return []

        validated: List[Dict[str, str]] = []
        text_lower = text.lower()

        for item in result:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name or len(name) < 3:
                continue

            # Verify the entity appears verbatim in text (case-insensitive)
            escaped = re.escape(name.lower())
            if not re.search(rf"\b{escaped}\b", text_lower):
                # Try exact substring match as fallback
                if name.lower() not in text_lower:
                    continue

            validated.append({"name": name})

        return validated
