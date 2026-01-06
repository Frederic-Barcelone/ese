# corpus_metadata/corpus_abbreviations/E_normalization/E02_disambiguator.py

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from A_core.A01_domain_models import (
    ExtractedEntity,
    ValidationStatus,
    FieldType,
)


class Disambiguator:
    """
    Abbreviation-only disambiguator.

    Goal:
      Resolve ambiguous SHORT_FORM_ONLY entities (orphans) into a best long_form
      using global document context (bag-of-words voting).

    What it updates (when confident):
      - entity.long_form
      - entity.normalized_value (adds a disambiguation payload)
      - entity.validation_flags (adds 'disambiguated')

    What it does NOT do:
      - It does not change entity.status (still VALIDATED if it was VALIDATED)
      - It does not modify provenance (audit trail stays intact)
    """

    def __init__(self, config: dict):
        self.config = config or {}

        # Minimum score needed to accept a meaning
        self.min_context_score: int = int(self.config.get("min_context_score", 2))

        # Require a margin over runner-up to avoid weak wins
        self.min_margin: int = int(self.config.get("min_margin", 1))

        # Only disambiguate if the entity is validated + orphan
        self.only_validated: bool = bool(self.config.get("only_validated", True))

        # Whether we are allowed to fill long_form for SHORT_FORM_ONLY entities
        self.fill_long_form: bool = bool(
            self.config.get("fill_long_form_for_orphans", True)
        )

        # Basic tokenization config
        self.lowercase: bool = bool(self.config.get("lowercase", True))

        # In production: load from JSON. Here: inline defaults (same as your sample)
        self.ambiguity_map: Dict[str, Dict[str, List[str]]] = self.config.get(
            "ambiguity_map"
        ) or {
            "MS": {
                "Multiple Sclerosis": [
                    "relapse",
                    "remission",
                    "neurology",
                    "brain",
                    "lesion",
                    "edss",
                ],
                "Mass Spectrometry": [
                    "chromatography",
                    "ion",
                    "mass",
                    "charge",
                    "spectrum",
                    "lc-ms",
                ],
                "Medical Services": ["admin", "hospital", "provider", "insurance"],
            },
            "PD": {
                "Pharmacodynamics": [
                    "pk",
                    "pharmacokinetics",
                    "drug",
                    "concentration",
                    "auc",
                ],
                "Parkinson's Disease": ["tremor", "motor", "dopamine", "neurology"],
                "Progressive Disease": [
                    "recist",
                    "tumor",
                    "oncology",
                    "cancer",
                    "response",
                ],
            },
            "AE": {
                "Adverse Event": ["safety", "toxicity", "grade", "serious"],
                "Anti-Epileptic": ["seizure", "drug", "epilepsy"],
            },
        }

    # -------------------------
    # Public API
    # -------------------------

    def resolve(
        self, entities: List[ExtractedEntity], full_doc_text: str
    ) -> List[ExtractedEntity]:
        """
        Resolve ambiguous orphans based on global document context.

        Args:
            entities: extracted entities (already verified)
            full_doc_text: document-level text (ideally from DocumentGraph)

        Returns:
            list of entities, with some orphans upgraded with a chosen long_form
        """
        profile = self._profile_document(full_doc_text)
        out: List[ExtractedEntity] = []

        for e in entities:
            if not self._should_attempt(e):
                out.append(e)
                continue

            sf = (e.short_form or "").strip()
            if not sf:
                out.append(e)
                continue

            sf_key = sf.upper()
            options = self.ambiguity_map.get(sf_key)
            if not options:
                out.append(e)
                continue

            decision = self._decide_meaning(sf_key, profile)
            if not decision:
                out.append(e)
                continue

            chosen_lf, score, runner_up_score = decision

            updates: Dict[str, Any] = {}
            flags = list(e.validation_flags or [])

            # Attach a structured explanation
            payload = {
                "disambiguation": {
                    "method": "global_context_voting",
                    "short_form": sf_key,
                    "chosen_long_form": chosen_lf,
                    "score": score,
                    "runner_up_score": runner_up_score,
                    "min_context_score": self.min_context_score,
                    "min_margin": self.min_margin,
                }
            }

            # Merge into normalized_value safely (could be str|dict|None)
            merged_norm = self._merge_normalized_value(e.normalized_value, payload)
            updates["normalized_value"] = merged_norm

            if "disambiguated" not in flags:
                flags.append("disambiguated")
            updates["validation_flags"] = flags

            # Only fill LF if allowed and LF is currently empty
            if self.fill_long_form and not e.long_form:
                updates["long_form"] = chosen_lf

            out.append(e.model_copy(update=updates))

        return out

    # -------------------------
    # Internal helpers
    # -------------------------

    def _should_attempt(self, e: ExtractedEntity) -> bool:
        if self.only_validated and e.status != ValidationStatus.VALIDATED:
            return False
        if e.field_type != FieldType.SHORT_FORM_ONLY:
            return False
        # Only orphans (no LF)
        if e.long_form:
            return False
        return True

    def _profile_document(self, text: str) -> Counter:
        """
        Create a bag-of-words profile.
        Very fast; good enough for theme detection.
        """
        if not text:
            return Counter()

        t = text
        if self.lowercase:
            t = t.lower()

        # light tokenization: split on non-alnum except hyphen
        tokens = []
        buf = []
        for ch in t:
            if ch.isalnum() or ch == "-":
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf))
                    buf = []
        if buf:
            tokens.append("".join(buf))

        return Counter(tokens)

    def _decide_meaning(
        self, sf_key: str, profile: Counter
    ) -> Optional[Tuple[str, int, int]]:
        """
        Returns (best_meaning, best_score, second_best_score) if confident, else None.
        """
        options = self.ambiguity_map.get(sf_key, {})
        if not options or not profile:
            return None

        scored: List[Tuple[str, int]] = []
        for meaning, keywords in options.items():
            score = 0
            for kw in keywords:
                k = kw.lower() if self.lowercase else kw
                score += int(profile.get(k, 0))
            scored.append((meaning, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_meaning, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0

        # Thresholding
        if best_score < self.min_context_score:
            return None
        if (best_score - second_score) < self.min_margin:
            return None

        return best_meaning, best_score, second_score

    def _merge_normalized_value(self, existing: Any, patch: Dict[str, Any]) -> Any:
        """
        normalized_value can be: None | str | dict.
        We prefer dict; if existing is str, we wrap it.
        """
        if existing is None:
            return patch
        if isinstance(existing, dict):
            merged = dict(existing)
            # shallow merge
            for k, v in patch.items():
                merged[k] = v
            return merged
        # if it's a string or other type
        return {"previous_normalized_value": existing, **patch}
