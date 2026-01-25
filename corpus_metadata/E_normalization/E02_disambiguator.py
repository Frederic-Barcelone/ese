# corpus_metadata/corpus_metadata/E_normalization/E02_disambiguator.py

from __future__ import annotations

import re
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
            "SSRS": {
                "Columbia Suicide Severity Rating Scale": [
                    "suicide",
                    "suicidal",
                    "ideation",
                    "depression",
                    "psychiatric",
                    "mental",
                    "safety",
                    "c-ssrs",
                ],
                "Simple Sequence Repeats": [
                    "microsatellite",
                    "genetic",
                    "dna",
                    "polymorphism",
                    "marker",
                    "pcr",
                ],
            },
            "ET": {
                "Early Termination": [
                    "discontinuation",
                    "withdrawal",
                    "dropout",
                    "visit",
                    "protocol",
                    "study",
                ],
                "Essential Thrombocythemia": [
                    "thrombocytosis",
                    "platelet",
                    "myeloproliferative",
                    "jak2",
                    "blood",
                ],
            },
            "MSD": {
                "Merck Sharp & Dohme": [
                    "merck",
                    "pharmaceutical",
                    "drug",
                    "clinical",
                    "trial",
                    "regulatory",
                    "fda",
                    "ema",
                    "approval",
                    "marketing",
                    "press",
                    "release",
                    "company",
                ],
                "Multiple sulfatase deficiency": [
                    "sulfatase",
                    "enzyme",
                    "lysosomal",
                    "mucopolysaccharidosis",
                    "genetic",
                    "metabolic",
                    "storage",
                ],
            },
            # ========================================
            # NEW: Guideline-specific ambiguous SFs
            # ========================================
            "ACR": {
                "American College of Rheumatology": [
                    "eular",
                    "rheumatology",
                    "guideline",
                    "recommendation",
                    "criteria",
                    "classification",
                    "rheumatoid",
                    "arthritis",
                    "lupus",
                    "vasculitis",
                    "spondyloarthritis",
                    "anca",
                    "gpa",
                    "mpa",
                    "egpa",
                    "aav",
                ],
                "Albumin-to-Creatinine Ratio": [
                    "albumin",
                    "urine",
                    "creatinine",
                    "kidney",
                    "nephropathy",
                    "proteinuria",
                    "uacr",
                    "microalbuminuria",
                ],
            },
            "LOA": {
                "Level of Agreement": [
                    "guideline",
                    "consensus",
                    "delphi",
                    "voting",
                    "agreement",
                    "panel",
                    "recommendation",
                    "statement",
                    "expert",
                ],
                "Leave of Absence": [
                    "employee",
                    "hr",
                    "work",
                    "personnel",
                    "leave",
                    "absence",
                ],
            },
            "FV": {
                "Final Vote": [
                    "guideline",
                    "consensus",
                    "voting",
                    "panel",
                    "delphi",
                    "recommendation",
                    "agreement",
                ],
                "Femoral Vein": [
                    "vein",
                    "venous",
                    "catheter",
                    "vascular",
                    "thrombosis",
                    "femoral",
                ],
            },
            "AI": {
                "Artificial Intelligence": [
                    "machine",
                    "learning",
                    "algorithm",
                    "deep",
                    "neural",
                    "network",
                    "model",
                    "prediction",
                    "computer",
                ],
                "Autoimmune": [
                    "autoimmune",
                    "immune",
                    "autoantibody",
                    "inflammation",
                    "disease",
                    "disorder",
                ],
                "Aortic Insufficiency": [
                    "aortic",
                    "valve",
                    "regurgitation",
                    "cardiac",
                    "heart",
                    "echocardiography",
                ],
            },
            "LOE": {
                "Level of Evidence": [
                    "guideline",
                    "recommendation",
                    "evidence",
                    "grade",
                    "quality",
                    "strength",
                    "systematic",
                    "review",
                ],
                "Loss of Expression": [
                    "gene",
                    "mutation",
                    "protein",
                    "expression",
                    "loss",
                    "tumor",
                ],
            },
            "SOC": {
                "Standard of Care": [
                    "trial",
                    "clinical",
                    "treatment",
                    "therapy",
                    "placebo",
                    "comparator",
                    "arm",
                    "randomized",
                    "control",
                    "intervention",
                    "patient",
                    "efficacy",
                    "safety",
                ],
                "System Organ Class": [
                    "adverse",
                    "event",
                    "meddra",
                    "safety",
                    "teae",
                    "ae",
                    "serious",
                ],
                "Superior Olivary Complex": [
                    "auditory",
                    "brainstem",
                    "cochlear",
                    "hearing",
                    "neuron",
                    "acoustic",
                ],
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

    def re_disambiguate_with_context(
        self, entities: List[ExtractedEntity], full_doc_text: str
    ) -> List[ExtractedEntity]:
        """
        Re-disambiguate entities that already have long_forms from lexicons.

        This handles the case where a lexicon (like UMLS) provides a wrong expansion
        for an ambiguous SF. For example:
        - ACR from UMLS might be "albumin-to-creatinine ratio" but in a rheumatology
          guideline context, it should be "American College of Rheumatology"
        - FV from UMLS might be "Immunoglobulin Fv Fragments" but in a guideline
          context, it should be "Final Vote"

        Args:
            entities: extracted entities (may have long_forms from lexicons)
            full_doc_text: document-level text for context scoring

        Returns:
            list of entities, with some lexicon LFs replaced if context suggests different meaning
        """
        profile = self._profile_document(full_doc_text)
        out: List[ExtractedEntity] = []

        for e in entities:
            # Skip non-validated
            if self.only_validated and e.status != ValidationStatus.VALIDATED:
                out.append(e)
                continue

            sf = (e.short_form or "").strip()
            if not sf:
                out.append(e)
                continue

            sf_key = sf.upper()
            options = self.ambiguity_map.get(sf_key)
            if not options:
                # SF not in ambiguity map, keep as-is
                out.append(e)
                continue

            # Get current long_form (might be None or from lexicon)
            current_lf = (e.long_form or "").strip()

            # Decide best meaning based on context
            decision = self._decide_meaning(sf_key, profile)
            if not decision:
                out.append(e)
                continue

            chosen_lf, score, runner_up_score = decision

            # Check if current LF matches one of the known meanings
            current_lf_lower = current_lf.lower()
            chosen_lf_lower = chosen_lf.lower()

            # If current LF is already the best match, no change needed
            if self._lf_matches(current_lf_lower, chosen_lf_lower):
                out.append(e)
                continue

            # Check if current LF matches ANY known meaning in ambiguity_map
            current_matches_any_known = False
            for meaning_lf in options.keys():
                if self._lf_matches(current_lf_lower, meaning_lf.lower()):
                    current_matches_any_known = True
                    break

            # Override if:
            # 1. Current LF is empty, OR
            # 2. Current LF matches a known wrong meaning (not the best), OR
            # 3. Current LF doesn't match ANY known meaning (e.g., "Immunoglobulin Fv Fragments" for FV)
            #    AND context strongly suggests a specific meaning
            should_override = (
                not current_lf or  # Empty
                current_matches_any_known or  # Known wrong meaning
                (not current_matches_any_known and score >= self.min_context_score * 2)  # Unknown LF but strong context
            )

            if not should_override:
                out.append(e)
                continue

            # Override with contextually correct meaning
            updates: Dict[str, Any] = {}
            flags = list(e.validation_flags or [])

            payload = {
                "re_disambiguation": {
                    "method": "context_override",
                    "short_form": sf_key,
                    "original_long_form": current_lf or None,
                    "chosen_long_form": chosen_lf,
                    "score": score,
                    "runner_up_score": runner_up_score,
                }
            }

            merged_norm = self._merge_normalized_value(e.normalized_value, payload)
            updates["normalized_value"] = merged_norm

            if "re_disambiguated" not in flags:
                flags.append("re_disambiguated")
            updates["validation_flags"] = flags

            # Replace long_form with contextually correct one
            updates["long_form"] = chosen_lf

            out.append(e.model_copy(update=updates))

        return out

    def _lf_matches(self, lf1: str, lf2: str) -> bool:
        """
        Check if two long forms match, allowing for minor variations.
        E.g., "albumin to creatinine ratio" matches "albumin-to-creatinine ratio"
        """
        # Normalize: lowercase, remove hyphens/punctuation, collapse spaces
        def normalize(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        return normalize(lf1) == normalize(lf2)

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

    def flag_unexpanded(
        self,
        entities: List[ExtractedEntity],
        whitelist_unexpanded: Optional[List[str]] = None,
    ) -> List[ExtractedEntity]:
        """
        Flag unexpanded abbreviations as AMBIGUOUS unless whitelisted.

        Abbreviations with no long_form (SF-only) may be ambiguous if they
        weren't expanded. Unless they're on a whitelist (known unambiguous SFs),
        we flag them as AMBIGUOUS to alert downstream consumers.

        Args:
            entities: List of extracted entities to check
            whitelist_unexpanded: List of SFs that are allowed to remain
                                  unexpanded without being flagged as AMBIGUOUS.
                                  These are typically unambiguous domain-specific SFs.

        Returns:
            Updated list of entities with AMBIGUOUS flags added where appropriate
        """
        whitelist = set((sf.upper() for sf in (whitelist_unexpanded or [])))

        out: List[ExtractedEntity] = []
        for e in entities:
            # Only check validated SF-only entities
            if e.status != ValidationStatus.VALIDATED:
                out.append(e)
                continue

            # Skip if has a long form
            if e.long_form and e.long_form.strip():
                out.append(e)
                continue

            sf = (e.short_form or "").strip().upper()
            if not sf:
                out.append(e)
                continue

            # Check whitelist
            if sf in whitelist:
                out.append(e)
                continue

            # Flag as ambiguous
            flags = list(e.validation_flags or [])
            flags.append("unexpanded_ambiguous")

            updated = e.model_copy(
                update={
                    "status": ValidationStatus.AMBIGUOUS,
                    "validation_flags": flags,
                    "rejection_reason": "No expansion found; may be ambiguous",
                }
            )
            out.append(updated)

        return out

    def decide_meaning_with_context(
        self,
        sf: str,
        local_context: str,
        global_profile: Optional[Counter] = None,
        proximity_window: int = 50,
    ) -> Optional[Tuple[str, float]]:
        """
        Decide meaning using proximity-weighted context scoring.

        Words closer to the SF get higher weight than distant words.
        This helps disambiguate when the local context strongly suggests
        one meaning even if the global document profile doesn't.

        Args:
            sf: Short form to disambiguate
            local_context: Text surrounding the SF occurrence
            global_profile: Optional pre-computed global document word profile
            proximity_window: Number of characters around SF to weight highly

        Returns:
            (best_meaning, confidence_score) or None if ambiguous
        """
        sf_key = sf.upper()
        options = self.ambiguity_map.get(sf_key)
        if not options:
            return None

        # Build weighted profile from local context
        local_lower = local_context.lower() if self.lowercase else local_context

        scored: List[Tuple[str, float]] = []
        for meaning, keywords in options.items():
            score = 0.0

            for kw in keywords:
                k = kw.lower() if self.lowercase else kw

                # Check local context with proximity weighting
                if k in local_lower:
                    # Find position relative to SF
                    sf_pos = local_lower.find(sf.lower())
                    kw_pos = local_lower.find(k)

                    if sf_pos >= 0 and kw_pos >= 0:
                        distance = abs(kw_pos - sf_pos)
                        # Higher weight for closer words
                        if distance <= proximity_window:
                            weight = 2.0 - (distance / proximity_window)  # 2.0 to 1.0
                        else:
                            weight = 0.5  # Further away
                        score += weight
                    else:
                        score += 1.0  # Default weight

                # Also check global profile if provided
                if global_profile and k in global_profile:
                    score += global_profile[k] * 0.3  # Lower weight for global

            scored.append((meaning, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_meaning, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0

        # Confidence based on margin
        if best_score < self.min_context_score:
            return None

        margin = best_score - second_score
        if margin < self.min_margin:
            return None

        # Normalize confidence to 0-1 range
        confidence = min(1.0, margin / 5.0)

        return best_meaning, confidence
