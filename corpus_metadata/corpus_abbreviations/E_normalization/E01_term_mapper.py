# corpus_metadata/corpus_abbreviations/E_normalization/E01_term_mapper.py

from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from typing import Any, Dict, Optional, List, Tuple

from A_core.A02_interfaces import BaseNormalizer
from A_core.A01_domain_models import (
    ExtractedEntity,
    ValidationStatus,
    FieldType,
)

# -------------------------
# TermMapper (Normalization)
# -------------------------


class TermMapper(BaseNormalizer):
    """
    Abbreviation-only normalizer.

    It does NOT replace the core extraction/verification logic.
    It only:
      1) Canonicalizes the long form (LF) when present
      2) Attaches standard_id (optional)
      3) Stores a normalization payload in normalized_value

    IMPORTANT:
      - By default, does NOT invent long_form for SHORT_FORM_ONLY.
      - You can enable filling LF from lexicon if you explicitly want it.
    """

    def __init__(self, config: dict):
        self.config = config or {}

        # Processed lexicon output (you said you'll manage lexicon processing later)
        # Expected to be a SINGLE file: abbreviation_lexicon.json
        self.mapping_file = self.config.get(
            "mapping_file_path", "abbreviation_lexicon.json"
        )

        # Optional fuzzy matching
        self.enable_fuzzy = bool(self.config.get("enable_fuzzy_matching", False))
        self.fuzzy_cutoff = float(self.config.get("fuzzy_cutoff", 0.90))

        # Controls for SHORT_FORM_ONLY behavior
        self.fill_long_form_for_orphans = bool(
            self.config.get("fill_long_form_for_orphans", False)
        )

        # Load lookup table (normalized_key -> {canonical_long_form, standard_id, ...})
        self.lookup_table: Dict[str, Dict[str, Any]] = self._load_mappings(
            self.mapping_file
        )
        self.valid_keys: List[str] = list(self.lookup_table.keys())

    # -------------------------
    # Public API
    # -------------------------

    def normalize(self, entity: ExtractedEntity) -> ExtractedEntity:
        """
        Normalizes a validated entity.
        - For DEFINITION_PAIR / GLOSSARY_ENTRY: normalize entity.long_form
        - For SHORT_FORM_ONLY: attach metadata; optionally fill long_form if enabled
        """
        if entity.status != ValidationStatus.VALIDATED:
            return entity

        # Decide what we want to map
        to_map, map_kind = self._select_term_to_map(entity)
        if not to_map:
            return entity

        norm_key = self._preprocess(to_map)
        match = self.lookup_table.get(norm_key)

        # Fuzzy fallback
        if not match and self.enable_fuzzy:
            match = self._fuzzy_lookup(norm_key)

        if not match:
            return entity

        # Build normalized payload (audit-friendly)
        payload = {
            "normalization_source": "term_mapper",
            "mapped_from": map_kind,  # "long_form" | "short_form"
            "original_term": to_map,
            "normalized_key": norm_key,
            "canonical_long_form": match.get("canonical_long_form")
            or match.get("name"),
            "standard_id": match.get("standard_id")
            or match.get("code")
            or match.get("id"),
        }

        updates: Dict[str, Any] = {}

        # Always attach normalized_value + standard_id if present
        updates["normalized_value"] = payload
        if payload.get("standard_id"):
            updates["standard_id"] = payload["standard_id"]

        # Canonicalize long_form (only when LF exists OR explicitly allowed for orphans)
        canonical_lf = payload.get("canonical_long_form")
        if canonical_lf:
            if entity.long_form:
                # Normal case: definition/glossary
                updates["long_form"] = canonical_lf
            else:
                # Orphan case: only if you explicitly allow filling
                if (
                    entity.field_type == FieldType.SHORT_FORM_ONLY
                    and self.fill_long_form_for_orphans
                ):
                    updates["long_form"] = canonical_lf

        # Add a small flag (keep the original flags too)
        new_flags = list(entity.validation_flags or [])
        if "normalized" not in new_flags:
            new_flags.append("normalized")
        updates["validation_flags"] = new_flags

        return entity.model_copy(update=updates)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _select_term_to_map(self, entity: ExtractedEntity) -> Tuple[Optional[str], str]:
        """
        Returns (term_to_map, kind).
        For abbreviations:
          - prefer long_form when present (definition/glossary)
          - fallback to short_form for orphans
        """
        if entity.long_form and entity.field_type in (
            FieldType.DEFINITION_PAIR,
            FieldType.GLOSSARY_ENTRY,
        ):
            return entity.long_form, "long_form"

        # Orphan / fallback
        if entity.short_form:
            return entity.short_form, "short_form"

        return None, "none"

    def _preprocess(self, text: str) -> str:
        """
        Aggressive but safe normalization for dictionary keys.
        """
        s = (text or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        # remove surrounding punctuation (not internal)
        s = s.strip(" \t\r\n\"'`.,;:()[]{}")
        return s

    def _fuzzy_lookup(self, norm_key: str) -> Optional[Dict[str, Any]]:
        if not self.valid_keys:
            return None
        matches = get_close_matches(
            norm_key, self.valid_keys, n=1, cutoff=self.fuzzy_cutoff
        )
        if matches:
            return self.lookup_table.get(matches[0])
        return None

    def _load_mappings(self, path: str) -> Dict[str, Dict[str, Any]]:
        """
        Loads the processed lexicon output.

        Supported shapes:
        A) Dict[str, Dict]: {
             "tnf": {"canonical_long_form": "Tumor Necrosis Factor", "standard_id": "NCI:C18247"},
             "tumor necrosis factor": {"canonical_long_form": "Tumor Necrosis Factor", "standard_id": "NCI:C18247"}
           }

        B) List[Dict]: [
             {"key": "tnf", "canonical_long_form": "...", "standard_id": "..."},
             ...
           ]

        C) Backward compatible keys: name/code/id
        """
        if not path or not os.path.exists(path):
            # Keep empty by default (you said lexicon processing will be managed later)
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN]  TermMapper: error loading mapping file {path}: {e}")
            return {}

        out: Dict[str, Dict[str, Any]] = {}

        # Case A: dict
        if isinstance(data, dict):
            for k, v in data.items():
                key = self._preprocess(str(k))
                if not key:
                    continue
                entry = self._coerce_entry(v)
                if entry:
                    out[key] = entry
            return out

        # Case B: list
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                raw_key = (
                    item.get("key")
                    or item.get("normalized_key")
                    or item.get("sf")
                    or item.get("term")
                )
                key = self._preprocess(str(raw_key)) if raw_key else ""
                if not key:
                    continue
                entry = self._coerce_entry(item)
                if entry:
                    out[key] = entry
            return out

        return {}

    def _coerce_entry(self, obj: Any) -> Optional[Dict[str, Any]]:
        """
        Tries to extract canonical_long_form + standard_id from different entry shapes.
        """
        if not isinstance(obj, dict):
            return None

        canonical = (
            obj.get("canonical_long_form")
            or obj.get("canonical_expansion")
            or obj.get("name")
            or obj.get("canonical")
        )

        standard_id = obj.get("standard_id") or obj.get("code") or obj.get("id")

        # Allow extra fields, but keep a consistent minimal shape
        entry: Dict[str, Any] = {}
        if canonical:
            entry["canonical_long_form"] = str(canonical).strip()
        if standard_id:
            entry["standard_id"] = str(standard_id).strip()

        # carry optional provenance-ish fields (safe)
        if "source" in obj:
            entry["source"] = obj["source"]
        if "sources" in obj:
            entry["sources"] = obj["sources"]

        return entry or None
