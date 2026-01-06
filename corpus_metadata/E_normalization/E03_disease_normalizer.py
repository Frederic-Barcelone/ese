# corpus_metadata/corpus_metadata/E_normalization/E03_disease_normalizer.py
"""
Disease-specific normalization.

Post-validation normalization for disease entities:
1. Extract primary codes from identifiers list (ICD-10, SNOMED, MONDO, ORPHA)
2. Categorize diseases by therapeutic area
3. Standardize disease names
"""

from __future__ import annotations

from typing import Dict, List, Optional

from A_core.A01_domain_models import ValidationStatus
from A_core.A05_disease_models import (
    DiseaseIdentifier,
    ExtractedDisease,
)


# -------------------------
# Disease Category Keywords
# -------------------------

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "nephrology": [
        "nephro",
        "kidney",
        "renal",
        "glomerul",
        "nephritic",
        "nephrotic",
        "uremia",
        "dialysis",
        "transplant kidney",
        "IgA nephropathy",
        "C3 glomerulopathy",
        "FSGS",
        "membranous",
    ],
    "pulmonology": [
        "pulmonary",
        "lung",
        "respiratory",
        "bronch",
        "pneumo",
        "asthma",
        "COPD",
        "fibrosis pulmonary",
        "PAH",
        "hypertension pulmonary",
    ],
    "cardiology": [
        "cardio",
        "heart",
        "cardiac",
        "coronary",
        "myocardial",
        "arrhythmia",
        "atrial",
        "ventricular",
        "valve",
        "aortic",
    ],
    "oncology": [
        "cancer",
        "tumor",
        "carcinoma",
        "lymphoma",
        "leukemia",
        "melanoma",
        "sarcoma",
        "neoplasm",
        "malignant",
        "metastatic",
        "oncolog",
    ],
    "neurology": [
        "neuro",
        "brain",
        "cerebral",
        "nervous",
        "alzheimer",
        "parkinson",
        "epilepsy",
        "seizure",
        "multiple sclerosis",
        "dementia",
        "stroke",
    ],
    "immunology": [
        "immune",
        "autoimmune",
        "immunodeficiency",
        "lupus",
        "rheumatoid",
        "vasculitis",
        "ANCA",
        "complement",
        "antibody",
    ],
    "hematology": [
        "hematolog",
        "blood",
        "anemia",
        "thrombocytopenia",
        "hemophilia",
        "coagulation",
        "platelet",
        "erythrocyte",
        "leukocyte",
    ],
    "endocrinology": [
        "endocrin",
        "diabetes",
        "thyroid",
        "adrenal",
        "pituitary",
        "hormone",
        "insulin",
        "metabolic",
    ],
    "gastroenterology": [
        "gastro",
        "intestin",
        "hepat",
        "liver",
        "bowel",
        "colitis",
        "crohn",
        "cirrhosis",
        "pancrea",
    ],
    "dermatology": [
        "dermato",
        "skin",
        "cutaneous",
        "psoriasis",
        "eczema",
        "dermatitis",
        "melanoma skin",
    ],
    "ophthalmology": [
        "ophthalm",
        "eye",
        "ocular",
        "retina",
        "macular",
        "glaucoma",
        "cataract",
    ],
    "rare_disease": [
        "orphan",
        "rare",
        "ultra-rare",
        "prevalence <1",
    ],
}


class DiseaseNormalizer:
    """
    Post-validation normalization for disease entities.

    Responsibilities:
    1. Extract primary codes from identifiers list
    2. Categorize diseases by therapeutic area
    3. Fill in missing code fields
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.category_keywords = CATEGORY_KEYWORDS

    def normalize(self, entity: ExtractedDisease) -> ExtractedDisease:
        """
        Normalize a validated disease entity.

        - Extracts primary codes (ICD-10, SNOMED, etc.) from identifiers
        - Categorizes by therapeutic area
        - Returns updated entity
        """
        if entity.status != ValidationStatus.VALIDATED:
            return entity

        updates: Dict = {}

        # Extract primary codes from identifiers
        code_updates = self._extract_primary_codes(entity.identifiers)
        updates.update(code_updates)

        # Categorize disease
        category = self._categorize_disease(
            entity.preferred_label,
            entity.abbreviation,
            entity.is_rare_disease,
        )
        if category:
            updates["disease_category"] = category

        # Apply updates if any
        if updates:
            return entity.model_copy(update=updates)

        return entity

    def normalize_batch(
        self, entities: List[ExtractedDisease]
    ) -> List[ExtractedDisease]:
        """Normalize a batch of disease entities."""
        return [self.normalize(e) for e in entities]

    def _extract_primary_codes(
        self, identifiers: List[DiseaseIdentifier]
    ) -> Dict[str, Optional[str]]:
        """
        Extract primary code for each code system from identifiers list.

        Maps system names to the primary code fields in ExtractedDisease.
        """
        updates: Dict[str, Optional[str]] = {}

        # Map of system names to field names
        system_to_field = {
            "ICD-10": "icd10_code",
            "ICD-10-CM": "icd10_code",  # Use same field for CM variant
            "ICD-11": "icd11_code",
            "SNOMED-CT": "snomed_code",
            "SNOMED": "snomed_code",
            "MONDO": "mondo_id",
            "ORPHA": "orpha_code",
            "Orphanet": "orpha_code",
            "UMLS": "umls_cui",
            "MeSH": "mesh_id",
        }

        for ident in identifiers:
            field = system_to_field.get(ident.system)
            if field and field not in updates:
                # Format code if needed
                code = ident.code
                if ident.system == "ORPHA" and not code.startswith("ORPHA:"):
                    code = f"ORPHA:{code}"
                elif ident.system == "MONDO" and not code.startswith("MONDO:"):
                    code = f"MONDO:{code}"
                updates[field] = code

        return updates

    def _categorize_disease(
        self,
        preferred_label: str,
        abbreviation: Optional[str],
        is_rare_disease: bool,
    ) -> Optional[str]:
        """
        Categorize disease by therapeutic area based on name/abbreviation.

        Returns the most likely category or None if undetermined.
        """
        # Combine label and abbreviation for matching
        search_text = preferred_label.lower()
        if abbreviation:
            search_text += " " + abbreviation.lower()

        # Score each category
        scores: Dict[str, int] = {}
        for category, keywords in self.category_keywords.items():
            score = 0
            for kw in keywords:
                if kw.lower() in search_text:
                    score += 1
            if score > 0:
                scores[category] = score

        # If rare disease flag is set, boost rare_disease category
        if is_rare_disease:
            scores["rare_disease"] = scores.get("rare_disease", 0) + 2

        if not scores:
            return None

        # Return highest scoring category
        # If rare_disease ties, prefer the more specific category
        sorted_categories = sorted(scores.items(), key=lambda x: -x[1])
        best_category = sorted_categories[0][0]

        # If best is rare_disease but there's another category with same score,
        # prefer the specific category
        if best_category == "rare_disease" and len(sorted_categories) > 1:
            second_best = sorted_categories[1]
            if second_best[1] >= scores["rare_disease"] - 2:  # Undo the boost
                best_category = second_best[0]

        return best_category

    def enrich_from_lexicon(
        self,
        entity: ExtractedDisease,
        lexicon_entry: Optional[dict],
    ) -> ExtractedDisease:
        """
        Enrich entity with additional data from lexicon entry.

        Can be used to fill in missing codes or metadata.
        """
        if not lexicon_entry:
            return entity

        updates: Dict = {}

        # Fill missing codes from lexicon
        code_mappings = {
            "ICD10": "icd10_code",
            "ICD10CM": "icd10_code",
            "ICD11": "icd11_code",
            "SNOMED_CT": "snomed_code",
            "ORPHA": "orpha_code",
            "UMLS_CUI": "umls_cui",
            "MESH": "mesh_id",
            "MONDO": "mondo_id",
        }

        identifiers = lexicon_entry.get("identifiers", {})
        for lex_key, entity_field in code_mappings.items():
            if getattr(entity, entity_field, None) is None:
                code = identifiers.get(lex_key)
                if code:
                    updates[entity_field] = str(code)

        if updates:
            return entity.model_copy(update=updates)

        return entity
