# corpus_metadata/E_normalization/E13_registry_enricher.py
"""
Registry Enricher for clinical trial feasibility extraction.

Extracts patient registry information that provides real-world cohort access,
natural history data, and external control arms for rare-disease trials.

Uses ZeroShotBioNER's zero-shot capability with registry-specific entity labels:
- registry_name: Registry identifiers (RaDaR, APPEAR-C3G, Kidgo, GARD)
- registry_size: Cohort sizes (N=1,247 patients, annual accrual 50 pts)
- geographic_coverage: Geographic scope (EU-wide, US centers, national)
- data_types: Available data (genomics, HPO phenotypes, labs, longitudinal)
- access_policy: Data access terms (research-only, IRB approval, federated query)
- eligibility_criteria: Registry inclusion criteria (confirmed DMD genotype)

Model: ProdicusII/ZeroShotBioNER (BioBERT-based zero-shot NER)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Entity labels for registry extraction
REGISTRY_LABELS = [
    "registry_name",        # Registry identifiers (RaDaR, APPEAR-C3G, Kidgo, GARD)
    "registry_size",        # Cohort sizes (N=1,247 patients)
    "geographic_coverage",  # Geographic scope (EU-wide, US centers, national)
    "data_types",           # Available data types (genomics, phenotypes, labs)
    "access_policy",        # Data access policy (research-only, federated query)
    "eligibility_criteria", # Registry inclusion criteria
]


# Known rare disease registries for post-processing linkage
KNOWN_REGISTRIES = {
    # Rare disease registries
    "radar": {"full_name": "RaDaR - Rare Diseases Registry", "scope": "UK", "type": "renal"},
    "appear-c3g": {"full_name": "APPEAR-C3G Registry", "scope": "International", "type": "C3G"},
    "kidgo": {"full_name": "KDIGO Registry", "scope": "International", "type": "kidney"},
    "gard": {"full_name": "Genetic and Rare Diseases Information Center", "scope": "US", "type": "rare_disease"},
    "ordr": {"full_name": "Office of Rare Diseases Research", "scope": "US", "type": "rare_disease"},
    "eurordis": {"full_name": "EURORDIS Rare Diseases Europe", "scope": "EU", "type": "rare_disease"},
    "nord": {"full_name": "National Organization for Rare Disorders", "scope": "US", "type": "rare_disease"},
    "orphanet": {"full_name": "Orphanet", "scope": "International", "type": "rare_disease"},
    # Disease-specific registries
    "cftr2": {"full_name": "CFTR2 - CF Mutation Database", "scope": "International", "type": "cystic_fibrosis"},
    "cinrg": {"full_name": "CINRG Duchenne Natural History Study", "scope": "International", "type": "DMD"},
    "treat-nmd": {"full_name": "TREAT-NMD Alliance", "scope": "International", "type": "neuromuscular"},
}


@dataclass
class RegistryEntity:
    """Single entity extracted for registry information."""

    text: str
    entity_type: str
    score: float
    start: int = 0
    end: int = 0
    # Linked registry info (if matched to known registry)
    linked_registry: Optional[Dict[str, str]] = None


@dataclass
class RegistryResult:
    """Structured result from registry extraction."""

    # Core registry profile
    registry_names: List[RegistryEntity] = field(default_factory=list)
    registry_sizes: List[RegistryEntity] = field(default_factory=list)
    geographic_coverages: List[RegistryEntity] = field(default_factory=list)

    # Data & access
    data_types: List[RegistryEntity] = field(default_factory=list)
    access_policies: List[RegistryEntity] = field(default_factory=list)
    eligibility_criteria: List[RegistryEntity] = field(default_factory=list)

    # Raw entities for debugging
    raw_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    extraction_time_seconds: float = 0.0

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dict for logging/export."""
        return {
            "registry_name": len(self.registry_names),
            "registry_size": len(self.registry_sizes),
            "geographic_coverage": len(self.geographic_coverages),
            "data_types": len(self.data_types),
            "access_policy": len(self.access_policies),
            "eligibility_criteria": len(self.eligibility_criteria),
            "total": self.total_entities,
            "extraction_time_seconds": self.extraction_time_seconds,
        }

    @property
    def total_entities(self) -> int:
        """Total number of extracted entities."""
        return (
            len(self.registry_names)
            + len(self.registry_sizes)
            + len(self.geographic_coverages)
            + len(self.data_types)
            + len(self.access_policies)
            + len(self.eligibility_criteria)
        )

    def get_linked_registries(self) -> List[Dict[str, Any]]:
        """Get list of registries that were linked to known databases."""
        linked = []
        for entity in self.registry_names:
            if entity.linked_registry:
                linked.append({
                    "extracted_text": entity.text,
                    "confidence": entity.score,
                    **entity.linked_registry,
                })
        return linked


class RegistryEnricher:
    """
    Enricher for patient registry extraction using ZeroShotBioNER.

    Reuses ZeroShotBioNER's extract_custom() method with registry-specific
    entity labels to extract cohort and registry information.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.run_id = config.get("run_id", "unknown")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.entity_labels = config.get("entity_labels", REGISTRY_LABELS)
        self.link_registries = config.get("link_registries", True)
        self._zeroshot = None

    def _load_zeroshot(self) -> bool:
        """Lazy load ZeroShotBioNER enricher."""
        if self._zeroshot is not None:
            return True

        try:
            from E_normalization.E09_zeroshot_bioner import ZeroShotBioNEREnricher

            self._zeroshot = ZeroShotBioNEREnricher(
                config={
                    "confidence_threshold": self.confidence_threshold,
                }
            )
            return True
        except ImportError as e:
            logger.warning(f"Failed to import ZeroShotBioNEREnricher: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ZeroShotBioNER: {e}")
            return False

    def _link_to_known_registry(self, text: str) -> Optional[Dict[str, str]]:
        """
        Attempt to link extracted registry name to known registry database.

        Args:
            text: Extracted registry name text

        Returns:
            Dict with linked registry info, or None if no match
        """
        if not self.link_registries:
            return None

        text_lower = text.lower().strip()

        # Direct match
        for key, info in KNOWN_REGISTRIES.items():
            if key in text_lower or text_lower in info["full_name"].lower():
                return info

        # Partial match on full name
        for key, info in KNOWN_REGISTRIES.items():
            full_name_lower = info["full_name"].lower()
            # Check if any significant word matches
            text_words = set(text_lower.split())
            name_words = set(full_name_lower.split())
            if text_words & name_words:  # Intersection
                return info

        return None

    def extract(self, text: str) -> RegistryResult:
        """
        Extract registry entities from text.

        Args:
            text: Input text (document, section, or abstract)

        Returns:
            RegistryResult with categorized entities
        """
        result = RegistryResult()
        start_time = time.time()

        if not text or not text.strip():
            return result

        if not self._load_zeroshot():
            logger.warning("ZeroShotBioNER not available, returning empty result")
            return result

        try:
            # Use extract_custom with registry labels
            raw_results = self._zeroshot.extract_custom(text, self.entity_labels)

            # Categorize extracted entities
            for entity_type, entities in raw_results.items():
                for entity in entities:
                    # Create RegistryEntity
                    linked = None
                    if entity_type == "registry_name":
                        linked = self._link_to_known_registry(entity.text)

                    reg_entity = RegistryEntity(
                        text=entity.text,
                        entity_type=entity_type,
                        score=entity.score,
                        start=entity.start,
                        end=entity.end,
                        linked_registry=linked,
                    )

                    # Store raw entity for debugging
                    result.raw_entities.append({
                        "text": entity.text,
                        "type": entity_type,
                        "score": entity.score,
                        "start": entity.start,
                        "end": entity.end,
                        "linked": linked,
                    })

                    # Route to appropriate list
                    if entity_type == "registry_name":
                        result.registry_names.append(reg_entity)
                    elif entity_type == "registry_size":
                        result.registry_sizes.append(reg_entity)
                    elif entity_type == "geographic_coverage":
                        result.geographic_coverages.append(reg_entity)
                    elif entity_type == "data_types":
                        result.data_types.append(reg_entity)
                    elif entity_type == "access_policy":
                        result.access_policies.append(reg_entity)
                    elif entity_type == "eligibility_criteria":
                        result.eligibility_criteria.append(reg_entity)

        except Exception as e:
            logger.error(f"Error during registry extraction: {e}")

        result.extraction_time_seconds = time.time() - start_time
        return result


def extract_registries(
    text: str, config: Optional[Dict[str, Any]] = None
) -> RegistryResult:
    """
    Convenience function for quick registry extraction.

    Args:
        text: Input text
        config: Optional configuration

    Returns:
        RegistryResult with extracted entities
    """
    enricher = RegistryEnricher(config)
    return enricher.extract(text)
