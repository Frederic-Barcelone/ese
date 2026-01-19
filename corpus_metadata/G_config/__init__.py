# corpus_metadata/G_config/__init__.py
"""
Configuration module for corpus_metadata extraction pipeline.

Provides selective extraction configuration:

    from G_config import ExtractionConfig, ExtractionPreset

    # Use presets
    config = ExtractionConfig.from_preset(ExtractionPreset.DRUGS_ONLY)
    config = ExtractionConfig.from_preset(ExtractionPreset.CLINICAL_ENTITIES)

    # Or build custom config
    config = ExtractionConfig(drugs=True, diseases=True, abbreviations=False)

    # Or from flags
    config = ExtractionConfig.from_flags("drugs", "diseases")

Available presets:
    - DRUGS_ONLY: Drug detection only
    - DISEASES_ONLY: Disease detection only
    - ABBREVIATIONS_ONLY: Abbreviation extraction only
    - FEASIBILITY_ONLY: Feasibility extraction only
    - ENTITIES_ONLY: Drugs + Diseases + Abbreviations
    - CLINICAL_ENTITIES: Drugs + Diseases
    - METADATA_ONLY: Authors + Citations + Doc metadata
    - STANDARD: Drugs + Diseases + Abbreviations + Feasibility
    - ALL: Everything
    - MINIMAL: Just abbreviations (fastest)
"""

from .extraction_config import (
    ExtractionConfig,
    ExtractionPreset,
    # Convenience functions
    drugs_only,
    diseases_only,
    abbreviations_only,
    feasibility_only,
    entities_only,
    clinical_entities,
    all_extractors,
    standard,
)

__all__ = [
    "ExtractionConfig",
    "ExtractionPreset",
    "drugs_only",
    "diseases_only",
    "abbreviations_only",
    "feasibility_only",
    "entities_only",
    "clinical_entities",
    "all_extractors",
    "standard",
]
