# corpus_metadata/G_config/__init__.py
"""
Configuration module for corpus_metadata extraction pipeline.

RECOMMENDED: Load configuration from config.yaml:

    from G_config import load_config

    # Load from config.yaml (recommended)
    config = load_config()
    print(config.enabled_extractors)

Configure which extractors run by editing G_config/config.yaml:

    extraction_pipeline:
      preset: null  # Or use a preset name below
      extractors:
        drugs: true
        diseases: true
        abbreviations: true
        feasibility: true
        pharma_companies: false
        authors: false
        citations: false
        document_metadata: false
        tables: true

Or use presets programmatically:

    from G_config import ExtractionConfig, ExtractionPreset

    config = ExtractionConfig.from_preset(ExtractionPreset.DRUGS_ONLY)
    config = ExtractionConfig.from_preset(ExtractionPreset.CLINICAL_ENTITIES)

Available presets:
    - drugs_only: Drug detection only
    - diseases_only: Disease detection only
    - abbreviations_only: Abbreviation extraction only
    - feasibility_only: Feasibility extraction only
    - entities_only: Drugs + Diseases + Abbreviations
    - clinical_entities: Drugs + Diseases
    - metadata_only: Authors + Citations + Doc metadata
    - standard: Drugs + Diseases + Abbreviations + Feasibility (DEFAULT)
    - all: Everything
    - minimal: Just abbreviations (fastest)
"""

from .extraction_config import (
    ExtractionConfig,
    ExtractionPreset,
    # Convenience functions
    load_config,
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
    "load_config",
    "drugs_only",
    "diseases_only",
    "abbreviations_only",
    "feasibility_only",
    "entities_only",
    "clinical_entities",
    "all_extractors",
    "standard",
]
