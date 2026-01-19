# corpus_metadata/G_config/extraction_config.py
"""
Extraction configuration for selective entity extraction.

Allows running specific extractors independently or in combination:
- Drug detection only
- Disease detection only
- Abbreviation detection only
- Feasibility extraction only
- Any combination of the above
- All extractors together

Usage:
    from G_config.extraction_config import ExtractionConfig, ExtractionPreset

    # Use a preset
    config = ExtractionConfig.from_preset(ExtractionPreset.DRUGS_ONLY)

    # Custom configuration
    config = ExtractionConfig(
        drugs=True,
        diseases=True,
        abbreviations=False,
    )

    # Run extraction
    from run_extraction import run_extraction
    results = run_extraction("document.pdf", config)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExtractionPreset(str, Enum):
    """Predefined extraction configurations."""

    # Single entity types
    DRUGS_ONLY = "drugs_only"
    DISEASES_ONLY = "diseases_only"
    ABBREVIATIONS_ONLY = "abbreviations_only"
    FEASIBILITY_ONLY = "feasibility_only"

    # Entity groups
    ENTITIES_ONLY = "entities_only"  # Drugs + Diseases + Abbreviations (no feasibility)
    CLINICAL_ENTITIES = "clinical_entities"  # Drugs + Diseases
    METADATA_ONLY = "metadata_only"  # Authors + Citations + Document metadata

    # Comprehensive
    ALL = "all"  # Everything
    MINIMAL = "minimal"  # Just abbreviations (fastest)
    STANDARD = "standard"  # Drugs + Diseases + Abbreviations + Feasibility


@dataclass
class ExtractionConfig:
    """
    Configuration for selective entity extraction.

    Each boolean flag controls whether that extractor runs.
    """

    # Core entity extractors
    drugs: bool = True
    diseases: bool = True
    abbreviations: bool = True

    # Feasibility & clinical
    feasibility: bool = True
    pharma_companies: bool = False

    # Document metadata
    authors: bool = False
    citations: bool = False
    document_metadata: bool = False

    # Table extraction
    tables: bool = True

    # Processing options
    use_llm_validation: bool = True  # LLM validation for abbreviations
    use_llm_feasibility: bool = True  # LLM-based feasibility extraction
    use_vlm_tables: bool = False  # Vision model for table extraction
    skip_normalization: bool = False  # Skip enrichment/normalization

    # Output options
    output_dir: Optional[Path] = None
    export_json: bool = True
    export_combined: bool = False  # Single combined output file

    # Advanced options
    parallel_extraction: bool = True  # Run independent extractors in parallel
    max_pages: Optional[int] = None  # Limit pages to process
    page_range: Optional[tuple] = None  # Specific page range (start, end)

    def __post_init__(self):
        if self.output_dir and isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_preset(cls, preset: ExtractionPreset) -> "ExtractionConfig":
        """Create configuration from a preset."""
        presets = {
            ExtractionPreset.DRUGS_ONLY: cls(
                drugs=True,
                diseases=False,
                abbreviations=False,
                feasibility=False,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=False,
            ),
            ExtractionPreset.DISEASES_ONLY: cls(
                drugs=False,
                diseases=True,
                abbreviations=False,
                feasibility=False,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=False,
            ),
            ExtractionPreset.ABBREVIATIONS_ONLY: cls(
                drugs=False,
                diseases=False,
                abbreviations=True,
                feasibility=False,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=False,
            ),
            ExtractionPreset.FEASIBILITY_ONLY: cls(
                drugs=False,
                diseases=False,
                abbreviations=False,
                feasibility=True,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=True,  # Tables often needed for feasibility
            ),
            ExtractionPreset.ENTITIES_ONLY: cls(
                drugs=True,
                diseases=True,
                abbreviations=True,
                feasibility=False,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=False,
            ),
            ExtractionPreset.CLINICAL_ENTITIES: cls(
                drugs=True,
                diseases=True,
                abbreviations=False,
                feasibility=False,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=False,
            ),
            ExtractionPreset.METADATA_ONLY: cls(
                drugs=False,
                diseases=False,
                abbreviations=False,
                feasibility=False,
                pharma_companies=False,
                authors=True,
                citations=True,
                document_metadata=True,
                tables=False,
            ),
            ExtractionPreset.ALL: cls(
                drugs=True,
                diseases=True,
                abbreviations=True,
                feasibility=True,
                pharma_companies=True,
                authors=True,
                citations=True,
                document_metadata=True,
                tables=True,
            ),
            ExtractionPreset.MINIMAL: cls(
                drugs=False,
                diseases=False,
                abbreviations=True,
                feasibility=False,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=False,
                use_llm_validation=False,
            ),
            ExtractionPreset.STANDARD: cls(
                drugs=True,
                diseases=True,
                abbreviations=True,
                feasibility=True,
                pharma_companies=False,
                authors=False,
                citations=False,
                document_metadata=False,
                tables=True,
            ),
        }
        return presets.get(preset, cls())

    @classmethod
    def from_flags(cls, *flags: str) -> "ExtractionConfig":
        """
        Create configuration from string flags.

        Args:
            flags: Entity type names to enable (e.g., "drugs", "diseases")

        Example:
            config = ExtractionConfig.from_flags("drugs", "diseases")
        """
        config = cls(
            drugs=False,
            diseases=False,
            abbreviations=False,
            feasibility=False,
            pharma_companies=False,
            authors=False,
            citations=False,
            document_metadata=False,
            tables=False,
        )

        flag_map = {
            "drugs": "drugs",
            "drug": "drugs",
            "diseases": "diseases",
            "disease": "diseases",
            "abbreviations": "abbreviations",
            "abbrev": "abbreviations",
            "abbr": "abbreviations",
            "feasibility": "feasibility",
            "feas": "feasibility",
            "pharma": "pharma_companies",
            "pharma_companies": "pharma_companies",
            "authors": "authors",
            "author": "authors",
            "citations": "citations",
            "citation": "citations",
            "refs": "citations",
            "metadata": "document_metadata",
            "doc_metadata": "document_metadata",
            "tables": "tables",
            "table": "tables",
        }

        for flag in flags:
            attr = flag_map.get(flag.lower())
            if attr:
                setattr(config, attr, True)

        return config

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractionConfig":
        """Create configuration from a dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "drugs": self.drugs,
            "diseases": self.diseases,
            "abbreviations": self.abbreviations,
            "feasibility": self.feasibility,
            "pharma_companies": self.pharma_companies,
            "authors": self.authors,
            "citations": self.citations,
            "document_metadata": self.document_metadata,
            "tables": self.tables,
            "use_llm_validation": self.use_llm_validation,
            "use_llm_feasibility": self.use_llm_feasibility,
            "use_vlm_tables": self.use_vlm_tables,
            "skip_normalization": self.skip_normalization,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "export_json": self.export_json,
            "export_combined": self.export_combined,
            "parallel_extraction": self.parallel_extraction,
            "max_pages": self.max_pages,
            "page_range": self.page_range,
        }

    @property
    def enabled_extractors(self) -> List[str]:
        """Get list of enabled extractor names."""
        extractors = []
        if self.drugs:
            extractors.append("drugs")
        if self.diseases:
            extractors.append("diseases")
        if self.abbreviations:
            extractors.append("abbreviations")
        if self.feasibility:
            extractors.append("feasibility")
        if self.pharma_companies:
            extractors.append("pharma_companies")
        if self.authors:
            extractors.append("authors")
        if self.citations:
            extractors.append("citations")
        if self.document_metadata:
            extractors.append("document_metadata")
        if self.tables:
            extractors.append("tables")
        return extractors

    @property
    def enabled_count(self) -> int:
        """Count of enabled extractors."""
        return len(self.enabled_extractors)

    def __str__(self) -> str:
        enabled = self.enabled_extractors
        if not enabled:
            return "ExtractionConfig(none enabled)"
        return f"ExtractionConfig({', '.join(enabled)})"

    def __repr__(self) -> str:
        return self.__str__()


# Convenience functions for quick configuration
def drugs_only() -> ExtractionConfig:
    """Configuration for drug extraction only."""
    return ExtractionConfig.from_preset(ExtractionPreset.DRUGS_ONLY)


def diseases_only() -> ExtractionConfig:
    """Configuration for disease extraction only."""
    return ExtractionConfig.from_preset(ExtractionPreset.DISEASES_ONLY)


def abbreviations_only() -> ExtractionConfig:
    """Configuration for abbreviation extraction only."""
    return ExtractionConfig.from_preset(ExtractionPreset.ABBREVIATIONS_ONLY)


def feasibility_only() -> ExtractionConfig:
    """Configuration for feasibility extraction only."""
    return ExtractionConfig.from_preset(ExtractionPreset.FEASIBILITY_ONLY)


def entities_only() -> ExtractionConfig:
    """Configuration for all entity types (drugs, diseases, abbreviations)."""
    return ExtractionConfig.from_preset(ExtractionPreset.ENTITIES_ONLY)


def clinical_entities() -> ExtractionConfig:
    """Configuration for clinical entities (drugs + diseases)."""
    return ExtractionConfig.from_preset(ExtractionPreset.CLINICAL_ENTITIES)


def all_extractors() -> ExtractionConfig:
    """Configuration to run all extractors."""
    return ExtractionConfig.from_preset(ExtractionPreset.ALL)


def standard() -> ExtractionConfig:
    """Standard configuration (drugs, diseases, abbreviations, feasibility)."""
    return ExtractionConfig.from_preset(ExtractionPreset.STANDARD)
