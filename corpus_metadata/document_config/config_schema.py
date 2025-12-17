#!/usr/bin/env python3
"""
corpus_metadata/document_config/config_schema.py
================================================
Configuration schema with validation and defaults.

FIXES APPLIED (Quality Check):
- FIX #4: Changed pubtator_enrichment default from True to False for privacy
- Added documentation about external API privacy implications
"""

import sys
from pathlib import Path

# Support direct script execution
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

@dataclass
class PathsConfig:
    """Paths configuration - relative to base path."""
    dictionaries: str = "corpus_dictionaries/output_datasources"
    databases: str = "corpus_db"
    logs: str = "corpus_logs"
    cache: str = "cache"
    
    # Resolved absolute paths (set during validation)
    _base_path: Optional[Path] = field(default=None, repr=False)
    
    def resolve(self, base_path: Path) -> None:
        """Resolve all paths relative to base path."""
        self._base_path = base_path
    
    def get_absolute(self, path_name: str) -> Path:
        """Get absolute path for a configured path."""
        if self._base_path is None:
            raise ValueError("Paths not resolved. Call resolve() first.")
        relative = getattr(self, path_name, None)
        if relative is None:
            raise ValueError(f"Unknown path: {path_name}")
        return self._base_path / relative


# ==============================================================================
# DEFAULTS CONFIGURATION
# ==============================================================================

@dataclass
class DefaultsConfig:
    """Global default values."""
    confidence_threshold: float = 0.75
    fuzzy_match_threshold: int = 85
    context_window: int = 100
    min_term_length: int = 3
    prefix_start_number: int = 1000  # Starting number for auto-incrementing file prefixes
    
    def __post_init__(self):
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be 0-1, got {self.confidence_threshold}")
        if not 0 <= self.fuzzy_match_threshold <= 100:
            raise ValueError(f"fuzzy_match_threshold must be 0-100, got {self.fuzzy_match_threshold}")
        if self.prefix_start_number < 0:
            raise ValueError(f"prefix_start_number must be >= 0, got {self.prefix_start_number}")


# ==============================================================================
# FEATURE FLAGS
# ==============================================================================

@dataclass
class FeaturesConfig:
    """
    Feature flags - all boolean with sensible defaults.
    
    PRIVACY NOTE: External API features (pubtator_enrichment, ai_validation)
    default to False to prevent accidental data exposure. Enable explicitly
    in config.yaml only for non-sensitive corpora.
    """
    # Core extraction
    drug_detection: bool = True
    disease_detection: bool = True
    abbreviation_extraction: bool = True  # NOTE: Code must use this exact key name
    
    # Document analysis
    classification: bool = True
    section_detection: bool = True
    table_extraction: bool = True
    
    # Metadata extraction
    title_extraction: bool = True
    date_extraction: bool = True
    description_extraction: bool = True
    
    # Citation/Reference extraction
    citation_extraction: bool = True
    person_extraction: bool = True
    reference_extraction: bool = True
    
    # External APIs
    # ⚠️ PRIVACY: These send data to third-party services
    # Default to False to protect sensitive/internal documents
    pubtator_enrichment: bool = False  # FIX #4: Changed from True to False
    ai_validation: bool = False  # Requires CLAUDE_API_KEY
    
    # File operations
    intelligent_rename: bool = True
    caching: bool = True


# ==============================================================================
# RESOURCES CONFIGURATION
# ==============================================================================

@dataclass
class ResourcesConfig:
    """Resource file names (relative to paths.dictionaries)."""
    # Abbreviations
    abbreviation_general: str = "2025_08_abbreviation_general.json"
    abbreviation_umls_biological: str = "2025_08_umls_biological_abbreviations_v5.tsv"
    abbreviation_umls_clinical: str = "2025_08_umls_clinical_abbreviations_v5.tsv"
    
    # Diseases
    disease_lexicon: str = "2025_08_lexicon_disease.json"
    disease_rare_acronyms: str = "2025_08_rare_disease_acronyms.json"
    
    # Drugs
    drug_lexicon: str = "2025_08_lexicon_drug.json"
    drug_alexion: str = "2025_08_alexion_drugs.json"
    drug_fda_approved: str = "2025_08_fda_approved_drugs.json"
    drug_investigational: str = "2025_08_investigational_drugs.json"
    
    # Other
    medical_terms_lexicon: str = "2025_08_lexicon_medical_terms.json"
    document_types: str = "2025_08_document_types.json"
    clinical_trial_metadata: str = "00966_clinical_trial_metadata.json"


@dataclass
class DatabasesConfig:
    """Database file names (relative to paths.databases)."""
    disease_ontology: str = "disease_ontology.db"
    disease_orphanet: str = "orphanet_nlp.db"


# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

@dataclass
class StageLimits:
    """Limits for a pipeline stage."""
    pdf_pages: Optional[int] = None
    text_chars: Optional[int] = None


@dataclass
class PipelineStage:
    """Single pipeline stage configuration."""
    name: str
    sequence: int
    tasks: List[str] = field(default_factory=list)
    limits: StageLimits = field(default_factory=StageLimits)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    stages: List[PipelineStage] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        stages = []
        for stage_data in data.get('stages', []):
            limits_data = stage_data.get('limits', {})
            limits = StageLimits(
                pdf_pages=limits_data.get('pdf_pages'),
                text_chars=limits_data.get('text_chars')
            )
            stage = PipelineStage(
                name=stage_data.get('name', 'unnamed'),
                sequence=stage_data.get('sequence', 0),
                tasks=stage_data.get('tasks', []),
                limits=limits
            )
            stages.append(stage)
        return cls(stages=stages)


# ==============================================================================
# API CONFIGURATION
# ==============================================================================

@dataclass
class PubtatorApiConfig:
    """PubTator API configuration."""
    base_url: str = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 30


@dataclass
class ClaudeApiConfig:
    """Claude API configuration."""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1500
    temperature: float = 0


@dataclass
class ApiConfig:
    """All API configurations."""
    pubtator: PubtatorApiConfig = field(default_factory=PubtatorApiConfig)
    claude: ClaudeApiConfig = field(default_factory=ClaudeApiConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiConfig':
        """Create from dictionary."""
        return cls(
            pubtator=PubtatorApiConfig(**data.get('pubtator', {})),
            claude=ClaudeApiConfig(**data.get('claude', {}))
        )


# ==============================================================================
# VALIDATION CONFIGURATION
# ==============================================================================

@dataclass
class DrugValidationConfig:
    """Drug validation settings."""
    confidence_threshold: float = 0.85
    stages: List[str] = field(default_factory=lambda: ["false_positive_filter", "knowledge_base", "claude_ai"])


@dataclass
class DiseaseValidationConfig:
    """Disease validation settings."""
    confidence_threshold: float = 0.75
    enrichment_mode: str = "balanced"
    stages: List[str] = field(default_factory=lambda: ["pattern_detection", "lexicon_matching", "claude_ai"])


@dataclass
class ValidationConfig:
    """All validation configurations."""
    drug: DrugValidationConfig = field(default_factory=DrugValidationConfig)
    disease: DiseaseValidationConfig = field(default_factory=DiseaseValidationConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        """Create from dictionary."""
        drug_data = data.get('drug', {})
        disease_data = data.get('disease', {})
        return cls(
            drug=DrugValidationConfig(
                confidence_threshold=drug_data.get('confidence_threshold', 0.85),
                stages=drug_data.get('stages', ["false_positive_filter", "knowledge_base", "claude_ai"])
            ),
            disease=DiseaseValidationConfig(
                confidence_threshold=disease_data.get('confidence_threshold', 0.75),
                enrichment_mode=disease_data.get('enrichment_mode', 'balanced'),
                stages=disease_data.get('stages', ["pattern_detection", "lexicon_matching", "claude_ai"])
            )
        )


# ==============================================================================
# DEDUPLICATION CONFIGURATION
# ==============================================================================

@dataclass
class DeduplicationConfig:
    """Deduplication settings."""
    enabled: bool = True
    priority_order: List[str] = field(default_factory=lambda: ["drug", "disease", "abbreviation"])
    remove_expansion_matches: bool = True


# ==============================================================================
# PROMOTION CONFIGURATION
# ==============================================================================

@dataclass
class PromotionConfig:
    """Abbreviation to entity promotion settings."""
    min_ids_required: int = 1
    confidence_boost: float = 0.05
    preferred_drug_ids: List[str] = field(default_factory=lambda: ["RxCUI", "MESH", "ATC", "DrugBank", "UNII"])
    preferred_disease_ids: List[str] = field(default_factory=lambda: ["ORPHA", "DOID", "SNOMED_CT", "ICD10", "UMLS_CUI"])


# ==============================================================================
# EXTRACTION CONFIGURATION
# ==============================================================================

@dataclass
class CitationExtractionConfig:
    """Citation extraction settings."""
    detect_style: bool = True
    link_inline: bool = True
    extract_identifiers: bool = True


@dataclass
class PersonExtractionConfig:
    """Person extraction settings."""
    extract_affiliations: bool = True
    classify_roles: bool = True
    validate_orcid: bool = True


@dataclass
class ReferenceExtractionConfig:
    """Reference extraction settings."""
    extract_all_types: bool = True
    reconstruct_urls: bool = True
    classify_roles: bool = True


@dataclass
class ExtractionConfig:
    """All extraction configurations."""
    citation: CitationExtractionConfig = field(default_factory=CitationExtractionConfig)
    person: PersonExtractionConfig = field(default_factory=PersonExtractionConfig)
    reference: ReferenceExtractionConfig = field(default_factory=ReferenceExtractionConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConfig':
        """Create from dictionary."""
        return cls(
            citation=CitationExtractionConfig(**data.get('citation', {})),
            person=PersonExtractionConfig(**data.get('person', {})),
            reference=ReferenceExtractionConfig(**data.get('reference', {}))
        )


# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

@dataclass
class OutputIncludeConfig:
    """What to include in output."""
    statistics: bool = True
    confidence: bool = True
    context: bool = True
    identifiers: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    format: str = "json"
    json_indent: int = 2
    include: OutputIncludeConfig = field(default_factory=OutputIncludeConfig)
    
    def __post_init__(self):
        valid_formats = ["json", "yaml", "csv"]
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputConfig':
        """Create from dictionary."""
        include_data = data.get('include', {})
        return cls(
            format=data.get('format', 'json'),
            json_indent=data.get('json_indent', 2),
            include=OutputIncludeConfig(**include_data) if include_data else OutputIncludeConfig()
        )


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    console: bool = False
    console_level: str = "ERROR"
    file: str = "corpus.log"
    max_size_mb: int = 10
    backup_count: int = 5
    
    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        if self.console_level not in valid_levels:
            raise ValueError(f"console_level must be one of {valid_levels}")


# ==============================================================================
# RUNTIME CONFIGURATION
# ==============================================================================

@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    batch_size: int = 10
    timeout_seconds: int = 300
    max_file_size_mb: int = 100
    on_error: str = "log_and_continue"
    
    def __post_init__(self):
        valid_error_handlers = ["log_and_continue", "raise", "skip"]
        if self.on_error not in valid_error_handlers:
            raise ValueError(f"on_error must be one of {valid_error_handlers}")


# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================

@dataclass
class SystemConfig:
    """System metadata."""
    name: str = "Rare Disease Document Extraction"
    version: str = "13.1"  # Updated version


# ==============================================================================
# MAIN CONFIG SCHEMA
# ==============================================================================

@dataclass
class CorpusConfigSchema:
    """
    Main configuration schema.
    
    Validates and provides defaults for all configuration.
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    databases: DatabasesConfig = field(default_factory=DatabasesConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorpusConfigSchema':
        """Create schema from dictionary (e.g., loaded YAML)."""
        return cls(
            system=SystemConfig(**data.get('system', {})),
            paths=PathsConfig(**data.get('paths', {})),
            defaults=DefaultsConfig(**data.get('defaults', {})),
            features=FeaturesConfig(**data.get('features', {})),
            resources=ResourcesConfig(**data.get('resources', {})),
            databases=DatabasesConfig(**data.get('databases', {})),
            pipeline=PipelineConfig.from_dict(data.get('pipeline', {})),
            api=ApiConfig.from_dict(data.get('api', {})),
            validation=ValidationConfig.from_dict(data.get('validation', {})),
            deduplication=DeduplicationConfig(**data.get('deduplication', {})),
            promotion=PromotionConfig(**data.get('promotion', {})),
            extraction=ExtractionConfig.from_dict(data.get('extraction', {})),
            output=OutputConfig.from_dict(data.get('output', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            runtime=RuntimeConfig(**data.get('runtime', {}))
        )
    
    def resolve_paths(self, base_path: Path) -> None:
        """Resolve all relative paths to absolute paths."""
        self.paths.resolve(base_path)
    
    def get_resource_path(self, resource_name: str) -> Path:
        """Get absolute path for a resource file."""
        if self.paths._base_path is None:
            raise ValueError("Paths not resolved. Call resolve_paths() first.")
        filename = getattr(self.resources, resource_name, None)
        if filename is None:
            raise ValueError(f"Unknown resource: {resource_name}")
        return self.paths._base_path / self.paths.dictionaries / filename
    
    def get_database_path(self, db_name: str) -> Path:
        """Get absolute path for a database file."""
        if self.paths._base_path is None:
            raise ValueError("Paths not resolved. Call resolve_paths() first.")
        filename = getattr(self.databases, db_name, None)
        if filename is None:
            raise ValueError(f"Unknown database: {db_name}")
        return self.paths._base_path / self.paths.databases / filename
    
    def get_log_path(self) -> Path:
        """Get absolute path for log file."""
        if self.paths._base_path is None:
            raise ValueError("Paths not resolved. Call resolve_paths() first.")
        return self.paths._base_path / self.paths.logs / self.logging.file


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_and_validate_config(yaml_path: Path, base_path: Optional[Path] = None) -> CorpusConfigSchema:
    """
    Load YAML config and validate against schema.
    
    Args:
        yaml_path: Path to config.yaml
        base_path: Base path for resolving relative paths (default: parent of yaml_path)
    
    Returns:
        Validated CorpusConfigSchema
    
    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    import yaml
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f) or {}
    
    # Create schema (validates during creation)
    schema = CorpusConfigSchema.from_dict(raw_config)
    
    # Resolve paths
    if base_path is None:
        base_path = yaml_path.parent.parent.parent  # corpus_metadata/document_config -> project root
    schema.resolve_paths(base_path)
    
    return schema