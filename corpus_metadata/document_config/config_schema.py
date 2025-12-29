#!/usr/bin/env python3
"""
corpus_metadata/document_config/config_schema.py
Configuration schema with validation and defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class PathsConfig:
    """Paths configuration - relative to base path."""
    dictionaries: str = "corpus_dictionaries/output_datasources"
    databases: str = "corpus_db"
    logs: str = "corpus_logs"
    cache: str = "cache"
    _base_path: Optional[Path] = field(default=None, repr=False)
    
    def resolve(self, base_path: Path) -> None:
        self._base_path = base_path
    
    def get_absolute(self, path_name: str) -> Path:
        if self._base_path is None:
            raise ValueError("Paths not resolved. Call resolve() first.")
        return self._base_path / getattr(self, path_name)


@dataclass
class DefaultsConfig:
    """Global default values."""
    confidence_threshold: float = 0.75
    fuzzy_match_threshold: int = 85
    context_window: int = 100
    min_term_length: int = 3
    prefix_start_number: int = 1000


@dataclass
class FeaturesConfig:
    """Feature flags."""
    drug_detection: bool = True
    disease_detection: bool = True
    classification: bool = True
    section_detection: bool = True
    table_extraction: bool = True
    title_extraction: bool = True
    date_extraction: bool = True
    description_extraction: bool = True
    citation_extraction: bool = True
    person_extraction: bool = True
    reference_extraction: bool = True
    pubtator_enrichment: bool = False
    ai_validation: bool = False
    intelligent_rename: bool = True
    caching: bool = True


@dataclass
class ResourcesConfig:
    """Resource file names (relative to paths.dictionaries)."""
    disease_lexicon: str = "2025_08_lexicon_disease.json"
    disease_rare_acronyms: str = "2025_08_rare_disease_acronyms.json"
    disease_lexicon_pah: str = "disease_lexicon_pah.json"
    disease_lexicon_anca: str = "disease_lexicon_anca.json"
    disease_lexicon_igan: str = "disease_lexicon_igan.json"
    drug_lexicon: str = "2025_08_lexicon_drug.json"
    drug_alexion: str = "2025_08_alexion_drugs.json"
    drug_fda_approved: str = "2025_08_fda_approved_drugs.json"
    drug_investigational: str = "2025_08_investigational_drugs.json"
    medical_terms_lexicon: str = "2025_08_lexicon_medical_terms.json"
    document_types: str = "2025_08_document_types.json"
    clinical_trial_metadata: str = "00966_clinical_trial_metadata.json"


@dataclass
class DatabasesConfig:
    """Database file names (relative to paths.databases)."""
    disease_ontology: str = "disease_ontology.db"
    disease_orphanet: str = "orphanet_nlp.db"


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
        stages = []
        for s in data.get('stages', []):
            limits = StageLimits(**s.get('limits', {})) if s.get('limits') else StageLimits()
            stages.append(PipelineStage(
                name=s.get('name', 'unnamed'),
                sequence=s.get('sequence', 0),
                tasks=s.get('tasks', []),
                limits=limits
            ))
        return cls(stages=stages)


@dataclass
class PubtatorApiConfig:
    """PubTator API configuration."""
    base_url: str = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 30


@dataclass
class ClaudeModelTierConfig:
    """Configuration for a Claude model tier."""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1500
    temperature: float = 0.0


@dataclass
class ClaudeApiConfig:
    """Claude API configuration with two-tier model support."""
    fast: ClaudeModelTierConfig = field(default_factory=ClaudeModelTierConfig)
    validation: ClaudeModelTierConfig = field(default_factory=lambda: ClaudeModelTierConfig(max_tokens=4096))
    
    def get_tier(self, tier: str) -> ClaudeModelTierConfig:
        return self.validation if tier == "validation" else self.fast


@dataclass
class ApiConfig:
    """All API configurations."""
    pubtator: PubtatorApiConfig = field(default_factory=PubtatorApiConfig)
    claude: ClaudeApiConfig = field(default_factory=ClaudeApiConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiConfig':
        claude_data = data.get('claude', {})
        fast = ClaudeModelTierConfig(**claude_data.get('fast', {}))
        val_defaults = {'max_tokens': 4096}
        val_defaults.update(claude_data.get('validation', {}))
        validation = ClaudeModelTierConfig(**val_defaults)
        return cls(
            pubtator=PubtatorApiConfig(**data.get('pubtator', {})),
            claude=ClaudeApiConfig(fast=fast, validation=validation)
        )


@dataclass
class DrugValidationConfig:
    """Drug validation settings."""
    enabled: bool = True
    model_tier: str = "validation"
    confidence_threshold: float = 0.85
    stages: List[str] = field(default_factory=lambda: ["false_positive_filter", "knowledge_base", "claude_ai"])


@dataclass
class DiseaseValidationConfig:
    """Disease validation settings."""
    enabled: bool = True
    model_tier: str = "validation"
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
        return cls(
            drug=DrugValidationConfig(**data.get('drug', {})),
            disease=DiseaseValidationConfig(**data.get('disease', {}))
        )


@dataclass
class DeduplicationConfig:
    """Deduplication settings."""
    enabled: bool = True
    priority_order: List[str] = field(default_factory=lambda: ["drug", "disease"])
    remove_expansion_matches: bool = True


@dataclass
class CitationExtractionConfig:
    detect_style: bool = True
    link_inline: bool = True
    extract_identifiers: bool = True


@dataclass
class PersonExtractionConfig:
    extract_affiliations: bool = True
    classify_roles: bool = True
    validate_orcid: bool = True


@dataclass
class ReferenceExtractionConfig:
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
        return cls(
            citation=CitationExtractionConfig(**data.get('citation', {})),
            person=PersonExtractionConfig(**data.get('person', {})),
            reference=ReferenceExtractionConfig(**data.get('reference', {}))
        )


@dataclass
class OutputIncludeConfig:
    statistics: bool = True
    confidence: bool = True
    context: bool = True
    identifiers: bool = True


@dataclass
class OutputConfig:
    format: str = "json"
    json_indent: int = 2
    include: OutputIncludeConfig = field(default_factory=OutputIncludeConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputConfig':
        include = OutputIncludeConfig(**data.get('include', {}))
        return cls(format=data.get('format', 'json'), json_indent=data.get('json_indent', 2), include=include)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    console: bool = False
    console_level: str = "ERROR"
    file: str = "corpus.log"
    max_size_mb: int = 10
    backup_count: int = 5


@dataclass
class RuntimeConfig:
    batch_size: int = 10
    timeout_seconds: int = 300
    max_file_size_mb: int = 100
    on_error: str = "log_and_continue"


@dataclass
class SystemConfig:
    name: str = "Rare Disease Document Extraction"
    version: str = "14.0"


@dataclass
class CorpusConfigSchema:
    """Main configuration schema."""
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
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorpusConfigSchema':
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
        return self.paths.get_absolute('dictionaries') / getattr(self.resources, resource_name)
    
    def get_database_path(self, db_name: str) -> Path:
        """Get absolute path for a database file."""
        return self.paths.get_absolute('databases') / getattr(self.databases, db_name)
    
    def get_log_path(self) -> Path:
        """Get absolute path for the log file."""
        return self.paths.get_absolute('logs') / self.logging.file


def load_and_validate_config(yaml_path: Path, base_path: Optional[Path] = None) -> CorpusConfigSchema:
    """Load YAML config and validate against schema."""
    import yaml
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f) or {}
    
    schema = CorpusConfigSchema.from_dict(raw_config)
    schema.resolve_paths(base_path or yaml_path.parent.parent.parent)
    return schema