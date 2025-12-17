# corpus_metadata/document_config/__init__.py
"""Configuration package for corpus metadata extraction."""

from .config_schema import (
    CorpusConfigSchema,
    FeaturesConfig,
    LoggingConfig,
    PathsConfig,
    DefaultsConfig,
    load_and_validate_config,
)