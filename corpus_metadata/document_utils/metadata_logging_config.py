#!/usr/bin/env python3
"""
corpus_metadata/document_utils/metadata_logging_config.py
=============================================================
Central configuration loader and logging utilities.
"""

import sys
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from contextlib import contextmanager

# Support direct script execution
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from corpus_metadata.document_utils.console_colors import Colors
from corpus_metadata.document_config.config_schema import (
    CorpusConfigSchema,
    load_and_validate_config,
)

_logger = logging.getLogger(__name__)


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


def _resolve_logger(logger_inst):
    """Resolve logger from string name, Logger object, or None."""
    if isinstance(logger_inst, str):
        return logging.getLogger(logger_inst)
    elif logger_inst is None:
        return _logger
    return logger_inst


@contextmanager
def timed_section(name: str, logger_inst=None, threshold: float = 0.0):
    """Context manager for timing code sections."""
    log = _resolve_logger(logger_inst)
    start_time = time.time()
    
    try:
        log.debug(f"Starting: {name}")
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed >= threshold:
            log.debug(f"Completed: {name} ({elapsed:.2f}s)")


def log_summary(logger_inst, title: str, stats: Dict[str, Any], level: int = logging.INFO):
    """Log a formatted summary of statistics."""
    log = _resolve_logger(logger_inst)
    
    log.log(level, f"\n{title}")
    log.log(level, "=" * len(title))
    
    for key, value in stats.items():
        if isinstance(value, dict):
            log.log(level, f"  {key}:")
            for sub_key, sub_value in value.items():
                log.log(level, f"    {sub_key}: {sub_value}")
        else:
            log.log(level, f"  {key}: {value}")
    
    log.log(level, "=" * len(title))


# ============================================================================
# CORPUS CONFIG (Wrapper around schema)
# ============================================================================

class CorpusConfig:
    """
    Configuration loader with logging setup.
    
    Wraps CorpusConfigSchema with logging initialization and console output.
    """
    
    def __init__(self, config_dir: str = "document_config", config_file: str = "config.yaml",
                 verbose: bool = True, use_colors: bool = True):
        self.verbose = verbose
        
        if not use_colors or not sys.stdout.isatty():
            Colors.disable()
        
        # Resolve config path
        config_dir_path = Path(config_dir)
        if not config_dir_path.is_absolute():
            config_dir_path = Path(__file__).parent.parent / config_dir
        
        config_path = config_dir_path / config_file
        
        if self.verbose:
            self._print_header()
        
        # Load and validate config using schema
        try:
            self.schema = load_and_validate_config(config_path)
            self._print_status(f"Loaded: {config_path.name}", "success")
        except Exception as e:
            self._print_status(f"Failed to load config: {e}", "error")
            raise
        
        # Setup logging
        self._setup_logging()
        
        if self.verbose:
            self._print_config_summary()
    
    # -------------------------------------------------------------------------
    # Delegate to schema (typed access)
    # -------------------------------------------------------------------------
    
    @property
    def features(self):
        return self.schema.features
    
    @property
    def defaults(self):
        return self.schema.defaults
    
    @property
    def logging_config(self):
        return self.schema.logging
    
    @property
    def api(self):
        return self.schema.api
    
    @property
    def validation(self):
        return self.schema.validation
    
    @property
    def pipeline(self):
        return self.schema.pipeline
    
    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------
    
    def get_resource_path(self, name: str) -> Path:
        """Get absolute path for a resource."""
        return self.schema.get_resource_path(name)
    
    def get_database_path(self, name: str) -> Path:
        """Get absolute path for a database."""
        return self.schema.get_database_path(name)
    
    def get_log_path(self) -> Path:
        """Get absolute path for log file."""
        return self.schema.get_log_path()
    
    # -------------------------------------------------------------------------
    # Legacy getters (for backward compatibility)
    # -------------------------------------------------------------------------
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation (legacy support).
        
        For dataclass attributes, returns a dict representation for backward
        compatibility with code that expects dict.get() behavior.
        """
        from dataclasses import asdict, is_dataclass
        
        parts = key.split('.')
        value = self.schema
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        # Convert dataclass to dict for backward compatibility
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        
        return value
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags as dict."""
        return {
            k: v for k, v in vars(self.schema.features).items()
            if not k.startswith('_')
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled."""
        return getattr(self.schema.features, feature, False)
    
    def get_all_stages(self):
        """Get pipeline stages."""
        return [(s.name, s) for s in self.schema.pipeline.stages]
    
    # -------------------------------------------------------------------------
    # Logging Setup
    # -------------------------------------------------------------------------
    
    def _setup_logging(self):
        """Configure logging system."""
        log_cfg = self.schema.logging
        log_path = self.get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_cfg.max_size_mb * 1024 * 1024,
            backupCount=log_cfg.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_cfg.level))
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)
        
        # Console handler (optional)
        if log_cfg.console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(getattr(logging, log_cfg.console_level))
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
            root_logger.addHandler(console_handler)
        
        root_logger.setLevel(getattr(logging, log_cfg.level))
        
        # Suppress noisy libraries
        for lib in ['httpx', 'spacy', 'urllib3', 'requests']:
            logging.getLogger(lib).setLevel(logging.WARNING)
        
        logging.getLogger('corpus_config').info(f"Logging initialized: {log_path}")
        
        if self.verbose:
            # Display relative path for cleaner output
            try:
                display_path = log_path.relative_to(Path.cwd())
            except ValueError:
                display_path = log_path  # Fallback to absolute if not relative to cwd
            self._print_status(f"Logging to: {display_path}", "success")
    
    # -------------------------------------------------------------------------
    # Console Output
    # -------------------------------------------------------------------------
    
    def _print_header(self):
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"  CORPUS METADATA SYSTEM - Configuration Loader")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}{Colors.ENDC}\n")
    
    def _print_config_summary(self):
        if not self.verbose:
            return
        
        features = self.get_feature_flags()
        enabled = sum(1 for v in features.values() if v)
        
        self._print_status(f"Features: {enabled}/{len(features)} enabled", "success")
        self._print_status(f"{self.schema.system.name} - Version {self.schema.system.version}", "success")
    
    def _print_status(self, message: str, status: str = "info"):
        if not self.verbose:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "success": f"{Colors.GREEN}[OK]{Colors.ENDC}",
            "error": f"{Colors.RED}[X]{Colors.ENDC}",
            "warning": f"{Colors.YELLOW}[!]{Colors.ENDC}",
        }
        icon = icons.get(status, "")
        print(f"[{timestamp}] {icon} {message}")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'CorpusConfig',
    'get_logger',
    'timed_section',
    'log_summary',
]