#!/usr/bin/env python3
"""
corpus_metadata/document_utils/metadata_logging_config.py
=============================================================
Central configuration loader and logging utilities for the corpus extraction system.
Enhanced with rich console output for better visibility.

FIXED: Added _setup_logging() method to actually configure logging handlers
"""

import os
import sys
import yaml
import logging
import logging.handlers
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    
    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.ENDC = ''


# ============================================================================
# CORPUS LOGGER CLASS - FIXED
# ============================================================================

class CorpusLogger:
    """
    Centralized logging utilities for the corpus system.
    Provides context managers and decorators for consistent logging.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for logger"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CorpusLogger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logger"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self._loggers = {}
            self._initialized = True
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance (for testing)"""
        with cls._lock:
            cls._instance = None
    
    @staticmethod
    @contextmanager
    def timed_section(name: str, logger_inst=None, threshold: float = 0.0):
        """
        Context manager for timing code sections.
        
        Args:
            name: Name of the section being timed
            logger_inst: Logger instance or name (default: module logger)
            threshold: Only log if execution time exceeds this threshold
        """
        start_time = time.time()
        
        # Handle both logger objects and strings
        if isinstance(logger_inst, str):
            log = logging.getLogger(logger_inst)
        elif logger_inst is None:
            log = logger
        else:
            log = logger_inst
        
        try:
            log.debug(f"Starting: {name}")
            yield
        finally:
            elapsed = time.time() - start_time
            if elapsed >= threshold:
                log.debug(f"Completed: {name} ({elapsed:.2f}s)")
    
    @staticmethod
    def log_summary(logger_inst: Union[str, logging.Logger], title: str, 
                   stats: Dict[str, Any], level: int = logging.INFO):
        """
        Log a formatted summary of statistics.
        
        Args:
            logger_inst: Logger instance or logger name string
            title: Title for the summary
            stats: Dictionary of statistics to log
            level: Logging level to use
        """
        # FIXED: Handle both string logger names and logger objects
        if isinstance(logger_inst, str):
            actual_logger = logging.getLogger(logger_inst)
        elif logger_inst is None:
            actual_logger = logger
        else:
            actual_logger = logger_inst
        
        # Now safely use the logger
        actual_logger.log(level, f"\n{title}")
        actual_logger.log(level, "=" * len(title))
        
        for key, value in stats.items():
            if isinstance(value, dict):
                actual_logger.log(level, f"  {key}:")
                for sub_key, sub_value in value.items():
                    actual_logger.log(level, f"    {sub_key}: {sub_value}")
            else:
                actual_logger.log(level, f"  {key}: {value}")
        
        actual_logger.log(level, "=" * len(title))


# ============================================================================
# LOGGING HELPER FUNCTIONS
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Name for the logger
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def singleton_logged(name: str):
    """
    Decorator to add logging to singleton classes.
    
    Args:
        name: Name for the logger
        
    Returns:
        Decorated class
    """
    def decorator(cls):
        original_new = cls.__new__
        
        @wraps(original_new)
        def logged_new(cls_arg, *args, **kwargs):
            instance = original_new(cls_arg)
            if not hasattr(instance, '_singleton_logged'):
                logger = get_logger(name)
                logger.debug(f"Creating singleton instance of {cls_arg.__name__}")
                instance._singleton_logged = True
            return instance
        
        cls.__new__ = logged_new
        return cls
    
    return decorator


def log_separator(logger_inst: Union[str, logging.Logger], style: str = 'major'):
    """
    Log a separator line for visual organization.
    
    Args:
        logger_inst: Logger instance or logger name string
        style: 'major' or 'minor' for different separator styles
    """
    # Handle both string and logger object
    if isinstance(logger_inst, str):
        actual_logger = logging.getLogger(logger_inst)
    else:
        actual_logger = logger_inst
    
    if style == 'major':
        actual_logger.debug("=" * 80)
    else:
        actual_logger.debug("-" * 60)


def log_metric(logger_inst: Union[str, logging.Logger], name: str, value: Any, unit: str = ''):
    """
    Log a metric with consistent formatting.
    
    Args:
        logger_inst: Logger instance or logger name string
        name: Name of the metric
        value: Value of the metric
        unit: Optional unit string
    """
    # Handle both string and logger object
    if isinstance(logger_inst, str):
        actual_logger = logging.getLogger(logger_inst)
    else:
        actual_logger = logger_inst
    
    if unit:
        actual_logger.debug(f"{name}: {value} {unit}")
    else:
        actual_logger.debug(f"{name}: {value}")


# ============================================================================
# CORPUS CONFIG CLASS - WITH LOGGING SETUP
# ============================================================================

class CorpusConfig:
    """
    Central configuration loader for the corpus extraction system.
    Enhanced with rich console output for better user feedback.
    NOW WITH ACTUAL LOGGING SETUP!
    """
    
    def __init__(self, config_dir: str = "corpus_config", config_file: str = "config.yaml", 
                 verbose: bool = True, use_colors: bool = True):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing config files
            config_file: Name of the configuration file (default: config.yaml)
            verbose: Enable detailed console output
            use_colors: Enable colored terminal output
        """
        self.config_dir = Path(config_dir)
        self.config_file = config_file
        self.verbose = verbose
        self.use_colors = use_colors
        
        # Disable colors if not in terminal or explicitly disabled
        if not use_colors or not sys.stdout.isatty():
            Colors.disable()
        
        # Print loading header
        if self.verbose:
            self._print_header()
            self._print_section("LOADING CONFIGURATION")
        
        self.config = self._load_config()
        self._resolved_config = self._resolve_all_references(self.config.copy())
        
        # CRITICAL: Set up logging after loading config
        self._setup_logging()
        
        # Print configuration summary
        if self.verbose:
            self._print_config_summary()
    
    def _setup_logging(self):
        """Actually configure the logging system with handlers"""
        log_config = self.get_logging_config()
        
        # Create log file path
        log_file = Path(log_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file  # Store for reference
        
        # Clear any existing handlers on root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Configure handlers
        handlers = []
        
        # File handler - always create
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_config['level'], logging.INFO))
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        handlers.append(file_handler)
        
        # Console handler - only if enabled in config
        if log_config.get('console', False):
            console_handler = logging.StreamHandler(sys.stderr)  # Use stderr to not mix with print()
            console_level = log_config.get('console_level', 'ERROR')
            console_handler.setLevel(getattr(logging, console_level, logging.ERROR))
            console_handler.setFormatter(logging.Formatter(
                '%(levelname)s - %(name)s - %(message)s'
            ))
            handlers.append(console_handler)
        
        # Configure root logger with handlers
        root_logger.setLevel(getattr(logging, log_config['level'], logging.INFO))
        for handler in handlers:
            root_logger.addHandler(handler)
        
        # Suppress noisy libraries but still log them to file
        noisy_libraries = [
            'DocumentClassifier',
            'corpus_metadata',
            'httpx',
            'spacy',
            'urllib3',
            'requests'
        ]
        
        for lib_name in noisy_libraries:
            lib_logger = logging.getLogger(lib_name)
            # Set to INFO so they still log to file, but not DEBUG spam
            lib_logger.setLevel(logging.INFO)
        
        # Log that setup is complete
        setup_logger = logging.getLogger('corpus_config')
        setup_logger.info("="*60)
        setup_logger.info("Logging system initialized")
        setup_logger.info(f"Log file: {log_file.absolute()}")
        setup_logger.info(f"File handler level: {log_config['level']}")
        setup_logger.info(f"Console output: {log_config.get('console', False)}")
        if log_config.get('console', False):
            setup_logger.info(f"Console level: {log_config.get('console_level', 'ERROR')}")
        setup_logger.info("="*60)
        
        if self.verbose:
            self._print_status(f"Logging to: {log_file}", "success")
    
    def _print_header(self):
        """Print the application header"""
        print(f"\n{Colors.HEADER}{'='*80}")
        print(f"  CORPUS METADATA SYSTEM - Configuration Loader")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}{Colors.ENDC}\n")
    
    def _print_section(self, title: str, char: str = '='):
        """Print a section header"""
        if not self.verbose:
            return
        print(f"\n{Colors.CYAN}{char*60}")
        print(f"  {title}")
        print(f"{char*60}{Colors.ENDC}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the main configuration file"""
        config_path = self.config_dir / self.config_file
        
        # Check if path exists
        if not config_path.exists():
            # Try parent directory
            parent_config = self.config_dir.parent / self.config_file
            if parent_config.exists():
                config_path = parent_config
            else:
                self._print_status(f"Config file not found: {config_path}", "error")
                return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                self._print_status(f"Loaded: {config_path.name}", "success")
                return config
        except Exception as e:
            self._print_status(f"Failed to load config: {e}", "error")
            return {}
    
    def _resolve_all_references(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all ${ref:...} references in the configuration"""
        def resolve_value(value: Any, depth: int = 0) -> Any:
            if depth > 10:
                return value
            
            if isinstance(value, str) and value.startswith('${ref:'):
                ref_path = value[6:-1]
                resolved = self._get_nested_value(config, ref_path)
                if resolved is not None:
                    return resolve_value(resolved, depth + 1)
                return value
            elif isinstance(value, dict):
                return {k: resolve_value(v, depth) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item, depth) for item in value]
            return value
        
        return resolve_value(config)
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get a nested value from config using dot notation"""
        parts = path.split('.')
        value = config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    def _print_config_summary(self):
        """Print a summary of loaded configuration"""
        if not self.verbose:
            return
        
        self._print_section("CONFIGURATION SUMMARY", "-")
        
        # Count enabled features
        features = self.get_feature_flags()
        enabled_count = sum(1 for v in features.values() if v)
        
        print(f"\n  {Colors.GREEN}✓{Colors.ENDC} Features enabled: {enabled_count}/{len(features)}")
        
        # Show key settings
        if 'output' in self._resolved_config:
            output = self._resolved_config['output']
            print(f"  {Colors.GREEN}✓{Colors.ENDC} Output format: {output.get('format', 'json')}")
        
        # Show logging status
        if hasattr(self, 'log_file'):
            print(f"  {Colors.GREEN}✓{Colors.ENDC} Logging configured: {self.log_file.name}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        parts = key.split('.')
        value = self._resolved_config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags"""
        return self._resolved_config.get('features', {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled"""
        return self.get_feature_flags().get(feature, False)
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return self._resolved_config.get('pipeline', {})
    
    def get_extraction_config(self) -> Dict[str, Any]:
        """Get extraction configuration"""
        return self._resolved_config.get('extraction', {})
    
    def get_abbreviations_config(self) -> Dict[str, Any]:
        """Get abbreviations configuration"""
        return self._resolved_config.get('abbreviations', {})
    
    def get_all_stages(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all pipeline stages configuration"""
        pipeline = self._resolved_config.get('pipeline', {})
        stages = pipeline.get('stages', [])
        # Return as list of tuples (stage_name, stage_config)
        return [(stage.get('name', 'unnamed'), stage) for stage in stages]
    
    def get_lexicon_config(self, lexicon_type: str) -> Dict[str, Any]:
        """Get configuration for a specific lexicon"""
        lexicons = self.get('lexicons', {})
        thresholds = self.get('thresholds', {})
        
        configs = {
            'drug': {
                'enabled': bool(lexicons.get('drug')),
                'path': lexicons.get('drug', ''),
                'min_term_length': thresholds.get('min_term_length', {}).get('drug', 3)
            },
            'disease': {
                'enabled': bool(lexicons.get('disease')),
                'path': lexicons.get('disease', ''),
                'min_term_length': thresholds.get('min_term_length', {}).get('disease', 4)
            },
            'medical_terms': {
                'enabled': bool(lexicons.get('medical_terms')),
                'path': lexicons.get('medical_terms', ''),
                'filter_mode': 'simple'
            }
        }
        
        return configs.get(lexicon_type, {})
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get API configuration"""
        return self._resolved_config.get('api_configuration', {}).get(api_name, {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        logging_cfg = self._resolved_config.get('logging', {})
        return {
            'level': logging_cfg.get('level', 'INFO'),  # Changed default to INFO
            'console': logging_cfg.get('console', False),
            'console_level': logging_cfg.get('console_level', 'ERROR'),
            'file': logging_cfg.get('file', 'corpus_logs/corpus.log')
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        output = self._resolved_config.get('output', {})
        include = output.get('include', {})
        
        return {
            'format': output.get('format', 'json'),
            'json_indent': output.get('json_indent', 2),
            'include_statistics': include.get('statistics', True),
            'include_metadata': include.get('metadata', True),
            'include_confidence': include.get('confidence_scores', True),
            'include_raw': include.get('raw_extractions', False)
        }
    
    def _print_status(self, message: str, status: str = "info"):
        """Print a processing status message with appropriate formatting"""
        if not self.verbose:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if status == "success":
            print(f"[{timestamp}] {Colors.GREEN}✓{Colors.ENDC} {message}")
        elif status == "error":
            print(f"[{timestamp}] {Colors.RED}✗{Colors.ENDC} {message}")
        elif status == "warning":
            print(f"[{timestamp}] {Colors.YELLOW}⚠{Colors.ENDC} {message}")
        elif status == "processing":
            print(f"[{timestamp}] {Colors.BLUE}➤{Colors.ENDC} {message}")
        else:
            print(f"[{timestamp}] {message}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete resolved configuration for debugging"""
        return self._resolved_config.copy()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'CorpusConfig',
    'CorpusLogger',
    'Colors',
    'get_logger',
    'singleton_logged',
    'log_separator',
    'log_metric',
    'load_corpus_config'
]

# Convenience function
def load_corpus_config(config_path: str = "corpus_config", verbose: bool = True) -> CorpusConfig:
    """Load corpus configuration with optional verbose output"""
    return CorpusConfig(config_dir=config_path, verbose=verbose)