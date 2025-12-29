#!/usr/bin/env python3
"""
Metadata System Initializer - ENHANCED VERSION (Version 15.0)
=============================================================================
Location: corpus_metadata/document_utils/metadata_system_initializer.py

CHANGES IN VERSION 15.0:
- Added comprehensive resource loading from config.yaml
- Support for TSV file loading (UMLS abbreviations)
- Display detailed statistics for each resource
- Dynamic resource loading based on config structure
- Show row/entry counts for all loaded resources
"""

import os
import yaml
import json
import csv
import sqlite3
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Union
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use centralized logging - no fallbacks
from corpus_metadata.document_utils.metadata_logging_config import (
    CorpusConfig,
    get_logger,
    timed_section,
    log_summary,
)

logger = get_logger('metadata_system_initializer')


 
class MetadataSystemInitializer:
    """
    System initialization for document metadata extraction.
    SINGLETON PATTERN: Only one instance exists across the entire application.
    Config.yaml is the single source of truth - no hardcoded defaults.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: str = "document_config/config.yaml"):
        """Singleton pattern: Ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetadataSystemInitializer, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "document_config/config.yaml"):
        """Initialize with configuration path."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self.config_path = Path(__file__).parent.parent / config_path
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            self.project_root = Path(__file__).parent.parent.parent
            self._initialize_data_structures()
            self._initialized = True
            self.initialize()
    
    def _initialize_data_structures(self):
        """Initialize all data structures."""
        self.config = {}
        self.models = {}
        self.resources = {}
        self.features = {}
        self.spacy_models_by_purpose = {}
        
        # Resource statistics
        self.resource_stats = {}
        
        # Lexicon data
        self.drug_lexicon = {}
        self.disease_lexicon = {}
        self.medical_terms_lexicon = set()
        self.medical_terms_normalized = set()
        
        # Abbreviation data
        self.abbreviations = {
            'medical': {},
            'alexion': {},
            'general': {},
            'umls_biological': [],
            'umls_clinical': []
        }
        
        # Investigational drugs - focus on interventionName
        self.investigational_drugs = []
        self.investigational_drug_names = set()
        self.investigational_drug_names_normalized = set()
        
        self.investigational_indices = {
            'by_intervention_name': defaultdict(list),
            'by_nct_id': defaultdict(list),
            'by_condition': defaultdict(list),
            'by_status': defaultdict(list),
            'combinations': defaultdict(list),
            'monotherapies': defaultdict(list)
        }
        
        self.lexicon_indices = {
            'drug': {
                'by_term': {},
                'by_normalized': {},
                'by_rxcui': {},
                'by_tty': defaultdict(list)
            },
            'disease': {
                'by_label': {},
                'by_normalized': {},
                'by_source': defaultdict(list),
                'by_id': {}
            },
            'medical_terms': {
                'by_term': set(),
                'by_normalized': set()
            }
        }
        
        self.status = {
            'config': False,
            'models': {},
            'resources': {},
            'lexicons': {},
            'investigational_drugs': False,
            'features': {},
            'errors': [],
            'warnings': []
        }
        
        # Loading counters for consistent output
        self._loading_counter = 0
        self._loading_total = 0
    
    # ========================================================================
    # CONSOLE OUTPUT HELPERS
    # ========================================================================
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format (MB)."""
        size_mb = size_bytes / (1024 * 1024)
        if size_mb >= 1.0:
            return f"{size_mb:.2f} MB"
        elif size_mb >= 0.01:
            return f"{size_mb:.2f} MB"
        else:
            size_kb = size_bytes / 1024
            return f"{size_kb:.2f} KB"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in HH:MM:SS format."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _print_loading_status(self, resource_name: str, status: str = "OK", 
                               size_bytes: int = 0, extra_info: str = ""):
        """
        Print loading status in consistent format.
        Format: [HH:MM:SS] [OK] Loading [N/TOTAL] resource_name (size)
        """
        from corpus_metadata.document_utils.console_colors import Colors
        
        timestamp = self._get_timestamp()
        counter_str = f"[{self._loading_counter:>2}/{self._loading_total}]"
        size_str = f"({self._format_size(size_bytes)})" if size_bytes > 0 else ""
        
        if status == "OK":
            status_icon = f"{Colors.GREEN}[OK]{Colors.ENDC}"
        elif status == "FAIL":
            status_icon = f"{Colors.RED}[FAIL]{Colors.ENDC}"
        elif status == "SKIP":
            status_icon = f"{Colors.YELLOW}[SKIP]{Colors.ENDC}"
        else:
            status_icon = f"[{status}]"
        
        info_str = f" {extra_info}" if extra_info else ""
        
        print(f"[{timestamp}] {status_icon} Loading {counter_str} {resource_name} {size_str}{info_str}")
    
    @classmethod
    def get_instance(cls, config_path: str = "document_config/config.yaml") -> 'MetadataSystemInitializer':
        """Get the singleton instance."""
        return cls(config_path)
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing only)."""
        with cls._lock:
            if cls._instance:
                logger.warning("Resetting MetadataSystemInitializer singleton")
                cls._instance._initialized = False
            cls._instance = None
 
    
    def initialize(self) -> 'MetadataSystemInitializer':
        """Run complete initialization."""
        if hasattr(self, '_fully_initialized') and self._fully_initialized:
            logger.debug("Already initialized")
            return self
        
        with timed_section("System initialization", logger, threshold=1.0):
            init_stats = {
                'config': None,
                'resources': 0,
                'models': 0,
                'features_enabled': 0,
                'investigational_drugs': 0,
                'warnings': 0,
                'errors': 0
            }
            
            try:
                # STEP 1: Load config (no defaults - config.yaml is required)
                logger.debug("Loading configuration...")
                self._load_config()
                init_stats['config'] = str(self.config_path.name)
                
                # STEP 2: EARLY API KEY CHECK - Validate Claude API key if AI features are enabled
                logger.debug("Checking Claude API key requirements...")
                self._validate_claude_api_key_early()
                
                # STEP 3: Load ALL resources from config dynamically
                logger.debug("Loading all resources from config...")
                total_resources = self._load_all_resources_from_config()
                init_stats['resources'] = total_resources
                
                # STEP 4: Load models from config
                logger.debug("Loading models...")
                model_count = self._load_models()
                init_stats['models'] = model_count
                
                # STEP 5: Configure features from config
                logger.debug("Configuring features...")
                feature_stats = self._configure_features()
                init_stats['features_enabled'] = len(feature_stats.get('enabled', []))
                
                init_stats['warnings'] = len(self.status.get('warnings', []))
                init_stats['errors'] = len(self.status.get('errors', []))
                
            except Exception as e:
                logger.error(f"Initialization failed: {e}")
                raise
            
            self._fully_initialized = True
        
        # Log summary
        log_summary(
            logger,
            "Metadata system ready",
            {
                'total_resources': init_stats['resources'],
                'models': init_stats['models'],
                'features': init_stats['features_enabled']
            }
        )
        
        return self
    
    def _load_config(self):
        """Load configuration from yaml - no defaults."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            if not self.config:
                raise ValueError(f"Empty or invalid config file: {self.config_path}")
        
        self.status['config'] = True
        logger.debug(f"Config loaded from {self.config_path}")
    
    def _validate_claude_api_key_early(self):
        """
        Validate Claude API key early in the initialization process.
        This happens immediately after loading config to fail fast if needed.
        """
        features = self.config.get('features', {})
        
        # Check if AI validation feature is enabled
        ai_validation_enabled = features.get('ai_validation', False)
        
        if ai_validation_enabled:
            # Check for Claude API key in environment
            api_key = os.getenv('CLAUDE_API_KEY')
            
            if not api_key:
                error_msg = (
                    f"AI validation enabled (ai_validation={ai_validation_enabled}) "
                    "but CLAUDE_API_KEY not set in environment. "
                    "Please set CLAUDE_API_KEY environment variable or disable AI features in config."
                )
                logger.error(error_msg)
                self.status['errors'].append(error_msg)
                raise ValueError(error_msg)
            
            # Optionally validate API key format (basic check)
            if len(api_key) < 10:  # Basic sanity check
                error_msg = "CLAUDE_API_KEY appears to be invalid (too short)"
                logger.error(error_msg)
                self.status['errors'].append(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Claude API key validated successfully")
            self.status['claude_api_validated'] = True
        else:
            logger.debug("AI validation disabled, Claude API key not required")
            self.status['claude_api_validated'] = False
    
    def _load_all_resources_from_config(self) -> int:
        """
        Load all resources defined in config.yaml dynamically.
        Processes the 'resources' section with the new naming convention.
        """
        from corpus_metadata.document_utils.console_colors import Colors
        from datetime import datetime
        
        total_loaded = 0
        
        # Get the dictionaries base path from config
        paths_config = self.config.get('paths', {})
        dictionaries_path = paths_config.get('dictionaries', 'corpus_dictionaries/output_datasources')
        
        # Resolve dictionaries path relative to project root
        if not Path(dictionaries_path).is_absolute():
            dictionaries_base = self.project_root / dictionaries_path
        else:
            dictionaries_base = Path(dictionaries_path)
        
        logger.debug(f"Dictionaries base path: {dictionaries_base}")
        
        # Process 'resources' section - this is now the main section
        if resources_config := self.config.get('resources'):
            # Count total resources first
            valid_resources = [(name, fname) for name, fname in resources_config.items() if fname]
            self._loading_total = len(valid_resources)
            self._loading_counter = 0
            
            # Print header
            timestamp = self._get_timestamp()
            print(f"\n[{timestamp}] {Colors.HEADER}Loading Resources ({self._loading_total} files){Colors.ENDC}")
            print(f"[{timestamp}] {'-' * 50}")
            
            for resource_name, resource_filename in valid_resources:
                self._loading_counter += 1
                
                # Resolve full path: dictionaries_base / filename
                full_path = dictionaries_base / resource_filename
                
                # Determine resource type based on naming convention
                special_type = None
                
                # Abbreviation resources
                if resource_name.startswith('abbreviation_'):
                    special_type = 'abbreviation'
                    abbrev_type = resource_name.replace('abbreviation_', '')
                    
                # Disease resources
                elif resource_name.startswith('disease_'):
                    if resource_name == 'disease_lexicon':
                        special_type = 'disease_lexicon'
                    elif 'lexicon' in resource_name:
                        # Supplemental disease lexicons (e.g., disease_lexicon_pah)
                        special_type = 'disease_lexicon_supplemental'
                    elif 'acronym' in resource_name:
                        special_type = 'disease_acronyms'
                    else:
                        special_type = 'disease_data'
                
                # Drug resources
                elif resource_name.startswith('drug_'):
                    if 'lexicon' in resource_name:
                        special_type = 'drug_lexicon'
                    elif 'investigational' in resource_name:
                        special_type = 'investigational_drugs'
                    elif 'alexion' in resource_name:
                        special_type = 'alexion_drugs'
                    elif 'fda' in resource_name:
                        special_type = 'fda_drugs'
                    # REMOVED: pattern handling
                    else:
                        special_type = 'drug_data'
                
                # Medical terms
                elif 'medical_terms' in resource_name:
                    special_type = 'medical_terms'
                
                # Clinical trial metadata
                elif 'clinical_trial' in resource_name:
                    special_type = 'clinical_trial'
                
                # Document types
                elif 'document_types' in resource_name:
                    special_type = 'document_types'
                
                count = self._load_resource_file(resource_name, str(full_path), special_type=special_type)
                if count > 0:
                    total_loaded += 1
            
            # Print footer
            timestamp = self._get_timestamp()
            print(f"[{timestamp}] {'-' * 50}")
        
        return total_loaded
    
    def _load_resource_file(self, name: str, path: str, special_type: Optional[str] = None) -> int:
        """
        Load a single resource file and return count of entries.
        Handles JSON, TSV files and SQLite databases.
        """
        try:
            resource_path = Path(path)
            
            if not resource_path.exists():
                self._print_loading_status(name, "FAIL", extra_info="NOT FOUND")
                logger.warning(f"Resource file not found: {name} at {path}")
                self.resource_stats[name] = {
                    'path': path,
                    'status': 'NOT_FOUND',
                    'count': 0,
                    'type': special_type
                }
                return 0
            
            # Get file size
            file_size = resource_path.stat().st_size
            
            # Check file extension to determine type
            suffix = resource_path.suffix.lower()
            
            if suffix in ['.json']:
                count = self._load_json_resource(name, resource_path, special_type)
            elif suffix in ['.tsv', '.txt']:
                count = self._load_tsv_resource(name, resource_path, special_type)
            elif suffix in ['.db', '.sqlite', '.sqlite3']:
                count = self._load_database_resource(name, resource_path, special_type)
            else:
                self._print_loading_status(name, "SKIP", file_size, f"Unknown type: {suffix}")
                logger.warning(f"Unknown resource type for {name}: {resource_path.suffix}")
                count = 0
            
            # Print loading status if successful
            if count > 0:
                self._print_loading_status(name, "OK", file_size)
            else:
                # Already printed in the loading method if it failed
                pass
            
            return count
            
        except Exception as e:
            self._print_loading_status(name, "FAIL", extra_info=str(e)[:50])
            logger.error(f"Failed to load resource {name}: {e}")
            self.resource_stats[name] = {
                'path': path,
                'status': 'ERROR',
                'count': 0,
                'error': str(e),
                'type': special_type
            }
            return 0
    
    def _load_json_resource(self, name: str, path: Path, special_type: Optional[str] = None) -> int:
        """
        Load a JSON resource and return count of entries.
        
        Intelligently counts entries based on resource type and structure.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine count based on data type and structure
            count = 0
            
            if isinstance(data, list):
                # Simple list - count items
                count = len(data)
                
            elif isinstance(data, dict):
                # Complex counting based on resource type
                
                # === DRUG RESOURCES ===
                if special_type == 'alexion_drugs':
                    # Count drugs in known_drugs section
                    if 'known_drugs' in data:
                        count = len(data['known_drugs'])
                    else:
                        count = len(data)
                        
                elif special_type == 'fda_drugs':
                    # FDA drugs might be in 'drugs', 'data', or 'entries'
                    if 'drugs' in data and isinstance(data['drugs'], (list, dict)):
                        count = len(data['drugs'])
                    elif 'data' in data and isinstance(data['data'], list):
                        count = len(data['data'])
                    elif 'entries' in data and isinstance(data['entries'], list):
                        count = len(data['entries'])
                    else:
                        # If it's a flat dict of drug entries
                        # Skip metadata keys
                        drug_keys = [k for k in data.keys() 
                                    if k not in ['metadata', 'version', 'source', 'created']]
                        count = len(drug_keys)
                        
                elif special_type == 'investigational_drugs':
                    # Similar to FDA drugs
                    if 'drugs' in data and isinstance(data['drugs'], (list, dict)):
                        count = len(data['drugs'])
                    elif 'trials' in data and isinstance(data['trials'], list):
                        count = len(data['trials'])
                    elif 'data' in data and isinstance(data['data'], list):
                        count = len(data['data'])
                    else:
                        count = len(data)
                        
                elif special_type == 'drug_lexicon':
                    # Drug lexicon might have various structures
                    if 'drugs' in data:
                        count = len(data['drugs'])
                    elif 'terms' in data:
                        count = len(data['terms'])
                    elif 'entries' in data:
                        count = len(data['entries'])
                    else:
                        # Count non-metadata keys
                        count = len([k for k in data.keys() 
                                if k not in ['metadata', 'version', 'source']])
                
                # === DISEASE RESOURCES ===
                elif special_type == 'disease_lexicon':
                    if 'diseases' in data:
                        count = len(data['diseases'])
                    elif 'terms' in data:
                        count = len(data['terms'])
                    elif 'entries' in data:
                        count = len(data['entries'])
                    else:
                        count = len(data)
                        
                elif special_type == 'disease_acronyms':
                    if 'acronyms' in data:
                        count = len(data['acronyms'])
                    elif 'abbreviations' in data:
                        count = len(data['abbreviations'])
                    else:
                        count = len(data)
                
                # === DOCUMENT TYPES ===
                elif special_type == 'document_types':
                    # Document types often have nested structure
                    if 'types' in data:
                        count = len(data['types'])
                    elif 'groups' in data:
                        # Count all types within groups
                        if isinstance(data['groups'], list):
                            count = len(data['groups'])
                        elif isinstance(data['groups'], dict):
                            # Sum up types in each group
                            total_types = 0
                            for group_name, group_data in data['groups'].items():
                                if isinstance(group_data, dict):
                                    if 'types' in group_data:
                                        total_types += len(group_data['types'])
                                    elif 'items' in group_data:
                                        total_types += len(group_data['items'])
                                    else:
                                        total_types += 1
                                elif isinstance(group_data, list):
                                    total_types += len(group_data)
                                else:
                                    total_types += 1
                            count = total_types if total_types > 0 else len(data['groups'])
                    else:
                        # Count top-level keys that aren't metadata
                        count = len([k for k in data.keys() 
                                if k not in ['metadata', 'version', 'description']])
                
                # === CLINICAL TRIALS ===
                elif special_type == 'clinical_trial':
                    if 'trials' in data:
                        count = len(data['trials'])
                    elif 'studies' in data:
                        count = len(data['studies'])
                    elif 'entries' in data:
                        count = len(data['entries'])
                    else:
                        count = len(data)
                
                # === MEDICAL TERMS ===
                elif special_type == 'medical_terms':
                    if 'terms' in data:
                        count = len(data['terms'])
                    elif 'vocabulary' in data:
                        count = len(data['vocabulary'])
                    elif 'entries' in data:
                        count = len(data['entries'])
                    else:
                        count = len(data)
                
                # === ABBREVIATIONS ===
                elif special_type and 'abbreviation' in special_type:
                    if 'abbreviations' in data:
                        count = len(data['abbreviations'])
                    elif 'abbrevs' in data:
                        count = len(data['abbrevs'])
                    elif 'entries' in data:
                        count = len(data['entries'])
                    elif 'mappings' in data:
                        count = len(data['mappings'])
                    else:
                        count = len(data)
                
                # === GENERIC PATTERNS ===
                else:
                    # Try common patterns for unknown types
                    common_keys = ['data', 'entries', 'items', 'records', 'terms', 
                                'elements', 'list', 'values', 'results']
                    
                    found_data = False
                    for key in common_keys:
                        if key in data and isinstance(data[key], (list, dict)):
                            count = len(data[key])
                            found_data = True
                            logger.debug(f"Found data in '{key}' field for {name}")
                            break
                    
                    if not found_data:
                        # Check if it's mostly data (not metadata)
                        metadata_keys = {'metadata', 'meta', 'info', 'version', 'source', 
                                        'created', 'updated', 'description', 'schema',
                                        'config', 'settings', '_metadata', '_info'}
                        data_keys = [k for k in data.keys() if k not in metadata_keys]
                        
                        if data_keys:
                            # If we have non-metadata keys, count them
                            count = len(data_keys)
                            logger.debug(f"Counting {count} non-metadata keys for {name}")
                        else:
                            # Last resort - count everything
                            count = len(data)
                            logger.warning(f"Using total key count ({count}) for {name} - structure unclear")
            else:
                # Not a list or dict - single item
                count = 1
            
            # Store in resources
            self.resources[name] = data
            
            # Process based on special type
            if special_type:
                self._process_resource_by_type(name, data, special_type)
            
            # Store statistics with more detail
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'LOADED',
                'count': count,
                'file_type': 'JSON',
                'data_structure': type(data).__name__,
                'resource_type': special_type or 'generic',
                'top_level_keys': list(data.keys()) if isinstance(data, dict) else None,
                'actual_count_source': self._determine_count_source(data, count, special_type)
            }
            
            logger.debug(f"Loaded {name}: {count} entries from {path.name}")
            return count
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {name}: {e}")
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'ERROR',
                'count': 0,
                'error': f"JSON decode error: {e}",
                'type': special_type
            }
            return 0
        except Exception as e:
            logger.error(f"Error loading JSON resource {name}: {e}")
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'ERROR',
                'count': 0,
                'error': str(e),
                'type': special_type
            }
            return 0

    def _determine_count_source(self, data: Any, count: int, special_type: Optional[str]) -> str:
        """
        Helper method to track where the count came from for debugging.
        """
        if isinstance(data, list):
            return "list_length"
        elif isinstance(data, dict):
            if special_type == 'alexion_drugs' and 'known_drugs' in data:
                return "known_drugs_section"
            elif special_type == 'document_types' and 'groups' in data:
                return "document_groups"
            elif any(key in data for key in ['data', 'entries', 'items', 'terms']):
                for key in ['data', 'entries', 'items', 'terms']:
                    if key in data:
                        return f"{key}_field"
            else:
                return "top_level_keys"
        return "single_item"
    
    def _load_tsv_resource(self, name: str, path: Path, special_type: Optional[str] = None) -> int:
        """Load a TSV resource and return count of entries."""
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    data.append(row)
            
            count = len(data)
            
            # Store in resources
            self.resources[name] = data
            
            # Process UMLS abbreviations specially
            if special_type == 'abbreviation' and 'umls' in name:
                abbrev_type = 'umls_biological' if 'biological' in name else 'umls_clinical'
                self.abbreviations[abbrev_type] = data
                logger.debug(f"Loaded {count} {abbrev_type} abbreviations")
            elif special_type:
                self._process_resource_by_type(name, data, special_type)
            
            # Store statistics
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'LOADED',
                'count': count,
                'file_type': 'TSV',
                'columns': list(data[0].keys()) if data else [],
                'resource_type': special_type or 'generic'
            }
            
            logger.debug(f"Loaded {name}: {count} entries from {path.name}")
            return count
            
        except Exception as e:
            logger.error(f"Error loading TSV resource {name}: {e}")
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'ERROR',
                'count': 0,
                'error': str(e),
                'type': special_type
            }
            return 0
    
    def _load_database_resource(self, name: str, path: Path, special_type: Optional[str] = None) -> int:
        """Load information about a SQLite database resource."""
        try:
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            table_counts = {}
            total_rows = 0
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                table_counts[table_name] = count
                total_rows += count
            
            conn.close()
            
            # Store database connection info (not the actual connection)
            self.resources[name] = {
                'type': 'database',
                'path': str(path),
                'tables': list(table_counts.keys()),
                'row_counts': table_counts
            }
            
            # Store statistics
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'LOADED',
                'count': total_rows,
                'file_type': 'SQLite',
                'tables': table_counts,
                'table_count': len(tables),
                'resource_type': special_type or 'database'
            }
            
            logger.debug(f"Loaded {name}: SQLite DB with {len(tables)} tables, {total_rows} total rows")
            return total_rows
            
        except Exception as e:
            logger.error(f"Error loading database resource {name}: {e}")
            self.resource_stats[name] = {
                'path': str(path),
                'status': 'ERROR',
                'count': 0,
                'error': str(e),
                'type': special_type
            }
            return 0
    
    def _process_resource_by_type(self, name: str, data: Any, special_type: str):
        """Process resource based on its special type."""
        
        # Abbreviation resources
        if special_type == 'abbreviation':
            abbrev_type = name.replace('abbreviation_', '')
            self.abbreviations[abbrev_type] = data
            self.status['resources'][f'abbreviation_{abbrev_type}'] = True
        
        # Drug lexicon
        elif special_type == 'drug_lexicon':
            self._process_drug_lexicon(data)
            self.status['lexicons']['drug'] = True
        
        # Disease lexicon
        elif special_type == 'disease_lexicon':
            self._process_disease_lexicon(data)
            self.status['lexicons']['disease'] = True
        
        # Supplemental disease lexicons (merged with main lexicon)
        elif special_type == 'disease_lexicon_supplemental':
            self._merge_supplemental_disease_lexicon(data, name)
            self.status['resources'][name] = True
        
        # Medical terms
        elif special_type == 'medical_terms':
            self._process_medical_terms(data)
            self.status['lexicons']['medical_terms'] = True
        
        # Investigational drugs
        elif special_type == 'investigational_drugs':
            self._process_investigational_drugs(data)
            self.status['investigational_drugs'] = True
        
        # Other drug data types
        elif special_type in ['alexion_drugs', 'fda_drugs']:
            self.status['resources'][name] = True
        
        # Disease data types
        elif special_type in ['disease_acronyms', 'disease_data']:
            self.status['resources'][name] = True
        
        # Clinical trial metadata
        elif special_type == 'clinical_trial':
            self.status['resources']['clinical_trial_metadata'] = True
        
        # Document types
        elif special_type == 'document_types':
            self.status['resources']['document_types'] = True
    
    def _process_drug_lexicon(self, data: Any):
        """Process drug lexicon data."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    term = item.get('term') or item.get('name') or item.get('drug_name', '')
                    if term:
                        self.drug_lexicon[term] = item
                elif isinstance(item, str):
                    self.drug_lexicon[item] = {'term': item}
        elif isinstance(data, dict):
            self.drug_lexicon = data
        
        # Build indices
        for term, info in self.drug_lexicon.items():
            normalized = term.lower().strip()
            self.lexicon_indices['drug']['by_term'][term] = info
            self.lexicon_indices['drug']['by_normalized'][normalized] = info
    
    def _process_disease_lexicon(self, data: Any):
        """Process disease lexicon data."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    label = item.get('label') or item.get('name') or item.get('disease_name', '')
                    if label:
                        self.disease_lexicon[label] = item
                elif isinstance(item, str):
                    self.disease_lexicon[item] = {'label': item}
        elif isinstance(data, dict):
            self.disease_lexicon = data
        
        # Build indices
        for label, info in self.disease_lexicon.items():
            normalized = label.lower().strip()
            self.lexicon_indices['disease']['by_label'][label] = info
            self.lexicon_indices['disease']['by_normalized'][normalized] = info
    
    def _merge_supplemental_disease_lexicon(self, data: Any, source_name: str):
        """
        Merge supplemental disease lexicon data into the main lexicon.
        
        This method adds entries from supplemental lexicons (e.g., PAH-specific)
        without replacing existing entries. If an entry already exists, the
        supplemental data is merged, with supplemental values taking precedence
        for any new fields while preserving existing fields.
        
        Args:
            data: The supplemental lexicon data (list or dict)
            source_name: Name of the supplemental resource for logging
        """
        added_count = 0
        updated_count = 0
        
        entries_to_merge = []
        
        # Normalize data to list of (label, info) tuples
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    label = item.get('label') or item.get('name') or item.get('disease_name', '')
                    if label:
                        entries_to_merge.append((label, item))
                elif isinstance(item, str):
                    entries_to_merge.append((item, {'label': item}))
        elif isinstance(data, dict):
            # Handle dict with 'diseases', 'entries', etc.
            if 'diseases' in data and isinstance(data['diseases'], (list, dict)):
                return self._merge_supplemental_disease_lexicon(data['diseases'], source_name)
            elif 'entries' in data and isinstance(data['entries'], list):
                return self._merge_supplemental_disease_lexicon(data['entries'], source_name)
            elif 'terms' in data and isinstance(data['terms'], list):
                return self._merge_supplemental_disease_lexicon(data['terms'], source_name)
            else:
                # Assume dict is label -> info mapping
                for label, info in data.items():
                    if label not in ['metadata', 'version', 'source', 'created', 'updated']:
                        if isinstance(info, dict):
                            entries_to_merge.append((label, info))
                        else:
                            entries_to_merge.append((label, {'label': label, 'value': info}))
        
        # Merge entries
        for label, info in entries_to_merge:
            normalized = label.lower().strip()
            
            # Mark source for traceability
            info['_supplemental_source'] = source_name
            
            if label in self.disease_lexicon:
                # Merge with existing entry
                existing = self.disease_lexicon[label]
                for key, value in info.items():
                    if key not in existing or not existing[key]:
                        existing[key] = value
                updated_count += 1
            else:
                # Add new entry
                self.disease_lexicon[label] = info
                added_count += 1
            
            # Update indices
            self.lexicon_indices['disease']['by_label'][label] = self.disease_lexicon[label]
            self.lexicon_indices['disease']['by_normalized'][normalized] = self.disease_lexicon[label]
        
        logger.info(f"Merged supplemental disease lexicon '{source_name}': "
                   f"{added_count} added, {updated_count} updated")
    
    def _process_medical_terms(self, data: Any):
        """Process medical terms data."""
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Try multiple possible field names
                    term = (item.get('term') or 
                           item.get('name') or 
                           item.get('medical_term') or
                           item.get('label') or
                           item.get('text') or
                           item.get('value', ''))
                    if term and isinstance(term, str):
                        self.medical_terms_lexicon.add(term)
                        self.medical_terms_normalized.add(term.lower().strip())
                elif isinstance(item, str):
                    self.medical_terms_lexicon.add(item)
                    self.medical_terms_normalized.add(item.lower().strip())
        elif isinstance(data, dict):
            # If it's a dict, try to extract terms from keys or specific fields
            if 'terms' in data and isinstance(data['terms'], list):
                self._process_medical_terms(data['terms'])
            elif 'data' in data and isinstance(data['data'], list):
                self._process_medical_terms(data['data'])
            else:
                # Otherwise treat keys as terms
                for key, value in data.items():
                    if isinstance(key, str) and key:
                        self.medical_terms_lexicon.add(key)
                        self.medical_terms_normalized.add(key.lower().strip())
        
        logger.debug(f"Processed medical terms: {len(self.medical_terms_lexicon)} unique terms")
    
    def _process_investigational_drugs(self, data: List[Dict[str, Any]]):
        """Process investigational drugs data."""
        if not isinstance(data, list):
            return
        
        self.investigational_drugs = data
        
        # Build indices focusing on interventionName
        for entry in self.investigational_drugs:
            intervention_name = entry.get('interventionName', '')
            if not intervention_name:
                continue
            
            # Add to name sets
            self.investigational_drug_names.add(intervention_name)
            self.investigational_drug_names_normalized.add(intervention_name.lower().strip())
            
            # Index by intervention name
            self.investigational_indices['by_intervention_name'][intervention_name].append(entry)
            
            # Categorize as combination or monotherapy
            if ' in combination with ' in intervention_name or ' + ' in intervention_name:
                self.investigational_indices['combinations'][intervention_name].append(entry)
            else:
                self.investigational_indices['monotherapies'][intervention_name].append(entry)
            
            # Index by NCT ID
            if nct_id := entry.get('nctId'):
                self.investigational_indices['by_nct_id'][nct_id].append(entry)
            
            # Index by conditions
            for condition in entry.get('conditions', []):
                self.investigational_indices['by_condition'][condition].append(entry)
            
            # Index by status
            if status := entry.get('overallStatus'):
                self.investigational_indices['by_status'][status].append(entry)
    

    
    
    
    
    def display_resource_statistics(self):
        """Display compact resource loading summary with timestamps in consistent format."""
        from datetime import datetime
        from corpus_metadata.document_utils.console_colors import Colors
        
        timestamp = self._get_timestamp()
        
        # Calculate totals by category using resource_stats
        categories = {
            'abbreviation': {'count': 0, 'loaded': 0, 'total': 3, 'label': 'Abbreviations'},
            'drug': {'count': 0, 'loaded': 0, 'total': 4, 'label': 'Drug Resources'},
            'disease': {'count': 0, 'loaded': 0, 'total': 4, 'label': 'Disease Resources'},
            'other': {'count': 0, 'loaded': 0, 'total': 3, 'label': 'Other Resources'}
        }
        
        for resource_name, stats in self.resource_stats.items():
            if stats.get('status') == 'LOADED':
                count = stats.get('count', 0)
                
                if resource_name.startswith('abbreviation_'):
                    categories['abbreviation']['count'] += count
                    categories['abbreviation']['loaded'] += 1
                elif resource_name.startswith('drug_'):
                    categories['drug']['count'] += count
                    categories['drug']['loaded'] += 1
                elif resource_name.startswith('disease_'):
                    categories['disease']['count'] += count
                    categories['disease']['loaded'] += 1
                else:
                    categories['other']['count'] += count
                    categories['other']['loaded'] += 1
        
        total_count = sum(c['count'] for c in categories.values())
        total_loaded = sum(c['loaded'] for c in categories.values())
        total_expected = sum(c['total'] for c in categories.values())
        
        # Print summary header
        print(f"\n[{timestamp}] {Colors.HEADER}Resource Loading Summary{Colors.ENDC}")
        print(f"[{timestamp}] {'Ã¢â€â‚¬' * 50}")
        
        # Print each category
        for key, cat in categories.items():
            status = f"{Colors.GREEN}[OK]{Colors.ENDC}" if cat['loaded'] == cat['total'] else f"{Colors.YELLOW}[!]{Colors.ENDC}"
            print(f"[{timestamp}] {status} {cat['label']:<18} {cat['count']:>8,} entries [{cat['loaded']:>2}/{cat['total']}]")
        
        # Print total
        print(f"[{timestamp}] {'Ã¢â€â‚¬' * 50}")
        total_status = f"{Colors.GREEN}[OK]{Colors.ENDC}" if total_loaded == total_expected else f"{Colors.YELLOW}[!]{Colors.ENDC}"
        print(f"[{timestamp}] {total_status} {'TOTAL':<18} {total_count:>8,} entries [{total_loaded:>2}/{total_expected}]")
    
    def _load_models(self) -> int:
        """Load models specified in config."""
        model_count = 0
        models_config = self.config.get('models', {})
        
        # Load SpaCy models if configured
        if spacy_models := models_config.get('spacy'):
            try:
                import spacy
                for model_spec in spacy_models:
                    if isinstance(model_spec, dict):
                        model_name = model_spec.get('name')
                        purpose = model_spec.get('purpose', model_name)
                    else:
                        model_name = model_spec
                        purpose = model_name
                    
                    try:
                        model = spacy.load(model_name)
                        self.models[model_name] = model
                        self.spacy_models_by_purpose[purpose] = model
                        model_count += 1
                        logger.debug(f"Loaded SpaCy model: {model_name}")
                    except Exception as e:
                        logger.warning(f"SpaCy model {model_name} not available: {e}")
                
                if model_count > 0:
                    self.status['models']['spacy'] = {
                        'loaded': True,
                        'count': model_count,
                        'models': list(self.models.keys())
                    }
            except ImportError:
                logger.warning("SpaCy not available")
        
        # Check Claude API if configured (already validated earlier)
        if claude_config := models_config.get('claude'):
            if os.getenv('CLAUDE_API_KEY'):
                self.status['models']['claude'] = {
                    'configured': True,
                    'model': claude_config.get('model'),
                    'api_validated': self.status.get('claude_api_validated', False)
                }
                model_count += 1
        
        return model_count
    
    def _configure_features(self) -> Dict[str, List[str]]:
        """Configure features from config."""
        enabled = []
        disabled = []
        
        features_config = self.config.get('features', {})
        
        for feature, is_enabled in features_config.items():
            if is_enabled:
                enabled.append(feature)
                self.features[feature] = True
            else:
                disabled.append(feature)
        
        self.status['features'] = {
            'enabled': enabled,
            'disabled': disabled
        }
        
        logger.debug(f"Features - Enabled: {len(enabled)}, Disabled: {len(disabled)}")
        
        return {'enabled': enabled, 'disabled': disabled}
    
    # ========================================================================
    # PUBLIC INTERFACE METHODS
    # ========================================================================
    def get_resource(self, resource_key: str) -> Optional[str]:
        """
        Get a resource path by key
        
        Args:
            resource_key: Key from config resources section
            
        Returns:
            Path string or None if not found
        """
        if not self.config or 'resources' not in self.config:
            return None
        
        resource = self.config['resources'].get(resource_key)
        
        # Handle both string paths and dict metadata objects
        if isinstance(resource, dict):
            return resource.get('path')
        elif isinstance(resource, str):
            return resource
        
        return None

    def get_lexicon(self, lexicon_name: str) -> Optional[Union[Dict, Set]]:
        """
        Get a lexicon by name
        
        Args:
            lexicon_name: Name of lexicon ('disease', 'drug', 'medical_terms')
            
        Returns:
            Lexicon dict/set or None if not found
        """
        lexicon_map = {
            'disease': self.disease_lexicon,
            'drug': self.drug_lexicon,
            'medical_terms': self.medical_terms_lexicon,
            'medical': self.medical_terms_lexicon,  # Alias
        }
        
        return lexicon_map.get(lexicon_name)
    
    def get_claude_client(self, tier: str = None):
        """
        Get or create a Claude client instance.
        
        The client is shared across all tiers - the tier parameter is for
        logging purposes and to validate the configuration exists.
        
        Args:
            tier: Optional model tier hint ("fast" or "validation")
                Used for logging; actual model selection happens at call time
        
        Returns:
            Anthropic client instance or None if API key not available
        """
        # Check if we already have a client
        if hasattr(self, '_claude_client') and self._claude_client is not None:
            return self._claude_client
        
        # Check for API key (support both env var names)
        api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY/CLAUDE_API_KEY not found in environment")
            return None
        
        try:
            from anthropic import Anthropic
            self._claude_client = Anthropic(api_key=api_key)
            
            # Log initialization with tier info
            if tier:
                model = self.get_claude_model(tier)
                logger.info(f"Claude client initialized (tier={tier}, model={model})")
            else:
                logger.info("Claude client initialized successfully")
            
            return self._claude_client
            
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            return None


    # ==============================================================================
    # ADD THESE NEW METHODS AFTER get_claude_client()
    # ==============================================================================

    def get_claude_config(self, tier: str = "fast") -> dict:
        """
        Get Claude configuration for specified tier from config.yaml.
        
        TWO-TIER STRATEGY:
        - "fast": Cheaper/faster model for basic tasks (classification, initial extraction)
        - "validation": Best model for final validation & enrichment
        
        Args:
            tier: "fast" or "validation"
            
        Returns:
            Dictionary with model, max_tokens, temperature
            
        Raises:
            ValueError: If config not loaded or tier config missing
        """
        if not hasattr(self, 'config') or not self.config:
            raise ValueError("Config not loaded. Call initialize() first.")
        
        api_config = self.config.get('api', {})
        claude_config = api_config.get('claude', {})
        
        if not claude_config:
            raise ValueError("No 'api.claude' section in config.yaml")
        
        # Get tier-specific config (required)
        tier_config = claude_config.get(tier, {})
        if not tier_config:
            raise ValueError(f"No 'api.claude.{tier}' section in config.yaml")
        
        # Model is required
        model = tier_config.get('model')
        if not model:
            raise ValueError(f"No 'model' specified in api.claude.{tier} config")
        
        return {
            "model": model,
            "max_tokens": tier_config.get('max_tokens', 4096),
            "temperature": tier_config.get('temperature', 0.0)
        }


    def get_claude_model(self, tier: str = "fast") -> str:
        """
        Get Claude model name for specified tier from config.yaml.
        
        Args:
            tier: "fast" or "validation"
            
        Returns:
            Model name string
        """
        config = self.get_claude_config(tier)
        return config["model"]


    def get_validation_config(self, entity_type: str) -> dict:
        """
        Get validation configuration for an entity type from config.yaml.
        
        Args:
            entity_type: "abbreviation", "drug", or "disease"
            
        Returns:
            Validation configuration dictionary
            
        Raises:
            ValueError: If config not loaded or validation config missing
        """
        if not hasattr(self, 'config') or not self.config:
            raise ValueError("Config not loaded. Call initialize() first.")
        
        validation_config = self.config.get('validation', {})
        if not validation_config:
            raise ValueError("No 'validation' section in config.yaml")
        
        entity_config = validation_config.get(entity_type, {})
        if not entity_config:
            raise ValueError(f"No 'validation.{entity_type}' section in config.yaml")
        
        return entity_config
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled in config."""
        return self.features.get(feature, False)
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self.models.get(model_name)
    
    def get_resource(self, resource_name: str) -> Optional[Any]:
        """Get a loaded resource by name."""
        return self.resources.get(resource_name)
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about all loaded resources."""
        return self.resource_stats.copy()
    
    def get_abbreviations(self, abbrev_type: str = None) -> Union[Dict, List]:
        """Get abbreviations by type or all abbreviations."""
        if abbrev_type:
            return self.abbreviations.get(abbrev_type, {})
        return self.abbreviations
    
    def is_medical_term(self, term: str) -> bool:
        """Check if a term exists in the medical terms lexicon."""
        return (term in self.medical_terms_lexicon or 
                term.lower() in self.medical_terms_normalized)
    
    def is_investigational_drug(self, drug_name: str) -> bool:
        """Check if a drug name is in the investigational drugs list."""
        return (drug_name in self.investigational_drug_names or 
                drug_name.lower() in self.investigational_drug_names_normalized)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the initialized system."""
        return {
            'config_loaded': self.status['config'],
            'total_resources': len(self.resource_stats),
            'resources_loaded': sum(1 for s in self.resource_stats.values() if s['status'] == 'LOADED'),
            'total_data_entries': sum(s.get('count', 0) for s in self.resource_stats.values()),
            'models_count': len(self.models),
            'features_enabled': len([f for f in self.features.values() if f]),
            'lexicons': {
                'drugs': len(self.drug_lexicon),
                'diseases': len(self.disease_lexicon),
                'medical_terms': len(self.medical_terms_lexicon),
                'investigational_drugs': len(self.investigational_drug_names)
            },
            'abbreviations': {
                'medical': len(self.abbreviations.get('medical', {})),
                'alexion': len(self.abbreviations.get('alexion', {})),
                'general': len(self.abbreviations.get('general', {})),
                'umls_biological': len(self.abbreviations.get('umls_biological', [])),
                'umls_clinical': len(self.abbreviations.get('umls_clinical', []))
            },
            'claude_api_validated': self.status.get('claude_api_validated', False),
            'warnings': len(self.status['warnings']),
            'errors': len(self.status['errors']),
            'ready': len(self.status['errors']) == 0
        }