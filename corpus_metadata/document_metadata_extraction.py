#!/usr/bin/env python3
"""
Documents SOTA Extractor - COMPLETE METADATA EXTRACTION PIPELINE
================================================================
 
Location: corpus_metadata/document_metadata_extraction.py

Pipeline Overview:

1. ABBREVIATION EXTRACTION (First)
   |-- Extract all abbreviations with expansions
   |-- Classify context (biological/disease/drug/clinical)
   `-- Output: {abbr, expansion, confidence, type, occurrences}
   
2. CITATION & REFERENCE EXTRACTION (New Stage)
   |-- Citation Extractor:
   |   |-- Detect citation styles (Vancouver, APA, Harvard, etc.)
   |   |-- Extract structured citations from references
   |   `-- Link inline citations to references
   |-- Person Extractor:
   |   |-- Extract authors, PIs, investigators
   |   |-- Handle Unicode names (Garcia-Lopez, O'Connor)
   |   |-- Extract ORCIDs and affiliations
   |   `-- Build co-author network
   `-- Reference Extractor:
       |-- Extract 60+ identifier types
       |-- Reconstruct URLs
       `-- Classify reference roles

3. ENRICHMENT PHASE
   |-- Drug Extractor receives:
   |   `-- Abbreviations where type IN (drug, clinical, biological)
   |       + Their full expansions
   `-- Disease Extractor receives:
       `-- Abbreviations where type IN (disease, clinical, biological)
           + Their full expansions

4. ENTITY EXTRACTION (Enhanced)
   |-- Drug Detector runs with:
   |   |-- Original text
   |   |-- Abbreviation candidates
   |   `-- Expanded forms
   `-- Disease Detector runs with:
       |-- Original text
       |-- Abbreviation candidates
       `-- Expanded forms

5. VALIDATION & DEDUPLICATION
   |-- Cross-validate findings
   |-- Resolve conflicts (MPA case)
   `-- Merge duplicates

6. INTELLIGENT RENAMING + PREFIX APPLICATION
   |-- Generate intelligent filename based on content
   |-- Apply auto-incrementing prefix (01000_, 01001_, etc.)
   `-- Rename file with new identifier
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.cli.info")
warnings.filterwarnings("ignore", category=FutureWarning, message="Possible nested set")

# =============================================================================================================================
# ENHANCED CONSOLE OUTPUT
# =============================================================================================================================
from corpus_metadata.document_utils.console_colors import Colors

# =============================================================================================================================
# EARLY SYSTEM INITIALIZATION
# =============================================================================================================================
# CorpusConfig  : Configuration manager - loads settings from config.yaml
# get_logger    : Factory function - creates configured logger instances 

from corpus_metadata.document_utils.metadata_logging_config import CorpusConfig, get_logger

config = CorpusConfig(config_dir=Path(__file__).parent / "document_config")
logger = get_logger('sota_extractor')
logger.debug("Centralized logging system initialized")

# =============================================================================================================================
# MODULE LOADING REGISTRY
# =============================================================================================================================
# Tracks initialization status of all pipeline components.
# Each module is set to True once successfully loaded.
#
# MODULE DESCRIPTIONS:
# -----------------------------------------------------------------------------------------------------------------------------
# system_initializer    : Singleton that loads config, resources, lexicons, and validates API keys. Central initialization hub.
# prefix_manager        : Manages auto-incrementing file prefixes (01000_, 01001_, etc.) for organized file naming.
# document_reader       : Reads PDF/DOCX/TXT files, extracts raw text, handles OCR for scanned documents.
# document_classifier   : Classifies document type (CSR, protocol, SmPC, IB, manuscript, etc.) using patterns and AI.
# document_router       : Routes documents to type-specific extractors based on classification results.
# basic_extractor       : Extracts core metadata: title, date, description, document structure, and section boundaries.
# drug_extractor        : Detects drug entities using RxNorm, FDA, and investigational drug databases. Validates via PubTator.
# disease_extractor     : Detects disease entities using Orphanet, DOID, SNOMED-CT. Enriches with rare disease identifiers.
# abbreviation_extractor: Extracts abbreviations and their expansions. Classifies context (biological/clinical/drug).
# entity_extraction     : Two-stage extraction orchestrator. Coordinates all entity extractors and handles deduplication.
# citation_extractor    : Extracts inline citations and detects citation style (Vancouver, APA, Harvard, etc.).
# person_extractor      : Extracts author names, PIs, investigators. Handles ORCIDs, affiliations, Unicode names.
# reference_extractor   : Extracts bibliographic references with 60+ ID types (DOI, PMID, NCT, URL, etc.).
# intelligent_renamer   : Generates descriptive filenames from extracted metadata (disease, drug, document type, date).
# =============================================================================================================================

modules_loaded = {
    'system_initializer': False,
    'prefix_manager': False,
    'document_reader': False,
    'document_classifier': False,
    'document_router': False,
    'basic_extractor': False,
    'drug_extractor': False,
    'disease_extractor': False,
    'abbreviation_extractor': False,
    'entity_extraction': False,
    'citation_extractor': False,  
    'person_extractor': False,    
    'reference_extractor': False,
    'intelligent_renamer': False,
}

# =============================================================================================================================
# CONSOLE OUTPUT HELPERS - UNIFIED FORMAT
# =============================================================================================================================
def get_timestamp() -> str:
    """Get current timestamp in HH:MM:SS format."""
    return datetime.now().strftime("%H:%M:%S")

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 10240:  # < 10 KB
        return f"{size_bytes:,} B"
    elif size_bytes < 1048576:  # < 1 MB
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / 1048576:.2f} MB"

def print_status(message: str, status: str = "OK", counter: str = "", size_info: str = ""):
    """
    Print status message with timestamp in unified format.
    Format: [HH:MM:SS] [STATUS] message [counter] (size)
    """
    timestamp = get_timestamp()
    status_icons = {
        "OK": f"{Colors.GREEN}[OK]{Colors.ENDC}",
        "FAIL": f"{Colors.RED}[FAIL]{Colors.ENDC}",
        "WARN": f"{Colors.YELLOW}[WARN]{Colors.ENDC}",
        "SKIP": f"{Colors.YELLOW}[SKIP]{Colors.ENDC}",
        "INFO": f"{Colors.CYAN}[INFO]{Colors.ENDC}",
    }
    icon = status_icons.get(status, f"[{status}]")
    
    # Build output string
    parts = [f"[{timestamp}]", icon, message]
    if counter:
        parts.append(counter)
    if size_info:
        parts.append(f"({size_info})")
    
    print(" ".join(parts))

def print_section_header(title: str):
    """Print a section header in unified format."""
    timestamp = get_timestamp()
    separator = "." * 60
    print(f"\n[{timestamp}] {Colors.BRIGHT_BLACK}{separator}{Colors.ENDC}")
    print(f"[{timestamp}] {Colors.BRIGHT_CYAN}{title}{Colors.ENDC}")
    print(f"[{timestamp}] {Colors.BRIGHT_BLACK}{separator}{Colors.ENDC}")

# =============================================================================================================================
# MODULE DEFINITIONS
# =============================================================================================================================
# Format: (key, module_path, class_name, is_critical, is_singleton)
# - is_critical : If True, exit on failure
# - is_singleton: If True, call .get_instance() after import
# =============================================================================================================================

module_definitions = [
    # Core modules (critical)
    ('system_initializer', 'corpus_metadata.document_utils.metadata_system_initializer', 'MetadataSystemInitializer', True, True),
    ('prefix_manager', 'corpus_metadata.document_utils.prefix_manager', 'DocumentPrefixManager', True, False),
    
    # Document processing modules
    ('document_reader', 'corpus_metadata.document_metadata_reader', 'DocumentReader', True, False),
    ('document_classifier', 'corpus_metadata.document_utils.metadata_classifier', 'DocumentClassifier', False, False),
    ('document_router', 'corpus_metadata.document_utils.metadata_document_type_router', 'DocumentTypeRouter', False, False),
    
    # Extraction modules
    ('basic_extractor', 'corpus_metadata.document_metadata_extraction_basic', 'RareDiseaseMetadataExtractor', False, False),
    ('drug_extractor', 'corpus_metadata.document_metadata_extraction_drug', 'DrugMetadataExtractor', False, False),
    ('disease_extractor', 'corpus_metadata.document_metadata_extraction_disease', 'DiseaseMetadataExtractor', False, False),
    ('abbreviation_extractor', 'corpus_metadata.document_metadata_extraction_abbreviation', 'AbbreviationExtractor', False, False),
    ('entity_extraction', 'corpus_metadata.document_utils.entity_extraction', 'process_document_two_stage', True, False),
    
    # Citation/Reference modules
    ('citation_extractor', 'corpus_metadata.document_metadata_extraction_citation', 'CitationExtractor', False, False),
    ('person_extractor', 'corpus_metadata.document_metadata_extraction_persons', 'PersonExtractor', False, False),
    ('reference_extractor', 'corpus_metadata.document_metadata_extraction_references', 'ReferenceExtractor', False, False),
    
    # File operations
    ('intelligent_renamer', 'corpus_metadata.document_intelligent_renamer', 'IntelligentDocumentRenamer', False, False),
]

# =============================================================================================================================
# MODULE LOADING
# =============================================================================================================================

print_section_header("INITIALIZING SYSTEM COMPONENTS")

loaded_classes = {}
system_initializer_instance = None
total_modules = len(module_definitions)

for i, (key, module_path, class_name, is_critical, is_singleton) in enumerate(module_definitions, 1):
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        loaded_classes[class_name] = cls
        modules_loaded[key] = True
        
        # Handle singleton pattern for MetadataSystemInitializer
        if is_singleton:
            system_initializer_instance = cls.get_instance()
        
        print_status(f"Loading {key}", "OK", f"[{i:>2}/{total_modules}]")
        logger.debug(f"Loaded module {key} successfully")
        
    except Exception as e:
        print_status(f"Loading {key} - {e}", "WARN", f"[{i:>2}/{total_modules}]")
        logger.error(f"Failed to load module {key}: {e}")
        
        if is_critical:
            print_status(f"CRITICAL: {key} is required. Exiting.", "FAIL")
            logger.error(f"Critical module {key} failed to load. Exiting.")
            sys.exit(1)

# =============================================================================================================================
# EXTRACT LOADED CLASSES
# =============================================================================================================================

MetadataSystemInitializer = loaded_classes.get('MetadataSystemInitializer')
DocumentPrefixManager = loaded_classes.get('DocumentPrefixManager')
DocumentReader = loaded_classes.get('DocumentReader')
DocumentClassifier = loaded_classes.get('DocumentClassifier')
DocumentTypeRouter = loaded_classes.get('DocumentTypeRouter')
RareDiseaseMetadataExtractor = loaded_classes.get('RareDiseaseMetadataExtractor')
DrugMetadataExtractor = loaded_classes.get('DrugMetadataExtractor')
DiseaseMetadataExtractor = loaded_classes.get('DiseaseMetadataExtractor')
AbbreviationExtractor = loaded_classes.get('AbbreviationExtractor')
process_document_two_stage = loaded_classes.get('process_document_two_stage')
CitationExtractor = loaded_classes.get('CitationExtractor')   
PersonExtractor = loaded_classes.get('PersonExtractor')       
ReferenceExtractor = loaded_classes.get('ReferenceExtractor')
IntelligentDocumentRenamer = loaded_classes.get('IntelligentDocumentRenamer')

# Get utility function from prefix_manager module
from corpus_metadata.document_utils.prefix_manager import apply_prefix_to_renamed_file

# ============================================================================
# ENHANCED DOCUMENT PROCESSOR CLASS
# ============================================================================

class EnhancedDocumentProcessor:
    """Enhanced document processor with rich console output"""
    
    def __init__(self):
        self.start_time = time.time()
        self.files_processed = 0
        self.total_files = 0
        self.current_file = None
        self.successful = 0
        self.failed = 0
        self.total_drugs = 0
        self.total_diseases = 0
        self.total_abbreviations = 0
        self.total_citations = 0      
        self.total_persons = 0        
        self.total_references = 0     
        self.texts_saved = 0
        self.files_renamed = 0
        self.files_prefixed = 0
        
    
    def print_section(self, title: str):
        """Print a section separator using unified format"""
        print_section_header(title)
        
    
    def print_loading_data(self):
        """
        Print loading status for all data resources defined in config.yaml.
        
        Dynamically reads resource names from config - DRY principle.
        Format: [HH:MM:SS] [OK] Loading [n/total] resource_name (size)
        """
        self.print_section("DATA SOURCES")
        
        # ======================================================================
        # GET RESOURCES DYNAMICALLY FROM CONFIG SCHEMA
        # ======================================================================
        try:
            resources_config = config.schema.resources
            if not resources_config:
                print_status("No resources configured", "WARN")
                logger.warning("No resources found in config schema")
                return
        except AttributeError as e:
            logger.warning(f"Unable to access resources from config schema: {e}")
            print_status("Unable to read resources from config", "WARN")
            return
        
        # ======================================================================
        # COLLECT ALL RESOURCE KEYS
        # ======================================================================
        resource_keys = []
        
        for resource_key in dir(resources_config):
            # Skip private/magic attributes
            if resource_key.startswith('_'):
                continue
            
            # Skip methods and non-string values
            value = getattr(resources_config, resource_key, None)
            if callable(value) or value is None:
                continue
            
            resource_keys.append(resource_key)
        
        # Sort alphabetically for consistent ordering
        resource_keys.sort()
        total_count = len(resource_keys)
        
        # ======================================================================
        # DISPLAY RESOURCES WITH CONSISTENT FORMAT
        # ======================================================================
        loaded_count = 0
        
        for i, resource_key in enumerate(resource_keys, 1):
            try:
                path_obj = config.get_resource_path(resource_key)
                
                if path_obj and path_obj.exists():
                    size_bytes = path_obj.stat().st_size
                    size_str = format_size(size_bytes)
                    
                    print_status(f"Loading {resource_key}", "OK", f"[{i:>2}/{total_count}]", size_str)
                    logger.debug(f"Loaded {resource_key} from {path_obj}")
                    loaded_count += 1
                else:
                    print_status(f"Loading {resource_key}", "WARN", f"[{i:>2}/{total_count}]", "not found")
                    logger.warning(f"{resource_key} not found at {path_obj}")
                    
            except (AttributeError, KeyError) as e:
                print_status(f"Loading {resource_key}", "FAIL", f"[{i:>2}/{total_count}]", "error")
                logger.debug(f"{resource_key} access error: {e}")
        
        # Summary line
        if loaded_count == total_count:
            print_status(f"Data sources: {loaded_count}/{total_count} loaded", "OK")
        else:
            print_status(f"Data sources: {loaded_count}/{total_count} loaded", "WARN")
        
        logger.info(f"Data sources loaded: {loaded_count}/{total_count}")
    
    
    def print_initialization_summary(self, modules_loaded: Dict[str, bool]):
        """Print initialization summary to console AND log."""
        loaded = sum(1 for v in modules_loaded.values() if v)
        total = len(modules_loaded)
        features = config.get_feature_flags()
        enabled = sum(1 for v in features.values() if v)
        
        print_section_header("INITIALIZATION SUMMARY")
        print_status(f"Modules loaded: {loaded}/{total}", "OK")
        print_status(f"Features enabled: {enabled}/{len(features)}", "OK")
        
        # Show which features are enabled - ALL of them, no truncation
        enabled_features = [k for k, v in features.items() if v]
        disabled_features = [k for k, v in features.items() if not v]
        
        if enabled_features:
            timestamp = get_timestamp()
            print(f"[{timestamp}]     Enabled: {', '.join(enabled_features)}")
        
        if disabled_features:
            timestamp = get_timestamp()
            print(f"[{timestamp}]     Disabled: {', '.join(disabled_features)}")
        
        # Also log for file output
        logger.debug(f"System initialized: {loaded}/{total} modules, {enabled}/{len(features)} features")

    def print_files_to_process(self, files: List[Path]):
        """Print files to be processed - ALL files listed"""
        self.total_files = len(files)
        self.print_section("FILES TO PROCESS")
        
        print_status(f"Total files: {len(files)}", "INFO")
        print_status("Order: Alphabetical", "INFO")
        
        # List ALL files, no truncation
        for i, filepath in enumerate(files, 1):
            timestamp = get_timestamp()
            print(f"[{timestamp}]     {i}. {filepath.name}")
    
    def print_file_processing_start(self, file_num: int, file_path: Path):
        """Print file processing start"""
        self.current_file = file_path.name
        self.files_processed = file_num
        size_str = format_size(file_path.stat().st_size)
        
        timestamp = get_timestamp()
        print(f"\n[{timestamp}] {Colors.BRIGHT_CYAN}[{file_num}/{self.total_files}]{Colors.ENDC} Processing: {file_path.name} ({size_str})")
    
    def print_stage_progress(self, stage: str, sequence: int, task: str, 
                            success: bool, detail: str = "", elapsed: float = 0):
        """Print stage progress"""
        status = "OK" if success else "FAIL"
        time_str = f"({elapsed:.1f}s)" if elapsed > 0 else ""
        detail_str = f" {detail}" if detail else ""
        print_status(f"{task}{detail_str} {time_str}", status)
    
    def print_extraction_results(self, results: Dict):
        """Print extraction results summary - UPDATED WITH NEW EXTRACTORS"""
        timestamp = get_timestamp()
        print(f"[{timestamp}] {Colors.CYAN}Extraction Results:{Colors.ENDC}")
        
        # Extract from pipeline_stages
        entities_stage = next(
            (stage for stage in results.get('pipeline_stages', []) 
            if stage['stage'] == 'entities'), 
            None
        )
        
        if entities_stage and entities_stage.get('results'):
            entity_results = entities_stage['results']
            drugs = entity_results.get('drugs', [])
            diseases = entity_results.get('diseases', [])
            abbreviations = entity_results.get('abbreviations', [])
            citations = entity_results.get('citations', [])
            persons = entity_results.get('persons', [])
            references = entity_results.get('references', [])
        else:
            drugs = results.get('drugs', [])
            diseases = results.get('diseases', [])
            abbreviations = results.get('abbreviations', [])
            citations = results.get('citations', [])
            persons = results.get('persons', [])
            references = results.get('references', [])
        
        self.total_drugs += len(drugs)
        self.total_diseases += len(diseases)
        self.total_abbreviations += len(abbreviations)
        self.total_citations += len(citations)
        self.total_persons += len(persons)
        self.total_references += len(references)
        
        # Print results in unified format
        print_status(f"Drugs found: {len(drugs)}", "INFO")
        if drugs and len(drugs) <= 3:
            for drug in drugs[:3]:
                name = drug.get('name', 'Unknown')
                conf = drug.get('confidence', 0)
                timestamp = get_timestamp()
                print(f"[{timestamp}]       - {name} ({conf:.0%})")
        
        print_status(f"Diseases found: {len(diseases)}", "INFO")
        if diseases and len(diseases) <= 3:
            for disease in diseases[:3]:
                name = disease.get('name', 'Unknown')
                conf = disease.get('confidence', 0)
                timestamp = get_timestamp()
                print(f"[{timestamp}]       - {name} ({conf:.0%})")
        
        print_status(f"Abbreviations found: {len(abbreviations)}", "INFO")
        if abbreviations and len(abbreviations) <= 3:
            for abbrev in abbreviations[:3]:
                abbr = abbrev.get('abbreviation', 'Unknown')
                exp = abbrev.get('expansion', 'Unknown')
                timestamp = get_timestamp()
                print(f"[{timestamp}]       - {abbr} -> {exp}")
        
        print_status(f"Citations found: {len(citations)}", "INFO")
        if citations and len(citations) <= 3:
            for citation in citations[:3]:
                if isinstance(citation, dict):
                    authors = citation.get('authors', [])
                    if authors and isinstance(authors, list) and len(authors) > 0:
                        first_author = authors[0].get('last_name', 'Unknown') if isinstance(authors[0], dict) else 'Unknown'
                    else:
                        first_author = 'Unknown'
                    year = citation.get('year', 'N/A')
                    journal = citation.get('journal', 'Unknown')[:20]
                    timestamp = get_timestamp()
                    print(f"[{timestamp}]       - {first_author} et al. ({year}) - {journal}")
        
        print_status(f"Persons found: {len(persons)}", "INFO")
        if persons and len(persons) <= 3:
            for person in persons[:3]:
                if isinstance(person, dict):
                    name_data = person.get('name', {})
                    if isinstance(name_data, dict):
                        full_name = name_data.get('full_name', 'Unknown')
                    else:
                        full_name = str(name_data)
                    role = person.get('role', 'unknown')
                    timestamp = get_timestamp()
                    print(f"[{timestamp}]       - {full_name} ({role})")
        
        print_status(f"References found: {len(references)}", "INFO")
        if references and len(references) <= 3:
            for reference in references[:3]:
                if isinstance(reference, dict):
                    ref_type = reference.get('reference_type', 'unknown')
                    normalized = reference.get('normalized_value', 'Unknown')[:30]
                    timestamp = get_timestamp()
                    print(f"[{timestamp}]       - {ref_type}: {normalized}")
    
    def print_file_complete(self, success: bool = True, error_msg: str = ""):
        """Print file completion status"""
        if success:
            self.successful += 1
            print_status("Completed successfully", "OK")
        else:
            self.failed += 1
            print_status(f"Failed: {error_msg[:50]}", "FAIL")
    
    def print_rename_status(self, old_name: str, new_name: str, was_prefixed: bool):
        """Print file rename status"""
        if old_name != new_name:
            self.files_renamed += 1
            if was_prefixed:
                self.files_prefixed += 1
                print_status(f"Renamed + Prefixed: {new_name}", "OK")
            else:
                print_status(f"Renamed: {new_name}", "OK")
            logger.info(f"File renamed: {old_name} -> {new_name}")
    
    def print_final_summary(self):
        """Print final processing summary - UPDATED"""
        elapsed = time.time() - self.start_time
        
        print_section_header("PROCESSING COMPLETE")
        
        print_status(f"Files processed: {self.files_processed}/{self.total_files}", "INFO")
        print_status(f"Successful: {self.successful}", "OK")
        if self.failed > 0:
            print_status(f"Failed: {self.failed}", "FAIL")
        else:
            print_status(f"Failed: {self.failed}", "OK")
        print_status(f"Text files saved: {self.texts_saved}", "INFO")
        print_status(f"Files renamed: {self.files_renamed}", "INFO")
        print_status(f"Files prefixed: {self.files_prefixed}", "INFO")
        
        timestamp = get_timestamp()
        print(f"\n[{timestamp}] {Colors.CYAN}Entities Extracted:{Colors.ENDC}")
        print_status(f"Drugs: {self.total_drugs}", "INFO")
        print_status(f"Diseases: {self.total_diseases}", "INFO")
        print_status(f"Abbreviations: {self.total_abbreviations}", "INFO")
        print_status(f"Citations: {self.total_citations}", "INFO")
        print_status(f"Persons: {self.total_persons}", "INFO")
        print_status(f"References: {self.total_references}", "INFO")
        
        print_status(f"Time elapsed: {elapsed:.1f} seconds", "INFO")
        print_status("All processing complete!", "OK")

console = EnhancedDocumentProcessor()

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_extraction_system():
    """Initialize the extraction system components - UPDATED WITH NEW EXTRACTORS"""
    try:
        logger.debug("Initializing extraction system components...")
        
        features = config.get_feature_flags()
        components = {
            'system_initializer': system_initializer_instance
        }
        
        # Check Claude availability
        use_claude = features.get('ai_validation', False)
        claude_api_key = os.getenv('CLAUDE_API_KEY')
        claude_available = use_claude and bool(claude_api_key)
        
        if use_claude:
            if claude_api_key:
                print_status("Claude validation: ENABLED", "OK")
                logger.info("Claude validation enabled")
            else:
                print_status("Claude validation: DISABLED (no API key)", "WARN")
                logger.warning("ai_validation is true but CLAUDE_API_KEY not found")
                claude_available = False
        else:
            print_status("Claude validation: DISABLED (config: ai_validation=false)", "WARN")
            logger.info("Claude validation disabled by config")
            claude_available = False
        
        # Initialize DocumentReader (required)
        if DocumentReader:
            try:
                components['document_reader'] = DocumentReader()
                logger.debug("Initialized DocumentReader")
            except Exception as e:
                print_status(f"Failed to initialize DocumentReader: {e}", "FAIL")
                logger.error(f"DocumentReader initialization failed: {e}")
                return {}
        else:
            print_status("DocumentReader class not available", "FAIL")
            logger.error("DocumentReader class not loaded")
            return {}
        
        # Initialize optional components
        if DocumentClassifier and features.get('classification', True):
            try:
                components['classifier'] = DocumentClassifier()
                logger.debug("Initialized DocumentClassifier")
            except Exception as e:
                logger.warning(f"DocumentClassifier initialization failed: {e}")
        
        if DocumentTypeRouter:
            try:
                components['router'] = DocumentTypeRouter(model_loader=system_initializer_instance)
                logger.debug("Initialized DocumentTypeRouter")
                
                # Reuse extractors from router's generic_extractor if available
                if hasattr(components['router'], 'generic_extractor') and components['router'].generic_extractor:
                    generic = components['router'].generic_extractor
                    
                    components['basic_extractor'] = generic
                    logger.debug("Reusing RareDiseaseMetadataExtractor from router")
                    
                    if hasattr(generic, 'drug_extractor') and generic.drug_extractor:
                        components['drug_extractor'] = generic.drug_extractor
                        logger.debug("Reusing DrugMetadataExtractor from router's extractor")
                    
                    if hasattr(generic, 'disease_extractor') and generic.disease_extractor:
                        components['disease_extractor'] = generic.disease_extractor
                        logger.debug("Reusing DiseaseMetadataExtractor from router's extractor")
                    
                    if 'classifier' not in components and hasattr(generic, 'classifier') and generic.classifier:
                        components['classifier'] = generic.classifier
                        logger.debug("Reusing DocumentClassifier from router's extractor")
                        
            except Exception as e:
                logger.warning(f"DocumentTypeRouter initialization failed: {e}")
        
        # Only create RareDiseaseMetadataExtractor if not already available from router
        if 'basic_extractor' not in components and RareDiseaseMetadataExtractor:
            try:
                components['basic_extractor'] = RareDiseaseMetadataExtractor(
                    system_initializer=system_initializer_instance
                )
                logger.debug("Initialized RareDiseaseMetadataExtractor")
            except Exception as e:
                logger.warning(f"RareDiseaseMetadataExtractor initialization failed: {e}")
        
        # Only create DrugMetadataExtractor if not already available
        if 'drug_extractor' not in components and DrugMetadataExtractor and features.get('drug_detection', True):
            try:
                components['drug_extractor'] = DrugMetadataExtractor(
                    system_initializer=system_initializer_instance,
                    use_claude=claude_available
                )
                logger.debug(f"Initialized DrugMetadataExtractor (Claude: {claude_available})")
            except Exception as e:
                logger.warning(f"DrugMetadataExtractor initialization failed: {e}")
        
        # Only create DiseaseMetadataExtractor if not already available
        if 'disease_extractor' not in components and DiseaseMetadataExtractor and features.get('disease_detection', True):
            try:
                components['disease_extractor'] = DiseaseMetadataExtractor(
                    mode='balanced',
                    use_claude=claude_available,
                    verbose=False
                )
                logger.debug(f"Initialized DiseaseMetadataExtractor (Claude: {claude_available})")
            except Exception as e:
                logger.warning(f"DiseaseMetadataExtractor initialization failed: {e}")
        
        # Get extraction-specific config sections
        extraction_config = config.get('extraction')
        
        # Helper to safely convert dataclass to dict
        def _dataclass_to_dict(obj):
            """Convert a dataclass to dict, or return empty dict if None."""
            if obj is None:
                return {}
            try:
                from dataclasses import asdict
                return asdict(obj)
            except (TypeError, AttributeError):
                return {} if not isinstance(obj, dict) else obj
        
        # Initialize Citation Extractor with proper config
        if CitationExtractor and features.get('citation_extraction', True):
            try:
                citation_config = _dataclass_to_dict(
                    getattr(extraction_config, 'citation', None) if extraction_config else None
                )
                components['citation_extractor'] = CitationExtractor(config=citation_config)
                print_status("Citation extraction: ENABLED", "OK")
                logger.debug("Initialized CitationExtractor")
            except Exception as e:
                print_status(f"Citation extraction: DISABLED ({str(e)[:30]})", "WARN")
                logger.warning(f"CitationExtractor initialization failed: {e}")
        
        # Initialize Person Extractor with proper config
        if PersonExtractor and features.get('person_extraction', True):
            try:
                person_config = _dataclass_to_dict(
                    getattr(extraction_config, 'person', None) if extraction_config else None
                )
                components['person_extractor'] = PersonExtractor(config=person_config)
                print_status("Person extraction: ENABLED", "OK")
                logger.debug("Initialized PersonExtractor")
            except Exception as e:
                print_status(f"Person extraction: DISABLED ({str(e)[:30]})", "WARN")
                logger.warning(f"PersonExtractor initialization failed: {e}")
        
        # Initialize Reference Extractor with proper config
        if ReferenceExtractor and features.get('reference_extraction', True):
            try:
                reference_config = _dataclass_to_dict(
                    getattr(extraction_config, 'reference', None) if extraction_config else None
                )
                components['reference_extractor'] = ReferenceExtractor(config=reference_config)
                print_status("Reference extraction: ENABLED", "OK")
                logger.debug("Initialized ReferenceExtractor")
            except Exception as e:
                print_status(f"Reference extraction: DISABLED ({str(e)[:30]})", "WARN")
                logger.warning(f"ReferenceExtractor initialization failed: {e}")
        
        # Initialize remaining components
        if IntelligentDocumentRenamer and features.get('intelligent_rename', True):
            try:
                components['renamer'] = IntelligentDocumentRenamer()
                print_status("Intelligent renaming: ENABLED", "OK")
                logger.debug("Initialized IntelligentDocumentRenamer")
            except Exception as e:
                logger.warning(f"IntelligentDocumentRenamer initialization failed: {e}")
        
        if AbbreviationExtractor and features.get('abbreviation_extraction', True):
            try:
                components['abbreviation_extractor'] = AbbreviationExtractor(
                    system_initializer=system_initializer_instance,
                    use_claude=claude_available
                )
                print_status("Abbreviation extraction: ENABLED", "OK")
                logger.debug(f"Initialized AbbreviationExtractor (Claude: {claude_available})")
            except Exception as e:
                logger.warning(f"AbbreviationExtractor initialization failed: {e}")
        
        print_status("Extraction components ready", "OK")
        logger.debug(f"System initialization complete with {len(components)} components")
        
        if 'document_reader' not in components:
            logger.error("Critical: DocumentReader not in components")
            return {}
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to initialize extraction components: {e}", exc_info=True)
        print_status(f"Failed to initialize components: {e}", "FAIL")
        return {}

# ============================================================================
# MAIN FUNCTION 
# ============================================================================

def main():
    """Main processing function - FIXED console reporting bug in v8.10.1"""
    
    try:
        # ====================================================================
        # DISPLAY SYSTEM STATUS - Show loaded resources and initialization summary
        # ====================================================================
        console.print_loading_data()
        console.print_initialization_summary(modules_loaded)
        
        
        # ====================================================================
        # CHECKING FILES - Locate documents folder and setup output paths
        # ====================================================================
        console.print_section("CHECKING FILES")
        documents_folder = Path("documents_sota")
        
        if not documents_folder.exists():
            documents_folder = Path(__file__).parent / "documents_sota"
        
        if not documents_folder.exists():
            print_status(f"Documents folder not found: {documents_folder}", "FAIL")
            logger.error(f"Documents folder not found: {documents_folder}")
            return
        
        print_status(f"Documents folder: {documents_folder.absolute()}", "OK")
        logger.debug(f"Using documents folder: {documents_folder.absolute()}")
        
        # Create output folders
        extracted_texts_folder = documents_folder / "extracted_texts"
        extracted_texts_folder.mkdir(exist_ok=True)
        
        print_status(f"Text output: {extracted_texts_folder.absolute()}", "OK")
        print_status(f"Metadata output: {documents_folder.absolute()}", "OK")



        # ====================================================================
        # PREFIX MANAGER - Auto-incrementing file prefixes (01000_, 01001_, etc.)
        # ====================================================================
        
        prefix_manager = DocumentPrefixManager(
            counter_dir=documents_folder,
            start_number=config.defaults.prefix_start_number
        )
        console.print_section("PREFIX MANAGER")
        print_status(f"Auto-prefix starting at: {prefix_manager.get_next_prefix()}", "OK")
        logger.info(f"DocumentPrefixManager initialized with starting prefix: {prefix_manager.get_next_prefix()}")




        
        # Scan for documents
        pdf_files = list(documents_folder.glob("*.pdf"))
        
        if not pdf_files:
            print_status("No PDF files found to process", "WARN")
            logger.warning("No PDF files found to process")
            return
        
        console.print_files_to_process(pdf_files)
        
        # Initialize system components
        console.print_section("PROCESSING PIPELINE")
        print_status("Initializing extraction components...", "INFO")
        components = initialize_extraction_system()
        
        if not components:
            print_status("Failed to initialize extraction system", "FAIL")
            logger.error("Failed to initialize extraction system")
            return
        
        components['prefix_manager'] = prefix_manager
        
        print_status("Components ready", "OK")
        
        # ====================================================================
        # PROCESS DOCUMENTS
        # ====================================================================
        for file_num, pdf_file in enumerate(pdf_files, 1):
            try:
                # Start processing and update console
                console.print_file_processing_start(file_num, pdf_file)
                
                # Process document and CAPTURE results
                results = process_document_two_stage(
                    pdf_file, 
                    components, 
                    documents_folder,
                    console=console,
                    config=config
                )
                
                # Display results and update tracking counters
                console.print_extraction_results(results)
                console.print_file_complete(success=True)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}", exc_info=True)
                console.print_file_complete(success=False, error_msg=str(e))
        
        # Print final summary with all tracked results
        console.print_final_summary()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("Process interrupted by user", "WARN")
        logger.warning("Process interrupted by user")
    except Exception as e:
        print_status(f"Fatal error: {e}", "FAIL")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()