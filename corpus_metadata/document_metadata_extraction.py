#!/usr/bin/env python3
"""
Documents SOTA Extractor - COMPLETE METADATA EXTRACTION PIPELINE
================================================================
 
Location: corpus_metadata/document_metadata_extraction.py

VERSION 8.10.1 - FIXED CONSOLE REPORTING BUG
=============================================
Changes in v8.10.1:
- FIXED: Console summary now correctly shows entity counts
- FIXED: Results from process_document_two_stage are now captured
- FIXED: print_extraction_results and print_file_complete are now called
- FIXED: File processing progress is now displayed correctly

Changes in v8.10.0:
- Added CitationExtractor for bibliographic citations
- Added PersonExtractor for authors and investigators
- Added ReferenceExtractor for external identifiers (DOI, PMID, NCT, etc.)
- Updated pipeline to include these extractors
- Enhanced output to show citation/person/reference statistics
 
Pipeline Overview:
1. ABBREVIATION EXTRACTION (First)
   ├── Extract all abbreviations with expansions
   ├── Classify context (biological/disease/drug/clinical)
   └── Output: {abbr, expansion, confidence, type, occurrences}
   
2. CITATION & REFERENCE EXTRACTION (New Stage)
   ├── Citation Extractor:
   │   ├── Detect citation styles (Vancouver, APA, Harvard, etc.)
   │   ├── Extract structured citations from references
   │   └── Link inline citations to references
   ├── Person Extractor:
   │   ├── Extract authors, PIs, investigators
   │   ├── Handle Unicode names (García-López, O'Connor)
   │   ├── Extract ORCIDs and affiliations
   │   └── Build co-author network
   └── Reference Extractor:
       ├── Extract 60+ identifier types
       ├── Reconstruct URLs
       └── Classify reference roles

3. ENRICHMENT PHASE
   ├── Drug Extractor receives:
   │   └── Abbreviations where type IN (drug, clinical, biological)
   │       + Their full expansions
   └── Disease Extractor receives:
       └── Abbreviations where type IN (disease, clinical, biological)
           + Their full expansions

4. ENTITY EXTRACTION (Enhanced)
   ├── Drug Detector runs with:
   │   ├── Original text
   │   ├── Abbreviation candidates
   │   └── Expanded forms
   └── Disease Detector runs with:
       ├── Original text
       ├── Abbreviation candidates
       └── Expanded forms

5. VALIDATION & DEDUPLICATION
   ├── Cross-validate findings
   ├── Resolve conflicts (MPA case)
   └── Merge duplicates

6. INTELLIGENT RENAMING + PREFIX APPLICATION
   ├── Generate intelligent filename based on content
   ├── Apply auto-incrementing prefix (01000_, 01001_, etc.)
   └── Rename file with new identifier
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="spacy.cli.info")
warnings.filterwarnings("ignore", category=FutureWarning, message="Possible nested set")

# ============================================================================
# ENHANCED CONSOLE OUTPUT
# ============================================================================

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
    BRIGHT_WHITE = '\033[97m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_CYAN = '\033[96m'
    
    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')

if not sys.stdout.isatty():
    Colors.disable()

# ============================================================================
# EARLY SYSTEM INITIALIZATION
# ============================================================================

os.environ['CORPUS_QUIET_INIT'] = '1'
from corpus_metadata.document_utils.metadata_logging_config import CorpusConfig, get_logger
script_dir = Path(__file__).parent.parent
config_dir = script_dir / "corpus_config"

config = CorpusConfig(config_dir=config_dir)
logger = get_logger('sota_extractor')
logger.debug("Centralized logging system initialized")

if 'CORPUS_QUIET_INIT' in os.environ:
    del os.environ['CORPUS_QUIET_INIT']

# ============================================================================
# MODULE LOADING WITH PROGRESS
# ============================================================================

print(f"\n{Colors.BRIGHT_CYAN}INITIALIZING SYSTEM COMPONENTS{Colors.ENDC}")
print(f"{Colors.BRIGHT_BLACK}{'─'*60}{Colors.ENDC}")

modules_loaded = {
    'system_initializer': False,
    'document_reader': False,
    'document_classifier': False,
    'document_router': False,
    'basic_extractor': False,
    'drug_extractor': False,
    'disease_extractor': False,
    'intelligent_renamer': False,
    'abbreviation_extractor': False,
    'entity_extraction': False,
    'prefix_manager': False,
    'citation_extractor': False,  
    'person_extractor': False,    
    'reference_extractor': False   
}

# Step 1: Load core modules
print(f"\n{Colors.YELLOW}Loading core modules:{Colors.ENDC}")
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    from corpus_metadata.document_utils.prefix_manager import (
        DocumentPrefixManager, 
        apply_prefix_to_renamed_file
    )
    
    system_initializer_instance = MetadataSystemInitializer.get_instance()
    modules_loaded['system_initializer'] = True
    modules_loaded['prefix_manager'] = True
    print(f"  {Colors.GREEN}✓{Colors.ENDC} [1/14] MetadataSystemInitializer")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} [2/14] DocumentPrefixManager")
    logger.debug("Core modules initialized successfully")
    
except Exception as e:
    print(f"  {Colors.RED}✗{Colors.ENDC} Core modules failed: {str(e)[:50]}")
    logger.error(f"Core modules failed: {e}")
    sys.exit(1)

# Define all modules to load
module_names = [
    ('document_reader', 'corpus_metadata.document_metadata_reader', 'DocumentReader'),
    ('document_classifier', 'corpus_metadata.document_utils.metadata_classifier', 'DocumentClassifier'),
    ('document_router', 'corpus_metadata.document_utils.metadata_document_type_router', 'DocumentTypeRouter'),
    ('basic_extractor', 'corpus_metadata.document_metadata_extraction_basic', 'RareDiseaseMetadataExtractor'),
    ('drug_extractor', 'corpus_metadata.document_metadata_extraction_drug', 'DrugMetadataExtractor'),
    ('disease_extractor', 'corpus_metadata.document_metadata_extraction_disease', 'DiseaseMetadataExtractor'),
    ('intelligent_renamer', 'corpus_metadata.document_intelligent_renamer', 'IntelligentDocumentRenamer'),
    ('abbreviation_extractor', 'corpus_metadata.document_metadata_extraction_abbreviation', 'AbbreviationExtractor'),
    ('entity_extraction', 'corpus_metadata.document_utils.entity_extraction', 'process_document_two_stage'),
    # NEW: Citation, Person, and Reference extractors
    ('citation_extractor', 'corpus_metadata.document_metadata_extraction_citation', 'CitationExtractor'),
    ('person_extractor', 'corpus_metadata.document_metadata_extraction_persons', 'PersonExtractor'),
    ('reference_extractor', 'corpus_metadata.document_metadata_extraction_references', 'ReferenceExtractor'),
]

loaded_classes = {
    'MetadataSystemInitializer': MetadataSystemInitializer,
    'DocumentPrefixManager': DocumentPrefixManager,
    'apply_prefix_to_renamed_file': apply_prefix_to_renamed_file
}

# Load modules
for i, (key, module_path, class_name) in enumerate(module_names, 3):
    try:
        module = __import__(module_path, fromlist=[class_name])
        loaded_classes[class_name] = getattr(module, class_name)
        modules_loaded[key] = True
        print(f"  {Colors.GREEN}✓{Colors.ENDC} [{i}/14] {key}")
        logger.debug(f"Loaded module {key} successfully")
    except ImportError as e:
        print(f"  {Colors.YELLOW}⚠{Colors.ENDC} [{i}/14] {key} - {str(e)[:50]}")
        logger.error(f"Failed to load module {key}: {e}")
        if key in ['document_reader', 'entity_extraction']:
            print(f"\n{Colors.RED}CRITICAL: {key} is required. Exiting.{Colors.ENDC}")
            logger.error(f"Critical module {key} failed to load. Exiting.")
            sys.exit(1)

# Extract loaded classes
DocumentReader = loaded_classes.get('DocumentReader')
DocumentClassifier = loaded_classes.get('DocumentClassifier')
DocumentTypeRouter = loaded_classes.get('DocumentTypeRouter')
RareDiseaseMetadataExtractor = loaded_classes.get('RareDiseaseMetadataExtractor')
DrugMetadataExtractor = loaded_classes.get('DrugMetadataExtractor')
DiseaseMetadataExtractor = loaded_classes.get('DiseaseMetadataExtractor')
IntelligentDocumentRenamer = loaded_classes.get('IntelligentDocumentRenamer')
AbbreviationExtractor = loaded_classes.get('AbbreviationExtractor')
process_document_two_stage = loaded_classes.get('process_document_two_stage')
DocumentPrefixManager = loaded_classes.get('DocumentPrefixManager')
apply_prefix_to_renamed_file = loaded_classes.get('apply_prefix_to_renamed_file')
CitationExtractor = loaded_classes.get('CitationExtractor')  # NEW
PersonExtractor = loaded_classes.get('PersonExtractor')      # NEW
ReferenceExtractor = loaded_classes.get('ReferenceExtractor')  # NEW

# Optional: Enable debugging
try:
    from corpus_metadata.abbreviation_debugger import integrate_debugging
    import corpus_metadata.document_utils.entity_extraction as entity_extraction
    abbreviation_debugger = integrate_debugging(entity_extraction)
    print(f"{Colors.GREEN}✓ Abbreviation debugging enabled{Colors.ENDC}")
    logger.info("Abbreviation debugging integrated successfully")
except ImportError as e:
    print(f"{Colors.YELLOW}⚠ Abbreviation debugging not available: {e}{Colors.ENDC}")
    logger.warning(f"Could not enable abbreviation debugging: {e}")

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
        self.total_citations = 0      # NEW
        self.total_persons = 0        # NEW
        self.total_references = 0     # NEW
        self.texts_saved = 0
        self.files_renamed = 0
        self.files_prefixed = 0
        
    def print_header(self):
        """Print the application header"""
        print(f"\n{Colors.BRIGHT_CYAN}{'═'*80}")
        print(f"  {Colors.BOLD}{Colors.BRIGHT_WHITE}CORPUS DOCUMENT PROCESSOR - SOTA EXTRACTOR v8.10.1{Colors.ENDC}")
        print(f"  {Colors.BRIGHT_BLACK}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
        print(f"{Colors.BRIGHT_CYAN}{'═'*80}{Colors.ENDC}\n")
    
    def print_section(self, title: str, char: str = "─"):
        """Print a section separator"""
        print(f"\n{Colors.BRIGHT_CYAN}{title}{Colors.ENDC}")
        print(f"{Colors.BRIGHT_BLACK}{char*60}{Colors.ENDC}")
    
    def print_loading_data(self):
        """Print data loading information"""
        self.print_section("DATA SOURCES")
        
        lexicons = {
            'Drug Lexicon': config.get('resources.drug_lexicon'),
            'Disease Lexicon': config.get('resources.disease_lexicon'),
            'Medical Terms': config.get('resources.medical_terms_lexicon'),
            'Abbreviations': config.get('resources.abbreviation_general')
        }
        
        for i, (name, path) in enumerate(lexicons.items(), 1):
            if path:
                path_obj = Path(path) if path else None
                if path_obj and path_obj.exists():
                    size = path_obj.stat().st_size / (1024 * 1024)
                    print(f"  {Colors.GREEN}✓{Colors.ENDC} {name:<25} {Colors.BRIGHT_BLACK}({size:.2f} MB){Colors.ENDC}")
                    logger.debug(f"Loaded {name} from {path} ({size:.2f} MB)")
                else:
                    print(f"  {Colors.YELLOW}⚠{Colors.ENDC} {name:<25} {Colors.BRIGHT_BLACK}(not found){Colors.ENDC}")
                    logger.warning(f"{name} not found at {path}")
            else:
                print(f"  {Colors.BRIGHT_BLACK}─{Colors.ENDC} {name:<25} {Colors.BRIGHT_BLACK}(not configured){Colors.ENDC}")
                logger.debug(f"{name} not configured")
    
    def print_initialization_summary(self, modules_loaded: Dict[str, bool]):
        """Print initialization summary"""
        loaded = sum(1 for v in modules_loaded.values() if v)
        total = len(modules_loaded)
        features = config.get_feature_flags()
        enabled = sum(1 for v in features.values() if v)
        
        print(f"\n{Colors.GREEN}✅ System Ready{Colors.ENDC}")
        print(f"{Colors.BRIGHT_BLACK}{'─'*40}{Colors.ENDC}")
        print(f"  {Colors.BRIGHT_WHITE}Modules:{Colors.ENDC} {loaded}/{total} loaded")
        print(f"  {Colors.BRIGHT_WHITE}Features:{Colors.ENDC} {enabled}/{len(features)} enabled")
        print(f"  {Colors.BRIGHT_WHITE}Pipeline:{Colors.ENDC} {len(config.get_all_stages())} stages configured")
        
        logger.debug(f"System initialized: {loaded}/{total} modules, {enabled}/{len(features)} features")
    
    def print_files_to_process(self, files: List[Path]):
        """Print files to be processed"""
        self.total_files = len(files)
        self.print_section("FILES TO PROCESS")
        
        print(f"  {Colors.BRIGHT_WHITE}Total files:{Colors.ENDC} {Colors.BOLD}{len(files)}{Colors.ENDC}")
        print(f"  {Colors.BRIGHT_WHITE}Order:{Colors.ENDC} Alphabetical\n")
        
        for i, filepath in enumerate(files[:5], 1):
            print(f"  {Colors.BRIGHT_BLACK}{i}.{Colors.ENDC} {filepath.name}")
        
        if len(files) > 5:
            print(f"  {Colors.BRIGHT_BLACK}... and {len(files) - 5} more files{Colors.ENDC}")
    
    def print_file_processing_start(self, file_num: int, file_path: Path):
        """Print file processing start"""
        self.current_file = file_path.name
        self.files_processed = file_num
        print(f"\n{Colors.BRIGHT_CYAN}[{file_num}/{self.total_files}]{Colors.ENDC} {Colors.BRIGHT_WHITE}Processing:{Colors.ENDC} {file_path.name}")
        print(f"  {Colors.BRIGHT_BLACK}Size: {file_path.stat().st_size / 1024:.1f} KB{Colors.ENDC}")
    
    def print_stage_progress(self, stage: str, sequence: int, task: str, 
                            success: bool, detail: str = "", elapsed: float = 0):
        """Print stage progress"""
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if success else f"{Colors.RED}✗{Colors.ENDC}"
        time_str = f"({elapsed:.1f}s)" if elapsed > 0 else ""
        print(f"    {status} {task:<30} {detail:<30} {time_str}")
    
    def print_extraction_results(self, results: Dict):
        """Print extraction results summary - UPDATED WITH NEW EXTRACTORS"""
        print(f"\n  {Colors.CYAN}Extraction Results:{Colors.ENDC}")
        
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
            citations = entity_results.get('citations', [])        # NEW
            persons = entity_results.get('persons', [])            # NEW
            references = entity_results.get('references', [])      # NEW
        else:
            drugs = results.get('drugs', [])
            diseases = results.get('diseases', [])
            abbreviations = results.get('abbreviations', [])
            citations = results.get('citations', [])               # NEW
            persons = results.get('persons', [])                   # NEW
            references = results.get('references', [])             # NEW
        
        self.total_drugs += len(drugs)
        self.total_diseases += len(diseases)
        self.total_abbreviations += len(abbreviations)
        self.total_citations += len(citations)                     # NEW
        self.total_persons += len(persons)                         # NEW
        self.total_references += len(references)                   # NEW
        
        # Print existing results
        print(f"    • Drugs found: {Colors.BOLD}{len(drugs)}{Colors.ENDC}")
        if drugs and len(drugs) <= 3:
            for drug in drugs[:3]:
                name = drug.get('name', 'Unknown')
                conf = drug.get('confidence', 0)
                print(f"      - {name} ({conf:.0%})")
        
        print(f"    • Diseases found: {Colors.BOLD}{len(diseases)}{Colors.ENDC}")
        if diseases and len(diseases) <= 3:
            for disease in diseases[:3]:
                name = disease.get('name', 'Unknown')
                conf = disease.get('confidence', 0)
                print(f"      - {name} ({conf:.0%})")
        
        print(f"    • Abbreviations found: {Colors.BOLD}{len(abbreviations)}{Colors.ENDC}")
        if abbreviations and len(abbreviations) <= 3:
            for abbrev in abbreviations[:3]:
                abbr = abbrev.get('abbreviation', 'Unknown')
                exp = abbrev.get('expansion', 'Unknown')
                print(f"      - {abbr} → {exp}")
        
        # NEW: Print citation results
        print(f"    • Citations found: {Colors.BOLD}{len(citations)}{Colors.ENDC}")
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
                    print(f"      - {first_author} et al. ({year}) - {journal}")
        
        # NEW: Print person results
        print(f"    • Persons found: {Colors.BOLD}{len(persons)}{Colors.ENDC}")
        if persons and len(persons) <= 3:
            for person in persons[:3]:
                if isinstance(person, dict):
                    name_data = person.get('name', {})
                    if isinstance(name_data, dict):
                        full_name = name_data.get('full_name', 'Unknown')
                    else:
                        full_name = str(name_data)
                    role = person.get('role', 'unknown')
                    print(f"      - {full_name} ({role})")
        
        # NEW: Print reference results
        print(f"    • References found: {Colors.BOLD}{len(references)}{Colors.ENDC}")
        if references and len(references) <= 3:
            for reference in references[:3]:
                if isinstance(reference, dict):
                    ref_type = reference.get('reference_type', 'unknown')
                    normalized = reference.get('normalized_value', 'Unknown')[:30]
                    print(f"      - {ref_type}: {normalized}")
    
    def print_file_complete(self, success: bool = True, error_msg: str = ""):
        """Print file completion status"""
        if success:
            self.successful += 1
            print(f"  {Colors.GREEN}✓ Completed successfully{Colors.ENDC}")
        else:
            self.failed += 1
            print(f"  {Colors.RED}✗ Failed: {error_msg[:50]}{Colors.ENDC}")
    
    def print_rename_status(self, old_name: str, new_name: str, was_prefixed: bool):
        """Print file rename status"""
        if old_name != new_name:
            self.files_renamed += 1
            if was_prefixed:
                self.files_prefixed += 1
                print(f"  {Colors.CYAN}↻ Renamed + Prefixed:{Colors.ENDC} {new_name}")
            else:
                print(f"  {Colors.CYAN}↻ Renamed:{Colors.ENDC} {new_name}")
            logger.info(f"File renamed: {old_name} → {new_name}")
    
    def print_final_summary(self):
        """Print final processing summary - UPDATED"""
        elapsed = time.time() - self.start_time
        
        print(f"\n{Colors.BRIGHT_CYAN}{'═'*60}")
        print(f"  PROCESSING COMPLETE")
        print(f"{'═'*60}{Colors.ENDC}")
        
        print(f"\n  {Colors.BRIGHT_WHITE}Summary:{Colors.ENDC}")
        print(f"    {Colors.BRIGHT_WHITE}Files processed:{Colors.ENDC} {self.files_processed}/{self.total_files}")
        print(f"    {Colors.BRIGHT_WHITE}Successful:{Colors.ENDC} {Colors.GREEN}{self.successful}{Colors.ENDC}")
        print(f"    {Colors.BRIGHT_WHITE}Failed:{Colors.ENDC} {Colors.RED}{self.failed}{Colors.ENDC}")
        print(f"    {Colors.BRIGHT_WHITE}Text files saved:{Colors.ENDC} {self.texts_saved}")
        print(f"    {Colors.BRIGHT_WHITE}Files renamed:{Colors.ENDC} {self.files_renamed}")
        print(f"    {Colors.BRIGHT_WHITE}Files prefixed:{Colors.ENDC} {self.files_prefixed}")
        
        print(f"\n  {Colors.BRIGHT_WHITE}Entities Extracted:{Colors.ENDC}")
        print(f"    {Colors.BRIGHT_WHITE}Drugs:{Colors.ENDC} {self.total_drugs}")
        print(f"    {Colors.BRIGHT_WHITE}Diseases:{Colors.ENDC} {self.total_diseases}")
        print(f"    {Colors.BRIGHT_WHITE}Abbreviations:{Colors.ENDC} {self.total_abbreviations}")
        print(f"    {Colors.BRIGHT_WHITE}Citations:{Colors.ENDC} {self.total_citations}")        # NEW
        print(f"    {Colors.BRIGHT_WHITE}Persons:{Colors.ENDC} {self.total_persons}")            # NEW
        print(f"    {Colors.BRIGHT_WHITE}References:{Colors.ENDC} {self.total_references}")      # NEW
        
        print(f"\n    {Colors.BRIGHT_WHITE}Time elapsed:{Colors.ENDC} {elapsed:.1f} seconds")
        print(f"\n{Colors.GREEN}✅ All processing complete!{Colors.ENDC}\n")

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
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Claude validation: {Colors.BOLD}ENABLED{Colors.ENDC}")
                logger.info("Claude validation enabled")
            else:
                print(f"  {Colors.YELLOW}⚠ {Colors.ENDC} Claude validation: {Colors.BOLD}DISABLED{Colors.ENDC} (no API key)")
                logger.warning("ai_validation is true but CLAUDE_API_KEY not found")
                claude_available = False
        else:
            print(f"  {Colors.YELLOW}⚠ {Colors.ENDC} Claude validation: {Colors.BOLD}DISABLED{Colors.ENDC} (config: ai_validation=false)")
            logger.info("Claude validation disabled by config")
            claude_available = False
        
        # Initialize DocumentReader (required)
        if DocumentReader:
            try:
                components['document_reader'] = DocumentReader()
                logger.debug("Initialized DocumentReader")
            except Exception as e:
                print(f"  {Colors.RED}✗{Colors.ENDC} Failed to initialize DocumentReader: {e}")
                logger.error(f"DocumentReader initialization failed: {e}")
                return {}
        else:
            print(f"  {Colors.RED}✗{Colors.ENDC} DocumentReader class not available")
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
            except Exception as e:
                logger.warning(f"DocumentTypeRouter initialization failed: {e}")
        
        if RareDiseaseMetadataExtractor:
            try:
                components['basic_extractor'] = RareDiseaseMetadataExtractor(
                    system_initializer=system_initializer_instance
                )
                logger.debug("Initialized RareDiseaseMetadataExtractor")
            except Exception as e:
                logger.warning(f"RareDiseaseMetadataExtractor initialization failed: {e}")
        
        if DrugMetadataExtractor and features.get('drug_detection', True):
            try:
                components['drug_extractor'] = DrugMetadataExtractor(
                    system_initializer=system_initializer_instance,
                    use_claude=claude_available
                )
                logger.debug(f"Initialized DrugMetadataExtractor (Claude: {claude_available})")
            except Exception as e:
                logger.warning(f"DrugMetadataExtractor initialization failed: {e}")
        
        if DiseaseMetadataExtractor and features.get('disease_detection', True):
            try:
                components['disease_extractor'] = DiseaseMetadataExtractor(
                    mode='balanced',
                    use_claude=claude_available,
                    verbose=False
                )
                logger.debug(f"Initialized DiseaseMetadataExtractor (Claude: {claude_available})")
            except Exception as e:
                logger.warning(f"DiseaseMetadataExtractor initialization failed: {e}")
        
        # NEW: Initialize Citation Extractor
        if CitationExtractor and features.get('citation_extraction', True):
            try:
                components['citation_extractor'] = CitationExtractor()
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Citation extraction: {Colors.BOLD}ENABLED{Colors.ENDC}")
                logger.debug("Initialized CitationExtractor")
            except Exception as e:
                print(f"  {Colors.YELLOW}⚠ {Colors.ENDC} Citation extraction: {Colors.BOLD}DISABLED{Colors.ENDC} ({str(e)[:30]})")
                logger.warning(f"CitationExtractor initialization failed: {e}")
        
        # NEW: Initialize Person Extractor
        if PersonExtractor and features.get('person_extraction', True):
            try:
                components['person_extractor'] = PersonExtractor()
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Person extraction: {Colors.BOLD}ENABLED{Colors.ENDC}")
                logger.debug("Initialized PersonExtractor")
            except Exception as e:
                print(f"  {Colors.YELLOW}⚠ {Colors.ENDC} Person extraction: {Colors.BOLD}DISABLED{Colors.ENDC} ({str(e)[:30]})")
                logger.warning(f"PersonExtractor initialization failed: {e}")
        
        # NEW: Initialize Reference Extractor
        if ReferenceExtractor and features.get('reference_extraction', True):
            try:
                components['reference_extractor'] = ReferenceExtractor()
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Reference extraction: {Colors.BOLD}ENABLED{Colors.ENDC}")
                logger.debug("Initialized ReferenceExtractor")
            except Exception as e:
                print(f"  {Colors.YELLOW}⚠ {Colors.ENDC} Reference extraction: {Colors.BOLD}DISABLED{Colors.ENDC} ({str(e)[:30]})")
                logger.warning(f"ReferenceExtractor initialization failed: {e}")
        
        # Initialize remaining components
        if IntelligentDocumentRenamer and features.get('intelligent_rename', True):
            try:
                components['renamer'] = IntelligentDocumentRenamer()
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Intelligent renaming: {Colors.BOLD}ENABLED{Colors.ENDC}")
                logger.debug("Initialized IntelligentDocumentRenamer")
            except Exception as e:
                logger.warning(f"IntelligentDocumentRenamer initialization failed: {e}")
        
        if AbbreviationExtractor and features.get('abbreviations', True):
            try:
                components['abbreviation_extractor'] = AbbreviationExtractor(
                    system_initializer=system_initializer_instance,
                    use_claude=claude_available
                )
                print(f"  {Colors.GREEN}✓{Colors.ENDC} Abbreviation extraction: {Colors.BOLD}ENABLED{Colors.ENDC}")
                logger.debug(f"Initialized AbbreviationExtractor (Claude: {claude_available})")
            except Exception as e:
                logger.warning(f"AbbreviationExtractor initialization failed: {e}")
        
        print(f"\n{Colors.GREEN}✅ Extraction components ready{Colors.ENDC}")
        logger.debug(f"System initialization complete with {len(components)} components")
        
        if 'document_reader' not in components:
            logger.error("Critical: DocumentReader not in components")
            return {}
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to initialize extraction components: {e}", exc_info=True)
        print(f"\n{Colors.RED}✗ Failed to initialize components: {e}{Colors.ENDC}")
        return {}

# ============================================================================
# MAIN FUNCTION - FIXED IN v8.10.1
# ============================================================================

def main():
    """Main processing function - FIXED console reporting bug in v8.10.1"""
    
    try:
        console.print_header()
        console.print_loading_data()
        console.print_initialization_summary(modules_loaded)
        
        # Check documents folder
        console.print_section("CHECKING FILES")
        documents_folder = Path("documents_sota")
        
        if not documents_folder.exists():
            documents_folder = Path(__file__).parent / "documents_sota"
        
        if not documents_folder.exists():
            print(f"  {Colors.RED}✗{Colors.ENDC} Documents folder not found: {documents_folder}")
            logger.error(f"Documents folder not found: {documents_folder}")
            return
        
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Documents folder: {Colors.BRIGHT_BLACK}{documents_folder.absolute()}{Colors.ENDC}")
        logger.debug(f"Using documents folder: {documents_folder.absolute()}")
        
        # Create output folders
        extracted_texts_folder = documents_folder / "extracted_texts"
        extracted_texts_folder.mkdir(exist_ok=True)
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Text output: {Colors.BRIGHT_BLACK}{extracted_texts_folder.absolute()}{Colors.ENDC}")
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Metadata output: {Colors.BRIGHT_BLACK}{documents_folder.absolute()}{Colors.ENDC}")

        # Initialize prefix manager
        prefix_manager = DocumentPrefixManager(
            counter_dir=documents_folder,
            start_number=1000
        )
        console.print_section("PREFIX MANAGER")
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Auto-prefix starting at: {prefix_manager.get_next_prefix()}")
        logger.info(f"DocumentPrefixManager initialized with starting prefix: {prefix_manager.get_next_prefix()}")
        
        # Scan for documents
        pdf_files = list(documents_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"  {Colors.YELLOW}⚠ No PDF files found to process{Colors.ENDC}")
            logger.warning("No PDF files found to process")
            return
        
        console.print_files_to_process(pdf_files)
        
        # Initialize system components
        console.print_section("PROCESSING PIPELINE")
        print("  Initializing extraction components...")
        components = initialize_extraction_system()
        
        if not components:
            print(f"  {Colors.RED}✗{Colors.ENDC} Failed to initialize extraction system")
            logger.error("Failed to initialize extraction system")
            return
        
        components['prefix_manager'] = prefix_manager
        
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Components ready\n")
        
        # ====================================================================
        # PROCESS DOCUMENTS - FIXED IN v8.10.1
        # ====================================================================
        # Fix: Capture results and call console reporting methods
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
        print(f"\n\n{Colors.YELLOW}⚠ Process interrupted by user{Colors.ENDC}")
        logger.warning("Process interrupted by user")
    except Exception as e:
        print(f"\n\n{Colors.RED}✗ Fatal error: {e}{Colors.ENDC}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()