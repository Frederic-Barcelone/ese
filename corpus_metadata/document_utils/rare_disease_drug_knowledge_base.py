#!/usr/bin/env python3
"""
Drug Knowledge Base for Rare Disease Document Processing
=========================================================
Location: corpus_metadata/document_utils/rare_disease_drug_knowledge_base.py
Version: 2.2
Last Updated: 2025-01-16

Purpose:
--------
Centralized drug knowledge management system that integrates multiple drug data sources
including FDA approved drugs, investigational drugs, Alexion drugs, and drug patterns.
Works with MetadataSystemInitializer to access all drug-related resources.

CHANGES IN VERSION 2.2:
-----------------------
- Updated source tracking to use specific lexicon names
- Changed source values from generic to specific:
  * 'FDA' → 'fda_approved'
  * 'ClinicalTrials.gov' → 'investigational'
  * 'Alexion' → 'alexion'
  * 'RxNorm' → 'drug_lexicon'
- This enables proper source identification in drug detection logs

Features:
---------
- Unified drug knowledge from multiple sources
- Drug classification (approved/investigational/alexion)
- Pattern-based drug detection support
- Drug name normalization and variant handling
- Integration with system initializer for resource access
- Proper source tracking for each lexicon
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
from dataclasses import dataclass, field
import re

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DrugInfo:
    """Comprehensive drug information container"""
    name: str
    drug_type: str  # 'approved', 'investigational', 'alexion', 'pattern'
    source: str  # Now uses specific lexicon names: 'fda_approved', 'investigational', 'alexion', 'drug_lexicon'
    synonyms: List[str] = field(default_factory=list)
    brand_names: List[str] = field(default_factory=list)
    generic_names: List[str] = field(default_factory=list)
    rxcui: Optional[str] = None
    nct_ids: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    status: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrugPattern:
    """Drug detection pattern"""
    pattern: str
    pattern_type: str  # 'suffix', 'prefix', 'regex'
    confidence: float
    description: str
    compiled_pattern: Optional[re.Pattern] = None


# ============================================================================
# Drug Knowledge Base Class
# ============================================================================

class DrugKnowledgeBase:
    """
    Centralized drug knowledge management system.
    Integrates all drug-related resources from config.yaml.
    """
    
    def __init__(self, system_initializer=None, config_path: str = "corpus_config/config.yaml"):
        """
        Initialize drug knowledge base.
        
        Args:
            system_initializer: MetadataSystemInitializer instance
            config_path: Path to configuration file
        """
        self.system_initializer = system_initializer
        self.config_path = Path(config_path)
        
        # Drug storage structures
        self.drugs = {}  # drug_name -> DrugInfo
        self.drug_index = {
            'by_type': defaultdict(set),
            'by_source': defaultdict(set),
            'by_rxcui': {},
            'by_nct': defaultdict(set),
            'by_condition': defaultdict(set),
            'by_name': {},
            'normalized': {},  # normalized_name -> canonical_name
        }
        
        # Pattern storage
        self.drug_patterns = []
        self.compiled_patterns = {
            'suffix': [],
            'prefix': [],
            'regex': []
        }
        
        # Statistics
        self.stats = {
            'fda_approved': 0,
            'investigational': 0,
            'alexion': 0,
            'patterns': 0,
            'total_drugs': 0,
            'total_synonyms': 0
        }
        
        # Initialize knowledge base
        self._initialize()
    
    def _initialize(self):
        """Initialize drug knowledge from all configured sources."""
        logger.info("Initializing Drug Knowledge Base...")
        
        if self.system_initializer:
            self._load_from_system_initializer()
        else:
            self._load_from_config()
        
        self._build_indices()
        self._compile_patterns()
        
        logger.info(f"Drug Knowledge Base initialized with {self.stats['total_drugs']} drugs")
        self._log_statistics()
    
    def _load_from_system_initializer(self):
        """Load drug data from system initializer resources."""
        
        # Load FDA approved drugs
        if fda_drugs := self.system_initializer.get_resource('drug_fda_approved'):
            self._process_fda_drugs(fda_drugs)
        
        # Load investigational drugs
        if inv_drugs := self.system_initializer.get_resource('drug_investigational'):
            self._process_investigational_drugs(inv_drugs)
        
        # Load Alexion drugs
        if alexion_drugs := self.system_initializer.get_resource('drug_alexion'):
            self._process_alexion_drugs(alexion_drugs)
        
        # Load drug patterns
        if drug_patterns := self.system_initializer.get_resource('drug_patterns'):
            self._process_drug_patterns(drug_patterns)
        
        # Load drug lexicon
        if drug_lexicon := self.system_initializer.drug_lexicon:
            self._process_drug_lexicon(drug_lexicon)
    
    def _load_from_config(self):
        """Load drug data directly from config file paths."""
        try:
            with open(self.config_path, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
            
            resources = config.get('resources', {})
            
            # Load each resource type
            for resource_name, resource_path in resources.items():
                if not resource_path or not Path(resource_path).exists():
                    continue
                
                if resource_name == 'drug_fda_approved':
                    self._load_and_process_json(resource_path, self._process_fda_drugs)
                elif resource_name == 'drug_investigational':
                    self._load_and_process_json(resource_path, self._process_investigational_drugs)
                elif resource_name == 'drug_alexion':
                    self._load_and_process_json(resource_path, self._process_alexion_drugs)
                elif resource_name == 'drug_patterns':
                    self._load_and_process_json(resource_path, self._process_drug_patterns)
                elif resource_name == 'drug_lexicon':
                    self._load_and_process_json(resource_path, self._process_drug_lexicon)
        
        except Exception as e:
            logger.error(f"Failed to load from config: {e}")
    
    def _load_and_process_json(self, file_path: str, processor_func):
        """Load JSON file and process with given function."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processor_func(data)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    def _process_fda_drugs(self, data: Union[List, Dict]):
        """
        Process FDA approved drugs data.
        Expected format: List of dictionaries with keys: key, drug_class, source_classes, meta
        """
        try:
            if not isinstance(data, list):
                logger.warning(f"Expected list format for FDA drugs, got {type(data)}")
                return
            
            logger.info(f"Processing {len(data)} FDA drug entries")
            
            # Group drugs by key (active ingredient) to consolidate duplicates
            drugs_by_key = defaultdict(lambda: {
                'drug_class': None,
                'brand_names': set(),
                'dosage_forms': set(),
                'routes': set(),
                'marketing_statuses': set(),
                'application_numbers': set(),
                'entries': []
            })
            
            # First pass: group all entries by drug key
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                
                drug_key = entry.get('key', '').strip()
                if not drug_key:
                    continue
                
                # Handle combination drugs (separated by |)
                if '|' in drug_key:
                    # For combination drugs, we'll create separate entries for each component
                    components = [comp.strip() for comp in drug_key.split('|')]
                    for component in components:
                        drug_group = drugs_by_key[component]
                        drug_group['entries'].append(entry)
                        
                        # Store drug class if available
                        if drug_class := entry.get('drug_class'):
                            drug_group['drug_class'] = drug_class
                        
                        # Extract metadata
                        if meta := entry.get('meta', {}):
                            if brand_name := meta.get('brand_name'):
                                drug_group['brand_names'].add(brand_name)
                            if dosage_form := meta.get('dosage_form'):
                                drug_group['dosage_forms'].add(dosage_form)
                            if route := meta.get('route'):
                                drug_group['routes'].add(route)
                            if marketing_status := meta.get('marketing_status'):
                                drug_group['marketing_statuses'].add(marketing_status)
                            if app_number := meta.get('application_number'):
                                drug_group['application_numbers'].add(app_number)
                else:
                    # Single drug entry
                    drug_group = drugs_by_key[drug_key]
                    drug_group['entries'].append(entry)
                    
                    # Store drug class if available
                    if drug_class := entry.get('drug_class'):
                        drug_group['drug_class'] = drug_class
                    
                    # Extract metadata
                    if meta := entry.get('meta', {}):
                        if brand_name := meta.get('brand_name'):
                            drug_group['brand_names'].add(brand_name)
                        if dosage_form := meta.get('dosage_form'):
                            drug_group['dosage_forms'].add(dosage_form)
                        if route := meta.get('route'):
                            drug_group['routes'].add(route)
                        if marketing_status := meta.get('marketing_status'):
                            drug_group['marketing_statuses'].add(marketing_status)
                        if app_number := meta.get('application_number'):
                            drug_group['application_numbers'].add(app_number)
            
            # Second pass: create DrugInfo objects from grouped data
            for drug_key, drug_group in drugs_by_key.items():
                # Skip if no entries
                if not drug_group['entries']:
                    continue
                
                # Determine if drug is currently marketed
                marketing_statuses = drug_group['marketing_statuses']
                is_active = any(status in ['Prescription', 'Over-the-counter'] 
                            for status in marketing_statuses)
                
                # Create DrugInfo object with specific source
                drug_info = DrugInfo(
                    name=drug_key,
                    drug_type='approved',
                    source='fda_approved',  # ← CHANGED: Using specific lexicon name
                    brand_names=list(drug_group['brand_names']),
                    status='Active' if is_active else 'Discontinued',
                    confidence=1.0,  # High confidence for FDA data
                    metadata={
                        'drug_class': drug_group['drug_class'],
                        'dosage_forms': list(drug_group['dosage_forms']),
                        'routes': list(drug_group['routes']),
                        'marketing_statuses': list(drug_group['marketing_statuses']),
                        'application_numbers': list(drug_group['application_numbers']),
                        'entry_count': len(drug_group['entries'])  # Number of different formulations
                    }
                )
                
                # Store with normalized name
                drug_normalized = drug_key.lower()
                self.drugs[drug_normalized] = drug_info
                self.drug_index['by_type']['approved'].add(drug_normalized)
                self.drug_index['by_source']['fda_approved'].add(drug_normalized)  # Index by source
                
                # Index the drug name and variants
                self._index_drug_name(drug_normalized, drug_info)
                
                # Also index without "hydrochloride", "disodium", etc. for better matching
                drug_clean = drug_normalized
                for suffix in ['hydrochloride', 'disodium', 'sodium', 'potassium', 'calcium', 
                            'maleate', 'tartrate', 'citrate', 'sulfate', 'acetate']:
                    drug_clean = drug_clean.replace(f' {suffix}', '').replace(suffix, '')
                
                drug_clean = drug_clean.strip()
                if drug_clean != drug_normalized:
                    self._index_drug_name(drug_clean, drug_info)
                
                # Index brand names
                for brand_name in drug_group['brand_names']:
                    if brand_name:
                        brand_normalized = brand_name.lower()
                        self._index_drug_name(brand_normalized, drug_info)
            
            # Update statistics
            self.stats['fda_approved'] = len(drugs_by_key)
            
            # Count active vs discontinued
            active_count = sum(1 for drug_group in drugs_by_key.values() 
                            if any(status in ['Prescription', 'Over-the-counter'] 
                                    for status in drug_group['marketing_statuses']))
            discontinued_count = len(drugs_by_key) - active_count
            
            logger.info(f"Loaded {self.stats['fda_approved']} unique FDA approved drugs")
            logger.info(f"  - Active: {active_count}")
            logger.info(f"  - Discontinued: {discontinued_count}")
            logger.info(f"  - Total formulations: {len(data)}")
            
        except Exception as e:
            logger.error(f"Error processing FDA drugs: {e}", exc_info=True)

    def _process_investigational_drugs(self, data: Union[List, Dict]):
        """
        Process investigational drugs data from clinical trials.
        Expected format: List of dictionaries with keys: nctId, title, overallStatus, conditions, interventionName, interventionType
        """
        try:
            if not isinstance(data, list):
                logger.warning(f"Expected list format for investigational drugs, got {type(data)}")
                return
            
            logger.info(f"Processing {len(data)} investigational drug entries")
            
            # Group drugs by intervention name to consolidate trials
            drugs_by_name = defaultdict(lambda: {
                'nct_ids': set(),
                'conditions': set(),
                'statuses': set(),
                'trial_titles': set(),
                'intervention_types': set(),
                'entries': []
            })
            
            # First pass: group all entries by drug name
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                
                intervention_name = entry.get('interventionName', '').strip()
                if not intervention_name:
                    continue
                
                # Handle combination drugs
                if ' in combination with ' in intervention_name:
                    # Split combination drugs
                    parts = intervention_name.split(' in combination with ')
                    drugs = [parts[0].strip()] + [d.strip() for d in parts[1].split(' and ')]
                elif ' and ' in intervention_name.lower():
                    # Handle "drug1 and drug2" format
                    drugs = [d.strip() for d in intervention_name.split(' and ')]
                elif '/' in intervention_name:
                    # Handle "drug1/drug2" format
                    drugs = [d.strip() for d in intervention_name.split('/')]
                else:
                    # Single drug
                    drugs = [intervention_name]
                
                # Process each drug component
                for drug in drugs:
                    if not drug:
                        continue
                        
                    drug_group = drugs_by_name[drug]
                    drug_group['entries'].append(entry)
                    
                    # Store trial information
                    if nct_id := entry.get('nctId'):
                        drug_group['nct_ids'].add(nct_id)
                    
                    if status := entry.get('overallStatus'):
                        drug_group['statuses'].add(status)
                    
                    if title := entry.get('title'):
                        drug_group['trial_titles'].add(title)
                    
                    if intervention_type := entry.get('interventionType'):
                        drug_group['intervention_types'].add(intervention_type)
                    
                    # Add conditions
                    conditions = entry.get('conditions', [])
                    if isinstance(conditions, list):
                        drug_group['conditions'].update(conditions)
            
            # Second pass: create DrugInfo objects from grouped data
            for drug_name, drug_group in drugs_by_name.items():
                # Skip if no entries
                if not drug_group['entries']:
                    continue
                
                # Determine overall status (prioritize active trials)
                statuses = drug_group['statuses']
                if 'RECRUITING' in statuses:
                    overall_status = 'Recruiting'
                elif 'ACTIVE_NOT_RECRUITING' in statuses:
                    overall_status = 'Active'
                elif 'ENROLLING_BY_INVITATION' in statuses:
                    overall_status = 'Enrolling'
                elif 'NOT_YET_RECRUITING' in statuses:
                    overall_status = 'Not Yet Recruiting'
                elif 'COMPLETED' in statuses:
                    overall_status = 'Completed'
                elif 'TERMINATED' in statuses or 'WITHDRAWN' in statuses:
                    overall_status = 'Terminated/Withdrawn'
                else:
                    overall_status = 'Unknown'
                
                # Determine intervention type
                intervention_types = drug_group['intervention_types']
                if 'BIOLOGICAL' in intervention_types:
                    drug_type = 'investigational_biological'
                else:
                    drug_type = 'investigational'
                
                # Create DrugInfo object with specific source
                drug_info = DrugInfo(
                    name=drug_name,
                    drug_type=drug_type,
                    source='investigational',  # ← CHANGED: Using specific lexicon name
                    nct_ids=sorted(list(drug_group['nct_ids'])),
                    conditions=sorted(list(drug_group['conditions'])),
                    status=overall_status,
                    confidence=0.95,  # High confidence for clinical trial data
                    metadata={
                        'trial_count': len(drug_group['nct_ids']),
                        'trial_statuses': list(drug_group['statuses']),
                        'intervention_types': list(drug_group['intervention_types']),
                        'trial_titles': list(drug_group['trial_titles'])[:5]  # Keep first 5 titles
                    }
                )
                
                # Store with normalized name
                drug_normalized = drug_name.lower()
                self.drugs[drug_normalized] = drug_info
                self.drug_index['by_type']['investigational'].add(drug_normalized)
                self.drug_index['by_source']['investigational'].add(drug_normalized)  # Index by source
                
                # Index the drug name
                self._index_drug_name(drug_normalized, drug_info)
                
                # Also index without common suffixes for better matching
                drug_clean = drug_normalized
                for suffix in ['hydrochloride', 'sodium', 'potassium', 'maleate', 'sulfate']:
                    drug_clean = drug_clean.replace(f' {suffix}', '').replace(suffix, '')
                
                drug_clean = drug_clean.strip()
                if drug_clean != drug_normalized:
                    self._index_drug_name(drug_clean, drug_info)
                
                # Index by NCT IDs
                for nct_id in drug_group['nct_ids']:
                    self.drug_index['by_nct'][nct_id].add(drug_normalized)
                
                # Index by conditions
                for condition in drug_group['conditions']:
                    condition_lower = condition.lower()
                    self.drug_index['by_condition'][condition_lower].add(drug_normalized)
            
            # Update statistics
            self.stats['investigational'] = len(drugs_by_name)
            
            # Count by status
            status_counts = defaultdict(int)
            for drug_group in drugs_by_name.values():
                for status in drug_group['statuses']:
                    status_counts[status] += 1
            
            logger.info(f"Loaded {self.stats['investigational']} unique investigational drugs")
            logger.info(f"  - Total clinical trial entries: {len(data)}")
            logger.info(f"  - Unique trials: {len(set(entry.get('nctId') for entry in data if entry.get('nctId')))}")
            
            # Log status breakdown
            logger.info("  Trial status breakdown:")
            for status, count in sorted(status_counts.items()):
                logger.info(f"    - {status}: {count}")
            
        except Exception as e:
            logger.error(f"Error processing investigational drugs: {e}", exc_info=True)

    def _process_alexion_drugs(self, data: Dict):
        """
        Process Alexion drugs data.
        Expected format: Dictionary with 'known_drugs' and 'drug_types' keys
        """
        try:
            if not isinstance(data, dict):
                logger.warning(f"Expected dict format for Alexion drugs, got {type(data)}")
                return
            
            # Extract the relevant sections
            known_drugs = data.get('known_drugs', {})
            drug_types = data.get('drug_types', {})
            metadata = data.get('metadata', {})
            
            logger.info(f"Processing {len(known_drugs)} Alexion drugs")
            
            # Process each drug and its variants
            for primary_name, variants in known_drugs.items():
                if not primary_name:
                    continue
                
                # Get the drug type (approved/investigational)
                drug_status = drug_types.get(primary_name, 'unknown')
                
                # Create DrugInfo object with specific source
                drug_info = DrugInfo(
                    name=primary_name,
                    drug_type='alexion',
                    source='alexion',  # ← CHANGED: Using specific lexicon name
                    status=drug_status,
                    synonyms=[] if not variants else [v for v in variants if v != primary_name],
                    confidence=1.0,  # High confidence for Alexion's own drugs
                    metadata={
                        'alexion_drug': True,
                        'approval_status': drug_status,
                        'source_metadata': metadata
                    }
                )
                
                # Separate brand names and code names
                brand_names = []
                code_names = []
                
                for variant in variants:
                    if variant != primary_name:
                        # Check if it's a code name (starts with ALXN)
                        if variant.upper().startswith('ALXN'):
                            code_names.append(variant)
                        # Check if it's likely a brand name (capitalized, single word)
                        elif variant[0].isupper() and ' ' not in variant:
                            brand_names.append(variant)
                        else:
                            # Keep as synonym
                            if variant not in drug_info.synonyms:
                                drug_info.synonyms.append(variant)
                
                # Store brand names and code names
                drug_info.brand_names = brand_names
                if code_names:
                    drug_info.metadata['code_names'] = code_names
                
                # Store in knowledge base with normalized name
                primary_normalized = primary_name.lower()
                self.drugs[primary_normalized] = drug_info
                
                # Add to Alexion index
                self.drug_index['by_type']['alexion'].add(primary_normalized)
                self.drug_index['by_source']['alexion'].add(primary_normalized)  # Index by source
                
                # Also categorize by approval status
                if drug_status == 'approved':
                    self.drug_index['by_type']['approved'].add(primary_normalized)
                elif drug_status == 'investigational':
                    self.drug_index['by_type']['investigational'].add(primary_normalized)
                
                # Index all variants for quick lookup
                all_variants = [primary_name] + (variants if variants else [])
                for variant in all_variants:
                    if variant:
                        variant_normalized = variant.lower()
                        self._index_drug_name(variant_normalized, drug_info)
                        
                        # Also index without hyphens/spaces for better matching
                        variant_clean = variant_normalized.replace('-', '').replace(' ', '')
                        if variant_clean != variant_normalized:
                            self._index_drug_name(variant_clean, drug_info)
                        
                        # Index ALXN codes in uppercase too
                        if variant.upper().startswith('ALXN'):
                            self._index_drug_name(variant.upper(), drug_info)
            
            # Update statistics
            self.stats['alexion'] = len(self.drug_index['by_type'].get('alexion', set()))
            logger.info(f"Loaded {self.stats['alexion']} Alexion drugs")
            
            # Log breakdown by status
            alexion_approved = sum(1 for drug in known_drugs if drug_types.get(drug) == 'approved')
            alexion_investigational = sum(1 for drug in known_drugs if drug_types.get(drug) == 'investigational')
            logger.info(f"  - Approved: {alexion_approved}")
            logger.info(f"  - Investigational: {alexion_investigational}")
            
        except Exception as e:
            logger.error(f"Error processing Alexion drugs: {e}", exc_info=True)

    def _process_drug_patterns(self, data: Union[List, Dict]):
        """Process drug detection patterns."""
        if isinstance(data, list):
            for pattern_entry in data:
                if isinstance(pattern_entry, dict):
                    self._add_drug_pattern(pattern_entry)
                elif isinstance(pattern_entry, str):
                    # Handle string patterns
                    self._add_drug_pattern({
                        'pattern': pattern_entry,
                        'pattern_type': 'regex',
                        'confidence': 0.7,
                        'description': ''
                    })
        elif isinstance(data, dict):
            # Process suffix patterns
            suffix_patterns = data.get('suffix_patterns', [])
            if isinstance(suffix_patterns, list):
                for pattern in suffix_patterns:
                    if isinstance(pattern, dict):
                        self._add_drug_pattern({
                            'pattern': pattern.get('pattern', ''),
                            'pattern_type': 'suffix',
                            'confidence': pattern.get('confidence', 0.8),
                            'description': pattern.get('description', '')
                        })
                    elif isinstance(pattern, str):
                        self._add_drug_pattern({
                            'pattern': pattern,
                            'pattern_type': 'suffix',
                            'confidence': 0.8,
                            'description': ''
                        })
            
            # Process prefix patterns
            prefix_patterns = data.get('prefix_patterns', [])
            if isinstance(prefix_patterns, list):
                for pattern in prefix_patterns:
                    if isinstance(pattern, dict):
                        self._add_drug_pattern({
                            'pattern': pattern.get('pattern', ''),
                            'pattern_type': 'prefix',
                            'confidence': pattern.get('confidence', 0.8),
                            'description': pattern.get('description', '')
                        })
                    elif isinstance(pattern, str):
                        self._add_drug_pattern({
                            'pattern': pattern,
                            'pattern_type': 'prefix',
                            'confidence': 0.8,
                            'description': ''
                        })
        
        self.stats['patterns'] = len(self.drug_patterns)
    
    def _add_drug_pattern(self, pattern_entry: Dict):
        """Add a drug detection pattern."""
        pattern = DrugPattern(
            pattern=pattern_entry.get('pattern', ''),
            pattern_type=pattern_entry.get('pattern_type', 'regex'),
            confidence=pattern_entry.get('confidence', 0.7),
            description=pattern_entry.get('description', '')
        )
        
        if pattern.pattern:
            self.drug_patterns.append(pattern)
    
    def _process_drug_lexicon(self, data: Union[List, Dict]):
        """
        Process drug lexicon data from RxNorm format.
        Handles both list and dict formats.
        
        Expected formats:
        - List: List of dictionaries with keys: term, term_normalized, rxcui, tty
        - Dict: Dictionary with drug names as keys and info as values
        
        TTY (Term Type) values:
        - IN: Ingredient
        - BN: Brand Name  
        - SY: Synonym
        - SCD: Semantic Clinical Drug
        - SCDC: Semantic Clinical Drug Component
        - SCDF: Semantic Clinical Drug Form
        - SBD: Semantic Branded Drug
        - SBDC: Semantic Branded Drug Component
        - SBDF: Semantic Branded Drug Form
        """
        try:
            # Handle dict format (convert to list format)
            if isinstance(data, dict):
                logger.info(f"Processing drug lexicon dictionary with {len(data)} entries")
                
                # Convert dict to list format
                list_data = []
                for drug_name, drug_info in data.items():
                    if isinstance(drug_info, dict):
                        # Add the drug name to the entry if not present
                        entry = drug_info.copy()
                        if 'term' not in entry:
                            entry['term'] = drug_name
                        list_data.append(entry)
                    elif isinstance(drug_info, str):
                        # Simple string value, create basic entry
                        list_data.append({
                            'term': drug_name,
                            'term_normalized': drug_info.lower()
                        })
                    else:
                        # Just use the drug name
                        list_data.append({
                            'term': drug_name,
                            'term_normalized': drug_name.lower()
                        })
                
                # Process as list
                data = list_data
            
            # Now process as list
            if not isinstance(data, list):
                logger.warning(f"Expected list or dict format for drug lexicon, got {type(data)}")
                return
            
            logger.info(f"Processing drug lexicon with {len(data)} entries")
            
            # Group drugs by rxcui to consolidate related terms
            drugs_by_rxcui = defaultdict(lambda: {
                'names': set(),
                'brand_names': set(),
                'synonyms': set(),
                'ingredients': set(),
                'rxcui': None
            })
            
            # First pass: group all terms by rxcui
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                    
                term = entry.get('term', '').strip()
                term_normalized = entry.get('term_normalized', term.lower())
                rxcui = entry.get('rxcui', '')
                tty = entry.get('tty', '')
                
                if not term:
                    continue
                
                # Store by rxcui for grouping
                if rxcui:
                    drug_group = drugs_by_rxcui[rxcui]
                    drug_group['rxcui'] = rxcui
                    
                    # Categorize by term type
                    if tty == 'BN':  # Brand Name
                        drug_group['brand_names'].add(term)
                    elif tty == 'IN':  # Ingredient
                        drug_group['ingredients'].add(term)
                    elif tty in ['SY', 'SCD', 'SBD', 'SCDC', 'SBDC', 'SCDF', 'SBDF']:  # Various synonyms
                        drug_group['synonyms'].add(term)
                    else:
                        drug_group['names'].add(term)
                else:
                    # No rxcui - create standalone entry with specific source
                    drug_info = DrugInfo(
                        name=term,
                        drug_type='lexicon',
                        source='drug_lexicon',  # ← CHANGED: Using specific lexicon name
                        confidence=0.9,
                        metadata={'tty': tty, 'term_normalized': term_normalized}
                    )
                    
                    self.drugs[term_normalized] = drug_info
                    self.drug_index['by_type']['lexicon'].add(term_normalized)
                    self.drug_index['by_source']['drug_lexicon'].add(term_normalized)  # Index by source
                    self._index_drug_name(term_normalized, drug_info)
            
            # Second pass: create DrugInfo objects from grouped data
            for rxcui, drug_group in drugs_by_rxcui.items():
                # Determine primary name (prefer ingredient name)
                primary_name = None
                if drug_group['ingredients']:
                    primary_name = list(drug_group['ingredients'])[0]
                elif drug_group['names']:
                    primary_name = list(drug_group['names'])[0]
                elif drug_group['brand_names']:
                    primary_name = list(drug_group['brand_names'])[0]
                elif drug_group['synonyms']:
                    primary_name = list(drug_group['synonyms'])[0]
                
                if not primary_name:
                    continue
                
                # Create consolidated drug info with specific source
                drug_info = DrugInfo(
                    name=primary_name,
                    drug_type='lexicon',
                    source='drug_lexicon',  # ← CHANGED: Using specific lexicon name
                    rxcui=rxcui,
                    brand_names=list(drug_group['brand_names']),
                    generic_names=list(drug_group['ingredients']),
                    synonyms=list(drug_group['synonyms'] | drug_group['names']),
                    confidence=0.95,  # Higher confidence for rxcui-grouped drugs
                    metadata={'rxcui': rxcui}
                )
                
                # Store under normalized primary name
                normalized_name = primary_name.lower().strip()
                self.drugs[normalized_name] = drug_info
                self.drug_index['by_type']['lexicon'].add(normalized_name)
                self.drug_index['by_source']['drug_lexicon'].add(normalized_name)  # Index by source
                self.drug_index['by_rxcui'][rxcui] = drug_info
                
                # Index all variants
                self._index_drug_name(normalized_name, drug_info)
                for variant in drug_group['brand_names'] | drug_group['synonyms'] | drug_group['names'] | drug_group['ingredients']:
                    self._index_drug_name(variant.lower().strip(), drug_info)
            
            self.stats['total_drugs'] = len(self.drugs)
            logger.info(f"Successfully processed {len(self.drugs)} unique drugs from lexicon")
            
        except Exception as e:
            logger.error(f"Failed to process drug lexicon: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _index_drug_name(self, name: str, drug_info: DrugInfo):
        """
        Helper method to index a drug name for quick lookup.
        Allows multiple drugs to be associated with the same name.
        """
        if not name:
            return
            
        if name not in self.drug_index['by_name']:
            self.drug_index['by_name'][name] = []
        
        # Only add if not already indexed for this drug
        if drug_info not in self.drug_index['by_name'][name]:
            self.drug_index['by_name'][name].append(drug_info)
    
    def _build_indices(self):
        """Build search indices for efficient lookup."""
        for drug_name, drug_info in self.drugs.items():
            # Index by type (already done in processing methods)
            # Skip to avoid duplication
            
            # Index by source is now done in processing methods
            # Skip to avoid duplication
            
            # Build normalized index
            normalized = self._normalize_drug_name(drug_name)
            self.drug_index['normalized'][normalized] = drug_name
            
            # Also index synonyms and brand names
            for synonym in drug_info.synonyms + drug_info.brand_names:
                normalized_syn = self._normalize_drug_name(synonym)
                self.drug_index['normalized'][normalized_syn] = drug_name
        
        self.stats['total_drugs'] = len(self.drugs)
        self.stats['total_synonyms'] = len(self.drug_index['normalized'])
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        for pattern in self.drug_patterns:
            try:
                if pattern.pattern_type == 'suffix':
                    regex = rf'\b\w+{re.escape(pattern.pattern)}\b'
                elif pattern.pattern_type == 'prefix':
                    regex = rf'\b{re.escape(pattern.pattern)}\w+\b'
                else:
                    regex = pattern.pattern
                
                compiled = re.compile(regex, re.IGNORECASE)
                pattern.compiled_pattern = compiled
                self.compiled_patterns[pattern.pattern_type].append((compiled, pattern))
            
            except Exception as e:
                logger.error(f"Failed to compile pattern {pattern.pattern}: {e}")
    
    def _normalize_drug_name(self, name: str) -> str:
        """Normalize drug name for matching."""
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove common suffixes
        for suffix in [' injection', ' tablet', ' capsule', ' solution', ' oral', ' iv']:
            normalized = normalized.replace(suffix, '')
        
        # Remove special characters but keep hyphens
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    def _log_statistics(self):
        """Log knowledge base statistics with source breakdown."""
        logger.info("Drug Knowledge Base Statistics:")
        logger.info(f"  - FDA Approved: {self.stats['fda_approved']}")
        logger.info(f"  - Investigational: {self.stats['investigational']}")
        logger.info(f"  - Alexion: {self.stats['alexion']}")
        logger.info(f"  - Patterns: {self.stats['patterns']}")
        logger.info(f"  - Total Drugs: {self.stats['total_drugs']}")
        logger.info(f"  - Total Synonyms/Variants: {self.stats['total_synonyms']}")
        
        # Log source breakdown
        logger.info("  Drugs by source:")
        for source, drugs in self.drug_index['by_source'].items():
            logger.info(f"    - {source}: {len(drugs)}")
    
    # ========================================================================
    # Public Query Methods
    # ========================================================================
    
    def get_drug_info(self, drug_name: str) -> Optional[DrugInfo]:
        """
        Get drug information by name.
        
        Args:
            drug_name: Drug name to look up
            
        Returns:
            DrugInfo object or None if not found
        """
        # Direct lookup
        if drug_name in self.drugs:
            return self.drugs[drug_name]
        
        # Try normalized lookup
        normalized = self._normalize_drug_name(drug_name)
        if canonical_name := self.drug_index['normalized'].get(normalized):
            return self.drugs.get(canonical_name)
        
        # Try looking up in by_name index
        if drug_name.lower() in self.drug_index['by_name']:
            drug_infos = self.drug_index['by_name'][drug_name.lower()]
            if drug_infos:
                return drug_infos[0]  # Return first match
        
        return None
    
    def is_known_drug(self, drug_name: str) -> bool:
        """Check if a drug name is in the knowledge base."""
        return self.get_drug_info(drug_name) is not None
    
    def is_fda_approved(self, drug_name: str) -> bool:
        """Check if a drug is FDA approved."""
        drug_info = self.get_drug_info(drug_name)
        return drug_info and (drug_info.drug_type == 'approved' or drug_info.source == 'fda_approved')
    
    def is_investigational(self, drug_name: str) -> bool:
        """Check if a drug is investigational."""
        drug_info = self.get_drug_info(drug_name)
        return drug_info and (drug_info.drug_type in ['investigational', 'investigational_biological'] or drug_info.source == 'investigational')
    
    def is_alexion_drug(self, drug_name: str) -> bool:
        """Check if a drug is an Alexion drug."""
        drug_info = self.get_drug_info(drug_name)
        return drug_info and (drug_info.drug_type == 'alexion' or drug_info.source == 'alexion')
    
    def get_drugs_by_type(self, drug_type: str) -> Set[str]:
        """Get all drugs of a specific type."""
        return self.drug_index['by_type'].get(drug_type, set())
    
    def get_drugs_by_source(self, source: str) -> Set[str]:
        """Get all drugs from a specific source/lexicon."""
        return self.drug_index['by_source'].get(source, set())
    
    def get_drugs_by_condition(self, condition: str) -> Set[str]:
        """Get all drugs being tested for a condition."""
        return self.drug_index['by_condition'].get(condition.lower(), set())
    
    def get_drug_by_rxcui(self, rxcui: str) -> Optional[DrugInfo]:
        """Get drug info by RxCUI."""
        return self.drug_index['by_rxcui'].get(rxcui)
    
    def search_drugs(self, query: str) -> List[Tuple[str, DrugInfo]]:
        """
        Search for drugs matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of (drug_name, DrugInfo) tuples
        """
        results = []
        query_lower = query.lower()
        query_normalized = self._normalize_drug_name(query)
        
        for drug_name, drug_info in self.drugs.items():
            # Check main name
            if query_lower in drug_name.lower():
                results.append((drug_name, drug_info))
                continue
            
            # Check synonyms and brand names
            for variant in drug_info.synonyms + drug_info.brand_names:
                if query_lower in variant.lower():
                    results.append((drug_name, drug_info))
                    break
        
        return results
    
    def match_patterns(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Match drug patterns in text.
        
        Args:
            text: Text to search
            
        Returns:
            List of (matched_text, confidence, pattern_description) tuples
        """
        matches = []
        
        for pattern_type, patterns in self.compiled_patterns.items():
            for compiled_pattern, pattern_info in patterns:
                for match in compiled_pattern.finditer(text):
                    matched_text = match.group()
                    matches.append((
                        matched_text,
                        pattern_info.confidence,
                        pattern_info.description
                    ))
        
        return matches
    
    def get_all_drug_names(self) -> Set[str]:
        """Get all drug names in the knowledge base."""
        return set(self.drugs.keys())
    
    def get_all_drug_variants(self) -> Set[str]:
        """Get all drug names including synonyms and brand names."""
        variants = set(self.drugs.keys())
        
        for drug_info in self.drugs.values():
            variants.update(drug_info.synonyms)
            variants.update(drug_info.brand_names)
        
        return variants
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics including source breakdown."""
        stats = self.stats.copy()
        
        # Add source breakdown
        stats['by_source'] = {
            source: len(drugs) 
            for source, drugs in self.drug_index['by_source'].items()
        }
        
        return stats


# ============================================================================
# Singleton Access Function
# ============================================================================

_knowledge_base_instance = None

def get_knowledge_base(system_initializer=None) -> DrugKnowledgeBase:
    """
    Get or create the singleton DrugKnowledgeBase instance.
    
    Args:
        system_initializer: MetadataSystemInitializer instance (optional)
        
    Returns:
        DrugKnowledgeBase singleton instance
    """
    global _knowledge_base_instance
    
    if _knowledge_base_instance is None:
        _knowledge_base_instance = DrugKnowledgeBase(system_initializer)
    
    return _knowledge_base_instance