#!/usr/bin/env python3
"""
Refactored Drug Validator with DrugKnowledgeBase Integration
============================================================
Location: corpus_metadata/document_utils/rare_disease_drug_validator.py

This refactored version:
1. Uses DrugKnowledgeBase as single source of truth for drug data
2. Uses config.yaml for all configuration (DRY principle)
3. Maintains medical terms filtering capability
4. Keeps Claude AI validation for final verification
5. Implements caching for performance

Version: 4.0 - Fully integrated with DrugKnowledgeBase
"""

import re
import json
import time
import logging
import hashlib
import warnings
import os
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

# Suppress regex warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Possible nested set")

# Import DrugKnowledgeBase (REQUIRED)
try:
    from corpus_metadata.document_utils.rare_disease_drug_knowledge_base import (
        DrugKnowledgeBase,
        get_knowledge_base
    )
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    raise ImportError("DrugKnowledgeBase is required for validation")

# Import system initializer for config and medical terms
try:
    from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
    SYSTEM_INIT_AVAILABLE = True
except ImportError:
    SYSTEM_INIT_AVAILABLE = False

# Optional Claude import
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: Anthropic not installed. Claude validation disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================

class ValidationDecision(Enum):
    """Validation decision types"""
    KEEP = "KEEP"
    FIX = "FIX"
    DROP = "DROP"

class DrugFormType(Enum):
    """Drug form types"""
    BASE = "base"
    SALT = "salt"
    COMBINATION = "combination"
    BIOSIMILAR = "biosimilar"
    VACCINE = "vaccine"

# ============================================================================
# Cache Implementation
# ============================================================================

class ValidationCache:
    """Simple in-memory cache for validation results"""
    
    def __init__(self, ttl_hours: int = 24, max_size: int = 1000):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, drug_list: List[Dict], context: Optional[str] = None) -> str:
        """Generate cache key"""
        drug_str = json.dumps(sorted([d.get('name', '') for d in drug_list]))
        context_str = (context or '')[:200]
        combined = f"{drug_str}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, drug_list: List[Dict], context: Optional[str] = None) -> Optional[Dict]:
        """Get cached result if available"""
        key = self._get_key(drug_list, context)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                self.hits += 1
                return entry['result']
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, drug_list: List[Dict], result: Dict, context: Optional[str] = None):
        """Cache validation result"""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        key = self._get_key(drug_list, context)
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now()
        }
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

# ============================================================================
# Main Validator Class
# ============================================================================

class DrugValidator:
    """
    Drug validator using DrugKnowledgeBase as single source of truth
    """
    
    def __init__(self, 
                 system_initializer: Optional['MetadataSystemInitializer'] = None,
                 claude_api_key: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize validator
        
        Args:
            system_initializer: MetadataSystemInitializer instance
            claude_api_key: Claude API key (uses env var if not provided)
            use_cache: Enable caching for Claude responses
        """
        # Initialize system if not provided
        if not system_initializer and SYSTEM_INIT_AVAILABLE:
            system_initializer = MetadataSystemInitializer()
        self.system_initializer = system_initializer
        
        # Initialize DrugKnowledgeBase (single source of truth)
        self.drug_kb = get_knowledge_base(system_initializer)
        logger.info(f"[OK] DrugKnowledgeBase initialized: {self.drug_kb.stats['total_drugs']} drugs")
        
        # Get configuration from config.yaml via system initializer
        self.config = self._load_config()
        
        # Initialize medical terms filter
        self.medical_terms = self._load_medical_terms()
        
        # Initialize Claude client
        self.claude_client = self._initialize_claude(claude_api_key)
        
        # Initialize cache
        self.cache = ValidationCache() if use_cache else None
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml via system initializer"""
        config = {
            # Default configuration
            'min_confidence': 0.5,
            'high_confidence': 0.8,
            'claude_model': 'claude-sonnet-4-5-20250929',
            'claude_max_tokens': 4096,
            'claude_temperature': 0.0,
            'max_context_length': 1000,
            
            # Common false positives (can be overridden by config.yaml)
            'common_false_positives': {
                'adapt', 'barrier', 'control', 'impact', 'information',
                'active', 'complete', 'central', 'nasal', 'pediatric',
                'therapeutic', 'net', 'acr', 'com', 'same', 'advantages',
                'gold', 'proteins', 'influenza', 'cartilage', 'http', 'https',
                'protein', 'creatinine', 'hemoglobin', 'albumin', 'glucose'
            },
            
            # Non-drug keywords
            'non_drug_keywords': {
                'cyp3a4', 'cathepsin c', 'factor b', 'anca',
                'membrane attack complex', 'c5ar1',
                'reduced-dose', 'standard-dose'
            }
        }
        
        # Try to get config from system initializer
        if self.system_initializer and hasattr(self.system_initializer, 'config'):
            sys_config = self.system_initializer.config
            
            # Get validation-specific config if available
            if validation_config := sys_config.get('validation'):
                config.update(validation_config)
            
            # Get features config
            if features := sys_config.get('features'):
                config['use_claude'] = features.get('ai_validation', True)
                config['use_medical_filter'] = features.get('medical_terms_filter', True)
        
        return config
    
    def _load_medical_terms(self) -> Set[str]:
        """Load medical terms for filtering"""
        medical_terms = set()
        
        # Basic medical terms (always included)
        basic_terms = {
            'protein', 'proteins', 'creatinine', 'hemoglobin', 'albumin',
            'glucose', 'cholesterol', 'triglycerides', 'enzyme', 'enzymes',
            'antibody', 'antibodies', 'antigen', 'antigens', 'cytokine',
            'receptor', 'receptors', 'hormone', 'hormones', 'vitamin',
            'mineral', 'electrolyte', 'lipid', 'lipids', 'carbohydrate',
            'barrier', 'control', 'impact', 'information', 'therapeutic'
        }
        medical_terms.update(basic_terms)
        
        # Try to get from system initializer
        if self.system_initializer:
            if hasattr(self.system_initializer, 'medical_terms_lexicon'):
                lexicon = self.system_initializer.medical_terms_lexicon
                if isinstance(lexicon, dict):
                    medical_terms.update(lexicon.keys())
                elif isinstance(lexicon, (list, set)):
                    medical_terms.update(lexicon)
                logger.info(f"[OK] Medical terms loaded: {len(medical_terms)} terms")
        
        return medical_terms
    
    def _initialize_claude(self, api_key: Optional[str] = None) -> Optional[anthropic.Anthropic]:
        """Initialize Claude client"""
        if not CLAUDE_AVAILABLE:
            return None
        
        if not api_key:
            api_key = os.getenv('CLAUDE_API_KEY')
        
        if not api_key:
            logger.warning("No Claude API key - AI validation disabled")
            return None
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            logger.info("[OK] Claude client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            return None
    
    def validate_drug_list(self,
                          drug_candidates: List[Any],
                          context_text: Optional[str] = None,
                          use_all_stages: bool = True) -> Dict[str, Any]:
        """
        Main validation method
        
        Args:
            drug_candidates: List of drug candidates (dicts, strings, or objects)
            context_text: Optional document context
            use_all_stages: Apply all validation stages
            
        Returns:
            Validation results dictionary
        """
        start_time = time.time()
        
        # Standardize input
        candidates = self._standardize_input(drug_candidates)
        logger.info(f"Validating {len(candidates)} drug candidates")
        
        # Track results
        results = {
            'input_count': len(candidates),
            'stages': [],
            'final_drugs': [],
            'removed_drugs': [],
            'processing_time': 0
        }
        
        # Stage 1: Remove common false positives
        if self.config.get('use_medical_filter', True):
            candidates, removed = self._filter_false_positives(candidates)
            results['stages'].append({
                'name': 'False Positives Filter',
                'removed': len(removed),
                'remaining': len(candidates)
            })
            results['removed_drugs'].extend(removed)
            logger.info(f"Stage 1: Removed {len(removed)} false positives")
        
        # Stage 2: Validate with DrugKnowledgeBase
        candidates, unknown = self._validate_with_kb(candidates)
        results['stages'].append({
            'name': 'DrugKnowledgeBase Validation',
            'validated': len(candidates),
            'unknown': len(unknown)
        })
        logger.info(f"Stage 2: Validated {len(candidates)} drugs with KB")
        
        # Stage 3: Claude validation (if enabled and available)
        if use_all_stages and self.claude_client and self.config.get('use_claude', True):
            candidates, rejected = self._validate_with_claude(candidates, context_text)
            results['stages'].append({
                'name': 'Claude AI Validation',
                'approved': len(candidates),
                'rejected': len(rejected)
            })
            results['removed_drugs'].extend(rejected)
            logger.info(f"Stage 3: Claude approved {len(candidates)}, rejected {len(rejected)}")
        
        # Final results
        results['final_drugs'] = candidates
        results['processing_time'] = time.time() - start_time
        
        # Update statistics
        self.stats['total_processed'] += results['input_count']
        self.stats['total_validated'] += len(candidates)
        
        logger.info(f"Validation complete: {len(candidates)}/{results['input_count']} drugs validated")
        
        return results
    
    def _standardize_input(self, candidates: List[Any]) -> List[Dict]:
        """Convert various input formats to standard dictionary"""
        standardized = []
        
        for candidate in candidates:
            if isinstance(candidate, dict):
                standardized.append(candidate)
            elif isinstance(candidate, str):
                standardized.append({'name': candidate})
            elif hasattr(candidate, '__dict__'):
                standardized.append(candidate.__dict__)
            elif hasattr(candidate, 'to_dict'):
                standardized.append(candidate.to_dict())
            else:
                standardized.append({'name': str(candidate)})
        
        return standardized
    
    def _filter_false_positives(self, candidates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Filter medical terms and false positives"""
        kept = []
        removed = []
        
        false_positives = self.config.get('common_false_positives', set())
        non_drug_keywords = self.config.get('non_drug_keywords', set())
        
        for candidate in candidates:
            name = candidate.get('name', '').lower()
            
            # Check against false positives and medical terms
            if (name in false_positives or 
                name in non_drug_keywords or
                name in self.medical_terms):
                removed.append(candidate)
                self.stats['filtered'] += 1
            else:
                kept.append(candidate)
        
        return kept, removed
    
    def _validate_with_kb(self, candidates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Validate and enrich using DrugKnowledgeBase"""
        validated = []
        unknown = []
        
        for candidate in candidates:
            name = candidate.get('name', '')
            
            # Look up in DrugKnowledgeBase
            drug_info = self.drug_kb.get_drug_info(name)
            
            if drug_info:
                # Enrich with KB data
                candidate['validated'] = True
                candidate['drug_type'] = drug_info.drug_type
                candidate['rxcui'] = drug_info.rxcui
                candidate['kb_source'] = drug_info.source
                candidate['kb_status'] = drug_info.status
                
                # Set confidence based on drug type
                if drug_info.drug_type == 'approved':
                    candidate['confidence'] = 0.95
                elif drug_info.drug_type == 'alexion':
                    candidate['confidence'] = 0.93
                elif drug_info.drug_type == 'investigational':
                    candidate['confidence'] = 0.90
                else:
                    candidate['confidence'] = 0.85
                
                # Add brand names and synonyms if available
                if drug_info.brand_names:
                    candidate['brand_names'] = drug_info.brand_names
                if drug_info.synonyms:
                    candidate['synonyms'] = drug_info.synonyms
                
                validated.append(candidate)
                self.stats['kb_validated'] += 1
            else:
                # Try searching for variants
                search_results = self.drug_kb.search_drugs(name)
                if search_results:
                    # Found as variant
                    drug_name, drug_info = search_results[0]
                    candidate['validated'] = True
                    candidate['canonical_name'] = drug_info.name
                    candidate['drug_type'] = drug_info.drug_type
                    candidate['rxcui'] = drug_info.rxcui
                    candidate['confidence'] = 0.85
                    validated.append(candidate)
                    self.stats['kb_validated'] += 1
                else:
                    # Not found in KB
                    candidate['validated'] = False
                    unknown.append(candidate)
                    
                    # Still include if high confidence from extraction
                    if candidate.get('confidence', 0) >= self.config.get('high_confidence', 0.8):
                        validated.append(candidate)
        
        return validated, unknown
    
    def _validate_with_claude(self, 
                             candidates: List[Dict], 
                             context: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """Validate with Claude AI"""
        if not candidates:
            return candidates, []
        
        # Check cache
        if self.cache:
            cached = self.cache.get(candidates, context)
            if cached:
                self.stats['cache_hits'] += 1
                return cached['approved'], cached['rejected']
        
        # Create prompt
        prompt = self._create_claude_prompt(candidates, context)
        
        try:
            # Call Claude
            response = self.claude_client.messages.create(
                model=self.config.get('claude_model'),
                max_tokens=self.config.get('claude_max_tokens'),
                temperature=self.config.get('claude_temperature'),
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            approved, rejected = self._parse_claude_response(
                response.content[0].text, 
                candidates
            )
            
            # Cache result
            if self.cache:
                self.cache.set(candidates, {
                    'approved': approved,
                    'rejected': rejected
                }, context)
            
            self.stats['api_calls'] += 1
            self.stats['claude_approved'] += len(approved)
            self.stats['claude_rejected'] += len(rejected)
            
            return approved, rejected
            
        except Exception as e:
            logger.error(f"Claude validation failed: {e}")
            self.stats['api_errors'] += 1
            # Return all as approved if Claude fails
            return candidates, []
    
    def _create_claude_prompt(self, candidates: List[Dict], context: Optional[str] = None) -> str:
        """Create Claude validation prompt"""
        # Prepare drug list
        drug_list = []
        for c in candidates[:100]:  # Limit to 100 drugs
            drug_list.append({
                'name': c.get('name'),
                'drug_type': c.get('drug_type'),
                'rxcui': c.get('rxcui'),
                'confidence': c.get('confidence', 0)
            })
        
        prompt = f"""You are a pharmacology expert. Validate these drug candidates.
Return ONLY a JSON array with your decisions.

Rules:
1. KEEP: Valid medications (ingredients, brand names, combinations)
2. DROP: Not drugs (enzymes, proteins, biomarkers, general terms)
3. For each drug, return: {{"name": "...", "decision": "KEEP/DROP", "reason": "..."}}

Drugs to validate:
{json.dumps(drug_list, indent=2)}
"""
        
        if context:
            prompt += f"\n\nContext:\n{context[:self.config.get('max_context_length', 1000)]}"
        
        prompt += "\n\nReturn ONLY the JSON array."
        
        return prompt
    
    def _parse_claude_response(self, response: str, candidates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Parse Claude response"""
        approved = []
        rejected = []
        
        try:
            # Extract JSON from response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if not match:
                logger.error("No JSON found in Claude response")
                return candidates, []
            
            decisions = json.loads(match.group())
            
            # Map decisions to candidates
            decision_map = {d['name'].lower(): d for d in decisions}
            
            for candidate in candidates:
                name = candidate.get('name', '').lower()
                decision = decision_map.get(name, {})
                
                if decision.get('decision') == 'KEEP':
                    candidate['claude_validated'] = True
                    candidate['claude_reason'] = decision.get('reason', '')
                    approved.append(candidate)
                else:
                    candidate['claude_validated'] = False
                    candidate['claude_reason'] = decision.get('reason', 'Not validated')
                    rejected.append(candidate)
            
            return approved, rejected
            
        except Exception as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return candidates, []
    
    def print_summary(self, results: Dict[str, Any]):
        """Print validation summary"""
        print("\n" + "="*60)
        print("DRUG VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nInput drugs: {results['input_count']}")
        print(f"Validated drugs: {len(results['final_drugs'])}")
        print(f"Removed drugs: {len(results['removed_drugs'])}")
        print(f"Processing time: {results['processing_time']:.2f}s")
        
        print("\nValidation Stages:")
        for stage in results['stages']:
            print(f"  - {stage['name']}: ", end="")
            for key, value in stage.items():
                if key != 'name':
                    print(f"{key}={value} ", end="")
            print()
        
        if results['final_drugs']:
            print("\nValidated Drugs (first 20):")
            for i, drug in enumerate(results['final_drugs'][:20], 1):
                name = drug.get('canonical_name', drug.get('name'))
                drug_type = drug.get('drug_type', 'unknown')
                confidence = drug.get('confidence', 0)
                print(f"  {i:2d}. {name:<30} [{drug_type}] (conf: {confidence:.2f})")
        
        print("\nStatistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics"""
        stats = dict(self.stats)
        stats['kb_stats'] = self.drug_kb.get_statistics()
        if self.cache:
            stats['cache_hit_rate'] = self.cache.get_hit_rate()
        return stats


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_drug_extraction(extraction_file: str,
                            output_file: Optional[str] = None,
                            context_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate drug extraction results from file
    
    Args:
        extraction_file: Path to extraction results
        output_file: Optional output file path
        context_file: Optional context file path
    
    Returns:
        Validation results
    """
    # Load extraction results
    with open(extraction_file, 'r') as f:
        data = json.load(f)
    
    # Extract drug candidates
    if isinstance(data, list):
        drugs = []
        for item in data:
            if isinstance(item, dict) and 'drugs' in item:
                drugs.extend(item['drugs'])
            else:
                drugs.append(item)
    elif isinstance(data, dict) and 'drugs' in data:
        drugs = data['drugs']
    else:
        drugs = [data] if not isinstance(data, list) else data
    
    # Load context if provided
    context = None
    if context_file and os.path.exists(context_file):
        with open(context_file, 'r') as f:
            context = f.read()
    
    # Create validator
    validator = DrugValidator()
    
    # Validate
    results = validator.validate_drug_list(drugs, context)
    
    # Print summary
    validator.print_summary(results)
    
    # Save if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results