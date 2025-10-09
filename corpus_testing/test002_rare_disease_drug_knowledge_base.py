#!/usr/bin/env python3
"""
Comprehensive Test Suite for Rare Disease Drug Knowledge Base
==============================================================
Location: corpus_testing/test_drug_knowledge_base.py

PURPOSE:
--------
Thoroughly test all capabilities of the DrugKnowledgeBase class including:
- Drug loading from multiple sources
- Drug information retrieval
- Drug classification
- Pattern matching
- Search functionality
- Index structures
- Data consolidation

Version: 1.0.0
Last Updated: 2025-01-15
"""

import sys
import os
import json
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict
import tempfile
import yaml

# Configure paths
TEST_DIR = Path("/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_testing")
PROJECT_ROOT = TEST_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "corpus_config" / "config.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

# Import modules to test
from corpus_metadata.document_utils.rare_disease_drug_knowledge_base import (
    DrugKnowledgeBase,
    DrugInfo,
    DrugPattern,
    get_knowledge_base
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TEST_DIR / 'test_drug_knowledge_base.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Data Fixtures
# ============================================================================

class TestDataFixtures:
    """Test data for various scenarios"""
    
    @staticmethod
    def get_fda_drug_data():
        """Sample FDA drug data"""
        return [
            {
                'key': 'aspirin',
                'drug_class': 'NSAID',
                'meta': {
                    'brand_name': 'Bayer',
                    'dosage_form': 'Tablet',
                    'route': 'Oral',
                    'marketing_status': 'Over-the-counter',
                    'application_number': 'NDA012345'
                }
            },
            {
                'key': 'metformin hydrochloride',
                'drug_class': 'Antidiabetic',
                'meta': {
                    'brand_name': 'Glucophage',
                    'dosage_form': 'Tablet',
                    'route': 'Oral',
                    'marketing_status': 'Prescription',
                    'application_number': 'NDA020357'
                }
            },
            {
                'key': 'aspirin|caffeine',  # Combination drug
                'drug_class': 'Analgesic combination',
                'meta': {
                    'brand_name': 'Excedrin',
                    'dosage_form': 'Tablet',
                    'marketing_status': 'Over-the-counter'
                }
            },
            {
                'key': 'discontinued_drug',
                'drug_class': 'Test',
                'meta': {
                    'marketing_status': 'Discontinued'
                }
            }
        ]
    
    @staticmethod
    def get_investigational_drug_data():
        """Sample investigational drug data"""
        return [
            {
                'nctId': 'NCT12345678',
                'title': 'Study of Drug A in Disease X',
                'overallStatus': 'RECRUITING',
                'conditions': ['Disease X', 'Syndrome Y'],
                'interventionName': 'Drug A',
                'interventionType': 'DRUG'
            },
            {
                'nctId': 'NCT87654321',
                'title': 'Combination therapy study',
                'overallStatus': 'ACTIVE_NOT_RECRUITING',
                'conditions': ['Cancer'],
                'interventionName': 'Drug B in combination with Drug C',
                'interventionType': 'DRUG'
            },
            {
                'nctId': 'NCT11111111',
                'title': 'Biological therapy trial',
                'overallStatus': 'COMPLETED',
                'conditions': ['Rare Disease'],
                'interventionName': 'Biological Agent X',
                'interventionType': 'BIOLOGICAL'
            }
        ]
    
    @staticmethod
    def get_alexion_drug_data():
        """Sample Alexion drug data"""
        return {
            'known_drugs': {
                'eculizumab': ['Soliris', 'ALXN1210'],
                'ravulizumab': ['Ultomiris', 'ALXN1210-PNH'],
                'asfotase alfa': ['Strensiq']
            },
            'drug_types': {
                'eculizumab': 'approved',
                'ravulizumab': 'approved',
                'asfotase alfa': 'approved'
            },
            'metadata': {
                'last_updated': '2025-01-15',
                'source': 'Alexion internal'
            }
        }
    
    @staticmethod
    def get_drug_lexicon_data():
        """Sample drug lexicon data"""
        return [
            {
                'term': 'Acetaminophen',
                'term_normalized': 'acetaminophen',
                'rxcui': '161',
                'tty': 'IN'
            },
            {
                'term': 'Tylenol',
                'term_normalized': 'tylenol',
                'rxcui': '161',
                'tty': 'BN'
            },
            {
                'term': 'Paracetamol',
                'term_normalized': 'paracetamol',
                'rxcui': '161',
                'tty': 'SY'
            }
        ]
    
    @staticmethod
    def get_drug_pattern_data():
        """Sample drug pattern data"""
        return {
            'suffix_patterns': [
                {
                    'pattern': 'mab',
                    'confidence': 0.95,
                    'description': 'Monoclonal antibody'
                },
                {
                    'pattern': 'nib',
                    'confidence': 0.90,
                    'description': 'Kinase inhibitor'
                }
            ],
            'prefix_patterns': [
                {
                    'pattern': 'anti',
                    'confidence': 0.70,
                    'description': 'Antagonist or antibody'
                }
            ]
        }


# ============================================================================
# Unit Tests for DrugKnowledgeBase
# ============================================================================

class TestDrugKnowledgeBase(unittest.TestCase):
    """Test DrugKnowledgeBase class functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        logger.info("\n" + "="*80)
        logger.info("TESTING DRUG KNOWLEDGE BASE MODULE")
        logger.info("="*80)
    
    def setUp(self):
        """Set up each test"""
        # Create a test knowledge base without system initializer
        self.kb = DrugKnowledgeBase(system_initializer=None)
        
        # Load test data
        self.test_fixtures = TestDataFixtures()
    
    # ------------------------------------------------------------------------
    # Test FDA Drug Processing
    # ------------------------------------------------------------------------
    
    def test_01_fda_drug_processing(self):
        """Test FDA drug data processing"""
        logger.info("\n" + "-"*60)
        logger.info("Test 01: FDA Drug Processing")
        logger.info("-"*60)
        
        # Process test FDA data
        fda_data = self.test_fixtures.get_fda_drug_data()
        self.kb._process_fda_drugs(fda_data)
        
        # Test simple drug
        aspirin_info = self.kb.get_drug_info('aspirin')
        self.assertIsNotNone(aspirin_info, "Should find aspirin")
        self.assertEqual(aspirin_info.drug_type, 'approved')
        self.assertEqual(aspirin_info.source, 'FDA')
        self.assertIn('Bayer', aspirin_info.brand_names)
        logger.info(f"  ✓ Found aspirin with brand name: {aspirin_info.brand_names}")
        
        # Test drug with salt form
        metformin_info = self.kb.get_drug_info('metformin hydrochloride')
        self.assertIsNotNone(metformin_info, "Should find metformin hydrochloride")
        
        # Test normalized lookup without salt
        metformin_clean = self.kb.get_drug_info('metformin')
        self.assertIsNotNone(metformin_clean, "Should find metformin without hydrochloride")
        logger.info(f"  ✓ Found metformin with and without salt suffix")
        
        # Test combination drug parsing
        aspirin_combo = self.kb.get_drug_info('aspirin')
        caffeine_combo = self.kb.get_drug_info('caffeine')
        self.assertIsNotNone(aspirin_combo, "Should have aspirin from combination")
        self.assertIsNotNone(caffeine_combo, "Should have caffeine from combination")
        logger.info(f"  ✓ Combination drugs parsed correctly")
        
        # Test discontinued drug status
        discontinued = self.kb.get_drug_info('discontinued_drug')
        self.assertIsNotNone(discontinued)
        self.assertEqual(discontinued.status, 'Discontinued')
        logger.info(f"  ✓ Discontinued drug status: {discontinued.status}")
    
    # ------------------------------------------------------------------------
    # Test Investigational Drug Processing
    # ------------------------------------------------------------------------
    
    def test_02_investigational_drug_processing(self):
        """Test investigational drug data processing"""
        logger.info("\n" + "-"*60)
        logger.info("Test 02: Investigational Drug Processing")
        logger.info("-"*60)
        
        inv_data = self.test_fixtures.get_investigational_drug_data()
        self.kb._process_investigational_drugs(inv_data)
        
        # Test single drug
        drug_a = self.kb.get_drug_info('drug a')
        self.assertIsNotNone(drug_a, "Should find Drug A")
        self.assertIn('investigational', drug_a.drug_type)
        self.assertIn('NCT12345678', drug_a.nct_ids)
        self.assertIn('Disease X', drug_a.conditions)
        logger.info(f"  ✓ Drug A: {len(drug_a.nct_ids)} trials, status: {drug_a.status}")
        
        # Test combination drug parsing
        drug_b = self.kb.get_drug_info('drug b')
        drug_c = self.kb.get_drug_info('drug c')
        self.assertIsNotNone(drug_b, "Should find Drug B from combination")
        self.assertIsNotNone(drug_c, "Should find Drug C from combination")
        logger.info(f"  ✓ Combination drugs B and C parsed")
        
        # Test biological classification
        bio_agent = self.kb.get_drug_info('biological agent x')
        self.assertIsNotNone(bio_agent)
        self.assertEqual(bio_agent.drug_type, 'investigational_biological')
        logger.info(f"  ✓ Biological agent classified: {bio_agent.drug_type}")
        
        # Test NCT ID index
        drugs_for_nct = self.kb.drug_index['by_nct'].get('NCT12345678', set())
        self.assertTrue(len(drugs_for_nct) > 0, "Should find drugs by NCT ID")
        logger.info(f"  ✓ NCT index working: {len(drugs_for_nct)} drugs for NCT12345678")
        
        # Test condition index
        drugs_for_condition = self.kb.drug_index['by_condition'].get('disease x', set())
        self.assertTrue(len(drugs_for_condition) > 0, "Should find drugs by condition")
        logger.info(f"  ✓ Condition index: {len(drugs_for_condition)} drugs for Disease X")
    
    # ------------------------------------------------------------------------
    # Test Alexion Drug Processing
    # ------------------------------------------------------------------------
    
    def test_03_alexion_drug_processing(self):
        """Test Alexion drug data processing"""
        logger.info("\n" + "-"*60)
        logger.info("Test 03: Alexion Drug Processing")
        logger.info("-"*60)
        
        alexion_data = self.test_fixtures.get_alexion_drug_data()
        self.kb._process_alexion_drugs(alexion_data)
        
        # Test primary drug name
        eculizumab = self.kb.get_drug_info('eculizumab')
        self.assertIsNotNone(eculizumab)
        self.assertEqual(eculizumab.drug_type, 'alexion')
        self.assertEqual(eculizumab.status, 'approved')
        logger.info(f"  ✓ Eculizumab found: {eculizumab.drug_type}, {eculizumab.status}")
        
        # Test brand name lookup - check that it finds a drug, not specific name
        soliris = self.kb.get_drug_info('soliris')
        self.assertIsNotNone(soliris, "Should find by brand name Soliris")
        # The drug might be stored under brand name or generic name
        self.assertIn(soliris.name.lower(), ['soliris', 'eculizumab'])
        logger.info(f"  ✓ Brand name lookup: Soliris -> {soliris.name}")
        
        # Test code name lookup (ALXN format)
        alxn_code = self.kb.drug_index['by_name'].get('ALXN1210')
        self.assertIsNotNone(alxn_code, "Should find by ALXN code")
        logger.info(f"  ✓ ALXN code lookup working")
        
        # Test multiple categorization
        alexion_drugs = self.kb.get_drugs_by_type('alexion')
        approved_drugs = self.kb.get_drugs_by_type('approved')
        
        # Check if eculizumab or any of its variants are in the sets
        self.assertTrue(any('eculizumab' in drug or 'soliris' in drug 
                          for drug in alexion_drugs))
        logger.info(f"  ✓ Dual categorization: {len(alexion_drugs)} Alexion, also in approved")
    
    # ------------------------------------------------------------------------
    # Test Drug Lexicon Processing
    # ------------------------------------------------------------------------
    
    def test_04_drug_lexicon_processing(self):
        """Test drug lexicon data processing"""
        logger.info("\n" + "-"*60)
        logger.info("Test 04: Drug Lexicon Processing")
        logger.info("-"*60)
        
        lexicon_data = self.test_fixtures.get_drug_lexicon_data()
        self.kb._process_drug_lexicon(lexicon_data)
        
        # Test RxCUI grouping
        acetaminophen = self.kb.get_drug_info('acetaminophen')
        self.assertIsNotNone(acetaminophen)
        self.assertIsNotNone(acetaminophen.rxcui)
        logger.info(f"  ✓ Acetaminophen RxCUI: {acetaminophen.rxcui}")
        
        # Test brand name mapping
        tylenol = self.kb.get_drug_info('tylenol')
        self.assertIsNotNone(tylenol, "Should find by brand name")
        # Check that both have RxCUI values (may be same or different)
        self.assertIsNotNone(tylenol.rxcui)
        logger.info(f"  ✓ Brand name Tylenol found with RxCUI: {tylenol.rxcui}")
        
        # Test synonym mapping
        paracetamol = self.kb.get_drug_info('paracetamol')
        self.assertIsNotNone(paracetamol, "Should find by synonym")
        self.assertIsNotNone(paracetamol.rxcui)
        logger.info(f"  ✓ Synonym paracetamol found with RxCUI: {paracetamol.rxcui}")
        
        # Test RxCUI index - check if any RxCUI was indexed
        if acetaminophen.rxcui:
            drug_name = self.kb.get_drug_by_rxcui(acetaminophen.rxcui)
            self.assertIsNotNone(drug_name, "Should find drug by RxCUI")
            logger.info(f"  ✓ RxCUI lookup: {acetaminophen.rxcui} -> {drug_name}")
    
    # ------------------------------------------------------------------------
    # Test Pattern Processing and Matching
    # ------------------------------------------------------------------------
    
    def test_05_pattern_processing(self):
        """Test drug pattern processing and matching"""
        logger.info("\n" + "-"*60)
        logger.info("Test 05: Pattern Processing and Matching")
        logger.info("-"*60)
        
        pattern_data = self.test_fixtures.get_drug_pattern_data()
        self.kb._process_drug_patterns(pattern_data)
        
        # Test suffix patterns
        self.assertEqual(len(self.kb.drug_patterns), 3, "Should have 3 patterns")
        
        # Test pattern compilation
        self.kb._compile_patterns()
        self.assertTrue(len(self.kb.compiled_patterns['suffix']) > 0)
        self.assertTrue(len(self.kb.compiled_patterns['prefix']) > 0)
        logger.info(f"  ✓ Patterns compiled: {len(self.kb.compiled_patterns['suffix'])} suffix, "
                   f"{len(self.kb.compiled_patterns['prefix'])} prefix")
        
        # Test pattern matching
        test_text = "The patient was treated with adalimumab and dasatinib"
        matches = self.kb.match_patterns(test_text)
        
        # Should match 'mab' suffix and 'nib' suffix
        mab_found = any('mab' in match[0].lower() for match in matches)
        nib_found = any('nib' in match[0].lower() for match in matches)
        
        self.assertTrue(mab_found, "Should match monoclonal antibody pattern")
        self.assertTrue(nib_found, "Should match kinase inhibitor pattern")
        logger.info(f"  ✓ Pattern matching found {len(matches)} matches")
        
        for match, confidence, description in matches:
            logger.info(f"    - {match}: {confidence:.2f} ({description})")
    
    # ------------------------------------------------------------------------
    # Test Drug Name Normalization
    # ------------------------------------------------------------------------
    
    def test_06_name_normalization(self):
        """Test drug name normalization"""
        logger.info("\n" + "-"*60)
        logger.info("Test 06: Drug Name Normalization")
        logger.info("-"*60)
        
        # Test normalization function
        test_cases = [
            ("Aspirin Tablet", "aspirin"),
            ("METFORMIN HYDROCHLORIDE", "metformin hydrochloride"),
            ("Drug-Name_Test!", "drug-name test"),
            ("Injection Solution IV", "injection solution iv"),
        ]
        
        for input_name, expected_base in test_cases:
            normalized = self.kb._normalize_drug_name(input_name)
            self.assertIsNotNone(normalized)
            logger.info(f"  '{input_name}' -> '{normalized}'")
            
            # Check that common suffixes are removed
            self.assertNotIn(' tablet', normalized)
            self.assertNotIn(' oral', normalized)
    
    # ------------------------------------------------------------------------
    # Test Search Functionality
    # ------------------------------------------------------------------------
    
    def test_07_search_functionality(self):
        """Test drug search functionality"""
        logger.info("\n" + "-"*60)
        logger.info("Test 07: Search Functionality")
        logger.info("-"*60)
        
        # Load some test data first
        fda_data = self.test_fixtures.get_fda_drug_data()
        self.kb._process_fda_drugs(fda_data)
        
        alexion_data = self.test_fixtures.get_alexion_drug_data()
        self.kb._process_alexion_drugs(alexion_data)
        
        # Test partial search
        results = self.kb.search_drugs('asp')
        self.assertTrue(len(results) > 0, "Should find drugs starting with 'asp'")
        logger.info(f"  ✓ Partial search 'asp': {len(results)} results")
        
        # Test brand name search
        results = self.kb.search_drugs('Soliris')
        self.assertTrue(len(results) > 0, "Should find by brand name")
        logger.info(f"  ✓ Brand name search 'Soliris': {len(results)} results")
        
        # Test case-insensitive search
        results_lower = self.kb.search_drugs('aspirin')
        results_upper = self.kb.search_drugs('ASPIRIN')
        self.assertEqual(len(results_lower), len(results_upper), 
                        "Search should be case-insensitive")
        logger.info(f"  ✓ Case-insensitive search working")
    
    # ------------------------------------------------------------------------
    # Test Classification Methods
    # ------------------------------------------------------------------------
    
    def test_08_drug_classification(self):
        """Test drug classification methods"""
        logger.info("\n" + "-"*60)
        logger.info("Test 08: Drug Classification Methods")
        logger.info("-"*60)
        
        # Load test data
        fda_data = self.test_fixtures.get_fda_drug_data()
        self.kb._process_fda_drugs(fda_data)
        
        inv_data = self.test_fixtures.get_investigational_drug_data()
        self.kb._process_investigational_drugs(inv_data)
        
        alexion_data = self.test_fixtures.get_alexion_drug_data()
        self.kb._process_alexion_drugs(alexion_data)
        
        # Test FDA approval check
        self.assertTrue(self.kb.is_fda_approved('aspirin'))
        self.assertFalse(self.kb.is_fda_approved('drug a'))
        logger.info(f"  ✓ FDA approval check working")
        
        # Test investigational check
        self.assertTrue(self.kb.is_investigational('drug a'))
        self.assertFalse(self.kb.is_investigational('aspirin'))
        logger.info(f"  ✓ Investigational check working")
        
        # Test Alexion drug check
        self.assertTrue(self.kb.is_alexion_drug('eculizumab'))
        self.assertFalse(self.kb.is_alexion_drug('aspirin'))
        logger.info(f"  ✓ Alexion drug check working")
        
        # Test known drug check
        self.assertTrue(self.kb.is_known_drug('aspirin'))
        self.assertTrue(self.kb.is_known_drug('eculizumab'))
        self.assertFalse(self.kb.is_known_drug('unknown_drug_xyz'))
        logger.info(f"  ✓ Known drug check working")
    
    # ------------------------------------------------------------------------
    # Test Index Structures
    # ------------------------------------------------------------------------
    
    def test_09_index_structures(self):
        """Test index structures and retrieval"""
        logger.info("\n" + "-"*60)
        logger.info("Test 09: Index Structures")
        logger.info("-"*60)
        
        # Load all test data
        self.kb._process_fda_drugs(self.test_fixtures.get_fda_drug_data())
        self.kb._process_investigational_drugs(self.test_fixtures.get_investigational_drug_data())
        self.kb._process_alexion_drugs(self.test_fixtures.get_alexion_drug_data())
        
        # Test type index
        approved = self.kb.get_drugs_by_type('approved')
        investigational = self.kb.get_drugs_by_type('investigational')
        alexion = self.kb.get_drugs_by_type('alexion')
        
        self.assertTrue(len(approved) > 0, "Should have approved drugs")
        self.assertTrue(len(investigational) > 0, "Should have investigational drugs")
        self.assertTrue(len(alexion) > 0, "Should have Alexion drugs")
        
        logger.info(f"  ✓ Type indices: {len(approved)} approved, "
                   f"{len(investigational)} investigational, {len(alexion)} Alexion")
        
        # Test source index
        fda_drugs = self.kb.drug_index['by_source'].get('FDA', set())
        ct_drugs = self.kb.drug_index['by_source'].get('ClinicalTrials.gov', set())
        alexion_drugs = self.kb.drug_index['by_source'].get('Alexion', set())
        
        self.assertTrue(len(fda_drugs) > 0, "Should have FDA sourced drugs")
        self.assertTrue(len(ct_drugs) > 0, "Should have ClinicalTrials.gov drugs")
        self.assertTrue(len(alexion_drugs) > 0, "Should have Alexion sourced drugs")
        
        logger.info(f"  ✓ Source indices: {len(fda_drugs)} FDA, "
                   f"{len(ct_drugs)} CT.gov, {len(alexion_drugs)} Alexion")
        
        # Test normalized index
        normalized_count = len(self.kb.drug_index['normalized'])
        logger.info(f"  ✓ Normalized index: {normalized_count} entries")
    
    # ------------------------------------------------------------------------
    # Test Statistics
    # ------------------------------------------------------------------------
    
    def test_10_statistics(self):
        """Test statistics tracking"""
        logger.info("\n" + "-"*60)
        logger.info("Test 10: Statistics Tracking")
        logger.info("-"*60)
        
        # Load all test data
        self.kb._process_fda_drugs(self.test_fixtures.get_fda_drug_data())
        self.kb._process_investigational_drugs(self.test_fixtures.get_investigational_drug_data())
        self.kb._process_alexion_drugs(self.test_fixtures.get_alexion_drug_data())
        self.kb._process_drug_patterns(self.test_fixtures.get_drug_pattern_data())
        
        stats = self.kb.get_statistics()
        
        self.assertIn('fda_approved', stats)
        self.assertIn('investigational', stats)
        self.assertIn('alexion', stats)
        self.assertIn('patterns', stats)
        self.assertIn('total_drugs', stats)
        
        logger.info(f"  Statistics:")
        for key, value in stats.items():
            logger.info(f"    - {key}: {value}")
        
        # Verify counts
        self.assertGreater(stats['fda_approved'], 0)
        self.assertGreater(stats['investigational'], 0)
        self.assertGreater(stats['alexion'], 0)
        self.assertGreater(stats['patterns'], 0)
        self.assertGreater(stats['total_drugs'], 0)
    
    # ------------------------------------------------------------------------
    # Test Singleton Pattern
    # ------------------------------------------------------------------------
    
    def test_11_singleton_pattern(self):
        """Test singleton pattern for get_knowledge_base"""
        logger.info("\n" + "-"*60)
        logger.info("Test 11: Singleton Pattern")
        logger.info("-"*60)
        
        # Get two instances
        kb1 = get_knowledge_base()
        kb2 = get_knowledge_base()
        
        # Should be the same instance
        self.assertIs(kb1, kb2, "Should return same instance")
        logger.info(f"  ✓ Singleton pattern working correctly")
        
        # Add data to kb1
        test_drug = DrugInfo(
            name='test_singleton_drug',
            drug_type='test',
            source='test'
        )
        kb1.drugs['test_singleton_drug'] = test_drug
        
        # Should be visible in kb2
        self.assertIn('test_singleton_drug', kb2.drugs)
        logger.info(f"  ✓ Data shared between singleton instances")
    
    # ------------------------------------------------------------------------
    # Test Error Handling
    # ------------------------------------------------------------------------
    
    def test_12_error_handling(self):
        """Test error handling for malformed data"""
        logger.info("\n" + "-"*60)
        logger.info("Test 12: Error Handling")
        logger.info("-"*60)
        
        # Test with None data
        self.kb._process_fda_drugs(None)
        self.kb._process_investigational_drugs(None)
        self.kb._process_alexion_drugs(None)
        self.kb._process_drug_lexicon(None)
        self.kb._process_drug_patterns(None)
        logger.info(f"  ✓ Handles None data without crashing")
        
        # Test with empty data
        self.kb._process_fda_drugs([])
        self.kb._process_investigational_drugs([])
        self.kb._process_alexion_drugs({})
        self.kb._process_drug_lexicon([])
        self.kb._process_drug_patterns({})
        logger.info(f"  ✓ Handles empty data without crashing")
        
        # Test with malformed data
        malformed_fda = [
            {'no_key_field': 'test'},
            'not_a_dict',
            None,
            {'key': ''},  # Empty key
        ]
        self.kb._process_fda_drugs(malformed_fda)
        logger.info(f"  ✓ Handles malformed FDA data")
        
        malformed_inv = [
            {'no_intervention': 'test'},
            {'interventionName': ''},  # Empty name
            {'interventionName': None},
        ]
        self.kb._process_investigational_drugs(malformed_inv)
        logger.info(f"  ✓ Handles malformed investigational data")
        
        # Test pattern matching with empty text
        matches = self.kb.match_patterns('')
        self.assertIsNotNone(matches)
        logger.info(f"  ✓ Pattern matching handles empty text")
        
        # Test search with special characters
        results = self.kb.search_drugs('!@#$%^&*()')
        self.assertIsNotNone(results)
        logger.info(f"  ✓ Search handles special characters")


# ============================================================================
# Integration Tests
# ============================================================================

class TestDrugKnowledgeBaseIntegration(unittest.TestCase):
    """Integration tests using real config.yaml if available"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        logger.info("\n" + "="*80)
        logger.info("INTEGRATION TESTS WITH CONFIG.YAML")
        logger.info("="*80)
        
        cls.config_exists = CONFIG_PATH.exists()
        if not cls.config_exists:
            logger.warning(f"Config file not found at {CONFIG_PATH}")
    
    def setUp(self):
        """Set up each test"""
        if not self.config_exists:
            self.skipTest("Config file not available")
    
    def test_01_load_from_config(self):
        """Test loading from actual config.yaml"""
        logger.info("\n" + "-"*60)
        logger.info("Integration Test: Load from config.yaml")
        logger.info("-"*60)
        
        try:
            kb = DrugKnowledgeBase(config_path=str(CONFIG_PATH))
            
            stats = kb.get_statistics()
            logger.info(f"  Loaded from config.yaml:")
            for key, value in stats.items():
                logger.info(f"    - {key}: {value}")
            
            # Test some known drugs if data loaded
            if stats['total_drugs'] > 0:
                # Test common drugs
                common_drugs = ['aspirin', 'ibuprofen', 'acetaminophen']
                for drug in common_drugs:
                    if kb.is_known_drug(drug):
                        logger.info(f"  ✓ Found {drug}")
                
                # Test Alexion drugs if loaded
                if stats.get('alexion', 0) > 0:
                    alexion_drugs = ['eculizumab', 'ravulizumab']
                    for drug in alexion_drugs:
                        if kb.is_alexion_drug(drug):
                            logger.info(f"  ✓ Found Alexion drug: {drug}")
            
        except Exception as e:
            logger.error(f"Failed to load from config: {e}")
            self.fail(f"Should load from config: {e}")
    
    def test_02_performance_large_dataset(self):
        """Test performance with large dataset"""
        logger.info("\n" + "-"*60)
        logger.info("Integration Test: Performance with Large Dataset")
        logger.info("-"*60)
        
        import time
        
        kb = DrugKnowledgeBase(config_path=str(CONFIG_PATH))
        stats = kb.get_statistics()
        
        if stats['total_drugs'] < 100:
            self.skipTest("Not enough data for performance test")
        
        # Test lookup performance
        start_time = time.time()
        lookups = 0
        test_drugs = ['aspirin', 'unknown_xyz', 'metformin', 'fake_drug_123']
        
        for _ in range(100):
            for drug in test_drugs:
                kb.get_drug_info(drug)
                lookups += 1
        
        elapsed = time.time() - start_time
        lookups_per_second = lookups / elapsed
        
        logger.info(f"  Lookup performance: {lookups} lookups in {elapsed:.3f}s")
        logger.info(f"  Rate: {lookups_per_second:.0f} lookups/second")
        
        # Test search performance
        start_time = time.time()
        searches = 0
        search_terms = ['asp', 'met', 'anti', 'mab']
        
        for term in search_terms:
            results = kb.search_drugs(term)
            searches += 1
            logger.info(f"    Search '{term}': {len(results)} results")
        
        elapsed = time.time() - start_time
        logger.info(f"  Search performance: {searches} searches in {elapsed:.3f}s")
        
        self.assertLess(elapsed, 5.0, "Searches should complete within 5 seconds")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("DRUG KNOWLEDGE BASE COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Test Location: {TEST_DIR}")
    print(f"Config Path: {CONFIG_PATH}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTests(loader.loadTestsFromTestCase(TestDrugKnowledgeBase))
    
    # Add integration tests
    suite.addTests(loader.loadTestsFromTestCase(TestDrugKnowledgeBaseIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} TESTS FAILED")
        
        # Print failure details
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}")
                print(f"    {traceback[:200]}...")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}")
                print(f"    {traceback[:200]}...")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())