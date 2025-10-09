#!/usr/bin/env python3
"""
corpus_testing/test001_rare_disease_pubtator3.py
==================================
Comprehensive test script for PubTator3 normalization functionality
Location: 17_Corpus/corpus_testing/test001_rare_disease_pubtator3.py

This script tests:
1. PubTator3Manager initialization
2. Single drug normalization
3. Batch drug normalization (the fixed normalize_drugs method)
4. Disease normalization
5. Cache functionality
6. Error handling
7. Integration with drug detector
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'corpus_metadata' / 'document_utils'))

# Import the fixed PubTator3Manager
from corpus_metadata.document_utils.rare_disease_pubtator3 import (
    PubTator3Manager,
    NormalizedEntity,
    setup_pubtator3
)

# Try importing the drug detector to test integration
try:
    from corpus_metadata.document_utils.rare_disease_drug_detector import EnhancedDrugDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    print("⚠️  Drug detector not available for integration testing")

# Test configuration
TEST_CONFIG = {
    'test_drugs': [
        'ravulizumab',
        'Ultomiris',
        'eculizumab',
        'Soliris',
        'ALXN1210',
        'rituximab',
        'cyclophosphamide',
        'prednisone',
        'mycophenolate mofetil',
        'avacopan',
        'invalid_drug_xyz123',  # Should return None
        'another_fake_drug'     # Should return None
    ],
    'test_diseases': [
        'PNH',
        'paroxysmal nocturnal hemoglobinuria',
        'aHUS',
        'atypical hemolytic uremic syndrome',
        'myasthenia gravis',
        'ANCA-associated vasculitis',
        'lupus nephritis',
        'fake_disease_xyz'  # Should return None
    ],
    'sample_text': """
    Ravulizumab (Ultomiris) is a long-acting C5 complement inhibitor approved for 
    treatment of paroxysmal nocturnal hemoglobinuria (PNH) and atypical hemolytic 
    uremic syndrome (aHUS). Unlike eculizumab (Soliris), ravulizumab requires less 
    frequent dosing. Patients with ANCA-associated vasculitis may benefit from 
    rituximab combined with cyclophosphamide and prednisone therapy.
    """
}

class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.start_time = time.time()
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"  ❌ {test_name}: {reason}")
    
    def add_skip(self, test_name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {test_name}: {reason}")
    
    def print_summary(self):
        duration = time.time() - self.start_time
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed:  {self.passed}")
        print(f"❌ Failed:  {self.failed}")
        print(f"⏭️  Skipped: {self.skipped}")
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        print("="*60)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.failed} tests failed")
        print("="*60)


def test_initialization(results: TestResults) -> Optional[PubTator3Manager]:
    """Test 1: Initialize PubTator3Manager"""
    print("\n1. TESTING INITIALIZATION")
    print("-" * 40)
    
    try:
        manager = setup_pubtator3()
        
        if manager.enabled:
            results.add_pass("PubTator3Manager initialized")
        else:
            results.add_skip("PubTator3Manager initialized but disabled", "Check configuration")
            return None
        
        # Test connection
        if manager.test_connection():
            results.add_pass("API connection verified")
        else:
            results.add_fail("API connection", "Could not connect to PubTator3 API")
        
        return manager
        
    except Exception as e:
        results.add_fail("Initialization", str(e))
        return None


def test_single_drug_normalization(manager: PubTator3Manager, results: TestResults):
    """Test 2: Single drug normalization"""
    print("\n2. TESTING SINGLE DRUG NORMALIZATION")
    print("-" * 40)
    
    test_cases = [
        ('ravulizumab', True, 'Should normalize ravulizumab'),
        ('Ultomiris', True, 'Should normalize brand name Ultomiris'),
        ('invalid_drug_xyz', False, 'Should return None for invalid drug'),
    ]
    
    for drug, should_normalize, description in test_cases:
        try:
            entity = manager.normalize_drug(drug)
            
            if should_normalize:
                if entity:
                    results.add_pass(f"{drug} → {entity.normalized_name}")
                    print(f"     Confidence: {entity.confidence}")
                    if entity.mesh_id:
                        print(f"     MeSH ID: {entity.mesh_id}")
                else:
                    results.add_fail(drug, f"Expected normalization but got None")
            else:
                if entity is None:
                    results.add_pass(f"{drug} correctly returned None")
                else:
                    results.add_fail(drug, f"Expected None but got {entity.normalized_name}")
                    
        except Exception as e:
            results.add_fail(drug, str(e))


def test_batch_drug_normalization(manager: PubTator3Manager, results: TestResults):
    """Test 3: Batch drug normalization (the fixed method)"""
    print("\n3. TESTING BATCH DRUG NORMALIZATION (normalize_drugs)")
    print("-" * 40)
    
    try:
        # Test the new normalize_drugs method
        drugs_to_test = TEST_CONFIG['test_drugs'][:6]  # Test first 6 drugs
        
        print(f"  Testing batch normalization of {len(drugs_to_test)} drugs...")
        batch_results = manager.normalize_drugs(drugs_to_test)
        
        # Verify we got results for each input
        if len(batch_results) != len(drugs_to_test):
            results.add_fail("Batch size", f"Expected {len(drugs_to_test)} results, got {len(batch_results)}")
        else:
            results.add_pass(f"Correct number of results returned ({len(batch_results)})")
        
        # Check individual results
        normalized_count = 0
        for drug, result in zip(drugs_to_test, batch_results):
            if result:
                normalized_count += 1
                print(f"     ✓ {drug} → {result.get('normalized_name', 'unknown')}")
            else:
                print(f"     - {drug} → None")
        
        results.add_pass(f"Normalized {normalized_count}/{len(drugs_to_test)} drugs")
        
        # Test empty list
        empty_results = manager.normalize_drugs([])
        if empty_results == []:
            results.add_pass("Empty list handled correctly")
        else:
            results.add_fail("Empty list", f"Expected empty list, got {empty_results}")
            
    except AttributeError as e:
        if "'normalize_drugs'" in str(e):
            results.add_fail("normalize_drugs method", "Method not found - fix not applied!")
        else:
            results.add_fail("Batch normalization", str(e))
    except Exception as e:
        results.add_fail("Batch normalization", str(e))


def test_disease_normalization(manager: PubTator3Manager, results: TestResults):
    """Test 4: Disease normalization"""
    print("\n4. TESTING DISEASE NORMALIZATION")
    print("-" * 40)
    
    diseases_to_test = ['PNH', 'aHUS', 'myasthenia gravis']
    
    for disease in diseases_to_test:
        try:
            entity = manager.normalize_disease(disease)
            
            if entity:
                results.add_pass(f"{disease} → {entity.normalized_name}")
                print(f"     Confidence: {entity.confidence}")
            else:
                results.add_fail(disease, "Failed to normalize")
                
        except Exception as e:
            results.add_fail(disease, str(e))


def test_cache_functionality(manager: PubTator3Manager, results: TestResults):
    """Test 5: Cache functionality"""
    print("\n5. TESTING CACHE FUNCTIONALITY")
    print("-" * 40)
    
    try:
        # Get initial cache stats
        initial_stats = manager.get_statistics()
        initial_cache_hits = initial_stats.get('cache_hits', 0)
        
        # Normalize a drug (should hit API)
        test_drug = 'methylprednisolone'
        entity1 = manager.normalize_drug(test_drug)
        
        # Get stats after first call
        stats1 = manager.get_statistics()
        api_calls1 = stats1.get('api_calls', 0)
        
        # Normalize same drug again (should hit cache)
        entity2 = manager.normalize_drug(test_drug)
        
        # Get stats after second call
        stats2 = manager.get_statistics()
        cache_hits2 = stats2.get('cache_hits', 0)
        api_calls2 = stats2.get('api_calls', 0)
        
        # Verify cache was used
        if cache_hits2 > initial_cache_hits:
            results.add_pass("Cache hit detected")
        else:
            results.add_fail("Cache", "No cache hit detected")
        
        if api_calls2 == api_calls1:
            results.add_pass("No additional API call for cached item")
        else:
            results.add_fail("Cache", "Unexpected API call for cached item")
        
        # Test cache persistence
        manager._save_cache()
        results.add_pass("Cache saved successfully")
        
    except Exception as e:
        results.add_fail("Cache functionality", str(e))


def test_process_batch_methods(manager: PubTator3Manager, results: TestResults):
    """Test 6: Process batch helper methods"""
    print("\n6. TESTING BATCH PROCESSING METHODS")
    print("-" * 40)
    
    try:
        # Test process_drug_batch
        drug_batch = ['rituximab', 'prednisone', 'fake_drug']
        drug_results = manager.process_drug_batch(drug_batch)
        
        if isinstance(drug_results, dict):
            results.add_pass(f"process_drug_batch returned dict with {len(drug_results)} results")
        else:
            results.add_fail("process_drug_batch", f"Expected dict, got {type(drug_results)}")
        
        # Test process_disease_batch
        disease_batch = ['lupus', 'vasculitis']
        disease_results = manager.process_disease_batch(disease_batch)
        
        if isinstance(disease_results, dict):
            results.add_pass(f"process_disease_batch returned dict with {len(disease_results)} results")
        else:
            results.add_fail("process_disease_batch", f"Expected dict, got {type(disease_results)}")
            
    except Exception as e:
        results.add_fail("Batch processing methods", str(e))


def test_statistics(manager: PubTator3Manager, results: TestResults):
    """Test 7: Statistics tracking"""
    print("\n7. TESTING STATISTICS")
    print("-" * 40)
    
    try:
        stats = manager.get_statistics()
        
        required_keys = ['api_calls', 'cache_hits', 'drugs_normalized', 
                         'diseases_normalized', 'errors', 'cache_size']
        
        for key in required_keys:
            if key in stats:
                results.add_pass(f"Stat '{key}': {stats[key]}")
            else:
                results.add_fail("Statistics", f"Missing key: {key}")
        
        print(f"\n  Full statistics:")
        print(json.dumps(stats, indent=2))
        
    except Exception as e:
        results.add_fail("Statistics", str(e))


def test_drug_detector_integration(results: TestResults):
    """Test 8: Integration with drug detector"""
    print("\n8. TESTING DRUG DETECTOR INTEGRATION")
    print("-" * 40)
    
    if not DETECTOR_AVAILABLE:
        results.add_skip("Drug detector integration", "EnhancedDrugDetector not available")
        return
    
    try:
        # Initialize drug detector with PubTator3
        detector = EnhancedDrugDetector(
            use_pubtator=True,
            use_lexicon=True,
            use_patterns=True
        )
        
        # Process sample text
        text = TEST_CONFIG['sample_text']
        detection_result = detector.detect_drugs(text)
        
        if detection_result and detection_result.drugs:
            results.add_pass(f"Detected {len(detection_result.drugs)} drugs")
            
            # Check if normalization worked
            normalized_drugs = [d for d in detection_result.drugs if d.normalized_name]
            if normalized_drugs:
                results.add_pass(f"Normalized {len(normalized_drugs)} drugs via PubTator3")
                for drug in normalized_drugs[:3]:  # Show first 3
                    print(f"     {drug.name} → {drug.normalized_name}")
            else:
                results.add_fail("Integration", "No drugs were normalized")
        else:
            results.add_fail("Integration", "No drugs detected")
            
    except AttributeError as e:
        if "'normalize_drugs'" in str(e):
            results.add_fail("Integration", "normalize_drugs method missing - fix not applied!")
        else:
            results.add_fail("Integration", str(e))
    except Exception as e:
        results.add_fail("Drug detector integration", str(e))


def main():
    """Run all tests"""
    print("="*60)
    print("PUBTATOR3 NORMALIZATION TEST SUITE")
    print("="*60)
    print(f"Test script: test001_rare_disease_pubtator3.py")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    results = TestResults()
    
    # Test 1: Initialization
    manager = test_initialization(results)
    
    if manager and manager.enabled:
        # Test 2: Single drug normalization
        test_single_drug_normalization(manager, results)
        
        # Test 3: Batch drug normalization (the fixed method)
        test_batch_drug_normalization(manager, results)
        
        # Test 4: Disease normalization
        test_disease_normalization(manager, results)
        
        # Test 5: Cache functionality
        test_cache_functionality(manager, results)
        
        # Test 6: Batch processing methods
        test_process_batch_methods(manager, results)
        
        # Test 7: Statistics
        test_statistics(manager, results)
        
        # Test 8: Drug detector integration
        test_drug_detector_integration(results)
    else:
        print("\n⚠️  Skipping remaining tests - PubTator3 not available")
    
    # Print summary
    results.print_summary()
    
    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit(main())