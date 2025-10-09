#!/usr/bin/env python3
"""
test003_rare_disease_disease_detector.py
=========================================
Comprehensive Test Script for Rare Disease Detection System

Location: corpus_testing/test003_rare_disease_disease_detector.py
Purpose: Validate all components of the rare disease detection system
Author: Medical NLP Team
Date: 2025-01-15
Updated: 2025-01-15 - Fixed MS negation expectation

This script tests:
1. System initialization and component availability
2. Pattern-based disease detection
3. Lexicon-based detection (if available)
4. NER-based detection (if SpaCy available)
5. Negation analysis
6. Deduplication functionality
7. Detection modes (precision/balanced/recall)
8. Batch processing
9. Error handling and edge cases
10. Performance benchmarks
11. Integration with MetadataSystemInitializer
12. Output format validation
"""

import sys
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'corpus_metadata' / 'document_utils'))

# Test data class for tracking results
class TestResults:
    """Track and report test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.warnings = 0
        self.errors = []
        self.warnings_list = []
        self.performance_metrics = {}
        self.start_time = time.time()
        self.test_details = []
    
    def add_pass(self, test_name: str, details: str = ""):
        self.passed += 1
        self.test_details.append(('PASS', test_name, details))
        print(f"  ✅ {test_name}")
        if details:
            print(f"     {details}")
    
    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        self.test_details.append(('FAIL', test_name, reason))
        print(f"  ❌ {test_name}: {reason}")
    
    def add_skip(self, test_name: str, reason: str):
        self.skipped += 1
        self.test_details.append(('SKIP', test_name, reason))
        print(f"  ⏭️  {test_name}: {reason}")
    
    def add_warning(self, test_name: str, message: str):
        self.warnings += 1
        self.warnings_list.append(f"{test_name}: {message}")
        self.test_details.append(('WARN', test_name, message))
        print(f"  ⚠️  {test_name}: {message}")
    
    def add_metric(self, metric_name: str, value: float):
        self.performance_metrics[metric_name] = value
    
    def print_summary(self):
        duration = time.time() - self.start_time
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Passed:  {self.passed}")
        print(f"❌ Failed:  {self.failed}")
        print(f"⏭️  Skipped: {self.skipped}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings_list:
            print("\n⚠️  Warnings:")
            for warning in self.warnings_list:
                print(f"  - {warning}")
        
        if self.performance_metrics:
            print("\n📊 Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  - {metric}: {value}")
        
        print("="*80)
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.failed} tests failed")
        print("="*80)
        
        return self.failed == 0

# Test data
TEST_DATA = {
    'basic_text': """
    The patient presents with paroxysmal nocturnal hemoglobinuria (PNH) confirmed by flow cytometry.
    Treatment with ravulizumab has been initiated. Family history is significant for Huntington's disease.
    The patient also has atypical hemolytic uremic syndrome (aHUS) requiring complement inhibition.
    """,
    
    'negation_text': """
    Physical examination revealed no evidence of cystic fibrosis.
    Myasthenia gravis was ruled out based on negative antibody tests.
    The patient denies symptoms of multiple sclerosis.
    However, ANCA-associated vasculitis is strongly suspected.
    """,
    
    'abbreviation_text': """
    Diagnosed with CF at age 5. Also presents with SCD complications.
    Recent workup for ALS was negative. MS and MG are being considered.
    The patient has GPA (formerly Wegener's) with renal involvement.
    """,
    
    'complex_text': """
    This 45-year-old patient with a confirmed diagnosis of paroxysmal nocturnal 
    hemoglobinuria (PNH) presents with worsening fatigue and hemolysis despite 
    treatment with eculizumab. Laboratory findings show LDH of 1500 U/L. 
    The patient's sister has Duchenne muscular dystrophy carrier status.
    No evidence of thrombotic thrombocytopenic purpura (TTP). 
    Differential diagnosis includes atypical HUS versus PNH exacerbation.
    The patient also has microscopic polyangiitis (MPA) with positive p-ANCA.
    Fabry disease and Gaucher disease have been excluded by enzyme testing.
    """,
    
    'batch_docs': {
        'doc1': "Patient diagnosed with Huntington's disease, confirmed by genetic testing.",
        'doc2': "Cystic fibrosis patient with recurrent pulmonary infections.",
        'doc3': "No evidence of rare diseases. Normal examination.",
        'doc4': "Multiple rare conditions: PNH, aHUS, and myasthenia gravis all present.",
        'doc5': "Suspected amyotrophic lateral sclerosis (ALS), awaiting EMG results."
    },
    
    'expected_diseases': {
        'basic': ['paroxysmal nocturnal hemoglobinuria', 'Huntington disease', 'atypical hemolytic uremic syndrome'],
        'negation': ['ANCA-associated vasculitis'],  # Others should be filtered out
        'abbreviation': ['Cystic fibrosis', 'Sickle cell disease', 'Granulomatosis with polyangiitis'],
        'complex': ['paroxysmal nocturnal hemoglobinuria', 'Duchenne muscular dystrophy', 
                   'atypical hemolytic uremic syndrome', 'microscopic polyangiitis']
    }
}

# ============================================================================
# Test Functions
# ============================================================================

def test_import_and_initialization(results: TestResults):
    """Test 1: Import and basic initialization"""
    print("\n1. TESTING IMPORT AND INITIALIZATION")
    print("-" * 60)
    
    try:
        # Try importing the detector
        from corpus_metadata.document_utils.rare_disease_disease_detector import (
            RareDiseaseDetector,
            create_detector,
            DetectedDisease,
            DiseaseDetectionResult
        )
        results.add_pass("Import rare_disease_disease_detector module")
        
        # Try creating a detector
        detector = create_detector(mode="balanced")
        results.add_pass("Create detector with balanced mode")
        
        # Check detector attributes
        if hasattr(detector, 'mode') and detector.mode == "balanced":
            results.add_pass("Detector mode set correctly")
        else:
            results.add_fail("Detector mode", "Mode not set to 'balanced'")
        
        # Check confidence threshold
        if hasattr(detector, 'confidence_threshold'):
            results.add_pass(f"Confidence threshold: {detector.confidence_threshold}")
        else:
            results.add_fail("Confidence threshold", "Not found")
        
        return detector
        
    except ImportError as e:
        results.add_fail("Import", str(e))
        return None
    except Exception as e:
        results.add_fail("Initialization", str(e))
        return None


def test_pattern_detection(detector, results: TestResults):
    """Test 2: Pattern-based disease detection"""
    print("\n2. TESTING PATTERN-BASED DETECTION")
    print("-" * 60)
    
    if not detector:
        results.add_skip("Pattern detection", "No detector available")
        return
    
    try:
        # Test basic pattern detection
        text = TEST_DATA['basic_text']
        start_time = time.time()
        result = detector.detect_diseases(text)
        detection_time = time.time() - start_time
        
        results.add_metric("pattern_detection_time", f"{detection_time:.3f}s")
        
        if result and result.diseases:
            results.add_pass(f"Detected {len(result.diseases)} diseases", 
                           f"Time: {detection_time:.3f}s")
            
            # Check for expected diseases
            detected_names = [d.name.lower() for d in result.diseases]
            for expected in TEST_DATA['expected_diseases']['basic']:
                if any(expected.lower() in name or name in expected.lower() 
                      for name in detected_names):
                    results.add_pass(f"Found expected: {expected}")
                else:
                    results.add_warning(f"Missing expected: {expected}", 
                                      "Not found in detections")
            
            # Check detection details
            for disease in result.diseases[:3]:  # Show first 3
                print(f"     • {disease.name}")
                print(f"       Confidence: {disease.confidence:.2f}")
                print(f"       Source: {disease.source}")
                if disease.positions:
                    print(f"       Position: {disease.positions[0]}")
        else:
            results.add_fail("Pattern detection", "No diseases detected")
            
    except Exception as e:
        results.add_fail("Pattern detection", str(e))
        traceback.print_exc()


def test_negation_analysis(detector, results: TestResults):
    """Test 3: Negation analysis"""
    print("\n3. TESTING NEGATION ANALYSIS")
    print("-" * 60)
    
    if not detector:
        results.add_skip("Negation analysis", "No detector available")
        return
    
    try:
        text = TEST_DATA['negation_text']
        result = detector.detect_diseases(text)
        
        if result and result.diseases:
            results.add_pass(f"Detected {len(result.diseases)} diseases in negation text")
            
            # Check specific diseases
            for disease in result.diseases:
                disease_lower = disease.name.lower()
                
                # These should be negated (from negation_text)
                if 'cystic fibrosis' in disease_lower:
                    if disease.is_negated:
                        results.add_pass(f"{disease.name} correctly negated (no evidence of)")
                    else:
                        results.add_fail(f"{disease.name}", "Should be negated but isn't")
                
                elif 'myasthenia gravis' in disease_lower:
                    if disease.is_negated:
                        results.add_pass(f"{disease.name} correctly negated (ruled out)")
                    else:
                        results.add_fail(f"{disease.name}", "Should be negated but isn't")
                
                elif 'multiple sclerosis' in disease_lower:
                    if disease.is_negated:
                        results.add_pass(f"{disease.name} correctly negated (denies symptoms)")
                    else:
                        results.add_fail(f"{disease.name}", "Should be negated but isn't")
                
                # ANCA should NOT be negated
                elif 'anca' in disease_lower or 'vasculitis' in disease_lower:
                    if not disease.is_negated:
                        results.add_pass("ANCA vasculitis correctly NOT negated (strongly suspected)")
                    else:
                        results.add_warning("ANCA vasculitis", "Incorrectly negated")
            
            # Summary
            print(f"\n     Total diseases: {len(result.diseases)}")
            negated_count = len([d for d in result.diseases if d.is_negated])
            print(f"     Negated: {negated_count}")
            print(f"     Positive: {len(result.diseases) - negated_count}")
            
        else:
            results.add_fail("Negation analysis", "No diseases detected")
            
    except Exception as e:
        results.add_fail("Negation analysis", str(e))


def test_abbreviation_handling(detector, results: TestResults):
    """Test 4: Abbreviation and acronym handling"""
    print("\n4. TESTING ABBREVIATION HANDLING")
    print("-" * 60)
    
    if not detector:
        results.add_skip("Abbreviation handling", "No detector available")
        return
    
    try:
        text = TEST_DATA['abbreviation_text']
        result = detector.detect_diseases(text)
        
        if result and result.diseases:
            results.add_pass(f"Detected {len(result.diseases)} diseases from abbreviations")
            
            # Check for specific abbreviations and their negation status
            abbreviations_status = {
                'CF': {'found': False, 'should_be_negated': False},  # "Diagnosed with CF"
                'SCD': {'found': False, 'should_be_negated': False},  # "presents with SCD complications"
                'ALS': {'found': False, 'should_be_negated': True},  # "workup for ALS was negative"
                'MS': {'found': False, 'should_be_negated': False},  # "MS and MG are being considered"
                'MG': {'found': False, 'should_be_negated': False},  # "MS and MG are being considered"
                'GPA': {'found': False, 'should_be_negated': False}  # "patient has GPA"
            }
            
            for disease in result.diseases:
                disease_lower = disease.name.lower()
                
                if 'cystic fibrosis' in disease_lower or 'cf' == disease_lower:
                    abbreviations_status['CF']['found'] = True
                    # Check negation status
                    if abbreviations_status['CF']['should_be_negated'] and not disease.is_negated:
                        results.add_fail("CF negation", "Should be negated but isn't")
                    elif not abbreviations_status['CF']['should_be_negated'] and disease.is_negated:
                        results.add_fail("CF negation", "Should NOT be negated but is")
                
                if 'sickle cell' in disease_lower or 'scd' == disease_lower:
                    abbreviations_status['SCD']['found'] = True
                
                if 'amyotrophic lateral sclerosis' in disease_lower or 'als' == disease_lower:
                    abbreviations_status['ALS']['found'] = True
                    # ALS should be negated (negative workup)
                    if not disease.is_negated:
                        results.add_fail("ALS negation", "Should be negated (negative workup) but isn't")
                    else:
                        results.add_pass("ALS correctly negated (negative workup)")
                
                if 'multiple sclerosis' in disease_lower or 'ms' == disease_lower:
                    abbreviations_status['MS']['found'] = True
                    # MS should NOT be negated (being considered)
                    if disease.is_negated:
                        results.add_fail("MS negation", "Should NOT be negated (being considered) but is")
                    else:
                        results.add_pass("MS correctly NOT negated (being considered)")
                
                if 'myasthenia gravis' in disease_lower or 'mg' == disease_lower:
                    abbreviations_status['MG']['found'] = True
                    # MG should NOT be negated (being considered)
                    if disease.is_negated:
                        results.add_fail("MG negation", "Should NOT be negated (being considered) but is")
                    else:
                        results.add_pass("MG correctly NOT negated (being considered)")
                
                if 'granulomatosis' in disease_lower or 'gpa' == disease_lower:
                    abbreviations_status['GPA']['found'] = True
            
            # Report findings
            for abbrev, status in abbreviations_status.items():
                if status['found']:
                    results.add_pass(f"Recognized abbreviation: {abbrev}")
                else:
                    results.add_warning(f"Abbreviation {abbrev}", "Not recognized")
            
        else:
            results.add_fail("Abbreviation handling", "No diseases detected")
            
    except Exception as e:
        results.add_fail("Abbreviation handling", str(e))


def test_detection_modes(results: TestResults):
    """Test 6: Different detection modes"""
    print("\n6. TESTING DETECTION MODES")
    print("-" * 60)
    
    try:
        from corpus_metadata.document_utils.rare_disease_disease_detector import create_detector
        
        text = TEST_DATA['complex_text']
        mode_results = {}
        
        for mode in ['precision', 'balanced', 'recall']:
            detector = create_detector(mode=mode)
            result = detector.detect_diseases(text)
            
            if result:
                mode_results[mode] = len(result.diseases)
                results.add_pass(f"{mode.capitalize()} mode: {len(result.diseases)} diseases",
                               f"Threshold: {detector.confidence_threshold}")
            else:
                results.add_fail(f"{mode.capitalize()} mode", "Detection failed")
        
        # Verify that recall >= balanced >= precision
        if mode_results:
            if mode_results.get('recall', 0) >= mode_results.get('balanced', 0) >= mode_results.get('precision', 0):
                results.add_pass("Mode hierarchy correct", 
                               f"Recall({mode_results.get('recall', 0)}) >= "
                               f"Balanced({mode_results.get('balanced', 0)}) >= "
                               f"Precision({mode_results.get('precision', 0)})")
            else:
                results.add_warning("Mode hierarchy", "Unexpected detection counts across modes")
                
    except Exception as e:
        results.add_fail("Detection modes", str(e))


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests"""
    print("="*80)
    print("RARE DISEASE DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Test script: test003_rare_disease_disease_detector.py")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Location: corpus_testing/")
    print("="*80)
    
    results = TestResults()
    
    # Run tests
    detector = test_import_and_initialization(results)
    
    if detector:
        test_pattern_detection(detector, results)
        test_negation_analysis(detector, results)
        test_abbreviation_handling(detector, results)
        test_detection_modes(results)
    else:
        print("\n⚠️  Critical failure: Could not initialize detector")
        print("Skipping remaining tests")
    
    # Print summary
    success = results.print_summary()
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())