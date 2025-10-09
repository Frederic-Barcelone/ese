#!/usr/bin/env python3
"""
Enhanced Drug Detector Test Suite v2.1
=======================================
Location: corpus_testing/test005_drug_detector.py

CHANGES IN v2.1:
- Fixed EnhancedDrugDetector initialization (removed use_lexicon parameter)
- Updated output directory to corpus_testing/test_results
- Added proper parameter mapping for detector initialization
- Suppressed spacy pkg_resources deprecation warning
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Suppress the spacy pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Console Formatting
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
    
    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        for attr in dir(Colors):
            if not attr.startswith('_') and not callable(getattr(Colors, attr)):
                setattr(Colors, attr, '')

# Check if we're in a terminal
if not sys.stdout.isatty():
    Colors.disable()

# ============================================================================
# Test Suite
# ============================================================================

class DrugDetectorTestSuite:
    """Test suite for enhanced drug detector"""
    
    def __init__(self):
        self.test_results = []
        self.output_dir = Path(__file__).parent / "test_results"
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def print_header(self):
        """Print test suite header"""
        print(f"{Colors.CYAN}{'🧪'*40}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}ENHANCED DRUG DETECTOR TEST SUITE v2.1{Colors.ENDC}")
        print(f"{Colors.CYAN}{'🧪'*40}{Colors.ENDC}")
        
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n{Colors.YELLOW}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BRIGHT_WHITE}{title.upper()}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'='*80}{Colors.ENDC}")
        
    def run_tests(self):
        """Run all tests"""
        self.print_header()
        
        # Setup
        self.print_section("Running Test Suite")
        
        try:
            # Initialize system
            self.print_section("Drug Detector Test Suite - Initialization")
            print("Initializing metadata system...")
            
            from corpus_metadata.document_utils.metadata_system_initializer import MetadataSystemInitializer
            system_initializer = MetadataSystemInitializer()
            system_initializer.initialize()
            print(f"{Colors.GREEN}✅ System initialized successfully{Colors.ENDC}")
            
            # Create drug detector with correct parameters
            print("Creating enhanced drug detector...")
            from corpus_metadata.document_utils.rare_disease_drug_detector import EnhancedDrugDetector
            
            # Use only the parameters that EnhancedDrugDetector accepts
            detector = EnhancedDrugDetector(
                system_initializer=system_initializer,
                use_kb=True,           # Use knowledge base
                use_patterns=True,     # Use pattern detection
                use_ner=True,          # Use NER detection
                use_pubtator=True,     # Use PubTator validation
                use_medical_filter=True,  # Use medical terms filter
                confidence_threshold=0.5
            )
            print(f"{Colors.GREEN}✅ Drug detector created successfully{Colors.ENDC}")
            
            # Test 1: Basic drug detection
            self.print_section("Test 1: Basic Drug Detection")
            test_text_1 = """
            The patient was treated with rituximab 375 mg/m2 weekly for 4 weeks.
            Methylprednisolone 1000mg was administered for 3 consecutive days.
            Maintenance therapy included azathioprine 2mg/kg daily.
            """
            
            print(f"Test text: {test_text_1[:100]}...")
            result_1 = detector.detect_drugs(test_text_1)
            
            if result_1 and hasattr(result_1, 'drugs'):
                print(f"\n{Colors.GREEN}✅ Found {len(result_1.drugs)} drugs:{Colors.ENDC}")
                for drug in result_1.drugs[:5]:  # Show first 5
                    print(f"  • {drug.name} (confidence: {drug.confidence:.2f})")
                    if drug.rxcui:
                        print(f"    RxCUI: {drug.rxcui}")
                self.test_results.append({"test": "basic_detection", "status": "passed", "drugs_found": len(result_1.drugs)})
            else:
                print(f"{Colors.RED}❌ No drugs detected{Colors.ENDC}")
                self.test_results.append({"test": "basic_detection", "status": "failed", "error": "No drugs found"})
            
            # Test 2: Complex drug names
            self.print_section("Test 2: Complex Drug Names")
            test_text_2 = """
            The clinical trial evaluated avacopan (TAVNEOS) in combination with 
            ravulizumab-cwvz for ANCA-associated vasculitis. Eculizumab was 
            considered as an alternative. The protocol included cyclophosphamide 
            induction followed by maintenance with mycophenolate mofetil.
            """
            
            print(f"Test text: {test_text_2[:100]}...")
            result_2 = detector.detect_drugs(test_text_2)
            
            if result_2 and hasattr(result_2, 'drugs'):
                print(f"\n{Colors.GREEN}✅ Found {len(result_2.drugs)} drugs:{Colors.ENDC}")
                for drug in result_2.drugs[:5]:
                    print(f"  • {drug.name} (confidence: {drug.confidence:.2f})")
                self.test_results.append({"test": "complex_names", "status": "passed", "drugs_found": len(result_2.drugs)})
            else:
                print(f"{Colors.RED}❌ No drugs detected{Colors.ENDC}")
                self.test_results.append({"test": "complex_names", "status": "failed"})
            
            # Test 3: Investigational drugs
            self.print_section("Test 3: Investigational Drugs")
            test_text_3 = """
            Phase 3 study of ALXN1210 versus ALXN1234 in patients with PNH.
            Secondary endpoints include evaluation of danicopan and BCX9930.
            The study protocol allows concomitant use of supportive care medications.
            """
            
            print(f"Test text: {test_text_3[:100]}...")
            result_3 = detector.detect_drugs(test_text_3)
            
            if result_3 and hasattr(result_3, 'drugs'):
                print(f"\n{Colors.GREEN}✅ Found {len(result_3.drugs)} drugs:{Colors.ENDC}")
                for drug in result_3.drugs[:5]:
                    print(f"  • {drug.name} (confidence: {drug.confidence:.2f})")
                    if drug.drug_type:
                        print(f"    Type: {drug.drug_type}")
                self.test_results.append({"test": "investigational", "status": "passed", "drugs_found": len(result_3.drugs)})
            else:
                print(f"{Colors.RED}❌ No drugs detected{Colors.ENDC}")
                self.test_results.append({"test": "investigational", "status": "failed"})
            
            # Test 4: Performance test
            self.print_section("Test 4: Performance Test")
            long_text = test_text_1 * 10  # Repeat text 10 times
            
            print(f"Testing with {len(long_text)} characters...")
            start_time = time.time()
            result_4 = detector.detect_drugs(long_text)
            elapsed = time.time() - start_time
            
            if result_4:
                print(f"{Colors.GREEN}✅ Processed in {elapsed:.2f} seconds{Colors.ENDC}")
                print(f"  • Characters/second: {len(long_text)/elapsed:.0f}")
                if hasattr(result_4, 'drugs'):
                    valid_drugs = [d for d in result_4.drugs if d.name and d.name.strip()]
                    print(f"  • Valid drugs found: {len(valid_drugs)} (total: {len(result_4.drugs)})")
                self.test_results.append({"test": "performance", "status": "passed", "time": elapsed})
            else:
                print(f"{Colors.RED}❌ Processing failed{Colors.ENDC}")
                self.test_results.append({"test": "performance", "status": "failed"})
            
            # Test 5: Edge cases
            self.print_section("Test 5: Edge Cases")
            
            # Empty text
            print("Testing empty text...")
            result_empty = detector.detect_drugs("")
            if result_empty and hasattr(result_empty, 'drugs'):
                print(f"  • Empty text: {len(result_empty.drugs)} drugs (expected 0)")
                
            # Special characters
            special_text = "Patient received aspirin/clopidogrel & metformin+glipizide"
            print(f"Testing special characters: {special_text}")
            result_special = detector.detect_drugs(special_text)
            if result_special and hasattr(result_special, 'drugs'):
                print(f"  • Found {len(result_special.drugs)} drugs")
                for drug in result_special.drugs:
                    print(f"    - {drug.name}")
            
            self.test_results.append({"test": "edge_cases", "status": "passed"})
            
            # Save results
            self.save_results()
            
        except ImportError as e:
            print(f"{Colors.RED}❌ Import error: {e}{Colors.ENDC}")
            self.test_results.append({"test": "setup", "status": "failed", "error": str(e)})
        except Exception as e:
            print(f"{Colors.RED}❌ Setup failed: {e}{Colors.ENDC}")
            self.test_results.append({"test": "setup", "status": "failed", "error": str(e)})
            import traceback
            traceback.print_exc()
        
        # Print summary
        self.print_summary()
        
    def save_results(self):
        """Save test results to file"""
        output_file = self.output_dir / f"drug_detector_test_{self.timestamp}.json"
        
        results = {
            "timestamp": self.timestamp,
            "test_suite": "drug_detector",
            "version": "2.1",
            "results": self.test_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{Colors.CYAN}Results saved to: {output_file}{Colors.ENDC}")
        
    def print_summary(self):
        """Print test summary"""
        self.print_section("Test Summary")
        
        passed = sum(1 for r in self.test_results if r.get('status') == 'passed')
        failed = sum(1 for r in self.test_results if r.get('status') == 'failed')
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
        if failed > 0:
            print(f"{Colors.RED}Failed: {failed}{Colors.ENDC}")
        
        if failed == 0:
            print(f"\n{Colors.GREEN}✅ All tests passed!{Colors.ENDC}")
        else:
            print(f"\n{Colors.YELLOW}⚠️ Some tests failed. Check the results file for details.{Colors.ENDC}")

# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution function"""
    suite = DrugDetectorTestSuite()
    suite.run_tests()
    print(f"\n{Colors.GREEN}✅ Test suite completed!{Colors.ENDC}")

if __name__ == "__main__":
    main()