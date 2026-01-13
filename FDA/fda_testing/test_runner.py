"""
FDA Syncer Test Suite Runner
=============================
Comprehensive test runner that executes all tests and generates detailed reports

Test Suites:
1. FDA Organization Validation (8 tests) - Checks if FDA data structure changed
2. Unit Tests (52 tests) - Tests all system components
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path


class TestRunner:
    """Runs all test suites and generates comprehensive reports"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
    
    def print_header(self):
        """Print test suite header"""
        print("\n" + "="*80)
        print(" "*20 + "FDA SYNCER TEST SUITE")
        print("="*80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version.split()[0]}")
        print("="*80 + "\n")
    
    def run_unit_tests(self):
        """Run offline unit tests"""
        print("\n" + "="*80)
        print("RUNNING: Unit Tests (52 tests)")
        print("="*80)
        
        try:
            # Get the directory where this script is located
            test_dir = Path(__file__).parent
            unit_test_file = test_dir / 'test_offline_unit_tests.py'
            
            result = subprocess.run(
                [sys.executable, str(unit_test_file)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            self.results['unit_tests'] = {
                'returncode': result.returncode,
                'passed': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Parse summary from output
            if 'Tests Run:' in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Tests Run:' in line:
                        print(f"\n{line}")
                    elif 'Successes:' in line:
                        print(line)
                    elif 'Failures:' in line:
                        print(line)
                    elif 'Errors:' in line:
                        print(line)
            
            if result.returncode == 0:
                print("\n=== All unit tests PASSED")
            else:
                print("\n=== Some unit tests FAILED")
                
        except Exception as e:
            print(f"\n=== Error running unit tests: {e}")
            self.results['unit_tests'] = {
                'passed': False,
                'error': str(e)
            }
    
    def run_fda_validation(self):
        """Run FDA organization validation (requires network)"""
        print("\n" + "="*80)
        print("RUNNING: FDA Organization Validation (8 tests)")
        print("Note: Requires network access")
        print("="*80)
        
        # Get the directory where this script is located
        test_dir = Path(__file__).parent
        fda_test_file = test_dir / 'test_fda_org.py'
        
        if not fda_test_file.exists():
            print("\n=== test_fda_org.py not found")
            self.results['fda_validation'] = {
                'passed': None,
                'skipped': True,
                'reason': 'Test file not found'
            }
            return
        
        try:
            result = subprocess.run(
                [sys.executable, str(fda_test_file)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            self.results['fda_validation'] = {
                'returncode': result.returncode,
                'passed': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Show summary
            if result.returncode == 0:
                print("\n=== FDA validation tests PASSED")
            else:
                print("\n=== FDA validation tests FAILED")
                if 'Network' in result.stdout or 'network' in result.stderr.lower():
                    print("  (May be due to network access)")
                print("\nTo see detailed output:")
                print("  python test_fda_org.py")
                
        except subprocess.TimeoutExpired:
            print("\n=== FDA validation tests timed out")
            self.results['fda_validation'] = {
                'passed': False,
                'error': 'Timeout after 120 seconds'
            }
        except Exception as e:
            print(f"\n=== Error running FDA validation: {e}")
            print("  This is normal if you don't have network access")
            self.results['fda_validation'] = {
                'passed': None,
                'skipped': True,
                'reason': str(e)
            }
    
    def print_summary(self):
        """Print comprehensive summary"""
        duration = datetime.now() - self.start_time
        
        print("\n" + "="*80)
        print(" "*25 + "TEST SUMMARY")
        print("="*80)
        print(f"Total Duration: {duration.total_seconds():.2f} seconds")
        print()
        
        # Count results
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for suite_name, suite_result in self.results.items():
            status = "=== PASS" if suite_result.get('passed') else "=== FAIL"
            if suite_result.get('skipped'):
                status = "=== SKIP"
                total_skipped += 1
            elif suite_result.get('passed'):
                total_passed += 1
            else:
                total_failed += 1
            
            print(f"{status} - {suite_name.replace('_', ' ').title()}")
            if 'reason' in suite_result:
                print(f"      Reason: {suite_result['reason']}")
        
        print("\n" + "-"*80)
        print(f"Suites Passed: {total_passed}")
        print(f"Suites Failed: {total_failed}")
        print(f"Suites Skipped: {total_skipped}")
        print("="*80 + "\n")
    
    def save_detailed_report(self):
        """Save detailed JSON report"""
        report = {
            'timestamp': self.start_time.isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'results': self.results,
            'summary': {
                'total_suites': len(self.results),
                'passed': sum(1 for r in self.results.values() if r.get('passed')),
                'failed': sum(1 for r in self.results.values() if not r.get('passed')),
                'skipped': sum(1 for r in self.results.values() if r.get('skipped'))
            }
        }
        
        # Save to current directory
        test_dir = Path(__file__).parent
        report_file = test_dir / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved: {report_file}")
    
    def run_all(self):
        """Run all test suites"""
        self.print_header()
        self.run_unit_tests()
        self.run_fda_validation()
        self.print_summary()
        self.save_detailed_report()
        
        # Return overall success
        return all(
            r.get('passed') or r.get('skipped') 
            for r in self.results.values()
        )


def print_test_documentation():
    """Print comprehensive test documentation"""
    doc = """
==========================================================================================================================================================================================================================================
===                     FDA SYNCER TEST SUITE DOCUMENTATION                    ===
==========================================================================================================================================================================================================================================

TEST SUITE 1: FDA Organization Validation
=======================================================================================================================================================================================================================================
Purpose: Detect breaking changes in FDA's data organization
Tests: 8
File: test_fda_organization.py
Network: Required

Tests:
  1. API Endpoint Accessibility (3 tests)
     - Drug Labels API
     - Drug Events API  
     - Drug Enforcement API
     
  2. API Response Structure (4 tests)
     - Label API structure
     - Adverse Events API structure
     - Enforcement API structure
     - Data type validation
     
  3. Website Accessibility (1 test)
     - Approval packages website

TEST SUITE 2: Unit Tests
=======================================================================================================================================================================================================================================
Purpose: Validate all system components work correctly
Tests: 52
File: test_offline_unit_tests.py
Network: Not required

Test Categories:

  A. Configuration (10 tests)
     - MODE validation
     - SYNC_AREAS validation
     - API key configuration
     - Therapeutic areas structure
     - Aliases structure
     - Config functions

  B. Therapeutic Areas (7 tests)
     - Disease count functions
     - Drug class count functions
     - Keyword expansion
     - Alias mapping
     - Duplicate detection

  C. Helper Functions (7 tests)
     - Directory creation
     - Drug name extraction
     - File existence checking
     - Deduplication logic

  D. HTTP Client (8 tests)
     - Initialization
     - Request handling
     - Retry logic (especially 404s)
     - Timeout configuration
     - SSL verification
     - Rate limiting
     - File download

  E. Labels Downloader (8 tests)
     - Initialization
     - Keyword batching (prevents URI too long)
     - Search query building
     - Config respect
     - Progress loading/saving
     - Finalization

  F. Approval Packages Downloader (7 tests)
     - Initialization
     - Document categorization
     - Filename sanitization
     - Application number extraction
     - Config respect

  G. Adverse Events Downloader (3 tests)
     - Initialization
     - Config respect
     - Progress loading

  H. Enforcement Downloader (2 tests)
     - Initialization
     - Config respect

RUNNING THE TESTS
=======================================================================================================================================================================================================================================

Run all tests:
  $ python test_runner.py

Run specific suite:
  $ python test_offline_unit_tests.py
  $ python test_fda_organization.py  # Requires network

INTERPRETING RESULTS
=======================================================================================================================================================================================================================================

=== PASS  - Test passed successfully
=== FAIL  - Test failed (needs investigation)
=== SKIP  - Test skipped (e.g., no network)

Common Failure Reasons:
  1. Network unavailable (FDA validation suite)
  2. Import errors (check project structure)
  3. Configuration changes (check syncher_keys.py)
  4. FDA API changes (check FDA organization validation)

TROUBLESHOOTING
=======================================================================================================================================================================================================================================

If FDA Organization Validation fails:
  === FDA may have changed their API structure
  === Review the failing tests to identify changes
  === Update downloaders accordingly

If Unit Tests fail:
  === Check import paths match your project structure
  === Verify all config files are present
  === Check Python version compatibility

For detailed logs:
  === Check test_report.json after running tests
  === Review stdout/stderr in test results

==========================================================================================================================================================================================================================================
"""
    print(doc)


if __name__ == '__main__':
    print_test_documentation()
    
    print("\n" + "="*80)
    response = input("Run all tests now? (y/n): ")
    print("="*80)
    
    if response.lower() in ['y', 'yes']:
        runner = TestRunner()
        success = runner.run_all()
        sys.exit(0 if success else 1)
    else:
        print("\nTests not run. Use 'python test_runner.py' to run them later.")