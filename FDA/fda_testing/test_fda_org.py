"""
FDA Data Organization Validation Test - VERBOSE VERSION
========================================================
Tests FDA APIs with detailed progress output

Run this to check if FDA's APIs are working before syncing data.
"""

import requests
import sys
from datetime import datetime, timedelta
import json
from pathlib import Path

class FDAValidator:
    """Validates FDA API endpoints with verbose logging"""
    
    def __init__(self):
        self.results = []
        self.failures = []
        
    def log(self, message, indent=0):
        """Print with indentation"""
        print("  " * indent + message)
    
    def log_test(self, name, passed, details=""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.log(f"{status} - {name}")
        if details:
            self.log(f"  → {details}", 1)
        
        self.results.append({
            'name': name,
            'passed': passed,
            'details': details
        })
        
        if not passed:
            self.failures.append(name)
    
    def test_labels_api(self):
        """Test 1: Drug Labels API"""
        print("\n" + "="*80)
        print("TEST 1: Drug Labels API")
        print("="*80)
        
        self.log("Testing basic connectivity...")
        url = "https://api.fda.gov/drug/label.json"
        
        try:
            # Test 1a: Basic endpoint
            self.log("1a. Testing endpoint accessibility...", 1)
            params = {"limit": 1}
            response = requests.get(url, params=params, timeout=10)
            self.log_test(
                "Labels API accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
            
            # Test 1b: Search by brand name
            self.log("1b. Testing brand name search...", 1)
            params = {
                "search": 'openfda.brand_name:"keytruda"',
                "limit": 1
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            has_results = 'results' in data and len(data['results']) > 0
            self.log_test(
                "Brand name search works",
                has_results,
                f"Found {len(data.get('results', []))} results"
            )
            
            if has_results:
                result = data['results'][0]
                self.log(f"Sample drug: {result.get('openfda', {}).get('brand_name', ['Unknown'])[0]}", 2)
            
            # Test 1c: Structure validation
            self.log("1c. Validating response structure...", 1)
            if has_results:
                result = data['results'][0]
                has_openfda = 'openfda' in result
                has_brand = has_openfda and 'brand_name' in result['openfda']
                has_app_num = has_openfda and 'application_number' in result['openfda']
                
                self.log_test(
                    "Response has required fields",
                    has_openfda and (has_brand or has_app_num),
                    f"openfda: {has_openfda}, brand_name: {has_brand}, app_number: {has_app_num}"
                )
            
        except Exception as e:
            self.log_test("Labels API test", False, f"Error: {str(e)}")
    
    def test_adverse_events_api(self):
        """Test 2: Adverse Events API"""
        print("\n" + "="*80)
        print("TEST 2: Adverse Events API")
        print("="*80)
        
        url = "https://api.fda.gov/drug/event.json"
        
        # Test 2a: Basic endpoint
        self.log("2a. Testing endpoint accessibility...", 1)
        try:
            params = {"limit": 1}
            response = requests.get(url, params=params, timeout=10)
            self.log_test(
                "Adverse Events API accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.log_test("Adverse Events API", False, f"Error: {str(e)}")
            return
        
        # Test 2b: Date range query (THE CRITICAL TEST)
        self.log("2b. Testing date range query...", 1)
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            
            self.log(f"Date range: {date_range}", 2)
            
            params = {
                "search": f"receivedate:{date_range}",
                "limit": 5
            }
            
            self.log("Sending query...", 2)
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            self.log(f"Response keys: {list(data.keys())}", 2)
            
            if 'error' in data:
                self.log(f"ERROR from API: {data['error']}", 2)
                self.log_test(
                    "Date range query works",
                    False,
                    f"API returned error: {data['error'].get('message', 'Unknown error')}"
                )
            elif 'results' in data:
                result_count = len(data.get('results', []))
                self.log(f"Got {result_count} results", 2)
                self.log_test(
                    "Date range query works",
                    result_count > 0,
                    f"Returned {result_count} events"
                )
            else:
                self.log_test(
                    "Date range query works",
                    False,
                    f"Unexpected response: {list(data.keys())}"
                )
                
        except Exception as e:
            self.log_test("Date range query", False, f"Error: {str(e)}")
        
        # Test 2c: Drug-specific query
        self.log("2c. Testing drug-specific query...", 1)
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Wider range
            date_range = f"[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            
            params = {
                "search": f'patient.drug.medicinalproduct:"aspirin" AND receivedate:{date_range}',
                "limit": 5
            }
            
            self.log("Query: aspirin in last year...", 2)
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'error' in data:
                self.log(f"ERROR: {data['error']}", 2)
                self.log_test(
                    "Drug-specific query works",
                    False,
                    f"API error: {data['error'].get('message', 'Unknown')}"
                )
            else:
                result_count = len(data.get('results', []))
                self.log(f"Got {result_count} results", 2)
                self.log_test(
                    "Drug-specific query works",
                    result_count > 0,
                    f"Found {result_count} events for aspirin"
                )
                
        except Exception as e:
            self.log_test("Drug-specific query", False, f"Error: {str(e)}")
    
    def test_enforcement_api(self):
        """Test 3: Enforcement API"""
        print("\n" + "="*80)
        print("TEST 3: Enforcement Reports API")
        print("="*80)
        
        url = "https://api.fda.gov/drug/enforcement.json"
        
        self.log("Testing enforcement reports...", 1)
        try:
            params = {"limit": 5}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            has_results = 'results' in data and len(data['results']) > 0
            self.log_test(
                "Enforcement API works",
                has_results,
                f"Retrieved {len(data.get('results', []))} reports"
            )
            
            if has_results:
                report = data['results'][0]
                self.log(f"Sample: {report.get('product_description', 'N/A')[:50]}...", 2)
            
        except Exception as e:
            self.log_test("Enforcement API", False, f"Error: {str(e)}")
    
    def test_approval_packages_site(self):
        """Test 4: Approval Packages Website"""
        print("\n" + "="*80)
        print("TEST 4: Approval Packages Website")
        print("="*80)
        
        base_url = "https://www.accessdata.fda.gov"
        
        self.log("Testing AccessData website...", 1)
        try:
            response = requests.get(base_url, timeout=10, verify=False)
            self.log_test(
                "AccessData website accessible",
                response.status_code == 200,
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.log_test("AccessData website", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*80)
        print("FDA API VALIDATION - VERBOSE MODE")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        self.test_labels_api()
        self.test_adverse_events_api()
        self.test_enforcement_api()
        self.test_approval_packages_site()
        
        self.print_summary()
        
        return len(self.failures) == 0
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        
        if failed > 0:
            print(f"\n⚠️  FAILING TESTS:")
            for failure in self.failures:
                print(f"  ✗ {failure}")
            
            print(f"\n⚠️  WARNING: Some FDA APIs have issues!")
            print("Check the failures above before running your sync.")
        else:
            print(f"\n✓ All tests passed! FDA APIs are working correctly.")
        
        print("="*80)
        
        # Save results
        test_dir = Path(__file__).parent
        results_file = test_dir / "fda_validation_verbose.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'tests': self.results
        }
        
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed results: {results_file}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    validator = FDAValidator()
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)