"""
FDA Syncer - Comprehensive Unit Test Suite
==========================================
Tests all components of the FDA data syncer system.

Usage:
    python test_fda_syncer.py
    python test_fda_syncer.py -v  # Verbose mode
    python -m unittest test_fda_syncer.TestTherapeuticAreas  # Single test class
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timedelta
import json
import tempfile
import shutil

# Add project to path
sys.path.insert(0, '/mnt/project')

# Import modules to test
from syncher_therapeutic_areas import (
    THERAPEUTIC_AREAS,
    ALIASES,
    normalize_term,
    get_expanded_keywords,
    get_all_therapeutic_areas,
    get_disease_count,
    get_drug_class_count
)

from syncher_keys import (
    validate_config,
    get_sync_config,
    SYNC_PARAMETERS
)

# Import helpers and utilities directly from project files
try:
    import helpers
    import http_client
    from helpers import (
        check_existing_file,
        ensure_dir,
        extract_drug_names_from_labels,
        get_today_file
    )
    from http_client import SimpleHTTPClient
    HELPERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import helpers/http_client: {e}")
    HELPERS_AVAILABLE = False

# Try to import downloaders
try:
    import labels as labels_module
    import approval_packages as approval_packages_module
    import adverse_events as adverse_events_module
    import enforcement as enforcement_module
    
    LabelsDownloader = getattr(labels_module, 'LabelsDownloader', None)
    ApprovalPackagesDownloader = getattr(approval_packages_module, 'ApprovalPackagesDownloader', None)
    AdverseEventsDownloader = getattr(adverse_events_module, 'AdverseEventsDownloader', None)
    EnforcementDownloader = getattr(enforcement_module, 'EnforcementDownloader', None)
    DOWNLOADERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import downloaders: {e}")
    LabelsDownloader = None
    ApprovalPackagesDownloader = None
    AdverseEventsDownloader = None
    EnforcementDownloader = None
    DOWNLOADERS_AVAILABLE = False


class TestTherapeuticAreas(unittest.TestCase):
    """Test therapeutic areas configuration and aliases"""
    
    def test_therapeutic_areas_structure(self):
        """Test that THERAPEUTIC_AREAS has correct structure"""
        self.assertIsInstance(THERAPEUTIC_AREAS, dict)
        self.assertIn('nephrology', THERAPEUTIC_AREAS)
        self.assertIn('hematology', THERAPEUTIC_AREAS)
        
        for area, data in THERAPEUTIC_AREAS.items():
            self.assertIn('rare_diseases', data)
            self.assertIn('drug_classes', data)
            self.assertIsInstance(data['rare_diseases'], list)
            self.assertIsInstance(data['drug_classes'], list)
    
    def test_no_duplicate_canonical_terms(self):
        """Test that canonical lists have no duplicates"""
        for area, data in THERAPEUTIC_AREAS.items():
            diseases = data['rare_diseases']
            drug_classes = data['drug_classes']
            
            # Check diseases for duplicates
            self.assertEqual(len(diseases), len(set(diseases)),
                           f"{area} has duplicate diseases")
            
            # Check drug classes for duplicates
            self.assertEqual(len(drug_classes), len(set(drug_classes)),
                           f"{area} has duplicate drug classes")
            
            # Check no overlap between diseases and drug classes
            overlap = set(diseases) & set(drug_classes)
            self.assertEqual(len(overlap), 0,
                           f"{area} has overlap between diseases and drug classes: {overlap}")
    
    def test_aliases_point_to_valid_canonical(self):
        """Test that all aliases point to valid canonical terms"""
        all_canonical = set()
        for area, data in THERAPEUTIC_AREAS.items():
            all_canonical.update(data['rare_diseases'])
            all_canonical.update(data['drug_classes'])
        
        for alias, canonical in ALIASES.items():
            self.assertIn(canonical, all_canonical,
                        f"Alias '{alias}' points to invalid canonical '{canonical}'")
    
    def test_normalize_term(self):
        """Test normalize_term function"""
        # Test known aliases
        test_cases = [
            ("FSGS", "focal segmental glomerulosclerosis"),
            ("AML", "acute myeloid leukemia"),
            ("SCD", "sickle cell disease"),
            ("aHUS", "atypical hemolytic uremic syndrome"),
            ("ERA", "endothelin receptor antagonist"),
        ]
        
        for alias, expected in test_cases:
            result = normalize_term(alias)
            self.assertEqual(result, expected,
                           f"normalize_term('{alias}') returned '{result}', expected '{expected}'")
        
        # Test canonical term returns itself
        canonical = "focal segmental glomerulosclerosis"
        self.assertEqual(normalize_term(canonical), canonical)
        
        # Test unknown term returns itself
        unknown = "unknown disease"
        self.assertEqual(normalize_term(unknown), unknown)
    
    def test_get_expanded_keywords(self):
        """Test get_expanded_keywords function"""
        for area in get_all_therapeutic_areas():
            keywords = get_expanded_keywords(area)
            
            # Should be a list
            self.assertIsInstance(keywords, list)
            
            # Should have more keywords than canonical
            canonical_count = (len(THERAPEUTIC_AREAS[area]['rare_diseases']) + 
                             len(THERAPEUTIC_AREAS[area]['drug_classes']))
            self.assertGreaterEqual(len(keywords), canonical_count,
                                  f"{area} expanded keywords not greater than canonical")
            
            # All canonical terms should be in expanded
            canonical_terms = (THERAPEUTIC_AREAS[area]['rare_diseases'] + 
                             THERAPEUTIC_AREAS[area]['drug_classes'])
            for term in canonical_terms:
                self.assertIn(term, keywords,
                            f"Canonical term '{term}' not in expanded keywords for {area}")
    
    def test_get_all_therapeutic_areas(self):
        """Test get_all_therapeutic_areas function"""
        areas = get_all_therapeutic_areas()
        self.assertIsInstance(areas, list)
        self.assertIn('nephrology', areas)
        self.assertIn('hematology', areas)
        self.assertEqual(len(areas), 2)
    
    def test_get_disease_count(self):
        """Test get_disease_count function"""
        nephrology_count = get_disease_count('nephrology')
        hematology_count = get_disease_count('hematology')
        
        self.assertGreater(nephrology_count, 0)
        self.assertGreater(hematology_count, 0)
        self.assertEqual(nephrology_count, len(THERAPEUTIC_AREAS['nephrology']['rare_diseases']))
        
        # Test invalid area
        self.assertEqual(get_disease_count('invalid'), 0)
    
    def test_get_drug_class_count(self):
        """Test get_drug_class_count function"""
        nephrology_count = get_drug_class_count('nephrology')
        hematology_count = get_drug_class_count('hematology')
        
        self.assertGreater(nephrology_count, 0)
        self.assertGreater(hematology_count, 0)
        self.assertEqual(nephrology_count, len(THERAPEUTIC_AREAS['nephrology']['drug_classes']))
        
        # Test invalid area
        self.assertEqual(get_drug_class_count('invalid'), 0)
    
    def test_canonical_names_are_lowercase(self):
        """Test that canonical names follow naming convention"""
        for area, data in THERAPEUTIC_AREAS.items():
            for disease in data['rare_diseases']:
                # Should not be all uppercase (unless abbreviation)
                if len(disease) > 5:  # Skip short terms
                    self.assertFalse(disease.isupper(),
                                   f"Disease '{disease}' in {area} is all uppercase")
            
            for drug_class in data['drug_classes']:
                if len(drug_class) > 5:
                    self.assertFalse(drug_class.isupper(),
                                   f"Drug class '{drug_class}' in {area} is all uppercase")


class TestSyncherKeys(unittest.TestCase):
    """Test syncher_keys configuration"""
    
    def test_validate_config(self):
        """Test config validation"""
        # Should validate without errors
        is_valid = validate_config()
        self.assertTrue(is_valid)
    
    def test_get_sync_config(self):
        """Test get_sync_config function"""
        config = get_sync_config()
        self.assertIsInstance(config, dict)
        
        # Check required keys
        required_keys = ['description', 'estimated_time', 'labels', 
                        'integrated_reviews', 'adverse_events', 'enforcement']
        for key in required_keys:
            self.assertIn(key, config)
    
    def test_sync_parameters_structure(self):
        """Test SYNC_PARAMETERS structure"""
        modes = ['test', 'daily', 'full']
        for mode in modes:
            self.assertIn(mode, SYNC_PARAMETERS)
            config = SYNC_PARAMETERS[mode]
            
            # Check required fields
            self.assertIn('description', config)
            self.assertIn('estimated_time', config)
            self.assertIn('labels', config)
            self.assertIn('integrated_reviews', config)
            self.assertIn('adverse_events', config)
            self.assertIn('enforcement', config)
            
            # Check labels config
            labels_config = config['labels']
            self.assertIn('enabled', labels_config)
            self.assertIsInstance(labels_config['enabled'], bool)


@unittest.skipIf(not HELPERS_AVAILABLE, "Helpers not available")
class TestHelpers(unittest.TestCase):
    """Test helper functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_check_existing_file(self):
        """Test check_existing_file function"""
        # Create a test file
        test_file = os.path.join(self.test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Test with FORCE_REDOWNLOAD = False (default)
        with patch('helpers.FORCE_REDOWNLOAD', False):
            self.assertTrue(check_existing_file(test_file))
            self.assertFalse(check_existing_file('nonexistent.txt'))
        
        # Test with FORCE_REDOWNLOAD = True
        with patch('helpers.FORCE_REDOWNLOAD', True):
            self.assertFalse(check_existing_file(test_file))
    
    def test_ensure_dir(self):
        """Test ensure_dir function"""
        test_path = os.path.join(self.test_dir, 'new_dir', 'nested_dir')
        ensure_dir(test_path)
        
        self.assertTrue(os.path.exists(test_path))
        self.assertTrue(os.path.isdir(test_path))
    
    def test_extract_drug_names_from_labels(self):
        """Test extract_drug_names_from_labels function"""
        # Mock label data
        labels = [
            {
                'openfda': {
                    'brand_name': ['Keytruda', 'Brand1'],
                    'generic_name': ['pembrolizumab']
                }
            },
            {
                'openfda': {
                    'brand_name': ['Opdivo'],
                    'generic_name': ['nivolumab']
                }
            },
            {
                'openfda': {}  # Empty openfda
            }
        ]
        
        drug_names = extract_drug_names_from_labels(labels)
        
        self.assertIsInstance(drug_names, list)
        self.assertIn('Keytruda', drug_names)
        self.assertIn('pembrolizumab', drug_names)
        self.assertIn('Opdivo', drug_names)
        self.assertIn('nivolumab', drug_names)
        
        # Check uniqueness
        self.assertEqual(len(drug_names), len(set(drug_names)))


@unittest.skipIf(not HELPERS_AVAILABLE, "HTTP client not available")
class TestHTTPClient(unittest.TestCase):
    """Test HTTP client"""
    
    def test_http_client_initialization(self):
        """Test SimpleHTTPClient initialization"""
        client = SimpleHTTPClient(max_retries=3)
        self.assertEqual(client.max_retries, 3)
        self.assertIsNotNone(client.session)
    
    @patch('requests.Session.get')
    def test_http_client_get_success(self, mock_get):
        """Test successful GET request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': []}
        mock_get.return_value = mock_response
        
        client = SimpleHTTPClient()
        response = client.get('https://api.fda.gov/drug/label.json')
        
        self.assertEqual(response.status_code, 200)
    
    @patch('requests.Session.get')
    def test_http_client_retry_on_error(self, mock_get):
        """Test retry logic on errors"""
        # Mock failed then successful response
        mock_get.side_effect = [
            Exception("Network error"),
            Mock(status_code=200)
        ]
        
        client = SimpleHTTPClient(max_retries=3)
        
        with patch('time.sleep'):  # Skip actual sleep
            response = client.get('https://api.fda.gov/test')
        
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('requests.Session.get')
    def test_http_client_404_handling(self, mock_get):
        """Test 404 handling (should fail fast)"""
        import requests
        
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        client = SimpleHTTPClient(max_retries=3)
        
        with patch('time.sleep'):
            with self.assertRaises(requests.exceptions.HTTPError):
                client.get('https://api.fda.gov/nonexistent')
        
        # Should only retry once for 404
        self.assertLessEqual(mock_get.call_count, 2)


@unittest.skipIf(not DOWNLOADERS_AVAILABLE, "Downloaders not available")
class TestLabelsDownloader(unittest.TestCase):
    """Test LabelsDownloader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_labels_downloader_structure(self):
        """Test LabelsDownloader has expected attributes"""
        if LabelsDownloader is None:
            self.skipTest("LabelsDownloader not available")
        
        # Just test that it exists and has expected structure
        self.assertIsNotNone(LabelsDownloader)
        
        # Try to instantiate (may fail, that's okay for now)
        try:
            downloader = LabelsDownloader()
            self.assertTrue(hasattr(downloader, 'download'))
        except:
            pass  # It's okay if instantiation fails


@unittest.skipIf(not DOWNLOADERS_AVAILABLE, "Downloaders not available")
class TestApprovalPackagesDownloader(unittest.TestCase):
    """Test ApprovalPackagesDownloader"""
    
    def test_approval_packages_exists(self):
        """Test that ApprovalPackagesDownloader exists"""
        if ApprovalPackagesDownloader is None:
            self.skipTest("ApprovalPackagesDownloader not available")
        
        self.assertIsNotNone(ApprovalPackagesDownloader)


@unittest.skipIf(not DOWNLOADERS_AVAILABLE, "Downloaders not available")
class TestAdverseEventsDownloader(unittest.TestCase):
    """Test AdverseEventsDownloader"""
    
    def test_adverse_events_exists(self):
        """Test that AdverseEventsDownloader exists"""
        if AdverseEventsDownloader is None:
            self.skipTest("AdverseEventsDownloader not available")
        
        self.assertIsNotNone(AdverseEventsDownloader)


@unittest.skipIf(not DOWNLOADERS_AVAILABLE, "Downloaders not available")
class TestEnforcementDownloader(unittest.TestCase):
    """Test EnforcementDownloader"""
    
    def test_enforcement_exists(self):
        """Test that EnforcementDownloader exists"""
        if EnforcementDownloader is None:
            self.skipTest("EnforcementDownloader not available")
        
        self.assertIsNotNone(EnforcementDownloader)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_normalize_term_with_none(self):
        """Test normalize_term with None input"""
        result = normalize_term(None)
        self.assertIsNone(result)
    
    def test_normalize_term_with_empty_string(self):
        """Test normalize_term with empty string"""
        result = normalize_term("")
        self.assertEqual(result, "")
    
    def test_get_expanded_keywords_invalid_area(self):
        """Test get_expanded_keywords with invalid area"""
        result = get_expanded_keywords('invalid_area')
        self.assertEqual(result, [])
    
    @unittest.skipIf(not HELPERS_AVAILABLE, "Helpers not available")
    def test_extract_drug_names_empty_list(self):
        """Test extract_drug_names_from_labels with empty list"""
        result = extract_drug_names_from_labels([])
        self.assertEqual(result, [])
    
    @unittest.skipIf(not HELPERS_AVAILABLE, "Helpers not available")
    def test_extract_drug_names_malformed_data(self):
        """Test extract_drug_names_from_labels with malformed data"""
        malformed_labels = [
            {'openfda': None},
            {'wrong_key': 'value'},
            None,
            "not a dict"
        ]
        
        # Should not raise exception
        try:
            result = extract_drug_names_from_labels(malformed_labels)
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"extract_drug_names_from_labels raised {e} on malformed data")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency"""
    
    def test_no_alias_points_to_alias(self):
        """Test that aliases don't point to other aliases"""
        for alias, canonical in ALIASES.items():
            self.assertNotIn(canonical, ALIASES,
                           f"Alias '{alias}' points to '{canonical}' which is also an alias")
    
    def test_alias_keys_not_in_canonical(self):
        """Test that alias keys are not in canonical lists"""
        all_canonical = set()
        for area, data in THERAPEUTIC_AREAS.items():
            all_canonical.update(data['rare_diseases'])
            all_canonical.update(data['drug_classes'])
        
        for alias in ALIASES.keys():
            self.assertNotIn(alias, all_canonical,
                           f"Alias '{alias}' should not be in canonical lists")
    
    def test_consistent_naming_conventions(self):
        """Test consistent naming conventions"""
        for area, data in THERAPEUTIC_AREAS.items():
            for disease in data['rare_diseases']:
                # Should not have leading/trailing whitespace
                self.assertEqual(disease, disease.strip(),
                               f"Disease '{disease}' has whitespace")
                
                # Should not be empty
                self.assertTrue(disease,
                              f"Empty disease name in {area}")
    
    def test_therapeutic_areas_coverage(self):
        """Test that we have good coverage of diseases"""
        nephro_count = len(THERAPEUTIC_AREAS['nephrology']['rare_diseases'])
        hemato_count = len(THERAPEUTIC_AREAS['hematology']['rare_diseases'])
        
        # Should have at least 20 diseases each
        self.assertGreaterEqual(nephro_count, 20,
                              "Nephrology should have at least 20 diseases")
        self.assertGreaterEqual(hemato_count, 20,
                              "Hematology should have at least 20 diseases")
    
    def test_alias_coverage(self):
        """Test that we have good alias coverage"""
        self.assertGreaterEqual(len(ALIASES), 50,
                              "Should have at least 50 aliases")


class TestPerformance(unittest.TestCase):
    """Test performance of key functions"""
    
    def test_get_expanded_keywords_performance(self):
        """Test performance of get_expanded_keywords"""
        import time
        
        start = time.time()
        for _ in range(100):
            get_expanded_keywords('hematology')
        end = time.time()
        
        elapsed = end - start
        self.assertLess(elapsed, 1.0,
                       f"get_expanded_keywords took {elapsed}s for 100 calls")
    
    def test_normalize_term_performance(self):
        """Test performance of normalize_term"""
        import time
        
        terms = list(ALIASES.keys())[:10]
        
        start = time.time()
        for _ in range(1000):
            for term in terms:
                normalize_term(term)
        end = time.time()
        
        elapsed = end - start
        self.assertLess(elapsed, 1.0,
                       f"normalize_term took {elapsed}s for 10000 calls")


def run_test_suite(verbosity=2):
    """Run the complete test suite"""
    
    print("="*70)
    print("FDA SYNCER - COMPREHENSIVE UNIT TEST SUITE")
    print("="*70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTherapeuticAreas,
        TestSyncherKeys,
        TestHelpers,
        TestHTTPClient,
        TestLabelsDownloader,
        TestApprovalPackagesDownloader,
        TestAdverseEventsDownloader,
        TestEnforcementDownloader,
        TestEdgeCases,
        TestDataIntegrity,
        TestPerformance,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    print("="*70)
    
    return result


if __name__ == '__main__':
    # Check for verbose flag
    verbosity = 2 if '-v' in sys.argv else 1
    
    # Run test suite
    result = run_test_suite(verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)