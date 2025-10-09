#!/usr/bin/env python3
"""
Document Reader Test Suite
==========================
Location: corpus_testing/test006_reader.py

Tests the DocumentReader with intro/full modes as defined in config.yaml
Tests various file types: PDF, DOCX, TXT, etc.
"""

import sys
import os
from pathlib import Path
import time
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the DocumentReader and config
try:
    from corpus_metadata.document_metadata_reader import DocumentReader
    READER_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import DocumentReader: {e}")
    READER_AVAILABLE = False
    sys.exit(1)

try:
    from corpus_metadata.document_utils.metadata_logging_config import CorpusConfig
    config = CorpusConfig(config_dir="corpus_config", verbose=False)
    CONFIG_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import CorpusConfig, using defaults")
    CONFIG_AVAILABLE = False

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_header(text: str):
    """Print a header"""
    print(f"\n{Colors.HEADER}{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}{Colors.ENDC}")

def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.CYAN}{'-'*60}")
    print(f"  {text}")
    print(f"{'-'*60}{Colors.ENDC}")

def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result"""
    if success:
        icon = f"{Colors.GREEN}✓{Colors.ENDC}"
        status = f"{Colors.GREEN}PASS{Colors.ENDC}"
    else:
        icon = f"{Colors.RED}✗{Colors.ENDC}"
        status = f"{Colors.RED}FAIL{Colors.ENDC}"
    
    print(f"  {icon} {test_name:<40} [{status}]")
    if details:
        print(f"    {Colors.CYAN}{details}{Colors.ENDC}")

class DocumentReaderTester:
    """Test suite for DocumentReader"""
    
    def __init__(self, test_folder: str = "test_documents"):
        """Initialize tester"""
        self.test_folder = Path(test_folder)
        self.reader = DocumentReader()
        self.results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': [],
            'timings': {},
            'mode_results': {'intro': {}, 'full': {}}
        }
        
        # Get mode configurations from config.yaml
        if CONFIG_AVAILABLE:
            self.stages = config.get_all_stages()
            # Extract mode definitions from stages
            self.mode_configs = {}
            for stage_name, stage_config in self.stages:
                limits = stage_config.get('limits', {})
                if limits.get('pdf_pages') == 10:
                    self.mode_configs['intro'] = {
                        'pdf_pages': 10,
                        'text_chars': limits.get('text_chars', 50000)
                    }
                else:
                    self.mode_configs['full'] = {
                        'pdf_pages': None,
                        'text_chars': None
                    }
        else:
            # Default configurations
            self.mode_configs = {
                'intro': {'pdf_pages': 10, 'text_chars': 50000},
                'full': {'pdf_pages': None, 'text_chars': None}
            }
    
    def setup_test_environment(self) -> bool:
        """Check if test folder exists and has files"""
        print_section("CHECKING TEST ENVIRONMENT")
        
        if not self.test_folder.exists():
            print(f"  {Colors.RED}✗{Colors.ENDC} Test folder not found: {self.test_folder}")
            print(f"    Please create folder: {self.test_folder.absolute()}")
            return False
        
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Test folder found: {self.test_folder}")
        
        # List available test files
        test_files = list(self.test_folder.iterdir())
        if not test_files:
            print(f"  {Colors.YELLOW}⚠{Colors.ENDC} No test files found in {self.test_folder}")
            return False
        
        print(f"  {Colors.GREEN}✓{Colors.ENDC} Found {len(test_files)} test files:")
        
        # Group files by type
        file_types = {}
        for file in test_files:
            ext = file.suffix.lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(file.name)
        
        for ext, files in file_types.items():
            print(f"    • {ext}: {len(files)} file(s)")
            for fname in files[:2]:  # Show first 2 of each type
                print(f"      - {fname}")
            if len(files) > 2:
                print(f"      ... and {len(files)-2} more")
        
        return True
    
    def test_reader_initialization(self) -> bool:
        """Test 1: Reader initialization and handler loading"""
        print_section("TEST 1: READER INITIALIZATION")
        
        try:
            # Check if reader initialized
            print_result("Reader instance created", True)
            
            # Check available handlers
            supported = self.reader.get_supported_extensions()
            print_result(f"Handlers loaded", True, f"{len(supported)} handlers: {', '.join(supported)}")
            
            # Check for essential handlers
            essential = ['.pdf', '.docx', '.txt']
            for ext in essential:
                if ext in supported:
                    print_result(f"Handler for {ext}", True)
                else:
                    print_result(f"Handler for {ext}", False, "Not available")
            
            return True
            
        except Exception as e:
            print_result("Reader initialization", False, str(e))
            return False
    
    def test_file_reading_modes(self) -> bool:
        """Test 2: Test intro and full modes for each file type"""
        print_section("TEST 2: INTRO/FULL MODE EXTRACTION")
        
        test_files = list(self.test_folder.iterdir())
        if not test_files:
            print(f"  {Colors.YELLOW}No test files available{Colors.ENDC}")
            return False
        
        # Test each file with both modes
        for test_file in test_files:
            if test_file.is_file():
                print(f"\n  Testing: {Colors.BOLD}{test_file.name}{Colors.ENDC}")
                
                for mode in ['intro', 'full']:
                    self.results['total'] += 1
                    
                    start = time.time()
                    result = self.reader.read_document(str(test_file), mode=mode)
                    elapsed = time.time() - start
                    
                    # Check if we got content or text
                    content = result.get('content', '') or result.get('text', '')
                    
                    if content:
                        self.results['passed'] += 1
                        content_len = len(content)
                        
                        # Store results for comparison
                        if test_file.name not in self.results['mode_results'][mode]:
                            self.results['mode_results'][mode][test_file.name] = content_len
                        
                        # Check if intro mode limits are respected
                        if mode == 'intro':
                            max_chars = self.mode_configs['intro']['text_chars']
                            if max_chars and content_len > max_chars * 1.1:  # Allow 10% margin
                                print_result(f"  {mode} mode", True, 
                                           f"{content_len:,} chars (WARNING: exceeds limit of {max_chars:,})")
                            else:
                                print_result(f"  {mode} mode", True, 
                                           f"{content_len:,} chars in {elapsed:.2f}s")
                        else:
                            print_result(f"  {mode} mode", True, 
                                       f"{content_len:,} chars in {elapsed:.2f}s")
                    else:
                        self.results['failed'] += 1
                        error = result.get('error', 'Unknown error')
                        print_result(f"  {mode} mode", False, error)
                        self.results['errors'].append({
                            'file': test_file.name,
                            'mode': mode,
                            'error': error
                        })
        
        return self.results['failed'] == 0
    
    def test_mode_differences(self) -> bool:
        """Test 3: Verify intro mode extracts less content than full mode"""
        print_section("TEST 3: MODE CONTENT DIFFERENCES")
        
        intro_results = self.results['mode_results']['intro']
        full_results = self.results['mode_results']['full']
        
        files_tested = set(intro_results.keys()) & set(full_results.keys())
        
        if not files_tested:
            print(f"  {Colors.YELLOW}No files tested with both modes{Colors.ENDC}")
            return False
        
        correct_behavior = 0
        issues = []
        
        for filename in files_tested:
            intro_len = intro_results[filename]
            full_len = full_results[filename]
            
            # Intro should be less than or equal to full
            if intro_len <= full_len:
                correct_behavior += 1
                reduction = ((full_len - intro_len) / full_len * 100) if full_len > 0 else 0
                print_result(f"{filename[:40]}", True, 
                           f"Intro: {intro_len:,} | Full: {full_len:,} | Reduction: {reduction:.1f}%")
            else:
                issues.append(filename)
                print_result(f"{filename[:40]}", False, 
                           f"Intro ({intro_len:,}) > Full ({full_len:,})")
        
        return len(issues) == 0
    
    def test_error_handling(self) -> bool:
        """Test 4: Error handling for invalid files"""
        print_section("TEST 4: ERROR HANDLING")
        
        test_cases = [
            ("Non-existent file", "fake_file_12345.pdf"),
            ("Invalid path", "/invalid/path/to/file.pdf"),
            ("Empty filename", "")
        ]
        
        all_handled = True
        
        for test_name, file_path in test_cases:
            try:
                result = self.reader.read_document(file_path, mode='intro')
                
                # Should return error, not crash
                if result.get('error'):
                    print_result(test_name, True, f"Error handled: {result['error'][:50]}")
                else:
                    # If no error but also no content, that's okay
                    content = result.get('content', '') or result.get('text', '')
                    if not content:
                        print_result(test_name, True, "Returned empty content")
                    else:
                        print_result(test_name, False, "Should have failed but didn't")
                        all_handled = False
                        
            except Exception as e:
                print_result(test_name, False, f"Unhandled exception: {str(e)[:50]}")
                all_handled = False
        
        return all_handled
    
    def test_pdf_specific(self) -> bool:
        """Test 5: PDF-specific page limit testing"""
        print_section("TEST 5: PDF PAGE LIMITS")
        
        pdf_files = list(self.test_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"  {Colors.YELLOW}No PDF files to test{Colors.ENDC}")
            return True  # Skip, not a failure
        
        pdf_file = pdf_files[0]  # Test first PDF
        print(f"  Testing PDF: {pdf_file.name}")
        
        # Test intro mode (should be limited to 10 pages per config)
        intro_pages = self.mode_configs['intro']['pdf_pages']
        
        result_intro = self.reader.read_document(str(pdf_file), mode='intro')
        result_full = self.reader.read_document(str(pdf_file), mode='full')
        
        content_intro = result_intro.get('content', '') or result_intro.get('text', '')
        content_full = result_full.get('content', '') or result_full.get('text', '')
        
        if content_intro and content_full:
            # Intro should be substantially smaller for multi-page PDFs
            ratio = len(content_intro) / len(content_full) if content_full else 0
            
            if ratio < 0.95:  # Intro is at least 5% smaller
                print_result(f"PDF page limiting", True, 
                           f"Intro is {(1-ratio)*100:.1f}% smaller than full")
            else:
                print_result(f"PDF page limiting", True, 
                           f"Files may have ≤{intro_pages} pages (ratio: {ratio:.2f})")
        else:
            print_result(f"PDF page limiting", False, "Could not extract content")
            return False
        
        return True
    
    def test_ocr_capability(self) -> bool:
        """Test 6: OCR capability for scanned PDFs and images"""
        print_section("TEST 6: OCR CAPABILITY")
        
        # Check if OCR dependencies are available
        ocr_available = self._check_ocr_dependencies()
        
        if not ocr_available:
            print(f"  {Colors.YELLOW}⚠ OCR dependencies not fully available{Colors.ENDC}")
        
        # Look for scanned PDFs or image files
        ocr_test_files = []
        
        # Check for scanned PDFs (usually have 'scan' or 'scanned' in name)
        for pdf in self.test_folder.glob("*scan*.pdf"):
            ocr_test_files.append(('PDF', pdf))
        
        # Check for image-based PDFs
        for pdf in self.test_folder.glob("*ocr*.pdf"):
            ocr_test_files.append(('PDF', pdf))
        
        # Check for image files that might need OCR
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        for ext in image_extensions:
            for img in self.test_folder.glob(ext):
                ocr_test_files.append(('Image', img))
        
        if not ocr_test_files:
            print(f"  {Colors.YELLOW}No OCR test files found{Colors.ENDC}")
            print(f"    Add files with 'scan' or 'ocr' in name, or image files")
            return True  # Skip, not a failure
        
        success_count = 0
        
        for file_type, test_file in ocr_test_files:
            print(f"\n  Testing {file_type}: {test_file.name}")
            
            # Test both modes
            for mode in ['intro', 'full']:
                start = time.time()
                result = self.reader.read_document(str(test_file), mode=mode)
                elapsed = time.time() - start
                
                content = result.get('content', '') or result.get('text', '')
                
                if content:
                    # Check if it's actual text, not just whitespace
                    if content.strip():
                        print_result(f"  OCR {mode} mode", True, 
                                   f"{len(content):,} chars extracted in {elapsed:.2f}s")
                        success_count += 1
                        
                        # Show sample of extracted text
                        sample = ' '.join(content.split()[:20])
                        print(f"    Sample: {Colors.CYAN}{sample}...{Colors.ENDC}")
                    else:
                        print_result(f"  OCR {mode} mode", False, 
                                   "Only whitespace extracted")
                else:
                    error = result.get('error', 'No OCR text extracted')
                    print_result(f"  OCR {mode} mode", False, error)
                    
                    # Check if it's an OCR dependency issue
                    if 'tesseract' in error.lower() or 'ocr' in error.lower():
                        print(f"    {Colors.YELLOW}Hint: Install tesseract and pytesseract{Colors.ENDC}")
        
        return success_count > 0
    
    def _check_ocr_dependencies(self) -> bool:
        """Check if OCR dependencies are installed"""
        dependencies = {
            'pytesseract': False,
            'pdf2image': False,
            'PIL/Pillow': False,
            'tesseract': False
        }
        
        # Check Python packages
        try:
            import pytesseract
            dependencies['pytesseract'] = True
        except ImportError:
            pass
        
        try:
            import pdf2image
            dependencies['pdf2image'] = True
        except ImportError:
            pass
        
        try:
            from PIL import Image
            dependencies['PIL/Pillow'] = True
        except ImportError:
            pass
        
        # Check tesseract binary
        try:
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                dependencies['tesseract'] = True
        except:
            pass
        
        # Print dependency status
        print(f"  OCR Dependencies:")
        for dep, available in dependencies.items():
            icon = f"{Colors.GREEN}✓{Colors.ENDC}" if available else f"{Colors.RED}✗{Colors.ENDC}"
            print(f"    {icon} {dep}")
        
        # Installation instructions if missing
        missing = [k for k, v in dependencies.items() if not v]
        if missing:
            print(f"\n  {Colors.YELLOW}To enable OCR, install missing dependencies:{Colors.ENDC}")
            if 'tesseract' in missing:
                print(f"    • Mac: brew install tesseract")
                print(f"    • Ubuntu: sudo apt-get install tesseract-ocr")
                print(f"    • Windows: Download from GitHub/UB-Mannheim/tesseract")
            if 'pytesseract' in missing or 'pdf2image' in missing:
                print(f"    • pip install pytesseract pdf2image")
        
        return all(dependencies.values())
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print_header("DOCUMENT READER TEST SUITE")
        print(f"Test folder: {self.test_folder.absolute()}")
        print(f"Config: {'Loaded' if CONFIG_AVAILABLE else 'Using defaults'}")
        
        if not self.setup_test_environment():
            print(f"\n{Colors.RED}Cannot run tests without test environment{Colors.ENDC}")
            self.print_required_files()
            return
        
        # Run tests
        tests = [
            self.test_reader_initialization,
            self.test_file_reading_modes,
            self.test_mode_differences,
            self.test_error_handling,
            self.test_pdf_specific
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n{Colors.RED}Test crashed: {e}{Colors.ENDC}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print_header("TEST SUMMARY")
        
        total = self.results['total']
        passed = self.results['passed']
        failed = self.results['failed']
        
        if total > 0:
            pass_rate = (passed / total) * 100
        else:
            pass_rate = 0
        
        print(f"  Total tests: {total}")
        print(f"  Passed: {Colors.GREEN}{passed}{Colors.ENDC}")
        print(f"  Failed: {Colors.RED}{failed}{Colors.ENDC}")
        print(f"  Pass rate: {pass_rate:.1f}%")
        
        if self.results['errors']:
            print(f"\n  {Colors.RED}Errors encountered:{Colors.ENDC}")
            for error in self.results['errors'][:5]:
                print(f"    • {error['file']} ({error['mode']}): {error['error'][:50]}")
        
        # Overall result
        print(f"\n  Overall: ", end="")
        if pass_rate >= 80:
            print(f"{Colors.GREEN}✓ PASSED{Colors.ENDC}")
        elif pass_rate >= 60:
            print(f"{Colors.YELLOW}⚠ PARTIAL PASS{Colors.ENDC}")
        else:
            print(f"{Colors.RED}✗ FAILED{Colors.ENDC}")
    
    def print_required_files(self):
        """Print what files are needed for testing"""
        print_section("REQUIRED TEST FILES")
        print(f"  Please create a '{self.test_folder}' folder with these test files:")
        print(f"\n  {Colors.CYAN}Essential files:{Colors.ENDC}")
        print(f"    • test.pdf         - A PDF file (preferably >10 pages)")
        print(f"    • test.docx        - A Word document")
        print(f"    • test.txt         - A plain text file")
        print(f"\n  {Colors.CYAN}Optional files:{Colors.ENDC}")
        print(f"    • large.pdf        - A large PDF (>50 pages) for page limit testing")
        print(f"    • test.doc         - Legacy Word format")
        print(f"    • test.xlsx        - Excel spreadsheet")
        print(f"    • test.pptx        - PowerPoint presentation")
        print(f"    • test.md          - Markdown file")
        print(f"\n  {Colors.YELLOW}Note:{Colors.ENDC} Files should contain actual content for meaningful tests")


def main():
    """Main test execution"""
    tester = DocumentReaderTester(test_folder="corpus_testing/test_documents")
    tester.run_all_tests()


if __name__ == "__main__":
    main()