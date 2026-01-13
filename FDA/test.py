#!/usr/bin/env python3
"""
Verification Script for FDA Syncer Critical Fixes
Checks that all 3 files have been updated correctly
"""

import os
import sys

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def find_file(filename):
    """Find file in current directory tree"""
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    return None

def check_file_exists(filepath, description):
    """Check if file exists"""
    if os.path.exists(filepath):
        print(f"{GREEN}✓{END} Found: {description}")
        print(f"  Path: {filepath}")
        return True
    else:
        print(f"{RED}✗{END} Missing: {description}")
        print(f"  Expected: {filepath}")
        return False

def check_file_content(filepath, search_strings, description):
    """Check if file contains expected content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing = []
        for search_str in search_strings:
            if search_str not in content:
                missing.append(search_str)
        
        if not missing:
            print(f"{GREEN}✓{END} {description}: All fixes present")
            return True
        else:
            print(f"{RED}✗{END} {description}: Missing fixes:")
            for m in missing:
                print(f"    - {m[:60]}...")
            return False
    except Exception as e:
        print(f"{RED}✗{END} {description}: Error reading file - {e}")
        return False

def compile_file(filepath, description):
    """Try to compile Python file"""
    try:
        import py_compile
        py_compile.compile(filepath, doraise=True)
        print(f"{GREEN}✓{END} {description}: Compiles successfully")
        return True
    except Exception as e:
        print(f"{RED}✗{END} {description}: Compilation error")
        print(f"    {str(e)}")
        return False

def main():
    print("="*70)
    print(f"{BLUE}FDA SYNCER - VERIFICATION SCRIPT{END}")
    print("Checking all 3 critical fixes")
    print("="*70 + "\n")
    
    # Define expected file locations (try both possible structures)
    file_configs = [
        {
            'name': 'syncher_keys.py',
            'possible_paths': ['FDA/syncher_keys.py', 'syncher_keys.py'],
            'description': 'Configuration (Fix #3: Reduce Scope)',
            'required_strings': [
                "'days_back': 90",
                "'max_drugs': 200",
                "OPTIMIZED"
            ]
        },
        {
            'name': 'http_client.py',
            'possible_paths': [
                'FDA/fda_syncher/utils/http_client.py',
                'fda_syncher/utils/http_client.py',
                'utils/http_client.py'
            ],
            'description': 'HTTP Client (Fix #1: Connection Pool)',
            'required_strings': [
                "self.request_count = 0",
                "_recycle_session_if_needed",
                "_adaptive_delay"
            ]
        },
        {
            'name': 'adverse_events.py',
            'possible_paths': [
                'FDA/fda_syncher/downloaders/adverse_events.py',
                'fda_syncher/downloaders/adverse_events.py',
                'downloaders/adverse_events.py'
            ],
            'description': 'Adverse Events (Fix #2: Circuit Breaker)',
            'required_strings': [
                "self.consecutive_errors = 0",
                "_check_circuit_breaker",
                "CIRCUIT BREAKER TRIGGERED"
            ]
        }
    ]
    
    results = []
    file_paths = {}
    
    # Step 1: Find all files
    print(f"{BLUE}STEP 1: Locating files...{END}\n")
    for config in file_configs:
        found = False
        for possible_path in config['possible_paths']:
            if os.path.exists(possible_path):
                check_file_exists(possible_path, config['description'])
                file_paths[config['name']] = possible_path
                found = True
                break
        
        if not found:
            # Try to find it anywhere
            found_path = find_file(config['name'])
            if found_path:
                print(f"{YELLOW}!{END} Found in unexpected location: {config['name']}")
                print(f"  Path: {found_path}")
                file_paths[config['name']] = found_path
            else:
                print(f"{RED}✗{END} Not found: {config['name']}")
                results.append(False)
        print()
    
    # Step 2: Check file contents
    print(f"{BLUE}STEP 2: Checking for fixes...{END}\n")
    for config in file_configs:
        if config['name'] in file_paths:
            result = check_file_content(
                file_paths[config['name']],
                config['required_strings'],
                config['description']
            )
            results.append(result)
        else:
            results.append(False)
        print()
    
    # Step 3: Compile files
    print(f"{BLUE}STEP 3: Compiling files...{END}\n")
    for config in file_configs:
        if config['name'] in file_paths:
            result = compile_file(
                file_paths[config['name']],
                config['description']
            )
            results.append(result)
        else:
            results.append(False)
        print()
    
    # Summary
    print("="*70)
    print(f"{BLUE}VERIFICATION SUMMARY{END}")
    print("="*70)
    
    total_checks = len(results)
    passed = sum(results)
    
    print(f"Total Checks: {total_checks}")
    print(f"{GREEN}Passed: {passed}{END}")
    print(f"{RED}Failed: {total_checks - passed}{END}")
    
    if all(results):
        print(f"\n{GREEN}✓ ALL CHECKS PASSED!{END}")
        print(f"\n{BLUE}Ready to restart sync:{END}")
        print("  cd FDA  # (if not already there)")
        print("  python sync.py")
        print("\n" + "="*70)
        return 0
    else:
        print(f"\n{RED}✗ SOME CHECKS FAILED{END}")
        print("\nPlease review the failed checks above and:")
        print("1. Ensure files are in the correct location")
        print("2. Verify all code changes were applied")
        print("3. Check for syntax errors")
        print("\n" + "="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())