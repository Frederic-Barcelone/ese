#!/usr/bin/env python3
"""
Verification Script for Entity Extraction Fix
==============================================
Run this after applying the fix to verify:
1. Variable order is correct in code
2. JSON output has drugs and diseases in correct arrays
3. Promotion counts are logical
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


class FixVerifier:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def check_code_fix(self, file_path: Path) -> bool:
        """Verify the code has correct variable order"""
        print("\n" + "="*80)
        print("CHECK 1: CODE VARIABLE ORDER")
        print("="*80)
        
        try:
            content = file_path.read_text()
            
            # Check for correct pattern
            import re
            correct_pattern = r'kept_abbreviations,\s*promoted_drugs,\s*promoted_diseases,\s*links\s*=\s*process_abbreviation_candidates'
            bug_pattern = r'promoted_drugs,\s*promoted_diseases,\s*kept_abbreviations,\s*links\s*=\s*process_abbreviation_candidates'
            
            has_correct = bool(re.search(correct_pattern, content))
            has_bug = bool(re.search(bug_pattern, content))
            
            if has_correct and not has_bug:
                print("✓ PASS: Variable order is correct")
                print("  Pattern: kept_abbreviations, promoted_drugs, promoted_diseases, links")
                self.checks_passed += 1
                return True
            elif has_bug:
                print("✗ FAIL: Bug pattern still present!")
                print("  Pattern: promoted_drugs, promoted_diseases, kept_abbreviations, links")
                self.checks_failed += 1
                return False
            else:
                print("? WARNING: Could not find expected pattern")
                self.warnings.append("Variable assignment pattern not found")
                return False
                
        except Exception as e:
            print(f"✗ FAIL: Error reading code file: {e}")
            self.checks_failed += 1
            return False
    
    def check_json_structure(self, json_path: Path) -> Dict[str, Any]:
        """Verify JSON output structure"""
        print("\n" + "="*80)
        print("CHECK 2: JSON OUTPUT STRUCTURE")
        print("="*80)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Find entities stage
            entities_stage = None
            for stage in data.get('pipeline_stages', []):
                if stage.get('stage') == 'entities':
                    entities_stage = stage
                    break
            
            if not entities_stage:
                print("✗ FAIL: Could not find entities stage in JSON")
                self.checks_failed += 1
                return {}
            
            results = entities_stage.get('results', {})
            drugs = results.get('drugs', [])
            diseases = results.get('diseases', [])
            
            print(f"\nFound {len(drugs)} drugs, {len(diseases)} diseases")
            
            # Analyze drugs array
            print("\n--- Drugs Array ---")
            valid_drugs = [d for d in drugs if d.get('name') or d.get('drug_name')]
            empty_drugs = [d for d in drugs if not (d.get('name') or d.get('drug_name'))]
            abbrev_structure_drugs = [d for d in drugs if d.get('abbreviation') == '']
            
            print(f"  Total entries: {len(drugs)}")
            print(f"  Valid drug entries: {len(valid_drugs)}")
            print(f"  Empty entries: {len(empty_drugs)}")
            print(f"  Abbreviation structures: {len(abbrev_structure_drugs)}")
            
            # Sample drugs
            if valid_drugs:
                sample = valid_drugs[0]
                print(f"\n  Sample drug entry:")
                print(f"    Name: {sample.get('name', sample.get('drug_name', 'N/A'))}")
                print(f"    Keys: {', '.join(list(sample.keys())[:5])}...")
            
            # Analyze diseases array
            print("\n--- Diseases Array ---")
            valid_diseases = [d for d in diseases if d.get('name') or d.get('disease_name')]
            
            print(f"  Total entries: {len(diseases)}")
            print(f"  Valid disease entries: {len(valid_diseases)}")
            
            # Sample diseases
            if valid_diseases:
                sample = valid_diseases[0]
                print(f"\n  Sample disease entry:")
                print(f"    Name: {sample.get('name', sample.get('disease_name', 'N/A'))}")
                print(f"    Keys: {', '.join(list(sample.keys())[:5])}...")
            
            # Check for drug names in disease array (indicates unfixed bug)
            drug_keywords = ['prednisone', 'methylprednisolone', 'rituximab', 'cyclophosphamide']
            drugs_in_disease_array = []
            
            for disease in diseases:
                name = (disease.get('name') or disease.get('disease_name', '')).lower()
                if any(keyword in name for keyword in drug_keywords):
                    drugs_in_disease_array.append(disease.get('name', disease.get('disease_name')))
            
            # Determine pass/fail
            checks_passed = True
            
            if len(valid_drugs) == 0 and len(abbrev_structure_drugs) > 0:
                print("\n✗ FAIL: Drugs array contains abbreviation structures (bug not fixed)")
                checks_passed = False
            elif drugs_in_disease_array:
                print(f"\n✗ FAIL: Disease array contains drugs: {drugs_in_disease_array}")
                checks_passed = False
            elif len(valid_drugs) > 0 and len(valid_diseases) > 0:
                print("\n✓ PASS: Both arrays contain appropriate entities")
            else:
                print("\n? WARNING: Arrays may be empty or structured unexpectedly")
                self.warnings.append("Unexpected array structure")
            
            if checks_passed:
                self.checks_passed += 1
            else:
                self.checks_failed += 1
            
            return results
            
        except Exception as e:
            print(f"✗ FAIL: Error reading JSON: {e}")
            self.checks_failed += 1
            return {}
    
    def check_promotion_logic(self, json_path: Path) -> bool:
        """Verify promotion counts are logical"""
        print("\n" + "="*80)
        print("CHECK 3: PROMOTION LOGIC")
        print("="*80)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Find entities stage
            for stage in data.get('pipeline_stages', []):
                if stage.get('stage') == 'entities':
                    summary = stage.get('results', {}).get('extraction_summary', {})
                    break
            else:
                print("✗ FAIL: Could not find extraction summary")
                self.checks_failed += 1
                return False
            
            drugs_direct = summary.get('drugs_direct', 0)
            drugs_promoted = summary.get('drugs_promoted', 0)
            drugs_total = summary.get('drugs_total', 0)
            
            diseases_direct = summary.get('diseases_direct', 0)
            diseases_promoted = summary.get('diseases_promoted', 0)
            diseases_total = summary.get('diseases_total', 0)
            
            print("\n--- Drugs ---")
            print(f"  Direct: {drugs_direct}")
            print(f"  Promoted: {drugs_promoted}")
            print(f"  Total: {drugs_total}")
            print(f"  Formula: {drugs_direct} + {drugs_promoted} = {drugs_direct + drugs_promoted}")
            
            print("\n--- Diseases ---")
            print(f"  Direct: {diseases_direct}")
            print(f"  Promoted: {diseases_promoted}")
            print(f"  Total: {diseases_total}")
            print(f"  Formula: {diseases_direct} + {diseases_promoted} = {diseases_direct + diseases_promoted}")
            
            # Check logic
            issues = []
            
            if drugs_promoted < 0:
                issues.append("Negative drug promotion count")
            
            if diseases_promoted < 0:
                issues.append("Negative disease promotion count")
            
            if drugs_direct + drugs_promoted != drugs_total:
                issues.append(f"Drug formula mismatch: {drugs_direct} + {drugs_promoted} ≠ {drugs_total}")
            
            if diseases_direct + diseases_promoted != diseases_total:
                issues.append(f"Disease formula mismatch: {diseases_direct} + {diseases_promoted} ≠ {diseases_total}")
            
            if drugs_total == 42 and diseases_total == 13:
                issues.append("Suspicious counts (42/13) suggest variables may still be swapped")
            
            if issues:
                print("\n✗ FAIL: Promotion logic issues:")
                for issue in issues:
                    print(f"  - {issue}")
                self.checks_failed += 1
                return False
            else:
                print("\n✓ PASS: Promotion counts are logical")
                self.checks_passed += 1
                return True
                
        except Exception as e:
            print(f"✗ FAIL: Error checking promotion logic: {e}")
            self.checks_failed += 1
            return False
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        total_checks = self.checks_passed + self.checks_failed
        
        print(f"\nTotal checks: {total_checks}")
        print(f"✓ Passed: {self.checks_passed}")
        print(f"✗ Failed: {self.checks_failed}")
        
        if self.warnings:
            print(f"\n⚠ Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        print()
        
        if self.checks_failed == 0:
            print("🎉 ALL CHECKS PASSED! Bug is fixed!")
            return True
        else:
            print("❌ SOME CHECKS FAILED - Bug may not be fully fixed")
            return False


def main():
    """Main verification"""
    print("\n" + "="*80)
    print("ENTITY EXTRACTION FIX VERIFICATION")
    print("="*80)
    
    # Get file paths
    if len(sys.argv) > 2:
        code_file = Path(sys.argv[1])
        json_file = Path(sys.argv[2])
    else:
        # Try default paths
        code_file = Path("corpus_metadata/document_utils/entity_extraction.py")
        json_file = Path("documents_sota/00954_Pediatric ANCA-Associated Vasculitis_ Current Evidence and Therapeutic Landscape_extracted.json")
        
        print(f"\nUsing default paths:")
        print(f"  Code: {code_file}")
        print(f"  JSON: {json_file}")
        print("\n(Pass paths as arguments to verify different files)")
    
    # Check files exist
    if not code_file.exists():
        print(f"\n✗ Code file not found: {code_file}")
        sys.exit(1)
    
    if not json_file.exists():
        print(f"\n✗ JSON file not found: {json_file}")
        print("\n⚠ Note: You need to re-run the extraction after fixing the code")
        print("  to generate a new JSON file for verification")
        sys.exit(1)
    
    # Run verification
    verifier = FixVerifier()
    
    verifier.check_code_fix(code_file)
    verifier.check_json_structure(json_file)
    verifier.check_promotion_logic(json_file)
    
    # Print summary
    success = verifier.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()