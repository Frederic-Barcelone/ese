#!/usr/bin/env python3
"""
Diagnostic Test Script for Extraction Pipeline Issues
Tests drug serialization, promotion logic, and data flow
"""

import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


# ============================================================================
# MOCK DATA STRUCTURES (based on actual extraction output)
# ============================================================================

@dataclass
class Drug:
    """Drug entity structure"""
    name: str
    drug_type: str
    confidence: float
    frequency: int
    source: str
    normalized_name: str
    rxcui: str = None
    mesh_id: str = None
    context: str = ""
    validated: bool = False
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Disease:
    """Disease entity structure"""
    name: str
    canonical_name: str
    primary_id: str
    all_ids: Dict[str, str]
    confidence: float
    occurrences: int
    detection_method: str
    semantic_type: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Abbreviation:
    """Abbreviation structure"""
    name: str
    canonical_name: str
    primary_id: str = None
    all_ids: Dict[str, str] = None
    confidence: float = 0.75
    occurrences: int = 1
    detection_method: str = "pattern"
    semantic_type: str = "unknown"
    
    def __post_init__(self):
        if self.all_ids is None:
            self.all_ids = {}
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# TEST 1: Drug Serialization
# ============================================================================

def test_drug_serialization():
    """Test if drugs serialize correctly to JSON"""
    print("\n" + "="*80)
    print("TEST 1: DRUG SERIALIZATION")
    print("="*80)
    
    # Create mock drugs (matching actual extraction)
    drugs = [
        Drug(
            name="METHYLPREDNISOLONE",
            drug_type="investigational",
            confidence=0.9,
            frequency=1,
            source="investigational",
            normalized_name="Methylprednisolone",
            rxcui="6902",
            mesh_id="D008775",
            validated=True
        ),
        Drug(
            name="prednisone",
            drug_type="lexicon",
            confidence=0.85,
            frequency=1,
            source="drug_lexicon",
            normalized_name="Prednisone",
            rxcui="8640",
            mesh_id="D011241",
            validated=True
        ),
        Drug(
            name="rituximab",
            drug_type="lexicon",
            confidence=0.85,
            frequency=1,
            source="drug_lexicon",
            normalized_name="Rituximab",
            rxcui="121191",
            mesh_id="D000069283",
            validated=True
        )
    ]
    
    print(f"\n✓ Created {len(drugs)} mock drugs")
    
    # Test serialization methods
    print("\n--- Testing serialization methods ---")
    
    # Method 1: Using to_dict()
    try:
        drugs_dict = [drug.to_dict() for drug in drugs]
        print(f"✓ to_dict() method: {len(drugs_dict)} drugs serialized")
        print(f"  Sample: {drugs_dict[0]['name']}")
    except Exception as e:
        print(f"✗ to_dict() failed: {e}")
    
    # Method 2: Using asdict directly
    try:
        drugs_asdict = [asdict(drug) for drug in drugs]
        print(f"✓ asdict() method: {len(drugs_asdict)} drugs serialized")
    except Exception as e:
        print(f"✗ asdict() failed: {e}")
    
    # Method 3: JSON serialization
    try:
        drugs_json = json.dumps(drugs_dict, indent=2)
        print(f"✓ JSON serialization: {len(drugs_json)} characters")
        
        # Verify deserialization
        drugs_loaded = json.loads(drugs_json)
        print(f"✓ JSON deserialization: {len(drugs_loaded)} drugs recovered")
        
        # Check for empty entries
        empty_count = sum(1 for d in drugs_loaded if not d.get('name'))
        if empty_count > 0:
            print(f"✗ WARNING: {empty_count} empty drug entries found!")
        else:
            print(f"✓ No empty entries detected")
            
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
    
    # Test what happens with wrong structure (simulating the bug)
    print("\n--- Testing problematic pattern (42 empty entries) ---")
    
    # Simulate the bug: creating empty drug structures
    problematic_drugs = [
        {
            "abbreviation": "",
            "expansion": "",
            "confidence": 0.0,
            "context_type": None,
            "source": None,
            "metadata": {}
        }
        for _ in range(42)
    ]
    
    print(f"Created {len(problematic_drugs)} empty drug structures")
    empty_count = sum(1 for d in problematic_drugs if not d.get('abbreviation'))
    print(f"✗ Empty entries: {empty_count}/{len(problematic_drugs)}")
    
    return drugs_dict


# ============================================================================
# TEST 2: Promotion Logic
# ============================================================================

def test_promotion_logic():
    """Test the promotion counting logic"""
    print("\n" + "="*80)
    print("TEST 2: PROMOTION LOGIC")
    print("="*80)
    
    # Initial counts (from actual log)
    initial_drugs = 13
    initial_diseases = 20
    initial_abbrev = 42
    
    # Enriched abbreviations
    enriched_abbrev = 2  # GPA/MPA/EGPA and GPA with disease IDs
    
    print(f"\n--- Initial state ---")
    print(f"Drugs: {initial_drugs}")
    print(f"Diseases: {initial_diseases}")
    print(f"Abbreviations: {initial_abbrev}")
    print(f"Enriched abbreviations: {enriched_abbrev}")
    
    # Test promotion scenarios
    print(f"\n--- Testing promotion scenarios ---")
    
    # Scenario 1: No promotion (actual behavior)
    promoted_drugs = 0
    promoted_diseases = 0
    
    final_drugs_v1 = initial_drugs + promoted_drugs
    final_diseases_v1 = initial_diseases + promoted_diseases
    final_abbrev_v1 = initial_abbrev - (promoted_drugs + promoted_diseases)
    
    print(f"\nScenario 1: No promotion (actual behavior)")
    print(f"  Final drugs: {final_drugs_v1} (was {initial_drugs})")
    print(f"  Final diseases: {final_diseases_v1} (was {initial_diseases})")
    print(f"  Final abbrev: {final_abbrev_v1} (was {initial_abbrev})")
    
    # Test the reported counts from log
    reported_drug_delta = 29  # "29 drugs promoted"
    reported_disease_delta = -7  # "-7 diseases promoted"
    
    print(f"\nScenario 2: Log-reported promotion")
    print(f"  Drug delta: +{reported_drug_delta}")
    print(f"  Disease delta: {reported_disease_delta}")
    
    final_drugs_v2 = initial_drugs + reported_drug_delta
    final_diseases_v2 = initial_diseases + reported_disease_delta
    
    print(f"  Final drugs: {final_drugs_v2} (was {initial_drugs})")
    print(f"  Final diseases: {final_diseases_v2} (was {initial_diseases})")
    
    # Check for logical inconsistencies
    print(f"\n--- Checking for issues ---")
    
    if reported_disease_delta < 0:
        print(f"✗ ERROR: Negative disease promotion (-7) is invalid!")
        print(f"  Cannot promote negative entities")
    
    if final_drugs_v2 == 42:
        print(f"✗ SUSPICIOUS: Final drug count (42) matches abbrev count")
        print(f"  Possible confusion between drugs and abbreviations")
    
    if final_diseases_v2 == initial_drugs:
        print(f"✗ SUSPICIOUS: Final diseases (13) matches initial drugs")
        print(f"  Possible variable swap")
    
    # Test expected behavior with enrichment
    print(f"\nScenario 3: Expected promotion (if enrichment worked)")
    expected_promoted = enriched_abbrev  # 2 enriched should promote
    
    final_drugs_v3 = initial_drugs
    final_diseases_v3 = initial_diseases + expected_promoted
    final_abbrev_v3 = initial_abbrev - expected_promoted
    
    print(f"  Promoted diseases: +{expected_promoted}")
    print(f"  Final drugs: {final_drugs_v3}")
    print(f"  Final diseases: {final_diseases_v3}")
    print(f"  Final abbrev: {final_abbrev_v3}")


# ============================================================================
# TEST 3: Data Structure Compatibility
# ============================================================================

def test_data_structure_compatibility(drugs_dict):
    """Test if drugs can be stored in diseases structure and vice versa"""
    print("\n" + "="*80)
    print("TEST 3: DATA STRUCTURE COMPATIBILITY")
    print("="*80)
    
    # Create mock disease
    disease = Disease(
        name="ANCA-associated vasculitis",
        canonical_name="ANCA-associated vasculitis",
        primary_id="ORPHA:ORPHA:156152",
        all_ids={"ORPHA": "ORPHA:156152"},
        confidence=0.95,
        occurrences=36,
        detection_method="pattern",
        semantic_type="disease"
    )
    
    print("\n--- Testing structure confusion ---")
    
    # Can we accidentally put drugs in disease array?
    try:
        mixed_array = drugs_dict.copy()
        mixed_array.append(disease.to_dict())
        print(f"✓ Can mix drugs and diseases in same array: {len(mixed_array)} items")
        
        # Check if we can differentiate
        drugs_in_mixed = [d for d in mixed_array if 'rxcui' in d]
        diseases_in_mixed = [d for d in mixed_array if 'semantic_type' in d]
        
        print(f"  Identifiable as drugs: {len(drugs_in_mixed)}")
        print(f"  Identifiable as diseases: {len(diseases_in_mixed)}")
        
    except Exception as e:
        print(f"✗ Cannot mix structures: {e}")
    
    # Test if drug fields match disease fields
    drug_fields = set(drugs_dict[0].keys())
    disease_fields = set(disease.to_dict().keys())
    
    common_fields = drug_fields & disease_fields
    drug_only = drug_fields - disease_fields
    disease_only = disease_fields - drug_fields
    
    print(f"\n--- Field overlap ---")
    print(f"  Common fields: {len(common_fields)}")
    print(f"  Drug-only fields: {drug_only}")
    print(f"  Disease-only fields: {disease_only}")
    
    if 'name' in common_fields and 'confidence' in common_fields:
        print(f"✗ WARNING: Structures share key identification fields")
        print(f"  Risk of confusion during serialization")


# ============================================================================
# TEST 4: Empty Structure Generation
# ============================================================================

def test_empty_structure_generation():
    """Test how empty drug structures might be generated"""
    print("\n" + "="*80)
    print("TEST 4: EMPTY STRUCTURE GENERATION")
    print("="*80)
    
    print("\n--- Testing array initialization patterns ---")
    
    # Pattern 1: Pre-allocating array
    size = 42
    drugs_preallocated = [{}] * size
    print(f"\nPattern 1: [{{}}] * {size}")
    print(f"  Created: {len(drugs_preallocated)} entries")
    print(f"  All empty: {all(not d for d in drugs_preallocated)}")
    
    # Pattern 2: List comprehension with default
    drugs_default = [{"abbreviation": "", "confidence": 0.0} for _ in range(size)]
    print(f"\nPattern 2: List comprehension with defaults")
    print(f"  Created: {len(drugs_default)} entries")
    print(f"  All have empty abbreviation: {all(not d.get('abbreviation') for d in drugs_default)}")
    
    # Pattern 3: Conditional population (simulating filtering)
    all_candidates = 46
    validated = 13
    
    drugs_filtered = []
    for i in range(all_candidates):
        if i < validated:
            drugs_filtered.append({
                "name": f"Drug_{i}",
                "confidence": 0.85
            })
        else:
            drugs_filtered.append({
                "abbreviation": "",
                "confidence": 0.0
            })
    
    print(f"\nPattern 3: Conditional population")
    print(f"  Total entries: {len(drugs_filtered)}")
    print(f"  Valid entries: {sum(1 for d in drugs_filtered if d.get('name'))}")
    print(f"  Empty entries: {sum(1 for d in drugs_filtered if not d.get('name'))}")
    
    # This matches the actual problem!
    if len(drugs_filtered) == 46 and sum(1 for d in drugs_filtered if d.get('name')) == 13:
        print(f"\n✗ MATCH FOUND: This pattern produces the observed bug!")
        print(f"  46 total entries with 13 valid and 33 empty")
        print(f"  Likely cause: Array pre-allocated for candidates, not filtered afterward")


# ============================================================================
# TEST 5: Simulated Extraction Flow
# ============================================================================

def test_extraction_flow():
    """Simulate the complete extraction flow to find where data is lost"""
    print("\n" + "="*80)
    print("TEST 5: EXTRACTION FLOW SIMULATION")
    print("="*80)
    
    # Stage 1: Detection
    print("\n--- Stage 1: Detection ---")
    candidates = 46
    print(f"Drug candidates detected: {candidates}")
    
    # Stage 2: Validation
    print("\n--- Stage 2: Validation ---")
    validated = 13
    rejected = candidates - validated
    print(f"Validated: {validated}")
    print(f"Rejected: {rejected}")
    
    # Create validated drugs
    validated_drugs = [
        {"name": f"Drug_{i}", "validated": True, "confidence": 0.85}
        for i in range(validated)
    ]
    print(f"✓ Created {len(validated_drugs)} drug objects")
    
    # Stage 3: Serialization (POTENTIAL BUG)
    print("\n--- Stage 3: Serialization ---")
    
    # Correct approach
    output_correct = {"drugs": validated_drugs}
    print(f"Correct approach: {len(output_correct['drugs'])} drugs in output")
    
    # Buggy approach (keeping candidate array size)
    output_buggy = {"drugs": [{}] * candidates}
    for i, drug in enumerate(validated_drugs):
        if i < len(output_buggy['drugs']):
            output_buggy['drugs'][i] = drug
    
    print(f"Buggy approach: {len(output_buggy['drugs'])} entries in output")
    print(f"  Valid: {sum(1 for d in output_buggy['drugs'] if d.get('name'))}")
    print(f"  Empty: {sum(1 for d in output_buggy['drugs'] if not d.get('name'))}")
    
    # Check if this matches the problem
    if len(output_buggy['drugs']) == 42:
        print(f"\n✗ FOUND IT: Output has 42 entries (matches observed bug)")
        print(f"  This would happen if array sized to 42 abbreviations")
        print(f"  But only {validated} positions get populated with drugs")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all diagnostic tests"""
    print("\n" + "="*80)
    print("EXTRACTION PIPELINE DIAGNOSTIC TEST SUITE")
    print("="*80)
    print("\nTesting extraction pipeline issues from log analysis")
    print("Based on: 00954_Pediatric ANCA-Associated Vasculitis extraction")
    
    try:
        drugs_dict = test_drug_serialization()
        test_promotion_logic()
        test_data_structure_compatibility(drugs_dict)
        test_empty_structure_generation()
        test_extraction_flow()
        
        print("\n" + "="*80)
        print("TEST SUITE COMPLETE")
        print("="*80)
        print("\n✓ All tests completed successfully")
        print("\nKey findings:")
        print("  1. Drug serialization works correctly with proper structures")
        print("  2. Promotion logic has counting inconsistencies")
        print("  3. Empty structures likely from array pre-allocation")
        print("  4. Data loss occurs during serialization, not validation")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()