#!/usr/bin/env python3
# corpus_metadata/corpus_abbreviations/F_evaluation/F04_generator_unit_test.py
"""
Generator Unit Test

Tests abbreviation extraction patterns using synthetic sentences built from gold pairs.
No PDF parsing required—validates regex logic in isolation.

Process:
    1. Load gold SF/LF pairs from JSON
    2. Generate 4 synthetic sentences per pair (different patterns)
    3. Run simplified extractor matching C01_strategy_abbrev.py logic
    4. Report recall by pattern type

Patterns tested:
    - LF (SF): "Tumor Necrosis Factor (TNF)"
    - SF (LF): "TNF (Tumor Necrosis Factor)"
    - Implicit: "TNF, defined as Tumor Necrosis Factor"
    - End position: "diagnosed with Tumor Necrosis Factor (TNF)"

Output: Per-pattern recall and overall recall with pass/fail threshold (≥90%).

Usage:
    python F04_generator_unit_test.py

Requires: Gold JSON from F03_process_nlp4rare.py

"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

GOLD_JSON = "/Users/frederictetard/Projects/ese/gold_data/nlp4rare_gold.json"

# =============================================================================
# SYNTHETIC SENTENCE GENERATOR
# =============================================================================

def generate_test_sentences(sf: str, lf: str) -> List[str]:
    """
    Generate synthetic sentences containing SF/LF pairs.
    Tests different patterns your generators should catch.
    """
    return [
        # Pattern A: LF (SF) - Schwartz-Hearst standard
        f"{lf} ({sf}) is a medical condition that affects patients.",
        
        # Pattern B: SF (LF) - Reverse explicit
        f"{sf} ({lf}) is commonly seen in clinical practice.",
        
        # Pattern C: Implicit phrasing
        f"{sf}, defined as {lf}, requires careful diagnosis.",
        
        # Pattern D: Abbreviation at end
        f"The patient was diagnosed with {lf} ({sf}).",
    ]


# =============================================================================
# SIMPLE GENERATOR (mirrors C01 logic)
# =============================================================================

import re

def extract_abbreviations_simple(text: str) -> List[Tuple[str, str]]:
    """
    Simplified extraction matching C01_strategy_abbrev.py patterns.
    Returns list of (SF, LF) tuples found.
    """
    results = []

    # Character classes for medical abbreviations
    # SF: uppercase letters, digits, hyphens, slashes, spaces
    SF_CHARS = r"A-Z0-9\-/\s"
    # LF: letters (including accented), digits, spaces, hyphens, apostrophes, commas, slashes
    LF_CHARS = r"a-zA-ZÀ-ÿ0-9\s\-',/"

    # Pattern A: LF (SF) - long form followed by short in parens
    # e.g., "Acanthosis nigricans (AN)" or "myelopathy (HAM/TSP)"
    pattern_a = re.compile(rf'([A-Za-zÀ-ÿ][{LF_CHARS}]{{3,120}})\s*\(([A-Z0-9][{SF_CHARS}]{{1,40}})\)', re.IGNORECASE)
    for m in pattern_a.finditer(text):
        lf = m.group(1).strip()
        sf = m.group(2).strip()
        if len(sf) >= 2 and len(lf) > len(sf):
            results.append((sf.upper(), lf))

    # Pattern B: SF (LF) - short form followed by long in parens
    # e.g., "HAM/TSP (HTLV-I associated myelopathy)" or "NHL (non-Hodgkin's lymphoma)"
    pattern_b = re.compile(rf'\b([A-Z0-9][{SF_CHARS}]{{1,40}})\s*\(([A-Za-zÀ-ÿ][{LF_CHARS}]{{3,120}})\)', re.IGNORECASE)
    for m in pattern_b.finditer(text):
        sf = m.group(1).strip()
        lf = m.group(2).strip()
        if len(sf) >= 2 and len(lf) > len(sf):
            results.append((sf.upper(), lf))

    # Pattern C: SF, defined as LF
    # e.g., "HAM/TSP, defined as HTLV-I associated myelopathy"
    pattern_c = re.compile(rf'\b([A-Z0-9][{SF_CHARS}]{{1,40}})\s*,\s*defined as\s+([A-Za-zÀ-ÿ][{LF_CHARS}]{{3,120}})', re.IGNORECASE)
    for m in pattern_c.finditer(text):
        sf = m.group(1).strip()
        lf = m.group(2).strip()
        if len(sf) >= 2:
            results.append((sf.upper(), lf))

    return results


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_unit_test(gold_path: str) -> Dict:
    """
    Run generator unit test on all gold pairs.
    """
    # Load gold
    with open(gold_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    print(f"Loaded {len(annotations)} gold pairs\n")
    
    # Deduplicate (same SF/LF can appear in multiple docs)
    unique_pairs: Set[Tuple[str, str]] = set()
    for a in annotations:
        sf = a['short_form'].upper()
        lf = a['long_form']
        unique_pairs.add((sf, lf))
    
    print(f"Unique pairs: {len(unique_pairs)}\n")
    
    # Test each pair
    results = {
        'total_pairs': len(unique_pairs),
        'total_tests': 0,
        'pattern_results': {
            'LF (SF)': {'tp': 0, 'fn': 0},
            'SF (LF)': {'tp': 0, 'fn': 0},
            'defined as': {'tp': 0, 'fn': 0},
            'end position': {'tp': 0, 'fn': 0},
        },
        'failures': [],
    }
    
    pattern_names = ['LF (SF)', 'SF (LF)', 'defined as', 'end position']
    
    for sf, lf in sorted(unique_pairs):
        sentences = generate_test_sentences(sf, lf)
        
        for i, sentence in enumerate(sentences):
            results['total_tests'] += 1
            pattern = pattern_names[i]
            
            # Run extraction
            found = extract_abbreviations_simple(sentence)
            
            # Check if we found the expected pair
            found_sfs = {f[0] for f in found}
            
            if sf in found_sfs:
                results['pattern_results'][pattern]['tp'] += 1
            else:
                results['pattern_results'][pattern]['fn'] += 1
                if len(results['failures']) < 10:  # Keep first 10 failures
                    results['failures'].append({
                        'expected_sf': sf,
                        'expected_lf': lf,
                        'pattern': pattern,
                        'sentence': sentence[:80] + '...',
                        'found': found,
                    })
    
    return results


def print_results(results: Dict):
    """Print test results."""
    print("=" * 60)
    print("GENERATOR UNIT TEST RESULTS")
    print("=" * 60)
    
    print(f"\nTotal unique pairs: {results['total_pairs']}")
    print(f"Total test cases:   {results['total_tests']}")
    
    print(f"\n{'Pattern':<15} {'TP':>6} {'FN':>6} {'Recall':>10}")
    print("-" * 40)
    
    total_tp = 0
    total_fn = 0
    
    for pattern, counts in results['pattern_results'].items():
        tp = counts['tp']
        fn = counts['fn']
        total = tp + fn
        recall = tp / total if total > 0 else 0
        print(f"{pattern:<15} {tp:>6} {fn:>6} {recall:>9.1%}")
        total_tp += tp
        total_fn += fn
    
    print("-" * 40)
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"{'OVERALL':<15} {total_tp:>6} {total_fn:>6} {overall_recall:>9.1%}")
    
    if results['failures']:
        print(f"\n⚠️  Sample failures (first {len(results['failures'])}):")
        for f in results['failures'][:5]:
            print(f"  {f['expected_sf']} → {f['expected_lf'][:30]}...")
            print(f"    Pattern: {f['pattern']}")
            print(f"    Found: {f['found']}")
    
    print("\n" + "=" * 60)
    
    # Interpretation
    if overall_recall >= 0.90:
        print("✅ PASS - Generators working correctly (≥90% recall)")
    elif overall_recall >= 0.70:
        print("⚠️  WARNING - Some patterns not captured (70-90%)")
    else:
        print("❌ FAIL - Generator logic needs review (<70%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not Path(GOLD_JSON).exists():
        print(f"❌ Gold file not found: {GOLD_JSON}")
        print("Run F03_process_nlp4rare.py first")
        return
    
    results = run_unit_test(GOLD_JSON)
    print_results(results)


if __name__ == "__main__":
    main()