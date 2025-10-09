#!/usr/bin/env python3
"""
Test Script: Investigate ANCA-Associated Vasculitis in Orphanet
================================================================
This script checks how AAV is stored and what the best query strategy is.
"""

import sqlite3
from pathlib import Path
import yaml

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def load_db_path():
    """Load Orphanet DB path from config"""
    config_path = Path('corpus_config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['resources']['disease_orphanet_db']

def test_disease_lookup(db_path: str):
    """Test how different disease names are stored"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Test diseases
    test_terms = [
        'ANCA-associated vasculitis',
        'anca-associated vasculitis',
        'ANCA associated vasculitis',
        'AAV',
        'granulomatosis with polyangiitis',
        'Wegener granulomatosis',
        "Wegener's granulomatosis",
        'eosinophilic granulomatosis with polyangiitis',
        'Churg-Strauss syndrome',
    ]
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}TESTING DISEASE NAME LOOKUPS IN ORPHANET{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    for term in test_terms:
        print(f"{Colors.BOLD}Searching for: '{term}'{Colors.ENDC}")
        print("-" * 80)
        
        # Test 1: Exact match on preferred term
        cursor.execute("""
            SELECT ce.orphacode, ce.entity_type, lr.text_value, lr.text_type
            FROM core_entities ce
            JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
            WHERE lr.text_type = 'preferred_term'
            AND ce.status = 'active'
            AND LOWER(lr.text_value) = LOWER(?)
        """, (term,))
        
        pref_results = cursor.fetchall()
        
        if pref_results:
            print(f"{Colors.GREEN}✓ Found as PREFERRED TERM:{Colors.ENDC}")
            for row in pref_results:
                print(f"  → ORPHA:{row['orphacode']}")
                print(f"     Term: {row['text_value']}")
                print(f"     Type: {row['entity_type']}")
        else:
            print(f"{Colors.YELLOW}✗ NOT found as preferred term{Colors.ENDC}")
        
        # Test 2: Check synonyms
        cursor.execute("""
            SELECT ce.orphacode, ce.entity_type, 
                   lr_syn.text_value as synonym,
                   lr_pref.text_value as preferred_term
            FROM core_entities ce
            JOIN linguistic_representations lr_syn ON ce.entity_id = lr_syn.entity_id
            JOIN linguistic_representations lr_pref ON ce.entity_id = lr_pref.entity_id
            WHERE lr_syn.text_type = 'synonym'
            AND lr_pref.text_type = 'preferred_term'
            AND ce.status = 'active'
            AND LOWER(lr_syn.text_value) = LOWER(?)
        """, (term,))
        
        syn_results = cursor.fetchall()
        
        if syn_results:
            print(f"{Colors.GREEN}✓ Found as SYNONYM:{Colors.ENDC}")
            for row in syn_results:
                print(f"  → ORPHA:{row['orphacode']}")
                print(f"     Synonym: {row['synonym']}")
                print(f"     Preferred: {row['preferred_term']}")
                print(f"     Type: {row['entity_type']}")
        else:
            print(f"{Colors.YELLOW}✗ NOT found as synonym{Colors.ENDC}")
        
        # Test 3: Fuzzy search (LIKE with wildcards)
        cursor.execute("""
            SELECT ce.orphacode, lr.text_value, lr.text_type
            FROM core_entities ce
            JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
            WHERE ce.status = 'active'
            AND LOWER(lr.text_value) LIKE LOWER(?)
            LIMIT 5
        """, (f'%{term}%',))
        
        fuzzy_results = cursor.fetchall()
        
        if fuzzy_results:
            print(f"{Colors.BLUE}➜ Found via FUZZY SEARCH (contains '{term}'):{Colors.ENDC}")
            for row in fuzzy_results:
                print(f"  → ORPHA:{row['orphacode']} - {row['text_value']} [{row['text_type']}]")
        
        print()
    
    conn.close()

def analyze_aav_entity(db_path: str):
    """Deep dive into AAV entity structure"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}DEEP DIVE: ANCA-ASSOCIATED VASCULITIS (ORPHA:156152){Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    # Get the entity
    cursor.execute("""
        SELECT * FROM core_entities
        WHERE orphacode = 156152
    """)
    
    entity = cursor.fetchone()
    
    if not entity:
        print(f"{Colors.RED}✗ Entity ORPHA:156152 not found!{Colors.ENDC}")
        conn.close()
        return
    
    print(f"{Colors.BOLD}Entity Details:{Colors.ENDC}")
    print(f"  Orphacode: {entity['orphacode']}")
    print(f"  Type: {entity['entity_type']}")
    print(f"  Status: {entity['status']}")
    print(f"  Disorder Type: {entity['disorder_type']}")
    print(f"  Classification: {entity['classification_level']}")
    
    # Get all linguistic representations
    print(f"\n{Colors.BOLD}All Linguistic Representations:{Colors.ENDC}")
    cursor.execute("""
        SELECT text_value, text_type, language_code, is_abbreviation
        FROM linguistic_representations
        WHERE entity_id = ?
        ORDER BY text_type, text_value
    """, (entity['entity_id'],))
    
    representations = cursor.fetchall()
    
    by_type = {}
    for rep in representations:
        text_type = rep['text_type']
        if text_type not in by_type:
            by_type[text_type] = []
        by_type[text_type].append(rep)
    
    for text_type, reps in by_type.items():
        print(f"\n  {Colors.CYAN}{text_type.upper()}:{Colors.ENDC} ({len(reps)} entries)")
        for rep in reps:
            abbrev_marker = f" {Colors.YELLOW}[ABBREV]{Colors.ENDC}" if rep['is_abbreviation'] else ""
            print(f"    • {rep['text_value']}{abbrev_marker}")
    
    # Get external mappings
    print(f"\n{Colors.BOLD}External ID Mappings:{Colors.ENDC}")
    cursor.execute("""
        SELECT external_system, external_code
        FROM external_mappings
        WHERE entity_id = ?
        ORDER BY external_system
    """, (entity['entity_id'],))
    
    mappings = cursor.fetchall()
    
    if mappings:
        for mapping in mappings:
            print(f"  • {mapping['external_system']}: {mapping['external_code']}")
    else:
        print(f"  {Colors.YELLOW}(No external mappings found){Colors.ENDC}")
    
    conn.close()

def test_case_sensitivity(db_path: str):
    """Test if case matters in queries"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}CASE SENSITIVITY TEST{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    test_variations = [
        'ANCA-associated vasculitis',
        'anca-associated vasculitis',
        'ANCA-Associated Vasculitis',
        'Anca-Associated Vasculitis',
    ]
    
    for variation in test_variations:
        # Test exact match
        cursor.execute("""
            SELECT ce.orphacode, lr.text_value
            FROM core_entities ce
            JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
            WHERE ce.status = 'active'
            AND lr.text_value = ?
        """, (variation,))
        
        exact_result = cursor.fetchone()
        
        # Test case-insensitive match
        cursor.execute("""
            SELECT ce.orphacode, lr.text_value
            FROM core_entities ce
            JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
            WHERE ce.status = 'active'
            AND LOWER(lr.text_value) = LOWER(?)
        """, (variation,))
        
        insensitive_result = cursor.fetchone()
        
        exact_status = f"{Colors.GREEN}✓{Colors.ENDC}" if exact_result else f"{Colors.RED}✗{Colors.ENDC}"
        insensitive_status = f"{Colors.GREEN}✓{Colors.ENDC}" if insensitive_result else f"{Colors.RED}✗{Colors.ENDC}"
        
        print(f"'{variation}':")
        print(f"  Exact match (case-sensitive):     {exact_status}")
        print(f"  LOWER() match (case-insensitive): {insensitive_status}")
        
        if insensitive_result:
            print(f"    → Found: ORPHA:{insensitive_result['orphacode']} - {insensitive_result['text_value']}")
        print()
    
    conn.close()

def check_all_vasculitis_diseases(db_path: str):
    """Find all vasculitis-related diseases"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}ALL VASCULITIS-RELATED DISEASES{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    cursor.execute("""
        SELECT DISTINCT ce.orphacode, lr.text_value as preferred_term
        FROM core_entities ce
        JOIN linguistic_representations lr ON ce.entity_id = lr.entity_id
        WHERE lr.text_type = 'preferred_term'
        AND ce.status = 'active'
        AND LOWER(lr.text_value) LIKE '%vasculitis%'
        ORDER BY lr.text_value
        LIMIT 20
    """)
    
    results = cursor.fetchall()
    
    print(f"Found {len(results)} vasculitis-related diseases:\n")
    
    for row in results:
        print(f"  • ORPHA:{row['orphacode']:<7} - {row['preferred_term']}")
    
    conn.close()

def main():
    """Run all tests"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*80)
    print(" ORPHANET DATABASE INVESTIGATION: ANCA-ASSOCIATED VASCULITIS")
    print("="*80)
    print(f"{Colors.ENDC}")
    
    db_path = load_db_path()
    print(f"Database: {db_path}\n")
    
    # Run tests
    test_disease_lookup(db_path)
    test_case_sensitivity(db_path)
    analyze_aav_entity(db_path)
    check_all_vasculitis_diseases(db_path)
    
    # Summary and recommendations
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}RECOMMENDATIONS{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Based on the results above:{Colors.ENDC}\n")
    
    print(f"1. If 'ANCA-associated vasculitis' is found as PREFERRED TERM:")
    print(f"   → Your current query should work (use case-insensitive LOWER())")
    print()
    
    print(f"2. If 'ANCA-associated vasculitis' is found as SYNONYM:")
    print(f"   → You MUST add synonym lookup to _query_orphanet_exact()")
    print(f"   → The preferred term might be different (e.g., 'Vasculitis associated with ANCA')")
    print()
    
    print(f"3. If 'ANCA-associated vasculitis' has punctuation variations:")
    print(f"   → Normalize by removing hyphens/punctuation before querying")
    print(f"   → Try both 'ANCA-associated' and 'ANCA associated'")
    print()
    
    print(f"4. Check the linguistic representations section to see:")
    print(f"   → Exact text as stored in database")
    print(f"   → Whether it's marked as abbreviation")
    print(f"   → All available synonyms")
    print()
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print(f"  • Review the output above")
    print(f"  • Update your query based on how AAV is actually stored")
    print(f"  • Test the updated query with test_02.py")
    print()

if __name__ == '__main__':
    main()