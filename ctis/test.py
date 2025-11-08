#!/usr/bin/env python3
"""
Analyze CTIS JSON files to find medical condition fields
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def find_all_keys(data, prefix="", max_depth=10, current_depth=0):
    """
    Recursively find all keys in JSON structure with their paths
    """
    results = []
    
    if current_depth > max_depth:
        return results
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            # Record this key with type info
            value_type = type(value).__name__
            if isinstance(value, list):
                value_type = f"list[{len(value)}]"
            elif isinstance(value, dict):
                value_type = f"dict[{len(value)}]"
            
            results.append({
                'path': current_path,
                'key': key,
                'type': value_type,
                'depth': current_depth
            })
            
            # Recurse into nested structures
            if isinstance(value, (dict, list)):
                results.extend(find_all_keys(value, current_path, max_depth, current_depth + 1))
    
    elif isinstance(data, list) and data:
        # For arrays, analyze first item
        if len(data) > 0:
            first_item = data[0]
            array_path = f"{prefix}[0]"
            if isinstance(first_item, (dict, list)):
                results.extend(find_all_keys(first_item, array_path, max_depth, current_depth + 1))
    
    return results

def search_medical_condition_keys(all_keys):
    """
    Filter keys that might contain medical condition information
    """
    search_terms = [
        'medical', 'condition', 'disease', 'illness', 'indication',
        'diagnosis', 'therapeutic', 'disorder', 'syndrome', 'pathology'
    ]
    
    matches = []
    for key_info in all_keys:
        key_lower = key_info['key'].lower()
        path_lower = key_info['path'].lower()
        
        if any(term in key_lower or term in path_lower for term in search_terms):
            matches.append(key_info)
    
    return matches

def extract_value_at_path(data, path_str):
    """
    Extract value from JSON using dot notation path
    """
    try:
        parts = path_str.split('.')
        current = data
        
        for part in parts:
            # Handle array indices like [0]
            if '[' in part and ']' in part:
                key = part.split('[')[0]
                index = int(part.split('[')[1].split(']')[0])
                if key:
                    current = current.get(key, [])[index]
                else:
                    current = current[index]
            else:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            
            if current is None:
                return None
        
        return current
    except:
        return None

def analyze_json_file(filepath):
    """
    Analyze a single JSON file for medical condition data
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*80}\n")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Handle both regular JSON and NDJSON
            if filepath.suffix == '.ndjson':
                # Read first line for NDJSON
                first_line = f.readline()
                if not first_line.strip():
                    print("❌ Empty NDJSON file")
                    return
                data = json.loads(first_line)
                print(f"Note: Analyzing first record from NDJSON file\n")
            else:
                data = json.load(f)
        
        # Get basic info
        print("BASIC INFO")
        print("-" * 80)
        if isinstance(data, dict):
            ct_number = data.get('ctNumber') or data.get('_id') or 'Unknown'
            print(f"CT Number: {ct_number}")
            print(f"Top-level keys: {len(data.keys())}")
            print(f"Keys: {', '.join(list(data.keys())[:10])}{'...' if len(data.keys()) > 10 else ''}")
        print()
        
        # Find all keys
        print("SEARCHING FOR MEDICAL CONDITION FIELDS...")
        print("-" * 80)
        all_keys = find_all_keys(data, max_depth=8)
        print(f"Total keys found: {len(all_keys)}")
        
        # Search for medical condition related keys
        medical_keys = search_medical_condition_keys(all_keys)
        print(f"Medical-related keys found: {len(medical_keys)}\n")
        
        if medical_keys:
            print("MEDICAL CONDITION RELATED FIELDS")
            print("-" * 80)
            
            for i, key_info in enumerate(medical_keys[:30], 1):  # Show first 30
                print(f"\n{i}. {key_info['path']}")
                print(f"   Type: {key_info['type']}")
                print(f"   Depth: {key_info['depth']}")
                
                # Try to extract and show value
                value = extract_value_at_path(data, key_info['path'])
                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        print(f"   Value: {value}")
                    elif isinstance(value, dict):
                        print(f"   Value (dict): {json.dumps(value, ensure_ascii=False)[:200]}...")
                    elif isinstance(value, list):
                        if len(value) > 0 and isinstance(value[0], str):
                            print(f"   Value (list): {', '.join(value[:3])}{'...' if len(value) > 3 else ''}")
                        else:
                            print(f"   Value (list): {len(value)} items")
            
            if len(medical_keys) > 30:
                print(f"\n... and {len(medical_keys) - 30} more medical-related fields")
        else:
            print("❌ No medical condition related fields found")
        
        # Check specific known paths
        print("\n\nCHECKING SPECIFIC KNOWN PATHS")
        print("-" * 80)
        
        known_paths = [
            'medicalCondition',
            'medicalConditions',
            'authorizedApplication.authorizedPartI.medicalCondition',
            'authorizedApplication.authorizedPartI.medicalConditions',
            'authorizedApplication.authorizedPartI.trialDetails.medicalCondition',
            'authorizedApplication.authorizedPartI.trialDetails.medicalConditions',
            'authorisedApplication.authorisedPartI.medicalCondition',
            'therapeuticAreas',
            'authorizedApplication.authorizedPartI.therapeuticAreas',
        ]
        
        for path in known_paths:
            value = extract_value_at_path(data, path)
            if value is not None:
                print(f"\n✓ FOUND: {path}")
                if isinstance(value, str):
                    print(f"  Value: {value}")
                elif isinstance(value, list):
                    print(f"  Type: list with {len(value)} items")
                    if len(value) > 0:
                        print(f"  First item: {json.dumps(value[0], ensure_ascii=False)[:300]}")
                elif isinstance(value, dict):
                    print(f"  Type: dict with keys: {', '.join(value.keys())}")
                    print(f"  Value: {json.dumps(value, ensure_ascii=False)[:300]}...")
            else:
                print(f"✗ Not found: {path}")
        
        # Output full structure for reference
        print("\n\nTOP-LEVEL STRUCTURE")
        print("-" * 80)
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                value = data[key]
                if isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*80)
    print("CTIS JSON MEDICAL CONDITION ANALYZER")
    print("="*80)
    
    # Check for JSON files
    possible_locations = [
        Path('/home/claude/ctis-out/ctis_full.ndjson'),
        Path('/home/claude/ctis-out/*.json'),
        Path('/home/claude/*.json'),
        Path('/home/claude/*.ndjson'),
        Path('ctis-out/ctis_full.ndjson'),
        Path('ctis_full.ndjson'),
    ]
    
    # Get file path from command line or search
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
        if not json_file.exists():
            print(f"❌ File not found: {json_file}")
            return 1
        analyze_json_file(json_file)
    else:
        print("\nSearching for JSON files...")
        print("-" * 80)
        
        found_files = []
        for pattern in possible_locations:
            if '*' in str(pattern):
                import glob
                found_files.extend([Path(f) for f in glob.glob(str(pattern))])
            elif pattern.exists():
                found_files.append(pattern)
        
        if not found_files:
            print("❌ No JSON files found")
            print("\nUsage:")
            print(f"  python3 {sys.argv[0]} <path-to-json-file>")
            print("\nOr place your JSON file in one of these locations:")
            for loc in possible_locations:
                print(f"  - {loc}")
            return 1
        
        print(f"Found {len(found_files)} file(s):\n")
        for i, f in enumerate(found_files, 1):
            print(f"  {i}. {f}")
        
        # Analyze first file
        print(f"\nAnalyzing first file: {found_files[0]}")
        analyze_json_file(found_files[0])
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())