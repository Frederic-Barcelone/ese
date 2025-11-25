#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS List Values Extractor
Extracts all controlled vocabularies from CTIS Excel file into JSON format

================================================================================
HOW TO USE THIS SCRIPT
================================================================================

OVERVIEW:
---------
This script extracts all 45 controlled vocabulary lists from the official EMA 
CTIS (Clinical Trial Information System) Excel file and creates JSON files 
that you can use to decode CTIS database codes into human-readable names.

WHAT IT DOES:
-------------
1. Reads the CTIS list values Excel file
2. Extracts all 45 reference lists (therapeutic areas, phases, status, etc.)
3. Creates two JSON files:
   - CTIS_List_Values_All.json: Complete data (all 45 lists)
   - CTIS_Key_Mappings.json: 6 most important mappings for quick access

KEY MAPPINGS CREATED:
---------------------
• therapeutic_areas (58 items) - MeSH disease/process categories [C], [E], [F], [G], [N]
• age_ranges (4 items)         - In utero, 0-17 years, 18-64 years, 65+ years
• trial_categories (3 items)   - Category 1, 2, 3
• trial_phases (11 items)      - Phase I, II, III, IV, Integrated phases
• trial_status (13 items)      - Pending, Authorised, Ended, etc.
• eea_countries (30 items)     - European Economic Area member states

REQUIREMENTS:
-------------
• Python 3.7+
• pandas library: pip install pandas openpyxl
• CTIS list values Excel file from EMA

FILE LOCATION:
--------------
Download the Excel file from:
https://www.ema.europa.eu/en/human-regulatory/research-development/clinical-trials-information-system

File name: clinical-trial-information-system-ctis-list-values_en.xlsx

USAGE:
------

Method 1: Excel file in same directory as script
    $ cd /path/to/your/ctis/directory
    $ python ctis_metadata_extract.py

Method 2: Excel file in parent or ctis subdirectory
    $ cd /path/to/your/project
    $ python ctis/ctis_metadata_extract.py

Method 3: Specify Excel file path explicitly
    $ python ctis_metadata_extract.py /path/to/excel/file.xlsx

OUTPUT FILES:
-------------
Two JSON files will be created in the same directory as the Excel file:

1. CTIS_List_Values_All.json (~76 KB)
   - Complete extraction of all 45 lists
   - Each list contains: tab_number, columns, row_count, data
   - Use this when you need access to all controlled vocabularies

2. CTIS_Key_Mappings.json (~6 KB)
   - Quick-access mappings for the 6 most commonly used lists
   - Simple code → name dictionary format
   - Use this for day-to-day coding/decoding

USING THE OUTPUT IN YOUR CODE:
-------------------------------

Example 1: Load and decode a single code
    import json
    
    with open('CTIS_Key_Mappings.json') as f:
        mappings = json.load(f)
    
    # Decode therapeutic area code
    area = mappings['therapeutic_areas']['20']
    # Returns: "Diseases [C] - Immune System Diseases [C20]"
    
    # Decode trial phase
    phase = mappings['trial_phases']['5']
    # Returns: "Therapeutic confirmatory (Phase III)"
    
    # Decode trial status
    status = mappings['trial_status']['7']
    # Returns: "Authorised"

Example 2: Decode database query results
    import json
    import sqlite3
    
    # Load mappings
    with open('CTIS_Key_Mappings.json') as f:
        mappings = json.load(f)
    
    # Query database
    conn = sqlite3.connect('ctis.db')
    cursor = conn.cursor()
    cursor.execute("SELECT ctNumber, therapeuticAreas, trialPhase FROM trials LIMIT 5")
    
    # Decode results
    for ct_number, area_code, phase_code in cursor.fetchall():
        area_name = mappings['therapeutic_areas'].get(str(area_code), 'Unknown')
        phase_name = mappings['trial_phases'].get(str(phase_code), 'Unknown')
        print(f"{ct_number}: {area_name}, {phase_name}")

Example 3: Use in Excel exports
    import pandas as pd
    import json
    
    # Load mappings
    with open('CTIS_Key_Mappings.json') as f:
        mappings = json.load(f)
    
    # Read data from database
    df = pd.read_sql("SELECT * FROM trials", conn)
    
    # Add decoded columns
    df['therapeutic_area_name'] = df['therapeuticAreas'].apply(
        lambda x: mappings['therapeutic_areas'].get(str(x), '')
    )
    df['phase_name'] = df['trialPhase'].apply(
        lambda x: mappings['trial_phases'].get(str(x), '')
    )
    
    # Export to Excel with human-readable names
    df.to_excel('trials_decoded.xlsx', index=False)

Example 4: Update your ctis_config.py
    import json
    from pathlib import Path
    
    # Load mappings at module level
    _mappings_path = Path(__file__).parent / 'CTIS_Key_Mappings.json'
    if _mappings_path.exists():
        with open(_mappings_path) as f:
            _CTIS_MAPPINGS = json.load(f)
    else:
        _CTIS_MAPPINGS = {}
    
    # Create easy-to-use dictionaries
    THERAPEUTIC_AREA_MAP = _CTIS_MAPPINGS.get('therapeutic_areas', {})
    AGE_CATEGORY_MAP = _CTIS_MAPPINGS.get('age_ranges', {})
    PHASE_MAP = _CTIS_MAPPINGS.get('trial_phases', {})
    TRIAL_STATUS_MAP = _CTIS_MAPPINGS.get('trial_status', {})
    TRIAL_CATEGORY_MAP = _CTIS_MAPPINGS.get('trial_categories', {})
    EEA_COUNTRY_MAP = _CTIS_MAPPINGS.get('eea_countries', {})

UNDERSTANDING THE CODES:
------------------------

THERAPEUTIC AREAS (MeSH Classification):
  Format: "Category [Letter] - Specific Area [Letter+Number]"
  
  Main categories:
  • [C]  = Diseases (C01-C23)
  • [E]  = Analytical, Diagnostic & Therapeutic Techniques (E01-E07)
  • [F]  = Psychiatry & Psychology (F01-F04)
  • [G]  = Phenomena & Processes (G01-G17)
  • [N]  = Health Care (N01-N06)
  
  Example: Code 20 → "Diseases [C] - Immune System Diseases [C20]"
           The [C] indicates top-level category "Diseases"
           The [C20] is the specific MeSH tree number

AGE RANGES:
  Code 1 = In utero
  Code 2 = 0-17 years
  Code 3 = 18-64 years
  Code 4 = 65+ years

TRIAL CATEGORIES:
  Code 1 = Category 1 (Phase I trials)
  Code 2 = Category 2 (Phase I/II, II, II/III, III trials)
  Code 3 = Category 3 (Phase III/IV, IV, low intervention trials)

TRIAL PHASES:
  Code 1  = Human Pharmacology (Phase I) - First administration to humans
  Code 4  = Therapeutic exploratory (Phase II)
  Code 5  = Therapeutic confirmatory (Phase III)
  Code 6  = Therapeutic use (Phase IV)
  Code 10 = Phase II and Phase III (Integrated)
  ... (11 total)

TRIAL STATUS:
  Code 7  = Authorised
  Code 12 = Ended
  Code 11 = Suspended
  ... (13 total)

IMPORTANT NOTES:
----------------
1. The script skips empty rows and invalid data automatically
2. Text values like "None" are preserved as-is (not converted to empty strings)
3. The Excel file contains trailing spaces in some reference names (handled automatically)
4. Always use the CODE field for lookups, not the NAME field
5. Some lists have more data in the full JSON than in the key mappings
6. The script is read-only - it doesn't modify the Excel file

TROUBLESHOOTING:
----------------
Error: "File not found"
  → Make sure the Excel file is in the same directory as the script,
    or provide the full path as an argument

Error: "No module named 'pandas'"
  → Install pandas: pip install pandas openpyxl

Error: "No module named 'openpyxl'"
  → Install openpyxl: pip install openpyxl

No output or empty JSON:
  → Check that the Excel file is the correct CTIS list values file
  → Verify the file has the "Reference data index" sheet

Wrong data extracted:
  → Make sure you're using the English version (_en.xlsx)
  → Re-download the Excel file from EMA if it seems corrupted

UPDATING THE MAPPINGS:
---------------------
EMA may update the CTIS list values file periodically. To get the latest:

1. Download the new Excel file from EMA website
2. Run this script again
3. Replace the old JSON files with the new ones
4. No code changes needed - your application will automatically use the new mappings

REFERENCES:
-----------
• CTIS Public Portal: https://euclinicaltrials.eu/
• EMA CTIS Info: https://www.ema.europa.eu/en/human-regulatory/research-development/clinical-trials-information-system
• MeSH Browser: https://meshb.nlm.nih.gov/
• MedDRA: https://www.meddra.org/

VERSION:
--------
Script Version: 1.0
Last Updated: November 2024
Compatible with: CTIS List Values file version November 2024

AUTHOR:
-------
Created for CTIS database integration projects
Based on EMA official controlled vocabularies

================================================================================
"""

import pandas as pd
import json
import sys
from pathlib import Path


def extract_ctis_list_values(excel_file, output_json='CTIS_List_Values.json'):
    """
    Extract all CTIS list values from Excel file
    
    Args:
        excel_file: Path to the CTIS list values Excel file
        output_json: Output JSON file path
    
    Returns:
        dict: All extracted list values
    """
    
    print("="*80)
    print("CTIS LIST VALUES EXTRACTOR")
    print("="*80)
    
    xl_file = pd.ExcelFile(excel_file)
    all_data = {}
    
    # Get reference index to map sheet numbers to names
    print("\n📋 Reading reference index...")
    df_ref_index = pd.read_excel(excel_file, sheet_name="Reference data index ")
    
    # Create mapping: TAB number -> Reference Entity name
    tab_to_name = {}
    for _, row in df_ref_index.iterrows():
        tab_to_name[str(row['TAB'])] = row['Reference Entity']
    
    print(f"   Found {len(tab_to_name)} reference entities\n")
    
    # Process each sheet
    print("📊 Extracting data from sheets...")
    print("-"*80)
    
    for sheet_name in xl_file.sheet_names:
        # Skip overview and index sheets
        if sheet_name in ['Overview', 'Reference data index ']:
            print(f"⏭️  Skipping: {sheet_name}")
            continue
        
        try:
            # Read sheet with header=1 to skip the "HOME" row
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=1)
            
            # Get the reference name
            ref_name = tab_to_name.get(sheet_name, f"Sheet_{sheet_name}")
            
            # Get column names
            columns = df.columns.tolist()
            
            # Convert to list of dicts - keep "None" as text, only replace actual NaN
            data_records = []
            for _, row in df.iterrows():
                record = {}
                for col in columns:
                    value = row[col]
                    # Keep actual None/NaN as empty string, but preserve the text "None"
                    if pd.isna(value):
                        record[col] = ''
                    else:
                        record[col] = value
                
                # Skip if CODE column is empty or not valid
                code_value = str(record.get(columns[0], '')).strip()
                
                # Skip empty rows
                if not code_value:
                    continue
                
                # Skip non-numeric codes (except for specific cases)
                # Keep if it's a number, or if all columns have some content
                if code_value.replace('.', '').replace('-', '').isdigit() or \
                   (len(columns) >= 2 and record.get(columns[1], '').strip()):
                    # Additional check: skip if it looks like a note/header
                    if code_value.upper() not in ['CODE', 'REGULATION', 'DEFINITIONS'] and \
                       not code_value.startswith('Regulation'):
                        data_records.append(record)
            
            # Store in output
            all_data[ref_name] = {
                'tab_number': sheet_name,
                'columns': columns,
                'row_count': len(df),
                'data': data_records
            }
            
            print(f"✓ {ref_name:50s} | Tab {sheet_name:4s} | {len(data_records):3d} rows (filtered from {len(df)})")
            
        except Exception as e:
            print(f"✗ Error processing sheet {sheet_name}: {e}")
    
    print("-"*80)
    print(f"\n✅ Successfully extracted {len(all_data)} lists")
    
    # Save to JSON
    print(f"\n💾 Saving to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to: {output_json}")
    print(f"📦 File size: {Path(output_json).stat().st_size / 1024:.1f} KB")
    
    return all_data


def create_key_mappings(all_data, output_json='CTIS_Key_Mappings.json'):
    """
    Create simplified mappings for the most commonly used lists
    
    Args:
        all_data: Dictionary with all CTIS list values
        output_json: Output JSON file path
    
    Returns:
        dict: Key mappings
    """
    
    print("\n" + "="*80)
    print("CREATING KEY MAPPINGS")
    print("="*80)
    
    def extract_code_name_mapping(ref_name):
        """Extract code->name mapping from a reference list"""
        if ref_name not in all_data:
            return {}
        
        mapping = {}
        items = all_data[ref_name]['data']
        columns = all_data[ref_name]['columns']
        
        if len(columns) < 2:
            return {}
        
        code_col = columns[0]  # Usually 'CODE'
        name_col = columns[1]  # Usually 'NAME'
        
        for item in items:
            code = str(item.get(code_col, '')).strip()
            name = str(item.get(name_col, '')).strip()
            
            # Skip empty or header-like rows
            if code and name and code not in ['CODE', 'code']:
                mapping[code] = name
        
        return mapping
    
    # Extract key mappings - NOTE: Use exact keys from JSON (with trailing spaces!)
    key_mappings = {
        'therapeutic_areas': extract_code_name_mapping('Therapeutic area'),
        'age_ranges': extract_code_name_mapping('Subject Age Range'),
        'trial_categories': extract_code_name_mapping('Trial Category'),
        'trial_phases': extract_code_name_mapping('Clinical trial type (Phase) '),  # Note trailing space!
        'trial_status': extract_code_name_mapping('Overall Trial Status'),
        'eea_countries': extract_code_name_mapping('European Economic Area [EEA] Member States'),
    }
    
    print("\n✅ Key mappings extracted:")
    for mapping_key, mapping in key_mappings.items():
        print(f"   {mapping_key:20s} : {len(mapping):3d} items")
    
    # Save key mappings
    print(f"\n💾 Saving to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(key_mappings, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to: {output_json}")
    print(f"📦 File size: {Path(output_json).stat().st_size / 1024:.1f} KB")
    
    return key_mappings


def main():
    """Main execution"""
    
    # Check for file path argument
    if len(sys.argv) > 1:
        excel_file = Path(sys.argv[1])
    else:
        # Try to find the file in current directory or common locations
        possible_paths = [
            Path('clinical-trial-information-system-ctis-list-values_en.xlsx'),
            Path('ctis/clinical-trial-information-system-ctis-list-values_en.xlsx'),
            Path('../clinical-trial-information-system-ctis-list-values_en.xlsx'),
        ]
        
        excel_file = None
        for path in possible_paths:
            if path.exists():
                excel_file = path
                break
    
    # Check if file exists
    if excel_file is None or not excel_file.exists():
        print(f"❌ Error: CTIS list values Excel file not found!")
        print(f"\nSearched in:")
        for path in possible_paths:
            print(f"   - {path}")
        print(f"\nUsage:")
        print(f"   python {sys.argv[0]} <path-to-excel-file>")
        print(f"\nExample:")
        print(f"   python {sys.argv[0]} ~/Downloads/clinical-trial-information-system-ctis-list-values_en.xlsx")
        return
    
    print(f"📂 Using file: {excel_file}\n")
    
    # Determine output directory (same as input file or current dir)
    output_dir = excel_file.parent if excel_file.parent != Path('.') else Path.cwd()
    
    # Extract all data
    all_data = extract_ctis_list_values(
        str(excel_file), 
        output_json=str(output_dir / 'CTIS_List_Values_All.json')
    )
    
    # Create key mappings
    key_mappings = create_key_mappings(
        all_data,
        output_json=str(output_dir / 'CTIS_Key_Mappings.json')
    )
    
    # Summary
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\n📁 Files created in: {output_dir}/")
    print("   1. CTIS_List_Values_All.json      - Complete data (all 45 lists)")
    print("   2. CTIS_Key_Mappings.json         - 6 key mappings for quick access")
    
    print("\n📊 Summary:")
    print(f"   Total reference lists: {len(all_data)}")
    print(f"   Key mappings:")
    for key, values in key_mappings.items():
        print(f"      - {key:25s} : {len(values):3d} items")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("""
1. The JSON files are now in your directory
2. Use them in your Python code:

   import json
   
   with open('CTIS_Key_Mappings.json') as f:
       mappings = json.load(f)
   
   # Decode a code
   therapeutic_area = mappings['therapeutic_areas']['20']
   # Returns: "Diseases [C] - Immune System Diseases [C20]"

3. Update your ctis_config.py to load these mappings
4. Use in your Excel generator and reports!
""")
    print("="*80)


if __name__ == "__main__":
    main()