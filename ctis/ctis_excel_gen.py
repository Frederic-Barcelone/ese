#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Excel Generator - ENHANCED VERSION WITH ISO CODES
Generates ONE Excel file with ALL trials from CTIS SQLite database
Optimized for feasibility assessment with ISO country codes

Version: 5.0.0 - ENHANCED WITH ISO CODES
Last Updated: 2024-11-18

NEW FEATURES:
- ✔ Country ISO codes in Sites_Locations sheet
- ✔ Condition_Synonyms and Condition_Abbreviations in Conditions sheet
- ✔ Enhanced MedDRA extraction
- ✔ Therapeutic areas properly decoded
- ✔ All 9 essential sheets for feasibility
"""

import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional

# Import utilities from project
sys.path.insert(0, '/mnt/project')
try:
    from ctis_config import AGE_CATEGORY_MAP, MSC_PUBLIC_STATUS_CODE_MAP
    from ctis_utils import parse_ts, log
except ImportError:
    print("WARNING: Could not import ctis modules, using fallback")
    AGE_CATEGORY_MAP = {
        "1": "Preterm newborn",
        "2": "Newborns (0-27 days)",
        "3": "Infants and toddlers (28 days-23 months)",
        "4": "Children (2-11 years)",
        "5": "Adolescents (12-17 years)",
        "6": "Adults (18-64 years)",
        "7": "Elderly (65-84 years)",
        "8": "85 years and over"
    }
    
    MSC_PUBLIC_STATUS_CODE_MAP = {
        1: "Authorised, recruitment pending",
        2: "Authorised, recruiting", 
        3: "Authorised, no longer recruiting",
        4: "Temporarily halted",
        5: "Ongoing, recruitment ended",
        6: "Restarted",
        7: "Withdrawn",
        8: "Ended",
        9: "Suspended",
    }
    
    def parse_ts(ts_str):
        if not ts_str:
            return None
        try:
            return datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None
    
    def log(msg):
        print(msg)


# ISO Country Code Mapping
ISO_COUNTRY_MAP = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czechia': 'CZ', 'Czech Republic': 'CZ', 'Denmark': 'DK',
    'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR', 'Germany': 'DE',
    'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE',
    'Italy': 'IT', 'Latvia': 'LV', 'Liechtenstein': 'LI', 'Lithuania': 'LT',
    'Luxembourg': 'LU', 'Malta': 'MT', 'Netherlands': 'NL', 'Norway': 'NO',
    'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Slovakia': 'SK',
    'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE',
    # Common non-EU countries
    'United States': 'US', 'United Kingdom': 'GB', 'Canada': 'CA',
    'Australia': 'AU', 'Switzerland': 'CH', 'Japan': 'JP', 'China': 'CN',
    'South Korea': 'KR', 'Brazil': 'BR', 'Mexico': 'MX', 'Argentina': 'AR',
    'Chile': 'CL', 'Colombia': 'CO', 'Peru': 'PE', 'South Africa': 'ZA',
    'India': 'IN', 'Israel': 'IL', 'Turkey': 'TR', 'Russia': 'RU',
    'Ukraine': 'UA', 'Serbia': 'RS', 'New Zealand': 'NZ', 'Singapore': 'SG',
    'Hong Kong': 'HK', 'Taiwan': 'TW', 'Thailand': 'TH', 'Malaysia': 'MY',
    'Philippines': 'PH', 'Indonesia': 'ID', 'Vietnam': 'VN', 'Egypt': 'EG',
    'Saudi Arabia': 'SA', 'United Arab Emirates': 'AE', 'Qatar': 'QA',
}


def get_country_iso2(country_name: str) -> str:
    """Convert country name to ISO 2-letter code"""
    if not country_name:
        return ""
    return ISO_COUNTRY_MAP.get(country_name.strip(), "")


def decode_age_categories(age_json: str) -> str:
    """Decode age category codes to CTIS format"""
    if not age_json:
        return ""
    try:
        codes = json.loads(age_json)
        if not codes:
            return ""
        names = [AGE_CATEGORY_MAP.get(str(code), f"Code {code}") for code in codes]
        return ", ".join(names)
    except (json.JSONDecodeError, TypeError):
        return str(age_json)


def decode_status_code(status_code: Any) -> str:
    """Decode ctPublicStatusCode to human-readable meaning"""
    if status_code is None or status_code == '':
        return ""
    
    try:
        code = int(status_code) if isinstance(status_code, str) else status_code
        return MSC_PUBLIC_STATUS_CODE_MAP.get(code, f"Unknown status code: {code}")
    except (ValueError, TypeError):
        return str(status_code)


def format_yes_no(value: Any) -> str:
    """Format boolean as Yes/No matching CTIS"""
    if value is None:
        return ""
    if isinstance(value, str):
        value = value.lower() in ('true', '1', 'yes')
    return "Yes" if value else "No"


def format_date(date_str: Any) -> str:
    """Format date to match CTIS style (YYYY-MM-DD) - DATE ONLY, no time"""
    if not date_str or date_str == '':
        return ""
    try:
        # If it's already a datetime object
        if hasattr(date_str, 'strftime'):
            return date_str.strftime('%Y-%m-%d')
        
        # Convert string to datetime
        dt = parse_ts(str(date_str))
        if dt:
            return dt.strftime('%Y-%m-%d')
        
        # Fallback: if it looks like an ISO timestamp, extract just the date part
        date_str_clean = str(date_str)
        if 'T' in date_str_clean:
            return date_str_clean.split('T')[0]
        
        return date_str_clean
    except Exception:
        # Last resort: if there's a T in the string, take everything before it
        date_str_clean = str(date_str)
        if 'T' in date_str_clean:
            return date_str_clean.split('T')[0]
        return date_str_clean



# ===================== Fetch ALL Data at Once =====================

def fetch_all_trials(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all trials overview data"""
    query = "SELECT * FROM trials"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber"
    
    df = pd.read_sql_query(query, conn, params=params)
    
    # Format all date columns immediately after fetching from database
    date_columns = [
        'decisionDate', 'publishDate', 'lastUpdated', 
        'estimatedRecruitmentStartDate', 'estimatedEndDate', 
        'global_end_date', 'pip_decision_date'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(format_date)
    
    # Decode age categories
    if 'ageCategories' in df.columns:
        df['Age_Range_Decoded'] = df['ageCategories'].apply(decode_age_categories)
    
    # Decode status
    if 'ctPublicStatusCode' in df.columns:
        df['Status_Meaning'] = df['ctPublicStatusCode'].apply(decode_status_code)
    
    return df


def fetch_all_inclusion(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all inclusion criteria"""
    query = "SELECT * FROM inclusion_criteria"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber, criterionNumber"
    
    return pd.read_sql_query(query, conn, params=params)


def fetch_all_exclusion(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all exclusion criteria"""
    query = "SELECT * FROM exclusion_criteria"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber, criterionNumber"
    
    return pd.read_sql_query(query, conn, params=params)


def fetch_all_endpoints(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all endpoints"""
    query = "SELECT * FROM endpoints"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber, endpointType, endpointNumber"
    
    return pd.read_sql_query(query, conn, params=params)


def fetch_all_products(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all products"""
    query = "SELECT * FROM trial_products"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber, productName"
    
    return pd.read_sql_query(query, conn, params=params)


def fetch_all_sites(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all sites"""
    query = "SELECT * FROM trial_sites"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber, country, city"
    
    return pd.read_sql_query(query, conn, params=params)


def fetch_all_people(conn: sqlite3.Connection, ct_numbers: Optional[List[str]] = None) -> pd.DataFrame:
    """Fetch all contacts"""
    query = "SELECT * FROM trial_people"
    params = ()
    
    if ct_numbers:
        placeholders = ','.join('?' * len(ct_numbers))
        query += f" WHERE ctNumber IN ({placeholders})"
        params = tuple(ct_numbers)
    
    query += " ORDER BY ctNumber, role, name"
    
    return pd.read_sql_query(query, conn, params=params)



# ===================== Generate Single Excel File =====================

def generate_single_excel_file(
    conn: sqlite3.Connection,
    output_path: Path,
    ct_numbers: Optional[List[str]] = None,
    filter_rare_disease: bool = False
) -> bool:
    """
    Generate ONE Excel file with ALL trials
    
    Args:
        conn: Database connection
        output_path: Path to output Excel file
        ct_numbers: List of specific trial numbers (None = all trials)
        filter_rare_disease: If True, only include rare disease trials
    
    Returns:
        True if successful
    """
    
    log("=" * 80)
    log("FETCHING DATA FROM DATABASE...")
    log("=" * 80)
    
    # Fetch all data at once
    trials_df = fetch_all_trials(conn, ct_numbers)
    
    if len(trials_df) == 0:
        log("ERROR: No trials found in database")
        return False
    
    # Apply rare disease filter if requested
    if filter_rare_disease:
        if 'isConditionRareDisease' in trials_df.columns:
            trials_df = trials_df[trials_df['isConditionRareDisease'] == 1]
            log(f"Filtered to {len(trials_df)} rare disease trials")
        else:
            log("WARNING: isConditionRareDisease column not found, cannot filter")
    
    if len(trials_df) == 0:
        log("ERROR: No trials match filter criteria")
        return False
    
    # Get filtered trial numbers
    filtered_ct_numbers = trials_df['ctNumber'].tolist()
    
    log(f"Processing {len(filtered_ct_numbers)} trials...")
    
    # Create lookup dictionaries for NCT_Number and Protocol_Code
    trial_lookup = trials_df.set_index('ctNumber')[['nct_number', 'shortTitle']].to_dict('index')
    
    def get_nct(ct_number):
        """Get NCT number for a trial"""
        return trial_lookup.get(ct_number, {}).get('nct_number', '')
    
    def get_protocol(ct_number):
        """Get protocol code (shortTitle) for a trial"""
        return trial_lookup.get(ct_number, {}).get('shortTitle', '')
    
    # Fetch all related data
    inclusion_df = fetch_all_inclusion(conn, filtered_ct_numbers)
    exclusion_df = fetch_all_exclusion(conn, filtered_ct_numbers)
    endpoints_df = fetch_all_endpoints(conn, filtered_ct_numbers)
    products_df = fetch_all_products(conn, filtered_ct_numbers)
    sites_df = fetch_all_sites(conn, filtered_ct_numbers)
    people_df = fetch_all_people(conn, filtered_ct_numbers)
    
    log("Data fetched successfully")
    log("=" * 80)
    log("PREPARING EXCEL SHEETS...")
    log("=" * 80)
    
    sheets = {}
    
    # ============================================================
    # Sheet 1: Trial_Overview
    # ============================================================
    overview_data = {
        'EUCT_Number': trials_df['ctNumber'],
        'NCT_Number': trials_df.get('nct_number', ''),
        'WHO_UTN': trials_df.get('who_utn', ''),
        'ISRCTN_Number': trials_df.get('isrctn_number', ''),
        'Protocol_Code': trials_df.get('shortTitle', ''),
        'Full_Title': trials_df.get('title', ''),
        
        # Trial characteristics
        'Trial_Phase': trials_df.get('trialPhase', ''),
        'Transition_Trial': trials_df.get('is_transition_trial', 0).apply(format_yes_no),
        'EudraCT_Number': trials_df.get('eudract_number', ''),
        
        # Sponsor
        'Sponsor': trials_df.get('sponsor', ''),
        
        # Population
        'Age_Range': trials_df.get('Age_Range_Decoded', ''),
        'Gender': trials_df.get('gender', ''),
        'Is_Adult': trials_df.get('isAdult', 0).apply(format_yes_no),
        'Is_Pediatric': trials_df.get('isPediatric', 0).apply(format_yes_no),
        
        # Geography
        'Countries': trials_df.get('countries', ''),
        
        # Objectives
        'Main_Objective': trials_df.get('mainObjective', ''),
        
        # Status
        'Status': trials_df.get('Status_Meaning', ''),
        
        # Timeline - Ensure all dates are formatted as YYYY-MM-DD
        'Decision_Date': trials_df['decisionDate'].apply(format_date) if 'decisionDate' in trials_df.columns else '',
        'Publish_Date': trials_df['publishDate'].apply(format_date) if 'publishDate' in trials_df.columns else '',
        'Estimated_Recruitment_Start': trials_df['estimatedRecruitmentStartDate'].apply(format_date) if 'estimatedRecruitmentStartDate' in trials_df.columns else '',
        'Estimated_End_Date_EU': trials_df['estimatedEndDate'].apply(format_date) if 'estimatedEndDate' in trials_df.columns else '',
        'Global_End_Date': trials_df['global_end_date'].apply(format_date) if 'global_end_date' in trials_df.columns else '',
        'Last_Updated': trials_df['lastUpdated'].apply(format_date) if 'lastUpdated' in trials_df.columns else '',
    }
    sheets['Trial_Overview'] = pd.DataFrame(overview_data)
    
    # ============================================================
    # Sheet 2: Conditions (ENHANCED - with Synonyms and Abbreviations)
    # ============================================================
    conditions_data = {
        'EUCT_Number': trials_df['ctNumber'],
        'NCT_Number': trials_df.get('nct_number', ''),
        'Protocol_Code': trials_df.get('shortTitle', ''),
        'Medical_Condition': trials_df.get('medicalCondition', ''),
        'MedDRA_Code': trials_df.get('conditionMeddraCode', ''),
        'MedDRA_Label': trials_df.get('conditionMeddraLabel', ''),
        'Condition_Synonyms': trials_df.get('conditionSynonyms', ''),  # NEW FIELD
        'Condition_Abbreviations': trials_df.get('conditionAbbreviations', ''),  # NEW FIELD
        'Therapeutic_Areas': trials_df.get('therapeuticAreas', ''),
        'Is_Rare_Disease': trials_df.get('isConditionRareDisease', 0).apply(format_yes_no),
    }
    sheets['Conditions'] = pd.DataFrame(conditions_data)
    
    # ============================================================
    # Sheet 3: Inclusion_Criteria
    # ============================================================
    if len(inclusion_df) > 0:
        inclusion_data = {
            'EUCT_Number': inclusion_df['ctNumber'],
            'NCT_Number': inclusion_df['ctNumber'].apply(get_nct),
            'Protocol_Code': inclusion_df['ctNumber'].apply(get_protocol),
            'Criterion_Number': inclusion_df['criterionNumber'],
            'Criterion_Type': 'Inclusion',
            'Criterion_Text': inclusion_df['criterionText']
        }
        sheets['Inclusion_Criteria'] = pd.DataFrame(inclusion_data)
    else:
        sheets['Inclusion_Criteria'] = pd.DataFrame()
    
    # ============================================================
    # Sheet 4: Exclusion_Criteria
    # ============================================================
    if len(exclusion_df) > 0:
        exclusion_data = {
            'EUCT_Number': exclusion_df['ctNumber'],
            'NCT_Number': exclusion_df['ctNumber'].apply(get_nct),
            'Protocol_Code': exclusion_df['ctNumber'].apply(get_protocol),
            'Criterion_Number': exclusion_df['criterionNumber'],
            'Criterion_Type': 'Exclusion',
            'Criterion_Text': exclusion_df['criterionText']
        }
        sheets['Exclusion_Criteria'] = pd.DataFrame(exclusion_data)
    else:
        sheets['Exclusion_Criteria'] = pd.DataFrame()
    
    # ============================================================
    # Sheet 5: Primary_Endpoints
    # ============================================================
    if len(endpoints_df) > 0:
        primary_df = endpoints_df[endpoints_df['endpointType'] == 'primary'].copy()
        if len(primary_df) > 0:
            primary_data = {
                'EUCT_Number': primary_df['ctNumber'],
                'NCT_Number': primary_df['ctNumber'].apply(get_nct),
                'Protocol_Code': primary_df['ctNumber'].apply(get_protocol),
                'Endpoint_Number': primary_df['endpointNumber'],
                'Endpoint_Type': 'Primary',
                'Endpoint_Description': primary_df['endpointText'],
                'Time_Frame': primary_df.get('timeFrame', '')
            }
            sheets['Primary_Endpoints'] = pd.DataFrame(primary_data)
        else:
            sheets['Primary_Endpoints'] = pd.DataFrame()
    else:
        sheets['Primary_Endpoints'] = pd.DataFrame()
    
    # ============================================================
    # Sheet 6: Secondary_Endpoints
    # ============================================================
    if len(endpoints_df) > 0:
        secondary_df = endpoints_df[endpoints_df['endpointType'] == 'secondary'].copy()
        if len(secondary_df) > 0:
            secondary_data = {
                'EUCT_Number': secondary_df['ctNumber'],
                'NCT_Number': secondary_df['ctNumber'].apply(get_nct),
                'Protocol_Code': secondary_df['ctNumber'].apply(get_protocol),
                'Endpoint_Number': secondary_df['endpointNumber'],
                'Endpoint_Type': 'Secondary',
                'Endpoint_Description': secondary_df['endpointText'],
                'Time_Frame': secondary_df.get('timeFrame', '')
            }
            sheets['Secondary_Endpoints'] = pd.DataFrame(secondary_data)
        else:
            sheets['Secondary_Endpoints'] = pd.DataFrame()
    else:
        sheets['Secondary_Endpoints'] = pd.DataFrame()
    
    # ============================================================
    # Sheet 7: Products
    # ============================================================
    if len(products_df) > 0:
        products_data = {
            'EUCT_Number': products_df['ctNumber'],
            'NCT_Number': products_df['ctNumber'].apply(get_nct),
            'Protocol_Code': products_df['ctNumber'].apply(get_protocol),
            'Product_Role': products_df.get('productRole', ''),
            'Product_Name': products_df.get('productName', ''),
            'Active_Substance': products_df.get('activeSubstance', ''),
            'ATC_Code': products_df.get('atcCode', ''),
            'Pharmaceutical_Form': products_df.get('pharmaceuticalForm', ''),
            'Route': products_df.get('route', ''),
            'Max_Daily_Dose': products_df.get('maxDailyDose', ''),
            'Daily_Dose_Unit': products_df.get('maxDailyDoseUnit', ''),
            'Max_Treatment_Period': products_df.get('maxTreatmentPeriod', ''),
            'Treatment_Period_Unit': products_df.get('maxTreatmentPeriodUnit', ''),
            'Is_Paediatric': products_df.get('isPaediatric', 0).apply(format_yes_no),
            'Is_Orphan_Drug': products_df.get('isOrphanDrug', 0).apply(format_yes_no),
            'Authorization_Status': products_df.get('authorizationStatus', '')
        }
        sheets['Products'] = pd.DataFrame(products_data)
    else:
        sheets['Products'] = pd.DataFrame()
    
    # ============================================================
    # Sheet 8: Sites_Locations (ENHANCED - with ISO Country Codes)
    # ============================================================
    if len(sites_df) > 0:
        # Add ISO country codes
        sites_df['country_iso2'] = sites_df['country'].apply(get_country_iso2)
        
        sites_data = {
            'EUCT_Number': sites_df['ctNumber'],
            'NCT_Number': sites_df['ctNumber'].apply(get_nct),
            'Protocol_Code': sites_df['ctNumber'].apply(get_protocol),
            'Country': sites_df.get('country', ''),
            'Country_ISO2': sites_df['country_iso2'],  # NEW FIELD
            'City': sites_df.get('city', ''),
            'Site_Name': sites_df.get('site_name', ''),
            'Organisation': sites_df.get('organisation', ''),
            'Address': sites_df.get('address', ''),
            'Postal_Code': sites_df.get('postal_code', '')
        }
        sheets['Sites_Locations'] = pd.DataFrame(sites_data)
    else:
        sheets['Sites_Locations'] = pd.DataFrame()
    
    # ============================================================
    # Sheet 9: Contacts (ENHANCED - with ISO Country Codes)
    # ============================================================
    if len(people_df) > 0:
        # Add ISO country codes
        people_df['country_iso2'] = people_df['country'].apply(get_country_iso2)
        
        contacts_data = {
            'EUCT_Number': people_df['ctNumber'],
            'NCT_Number': people_df['ctNumber'].apply(get_nct),
            'Protocol_Code': people_df['ctNumber'].apply(get_protocol),
            'Country': people_df.get('country', ''),
            'Country_ISO2': people_df['country_iso2'],  # NEW FIELD
            'City': people_df.get('city', ''),
            'Contact_Role': people_df.get('role', ''),
            'Name': people_df.get('name', ''),
            'Email': people_df.get('email', ''),
            'Phone': people_df.get('phone', ''),
            'Organisation': people_df.get('organisation', ''),
            'Site_Name': people_df.get('site_name', '')
        }
        sheets['Contacts'] = pd.DataFrame(contacts_data)
    else:
        sheets['Contacts'] = pd.DataFrame()
    
    # ============================================================
    # Write to Excel
    # ============================================================
    log("=" * 80)
    log("WRITING EXCEL FILE...")
    log("=" * 80)
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                log(f"Writing sheet: {sheet_name} ({len(df)} rows)")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max() if len(df) > 0 else 0,
                        len(col)
                    )
                    adjusted_width = min(max_length + 2, 100)
                    
                    # Excel column letters
                    if idx < 26:
                        col_letter = chr(65 + idx)
                    else:
                        first = (idx // 26) - 1
                        second = idx % 26
                        if first >= 0:
                            col_letter = chr(65 + first) + chr(65 + second)
                        else:
                            col_letter = chr(65 + second)
                    worksheet.column_dimensions[col_letter].width = adjusted_width
        
        log("=" * 80)
        log("SUCCESS!")
        log("=" * 80)
        log(f"Excel file created: {output_path}")
        log(f"Total trials: {len(filtered_ct_numbers)}")
        log(f"Total sheets: {len(sheets)}")
        log("")
        log("Sheet Summary:")
        for sheet_name, df in sheets.items():
            log(f"  - {sheet_name}: {len(df)} rows")
        log("=" * 80)
        
        return True
        
    except Exception as e:
        log(f"ERROR: Failed to write Excel file: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===================== Main Entry Point =====================

if __name__ == "__main__":
    """
    Main entry point - creates ONE Excel file with ALL trials
    ENHANCED VERSION v5.0 with ISO codes
    """
    
    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Database path
    DB_PATH = Path("ctis-out/ctis.db")  # Adjust to your database
    
    # Output file name - add date automatically (format: YYYY_MM_DD)
    today = datetime.now().strftime('%Y_%m_%d')
    OUTPUT_FILE = f"{today}_CTIS_FEASIBILITY.xlsx"
    
    # Output directory
    OUTPUT_DIR = Path("ctis-out")
    
    # Filter options - GET ALL TRIALS
    FILTER_RARE_DISEASE_ONLY = False  # False = ALL trials from database
    
    # Specific trials (set to None to process ALL trials in database)
    SPECIFIC_TRIALS = None  # None = Process ALL trials
    
    # ============================================================
    
    print()
    print("=" * 80)
    print("CTIS EXCEL GENERATOR v5.0 - ENHANCED WITH ISO CODES")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rare disease filter: {FILTER_RARE_DISEASE_ONLY}")
    print("=" * 80)
    
    if not DB_PATH.exists():
        print(f"\nERROR: Database not found: {DB_PATH}")
        print("\nPlease update DB_PATH in the script to point to your database.")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output_path = OUTPUT_DIR / OUTPUT_FILE
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Generate single Excel file
    success = generate_single_excel_file(
        conn=conn,
        output_path=output_path,
        ct_numbers=SPECIFIC_TRIALS,
        filter_rare_disease=FILTER_RARE_DISEASE_ONLY
    )
    
    conn.close()
    
    # Exit
    if success:
        print("\n✓ Excel file generated successfully!")
        print(f"  Location: {output_path}")
        sys.exit(0)
    else:
        print("\n✗ Failed to generate Excel file")
        sys.exit(1)