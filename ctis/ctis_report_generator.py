#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Report Generator - CTIS-COMPLIANT VERSION
Generates reports using EXACT CTIS labels from the official HTML download
ctis/ctis_report_generator_ctis_format.py

Version: 1.4.1
Last Updated: 2024-11-10

This generator uses the EXACT labels from CTIS HTML downloads,
with database field names shown in parentheses for reference.
"""

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Import utilities
try:
    from ctis_config import AGE_CATEGORY_MAP
    from ctis_utils import parse_ts, log
except ImportError:
    print("ERROR: Could not import ctis modules")
    sys.exit(1)


# ===================== CTIS Label Mappings =====================

# Map database fields to CTIS official labels
CTIS_LABELS = {
    # Trial Identification
    'ctNumber': 'EUCT number',
    'title': 'Full title (English)',
    'shortTitle': 'Protocol code',
    'sponsor': 'Sponsor',
    'trialPhase': 'Trial Phase',
    'medicalCondition': 'Medical condition(s)',
    'therapeuticAreas': 'Therapeutic areas',
    'countries': 'Locations',
    'mainObjective': 'Main objective (English)',
    'is_transition_trial': 'Transition Trial',
    
    # Trial Status & Dates
    'ctPublicStatusCode': 'Overall trial status',
    'decisionDate': 'Decision Date',
    'publishDate': 'Publication Date',
    'lastUpdated': 'Last Updated',
    'estimatedRecruitmentStartDate': 'Estimated recruitment start date in EU/EEA',
    'estimatedEndDate': 'Estimated end of trial date in EU/EEA',
    'global_end_date': 'Estimated global end date of the trial',
    
    # Population
    'ageCategories': 'Age range',
    'isPediatric': 'Pediatric',
    'isAdult': 'Adult',
    'gender': 'Gender',
    'isConditionRareDisease': 'Is the medical condition considered to be a rare disease',
    
    # Design
    'isRandomised': 'Randomised',
    'blindingType': 'Blinding type',
    'trialScope': 'Trial scope',
    
    # Endpoints
    'primaryEndpointsCount': 'Number of primary endpoints',
    'secondaryEndpointsCount': 'Number of secondary endpoints',
    
    # Products
    'productName': 'Product name',
    'activeSubstance': 'Active Substance name',
    'atcCode': 'Anatomical Therapeutic Chemical (ATC) Codes',
    'pharmaceuticalForm': 'Pharmaceutical form',
    'route': 'Route of administration',
    'maxDailyDose': 'Maximum daily dose',
    'maxDailyDoseUnit': 'Daily dose unit of measure',
    'maxTreatmentPeriod': 'Maximum treatment period',
    'maxTreatmentPeriodUnit': 'Treatment period unit',
    'isPaediatric': 'Paediatric formulation',
    'isOrphanDrug': 'Does this product have an orphan drug designation',
    'authorizationStatus': 'Product authorisation status',
    
    # Sites
    'site_name': 'Site location',
    'address': 'Site street address',
    'city': 'Site city',
    'postal_code': 'Site post code',
    'country': 'Site country',
    'organisation': 'Organisation name',
    
    # Contacts
    'name': 'Name',
    'email': 'Email address',
    'phone': 'Phone number',
    'role': 'Role',
    
    # Member State
    'member_state': 'Member State',
    'status': 'Application Trial Status',
    'decision_date': 'Decision Date',
    'start_date': 'Start of trial',
    'recruitment_start': 'Start of recruitment',
    'recruitment_end': 'End of recruitment',
    'end_date': 'End of trial',
    'early_termination_date': 'Early termination',
    'early_termination_reason': 'Reason for early termination',
    'temporary_halt': 'Temporary Halt',
    'restart_date': 'Restart trial',
}


# ===================== Format Functions =====================

def format_section_header(title: str, level: int = 1) -> str:
    """Format section headers matching CTIS style"""
    if level == 1:
        return f"\n{'=' * 80}\n{title.upper()}\n{'=' * 80}\n"
    elif level == 2:
        return f"\n{'-' * 80}\n{title}\n{'-' * 80}\n"
    else:
        return f"\n{title}:\n"


def format_field_ctis(ctis_label: str, value: Any, db_field: str = "", indent: int = 0) -> str:
    """
    Format a field using CTIS label with database field in parentheses
    """
    indent_str = "  " * indent
    
    # Handle None/empty values
    if value is None or value == "":
        display_value = ""
    else:
        display_value = str(value)
    
    # Show CTIS label with database field in parentheses
    if db_field:
        label_display = f"{ctis_label} ({db_field})"
    else:
        label_display = ctis_label
    
    return f"{indent_str}{label_display:60s}: {display_value}\n"


def get_ctis_label(db_field: str) -> str:
    """Get CTIS label for a database field"""
    return CTIS_LABELS.get(db_field, db_field)


def format_date(date_str: Any) -> str:
    """Format date to match CTIS style (YYYY-MM-DD)"""
    if not date_str:
        return ""
    try:
        dt = parse_ts(date_str)
        if dt:
            return dt.strftime('%Y-%m-%d')
        return str(date_str)
    except (ValueError, TypeError):
        return str(date_str)


def decode_age_categories(age_json: str) -> str:
    """Decode age category codes to CTIS format"""
    if not age_json:
        return ""
    try:
        codes = json.loads(age_json)
        if not codes:
            return ""
        # CTIS format: "18-64 years,65+ years"
        # Convert codes to strings for lookup (handles both int and str codes)
        names = [AGE_CATEGORY_MAP.get(str(code), f"Code {code}") for code in codes]
        return ",".join(names)
    except (json.JSONDecodeError, TypeError):
        return str(age_json)


def format_yes_no(value: Any) -> str:
    """Format boolean as Yes/No matching CTIS"""
    if value is None:
        return ""
    if isinstance(value, str):
        value = value.lower() in ('true', '1', 'yes')
    return "Yes" if value else "No"


# ===================== Data Retrieval Functions =====================

def get_trial_data(conn: sqlite3.Connection, ct_number: str) -> Optional[Dict[str, Any]]:
    """Retrieve trial data from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trials WHERE ctNumber = ?", (ct_number,))
    row = cursor.fetchone()
    
    if not row:
        return None
    
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row))


def get_trial_sites(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve trial sites"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT site_name, organisation, country, city, address, postal_code
        FROM trial_sites
        WHERE ctNumber = ?
        ORDER BY country, city, site_name
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_trial_people(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve trial people"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, role, email, phone, country, city, site_name, organisation
        FROM trial_people
        WHERE ctNumber = ?
        ORDER BY role, name
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_inclusion_criteria(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve inclusion criteria"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT criterionNumber, criterionText
        FROM inclusion_criteria
        WHERE ctNumber = ?
        ORDER BY criterionNumber
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_exclusion_criteria(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve exclusion criteria"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT criterionNumber, criterionText
        FROM exclusion_criteria
        WHERE ctNumber = ?
        ORDER BY criterionNumber
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_endpoints(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve endpoints"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT endpointType, endpointNumber, endpointText, timeFrame
        FROM endpoints
        WHERE ctNumber = ?
        ORDER BY endpointType, endpointNumber
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_trial_products(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve trial products"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT productRole, productName, activeSubstance, atcCode, pharmaceuticalForm,
               route, maxDailyDose, maxDailyDoseUnit, maxTreatmentPeriod, maxTreatmentPeriodUnit,
               isPaediatric, isOrphanDrug, authorizationStatus
        FROM trial_products
        WHERE ctNumber = ?
        ORDER BY productRole, productName
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_ms_status(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve member state status"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT member_state, status, decision_date, start_date, recruitment_start,
               recruitment_end, temporary_halt, restart_date, end_date,
               early_termination_date, early_termination_reason, last_update
        FROM ms_status
        WHERE ctNumber = ?
        ORDER BY member_state, captured_at DESC
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    
    # Get unique records per member state (most recent)
    seen_states = set()
    results = []
    for row in cursor.fetchall():
        record = dict(zip(columns, row))
        if record['member_state'] not in seen_states:
            seen_states.add(record['member_state'])
            results.append(record)
    
    return results


def get_country_planning(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve country planning"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT country, planned_participants
        FROM country_planning
        WHERE ctNumber = ?
        ORDER BY country
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_site_contacts(conn: sqlite3.Connection, ct_number: str) -> List[Dict[str, Any]]:
    """Retrieve site contacts"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT country, country_iso2, org_name, site_name, address, city, postal_code,
               pi_name, pi_email, pi_phone
        FROM site_contact
        WHERE ctNumber = ?
        ORDER BY country, org_name
    """, (ct_number,))
    
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ===================== Report Generation =====================

def generate_ctis_format_report(conn: sqlite3.Connection, ct_number: str, output_path: Path):
    """
    Generate a report using EXACT CTIS labels from official HTML downloads
    Database field names shown in parentheses for reference
    """
    
    # Retrieve all data
    trial = get_trial_data(conn, ct_number)
    if not trial:
        print(f"ERROR: Trial {ct_number} not found in database")
        return False
    
    sites = get_trial_sites(conn, ct_number)
    people = get_trial_people(conn, ct_number)
    inclusion = get_inclusion_criteria(conn, ct_number)
    exclusion = get_exclusion_criteria(conn, ct_number)
    endpoints = get_endpoints(conn, ct_number)
    products = get_trial_products(conn, ct_number)
    ms_statuses = get_ms_status(conn, ct_number)

    # Generate report
    with output_path.open('w', encoding='utf-8') as f:
        # Header - matching CTIS HTML format
        f.write(format_section_header("CTIS Clinical Trial Information", 1))
        f.write(f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write("Data extracted from CTIS database\n")
        f.write("Labels match official CTIS HTML download format\n")
        f.write("Database field names shown in parentheses\n")
        
        # ============================================================
        # 1. SUMMARY - TRIAL INFORMATION
        # ============================================================
        f.write(format_section_header("1. Summary", 1))
        f.write(format_section_header("1.1 Trial Information", 2))
        
        f.write(format_field_ctis(get_ctis_label('ctNumber'), trial['ctNumber'], 'ctNumber'))
        f.write(format_field_ctis(get_ctis_label('title'), trial['title'], 'title'))
        f.write(format_field_ctis(get_ctis_label('shortTitle'), trial['shortTitle'], 'shortTitle'))
        f.write(format_field_ctis(get_ctis_label('medicalCondition'), trial['medicalCondition'], 'medicalCondition'))
        f.write(format_field_ctis(get_ctis_label('trialPhase'), trial['trialPhase'], 'trialPhase'))
        f.write(format_field_ctis(get_ctis_label('is_transition_trial'), format_yes_no(trial.get('is_transition_trial')), 'is_transition_trial'))
        f.write(format_field_ctis(get_ctis_label('sponsor'), trial['sponsor'], 'sponsor'))
        
        # Age range - decode using AGE_CATEGORY_MAP
        age_codes = trial.get('ageCategories', '[]')
        age_display = decode_age_categories(age_codes)
        
        f.write(format_field_ctis(get_ctis_label('ageCategories'), age_display, 'ageCategories'))
        f.write(format_field_ctis(get_ctis_label('isAdult'), format_yes_no(trial.get('isAdult')), 'isAdult'))
        f.write(format_field_ctis(get_ctis_label('isPediatric'), format_yes_no(trial.get('isPediatric')), 'isPediatric'))
        
        f.write(format_field_ctis(get_ctis_label('countries'), trial['countries'], 'countries'))
        f.write(format_field_ctis(get_ctis_label('mainObjective'), trial.get('mainObjective', ''), 'mainObjective'))
        
        # ============================================================
        # 1.2 OVERALL TRIAL STATUS
        # ============================================================
        f.write(format_section_header("1.2 Overall Trial Status", 2))
        
        f.write(format_field_ctis(get_ctis_label('ctPublicStatusCode'), trial['ctPublicStatusCode'], 'ctPublicStatusCode'))
        f.write(format_field_ctis(get_ctis_label('estimatedRecruitmentStartDate'), format_date(trial.get('estimatedRecruitmentStartDate')), 'estimatedRecruitmentStartDate'))
        f.write(format_field_ctis(get_ctis_label('estimatedEndDate'), format_date(trial.get('estimatedEndDate')), 'estimatedEndDate'))
        f.write(format_field_ctis(get_ctis_label('global_end_date'), format_date(trial.get('global_end_date')), 'global_end_date'))
        
        # Application Trial Status table
        if ms_statuses:
            f.write(f"\n{get_ctis_label('status')} by {get_ctis_label('member_state')}:\n")
            f.write(f"{'Member State':<30s} {'Status':<40s} {'Decision Date':<15s}\n")
            f.write(f"{'-' * 85}\n")
            for ms in sorted(ms_statuses, key=lambda x: x['member_state']):
                ms_name = ms['member_state'] or ''
                ms_status = ms['status'] or ''
                ms_decision = format_date(ms['decision_date'])
                f.write(f"{ms_name:<30s} {ms_status:<40s} {ms_decision:<15s}\n")
        
        # ============================================================
        # 1.3 TRIAL NOTIFICATIONS (by Member State)
        # ============================================================
        f.write(format_section_header("1.3 Trial Notifications", 2))
        
        if ms_statuses:
            for ms in sorted(ms_statuses, key=lambda x: x['member_state']):
                f.write(f"\n{ms['member_state']}\n")
                f.write(f"{'-' * 80}\n")
                f.write(format_field_ctis(get_ctis_label('start_date'), format_date(ms['start_date']), 'start_date', indent=1))
                f.write(format_field_ctis(get_ctis_label('restart_date'), format_date(ms['restart_date']), 'restart_date', indent=1))
                f.write(format_field_ctis(get_ctis_label('end_date'), format_date(ms['end_date']), 'end_date', indent=1))
                f.write(format_field_ctis(get_ctis_label('early_termination_date'), format_date(ms['early_termination_date']), 'early_termination_date', indent=1))
                if ms['early_termination_reason']:
                    f.write(format_field_ctis(get_ctis_label('early_termination_reason'), ms['early_termination_reason'], 'early_termination_reason', indent=1))
        
        # ============================================================
        # 1.4 RECRUITMENT NOTIFICATIONS
        # ============================================================
        f.write(format_section_header("1.4 Recruitment Notifications", 2))
        
        if ms_statuses:
            for ms in sorted(ms_statuses, key=lambda x: x['member_state']):
                if ms['recruitment_start'] or ms['recruitment_end']:
                    f.write(f"\n{ms['member_state']}\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(format_field_ctis(get_ctis_label('recruitment_start'), format_date(ms['recruitment_start']), 'recruitment_start', indent=1))
                    f.write(format_field_ctis(get_ctis_label('recruitment_end'), format_date(ms['recruitment_end']), 'recruitment_end', indent=1))
        
        # ============================================================
        # 2. FULL TRIAL INFORMATION
        # ============================================================
        f.write(format_section_header("2. Full Trial Information", 1))
        f.write(format_section_header("2.1 Trial Details", 2))
        
        # Population
        f.write("\nPopulation:\n")
        f.write(format_field_ctis(get_ctis_label('gender'), trial.get('gender', ''), 'gender', indent=1))
        f.write(format_field_ctis(get_ctis_label('ageCategories'), age_display, 'ageCategories', indent=1))
        f.write(format_field_ctis(get_ctis_label('isPediatric'), format_yes_no(trial.get('isPediatric')), 'isPediatric', indent=1))
        f.write(format_field_ctis(get_ctis_label('isAdult'), format_yes_no(trial.get('isAdult')), 'isAdult', indent=1))
        f.write(format_field_ctis(get_ctis_label('isConditionRareDisease'), format_yes_no(trial.get('isConditionRareDisease')), 'isConditionRareDisease', indent=1))
        
        # Trial Design
        f.write("\nTrial Design:\n")
        f.write(format_field_ctis(get_ctis_label('isRandomised'), format_yes_no(trial.get('isRandomised')), 'isRandomised', indent=1))
        f.write(format_field_ctis(get_ctis_label('blindingType'), trial.get('blindingType', ''), 'blindingType', indent=1))
        
        # Inclusion Criteria
        f.write("\nInclusion Criteria:\n")
        if inclusion:
            for criterion in inclusion:
                f.write(f"  {criterion['criterionNumber']}. {criterion['criterionText']}\n")
        else:
            f.write("  (None specified)\n")
        
        # Exclusion Criteria
        f.write("\nExclusion Criteria:\n")
        if exclusion:
            for criterion in exclusion:
                f.write(f"  {criterion['criterionNumber']}. {criterion['criterionText']}\n")
        else:
            f.write("  (None specified)\n")
        
        # Endpoints
        f.write("\nEndpoints:\n")
        f.write(format_field_ctis(get_ctis_label('primaryEndpointsCount'), trial.get('primaryEndpointsCount', 0), 'primaryEndpointsCount', indent=1))
        f.write(format_field_ctis(get_ctis_label('secondaryEndpointsCount'), trial.get('secondaryEndpointsCount', 0), 'secondaryEndpointsCount', indent=1))
        
        primary_eps = [ep for ep in endpoints if ep['endpointType'] == 'primary']
        secondary_eps = [ep for ep in endpoints if ep['endpointType'] == 'secondary']
        
        if primary_eps:
            f.write("\n  Primary Endpoints:\n")
            for ep in primary_eps:
                f.write(f"    {ep['endpointNumber']}. {ep['endpointText']}\n")
                if ep['timeFrame']:
                    f.write(f"       Timeframe: {ep['timeFrame']}\n")
        
        if secondary_eps:
            f.write("\n  Secondary Endpoints:\n")
            for ep in secondary_eps:
                f.write(f"    {ep['endpointNumber']}. {ep['endpointText']}\n")
                if ep['timeFrame']:
                    f.write(f"       Timeframe: {ep['timeFrame']}\n")
        
        # ============================================================
        # 2.2 PRODUCTS
        # ============================================================
        f.write(format_section_header("2.2 Products", 2))
        
        if products:
            for idx, prod in enumerate(products, 1):
                f.write(f"\nProduct {idx} ({prod['productRole']})\n")
                f.write(f"{'-' * 80}\n")
                f.write(format_field_ctis(get_ctis_label('productName'), prod['productName'], 'productName', indent=1))
                f.write(format_field_ctis(get_ctis_label('activeSubstance'), prod['activeSubstance'], 'activeSubstance', indent=1))
                f.write(format_field_ctis(get_ctis_label('atcCode'), prod['atcCode'], 'atcCode', indent=1))
                f.write(format_field_ctis(get_ctis_label('pharmaceuticalForm'), prod['pharmaceuticalForm'], 'pharmaceuticalForm', indent=1))
                f.write(format_field_ctis(get_ctis_label('route'), prod['route'], 'route', indent=1))
                f.write(format_field_ctis(get_ctis_label('maxDailyDose'), prod['maxDailyDose'], 'maxDailyDose', indent=1))
                f.write(format_field_ctis(get_ctis_label('maxDailyDoseUnit'), prod['maxDailyDoseUnit'], 'maxDailyDoseUnit', indent=1))
                f.write(format_field_ctis(get_ctis_label('maxTreatmentPeriod'), prod['maxTreatmentPeriod'], 'maxTreatmentPeriod', indent=1))
                f.write(format_field_ctis(get_ctis_label('maxTreatmentPeriodUnit'), prod['maxTreatmentPeriodUnit'], 'maxTreatmentPeriodUnit', indent=1))
                f.write(format_field_ctis(get_ctis_label('isPaediatric'), format_yes_no(prod['isPaediatric']), 'isPaediatric', indent=1))
                f.write(format_field_ctis(get_ctis_label('isOrphanDrug'), format_yes_no(prod['isOrphanDrug']), 'isOrphanDrug', indent=1))
                f.write(format_field_ctis(get_ctis_label('authorizationStatus'), prod['authorizationStatus'], 'authorizationStatus', indent=1))
        else:
            f.write("  (No products specified)\n")
        
        # ============================================================
        # 4. LOCATIONS AND CONTACT POINTS
        # ============================================================
        f.write(format_section_header("4. Locations and Contact Points", 1))
        f.write(format_section_header("4.1 Locations", 2))
        
        if sites:
            # Group by country
            sites_by_country = defaultdict(list)
            for site in sites:
                country = site['country'] or 'Unknown'
                sites_by_country[country].append(site)
            
            for country in sorted(sites_by_country.keys()):
                f.write(f"\n{country} ({len(sites_by_country[country])} sites)\n")
                f.write(f"{'-' * 80}\n")
                
                for idx, site in enumerate(sites_by_country[country], 1):
                    f.write(f"\n  Site {idx}:\n")
                    f.write(format_field_ctis(get_ctis_label('site_name'), site['site_name'], 'site_name', indent=2))
                    f.write(format_field_ctis(get_ctis_label('organisation'), site['organisation'], 'organisation', indent=2))
                    f.write(format_field_ctis(get_ctis_label('address'), site['address'], 'address', indent=2))
                    f.write(format_field_ctis(get_ctis_label('city'), site['city'], 'city', indent=2))
                    f.write(format_field_ctis(get_ctis_label('postal_code'), site['postal_code'], 'postal_code', indent=2))
                    f.write(format_field_ctis(get_ctis_label('country'), site['country'], 'country', indent=2))
        else:
            f.write("  (No sites specified)\n")
        
        # ============================================================
        # 4.2 SPONSORS / CONTACTS
        # ============================================================
        f.write(format_section_header("4.2 Sponsors and Contacts", 2))
        
        if people:
            # Group by role
            people_by_role = defaultdict(list)
            for person in people:
                role = person['role'] or 'Unknown Role'
                people_by_role[role].append(person)
            
            for role in sorted(people_by_role.keys()):
                f.write(f"\n{role} ({len(people_by_role[role])} contacts)\n")
                f.write(f"{'-' * 80}\n")
                
                for person in people_by_role[role]:
                    f.write("\n")
                    f.write(format_field_ctis(get_ctis_label('name'), person['name'], 'name', indent=1))
                    f.write(format_field_ctis(get_ctis_label('email'), person['email'], 'email', indent=1))
                    f.write(format_field_ctis(get_ctis_label('phone'), person['phone'], 'phone', indent=1))
                    f.write(format_field_ctis(get_ctis_label('organisation'), person['organisation'], 'organisation', indent=1))
                    if person['site_name']:
                        f.write(format_field_ctis('Site', person['site_name'], 'site_name', indent=1))
                    if person['country']:
                        location = f"{person['city']}, {person['country']}" if person['city'] else person['country']
                        f.write(format_field_ctis('Location', location, 'city, country', indent=1))
        else:
            f.write("  (No contacts specified)\n")
        
        # ============================================================
        # FOOTER
        # ============================================================
        f.write(format_section_header("End of Report", 1))
        f.write(f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"CT Number: {ct_number}\n")
        f.write("Labels match official CTIS HTML download format\n")
        f.write("Database field names shown in parentheses\n")
        f.write("=" * 80 + "\n")
    
    return True


# ===================== Main / CLI =====================

if __name__ == "__main__":
    """Command-line interface"""
    
    if len(sys.argv) < 3:
        print("Usage: python ctis_report_generator_ctis_format.py <db_path> <ct_number> [output_path]")
        print("\nExample:")
        print("  python ctis_report_generator_ctis_format.py ctis-out/ctis.db 2024-514133-38-00")
        sys.exit(1)
    
    db_path = Path(sys.argv[1])
    ct_number = sys.argv[2]
    
    if len(sys.argv) > 3:
        output_path = Path(sys.argv[3])
    else:
        output_path = Path(f"{ct_number}_ctis_format_report.txt")
    
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    
    print(f"Generating CTIS-format report for {ct_number}...")
    print(f"Output: {output_path}")
    
    if generate_ctis_format_report(conn, ct_number, output_path):
        print("âœ“ CTIS-format report generated successfully!")
        print(f"  {output_path}")
    else:
        print("âœ— Failed to generate report")
        sys.exit(1)
    
    conn.close()