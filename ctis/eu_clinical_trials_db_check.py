#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS DB Enhanced Audit Tool
Provides comprehensive field-by-field analysis with:
- Unique value counts and distributions
- Data quality metrics per field
- Statistical summaries
- Temporal analysis
- Relationship patterns
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# ========= Configuration =========
OUT_DIR = Path("ctis-out")
DB_PATH = OUT_DIR / "ctis.db"
NDJSON_PATH = OUT_DIR / "ctis_full.ndjson"
REPORT_PATH = OUT_DIR / "audit_report_enhanced.md"
JSON_REPORT_PATH = OUT_DIR / "audit_report_enhanced.json"

EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?$")

# ========= Utility Functions =========

def is_blank(x):
    return x is None or (isinstance(x, str) and x.strip() == "")

def pct(n, d):
    return 0.0 if d == 0 else (100.0 * n / d)

def safe_execute(cur, query, params=()):
    """Execute query with error handling."""
    try:
        cur.execute(query, params)
        return cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"[WARN] Query failed: {e}")
        return []

def count_lines(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

def parse_year(date_str: Any) -> Optional[int]:
    """Extract year from various date formats."""
    if is_blank(date_str):
        return None
    s = str(date_str).strip()
    match = re.match(r'^(\d{4})', s)
    return int(match.group(1)) if match else None

def parse_date(date_str: Any) -> Optional[datetime]:
    """Parse date string to datetime object."""
    if is_blank(date_str):
        return None
    s = str(date_str).strip()
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

# ========= Field Analysis Functions =========

def analyze_field_basic(values: List[Any], field_name: str) -> Dict[str, Any]:
    """Basic field analysis: nulls, blanks, unique values."""
    total = len(values)
    
    null_count = sum(1 for v in values if v is None)
    blank_count = sum(1 for v in values if is_blank(v))
    non_blank = [v for v in values if not is_blank(v)]
    unique_values = set(str(v) for v in non_blank)
    
    return {
        "field": field_name,
        "total_rows": total,
        "null_count": null_count,
        "blank_count": blank_count,
        "non_blank_count": len(non_blank),
        "unique_count": len(unique_values),
        "fill_rate": pct(len(non_blank), total),
        "cardinality_ratio": pct(len(unique_values), len(non_blank)) if non_blank else 0,
    }

def analyze_text_field(values: List[Any], field_name: str, top_n: int = 10) -> Dict[str, Any]:
    """Detailed text field analysis."""
    basic = analyze_field_basic(values, field_name)
    non_blank = [str(v).strip() for v in values if not is_blank(v)]
    
    if not non_blank:
        return {**basic, "top_values": [], "length_stats": {}}
    
    # Length statistics
    lengths = [len(v) for v in non_blank]
    length_stats = {
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
        "median": sorted(lengths)[len(lengths) // 2],
    }
    
    # Top values
    counter = Counter(non_blank)
    top_values = [
        {"value": val, "count": cnt, "percentage": pct(cnt, len(non_blank))}
        for val, cnt in counter.most_common(top_n)
    ]
    
    return {
        **basic,
        "length_stats": length_stats,
        "top_values": top_values,
    }

def analyze_numeric_field(values: List[Any], field_name: str) -> Dict[str, Any]:
    """Numeric field analysis with statistics."""
    basic = analyze_field_basic(values, field_name)
    
    numeric_values = []
    for v in values:
        if is_blank(v):
            continue
        try:
            numeric_values.append(float(v) if '.' in str(v) else int(v))
        except (ValueError, TypeError):
            pass
    
    if not numeric_values:
        return {**basic, "numeric_stats": None}
    
    sorted_vals = sorted(numeric_values)
    n = len(sorted_vals)
    
    stats = {
        "count": n,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": sum(sorted_vals) / n,
        "median": sorted_vals[n // 2],
        "q25": sorted_vals[n // 4],
        "q75": sorted_vals[3 * n // 4],
    }
    
    # Value distribution
    counter = Counter(numeric_values)
    top_values = [
        {"value": val, "count": cnt}
        for val, cnt in counter.most_common(10)
    ]
    
    return {
        **basic,
        "numeric_stats": stats,
        "top_values": top_values,
    }

def analyze_date_field(values: List[Any], field_name: str) -> Dict[str, Any]:
    """Date field analysis with temporal patterns."""
    basic = analyze_field_basic(values, field_name)
    
    # ISO format validation
    non_blank = [v for v in values if not is_blank(v)]
    valid_iso = sum(1 for v in non_blank if ISO_TS_RE.match(str(v)))
    
    # Parse dates
    dates = [parse_date(v) for v in non_blank]
    dates = [d for d in dates if d is not None]
    
    if not dates:
        return {
            **basic,
            "iso_valid_count": valid_iso,
            "iso_valid_rate": pct(valid_iso, len(non_blank)),
            "temporal_stats": None,
        }
    
    sorted_dates = sorted(dates)
    
    # Year distribution
    years = [d.year for d in dates]
    year_dist = Counter(years)
    
    temporal_stats = {
        "earliest": sorted_dates[0].isoformat(),
        "latest": sorted_dates[-1].isoformat(),
        "span_days": (sorted_dates[-1] - sorted_dates[0]).days,
        "year_distribution": [
            {"year": year, "count": cnt}
            for year, cnt in sorted(year_dist.items())
        ],
    }
    
    return {
        **basic,
        "iso_valid_count": valid_iso,
        "iso_valid_rate": pct(valid_iso, len(non_blank)),
        "temporal_stats": temporal_stats,
    }

def analyze_email_field(values: List[Any], field_name: str) -> Dict[str, Any]:
    """Email field analysis with validation."""
    text_analysis = analyze_text_field(values, field_name, top_n=15)
    
    non_blank = [str(v).strip() for v in values if not is_blank(v)]
    valid_emails = [e for e in non_blank if EMAIL_RE.match(e)]
    
    # Domain analysis
    domains = []
    for email in valid_emails:
        if '@' in email:
            domains.append(email.split('@')[1].lower())
    
    domain_dist = Counter(domains).most_common(15)
    
    return {
        **text_analysis,
        "valid_count": len(valid_emails),
        "valid_rate": pct(len(valid_emails), len(non_blank)),
        "top_domains": [
            {"domain": domain, "count": cnt}
            for domain, cnt in domain_dist
        ],
    }

def analyze_phone_field(values: List[Any], field_name: str) -> Dict[str, Any]:
    """Phone field analysis with format patterns."""
    text_analysis = analyze_text_field(values, field_name, top_n=10)
    
    non_blank = [str(v).strip() for v in values if not is_blank(v)]
    valid_phones = [p for p in non_blank if PHONE_RE.search(p)]
    
    # Country code analysis (heuristic)
    country_codes = []
    for phone in valid_phones:
        if phone.startswith('+'):
            # Extract country code (1-3 digits after +)
            match = re.match(r'\+(\d{1,3})', phone)
            if match:
                country_codes.append(match.group(1))
    
    code_dist = Counter(country_codes).most_common(10)
    
    return {
        **text_analysis,
        "valid_count": len(valid_phones),
        "valid_rate": pct(len(valid_phones), len(non_blank)),
        "top_country_codes": [
            {"code": f"+{code}", "count": cnt}
            for code, cnt in code_dist
        ] if code_dist else [],
    }

# ========= Table-Specific Analysis =========

def analyze_trials_table(con: sqlite3.Connection) -> Dict[str, Any]:
    """Comprehensive trials table analysis."""
    cur = con.cursor()
    
    print("[INFO] Analyzing trials table...")
    
    # Get all data
    rows = safe_execute(cur, """
        SELECT ctNumber, ctStatus, ctPublicStatusCode, title, shortTitle, sponsor,
               trialPhase, therapeuticAreas, countries, decisionDate, publishDate, lastUpdated
        FROM trials
    """)
    
    if not rows:
        return {"error": "No data in trials table"}
    
    # Transpose for field analysis
    fields = ["ctNumber", "ctStatus", "ctPublicStatusCode", "title", "shortTitle", 
              "sponsor", "trialPhase", "therapeuticAreas", "countries", 
              "decisionDate", "publishDate", "lastUpdated"]
    
    field_data = {field: [row[i] for row in rows] for i, field in enumerate(fields)}
    
    analysis = {
        "total_trials": len(rows),
        "fields": {}
    }
    
    # Analyze each field with appropriate method
    print("  - Analyzing ctNumber...")
    analysis["fields"]["ctNumber"] = analyze_text_field(field_data["ctNumber"], "ctNumber", top_n=5)
    
    print("  - Analyzing ctStatus...")
    analysis["fields"]["ctStatus"] = analyze_numeric_field(field_data["ctStatus"], "ctStatus")
    
    print("  - Analyzing ctPublicStatusCode...")
    analysis["fields"]["ctPublicStatusCode"] = analyze_text_field(field_data["ctPublicStatusCode"], "ctPublicStatusCode")
    
    print("  - Analyzing title...")
    analysis["fields"]["title"] = analyze_text_field(field_data["title"], "title", top_n=5)
    
    print("  - Analyzing shortTitle...")
    analysis["fields"]["shortTitle"] = analyze_text_field(field_data["shortTitle"], "shortTitle", top_n=5)
    
    print("  - Analyzing sponsor...")
    analysis["fields"]["sponsor"] = analyze_text_field(field_data["sponsor"], "sponsor", top_n=20)
    
    print("  - Analyzing trialPhase...")
    analysis["fields"]["trialPhase"] = analyze_text_field(field_data["trialPhase"], "trialPhase")
    
    print("  - Analyzing therapeuticAreas...")
    analysis["fields"]["therapeuticAreas"] = analyze_text_field(field_data["therapeuticAreas"], "therapeuticAreas", top_n=15)
    
    print("  - Analyzing countries...")
    analysis["fields"]["countries"] = analyze_text_field(field_data["countries"], "countries", top_n=20)
    
    print("  - Analyzing decisionDate...")
    analysis["fields"]["decisionDate"] = analyze_date_field(field_data["decisionDate"], "decisionDate")
    
    print("  - Analyzing publishDate...")
    analysis["fields"]["publishDate"] = analyze_date_field(field_data["publishDate"], "publishDate")
    
    print("  - Analyzing lastUpdated...")
    analysis["fields"]["lastUpdated"] = analyze_date_field(field_data["lastUpdated"], "lastUpdated")
    
    # Additional cross-field analysis
    print("  - Performing cross-field analysis...")
    
    # Trials per sponsor
    sponsor_trial_counts = Counter([row[5] for row in rows if not is_blank(row[5])])
    analysis["sponsor_statistics"] = {
        "total_unique_sponsors": len(sponsor_trial_counts),
        "avg_trials_per_sponsor": sum(sponsor_trial_counts.values()) / len(sponsor_trial_counts) if sponsor_trial_counts else 0,
        "max_trials_single_sponsor": max(sponsor_trial_counts.values()) if sponsor_trial_counts else 0,
    }
    
    # Phase distribution with complete mapping
    phase_map = {
        "Phase I": "Phase I",
        "Phase II": "Phase II",
        "Phase III": "Phase III",
        "Phase IV": "Phase IV",
        "Expanded Access": "Expanded Access",
        "Phase I/II": "Phase I/II (Integrated)",
        "Phase II/III": "Phase II/III (Integrated)",
        "Phase III/IV": "Phase III/IV (Integrated)",
        "Non-Interventional": "Non-Interventional",
        "Compassionate Use": "Compassionate Use",
    }
    
    raw_phases = [row[6] for row in rows if not is_blank(row[6])]
    phase_counts = Counter(raw_phases)
    
    # Categorize phases for better reporting
    categorized_phases = []
    for phase, cnt in phase_counts.items():
        display_name = phase_map.get(phase, phase)  # Use mapping or original value
        categorized_phases.append({
            "phase": display_name,
            "raw_value": phase,
            "count": cnt,
            "percentage": pct(cnt, len(rows))
        })
    
    # Sort by standard phase order, then by count
    phase_order = {
        "Phase I": 1,
        "Phase I/II (Integrated)": 2,
        "Phase II": 3,
        "Phase II/III (Integrated)": 4,
        "Phase III": 5,
        "Phase III/IV (Integrated)": 6,
        "Phase IV": 7,
        "Expanded Access": 8,
        "Compassionate Use": 9,
        "Non-Interventional": 10,
    }
    
    categorized_phases.sort(key=lambda x: (phase_order.get(x["phase"], 999), -x["count"]))
    analysis["phase_distribution"] = categorized_phases
    
    # Temporal trends (trials per year)
    decision_years = [parse_year(row[9]) for row in rows]
    decision_years = [y for y in decision_years if y is not None]
    year_counts = Counter(decision_years)
    analysis["trials_per_year"] = [
        {"year": year, "count": cnt}
        for year, cnt in sorted(year_counts.items())
    ]
    
    return analysis

def analyze_people_table(con: sqlite3.Connection) -> Dict[str, Any]:
    """Comprehensive trial_people table analysis."""
    cur = con.cursor()
    
    print("[INFO] Analyzing trial_people table...")
    
    rows = safe_execute(cur, """
        SELECT ctNumber, name, role, email, phone, country, city, site_name, organisation
        FROM trial_people
    """)
    
    if not rows:
        return {"error": "No data in trial_people table"}
    
    fields = ["ctNumber", "name", "role", "email", "phone", "country", "city", "site_name", "organisation"]
    field_data = {field: [row[i] for row in rows] for i, field in enumerate(fields)}
    
    analysis = {
        "total_people": len(rows),
        "fields": {}
    }
    
    print("  - Analyzing name...")
    analysis["fields"]["name"] = analyze_text_field(field_data["name"], "name", top_n=20)
    
    print("  - Analyzing role...")
    analysis["fields"]["role"] = analyze_text_field(field_data["role"], "role", top_n=15)
    
    print("  - Analyzing email...")
    analysis["fields"]["email"] = analyze_email_field(field_data["email"], "email")
    
    print("  - Analyzing phone...")
    analysis["fields"]["phone"] = analyze_phone_field(field_data["phone"], "phone")
    
    print("  - Analyzing country...")
    analysis["fields"]["country"] = analyze_text_field(field_data["country"], "country", top_n=20)
    
    print("  - Analyzing city...")
    analysis["fields"]["city"] = analyze_text_field(field_data["city"], "city", top_n=20)
    
    print("  - Analyzing site_name...")
    analysis["fields"]["site_name"] = analyze_text_field(field_data["site_name"], "site_name", top_n=15)
    
    print("  - Analyzing organisation...")
    analysis["fields"]["organisation"] = analyze_text_field(field_data["organisation"], "organisation", top_n=20)
    
    # People per trial statistics
    print("  - Calculating people per trial statistics...")
    people_per_trial = Counter(field_data["ctNumber"])
    trial_counts = list(people_per_trial.values())
    if trial_counts:
        analysis["people_per_trial"] = {
            "min": min(trial_counts),
            "max": max(trial_counts),
            "avg": sum(trial_counts) / len(trial_counts),
            "median": sorted(trial_counts)[len(trial_counts) // 2],
        }
    
    # Role-specific statistics
    roles = [r for r in field_data["role"] if not is_blank(r)]
    role_counts = Counter(roles)
    analysis["role_statistics"] = {
        "total_unique_roles": len(role_counts),
        "role_distribution": [
            {"role": role, "count": cnt, "percentage": pct(cnt, len(roles))}
            for role, cnt in role_counts.most_common(20)
        ],
    }
    
    return analysis

def analyze_sites_table(con: sqlite3.Connection) -> Dict[str, Any]:
    """Comprehensive trial_sites table analysis."""
    cur = con.cursor()
    
    print("[INFO] Analyzing trial_sites table...")
    
    rows = safe_execute(cur, """
        SELECT ctNumber, site_name, organisation, country, city, address, postal_code
        FROM trial_sites
    """)
    
    if not rows:
        return {"error": "No data in trial_sites table"}
    
    fields = ["ctNumber", "site_name", "organisation", "country", "city", "address", "postal_code"]
    field_data = {field: [row[i] for row in rows] for i, field in enumerate(fields)}
    
    analysis = {
        "total_sites": len(rows),
        "fields": {}
    }
    
    print("  - Analyzing site_name...")
    analysis["fields"]["site_name"] = analyze_text_field(field_data["site_name"], "site_name", top_n=20)
    
    print("  - Analyzing organisation...")
    analysis["fields"]["organisation"] = analyze_text_field(field_data["organisation"], "organisation", top_n=20)
    
    print("  - Analyzing country...")
    analysis["fields"]["country"] = analyze_text_field(field_data["country"], "country", top_n=25)
    
    print("  - Analyzing city...")
    analysis["fields"]["city"] = analyze_text_field(field_data["city"], "city", top_n=25)
    
    print("  - Analyzing address...")
    analysis["fields"]["address"] = analyze_text_field(field_data["address"], "address", top_n=10)
    
    print("  - Analyzing postal_code...")
    analysis["fields"]["postal_code"] = analyze_text_field(field_data["postal_code"], "postal_code", top_n=15)
    
    # Sites per trial statistics
    print("  - Calculating sites per trial statistics...")
    sites_per_trial = Counter(field_data["ctNumber"])
    trial_counts = list(sites_per_trial.values())
    if trial_counts:
        analysis["sites_per_trial"] = {
            "min": min(trial_counts),
            "max": max(trial_counts),
            "avg": sum(trial_counts) / len(trial_counts),
            "median": sorted(trial_counts)[len(trial_counts) // 2],
            "trials_single_site": sum(1 for c in trial_counts if c == 1),
            "trials_multi_site": sum(1 for c in trial_counts if c > 1),
        }
    
    # Geographic distribution
    countries = [c for c in field_data["country"] if not is_blank(c)]
    country_counts = Counter(countries)
    analysis["geographic_distribution"] = {
        "total_unique_countries": len(country_counts),
        "top_countries": [
            {"country": country, "count": cnt, "percentage": pct(cnt, len(countries))}
            for country, cnt in country_counts.most_common(25)
        ],
    }
    
    # Site completeness
    complete_sites = sum(
        1 for row in rows
        if not is_blank(row[3]) and (not is_blank(row[4]) or not is_blank(row[5]))
    )
    analysis["data_completeness"] = {
        "complete_sites": complete_sites,
        "complete_rate": pct(complete_sites, len(rows)),
    }
    
    return analysis

def analyze_relationships(con: sqlite3.Connection) -> Dict[str, Any]:
    """Analyze relationships between tables."""
    cur = con.cursor()
    
    print("[INFO] Analyzing table relationships...")
    
    analysis = {}
    
    # Orphan checks
    orphan_people = safe_execute(cur, """
        SELECT COUNT(*) FROM trial_people p
        LEFT JOIN trials t ON t.ctNumber = p.ctNumber
        WHERE t.ctNumber IS NULL
    """)
    
    orphan_sites = safe_execute(cur, """
        SELECT COUNT(*) FROM trial_sites s
        LEFT JOIN trials t ON t.ctNumber = s.ctNumber
        WHERE t.ctNumber IS NULL
    """)
    
    analysis["referential_integrity"] = {
        "orphan_people": orphan_people[0][0] if orphan_people else 0,
        "orphan_sites": orphan_sites[0][0] if orphan_sites else 0,
    }
    
    # Trials with no sites/people
    trials_no_sites = safe_execute(cur, """
        SELECT COUNT(*) FROM trials t
        LEFT JOIN trial_sites s ON s.ctNumber = t.ctNumber
        WHERE s.ctNumber IS NULL
    """)
    
    trials_no_people = safe_execute(cur, """
        SELECT COUNT(*) FROM trials t
        LEFT JOIN trial_people p ON p.ctNumber = t.ctNumber
        WHERE p.ctNumber IS NULL
    """)
    
    total_trials = safe_execute(cur, "SELECT COUNT(*) FROM trials")
    total = total_trials[0][0] if total_trials else 0
    
    analysis["data_coverage"] = {
        "trials_without_sites": trials_no_sites[0][0] if trials_no_sites else 0,
        "trials_without_sites_pct": pct(trials_no_sites[0][0] if trials_no_sites else 0, total),
        "trials_without_people": trials_no_people[0][0] if trials_no_people else 0,
        "trials_without_people_pct": pct(trials_no_people[0][0] if trials_no_people else 0, total),
    }
    
    # Joint statistics
    trial_stats = safe_execute(cur, """
        SELECT 
            t.ctNumber,
            COUNT(DISTINCT s.id) as site_count,
            COUNT(DISTINCT p.id) as people_count
        FROM trials t
        LEFT JOIN trial_sites s ON s.ctNumber = t.ctNumber
        LEFT JOIN trial_people p ON p.ctNumber = t.ctNumber
        GROUP BY t.ctNumber
    """)
    
    if trial_stats:
        site_counts = [row[1] for row in trial_stats]
        people_counts = [row[2] for row in trial_stats]
        
        analysis["combined_statistics"] = {
            "avg_sites_per_trial": sum(site_counts) / len(site_counts),
            "avg_people_per_trial": sum(people_counts) / len(people_counts),
            "max_sites_single_trial": max(site_counts),
            "max_people_single_trial": max(people_counts),
        }
    
    return analysis

# ========= Report Generation =========

def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generate enhanced markdown report."""
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    
    lines.append(f"# CTIS Database - Enhanced Audit Report\n")
    lines.append(f"_Generated: {now}_\n\n")
    lines.append("---\n\n")
    
    # Executive Summary
    lines.append("## Executive Summary\n\n")
    if "trials" in data:
        lines.append(f"- **Total Trials**: {data['trials']['total_trials']:,}\n")
        
        # Add phase summary
        if "phase_distribution" in data["trials"] and data["trials"]["phase_distribution"]:
            top_phase = data["trials"]["phase_distribution"][0]
            lines.append(f"- **Most Common Phase**: {top_phase['phase']} ({top_phase['count']:,} trials, {top_phase['percentage']:.1f}%)\n")
            
            # Count phase categories
            standard = sum(1 for p in data["trials"]["phase_distribution"] 
                          if p['phase'] in ["Phase I", "Phase II", "Phase III", "Phase IV"])
            integrated = sum(1 for p in data["trials"]["phase_distribution"] 
                           if 'Integrated' in p['phase'])
            special = len(data["trials"]["phase_distribution"]) - standard - integrated
            
            if integrated > 0:
                lines.append(f"- **Integrated Phase Trials**: {integrated} phase type(s)\n")
            if special > 0:
                lines.append(f"- **Special Programs**: {special} program type(s)\n")
    
    if "people" in data:
        lines.append(f"- **Total People**: {data['people']['total_people']:,}\n")
    if "sites" in data:
        lines.append(f"- **Total Sites**: {data['sites']['total_sites']:,}\n")
    lines.append("\n")
    
    # Trials Analysis
    if "trials" in data and "fields" in data["trials"]:
        lines.append("## Trials Table Analysis\n\n")
        
        for field_name, field_data in data["trials"]["fields"].items():
            lines.append(f"### Field: `{field_name}`\n\n")
            lines.append(f"- **Fill Rate**: {field_data['fill_rate']:.1f}%\n")
            lines.append(f"- **Unique Values**: {field_data['unique_count']:,}\n")
            lines.append(f"- **Cardinality**: {field_data['cardinality_ratio']:.1f}%\n")
            
            if "length_stats" in field_data and field_data["length_stats"]:
                ls = field_data["length_stats"]
                lines.append(f"- **Length**: min={ls['min']}, max={ls['max']}, avg={ls['avg']:.1f}\n")
            
            if "numeric_stats" in field_data and field_data["numeric_stats"]:
                ns = field_data["numeric_stats"]
                lines.append(f"- **Range**: {ns['min']} to {ns['max']}, mean={ns['mean']:.2f}, median={ns['median']}\n")
            
            if "temporal_stats" in field_data and field_data["temporal_stats"]:
                ts = field_data["temporal_stats"]
                lines.append(f"- **Date Range**: {ts['earliest']} to {ts['latest']} ({ts['span_days']} days)\n")
            
            if "top_values" in field_data and field_data["top_values"]:
                lines.append(f"\n**Top Values**:\n\n")
                for item in field_data["top_values"][:10]:
                    if "percentage" in item:
                        lines.append(f"- {item['value']}: {item['count']:,} ({item['percentage']:.1f}%)\n")
                    else:
                        lines.append(f"- {item['value']}: {item['count']:,}\n")
            
            lines.append("\n")
        
        # Phase distribution with categorization
        if "phase_distribution" in data["trials"]:
            lines.append("### Phase Distribution\n\n")
            lines.append("This shows the distribution of trials across different clinical trial phases:\n\n")
            
            # Group phases by category
            standard_phases = []
            integrated_phases = []
            special_phases = []
            
            for item in data["trials"]["phase_distribution"]:
                phase = item['phase']
                if 'Integrated' in phase:
                    integrated_phases.append(item)
                elif phase in ["Phase I", "Phase II", "Phase III", "Phase IV"]:
                    standard_phases.append(item)
                else:
                    special_phases.append(item)
            
            if standard_phases:
                lines.append("**Standard Phases:**\n\n")
                for item in standard_phases:
                    lines.append(f"- **{item['phase']}**: {item['count']:,} trials ({item['percentage']:.1f}%)\n")
                lines.append("\n")
            
            if integrated_phases:
                lines.append("**Integrated Phases:**\n\n")
                for item in integrated_phases:
                    lines.append(f"- **{item['phase']}**: {item['count']:,} trials ({item['percentage']:.1f}%)\n")
                lines.append("\n")
            
            if special_phases:
                lines.append("**Special Programs:**\n\n")
                for item in special_phases:
                    lines.append(f"- **{item['phase']}**: {item['count']:,} trials ({item['percentage']:.1f}%)\n")
                lines.append("\n")
        
        if "trials_per_year" in data["trials"]:
            lines.append("### Trials by Decision Year\n\n")
            for item in data["trials"]["trials_per_year"][-10:]:  # Last 10 years
                lines.append(f"- **{item['year']}**: {item['count']:,}\n")
            lines.append("\n")
    
    # People Analysis
    if "people" in data and "fields" in data["people"]:
        lines.append("## Trial People Table Analysis\n\n")
        
        # Key metrics
        if "people_per_trial" in data["people"]:
            ppt = data["people"]["people_per_trial"]
            lines.append(f"### People per Trial Statistics\n\n")
            lines.append(f"- **Average**: {ppt['avg']:.1f}\n")
            lines.append(f"- **Median**: {ppt['median']}\n")
            lines.append(f"- **Range**: {ppt['min']} to {ppt['max']}\n\n")
        
        # Selected fields
        for field_name in ["role", "email", "phone", "country"]:
            if field_name in data["people"]["fields"]:
                field_data = data["people"]["fields"][field_name]
                lines.append(f"### Field: `{field_name}`\n\n")
                lines.append(f"- **Fill Rate**: {field_data['fill_rate']:.1f}%\n")
                lines.append(f"- **Unique Values**: {field_data['unique_count']:,}\n")
                
                if field_name == "email" and "valid_rate" in field_data:
                    lines.append(f"- **Valid Emails**: {field_data['valid_rate']:.1f}%\n")
                    if "top_domains" in field_data and field_data["top_domains"]:
                        lines.append(f"\n**Top Email Domains**:\n\n")
                        for item in field_data["top_domains"][:10]:
                            lines.append(f"- {item['domain']}: {item['count']:,}\n")
                
                elif field_name == "phone" and "valid_rate" in field_data:
                    lines.append(f"- **Valid Phones**: {field_data['valid_rate']:.1f}%\n")
                
                elif "top_values" in field_data and field_data["top_values"]:
                    lines.append(f"\n**Top Values**:\n\n")
                    for item in field_data["top_values"][:10]:
                        if "percentage" in item:
                            lines.append(f"- {item['value']}: {item['count']:,} ({item['percentage']:.1f}%)\n")
                
                lines.append("\n")
    
    # Sites Analysis
    if "sites" in data and "fields" in data["sites"]:
        lines.append("## Trial Sites Table Analysis\n\n")
        
        if "sites_per_trial" in data["sites"]:
            spt = data["sites"]["sites_per_trial"]
            lines.append(f"### Sites per Trial Statistics\n\n")
            lines.append(f"- **Average**: {spt['avg']:.1f}\n")
            lines.append(f"- **Median**: {spt['median']}\n")
            lines.append(f"- **Range**: {spt['min']} to {spt['max']}\n")
            lines.append(f"- **Single-site Trials**: {spt['trials_single_site']:,}\n")
            lines.append(f"- **Multi-site Trials**: {spt['trials_multi_site']:,}\n\n")
        
        if "data_completeness" in data["sites"]:
            dc = data["sites"]["data_completeness"]
            lines.append(f"### Data Completeness\n\n")
            lines.append(f"- **Complete Sites**: {dc['complete_sites']:,} ({dc['complete_rate']:.1f}%)\n\n")
        
        if "geographic_distribution" in data["sites"]:
            gd = data["sites"]["geographic_distribution"]
            lines.append(f"### Geographic Distribution\n\n")
            lines.append(f"- **Unique Countries**: {gd['total_unique_countries']}\n\n")
            lines.append(f"**Top Countries by Site Count**:\n\n")
            for item in gd["top_countries"][:15]:
                lines.append(f"- **{item['country']}**: {item['count']:,} ({item['percentage']:.1f}%)\n")
            lines.append("\n")
    
    # Relationships
    if "relationships" in data:
        lines.append("## Cross-Table Analysis\n\n")
        
        if "referential_integrity" in data["relationships"]:
            ri = data["relationships"]["referential_integrity"]
            lines.append(f"### Referential Integrity\n\n")
            lines.append(f"- **Orphan People Records**: {ri['orphan_people']:,}\n")
            lines.append(f"- **Orphan Sites Records**: {ri['orphan_sites']:,}\n\n")
        
        if "data_coverage" in data["relationships"]:
            dc = data["relationships"]["data_coverage"]
            lines.append(f"### Data Coverage\n\n")
            lines.append(f"- **Trials Without Sites**: {dc['trials_without_sites']:,} ({dc['trials_without_sites_pct']:.1f}%)\n")
            lines.append(f"- **Trials Without People**: {dc['trials_without_people']:,} ({dc['trials_without_people_pct']:.1f}%)\n\n")
        
        if "combined_statistics" in data["relationships"]:
            cs = data["relationships"]["combined_statistics"]
            lines.append(f"### Combined Statistics\n\n")
            lines.append(f"- **Average Sites per Trial**: {cs['avg_sites_per_trial']:.1f}\n")
            lines.append(f"- **Average People per Trial**: {cs['avg_people_per_trial']:.1f}\n")
            lines.append(f"- **Max Sites (Single Trial)**: {cs['max_sites_single_trial']:,}\n")
            lines.append(f"- **Max People (Single Trial)**: {cs['max_people_single_trial']:,}\n\n")
    
    lines.append("---\n\n")
    lines.append(f"_Report generated by CTIS Enhanced Audit Tool_\n")
    
    return "".join(lines)

# ========= Main =========

def main():
    print("\n" + "="*80)
    print("CTIS Database Enhanced Audit Tool")
    print("="*80 + "\n")
    
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        return 1
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        con = sqlite3.connect(DB_PATH)
        
        # Collect all analysis data
        analysis_data = {}
        
        # Analyze tables
        analysis_data["trials"] = analyze_trials_table(con)
        analysis_data["people"] = analyze_people_table(con)
        analysis_data["sites"] = analyze_sites_table(con)
        analysis_data["relationships"] = analyze_relationships(con)
        
        con.close()
        
        # Generate reports
        print("\n[INFO] Generating reports...")
        
        # Markdown report
        md_report = generate_markdown_report(analysis_data)
        REPORT_PATH.write_text(md_report, encoding="utf-8")
        print(f"[SUCCESS] Markdown report written to: {REPORT_PATH}")
        
        # JSON report (full data)
        JSON_REPORT_PATH.write_text(json.dumps(analysis_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SUCCESS] JSON report written to: {JSON_REPORT_PATH}")
        
        # Console summary
        print("\n" + "="*80)
        print("AUDIT COMPLETE")
        print("="*80)
        if "trials" in analysis_data:
            print(f"Trials analyzed: {analysis_data['trials']['total_trials']:,}")
        if "people" in analysis_data:
            print(f"People records analyzed: {analysis_data['people']['total_people']:,}")
        if "sites" in analysis_data:
            print(f"Sites analyzed: {analysis_data['sites']['total_sites']:,}")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())