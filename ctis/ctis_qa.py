#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS QA Checks - Standalone Version
Run quality checks on the CTIS database
Usage: python ctis_qa.py [path_to_database]
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime


def log(msg, level="INFO"):
    """Simple logger"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


# ===================== QA Check Functions =====================

def check_ms_status_completeness(db_path):
    """Check that MS status records have required fields"""
    issues = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check for null status
            cursor.execute("""
                SELECT ctNumber, member_state
                FROM ms_status
                WHERE status IS NULL OR status = ''
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "ms_status_completeness",
                    "severity": "ERROR",
                    "trial": row[0],
                    "country": row[1],
                    "issue": "Status is missing"
                })
            
            # Check #1: Authorised trials must have decision_date
            cursor.execute("""
                SELECT ctNumber, member_state, status
                FROM ms_status
                WHERE status LIKE '%Authorised%'
                AND (decision_date IS NULL OR decision_date = '')
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "ms_status_completeness",
                    "severity": "ERROR",
                    "trial": row[0],
                    "country": row[1],
                    "issue": f"Decision date missing for status: {row[2]}"
                })
            
            # Check #2: Recruiting trials must have start_date or recruitment_start
            cursor.execute("""
                SELECT ctNumber, member_state, status
                FROM ms_status
                WHERE status IN ('Authorised, recruiting', 'Ongoing, recruiting')
                AND (start_date IS NULL OR start_date = '')
                AND (recruitment_start IS NULL OR recruitment_start = '')
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "ms_status_completeness",
                    "severity": "WARNING",
                    "trial": row[0],
                    "country": row[1],
                    "issue": f"Recruiting but missing start_date and recruitment_start: {row[2]}"
                })
            
            # Check #3: Temporarily halted trials must have temporary_halt date
            cursor.execute("""
                SELECT ctNumber, member_state, status
                FROM ms_status
                WHERE status = 'Temporarily halted'
                AND (temporary_halt IS NULL OR temporary_halt = '')
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "ms_status_completeness",
                    "severity": "WARNING",
                    "trial": row[0],
                    "country": row[1],
                    "issue": "Temporarily halted but missing temporary_halt date"
                })
    except Exception as e:
        log(f"Error in check_ms_status_completeness: {e}", "ERROR")
    
    return issues


def check_country_planning_for_recruiting(db_path):
    """Check that recruiting countries have planned participant numbers"""
    issues = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ms.ctNumber, ms.member_state, ms.status
                FROM ms_status ms
                WHERE (ms.status LIKE '%recruiting%' OR ms.status LIKE '%Ongoing%')
                AND NOT EXISTS (
                    SELECT 1 FROM country_planning cp
                    WHERE cp.ctNumber = ms.ctNumber
                    AND cp.country = ms.member_state
                    AND cp.planned_participants IS NOT NULL
                )
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "country_planning_for_recruiting",
                    "severity": "WARNING",
                    "trial": row[0],
                    "country": row[1],
                    "issue": f"Recruiting but no planned participants ({row[2]})"
                })
    except Exception as e:
        log(f"Error in check_country_planning_for_recruiting: {e}", "ERROR")
    return issues


def check_dosage_disclosure_timing(db_path):
    """Check that dosage fields respect disclosure timing rules"""
    issues = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.ctNumber, t.trialCategory, t.expectedDosageDisclosureDate,
                       t.dosageVisibleNow, COUNT(p.id) as product_count
                FROM trials t
                LEFT JOIN trial_products p ON t.ctNumber = p.ctNumber
                WHERE t.trialCategory = '1'
                AND t.dosageVisibleNow = 0
                AND t.expectedDosageDisclosureDate IS NOT NULL
                GROUP BY t.ctNumber
            """)
            for row in cursor.fetchall():
                ct_number, category, expected_date, visible, product_count = row
                if product_count > 0:
                    cursor.execute("""
                        SELECT COUNT(*)
                        FROM trial_products
                        WHERE ctNumber = ?
                        AND (maxDailyDose IS NOT NULL AND maxDailyDose != ''
                             OR maxTreatmentPeriod IS NOT NULL)
                    """, (ct_number,))
                    dosage_count = cursor.fetchone()[0]
                    if dosage_count > 0:
                        issues.append({
                            "check": "dosage_disclosure_timing",
                            "severity": "INFO",
                            "trial": ct_number,
                            "issue": f"Category 1 trial with dosage data not yet publicly disclosed (expected: {expected_date})"
                        })
    except Exception as e:
        log(f"Error in check_dosage_disclosure_timing: {e}", "ERROR")
    return issues


def check_meddra_completeness(db_path):
    """Check that trials with medical conditions have MedDRA codes"""
    issues = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ctNumber, medicalCondition
                FROM trials
                WHERE medicalCondition IS NOT NULL 
                AND medicalCondition != ''
                AND (conditionMeddraCode IS NULL OR conditionMeddraCode = '')
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "meddra_completeness",
                    "severity": "INFO",
                    "trial": row[0],
                    "issue": f"Medical condition '{row[1][:50]}...' missing MedDRA code"
                })
    except Exception as e:
        log(f"Error in check_meddra_completeness: {e}", "ERROR")
    return issues


def check_atc_completeness(db_path):
    """Check that products have ATC codes where expected"""
    issues = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ctNumber, productName, activeSubstance
                FROM trial_products
                WHERE activeSubstance IS NOT NULL 
                AND activeSubstance != ''
                AND (atcCode IS NULL OR atcCode = '')
                AND productRole = 'test'
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "atc_completeness",
                    "severity": "INFO",
                    "trial": row[0],
                    "issue": f"Test product '{row[1]}' missing ATC code"
                })
    except Exception as e:
        log(f"Error in check_atc_completeness: {e}", "ERROR")
    return issues


def check_site_contact_completeness(db_path):
    """Check that recruiting countries have site contact information"""
    issues = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ms.ctNumber, ms.member_state, ms.status
                FROM ms_status ms
                WHERE ms.status LIKE '%recruiting%'
                AND NOT EXISTS (
                    SELECT 1 FROM site_contact sc
                    WHERE sc.ctNumber = ms.ctNumber
                    AND sc.country = ms.member_state
                    AND (sc.pi_name IS NOT NULL OR sc.pi_email IS NOT NULL)
                )
            """)
            for row in cursor.fetchall():
                issues.append({
                    "check": "site_contact_completeness",
                    "severity": "WARNING",
                    "trial": row[0],
                    "country": row[1],
                    "issue": f"Recruiting but no PI contact info ({row[2]})"
                })
    except Exception as e:
        log(f"Error in check_site_contact_completeness: {e}", "ERROR")
    return issues


# ===================== Main QA Runner =====================

def run_all_qa_checks(db_path):
    """Run all QA checks and return results"""
    log("=" * 80)
    log("Running QA Checks")
    log("=" * 80)
    
    all_checks = {
        "ms_status_completeness": check_ms_status_completeness(db_path),
        "country_planning_for_recruiting": check_country_planning_for_recruiting(db_path),
        "dosage_disclosure_timing": check_dosage_disclosure_timing(db_path),
        "meddra_completeness": check_meddra_completeness(db_path),
        "atc_completeness": check_atc_completeness(db_path),
        "site_contact_completeness": check_site_contact_completeness(db_path),
    }
    
    # Count issues by severity
    error_count = 0
    warning_count = 0
    info_count = 0
    
    for check_name, issues in all_checks.items():
        for issue in issues:
            severity = issue.get("severity", "INFO")
            if severity == "ERROR":
                error_count += 1
            elif severity == "WARNING":
                warning_count += 1
            else:
                info_count += 1
    
    log("")
    log("QA Check Summary:")
    log(f"  Errors:   {error_count}")
    log(f"  Warnings: {warning_count}")
    log(f"  Info:     {info_count}")
    log("")
    
    # Print issues grouped by severity
    for severity in ["ERROR", "WARNING", "INFO"]:
        severity_issues = []
        for check_name, issues in all_checks.items():
            for issue in issues:
                if issue.get("severity") == severity:
                    severity_issues.append((check_name, issue))
        
        if severity_issues:
            log(f"{severity}S:")
            for check_name, issue in severity_issues[:20]:  # Limit to first 20
                trial = issue.get("trial", "N/A")
                country = issue.get("country", "")
                msg = issue.get("issue", "")
                if country:
                    log(f"  [{check_name}] {trial} ({country}): {msg}")
                else:
                    log(f"  [{check_name}] {trial}: {msg}")
            
            if len(severity_issues) > 20:
                log(f"  ... and {len(severity_issues) - 20} more {severity.lower()}s")
            log("")
    
    log("=" * 80)
    
    return all_checks


def print_qa_report(db_path, output_path):
    """Generate and save detailed QA report"""
    
    # Run checks
    results = {}
    
    log("Generating detailed QA report...")
    
    results["ms_status_completeness"] = check_ms_status_completeness(db_path)
    results["country_planning_for_recruiting"] = check_country_planning_for_recruiting(db_path)
    results["dosage_disclosure_timing"] = check_dosage_disclosure_timing(db_path)
    results["meddra_completeness"] = check_meddra_completeness(db_path)
    results["atc_completeness"] = check_atc_completeness(db_path)
    results["site_contact_completeness"] = check_site_contact_completeness(db_path)
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write("CTIS QA Report\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write("=" * 80 + "\n\n")
        
        for check_name, issues in results.items():
            f.write(f"\n{check_name.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")
            
            if not issues:
                f.write("No issues found.\n")
            else:
                for issue in issues:
                    trial = issue.get("trial", "N/A")
                    country = issue.get("country", "")
                    severity = issue.get("severity", "INFO")
                    msg = issue.get("issue", "")
                    
                    if country:
                        f.write(f"[{severity}] {trial} ({country}): {msg}\n")
                    else:
                        f.write(f"[{severity}] {trial}: {msg}\n")
            
            f.write("\n")
    
    log(f"QA report saved to: {output_path}")


def main():
    """Main entry point"""
    
    # Determine database path
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        db_path = Path("ctis-out/ctis.db")
    
    # Check if database exists
    if not db_path.exists():
        log(f"Database not found: {db_path}", "ERROR")
        log("Usage: python ctis_qa.py [path_to_database]", "INFO")
        log("Example: python ctis_qa.py ctis-out/ctis.db", "INFO")
        return 1
    
    log(f"Checking database: {db_path}")
    log("")
    
    # Run all checks
    results = run_all_qa_checks(db_path)
    
    # Save detailed report
    report_path = db_path.parent / "qa_report.txt"
    print_qa_report(db_path, report_path)
    
    # Calculate totals
    total_issues = sum(len(issues) for issues in results.values())
    
    log("")
    log("=" * 80)
    log(f"QA checks complete. Total issues found: {total_issues}")
    if total_issues == 0:
        log("All checks passed! Database quality is excellent.")
    else:
        log(f"See detailed report: {report_path}")
    log("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())