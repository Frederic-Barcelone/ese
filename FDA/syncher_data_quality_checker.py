#!/usr/bin/env python3
"""
FDA Data Quality Control Script
================================
Validates downloaded data quality, provides KPIs, and identifies issues.

Usage:
    python data_quality_checker.py

Output:
    - Console report with KPIs
    - Issues log: data_quality_report.json
"""

import json
import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
import PyPDF2
from collections import defaultdict
import re

# Import configuration
from syncher_keys import OUTPUT_DIR, SYNC_AREAS
from syncher_therapeutic_areas import THERAPEUTIC_AREAS

# ============================================================================
# QUALITY CHECK CONFIGURATION
# ============================================================================

# Expected minimum values
EXPECTED_MINIMUMS = {
    'nephrology': {
        'labels': 150,  # Minimum expected drug labels
        'orphan_drugs': 80,  # Minimum orphan designations
        'approval_packages': 20,  # Minimum approval packages with PDFs
        'adverse_events': 100,  # Minimum adverse event reports
        'enforcement': 5  # Minimum enforcement reports
    },
    'hematology': {
        'labels': 200,
        'orphan_drugs': 120,
        'approval_packages': 30,
        'adverse_events': 200,
        'enforcement': 10
    }
}

# Critical JSON fields that must exist
REQUIRED_LABEL_FIELDS = [
    'openfda',
    'indications_and_usage',
    'set_id'
]

# Expected document types in approval packages
EXPECTED_DOC_TYPES = [
    'approval_letter',
    'label',
    'medical_review',
    'clinical_pharm_review',
    'statistical_review'
]

# ============================================================================
# QUALITY CHECKER CLASS
# ============================================================================

class FDADataQualityChecker:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.issues = []
        self.kpis = {}
        self.warnings = []
        self.timestamp = datetime.now()
        
    def log_issue(self, severity, category, message, details=None):
        """Log a quality issue"""
        issue = {
            'severity': severity,  # 'critical', 'warning', 'info'
            'category': category,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        if severity == 'critical':
            self.issues.append(issue)
        elif severity == 'warning':
            self.warnings.append(issue)
        
        # Print to console
        icon = '🔴' if severity == 'critical' else '⚠️' if severity == 'warning' else 'ℹ️'
        print(f"  {icon} [{category}] {message}")
        if details:
            print(f"      Details: {details}")
    
    def log_kpi(self, category, name, value, expected=None, status='ok'):
        """Log a KPI metric"""
        if category not in self.kpis:
            self.kpis[category] = {}
        
        self.kpis[category][name] = {
            'value': value,
            'expected': expected,
            'status': status  # 'ok', 'warning', 'critical'
        }
    
    # ========== CHECK 1: DRUG LABELS (JSON) ==========
    def check_drug_labels(self, therapeutic_area):
        """Validate drug label JSON files"""
        print(f"\n{'='*70}")
        print(f"CHECKING DRUG LABELS: {therapeutic_area.upper()}")
        print(f"{'='*70}")
        
        label_files = glob.glob(f"{self.output_dir}/labels/{therapeutic_area}_labels_*.json")
        
        if not label_files:
            self.log_issue('critical', 'labels', 
                          f"No label files found for {therapeutic_area}",
                          f"Expected files in: {self.output_dir}/labels/")
            self.log_kpi(therapeutic_area, 'labels_count', 0, 
                        EXPECTED_MINIMUMS[therapeutic_area]['labels'], 'critical')
            return
        
        # Use most recent file
        latest_file = max(label_files, key=os.path.getmtime)
        print(f"  Checking: {os.path.basename(latest_file)}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        except json.JSONDecodeError as e:
            self.log_issue('critical', 'labels', 
                          f"Invalid JSON in {latest_file}", str(e))
            return
        except Exception as e:
            self.log_issue('critical', 'labels', 
                          f"Cannot read {latest_file}", str(e))
            return
        
        # KPI: Total labels
        total_labels = len(labels)
        expected = EXPECTED_MINIMUMS[therapeutic_area]['labels']
        status = 'ok' if total_labels >= expected else 'critical'
        self.log_kpi(therapeutic_area, 'labels_count', total_labels, expected, status)
        
        if total_labels < expected:
            self.log_issue('critical', 'labels',
                          f"Only {total_labels} labels found, expected at least {expected}")
        else:
            print(f"  ✓ Found {total_labels} drug labels (expected ≥{expected})")
        
        # Check individual labels
        labels_with_issues = 0
        labels_missing_key_fields = 0
        labels_with_empty_content = 0
        drug_names = set()
        
        for i, label in enumerate(labels):
            # Check required fields
            missing_fields = []
            for field in REQUIRED_LABEL_FIELDS:
                if field not in label or not label[field]:
                    missing_fields.append(field)
            
            if missing_fields:
                labels_with_issues += 1
                labels_missing_key_fields += 1
                if labels_with_issues <= 5:  # Only log first 5
                    self.log_issue('warning', 'labels',
                                  f"Label {i} missing fields: {missing_fields}",
                                  f"set_id: {label.get('set_id', 'unknown')}")
            
            # Check for empty critical content
            if 'indications_and_usage' in label:
                content = label['indications_and_usage']
                if isinstance(content, list):
                    content = ' '.join(content)
                if len(str(content).strip()) < 50:
                    labels_with_empty_content += 1
                    if labels_with_empty_content <= 3:
                        drug_name = label.get('openfda', {}).get('brand_name', ['unknown'])[0]
                        self.log_issue('warning', 'labels',
                                      f"Label has minimal content: {drug_name}")
            
            # Collect drug names
            if 'openfda' in label and 'brand_name' in label['openfda']:
                drug_names.update(label['openfda']['brand_name'])
        
        # KPIs
        self.log_kpi(therapeutic_area, 'labels_with_issues', labels_with_issues, 
                    0, 'warning' if labels_with_issues > total_labels * 0.05 else 'ok')
        self.log_kpi(therapeutic_area, 'unique_drugs', len(drug_names))
        
        print(f"  ✓ Unique drugs: {len(drug_names)}")
        print(f"  ✓ Labels with issues: {labels_with_issues}/{total_labels} ({labels_with_issues/total_labels*100:.1f}%)")
        
        # Check for expected diseases
        diseases = THERAPEUTIC_AREAS[therapeutic_area]['rare_diseases']
        disease_coverage = self.check_disease_coverage(labels, diseases[:10])  # Check first 10
        self.log_kpi(therapeutic_area, 'disease_coverage', f"{disease_coverage['covered']}/{disease_coverage['checked']}")
        
        return {
            'total': total_labels,
            'unique_drugs': len(drug_names),
            'with_issues': labels_with_issues
        }
    
    def check_disease_coverage(self, labels, diseases):
        """Check which diseases are covered in the labels"""
        covered = 0
        all_text = ' '.join([json.dumps(label).lower() for label in labels])
        
        for disease in diseases:
            if disease.lower() in all_text:
                covered += 1
        
        return {'covered': covered, 'checked': len(diseases)}
    
    # ========== CHECK 2: ORPHAN DRUGS (CSV) ==========
    def check_orphan_drugs(self, therapeutic_area):
        """Validate orphan drug CSV files"""
        print(f"\n{'='*70}")
        print(f"CHECKING ORPHAN DRUGS: {therapeutic_area.upper()}")
        print(f"{'='*70}")
        
        orphan_files = glob.glob(f"{self.output_dir}/orphan_drugs/{therapeutic_area}_orphan_drugs_*.csv")
        
        if not orphan_files:
            self.log_issue('warning', 'orphan_drugs',
                          f"No orphan drug files found for {therapeutic_area}")
            self.log_kpi(therapeutic_area, 'orphan_drugs_count', 0,
                        EXPECTED_MINIMUMS[therapeutic_area]['orphan_drugs'], 'warning')
            return
        
        latest_file = max(orphan_files, key=os.path.getmtime)
        print(f"  Checking: {os.path.basename(latest_file)}")
        
        try:
            df = pd.read_csv(latest_file)
        except Exception as e:
            self.log_issue('critical', 'orphan_drugs',
                          f"Cannot read {latest_file}", str(e))
            return
        
        # KPI: Total orphan drugs
        total_orphan = len(df)
        expected = EXPECTED_MINIMUMS[therapeutic_area]['orphan_drugs']
        status = 'ok' if total_orphan >= expected else 'warning'
        self.log_kpi(therapeutic_area, 'orphan_drugs_count', total_orphan, expected, status)
        
        if total_orphan < expected:
            self.log_issue('warning', 'orphan_drugs',
                          f"Only {total_orphan} orphan drugs found, expected at least {expected}")
        else:
            print(f"  ✓ Found {total_orphan} orphan drug designations (expected ≥{expected})")
        
        # Check required columns
        required_cols = ['generic_name', 'indication', 'sponsor', 'designation_status']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.log_issue('critical', 'orphan_drugs',
                          f"Missing required columns: {missing_cols}")
        
        # Check data quality
        if 'designation_status' in df.columns:
            approved_count = len(df[df['designation_status'] == 'Approved'])
            designated_count = len(df[df['designation_status'] == 'Designated'])
            
            self.log_kpi(therapeutic_area, 'orphan_approved', approved_count)
            self.log_kpi(therapeutic_area, 'orphan_designated_only', designated_count)
            
            print(f"  ✓ Approved: {approved_count}")
            print(f"  ✓ Designated only: {designated_count}")
        
        # Check for empty/null values
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                empty_count = (df[col] == '').sum()
                total_bad = null_count + empty_count
                
                if total_bad > 0:
                    self.log_issue('warning', 'orphan_drugs',
                                  f"Column '{col}' has {total_bad} empty/null values")
        
        return {
            'total': total_orphan,
            'approved': approved_count if 'designation_status' in df.columns else 0
        }
    
    # ========== CHECK 3: APPROVAL PACKAGES (PDFs) ==========
    def check_approval_packages(self, therapeutic_area):
        """Validate approval package PDFs"""
        print(f"\n{'='*70}")
        print(f"CHECKING APPROVAL PACKAGES: {therapeutic_area.upper()}")
        print(f"{'='*70}")
        
        packages_dir = f"{self.output_dir}/approval_packages/{therapeutic_area}"
        
        if not os.path.exists(packages_dir):
            self.log_issue('warning', 'approval_packages',
                          f"No approval packages directory found for {therapeutic_area}",
                          f"Expected: {packages_dir}")
            self.log_kpi(therapeutic_area, 'approval_packages_count', 0,
                        EXPECTED_MINIMUMS[therapeutic_area]['approval_packages'], 'warning')
            return
        
        # Find all drug folders
        drug_folders = [f for f in os.listdir(packages_dir) 
                       if os.path.isdir(os.path.join(packages_dir, f))]
        
        if not drug_folders:
            self.log_issue('warning', 'approval_packages',
                          f"No drug folders found in {packages_dir}")
            self.log_kpi(therapeutic_area, 'approval_packages_count', 0,
                        EXPECTED_MINIMUMS[therapeutic_area]['approval_packages'], 'warning')
            return
        
        print(f"  Found {len(drug_folders)} drug packages")
        
        total_pdfs = 0
        total_size_mb = 0
        corrupted_pdfs = 0
        packages_by_doc_type = defaultdict(int)
        packages_missing_key_docs = []
        
        for drug_folder in drug_folders:
            drug_path = os.path.join(packages_dir, drug_folder)
            pdfs = glob.glob(f"{drug_path}/**/*.pdf", recursive=True)
            
            if not pdfs:
                self.log_issue('warning', 'approval_packages',
                              f"No PDFs found for {drug_folder}")
                continue
            
            total_pdfs += len(pdfs)
            
            # Check document types present
            doc_types_present = set()
            
            for pdf_path in pdfs:
                # Check file size
                size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                total_size_mb += size_mb
                
                if size_mb < 0.01:  # Less than 10KB - likely corrupt
                    corrupted_pdfs += 1
                    self.log_issue('critical', 'approval_packages',
                                  f"Suspiciously small PDF: {os.path.basename(pdf_path)}",
                                  f"Size: {size_mb:.3f} MB in {drug_folder}")
                
                # Try to open PDF
                try:
                    with open(pdf_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        page_count = len(pdf_reader.pages)
                        
                        if page_count == 0:
                            corrupted_pdfs += 1
                            self.log_issue('critical', 'approval_packages',
                                          f"PDF has 0 pages: {os.path.basename(pdf_path)}",
                                          f"File: {drug_folder}")
                except Exception as e:
                    corrupted_pdfs += 1
                    self.log_issue('critical', 'approval_packages',
                                  f"Cannot read PDF: {os.path.basename(pdf_path)}",
                                  f"Error: {str(e)}")
                
                # Determine document type from path
                path_parts = pdf_path.split(os.sep)
                if len(path_parts) >= 2:
                    doc_type = path_parts[-2]  # Parent folder name
                    doc_types_present.add(doc_type)
                    packages_by_doc_type[doc_type] += 1
            
            # Check for missing key document types
            missing_types = [dt for dt in EXPECTED_DOC_TYPES if dt not in doc_types_present]
            if missing_types:
                packages_missing_key_docs.append({
                    'drug': drug_folder,
                    'missing': missing_types,
                    'has': list(doc_types_present)
                })
        
        # KPIs
        expected = EXPECTED_MINIMUMS[therapeutic_area]['approval_packages']
        status = 'ok' if len(drug_folders) >= expected else 'warning'
        self.log_kpi(therapeutic_area, 'approval_packages_count', len(drug_folders), expected, status)
        self.log_kpi(therapeutic_area, 'total_pdfs', total_pdfs)
        self.log_kpi(therapeutic_area, 'total_pdf_size_mb', round(total_size_mb, 2))
        self.log_kpi(therapeutic_area, 'corrupted_pdfs', corrupted_pdfs, 0, 
                    'critical' if corrupted_pdfs > 0 else 'ok')
        
        print(f"  ✓ Drug packages: {len(drug_folders)} (expected ≥{expected})")
        print(f"  ✓ Total PDFs: {total_pdfs}")
        print(f"  ✓ Total size: {total_size_mb:.2f} MB")
        print(f"  ✓ Corrupted PDFs: {corrupted_pdfs}")
        
        # Report document type distribution
        print(f"\n  Document type distribution:")
        for doc_type, count in sorted(packages_by_doc_type.items()):
            print(f"    - {doc_type.replace('_', ' ').title()}: {count}")
        
        # Report packages missing key documents
        if packages_missing_key_docs:
            missing_count = len(packages_missing_key_docs)
            self.log_issue('warning', 'approval_packages',
                          f"{missing_count} packages missing key document types")
            
            if missing_count <= 5:
                for pkg in packages_missing_key_docs:
                    self.log_issue('info', 'approval_packages',
                                  f"{pkg['drug']} missing: {', '.join(pkg['missing'])}")
        
        return {
            'packages': len(drug_folders),
            'pdfs': total_pdfs,
            'size_mb': total_size_mb,
            'corrupted': corrupted_pdfs
        }
    
    # ========== CHECK 4: ADVERSE EVENTS (JSON) ==========
    def check_adverse_events(self, therapeutic_area):
        """Validate adverse events JSON files"""
        print(f"\n{'='*70}")
        print(f"CHECKING ADVERSE EVENTS: {therapeutic_area.upper()}")
        print(f"{'='*70}")
        
        ae_files = glob.glob(f"{self.output_dir}/adverse_events/{therapeutic_area}_adverse_events_*.json")
        
        if not ae_files:
            self.log_issue('warning', 'adverse_events',
                          f"No adverse event files found for {therapeutic_area}")
            self.log_kpi(therapeutic_area, 'adverse_events_count', 0,
                        EXPECTED_MINIMUMS[therapeutic_area]['adverse_events'], 'warning')
            return
        
        latest_file = max(ae_files, key=os.path.getmtime)
        print(f"  Checking: {os.path.basename(latest_file)}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
        except Exception as e:
            self.log_issue('critical', 'adverse_events',
                          f"Cannot read {latest_file}", str(e))
            return
        
        # KPI: Total events
        total_events = len(events)
        expected = EXPECTED_MINIMUMS[therapeutic_area]['adverse_events']
        status = 'ok' if total_events >= expected else 'warning'
        self.log_kpi(therapeutic_area, 'adverse_events_count', total_events, expected, status)
        
        print(f"  ✓ Found {total_events} adverse event reports (expected ≥{expected})")
        
        # Analyze events
        if events:
            drugs_with_events = set()
            serious_events = 0
            
            for event in events:
                if 'query_drug' in event:
                    drugs_with_events.add(event['query_drug'])
                if event.get('serious') == '1':
                    serious_events += 1
            
            self.log_kpi(therapeutic_area, 'drugs_with_ae', len(drugs_with_events))
            self.log_kpi(therapeutic_area, 'serious_ae', serious_events)
            
            print(f"  ✓ Drugs with events: {len(drugs_with_events)}")
            print(f"  ✓ Serious events: {serious_events} ({serious_events/total_events*100:.1f}%)")
        
        return {
            'total': total_events,
            'serious': serious_events if events else 0
        }
    
    # ========== CHECK 5: ENFORCEMENT REPORTS (JSON) ==========
    def check_enforcement_reports(self, therapeutic_area):
        """Validate enforcement reports JSON files"""
        print(f"\n{'='*70}")
        print(f"CHECKING ENFORCEMENT REPORTS: {therapeutic_area.upper()}")
        print(f"{'='*70}")
        
        enforcement_files = glob.glob(f"{self.output_dir}/enforcement/{therapeutic_area}_enforcement_*.json")
        
        if not enforcement_files:
            self.log_issue('info', 'enforcement',
                          f"No enforcement files found for {therapeutic_area}")
            self.log_kpi(therapeutic_area, 'enforcement_count', 0,
                        EXPECTED_MINIMUMS[therapeutic_area]['enforcement'], 'info')
            return
        
        latest_file = max(enforcement_files, key=os.path.getmtime)
        print(f"  Checking: {os.path.basename(latest_file)}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                reports = json.load(f)
        except Exception as e:
            self.log_issue('critical', 'enforcement',
                          f"Cannot read {latest_file}", str(e))
            return
        
        # KPI: Total reports
        total_reports = len(reports)
        expected = EXPECTED_MINIMUMS[therapeutic_area]['enforcement']
        status = 'ok' if total_reports >= expected else 'info'
        self.log_kpi(therapeutic_area, 'enforcement_count', total_reports, expected, status)
        
        print(f"  ✓ Found {total_reports} enforcement reports (expected ≥{expected})")
        
        # Analyze reports by classification
        if reports:
            by_class = defaultdict(int)
            for report in reports:
                classification = report.get('classification', 'unknown')
                by_class[classification] += 1
            
            print(f"  ✓ By classification:")
            for cls, count in sorted(by_class.items()):
                print(f"    - {cls}: {count}")
        
        return {
            'total': total_reports
        }
    
    # ========== MAIN CHECK FUNCTION ==========
    def run_all_checks(self):
        """Run all quality checks"""
        print(f"\n{'#'*70}")
        print(f"FDA DATA QUALITY CONTROL")
        print(f"Started: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'#'*70}")
        
        results = {}
        
        for area in SYNC_AREAS:
            results[area] = {}
            
            # Check 1: Drug Labels
            results[area]['labels'] = self.check_drug_labels(area)
            
            # Check 2: Orphan Drugs
            results[area]['orphan_drugs'] = self.check_orphan_drugs(area)
            
            # Check 3: Approval Packages
            results[area]['approval_packages'] = self.check_approval_packages(area)
            
            # Check 4: Adverse Events
            results[area]['adverse_events'] = self.check_adverse_events(area)
            
            # Check 5: Enforcement Reports
            results[area]['enforcement'] = self.check_enforcement_reports(area)
        
        return results
    
    # ========== REPORT GENERATION ==========
    def generate_summary_report(self, results):
        """Generate summary report"""
        print(f"\n{'='*70}")
        print(f"QUALITY CONTROL SUMMARY")
        print(f"{'='*70}")
        
        # Overall status
        critical_count = len(self.issues)
        warning_count = len(self.warnings)
        
        if critical_count == 0 and warning_count == 0:
            print(f"\n✅ ALL CHECKS PASSED - Data quality is excellent!")
        elif critical_count == 0:
            print(f"\n⚠️  {warning_count} WARNINGS - Data quality is good with minor issues")
        else:
            print(f"\n🔴 {critical_count} CRITICAL ISSUES - Data quality needs attention!")
            print(f"⚠️  {warning_count} warnings")
        
        # Print KPI summary
        print(f"\n{'='*70}")
        print(f"KEY PERFORMANCE INDICATORS")
        print(f"{'='*70}")
        
        for area, kpis in self.kpis.items():
            print(f"\n{area.upper()}:")
            for name, data in kpis.items():
                value = data['value']
                expected = data.get('expected')
                status = data['status']
                
                status_icon = '✓' if status == 'ok' else '⚠️' if status == 'warning' else '✗'
                
                if expected:
                    print(f"  {status_icon} {name}: {value} (expected ≥{expected})")
                else:
                    print(f"  {status_icon} {name}: {value}")
        
        # Print issues
        if self.issues:
            print(f"\n{'='*70}")
            print(f"CRITICAL ISSUES ({len(self.issues)})")
            print(f"{'='*70}")
            
            for i, issue in enumerate(self.issues, 1):
                print(f"\n{i}. [{issue['category']}] {issue['message']}")
                if issue['details']:
                    print(f"   Details: {issue['details']}")
        
        if self.warnings:
            print(f"\n{'='*70}")
            print(f"WARNINGS ({len(self.warnings)})")
            print(f"{'='*70}")
            
            for i, warning in enumerate(self.warnings[:10], 1):  # Show first 10
                print(f"\n{i}. [{warning['category']}] {warning['message']}")
            
            if len(self.warnings) > 10:
                print(f"\n... and {len(self.warnings) - 10} more warnings")
        
        # Save detailed report
        self.save_json_report()
    
    def save_json_report(self):
        """Save detailed JSON report"""
        report = {
            'timestamp': self.timestamp.isoformat(),
            'output_dir': self.output_dir,
            'summary': {
                'critical_issues': len(self.issues),
                'warnings': len(self.warnings),
                'status': 'pass' if len(self.issues) == 0 else 'fail'
            },
            'kpis': self.kpis,
            'issues': self.issues,
            'warnings': self.warnings
        }
        
        report_file = 'data_quality_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run quality checks"""
    checker = FDADataQualityChecker()
    results = checker.run_all_checks()
    checker.generate_summary_report(results)
    
    # Exit code based on results
    if checker.issues:
        exit(1)  # Critical issues found
    else:
        exit(0)  # All good

if __name__ == "__main__":
    main()