"""
FDA Data Quality Checker - Comprehensive Analysis
==================================================
Analyzes all FDA_DATA folders for:
- Duplicates
- Data integrity issues
- Structure validation
- File health
- Size analysis
- Coverage gaps

Usage:
    python fda_data_quality_checker.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import hashlib

class FDADataQualityChecker:
    """Comprehensive quality checker for FDA data"""
    
    def __init__(self, data_dir="FDA_DATA"):
        self.data_dir = Path(data_dir)
        self.issues = defaultdict(list)
        self.stats = defaultdict(dict)
        self.duplicates = defaultdict(list)
        
    def run_all_checks(self):
        """Run all quality checks"""
        print("\n" + "="*80)
        print("FDA DATA QUALITY CHECKER")
        print("="*80)
        print(f"Analyzing: {self.data_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        # Run all checks
        self.check_directory_structure()
        self.check_labels()
        self.check_adverse_events()
        self.check_enforcement()
        self.check_approval_packages()
        self.check_cross_data_consistency()
        
        # Generate report
        self.generate_report()
        self.save_report()
    
    def check_directory_structure(self):
        """Check if all expected directories exist"""
        print("📁 Checking directory structure...")
        
        expected_dirs = [
            'labels',
            'adverse_events',
            'enforcement',
            'approval_packages'
        ]
        
        for dir_name in expected_dirs:
            dir_path = self.data_dir / dir_name
            if dir_path.exists():
                size = self._get_dir_size(dir_path)
                file_count = len(list(dir_path.rglob('*')))
                self.stats['directories'][dir_name] = {
                    'exists': True,
                    'size_gb': size / (1024**3),
                    'file_count': file_count
                }
                print(f"  ✅ {dir_name}: {size/(1024**3):.2f} GB, {file_count} files")
            else:
                self.issues['missing_directories'].append(dir_name)
                print(f"  ❌ {dir_name}: NOT FOUND")
        
        print()
    
    def check_labels(self):
        """Check drug labels data quality"""
        print("💊 Checking drug labels...")
        
        labels_dir = self.data_dir / 'labels'
        if not labels_dir.exists():
            self.issues['labels'].append("Labels directory not found")
            return
        
        # Find all label JSON files
        label_files = list(labels_dir.glob('*_labels_*.json'))
        
        if not label_files:
            self.issues['labels'].append("No label files found")
            return
        
        total_labels = 0
        duplicate_set_ids = []
        seen_set_ids = set()
        missing_fields = defaultdict(int)
        
        for file_path in label_files:
            try:
                print(f"  📄 Analyzing: {file_path.name}")
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    self.issues['labels'].append(f"{file_path.name}: Not a list")
                    continue
                
                file_label_count = len(data)
                total_labels += file_label_count
                
                # Check each label
                for idx, label in enumerate(data):
                    if not isinstance(label, dict):
                        self.issues['labels'].append(f"{file_path.name}[{idx}]: Not a dict")
                        continue
                    
                    # Check for duplicate set_id
                    set_id = label.get('set_id')
                    if set_id:
                        if set_id in seen_set_ids:
                            duplicate_set_ids.append(set_id)
                        seen_set_ids.add(set_id)
                    else:
                        missing_fields['set_id'] += 1
                    
                    # Check for required fields
                    required_fields = ['id', 'openfda']
                    for field in required_fields:
                        if field not in label:
                            missing_fields[field] += 1
                    
                    # Check openfda structure
                    if 'openfda' in label:
                        openfda = label['openfda']
                        if not isinstance(openfda, dict):
                            self.issues['labels'].append(f"{file_path.name}[{idx}]: openfda not a dict")
                
                print(f"    ✅ {file_label_count:,} labels")
                
            except json.JSONDecodeError as e:
                self.issues['labels'].append(f"{file_path.name}: JSON decode error - {e}")
            except Exception as e:
                self.issues['labels'].append(f"{file_path.name}: Error - {e}")
        
        # Store stats
        self.stats['labels'] = {
            'total_labels': total_labels,
            'unique_set_ids': len(seen_set_ids),
            'duplicate_set_ids': len(duplicate_set_ids),
            'missing_fields': dict(missing_fields)
        }
        
        # Report duplicates
        if duplicate_set_ids:
            self.duplicates['labels'] = duplicate_set_ids[:10]  # First 10
            print(f"  ⚠️  Found {len(duplicate_set_ids)} duplicate set_ids")
        
        print(f"  📊 Total: {total_labels:,} labels, {len(seen_set_ids):,} unique")
        print()
    
    def check_adverse_events(self):
        """Check adverse events data quality"""
        print("🚨 Checking adverse events...")
        
        ae_dir = self.data_dir / 'adverse_events'
        if not ae_dir.exists():
            self.issues['adverse_events'].append("Adverse events directory not found")
            return
        
        # Find all adverse event JSON files (excluding progress files)
        ae_files = [f for f in ae_dir.glob('*_adverse_events_*.json') 
                    if not f.parent.name.startswith('.')]
        
        if not ae_files:
            self.issues['adverse_events'].append("No adverse event files found")
            return
        
        total_events = 0
        duplicate_report_ids = []
        seen_report_ids = set()
        missing_fields = defaultdict(int)
        drugs_covered = set()
        
        for file_path in ae_files:
            try:
                print(f"  📄 Analyzing: {file_path.name}")
                print(f"     Size: {file_path.stat().st_size / (1024**3):.2f} GB")
                
                # Load with progress indicator for large files
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    self.issues['adverse_events'].append(f"{file_path.name}: Not a list")
                    continue
                
                file_event_count = len(data)
                total_events += file_event_count
                
                # Sample check (first 1000 events for speed)
                sample_size = min(1000, len(data))
                print(f"     Checking {sample_size:,} events (sample)...")
                
                for idx, event in enumerate(data[:sample_size]):
                    if not isinstance(event, dict):
                        self.issues['adverse_events'].append(f"{file_path.name}[{idx}]: Not a dict")
                        continue
                    
                    # Check for duplicate safetyreportid
                    report_id = event.get('safetyreportid')
                    if report_id:
                        if report_id in seen_report_ids:
                            duplicate_report_ids.append(report_id)
                        seen_report_ids.add(report_id)
                    else:
                        missing_fields['safetyreportid'] += 1
                    
                    # Check for required fields
                    required_fields = ['receivedate', 'patient']
                    for field in required_fields:
                        if field not in event:
                            missing_fields[field] += 1
                    
                    # Track drugs covered
                    if 'query_drug' in event:
                        drugs_covered.add(event['query_drug'])
                
                print(f"    ✅ {file_event_count:,} events")
                
            except json.JSONDecodeError as e:
                self.issues['adverse_events'].append(f"{file_path.name}: JSON decode error - {e}")
            except Exception as e:
                self.issues['adverse_events'].append(f"{file_path.name}: Error - {e}")
        
        # Store stats
        self.stats['adverse_events'] = {
            'total_events': total_events,
            'unique_report_ids': len(seen_report_ids),
            'duplicate_report_ids': len(duplicate_report_ids),
            'drugs_covered': len(drugs_covered),
            'missing_fields': dict(missing_fields)
        }
        
        # Report duplicates
        if duplicate_report_ids:
            self.duplicates['adverse_events'] = duplicate_report_ids[:10]
            print(f"  ⚠️  Found {len(duplicate_report_ids)} duplicate report_ids")
        
        print(f"  📊 Total: {total_events:,} events, {len(seen_report_ids):,} unique")
        print(f"  💊 Drugs covered: {len(drugs_covered)}")
        print()
    
    def check_enforcement(self):
        """Check enforcement reports data quality"""
        print("📋 Checking enforcement reports...")
        
        enf_dir = self.data_dir / 'enforcement'
        if not enf_dir.exists():
            self.issues['enforcement'].append("Enforcement directory not found")
            return
        
        # Find all enforcement JSON files
        enf_files = list(enf_dir.glob('*_enforcement_*.json'))
        
        if not enf_files:
            self.issues['enforcement'].append("No enforcement files found")
            return
        
        total_reports = 0
        duplicate_recall_numbers = []
        seen_recall_numbers = set()
        missing_fields = defaultdict(int)
        
        for file_path in enf_files:
            try:
                print(f"  📄 Analyzing: {file_path.name}")
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    self.issues['enforcement'].append(f"{file_path.name}: Not a list")
                    continue
                
                file_report_count = len(data)
                total_reports += file_report_count
                
                # Check each report
                for idx, report in enumerate(data):
                    if not isinstance(report, dict):
                        self.issues['enforcement'].append(f"{file_path.name}[{idx}]: Not a dict")
                        continue
                    
                    # Check for duplicate recall_number
                    recall_number = report.get('recall_number')
                    if recall_number:
                        if recall_number in seen_recall_numbers:
                            duplicate_recall_numbers.append(recall_number)
                        seen_recall_numbers.add(recall_number)
                    else:
                        missing_fields['recall_number'] += 1
                    
                    # Check for required fields
                    required_fields = ['report_date', 'product_description', 'reason_for_recall']
                    for field in required_fields:
                        if field not in report:
                            missing_fields[field] += 1
                
                print(f"    ✅ {file_report_count:,} reports")
                
            except json.JSONDecodeError as e:
                self.issues['enforcement'].append(f"{file_path.name}: JSON decode error - {e}")
            except Exception as e:
                self.issues['enforcement'].append(f"{file_path.name}: Error - {e}")
        
        # Store stats
        self.stats['enforcement'] = {
            'total_reports': total_reports,
            'unique_recall_numbers': len(seen_recall_numbers),
            'duplicate_recall_numbers': len(duplicate_recall_numbers),
            'missing_fields': dict(missing_fields)
        }
        
        # Report duplicates
        if duplicate_recall_numbers:
            self.duplicates['enforcement'] = duplicate_recall_numbers[:10]
            print(f"  ⚠️  Found {len(duplicate_recall_numbers)} duplicate recall_numbers")
        
        print(f"  📊 Total: {total_reports:,} reports, {len(seen_recall_numbers):,} unique")
        print()
    
    def check_approval_packages(self):
        """Check approval packages data quality"""
        print("📦 Checking approval packages...")
        
        packages_dir = self.data_dir / 'approval_packages'
        if not packages_dir.exists():
            self.issues['approval_packages'].append("Approval packages directory not found")
            return
        
        # Find all therapeutic areas
        therapeutic_areas = [d for d in packages_dir.iterdir() if d.is_dir()]
        
        if not therapeutic_areas:
            self.issues['approval_packages'].append("No therapeutic area folders found")
            return
        
        total_packages = 0
        total_pdfs = 0
        empty_packages = []
        missing_index = []
        
        for area in therapeutic_areas:
            print(f"  📂 {area.name}:")
            
            # Find all drug packages
            drug_packages = [d for d in area.iterdir() if d.is_dir()]
            area_package_count = len(drug_packages)
            total_packages += area_package_count
            
            for package in drug_packages:
                # Count PDFs
                pdfs = list(package.rglob('*.pdf'))
                pdf_count = len(pdfs)
                total_pdfs += pdf_count
                
                if pdf_count == 0:
                    empty_packages.append(str(package.relative_to(packages_dir)))
                
                # Check for INDEX.md
                index_file = package / 'INDEX.md'
                if not index_file.exists():
                    missing_index.append(str(package.relative_to(packages_dir)))
            
            print(f"    ✅ {area_package_count} packages")
        
        # Store stats
        self.stats['approval_packages'] = {
            'total_packages': total_packages,
            'total_pdfs': total_pdfs,
            'empty_packages': len(empty_packages),
            'missing_index': len(missing_index),
            'avg_pdfs_per_package': total_pdfs / total_packages if total_packages > 0 else 0
        }
        
        # Report issues
        if empty_packages:
            self.issues['approval_packages'].extend([f"Empty package: {p}" for p in empty_packages[:5]])
            print(f"  ⚠️  {len(empty_packages)} empty packages")
        
        if missing_index:
            print(f"  ⚠️  {len(missing_index)} packages missing INDEX.md")
        
        print(f"  📊 Total: {total_packages} packages, {total_pdfs:,} PDFs")
        print(f"  📄 Average: {total_pdfs/total_packages:.1f} PDFs per package")
        print()
    
    def check_cross_data_consistency(self):
        """Check consistency across different data types"""
        print("🔗 Checking cross-data consistency...")
        
        # Get drug names from labels
        labels_drugs = set()
        labels_dir = self.data_dir / 'labels'
        if labels_dir.exists():
            for label_file in labels_dir.glob('*_labels_*.json'):
                try:
                    with open(label_file, 'r') as f:
                        labels = json.load(f)
                    for label in labels:
                        openfda = label.get('openfda', {})
                        if 'brand_name' in openfda:
                            labels_drugs.update(openfda['brand_name'])
                        if 'generic_name' in openfda:
                            labels_drugs.update(openfda['generic_name'])
                except:
                    pass
        
        # Get drug names from adverse events
        ae_drugs = set()
        ae_dir = self.data_dir / 'adverse_events'
        if ae_dir.exists():
            for ae_file in [f for f in ae_dir.glob('*_adverse_events_*.json') 
                           if not f.parent.name.startswith('.')]:
                try:
                    with open(ae_file, 'r') as f:
                        events = json.load(f)
                    # Sample first 1000
                    for event in events[:1000]:
                        if 'query_drug' in event:
                            ae_drugs.add(event['query_drug'])
                except:
                    pass
        
        # Get package names
        package_drugs = set()
        packages_dir = self.data_dir / 'approval_packages'
        if packages_dir.exists():
            for area in packages_dir.iterdir():
                if area.is_dir():
                    for package in area.iterdir():
                        if package.is_dir():
                            package_drugs.add(package.name)
        
        # Compare
        self.stats['consistency'] = {
            'drugs_with_labels': len(labels_drugs),
            'drugs_with_adverse_events': len(ae_drugs),
            'drugs_with_packages': len(package_drugs),
            'drugs_in_all_three': len(labels_drugs & ae_drugs & package_drugs)
        }
        
        print(f"  💊 Drugs with labels: {len(labels_drugs):,}")
        print(f"  🚨 Drugs with adverse events: {len(ae_drugs):,}")
        print(f"  📦 Drugs with approval packages: {len(package_drugs):,}")
        print(f"  ✅ Drugs in all three: {len(labels_drugs & ae_drugs & package_drugs):,}")
        print()
    
    def _get_dir_size(self, path):
        """Get total size of directory"""
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    
    def generate_report(self):
        """Generate comprehensive quality report"""
        print("\n" + "="*80)
        print("QUALITY REPORT SUMMARY")
        print("="*80)
        
        # Overall health score
        total_issues = sum(len(v) for v in self.issues.values())
        total_duplicates = sum(len(v) for v in self.duplicates.values())
        
        print(f"\n📊 Overall Health:")
        print(f"  Issues Found: {total_issues}")
        print(f"  Duplicates Found: {total_duplicates}")
        
        # Data completeness
        print(f"\n📈 Data Completeness:")
        for data_type, stats in self.stats.items():
            if data_type == 'directories':
                continue
            print(f"  {data_type.replace('_', ' ').title()}:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    if value:  # Only show if not empty
                        print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value:,}" if isinstance(value, int) else f"    {key}: {value}")
        
        # Issues breakdown
        if total_issues > 0:
            print(f"\n⚠️  Issues by Category:")
            for category, issue_list in self.issues.items():
                if issue_list:
                    print(f"  {category}: {len(issue_list)} issues")
                    for issue in issue_list[:3]:  # Show first 3
                        print(f"    - {issue}")
                    if len(issue_list) > 3:
                        print(f"    ... and {len(issue_list) - 3} more")
        
        # Duplicates
        if total_duplicates > 0:
            print(f"\n🔄 Duplicates Found:")
            for category, dup_list in self.duplicates.items():
                if dup_list:
                    print(f"  {category}: {len(dup_list)} duplicates")
                    print(f"    Examples: {', '.join(str(d) for d in dup_list[:3])}")
        
        # Health assessment
        print(f"\n🏥 Health Assessment:")
        if total_issues == 0 and total_duplicates == 0:
            print("  ✅ EXCELLENT - No issues found!")
        elif total_issues < 10 and total_duplicates < 100:
            print("  ✅ GOOD - Minor issues only")
        elif total_issues < 50 and total_duplicates < 500:
            print("  ⚠️  FAIR - Some issues need attention")
        else:
            print("  ❌ POOR - Significant issues require immediate attention")
        
        print("\n" + "="*80)
    
    def save_report(self):
        """Save detailed report to JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'statistics': self.stats,
            'issues': dict(self.issues),
            'duplicates': dict(self.duplicates)
        }
        
        output_file = self.data_dir.parent / 'FDA_DATA_QUALITY_REPORT.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {output_file}")
        
        # Also save a text summary
        summary_file = self.data_dir.parent / 'FDA_DATA_QUALITY_SUMMARY.txt'
        with open(summary_file, 'w') as f:
            f.write("FDA DATA QUALITY CHECK SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.data_dir}\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-"*80 + "\n")
            for category, stats in self.stats.items():
                f.write(f"\n{category.upper()}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
            
            total_issues = sum(len(v) for v in self.issues.values())
            f.write(f"\n\nISSUES FOUND: {total_issues}\n")
            f.write("-"*80 + "\n")
            for category, issue_list in self.issues.items():
                if issue_list:
                    f.write(f"\n{category.upper()} ({len(issue_list)}):\n")
                    for issue in issue_list[:10]:
                        f.write(f"  - {issue}\n")
            
            total_duplicates = sum(len(v) for v in self.duplicates.values())
            f.write(f"\n\nDUPLICATES FOUND: {total_duplicates}\n")
            f.write("-"*80 + "\n")
            for category, dup_list in self.duplicates.items():
                if dup_list:
                    f.write(f"\n{category.upper()}: {len(dup_list)}\n")
        
        print(f"📄 Summary saved: {summary_file}")


def main():
    """Main execution"""
    checker = FDADataQualityChecker("FDA_DATA")
    checker.run_all_checks()
    
    print("\n" + "="*80)
    print("Quality check complete!")
    print("="*80)


if __name__ == "__main__":
    main()