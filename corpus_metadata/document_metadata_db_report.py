#!/usr/bin/env python3
"""
Database Integrity Test Script for Biomedical Entity Extraction
Tests the extraction_results.db database and displays entity lists in table format
"""

import sqlite3
from typing import Dict, List, Any
from pathlib import Path

class DatabaseIntegrityTester:
    """Test database integrity and display entity lists"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the tester with database path
        
        Args:
            db_path: Path to the SQLite database
        """
        if db_path is None:
            db_path = '/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_db/extraction_results.db'
        
        self.db_path = Path(db_path)
        self.conn = None
        
    def connect(self):
        """Connect to the database"""
        try:
            if not self.db_path.exists():
                print(f"✗ Database not found at: {self.db_path}")
                return False
                
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            print(f"✓ Connected to database: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to database: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_abbreviations_table(self, run_id: int = None) -> List[Dict]:
        """Get abbreviations in table format"""
        cursor = self.conn.cursor()
        
        if run_id is None:
            cursor.execute("SELECT MAX(run_id) as latest FROM extraction_runs")
            run_id = cursor.fetchone()['latest']
        
        # Count unique abbreviation-expansion pairs
        query = """
        SELECT 
            abbreviation,
            expansion,
            context_type,
            source_method,
            confidence,
            COUNT(*) as occurrences
        FROM abbreviations
        WHERE run_id = ?
        GROUP BY abbreviation, expansion
        ORDER BY COUNT(*) DESC, abbreviation
        """
        
        cursor.execute(query, (run_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_drugs_table(self, run_id: int = None) -> List[Dict]:
        """Get drugs in table format"""
        cursor = self.conn.cursor()
        
        if run_id is None:
            cursor.execute("SELECT MAX(run_id) as latest FROM extraction_runs")
            run_id = cursor.fetchone()['latest']
        
        query = """
        SELECT 
            drug_name,
            normalized_name,
            drug_type,
            source,
            confidence,
            occurrences
        FROM drugs
        WHERE run_id = ?
        ORDER BY occurrences DESC, drug_name
        """
        
        cursor.execute(query, (run_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_diseases_table(self, run_id: int = None) -> List[Dict]:
        """Get diseases in table format"""
        cursor = self.conn.cursor()
        
        if run_id is None:
            cursor.execute("SELECT MAX(run_id) as latest FROM extraction_runs")
            run_id = cursor.fetchone()['latest']
        
        query = """
        SELECT 
            disease_name,
            canonical_name,
            context_type,
            detection_method,
            confidence,
            occurrences
        FROM diseases
        WHERE run_id = ?
        ORDER BY occurrences DESC, disease_name
        """
        
        cursor.execute(query, (run_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def print_entity_tables(self, run_id: int = None):
        """Print all entities in table format"""
        
        # ABBREVIATIONS TABLE
        print(f"\n{'='*120}")
        print("ABBREVIATIONS TABLE")
        print(f"{'='*120}")
        
        abbreviations = self.get_abbreviations_table(run_id)
        
        if abbreviations:
            # Header
            print(f"{'#':<4} {'Abbreviation':<15} {'Expansion':<50} {'Type':<15} {'Source':<12} {'Conf':<6} {'Count':<6}")
            print(f"{'-'*4} {'-'*15} {'-'*50} {'-'*15} {'-'*12} {'-'*6} {'-'*6}")
            
            # Data rows
            for i, abbr in enumerate(abbreviations, 1):
                abbrev = str(abbr['abbreviation'])[:14]
                expansion = str(abbr['expansion'] or 'N/A')[:49]
                context = str(abbr['context_type'] or 'N/A')[:14]
                source = str(abbr['source_method'] or 'N/A')[:11]
                conf = float(abbr['confidence'] or 0)
                count = int(abbr['occurrences'])
                
                print(f"{i:<4} {abbrev:<15} {expansion:<50} {context:<15} {source:<12} {conf:<6.2f} {count:<6}")
            
            print(f"\nTotal: {len(abbreviations)} unique abbreviations (sorted by occurrences)")
        else:
            print("No abbreviations found.")
        
        # DRUGS TABLE
        print(f"\n{'='*120}")
        print("DRUGS TABLE")
        print(f"{'='*120}")
        
        drugs = self.get_drugs_table(run_id)
        
        if drugs:
            # Header
            print(f"{'#':<4} {'Drug Name':<30} {'Normalized Name':<30} {'Type':<20} {'Source':<15} {'Conf':<6} {'Count':<6}")
            print(f"{'-'*4} {'-'*30} {'-'*30} {'-'*20} {'-'*15} {'-'*6} {'-'*6}")
            
            # Data rows
            for i, drug in enumerate(drugs, 1):
                drug_name = str(drug['drug_name'])[:29]
                normalized = str(drug['normalized_name'] or drug_name)[:29]
                drug_type = str(drug['drug_type'] or 'N/A')[:19]
                source = str(drug['source'] or 'N/A')[:14]
                conf = float(drug['confidence'] or 0)
                count = int(drug['occurrences'])
                
                print(f"{i:<4} {drug_name:<30} {normalized:<30} {drug_type:<20} {source:<15} {conf:<6.2f} {count:<6}")
            
            print(f"\nTotal: {len(drugs)} unique drugs (sorted by occurrences)")
        else:
            print("No drugs found.")
        
        # DISEASES TABLE
        print(f"\n{'='*120}")
        print("DISEASES TABLE")
        print(f"{'='*120}")
        
        diseases = self.get_diseases_table(run_id)
        
        if diseases:
            # Header
            print(f"{'#':<4} {'Disease Name':<35} {'Canonical Name':<35} {'Context':<15} {'Method':<12} {'Conf':<6} {'Count':<6}")
            print(f"{'-'*4} {'-'*35} {'-'*35} {'-'*15} {'-'*12} {'-'*6} {'-'*6}")
            
            # Data rows
            for i, disease in enumerate(diseases, 1):
                disease_name = str(disease['disease_name'])[:34]
                canonical = str(disease['canonical_name'] or disease_name)[:34]
                context = str(disease['context_type'] or 'N/A')[:14]
                method = str(disease['detection_method'] or 'N/A')[:11]
                conf = float(disease['confidence'] or 0)
                count = int(disease['occurrences'])
                
                print(f"{i:<4} {disease_name:<35} {canonical:<35} {context:<15} {method:<12} {conf:<6.2f} {count:<6}")
            
            print(f"\nTotal: {len(diseases)} unique diseases (sorted by occurrences)")
        else:
            print("No diseases found.")
    
    def run_integrity_tests(self, run_id: int = None):
        """Run basic integrity tests"""
        cursor = self.conn.cursor()
        
        if run_id is None:
            cursor.execute("SELECT MAX(run_id) as latest FROM extraction_runs")
            run_id = cursor.fetchone()['latest']
        
        print(f"\n{'='*120}")
        print(f"DATABASE INTEGRITY CHECK - RUN ID: {run_id}")
        print(f"{'='*120}")
        
        # Get extraction run stats
        cursor.execute("""
            SELECT 
                total_abbreviations,
                total_drugs,
                total_diseases,
                drugs_from_abbreviations,
                diseases_from_abbreviations
            FROM extraction_runs
            WHERE run_id = ?
        """, (run_id,))
        
        run_stats = cursor.fetchone()
        
        if run_stats:
            print(f"\nExtraction Run Statistics:")
            print(f"  Abbreviations: {run_stats['total_abbreviations']}")
            print(f"  Drugs: {run_stats['total_drugs']}")
            print(f"  Diseases: {run_stats['total_diseases']}")
            print(f"  Drugs from abbreviations: {run_stats['drugs_from_abbreviations']}")
            print(f"  Diseases from abbreviations: {run_stats['diseases_from_abbreviations']}")
        
        # Get actual counts
        cursor.execute("SELECT COUNT(*) as count FROM abbreviations WHERE run_id = ?", (run_id,))
        actual_abbr = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM drugs WHERE run_id = ?", (run_id,))
        actual_drugs = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM diseases WHERE run_id = ?", (run_id,))
        actual_diseases = cursor.fetchone()['count']
        
        print(f"\nActual counts in tables:")
        print(f"  Abbreviations: {actual_abbr}")
        print(f"  Drugs: {actual_drugs}")
        print(f"  Diseases: {actual_diseases}")
        
        # Check for consistency
        if (actual_abbr == run_stats['total_abbreviations'] and 
            actual_drugs == run_stats['total_drugs'] and 
            actual_diseases == run_stats['total_diseases']):
            print("\n✓ Database counts are consistent")
        else:
            print("\n⚠️ Database count mismatch detected")
    
    def find_document_runs(self, filename_pattern: str = '00954'):
        """Find all extraction runs for a document"""
        print(f"\n{'='*120}")
        print(f"DOCUMENT SEARCH: {filename_pattern}")
        print(f"{'='*120}")
        
        cursor = self.conn.cursor()
        
        query = """
        SELECT 
            er.run_id,
            er.run_date,
            er.extraction_mode,
            er.status,
            d.filename,
            d.document_id,
            er.total_abbreviations,
            er.total_drugs,
            er.total_diseases
        FROM extraction_runs er
        JOIN documents d ON er.document_id = d.document_id
        WHERE d.filename LIKE ?
        ORDER BY er.run_date DESC
        """
        
        cursor.execute(query, (f'%{filename_pattern}%',))
        runs = cursor.fetchall()
        
        if runs:
            print(f"\nFound {len(runs)} extraction run(s):")
            for run in runs:
                print(f"\n  Run ID: {run['run_id']}")
                print(f"  Document: {run['filename']}")
                print(f"  Date: {run['run_date']}")
                print(f"  Mode: {run['extraction_mode']}")
                print(f"  Status: {run['status']}")
                print(f"  Entities: {run['total_abbreviations']} abbr, {run['total_drugs']} drugs, {run['total_diseases']} diseases")
            
            return [run['run_id'] for run in runs]
        else:
            print(f"No runs found for document pattern: {filename_pattern}")
            return []


def main():
    """Main execution"""
    print(f"{'='*120}")
    print("BIOMEDICAL ENTITY EXTRACTION - DATABASE REPORT")
    print(f"{'='*120}")
    
    # Create tester with default database path
    tester = DatabaseIntegrityTester()
    
    if not tester.connect():
        return 1
    
    try:
        # Find runs for document '00954'
        run_ids = tester.find_document_runs('00954')
        
        if run_ids:
            # Use the most recent run
            run_id = run_ids[0]
            
            # Run integrity check
            tester.run_integrity_tests(run_id)
            
            # Print entity tables
            tester.print_entity_tables(run_id)
            
            print(f"\n{'='*120}")
            print("REPORT COMPLETE")
            print(f"{'='*120}\n")
        else:
            print("\nNo extraction runs found for document 00954.")
        
    finally:
        tester.close()
    
    return 0


if __name__ == "__main__":
    exit(main())