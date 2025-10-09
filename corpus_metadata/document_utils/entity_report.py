#!/usr/bin/env python3
"""
Entity Extraction Report Generator - Enhanced with Summary Tables & Prefix Manager
===================================================================================
Location: corpus_metadata/document_utils/entity_report.py

Generates comprehensive TXT reports showing all database information
for a single processed document. Includes formatted summary tables.

Version: 1.3.0 - Added prefix manager support and improved file handling
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import sqlite3

# Import centralized logging
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('entity_report')

class EntityExtractionReport:
    """Generate comprehensive text report for a single processed document"""
    
    def __init__(self, db_path: Optional[str] = None, prefix_manager=None):
        """Initialize report generator
        
        Args:
            db_path: Path to database file. If None, uses default location.
            prefix_manager: DocumentPrefixManager instance for handling file prefixes
        """
        if db_path is None:
            db_path = '/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_db/extraction_results.db'
        
        self.db_path = Path(db_path)
        self.prefix_manager = prefix_manager
        logger.info(f"Report generator initialized with database: {self.db_path}")
        if prefix_manager:
            logger.info("Prefix manager enabled for file naming")
    
    def generate_document_report(self, run_id: int, output_folder: Path) -> str:
        """Generate comprehensive report for a single document extraction
        
        Args:
            run_id: Database run ID for the current extraction
            output_folder: Folder to save the report
            
        Returns:
            Path to the generated report file
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get document info to create filename
            cursor.execute("""
                SELECT d.filename, d.document_id
                FROM documents d
                JOIN extraction_runs er ON d.document_id = er.document_id
                WHERE er.run_id = ?
            """, (run_id,))
            
            doc_result = cursor.fetchone()
            if not doc_result:
                raise ValueError(f"No document found for run_id: {run_id}")
            
            filename = doc_result['filename']
            doc_id = doc_result['document_id']
            
            # Create report filename with proper prefix handling
            report_filename = self._generate_report_filename(filename)
            report_path = output_folder / report_filename
            
            # Build the complete report
            report_lines = []
            
            # Add header
            report_lines.extend(self._generate_header())
            
            # Add document information
            doc_info = self._get_document_info(cursor, run_id)
            if doc_info:
                report_lines.extend(self._format_document_section(doc_info))
            
            # Add extraction run information
            run_info = self._get_run_info(cursor, run_id)
            if run_info:
                report_lines.extend(self._format_run_section(run_info))
            
            # Add extraction summary
            summary = self._get_extraction_summary(cursor, run_id)
            report_lines.extend(self._format_summary_section(summary))
            
            # Get all entities for summary tables
            abbreviations = self._get_abbreviations(cursor, run_id)
            drugs = self._get_drugs(cursor, run_id)
            diseases = self._get_diseases(cursor, run_id)
            
            # ADD SECTION: Summary Tables
            report_lines.extend(self._format_summary_tables(abbreviations, drugs, diseases))
            
            # Add detailed sections
            report_lines.extend(self._format_abbreviations_section(abbreviations))
            report_lines.extend(self._format_drugs_section(drugs))
            report_lines.extend(self._format_diseases_section(diseases))
            
            # Add linkage analysis
            linkages = self._analyze_abbreviation_linkages(cursor, run_id)
            report_lines.extend(self._format_linkage_section(linkages))
            
            # Add footer
            report_lines.extend(self._generate_footer())
            
            # Join all lines into final report
            report = '\n'.join(report_lines)
            
            # Save to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Report saved to: {report_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate document report: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _generate_report_filename(self, original_filename: str) -> str:
        """Generate report filename with prefix if manager available
        
        Args:
            original_filename: Original document filename
            
        Returns:
            Report filename (with prefix if applicable)
        """
        # Strip existing prefix if present
        base_name = Path(original_filename).stem
        if self.prefix_manager:
            base_name = self.prefix_manager.strip_prefix(base_name)
        
        # Create report name
        report_filename = f"{base_name}_extraction_report.txt"
        
        # Apply prefix if manager available and file doesn't have one
        if self.prefix_manager and not self.prefix_manager.has_prefix(report_filename):
            report_filename = self.prefix_manager.apply_prefix(report_filename)
        
        return report_filename
    
    def _format_summary_tables(self, abbreviations: List[Dict], drugs: List[Dict], diseases: List[Dict]) -> List[str]:
        """Format summary tables for all entities with key attributes"""
        lines = [
            "ENTITY EXTRACTION SUMMARY TABLES",
            "=" * 100,
            "",
            "Complete list of all extracted entities with their key attributes.",
            "For detailed information, see the full sections below.",
            "",
        ]
        
        # Filter out None entries from lists
        abbreviations = [a for a in abbreviations if a is not None]
        drugs = [d for d in drugs if d is not None]
        diseases = [d for d in diseases if d is not None]
        
        # ABBREVIATIONS TABLE
        lines.extend([
            "ABBREVIATIONS SUMMARY TABLE",
            "-" * 100,
        ])
        
        if abbreviations:
            # Table header
            lines.append(
                f"{'#':<4} {'Abbreviation':<15} {'Expansion':<40} {'Type':<12} {'Source':<12} {'Dictionary':<20} {'Conf':<6} {'Count':<6}"
            )
            lines.append("-" * 135)
            
            # Sort and show ALL abbreviations - handle None values in sorting
            sorted_abbrevs = sorted(
                abbreviations, 
                key=lambda x: (x.get('occurrences') if x.get('occurrences') is not None else 0, 
                            x.get('confidence') if x.get('confidence') is not None else 0), 
                reverse=True
            )
            
            for i, abbrev in enumerate(sorted_abbrevs, 1):
                if not abbrev:
                    continue
                abbr = str(abbrev.get('abbreviation', 'N/A'))[:14]
                exp = str(abbrev.get('expansion', 'N/A'))[:39]
                ctx_type = str(abbrev.get('context_type', 'N/A'))[:11]
                source = str(abbrev.get('source_method', 'N/A'))[:11]
                dict_sources = self._format_json_field(abbrev.get('dictionary_sources'))[:19]
                
                # Handle None values for numeric fields
                conf_value = abbrev.get('confidence')
                conf = float(conf_value) if conf_value is not None else 0.0
                
                count_value = abbrev.get('occurrences')
                count = int(count_value) if count_value is not None else 0
                
                # Add the actual data row
                lines.append(
                    f"{i:<4} {abbr:<15} {exp:<40} {ctx_type:<12} {source:<12} {dict_sources:<20} {conf:<6.2f} {count:<6}"
                )
            
            # Add summary message
            lines.append("")
            lines.append(f"Total: {len(abbreviations)} abbreviations (sorted by occurrence)")
        else:
            lines.append("No abbreviations found.")
        
        lines.append("")
        
        # DRUGS TABLE
        lines.extend([
            "DRUGS SUMMARY TABLE",
            "-" * 100,
        ])
        
        if drugs:
            # Table header
            lines.append(
                f"{'#':<4} {'Drug Name':<30} {'Type':<15} {'Source':<12} {'Dictionary':<15} {'Conf':<6} {'Method':<12} {'IDs':<20}"
            )
            lines.append("-" * 134)
            
            # Show ALL drugs
            for i, drug in enumerate(drugs, 1):
                if not drug:
                    continue
                name = str(drug.get('drug_name', 'N/A'))[:29]
                dtype = str(drug.get('type', 'N/A'))[:14]
                source = str(drug.get('source', 'N/A'))[:11]
                dict_src = str(drug.get('dictionary', 'N/A'))[:14]
                
                # Handle None for confidence
                conf_value = drug.get('confidence')
                conf = float(conf_value) if conf_value is not None else 0.0
                
                method = str(drug.get('extraction_method', 'N/A'))[:11]
                ids = self._format_identifiers(drug)[:19]
                
                lines.append(
                    f"{i:<4} {name:<30} {dtype:<15} {source:<12} {dict_src:<15} {conf:<6.2f} {method:<12} {ids:<20}"
                )
            
            lines.append("")
            lines.append(f"Total: {len(drugs)} drugs")
        else:
            lines.append("No drugs found.")
        
        lines.append("")
        
        # DISEASES TABLE
        lines.extend([
            "DISEASES/BIOMARKERS SUMMARY TABLE",
            "-" * 100,
        ])
        
        if diseases:
            # Table header
            lines.append(
                f"{'#':<4} {'Name':<35} {'Type':<15} {'Source':<12} {'Dictionary':<15} {'Conf':<6} {'From Abbr':<10}"
            )
            lines.append("-" * 117)
            
            # Show ALL diseases
            for i, disease in enumerate(diseases, 1):
                if not disease:
                    continue
                name = str(disease.get('disease_name', 'N/A'))[:34]
                dtype = str(disease.get('type', 'N/A'))[:14]
                source = str(disease.get('source', 'N/A'))[:11]
                dict_src = str(disease.get('dictionary', 'N/A'))[:14]
                
                # Handle None for confidence
                conf_value = disease.get('confidence')
                conf = float(conf_value) if conf_value is not None else 0.0
                
                from_abbr = 'Yes' if disease.get('from_abbreviation_id') else '-'
                
                lines.append(
                    f"{i:<4} {name:<35} {dtype:<15} {source:<12} {dict_src:<15} {conf:<6.2f} {from_abbr:<10}"
                )
            
            lines.append("")
            lines.append(f"Total: {len(diseases)} diseases/biomarkers")
        else:
            lines.append("No diseases found.")
        
        lines.append("")
        
        # EXTRACTION STATISTICS
        lines.extend([
            "EXTRACTION STATISTICS",
            "-" * 100,
        ])
        
        # Calculate statistics with proper None handling
        high_conf_abbrevs = sum(1 for a in abbreviations 
                            if a and a.get('confidence') is not None 
                            and float(a.get('confidence')) >= 0.90)
        high_conf_drugs = sum(1 for d in drugs 
                            if d and d.get('confidence') is not None 
                            and float(d.get('confidence')) >= 0.90)
        high_conf_diseases = sum(1 for d in diseases 
                            if d and d.get('confidence') is not None 
                            and float(d.get('confidence')) >= 0.90)
        
        drugs_from_abbrev = sum(1 for d in drugs if d and d.get('from_abbreviation_id'))
        diseases_from_abbrev = sum(1 for d in diseases if d and d.get('from_abbreviation_id'))
        
        drugs_with_ids = sum(1 for d in drugs if d and self._format_identifiers(d) != 'None')
        diseases_with_ids = sum(1 for d in diseases if d and self._format_identifiers(d) != 'None')
        
        lines.extend([
            "High Confidence (≥0.90):",
            f"  • Abbreviations: {high_conf_abbrevs}/{len(abbreviations)} ({high_conf_abbrevs*100/len(abbreviations):.1f}%)" if abbreviations else "  • Abbreviations: 0/0",
            f"  • Drugs: {high_conf_drugs}/{len(drugs)} ({high_conf_drugs*100/len(drugs):.1f}%)" if drugs else "  • Drugs: 0/0",
            f"  • Diseases: {high_conf_diseases}/{len(diseases)} ({high_conf_diseases*100/len(diseases):.1f}%)" if diseases else "  • Diseases: 0/0",
            "",
            "Entity Sources:",
            f"  • Drugs from abbreviations: {drugs_from_abbrev}/{len(drugs)}" if drugs else "  • Drugs from abbreviations: 0/0",
            f"  • Diseases from abbreviations: {diseases_from_abbrev}/{len(diseases)}" if diseases else "  • Diseases from abbreviations: 0/0",
            "",
            "Entity Identifiers:",
            f"  • Drugs with identifiers (RxCUI/MeSH): {drugs_with_ids}/{len(drugs)}" if drugs else "  • Drugs with identifiers: 0/0",
            f"  • Diseases with identifiers (ORPHA/MeSH/DOID): {diseases_with_ids}/{len(diseases)}" if diseases else "  • Diseases with identifiers: 0/0",
        ])
        
        lines.extend(["", "=" * 100, ""])
        
        return lines
    
    def _generate_header(self) -> List[str]:
        """Generate report header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [
            "=" * 100,
            "ENTITY EXTRACTION REPORT - SINGLE DOCUMENT",
            "=" * 100,
            f"Report Generated: {timestamp}",
            "=" * 100,
            ""
        ]
    
    def _get_document_info(self, cursor, run_id) -> Dict:
        """Get document information"""
        cursor.execute("""
            SELECT d.* 
            FROM documents d
            JOIN extraction_runs er ON d.document_id = er.document_id
            WHERE er.run_id = ?
        """, (run_id,))
        
        result = cursor.fetchone()
        return dict(result) if result else {}
    
    def _get_run_info(self, cursor, run_id) -> Dict:
        """Get extraction run information"""
        cursor.execute("""
            SELECT * FROM extraction_runs WHERE run_id = ?
        """, (run_id,))
        
        result = cursor.fetchone()
        return dict(result) if result else {}
    
    def _get_extraction_summary(self, cursor, run_id) -> Dict:
        """Get extraction summary statistics"""
        cursor.execute("""
            SELECT 
                total_abbreviations,
                total_drugs,
                total_diseases,
                drugs_from_abbreviations,
                diseases_from_abbreviations,
                processing_time_seconds,
                text_length,
                extraction_mode,
                validation_method
            FROM extraction_runs 
            WHERE run_id = ?
        """, (run_id,))
        
        result = cursor.fetchone()
        return dict(result) if result else {}
    
    def _get_abbreviations(self, cursor, run_id) -> List[Dict]:
        """Get all abbreviations with full details"""
        cursor.execute("""
            SELECT * FROM abbreviations 
            WHERE run_id = ? 
            ORDER BY confidence DESC, occurrences DESC
        """, (run_id,))
        
        return [dict(row) if row else None for row in cursor.fetchall()]
    
    def _get_drugs(self, cursor, run_id) -> List[Dict]:
        """Get all drugs with full details"""
        cursor.execute("""
            SELECT d.*, a.abbreviation as source_abbreviation
            FROM drugs d
            LEFT JOIN abbreviations a ON d.from_abbreviation_id = a.abbrev_id
            WHERE d.run_id = ?
            ORDER BY d.confidence DESC, d.occurrences DESC
        """, (run_id,))
        
        return [dict(row) if row else None for row in cursor.fetchall()]
    
    def _get_diseases(self, cursor, run_id) -> List[Dict]:
        """Get all diseases with full details"""
        cursor.execute("""
            SELECT d.*, a.abbreviation as source_abbreviation
            FROM diseases d
            LEFT JOIN abbreviations a ON d.from_abbreviation_id = a.abbrev_id
            WHERE d.run_id = ?
            ORDER BY d.confidence DESC, d.occurrences DESC
        """, (run_id,))
        
        return [dict(row) if row else None for row in cursor.fetchall()]
    
    def _analyze_abbreviation_linkages(self, cursor, run_id) -> Dict:
        """Analyze which entities were derived from abbreviations"""
        # Drugs from abbreviations
        cursor.execute("""
            SELECT 
                a.abbreviation,
                a.expansion,
                COUNT(d.drug_id) as drugs_generated,
                GROUP_CONCAT(d.drug_name, ', ') as drug_names
            FROM abbreviations a
            JOIN drugs d ON d.from_abbreviation_id = a.abbrev_id
            WHERE a.run_id = ?
            GROUP BY a.abbrev_id
        """, (run_id,))
        drugs_from_abbrev = [dict(row) if row else None for row in cursor.fetchall()]
        
        # Diseases from abbreviations
        cursor.execute("""
            SELECT 
                a.abbreviation,
                a.expansion,
                COUNT(d.disease_id) as diseases_generated,
                GROUP_CONCAT(d.disease_name, ', ') as disease_names
            FROM abbreviations a
            JOIN diseases d ON d.from_abbreviation_id = a.abbrev_id
            WHERE a.run_id = ?
            GROUP BY a.abbrev_id
        """, (run_id,))
        diseases_from_abbrev = [dict(row) if row else None for row in cursor.fetchall()]
        
        return {
            'drugs_from_abbreviations': [d for d in drugs_from_abbrev if d],
            'diseases_from_abbreviations': [d for d in diseases_from_abbrev if d]
        }
    
    def _format_document_section(self, doc_info: Dict) -> List[str]:
        """Format document information section"""
        lines = [
            "DOCUMENT INFORMATION",
            "-" * 100,
            f"Filename:                {doc_info.get('filename', 'N/A')}",
            f"File Path:               {doc_info.get('file_path', 'N/A')}",
            f"File Hash:               {doc_info.get('file_hash', 'N/A')}",
            f"File Size:               {self._format_file_size(doc_info.get('file_size', 0))}",
            f"Page Count:              {doc_info.get('page_count', 'N/A')}",
            "",
            "Document Classification:",
            f"  Type:                  {doc_info.get('document_type', 'N/A')}",
            f"  Subtype:               {doc_info.get('document_subtype', 'N/A')}",
            f"  Disease Classification: {doc_info.get('disease_classification', 'N/A')}",
            "",
            "Document Metadata:",
            f"  Title:                 {doc_info.get('title', 'N/A')}",
            f"  Description:           {doc_info.get('short_description', 'N/A')}",
            f"  Language:              {doc_info.get('language', 'N/A')}",
            f"  Document Date:         {doc_info.get('document_date', 'N/A')}",
            f"  Extraction Date:       {doc_info.get('extraction_date', 'N/A')}",
            f"  Pipeline Version:      {doc_info.get('pipeline_version', 'N/A')}",
            "",
            "=" * 100,
            ""
        ]
        return lines
    
    def _format_run_section(self, run_info: Dict) -> List[str]:
        """Format extraction run information section"""
        lines = [
            "EXTRACTION RUN DETAILS",
            "-" * 100,
            f"Run ID:                  {run_info.get('run_id', 'N/A')}",
            f"Run Date:                {run_info.get('run_date', 'N/A')}",
            f"Extraction Mode:         {run_info.get('extraction_mode', 'N/A')}",
            f"Validation Method:       {run_info.get('validation_method', 'N/A')}",
            f"Processing Time:         {self._format_time(run_info.get('processing_time_seconds', 0))}",
            f"Text Length:             {run_info.get('text_length', 0):,} characters",
            f"Status:                  {run_info.get('status', 'N/A')}",
            f"Error Message:           {run_info.get('error_message', 'None')}",
            "",
            "=" * 100,
            ""
        ]
        return lines
    
    def _format_summary_section(self, summary: Dict) -> List[str]:
        """Format extraction summary section"""
        total_drugs = summary.get('total_drugs', 0)
        drugs_from_abbrev = summary.get('drugs_from_abbreviations', 0)
        total_diseases = summary.get('total_diseases', 0)
        diseases_from_abbrev = summary.get('diseases_from_abbreviations', 0)
        
        lines = [
            "EXTRACTION SUMMARY",
            "-" * 100,
            f"Total Abbreviations:     {summary.get('total_abbreviations', 0)}",
            f"Total Drugs:             {total_drugs}",
            f"  - From Abbreviations:  {drugs_from_abbrev}",
            f"  - Direct Detection:    {total_drugs - drugs_from_abbrev}",
            f"Total Diseases:          {total_diseases}",
            f"  - From Abbreviations:  {diseases_from_abbrev}",
            f"  - Direct Detection:    {total_diseases - diseases_from_abbrev}",
            "",
            "=" * 100,
            ""
        ]
        return lines
    
    def _format_abbreviations_section(self, abbreviations: List[Dict]) -> List[str]:
        """Format abbreviations section with full details"""
        # Filter out None entries
        abbreviations = [a for a in abbreviations if a is not None]
        
        lines = [
            "ABBREVIATIONS EXTRACTED",
            "-" * 100,
            f"Total Count: {len(abbreviations)}",
            ""
        ]
        
        if not abbreviations:
            lines.append("No abbreviations found.")
        else:
            for i, abbrev in enumerate(abbreviations, 1):
                if not abbrev:
                    continue
                
                conf_value = abbrev.get('confidence')
                conf = float(conf_value) if conf_value is not None else 0.0
                
                occ_value = abbrev.get('occurrences')
                occurrences = int(occ_value) if occ_value is not None else 0
                
                lines.extend([
                    f"[{i}] {abbrev.get('abbreviation', 'N/A')}",
                    f"    Expansion:           {abbrev.get('expansion', 'N/A')}",
                    f"    Context Type:        {abbrev.get('context_type', 'N/A')}",
                    f"    Semantic Type:       {abbrev.get('semantic_type', 'N/A')}",
                    f"    Confidence:          {conf:.2f}",
                    f"    Occurrences:         {occurrences}",
                    f"    Source Method:       {abbrev.get('source_method', 'N/A')}",
                    f"    Dictionary Sources:  {self._format_json_field(abbrev.get('dictionary_sources'))}",
                    f"    Conflict Resolved:   {abbrev.get('conflict_resolved', False)}",
                ])
                
                if abbrev.get('alternative_expansions'):
                    alts = self._parse_json_field(abbrev.get('alternative_expansions'))
                    if alts:
                        lines.append(f"    Alternatives:        {', '.join(alts)}")
                
                lines.append("")
        
        lines.extend(["", "=" * 100, ""])
        return lines
    
    def _format_drugs_section(self, drugs: List[Dict]) -> List[str]:
        """Format drugs section with full details"""
        drugs = [d for d in drugs if d is not None]
        
        lines = [
            "DRUGS EXTRACTED",
            "-" * 100,
            f"Total Count: {len(drugs)}",
            ""
        ]
        
        if not drugs:
            lines.append("No drugs found.")
        else:
            for i, drug in enumerate(drugs, 1):
                if not drug:
                    continue
                lines.extend([
                    f"[{i}] {drug.get('drug_name', 'N/A')}",
                    f"    Normalized Name:     {drug.get('normalized_name', 'N/A')}",
                    f"    Canonical Name:      {drug.get('canonical_name', 'N/A')}",
                    f"    Drug Type:           {drug.get('drug_type', 'N/A')}",
                    f"    Drug Class:          {drug.get('drug_class', 'N/A')}",
                    f"    Mechanism:           {drug.get('mechanism', 'N/A')}",
                    f"    Source:              {drug.get('source', 'N/A')}",
                    f"    Approval Status:     {drug.get('approval_status', 'N/A')}",
                    f"    FDA Approval Date:   {drug.get('fda_approval_date', 'N/A')}",
                ])
                
                identifiers = []
                if drug.get('rxcui'): identifiers.append(f"RxCUI: {drug['rxcui']}")
                if drug.get('mesh_id'): identifiers.append(f"MeSH: {drug['mesh_id']}")
                if drug.get('chebi_id'): identifiers.append(f"ChEBI: {drug['chebi_id']}")
                if drug.get('unii'): identifiers.append(f"UNII: {drug['unii']}")
                if drug.get('drugbank_id'): identifiers.append(f"DrugBank: {drug['drugbank_id']}")
                if drug.get('atc_code'): identifiers.append(f"ATC: {drug['atc_code']}")
                
                if identifiers:
                    lines.append(f"    Identifiers:         {', '.join(identifiers)}")
                
                lines.extend([
                    f"    Confidence:          {float(drug.get('confidence', 0)):.2f}",
                    f"    Occurrences:         {drug.get('occurrences', 0)}",
                    f"    Detection Method:    {drug.get('detection_method', 'N/A')}",
                ])
                
                if drug.get('source_abbreviation'):
                    lines.append(f"    From Abbreviation:   {drug['source_abbreviation']}")
                
                if drug.get('validation_status'):
                    lines.append(f"    Validation Status:   {drug['validation_status']}")
                if drug.get('claude_approved') is not None:
                    lines.append(f"    Claude Approved:     {drug['claude_approved']}")
                    if drug.get('claude_reason'):
                        lines.append(f"    Claude Reason:       {drug['claude_reason']}")
                
                if drug.get('brand_names'):
                    brands = self._parse_json_field(drug.get('brand_names'))
                    if brands:
                        lines.append(f"    Brand Names:         {', '.join(brands)}")
                
                if drug.get('synonyms'):
                    syns = self._parse_json_field(drug.get('synonyms'))
                    if syns and len(syns) <= 5:
                        lines.append(f"    Synonyms:            {', '.join(syns)}")
                    elif syns:
                        lines.append(f"    Synonyms:            {', '.join(syns[:5])} ... ({len(syns)} total)")
                
                lines.append("")
        
        lines.extend(["", "=" * 100, ""])
        return lines
    
    def _format_diseases_section(self, diseases: List[Dict]) -> List[str]:
        """Format diseases section with full details"""
        diseases = [d for d in diseases if d is not None]
        
        lines = [
            "DISEASES EXTRACTED",
            "-" * 100,
            f"Total Count: {len(diseases)}",
            ""
        ]
        
        if not diseases:
            lines.append("No diseases found.")
        else:
            for i, disease in enumerate(diseases, 1):
                if not disease:
                    continue
                lines.extend([
                    f"[{i}] {disease.get('disease_name', 'N/A')}",
                    f"    Normalized Name:     {disease.get('normalized_name', 'N/A')}",
                    f"    Canonical Name:      {disease.get('canonical_name', 'N/A')}",
                    f"    Disease Category:    {disease.get('disease_category', 'N/A')}",
                    f"    Disease Group:       {disease.get('disease_group', 'N/A')}",
                    f"    Inheritance Pattern: {disease.get('inheritance_pattern', 'N/A')}",
                    f"    Is Rare:             {disease.get('is_rare', False)}",
                    f"    Prevalence:          {disease.get('prevalence', 'N/A')}",
                    f"    Onset Age:           {disease.get('onset_age', 'N/A')}",
                ])
                
                identifiers = []
                if disease.get('orpha_code'): identifiers.append(f"ORPHA: {disease['orpha_code']}")
                if disease.get('doid'): identifiers.append(f"DOID: {disease['doid']}")
                if disease.get('mesh_id'): identifiers.append(f"MeSH: {disease['mesh_id']}")
                if disease.get('omim_id'): identifiers.append(f"OMIM: {disease['omim_id']}")
                if disease.get('mondo_id'): identifiers.append(f"MONDO: {disease['mondo_id']}")
                
                if identifiers:
                    lines.append(f"    Identifiers:         {', '.join(identifiers)}")
                
                if disease.get('icd10_codes'):
                    icd10 = self._parse_json_field(disease.get('icd10_codes'))
                    if icd10:
                        lines.append(f"    ICD-10 Codes:        {', '.join(icd10)}")
                
                lines.extend([
                    f"    Detection Method:    {disease.get('detection_method', 'N/A')}",
                    f"    Source:              {disease.get('source', 'N/A')}",
                    f"    Lexicon Source:      {disease.get('lexicon_source', 'N/A')}",
                    f"    Confidence:          {float(disease.get('confidence', 0)):.2f}",
                    f"    Occurrences:         {disease.get('occurrences', 0)}",
                    f"    Is Primary:          {disease.get('is_primary', False)}",
                ])
                
                if disease.get('source_abbreviation'):
                    lines.append(f"    From Abbreviation:   {disease['source_abbreviation']}")
                
                if disease.get('validation_status'):
                    lines.append(f"    Validation Status:   {disease['validation_status']}")
                if disease.get('claude_approved') is not None:
                    lines.append(f"    Claude Approved:     {disease['claude_approved']}")
                
                if disease.get('synonyms'):
                    syns = self._parse_json_field(disease.get('synonyms'))
                    if syns:
                        lines.append(f"    Synonyms:            {', '.join(syns)}")
                
                lines.append("")
        
        lines.extend(["", "=" * 100, ""])
        return lines
    
    def _format_linkage_section(self, linkages: Dict) -> List[str]:
        """Format abbreviation linkage analysis section"""
        lines = [
            "ABBREVIATION-TO-ENTITY LINKAGE ANALYSIS",
            "-" * 100,
            ""
        ]
        
        drugs_from = linkages.get('drugs_from_abbreviations', [])
        if drugs_from:
            lines.append(f"Drugs Generated from Abbreviations ({len(drugs_from)} abbreviations):")
            lines.append("")
            for item in drugs_from:
                if item:
                    lines.extend([
                        f"  • {item.get('abbreviation', 'N/A')} → {item.get('expansion', 'N/A')}",
                        f"    Generated {item.get('drugs_generated', 0)} drug(s): {item.get('drug_names', 'N/A')}",
                        ""
                    ])
        else:
            lines.append("No drugs were generated from abbreviations.")
        
        lines.append("")
        
        diseases_from = linkages.get('diseases_from_abbreviations', [])
        if diseases_from:
            lines.append(f"Diseases Generated from Abbreviations ({len(diseases_from)} abbreviations):")
            lines.append("")
            for item in diseases_from:
                if item:
                    lines.extend([
                        f"  • {item.get('abbreviation', 'N/A')} → {item.get('expansion', 'N/A')}",
                        f"    Generated {item.get('diseases_generated', 0)} disease(s): {item.get('disease_names', 'N/A')}",
                        ""
                    ])
        else:
            lines.append("No diseases were generated from abbreviations.")
        
        lines.extend(["", "=" * 100, ""])
        return lines
    
    def _generate_footer(self) -> List[str]:
        """Generate report footer"""
        return [
            "",
            "END OF REPORT",
            "=" * 100
        ]
    
    # Helper methods
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if not size_bytes:
            return "N/A"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration"""
        if not seconds:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    def _format_json_field(self, json_str: str) -> str:
        """Format JSON field for display"""
        if not json_str:
            return "None"
        
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if isinstance(data, list):
                return ', '.join(str(item) for item in data) if data else "None"
            return str(data)
        except:
            return str(json_str) if json_str else "None"
    
    def _format_identifiers(self, entity: Dict) -> str:
        """Format entity identifiers for display"""
        identifiers = []
        
        # Drug identifiers
        if entity.get('rxcui'):
            identifiers.append(f"RxCUI:{entity['rxcui']}")
        if entity.get('mesh_id'):
            identifiers.append(f"MeSH:{entity['mesh_id']}")
        if entity.get('atc_code'):
            identifiers.append(f"ATC:{entity['atc_code']}")
        
        # Disease identifiers  
        if entity.get('orphanet_id'):
            identifiers.append(f"ORPHA:{entity['orphanet_id']}")
        if entity.get('omim_id'):
            identifiers.append(f"OMIM:{entity['omim_id']}")
        if entity.get('doid'):
            identifiers.append(f"DOID:{entity['doid']}")
        if entity.get('icd10'):
            identifiers.append(f"ICD10:{entity['icd10']}")
        
        # Generic identifiers field
        if entity.get('identifiers'):
            if isinstance(entity['identifiers'], str):
                identifiers.append(entity['identifiers'])
            elif isinstance(entity['identifiers'], dict):
                for key, value in entity['identifiers'].items():
                    if value:
                        identifiers.append(f"{key}:{value}")
        
        return ', '.join(identifiers[:2]) if identifiers else 'None'
    
    def _parse_json_field(self, json_str: str) -> List:
        """Parse JSON field to list"""
        if not json_str:
            return []
        
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            return data if isinstance(data, list) else []
        except:
            return []


# Function to be called from entity_extraction.py
def generate_extraction_report(run_id: int, output_folder: Path, prefix_manager=None) -> Optional[str]:
    """Generate extraction report for the current document
    
    This function is called at the end of process_document_two_stage()
    
    Args:
        run_id: Database run ID for the current extraction
        output_folder: Folder where the report should be saved
        prefix_manager: Optional DocumentPrefixManager for file naming
        
    Returns:
        Path to the generated report or None if failed
    """
    try:
        reporter = EntityExtractionReport(prefix_manager=prefix_manager)
        report_path = reporter.generate_document_report(run_id, output_folder)
        logger.info(f"Generated extraction report: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Failed to generate extraction report: {e}")
        return None