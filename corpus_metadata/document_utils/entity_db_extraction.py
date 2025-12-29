#!/usr/bin/env python3
"""
Entity Extraction Database Setup - v12.0.0
==========================================
Location: corpus_metadata/document_utils/entity_db_extraction.py
Version: 12.0.0 - REMOVED ABBREVIATION TABLES AND FUNCTIONALITY

Sets up SQLite database for storing extraction pipeline results.
Updated with proper defaults to prevent NULL value errors in report generation.

CHANGES IN v12.0.0:
- REMOVED abbreviations table
- REMOVED from_abbreviation_id foreign keys
- REMOVED abbreviation-related views
- REMOVED save_abbreviations_with_counts method
- SIMPLIFIED extraction_runs table
"""

import sqlite3
import os
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

# Import centralized logging
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('entity_db_extraction')

class ExtractionDatabase:
    """Manages SQLite database for extraction results"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection
        
        Args:
            db_path: Path to database file. If None, uses default location.
        """
        if db_path is None:
            db_path = '/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_db/extraction_results.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing database at: {self.db_path}")
        self.conn = None
        self.setup_database()
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn
    
    def setup_database(self):
        """Create all required tables and indexes"""
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                
                # Create tables
                self._create_documents_table(cursor)
                self._create_extraction_runs_table(cursor)
                self._create_drugs_table(cursor)
                self._create_diseases_table(cursor)
                self._create_identifier_tables(cursor)
                self._create_lexicon_tables(cursor)
                self._create_indexes(cursor)
                self._create_views(cursor)
                
                conn.commit()
                logger.info("Database setup completed successfully")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def _create_documents_table(self, cursor):
        """Create documents table with proper defaults"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT DEFAULT '',
                file_hash TEXT UNIQUE,
                file_size INTEGER DEFAULT 0,
                page_count INTEGER DEFAULT 0,
                
                -- Document classification with defaults
                document_type VARCHAR(50) DEFAULT 'unknown',
                document_subtype VARCHAR(50) DEFAULT '',
                disease_classification VARCHAR(100) DEFAULT '',
                
                -- Metadata with defaults
                title TEXT DEFAULT '',
                short_description TEXT DEFAULT '',
                language VARCHAR(10) DEFAULT 'en',
                
                -- Dates
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                document_date DATE,
                
                -- Pipeline info with defaults
                pipeline_version VARCHAR(20) DEFAULT 'unknown',
                status VARCHAR(20) DEFAULT 'pending',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.debug("Created documents table")
    
    def _create_extraction_runs_table(self, cursor):
        """Create extraction runs table - simplified without abbreviation counts"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                
                -- Run metadata with defaults
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                extraction_mode VARCHAR(20) DEFAULT 'full',
                validation_method VARCHAR(50) DEFAULT 'confidence_threshold',
                
                -- Statistics with defaults
                total_drugs INTEGER DEFAULT 0,
                total_diseases INTEGER DEFAULT 0,
                
                -- Performance with defaults
                processing_time_seconds FLOAT DEFAULT 0.0,
                text_length INTEGER DEFAULT 0,
                
                -- Status with defaults
                status VARCHAR(20) DEFAULT 'pending',
                error_message TEXT DEFAULT '',
                
                FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created extraction_runs table")
    
    def _create_drugs_table(self, cursor):
        """Create drugs table - simplified without abbreviation references"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drugs (
                drug_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                
                -- Core drug data with defaults
                drug_name TEXT NOT NULL,
                normalized_name TEXT DEFAULT '',
                
                -- Identifiers with defaults
                rxcui VARCHAR(20) DEFAULT '',
                unii VARCHAR(20) DEFAULT '',
                drugbank_id VARCHAR(20) DEFAULT '',
                mesh_id VARCHAR(20) DEFAULT '',
                atc_code VARCHAR(20) DEFAULT '',
                ndc_code VARCHAR(20) DEFAULT '',
                
                -- Drug classification with defaults
                drug_type VARCHAR(50) DEFAULT 'unknown',
                route VARCHAR(50) DEFAULT '',
                mechanism_class VARCHAR(100) DEFAULT '',
                therapeutic_class VARCHAR(100) DEFAULT '',
                
                -- Detection metadata with defaults
                source VARCHAR(50) DEFAULT 'kb',
                detection_method VARCHAR(50) DEFAULT 'pattern',
                confidence FLOAT DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                
                -- Approval info with defaults
                approval_status VARCHAR(50) DEFAULT 'unknown',
                approval_year INTEGER,
                first_marketed VARCHAR(50) DEFAULT '',
                
                -- Context with defaults
                context TEXT DEFAULT '',
                positions TEXT DEFAULT '[]',
                
                -- Provenance
                provenance_span TEXT DEFAULT '{}',
                
                FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created drugs table")
    
    def _create_diseases_table(self, cursor):
        """Create diseases table - simplified without abbreviation references"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diseases (
                disease_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                
                -- Core disease data with defaults
                disease_name TEXT NOT NULL,
                canonical_name TEXT DEFAULT '',
                
                -- Identifiers with defaults
                orpha_code VARCHAR(20) DEFAULT '',
                doid VARCHAR(30) DEFAULT '',
                umls_cui VARCHAR(20) DEFAULT '',
                snomed_ct VARCHAR(30) DEFAULT '',
                mesh_id VARCHAR(20) DEFAULT '',
                mondo_id VARCHAR(30) DEFAULT '',
                omim_id VARCHAR(20) DEFAULT '',
                icd10_code VARCHAR(20) DEFAULT '',
                icd9_code VARCHAR(20) DEFAULT '',
                
                -- Disease classification with defaults
                is_rare BOOLEAN DEFAULT 0,
                semantic_type VARCHAR(50) DEFAULT 'disease',
                inheritance_pattern VARCHAR(50) DEFAULT '',
                
                -- Detection metadata with defaults
                source VARCHAR(50) DEFAULT 'kb',
                detection_method VARCHAR(50) DEFAULT 'pattern',
                confidence FLOAT DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                
                -- Context with defaults
                context TEXT DEFAULT '',
                positions TEXT DEFAULT '[]',
                matched_terms TEXT DEFAULT '[]',
                
                -- Provenance
                provenance_span TEXT DEFAULT '{}',
                
                FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created diseases table")
    
    def _create_identifier_tables(self, cursor):
        """Create tables for storing additional identifiers"""
        
        # Drug identifiers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_identifiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_id INTEGER NOT NULL,
                identifier_type VARCHAR(50) NOT NULL DEFAULT '',
                identifier_value VARCHAR(100) NOT NULL DEFAULT '',
                source VARCHAR(50) DEFAULT '',
                FOREIGN KEY (drug_id) REFERENCES drugs(drug_id) ON DELETE CASCADE
            )
        """)
        
        # Disease identifiers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_identifiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_id INTEGER NOT NULL,
                identifier_type VARCHAR(50) NOT NULL DEFAULT '',
                identifier_value VARCHAR(100) NOT NULL DEFAULT '',
                source VARCHAR(50) DEFAULT '',
                FOREIGN KEY (disease_id) REFERENCES diseases(disease_id) ON DELETE CASCADE
            )
        """)
        
        logger.debug("Created identifier tables")
    
    def _create_lexicon_tables(self, cursor):
        """Create tables for lexicon management"""
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lexicon_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                lexicon_type VARCHAR(50) NOT NULL DEFAULT '',
                term TEXT NOT NULL,
                normalized_term TEXT DEFAULT '',
                identifiers TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                source VARCHAR(50) DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lexicon_type, term)
            )
        """)
        
        logger.debug("Created lexicon tables")
    
    def _create_indexes(self, cursor):
        """Create performance indexes"""
        indexes = [
            # Document indexes
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_disease ON documents(disease_classification)",
            
            # Extraction run indexes
            "CREATE INDEX IF NOT EXISTS idx_runs_document ON extraction_runs(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON extraction_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_date ON extraction_runs(run_date)",
            
            # Drug indexes
            "CREATE INDEX IF NOT EXISTS idx_drugs_run ON drugs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_name ON drugs(drug_name)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_rxcui ON drugs(rxcui)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_mesh ON drugs(mesh_id)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_type ON drugs(drug_type)",
            
            # Disease indexes
            "CREATE INDEX IF NOT EXISTS idx_diseases_run ON diseases(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_name ON diseases(disease_name)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_orpha ON diseases(orpha_code)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_doid ON diseases(doid)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_mesh ON diseases(mesh_id)",
            
            # Identifier indexes
            "CREATE INDEX IF NOT EXISTS idx_drug_ident_rxcui ON drug_identifiers(identifier_value)",
            "CREATE INDEX IF NOT EXISTS idx_disease_ident_orpha ON disease_identifiers(identifier_value)"
        ]
        
        for idx in indexes:
            cursor.execute(idx)
        
        logger.debug(f"Created {len(indexes)} indexes")
    
    def _create_views(self, cursor):
        """Create analysis views"""
        
        # Comprehensive extraction summary
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS extraction_summary AS
            SELECT 
                d.filename,
                d.document_type,
                d.disease_classification,
                e.run_id,
                e.run_date,
                e.validation_method,
                e.extraction_mode,
                e.total_drugs,
                e.total_diseases,
                e.processing_time_seconds,
                e.status
            FROM documents d
            JOIN extraction_runs e ON d.document_id = e.document_id
        """)
        
        # Drug analysis view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS drug_analysis AS
            SELECT 
                dr.drug_name,
                dr.rxcui,
                dr.mesh_id,
                dr.drug_type,
                dr.source,
                dr.approval_status,
                COUNT(DISTINCT dr.run_id) as extraction_count,
                AVG(dr.confidence) as avg_confidence,
                SUM(dr.occurrences) as total_occurrences
            FROM drugs dr
            GROUP BY dr.drug_name, dr.rxcui
        """)
        
        # Disease analysis view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS disease_analysis AS
            SELECT 
                di.disease_name,
                di.orpha_code,
                di.doid,
                di.is_rare,
                COUNT(DISTINCT di.run_id) as extraction_count,
                AVG(di.confidence) as avg_confidence,
                SUM(di.occurrences) as total_occurrences
            FROM diseases di
            GROUP BY di.disease_name, di.orpha_code
        """)
        
        logger.debug("Created analysis views")
    
    # =========================================================================
    # Data Storage Methods
    # =========================================================================
    
    def start_extraction(self, filename: str, file_path: str = "", 
                        document_metadata: Dict = None,
                        pipeline_version: str = "unknown",
                        validation_method: str = "confidence_threshold") -> Tuple[int, int]:
        """Start a new extraction run, creating document if needed"""
        
        document_metadata = document_metadata or {}
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            # Calculate file hash
            file_hash = None
            if file_path and Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Check for existing document
            if file_hash:
                cursor.execute(
                    "SELECT document_id FROM documents WHERE file_hash = ?",
                    (file_hash,)
                )
                result = cursor.fetchone()
                if result:
                    document_id = result[0]
                    logger.debug(f"Found existing document: {document_id}")
                else:
                    document_id = self._create_document(cursor, filename, file_path, 
                                                       file_hash, document_metadata, 
                                                       pipeline_version)
            else:
                document_id = self._create_document(cursor, filename, file_path,
                                                   file_hash, document_metadata,
                                                   pipeline_version)
            
            # Create extraction run
            cursor.execute("""
                INSERT INTO extraction_runs (
                    document_id, validation_method, status
                ) VALUES (?, ?, 'in_progress')
            """, (document_id, validation_method))
            
            run_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Started extraction run {run_id} for document {document_id}")
            return document_id, run_id
    
    def _create_document(self, cursor, filename: str, file_path: str,
                        file_hash: str, metadata: Dict,
                        pipeline_version: str) -> int:
        """Create a new document record"""
        
        file_size = 0
        if file_path and Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
        
        cursor.execute("""
            INSERT INTO documents (
                filename, file_path, file_hash, file_size,
                document_type, document_subtype, disease_classification,
                title, short_description, pipeline_version, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            filename,
            file_path,
            file_hash,
            file_size,
            metadata.get('document_type', 'unknown'),
            metadata.get('document_subtype', ''),
            metadata.get('disease_classification', ''),
            metadata.get('title', ''),
            metadata.get('short_description', ''),
            pipeline_version
        ))
        
        return cursor.lastrowid
    
    def complete_extraction(self, run_id: int, 
                           total_abbreviations: int = 0,  # Kept for API compatibility
                           total_drugs: int = 0,
                           total_diseases: int = 0,
                           drugs_from_abbrev: int = 0,  # Kept for API compatibility
                           diseases_from_abbrev: int = 0,  # Kept for API compatibility
                           processing_time: float = 0.0,
                           text_length: int = 0):
        """Complete an extraction run with statistics"""
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE extraction_runs SET
                    total_drugs = ?,
                    total_diseases = ?,
                    processing_time_seconds = ?,
                    text_length = ?,
                    status = 'completed'
                WHERE run_id = ?
            """, (total_drugs, total_diseases, processing_time, text_length, run_id))
            
            # Update document status
            cursor.execute("""
                UPDATE documents SET status = 'completed'
                WHERE document_id = (
                    SELECT document_id FROM extraction_runs WHERE run_id = ?
                )
            """, (run_id,))
            
            conn.commit()
            logger.info(f"Completed extraction run {run_id}: {total_drugs} drugs, {total_diseases} diseases")
    
    def save_drugs_with_counts(self, run_id: int, drugs: List[Dict[str, Any]], 
                              abbrev_id_map: Dict = None,
                              text_content: str = "") -> int:
        """Save drugs with occurrence counts"""
        saved_count = 0
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            for drug in drugs:
                # Count occurrences in text
                occurrences = 1
                if text_content and drug.get('name'):
                    pattern = re.escape(drug.get('name'))
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    occurrences = len(matches) or 1
                
                positions = json.dumps(drug.get('positions', []))
                provenance = json.dumps(drug.get('provenance_span', {}))
                
                cursor.execute("""
                    INSERT INTO drugs (
                        run_id, drug_name, normalized_name,
                        rxcui, unii, drugbank_id, mesh_id, atc_code, ndc_code,
                        drug_type, route, mechanism_class, therapeutic_class,
                        source, detection_method, confidence, occurrences,
                        approval_status, approval_year, first_marketed,
                        context, positions, provenance_span
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    drug.get('name') or '',
                    drug.get('normalized_name') or drug.get('name') or '',
                    drug.get('rxcui') or '',
                    drug.get('unii') or '',
                    drug.get('drugbank_id') or '',
                    drug.get('mesh_id') or '',
                    drug.get('atc_code') or '',
                    drug.get('ndc_code') or '',
                    drug.get('drug_type') or 'unknown',
                    drug.get('route') or '',
                    drug.get('mechanism_class') or '',
                    drug.get('therapeutic_class') or '',
                    drug.get('source') or 'kb',
                    drug.get('detection_method') or 'pattern',
                    drug.get('confidence') or 0.0,
                    occurrences,
                    drug.get('approval_status') or 'unknown',
                    drug.get('approval_year'),
                    drug.get('first_marketed') or '',
                    drug.get('context') or '',
                    positions,
                    provenance
                ))
                
                drug_id = cursor.lastrowid
                
                # Save additional identifiers
                all_ids = drug.get('all_ids', {})
                for id_type, id_value in all_ids.items():
                    if id_value:
                        cursor.execute("""
                            INSERT INTO drug_identifiers (drug_id, identifier_type, identifier_value, source)
                            VALUES (?, ?, ?, 'extraction')
                        """, (drug_id, id_type, str(id_value)))
                
                saved_count += 1
            
            conn.commit()
            logger.debug(f"Saved {saved_count} drugs for run {run_id}")
        
        return saved_count
    
    def save_diseases_with_counts(self, run_id: int, diseases: List[Dict[str, Any]],
                                 abbrev_id_map: Dict = None,
                                 text_content: str = "") -> int:
        """Save diseases with occurrence counts"""
        saved_count = 0
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            for disease in diseases:
                # Count occurrences
                occurrences = 1
                if text_content and disease.get('name'):
                    pattern = re.escape(disease.get('name'))
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    occurrences = len(matches) or 1
                
                positions = json.dumps(disease.get('positions', []))
                matched_terms = json.dumps(disease.get('matched_terms', []))
                provenance = json.dumps(disease.get('provenance_span', {}))
                
                cursor.execute("""
                    INSERT INTO diseases (
                        run_id, disease_name, canonical_name,
                        orpha_code, doid, umls_cui, snomed_ct, mesh_id, mondo_id,
                        omim_id, icd10_code, icd9_code,
                        is_rare, semantic_type, inheritance_pattern,
                        source, detection_method, confidence, occurrences,
                        context, positions, matched_terms, provenance_span
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    disease.get('name') or '',
                    disease.get('canonical_name') or disease.get('name') or '',
                    disease.get('orpha_code') or disease.get('ORPHA') or '',
                    disease.get('doid') or disease.get('DOID') or '',
                    disease.get('umls_cui') or disease.get('UMLS') or '',
                    disease.get('snomed_ct') or disease.get('SNOMED') or '',
                    disease.get('mesh_id') or disease.get('MESH') or '',
                    disease.get('mondo_id') or disease.get('MONDO') or '',
                    disease.get('omim_id') or disease.get('OMIM') or '',
                    disease.get('icd10_code') or disease.get('ICD10') or '',
                    disease.get('icd9_code') or disease.get('ICD9') or '',
                    disease.get('is_rare', False),
                    disease.get('semantic_type') or 'disease',
                    disease.get('inheritance_pattern') or '',
                    disease.get('source') or 'kb',
                    disease.get('detection_method') or 'pattern',
                    disease.get('confidence') or 0.0,
                    occurrences,
                    disease.get('context') or '',
                    positions,
                    matched_terms,
                    provenance
                ))
                
                disease_id = cursor.lastrowid
                
                # Save additional identifiers
                all_ids = disease.get('all_ids', {})
                for id_type, id_value in all_ids.items():
                    if id_value:
                        cursor.execute("""
                            INSERT INTO disease_identifiers (disease_id, identifier_type, identifier_value, source)
                            VALUES (?, ?, ?, 'extraction')
                        """, (disease_id, id_type, str(id_value)))
                
                saved_count += 1
            
            conn.commit()
            logger.debug(f"Saved {saved_count} diseases for run {run_id}")
        
        return saved_count
    
    def get_extraction_summary(self, run_id: int = None) -> Dict[str, Any]:
        """Get extraction summary statistics"""
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            if run_id:
                cursor.execute("""
                    SELECT 
                        e.run_id,
                        d.filename,
                        d.document_type,
                        e.total_drugs,
                        e.total_diseases,
                        e.processing_time_seconds,
                        e.status
                    FROM extraction_runs e
                    JOIN documents d ON e.document_id = d.document_id
                    WHERE e.run_id = ?
                """, (run_id,))
            else:
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT e.run_id) as total_runs,
                        SUM(e.total_drugs) as total_drugs,
                        SUM(e.total_diseases) as total_diseases,
                        AVG(e.processing_time_seconds) as avg_processing_time,
                        COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as completed_runs
                    FROM extraction_runs e
                """)
            
            result = cursor.fetchone()
            
            if result:
                return dict(result)
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None