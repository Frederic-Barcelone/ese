#!/usr/bin/env python3
"""
Entity Extraction Database Setup
=================================
Location: corpus_metadata/document_utils/entity_db_extraction.py

Sets up SQLite database for storing comprehensive extraction pipeline results.
Updated with proper defaults to prevent NULL value errors in report generation.
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
    """Manages SQLite database for comprehensive extraction results"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection
        
        Args:
            db_path: Path to database file. If None, uses default location.
        """
        if db_path is None:
            # Set to specified corpus_db folder
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
        # Enable foreign keys
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
                self._create_abbreviations_table(cursor)
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
        """Create extraction runs table with proper defaults"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                
                -- Run metadata with defaults
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                extraction_mode VARCHAR(20) DEFAULT 'full',
                validation_method VARCHAR(50) DEFAULT 'confidence_threshold',
                
                -- Statistics with defaults
                total_abbreviations INTEGER DEFAULT 0,
                total_drugs INTEGER DEFAULT 0,
                total_diseases INTEGER DEFAULT 0,
                drugs_from_abbreviations INTEGER DEFAULT 0,
                diseases_from_abbreviations INTEGER DEFAULT 0,
                
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
    
    def _create_abbreviations_table(self, cursor):
        """Create comprehensive abbreviations table with defaults"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abbreviations (
                abbrev_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                
                -- Core abbreviation data with defaults
                abbreviation VARCHAR(50) NOT NULL,
                expansion TEXT DEFAULT '',
                
                -- Classification with defaults
                context_type VARCHAR(50) DEFAULT 'general',
                semantic_type VARCHAR(100) DEFAULT '',
                
                -- Detection details with defaults
                confidence FLOAT DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                source_method VARCHAR(100) DEFAULT 'unknown',
                dictionary_sources TEXT DEFAULT '[]',
                
                -- Conflict resolution with defaults
                conflict_resolved BOOLEAN DEFAULT 0,
                resolution_applied TEXT DEFAULT '',
                original_expansion TEXT DEFAULT '',
                alternative_expansions TEXT DEFAULT '[]',
                
                -- Position information with defaults
                first_occurrence INTEGER DEFAULT 0,
                positions TEXT DEFAULT '[]',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE CASCADE
            )
        """)
        logger.debug("Created abbreviations table")
    
    def _create_drugs_table(self, cursor):
        """Create comprehensive drugs table with defaults"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drugs (
                drug_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                
                -- Core drug information with defaults
                drug_name TEXT NOT NULL,
                normalized_name TEXT DEFAULT '',
                canonical_name TEXT DEFAULT '',
                
                -- Multiple identifier systems (can be NULL)
                rxcui VARCHAR(20),
                mesh_id VARCHAR(50),
                chebi_id VARCHAR(50),
                unii VARCHAR(20),
                ndc VARCHAR(20),
                atc_code VARCHAR(20),
                drugbank_id VARCHAR(20),
                pubchem_cid VARCHAR(20),
                
                -- Drug classification with defaults
                drug_type VARCHAR(100) DEFAULT '',
                drug_class VARCHAR(100) DEFAULT '',
                mechanism VARCHAR(200) DEFAULT '',
                tty VARCHAR(20) DEFAULT '',
                
                -- Source and status with defaults
                source VARCHAR(100) DEFAULT 'unknown',
                approval_status VARCHAR(50) DEFAULT '',
                fda_approval_date DATE,
                
                -- Names with defaults (JSON arrays)
                brand_names TEXT DEFAULT '[]',
                generic_names TEXT DEFAULT '[]',
                synonyms TEXT DEFAULT '[]',
                
                -- Clinical trial info with defaults
                nct_ids TEXT DEFAULT '[]',
                trial_phase VARCHAR(20) DEFAULT '',
                conditions_studied TEXT DEFAULT '[]',
                
                -- Detection metadata with defaults
                confidence FLOAT DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                detection_method VARCHAR(100) DEFAULT 'unknown',
                from_abbreviation_id INTEGER,
                
                -- Validation with defaults
                validation_status VARCHAR(20) DEFAULT 'unvalidated',
                claude_approved BOOLEAN DEFAULT 0,
                claude_decision TEXT DEFAULT '',
                claude_reason TEXT DEFAULT '',
                pubtator_validated BOOLEAN DEFAULT 0,
                
                -- Additional metadata with default
                metadata TEXT DEFAULT '{}',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (from_abbreviation_id) REFERENCES abbreviations(abbrev_id) ON DELETE SET NULL
            )
        """)
        logger.debug("Created drugs table")
    
    def _create_diseases_table(self, cursor):
        """Create comprehensive diseases table with defaults"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diseases (
                disease_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                
                -- Core disease information with defaults
                disease_name TEXT NOT NULL,
                normalized_name TEXT DEFAULT '',
                canonical_name TEXT DEFAULT '',
                
                -- Multiple identifier systems (can be NULL)
                orpha_code VARCHAR(20),
                doid VARCHAR(20),
                mesh_id VARCHAR(50),
                omim_id VARCHAR(20),
                icd10_codes TEXT DEFAULT '[]',
                icd11_codes TEXT DEFAULT '[]',
                snomed_ct VARCHAR(20),
                umls_cui VARCHAR(20),
                mondo_id VARCHAR(20),
                hp_terms TEXT DEFAULT '[]',
                
                -- Disease classification with defaults
                disease_category VARCHAR(100) DEFAULT '',
                disease_group VARCHAR(100) DEFAULT '',
                inheritance_pattern VARCHAR(50) DEFAULT '',
                
                -- Epidemiology with defaults
                is_rare BOOLEAN DEFAULT 0,
                prevalence VARCHAR(100) DEFAULT '',
                onset_age VARCHAR(50) DEFAULT '',
                
                -- Source information with defaults
                detection_method VARCHAR(100) DEFAULT 'unknown',
                source VARCHAR(100) DEFAULT 'unknown',
                lexicon_source VARCHAR(50) DEFAULT '',
                
                -- Detection metadata with defaults
                confidence FLOAT DEFAULT 0.0,
                occurrences INTEGER DEFAULT 1,
                is_primary BOOLEAN DEFAULT 0,
                context_type VARCHAR(50) DEFAULT 'general',
                
                -- Abbreviation linkage
                from_abbreviation_id INTEGER,
                matched_terms TEXT DEFAULT '[]',
                
                -- Validation with defaults
                validation_status VARCHAR(20) DEFAULT 'unvalidated',
                claude_approved BOOLEAN DEFAULT 0,
                pubtator_validated BOOLEAN DEFAULT 0,
                
                -- Additional data with defaults
                synonyms TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES extraction_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (from_abbreviation_id) REFERENCES abbreviations(abbrev_id) ON DELETE SET NULL
            )
        """)
        logger.debug("Created diseases table")
    
    def _create_identifier_tables(self, cursor):
        """Create identifier mapping tables"""
        
        # Drug identifier mappings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_identifiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rxcui VARCHAR(20),
                mesh_id VARCHAR(50),
                chebi_id VARCHAR(50),
                unii VARCHAR(20),
                drugbank_id VARCHAR(20),
                canonical_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(rxcui, mesh_id, chebi_id)
            )
        """)
        
        # Disease identifier mappings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_identifiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                orpha_code VARCHAR(20),
                doid VARCHAR(20),
                mesh_id VARCHAR(50),
                omim_id VARCHAR(20),
                mondo_id VARCHAR(20),
                canonical_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(orpha_code, doid, mesh_id)
            )
        """)
        
        logger.debug("Created identifier mapping tables")
    
    def _create_lexicon_tables(self, cursor):
        """Create reference lexicon tables"""
        
        # Drug lexicon
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drug_lexicon (
                lexicon_id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_name TEXT UNIQUE NOT NULL,
                drug_type VARCHAR(100),
                source VARCHAR(50),
                rxcui VARCHAR(20),
                mesh_id VARCHAR(50),
                is_investigational BOOLEAN DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Disease lexicon
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_lexicon (
                lexicon_id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_name TEXT UNIQUE NOT NULL,
                orpha_code VARCHAR(20),
                doid VARCHAR(20),
                icd10_code VARCHAR(20),
                is_rare BOOLEAN DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        logger.debug("Created lexicon tables")
    
    def _create_indexes(self, cursor):
        """Create performance indexes"""
        indexes = [
            # Document indexes
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
            
            # Run indexes
            "CREATE INDEX IF NOT EXISTS idx_runs_document ON extraction_runs(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_date ON extraction_runs(run_date)",
            
            # Abbreviation indexes
            "CREATE INDEX IF NOT EXISTS idx_abbrev_context ON abbreviations(context_type)",
            "CREATE INDEX IF NOT EXISTS idx_abbrev_run ON abbreviations(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_abbrev_text ON abbreviations(abbreviation)",
            
            # Drug indexes
            "CREATE INDEX IF NOT EXISTS idx_drugs_run ON drugs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_rxcui ON drugs(rxcui)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_mesh ON drugs(mesh_id)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_source ON drugs(source)",
            "CREATE INDEX IF NOT EXISTS idx_drugs_from_abbrev ON drugs(from_abbreviation_id)",
            
            # Disease indexes
            "CREATE INDEX IF NOT EXISTS idx_diseases_run ON diseases(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_orpha ON diseases(orpha_code)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_doid ON diseases(doid)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_mesh ON diseases(mesh_id)",
            "CREATE INDEX IF NOT EXISTS idx_diseases_from_abbrev ON diseases(from_abbreviation_id)",
            
            # Identifier indexes
            "CREATE INDEX IF NOT EXISTS idx_drug_ident_rxcui ON drug_identifiers(rxcui)",
            "CREATE INDEX IF NOT EXISTS idx_disease_ident_orpha ON disease_identifiers(orpha_code)"
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
                e.total_abbreviations,
                e.total_drugs,
                e.total_diseases,
                e.drugs_from_abbreviations,
                e.diseases_from_abbreviations,
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
                SUM(dr.occurrences) as total_occurrences,
                COUNT(DISTINCT dr.from_abbreviation_id) as from_abbreviations
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
                SUM(di.occurrences) as total_occurrences,
                COUNT(DISTINCT di.from_abbreviation_id) as from_abbreviations
            FROM diseases di
            GROUP BY di.disease_name, di.orpha_code
        """)
        
        # Abbreviation effectiveness view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS abbreviation_effectiveness AS
            SELECT 
                a.context_type,
                COUNT(DISTINCT a.abbrev_id) as total_abbreviations,
                COUNT(DISTINCT dr.drug_id) as drugs_generated,
                COUNT(DISTINCT di.disease_id) as diseases_generated,
                AVG(a.confidence) as avg_confidence
            FROM abbreviations a
            LEFT JOIN drugs dr ON dr.from_abbreviation_id = a.abbrev_id
            LEFT JOIN diseases di ON di.from_abbreviation_id = a.abbrev_id
            GROUP BY a.context_type
        """)
        
        logger.debug("Created analysis views")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def calculate_document_occurrences(self, text: str, entity_name: str) -> int:
        """Count actual occurrences of entity in document text"""
        if not text or not entity_name:
            return 1
        
        escaped_name = re.escape(entity_name)
        pattern = r'\b' + escaped_name + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return max(len(matches), 1)
    
    def start_extraction(self, filename: str, file_path: str = None, 
                        document_metadata: Dict = None, pipeline_version: str = None,
                        validation_method: str = None, extraction_mode: str = 'full') -> Tuple[int, int]:
        """Start new extraction: delete old data, create new document and run"""
        file_hash = None
        file_size = 0
        
        if file_path and os.path.exists(file_path):
            file_hash = self.calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            # Enable foreign keys for CASCADE
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Find and delete existing documents
            cursor.execute("""
                SELECT document_id 
                FROM documents 
                WHERE filename = ? OR (file_hash = ? AND file_hash IS NOT NULL)
            """, (filename, file_hash))
            existing_docs = cursor.fetchall()
            
            for doc in existing_docs:
                doc_id = doc['document_id']
                cursor.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,))
                logger.info(f"Deleted existing document {doc_id}")
            
            conn.commit()
            
            # Insert new document with defaults for NULL values
            cursor.execute("""
                INSERT INTO documents (
                    filename, file_path, file_hash, file_size,
                    document_type, document_subtype, disease_classification,
                    title, short_description, language,
                    pipeline_version, page_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename, 
                file_path or '',
                file_hash,
                file_size or 0,
                (document_metadata.get('document_type') if document_metadata else None) or 'unknown',
                (document_metadata.get('document_subtype') if document_metadata else None) or '',
                (document_metadata.get('disease_classification') if document_metadata else None) or '',
                (document_metadata.get('title') if document_metadata else None) or '',
                (document_metadata.get('short_description') if document_metadata else None) or '',
                (document_metadata.get('language') if document_metadata else None) or 'en',
                pipeline_version or 'unknown',
                (document_metadata.get('page_count') if document_metadata else None) or 0
            ))
            
            document_id = cursor.lastrowid
            
            # Insert new extraction run with defaults
            cursor.execute("""
                INSERT INTO extraction_runs (
                    document_id, validation_method, extraction_mode, status, text_length
                )
                VALUES (?, ?, ?, 'running', 0)
            """, (document_id, validation_method or 'confidence_threshold', extraction_mode or 'full'))
            
            run_id = cursor.lastrowid
            
            logger.info(f"Started new extraction - Document: {document_id}, Run: {run_id}, File: {filename}")
            
            return document_id, run_id
    
    def save_abbreviations_with_counts(self, run_id: int, abbreviations: List[Dict[str, Any]], 
                                      text_content: str = None) -> Dict[str, int]:
        """Save abbreviations with actual document occurrence counts"""
        abbrev_id_map = {}
        
        with self.connect() as conn:
            cursor = conn.cursor()
            
            for abbrev in abbreviations:
                if text_content and abbrev.get('abbreviation'):
                    actual_occurrences = self.calculate_document_occurrences(
                        text_content, 
                        abbrev.get('abbreviation')
                    )
                else:
                    actual_occurrences = 1
                
                dict_sources = json.dumps(abbrev.get('dictionary_sources', []))
                alternatives = json.dumps(abbrev.get('alternative_expansions', []))
                positions = json.dumps(abbrev.get('positions', []))
                
                cursor.execute("""
                    INSERT INTO abbreviations (
                        run_id, abbreviation, expansion, context_type,
                        semantic_type, confidence, occurrences, source_method,
                        dictionary_sources, conflict_resolved, resolution_applied,
                        original_expansion, alternative_expansions, first_occurrence, positions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    abbrev.get('abbreviation'),
                    abbrev.get('expansion') or '',
                    abbrev.get('context_type') or 'general',
                    abbrev.get('semantic_type') or '',
                    abbrev.get('confidence') or 0.0,
                    actual_occurrences,
                    abbrev.get('source_method') or abbrev.get('source') or 'unknown',
                    dict_sources,
                    abbrev.get('conflict_resolved', False),
                    abbrev.get('resolution_applied') or '',
                    abbrev.get('original_expansion') or '',
                    alternatives,
                    abbrev.get('first_occurrence') or 0,
                    positions
                ))
                
                abbrev_id_map[abbrev.get('abbreviation')] = cursor.lastrowid
            
            logger.debug(f"Saved {len(abbreviations)} abbreviations for run {run_id}")
        
        return abbrev_id_map
    
    def save_drugs_with_counts(self, run_id: int, drugs: List[Dict[str, Any]], 
                              abbrev_map: Dict[str, int] = None, text_content: str = None):
        """Save drug entities with actual document occurrence counts"""
        with self.connect() as conn:
            cursor = conn.cursor()
            
            for drug in drugs:
                drug_name = drug.get('name') or drug.get('drug_name')
                if text_content and drug_name:
                    actual_occurrences = self.calculate_document_occurrences(
                        text_content, 
                        drug_name
                    )
                else:
                    actual_occurrences = 1
                
                from_abbrev_id = None
                if abbrev_map and drug.get('from_abbreviation'):
                    from_abbrev_id = abbrev_map.get(drug['from_abbreviation'])
                
                brand_names = json.dumps(drug.get('brand_names', []))
                generic_names = json.dumps(drug.get('generic_names', []))
                synonyms = json.dumps(drug.get('synonyms', []))
                nct_ids = json.dumps(drug.get('nct_ids', []))
                conditions = json.dumps(drug.get('conditions_studied', []))
                metadata = json.dumps(drug.get('metadata', {}))
                
                cursor.execute("""
                    INSERT INTO drugs (
                        run_id, drug_name, normalized_name, canonical_name,
                        rxcui, mesh_id, chebi_id, unii, ndc, atc_code,
                        drugbank_id, pubchem_cid, drug_type, drug_class,
                        mechanism, tty, source, approval_status, fda_approval_date,
                        brand_names, generic_names, synonyms, nct_ids, trial_phase,
                        conditions_studied, confidence, occurrences, detection_method,
                        from_abbreviation_id, validation_status, claude_approved,
                        claude_decision, claude_reason, pubtator_validated, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    drug_name,
                    drug.get('normalized_name') or '',
                    drug.get('canonical_name') or '',
                    drug.get('rxcui'),
                    drug.get('mesh_id'),
                    drug.get('chebi_id'),
                    drug.get('unii'),
                    drug.get('ndc'),
                    drug.get('atc_code'),
                    drug.get('drugbank_id'),
                    drug.get('pubchem_cid'),
                    drug.get('drug_type') or '',
                    drug.get('drug_class') or '',
                    drug.get('mechanism') or '',
                    drug.get('tty') or '',
                    drug.get('source') or 'unknown',
                    drug.get('approval_status') or '',
                    drug.get('fda_approval_date'),
                    brand_names,
                    generic_names,
                    synonyms,
                    nct_ids,
                    drug.get('trial_phase') or '',
                    conditions,
                    drug.get('confidence') or 0.0,
                    actual_occurrences,
                    drug.get('detection_method') or 'unknown',
                    from_abbrev_id,
                    drug.get('validation_status') or 'unvalidated',
                    drug.get('claude_approved') or False,
                    drug.get('claude_decision') or '',
                    drug.get('claude_reason') or '',
                    drug.get('pubtator_validated') or False,
                    metadata
                ))
            
            logger.debug(f"Saved {len(drugs)} drugs for run {run_id}")
    
    def save_diseases_with_counts(self, run_id: int, diseases: List[Dict[str, Any]], 
                                 abbrev_map: Dict[str, int] = None, text_content: str = None):
        """Save disease entities with actual document occurrence counts"""
        with self.connect() as conn:
            cursor = conn.cursor()
            
            for disease in diseases:
                disease_name = disease.get('name') or disease.get('disease_name')
                if text_content and disease_name:
                    actual_occurrences = self.calculate_document_occurrences(
                        text_content, 
                        disease_name
                    )
                else:
                    actual_occurrences = 1
                
                from_abbrev_id = None
                if abbrev_map and disease.get('from_abbreviation'):
                    from_abbrev_id = abbrev_map.get(disease['from_abbreviation'])
                
                icd10_codes = json.dumps(disease.get('icd10_codes', []))
                icd11_codes = json.dumps(disease.get('icd11_codes', []))
                hp_terms = json.dumps(disease.get('hp_terms', []))
                matched_terms = json.dumps(disease.get('matched_terms', []))
                synonyms = json.dumps(disease.get('synonyms', []))
                metadata = json.dumps(disease.get('metadata', {}))
                
                cursor.execute("""
                    INSERT INTO diseases (
                        run_id, disease_name, normalized_name, canonical_name,
                        orpha_code, doid, mesh_id, omim_id, icd10_codes,
                        icd11_codes, snomed_ct, umls_cui, mondo_id, hp_terms,
                        disease_category, disease_group, inheritance_pattern,
                        is_rare, prevalence, onset_age, detection_method,
                        source, lexicon_source, confidence, occurrences,
                        is_primary, context_type, from_abbreviation_id,
                        matched_terms, validation_status, claude_approved,
                        pubtator_validated, synonyms, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    disease_name,
                    disease.get('normalized_name') or '',
                    disease.get('canonical_name') or '',
                    disease.get('orpha_code') or disease.get('orphacode'),
                    disease.get('doid'),
                    disease.get('mesh_id'),
                    disease.get('omim_id'),
                    icd10_codes,
                    icd11_codes,
                    disease.get('snomed_ct'),
                    disease.get('umls_cui'),
                    disease.get('mondo_id'),
                    hp_terms,
                    disease.get('disease_category') or '',
                    disease.get('disease_group') or '',
                    disease.get('inheritance_pattern') or '',
                    disease.get('is_rare', False),
                    disease.get('prevalence') or '',
                    disease.get('onset_age') or '',
                    disease.get('method') or disease.get('detection_method') or 'unknown',
                    disease.get('source') or 'unknown',
                    disease.get('lexicon_source') or '',
                    disease.get('confidence') or 0.0,
                    actual_occurrences,
                    disease.get('is_primary', False),
                    disease.get('context_type') or 'general',
                    from_abbrev_id,
                    matched_terms,
                    disease.get('validation_status') or 'unvalidated',
                    disease.get('claude_approved') or False,
                    disease.get('pubtator_validated') or False,
                    synonyms,
                    metadata
                ))
            
            logger.debug(f"Saved {len(diseases)} diseases for run {run_id}")
    
    def complete_extraction(self, 
                        run_id: int,
                        total_abbreviations: int = 0,
                        total_drugs: int = 0,
                        total_diseases: int = 0,
                        drugs_from_abbreviations: int = 0,
                        diseases_from_abbreviations: int = 0,
                        processing_time: float = 0.0,
                        text_length: int = 0,
                        status: str = 'completed',
                        error_message: str = None):
        """
        Mark extraction as complete and update statistics
        
        Args:
            run_id: The extraction run ID
            total_abbreviations: Total abbreviations extracted
            total_drugs: Total drugs extracted
            total_diseases: Total diseases extracted
            drugs_from_abbreviations: Number of drugs promoted from abbreviations
            diseases_from_abbreviations: Number of diseases promoted from abbreviations
            processing_time: Processing time in seconds
            text_length: Length of processed text
            status: Extraction status ('completed', 'failed', etc.)
            error_message: Optional error message if status is 'failed'
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            
            # If counts not provided, get from database
            if total_abbreviations == 0:
                cursor.execute("SELECT COUNT(*) FROM abbreviations WHERE run_id = ?", (run_id,))
                total_abbreviations = cursor.fetchone()[0]
            
            if total_drugs == 0:
                cursor.execute("SELECT COUNT(*) FROM drugs WHERE run_id = ?", (run_id,))
                total_drugs = cursor.fetchone()[0]
            
            if total_diseases == 0:
                cursor.execute("SELECT COUNT(*) FROM diseases WHERE run_id = ?", (run_id,))
                total_diseases = cursor.fetchone()[0]
            
            # Get promoted counts if not provided
            if drugs_from_abbreviations == 0:
                cursor.execute("""
                    SELECT COUNT(*) FROM drugs 
                    WHERE run_id = ? AND from_abbreviation_id IS NOT NULL
                """, (run_id,))
                drugs_from_abbreviations = cursor.fetchone()[0]
            
            if diseases_from_abbreviations == 0:
                cursor.execute("""
                    SELECT COUNT(*) FROM diseases 
                    WHERE run_id = ? AND from_abbreviation_id IS NOT NULL
                """, (run_id,))
                diseases_from_abbreviations = cursor.fetchone()[0]
            
            # Update extraction_runs table with all statistics
            cursor.execute("""
                UPDATE extraction_runs 
                SET total_abbreviations = ?,
                    total_drugs = ?,
                    total_diseases = ?,
                    drugs_from_abbreviations = ?,
                    diseases_from_abbreviations = ?,
                    processing_time_seconds = ?,
                    text_length = ?,
                    status = ?,
                    error_message = ?
                WHERE run_id = ?
            """, (
                total_abbreviations,
                total_drugs,
                total_diseases,
                drugs_from_abbreviations,
                diseases_from_abbreviations,
                processing_time or 0.0,
                text_length or 0,
                status,
                error_message or '',
                run_id
            ))
            
            conn.commit()
            
            logger.info(f"Completed extraction run {run_id}: "
                    f"{total_abbreviations} abbrev, {total_drugs} drugs, {total_diseases} diseases "
                    f"({drugs_from_abbreviations} drugs + {diseases_from_abbreviations} diseases from abbrev)")
    
    def get_extraction_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent extraction summaries"""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM extraction_summary 
                ORDER BY run_date DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_drug_statistics(self) -> Dict:
        """Get drug extraction statistics"""
        with self.connect() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(DISTINCT drug_name) FROM drugs")
            stats['unique_drugs'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT rxcui) FROM drugs WHERE rxcui IS NOT NULL")
            stats['drugs_with_rxcui'] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT source, COUNT(*) 
                FROM drugs 
                GROUP BY source
            """)
            stats['by_source'] = dict(cursor.fetchall())
            
            cursor.execute("SELECT AVG(confidence) FROM drugs")
            stats['avg_confidence'] = cursor.fetchone()[0] or 0.0
            
            return stats
    
    def get_disease_statistics(self) -> Dict:
        """Get disease extraction statistics"""
        with self.connect() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(DISTINCT disease_name) FROM diseases")
            stats['unique_diseases'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT disease_name) FROM diseases WHERE is_rare = 1")
            stats['rare_diseases'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT orpha_code) FROM diseases WHERE orpha_code IS NOT NULL")
            stats['with_orpha_code'] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT detection_method, COUNT(*) 
                FROM diseases 
                GROUP BY detection_method
            """)
            stats['by_method'] = dict(cursor.fetchall())
            
            return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")


if __name__ == "__main__":
    # Initialize database only
    db = ExtractionDatabase()
    logger.info("Database initialized successfully")
    db.close()