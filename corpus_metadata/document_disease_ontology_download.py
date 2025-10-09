#!/usr/bin/env python3
"""
Disease Ontology (DO) Database Builder
======================================
Downloads and processes the Disease Ontology to create a standalone disease database.

Creates a separate SQLite database with:
- Disease names and synonyms
- DO IDs and cross-references
- Optimized search indices
- Integration-ready with your existing system

The database is kept separate from Orphanet to maintain:
- Data source integrity
- Independent updates
- Flexible usage options

Usage:
    python download_disease_ontology.py
    
Output:
    ./corpus_db/disease_ontology.db
"""

import os
import sqlite3
import requests
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Set, Tuple, Optional
import re
from collections import defaultdict
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiseaseOntologyBuilder:
    """Downloads and builds Disease Ontology database"""
    
    # Disease Ontology download URL
    DO_URL = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/HumanDO.obo"
    
    def __init__(self, output_dir: str = "./corpus_db"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "disease_ontology.db"
        self.obo_path = self.output_dir / "HumanDO.obo"
        
        # Disease data storage
        self.diseases = {}  # doid -> disease info
        self.name_to_doid = {}  # normalized name -> doid
        self.stats = defaultdict(int)
        
    def download_ontology(self) -> bool:
        """Download the latest Disease Ontology OBO file"""
        logger.info("Downloading Disease Ontology...")
        
        try:
            response = requests.get(self.DO_URL, stream=True)
            response.raise_for_status()
            
            # Save to file
            total_size = int(response.headers.get('content-length', 0))
            with open(self.obo_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✅ Downloaded to {self.obo_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def parse_obo_file(self):
        """Parse the OBO file to extract disease information"""
        logger.info("Parsing Disease Ontology OBO file...")
        
        with open(self.obo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into term blocks
        term_blocks = content.split('[Term]')[1:]  # Skip header
        
        for block in tqdm(term_blocks, desc="Parsing diseases"):
            disease_info = self._parse_term_block(block)
            if disease_info and not disease_info.get('is_obsolete'):
                doid = disease_info['id']
                self.diseases[doid] = disease_info
                
                # Index by name
                name = disease_info['name'].lower().strip()
                self.name_to_doid[name] = doid
                
                # Index by synonyms
                for syn in disease_info.get('synonyms', []):
                    syn_lower = syn.lower().strip()
                    if syn_lower and syn_lower != name:
                        self.name_to_doid[syn_lower] = doid
        
        logger.info(f"✅ Parsed {len(self.diseases)} active diseases")
        logger.info(f"✅ Created {len(self.name_to_doid)} name mappings")
    
    def _parse_term_block(self, block: str) -> Optional[Dict]:
        """Parse a single term block from OBO file"""
        lines = block.strip().split('\n')
        disease_info = {
            'synonyms': [],
            'xrefs': [],
            'alt_ids': [],
            'subsets': [],
            'is_obsolete': False
        }
        
        for line in lines:
            line = line.strip()
            if not line or not ':' in line:
                continue
                
            key, value = line.split(':', 1)
            value = value.strip()
            
            if key == 'id':
                disease_info['id'] = value
            elif key == 'name':
                disease_info['name'] = value
            elif key == 'def':
                # Extract definition text
                match = re.match(r'"([^"]*)"', value)
                if match:
                    disease_info['definition'] = match.group(1)
            elif key == 'synonym':
                # Extract synonym text
                match = re.match(r'"([^"]*)"', value)
                if match:
                    disease_info['synonyms'].append(match.group(1))
            elif key == 'xref':
                disease_info['xrefs'].append(value)
            elif key == 'alt_id':
                disease_info['alt_ids'].append(value)
            elif key == 'subset':
                disease_info['subsets'].append(value)
            elif key == 'is_obsolete' and value == 'true':
                disease_info['is_obsolete'] = True
            elif key == 'replaced_by':
                disease_info['replaced_by'] = value
        
        # Must have ID and name
        if 'id' in disease_info and 'name' in disease_info:
            self.stats['total_parsed'] += 1
            if disease_info['is_obsolete']:
                self.stats['obsolete'] += 1
            return disease_info
        
        return None
    
    def create_database(self):
        """Create SQLite database with disease data"""
        logger.info(f"Creating database at {self.db_path}...")
        
        # Remove existing database
        if self.db_path.exists():
            self.db_path.unlink()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE diseases (
                doid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition TEXT,
                is_obsolete BOOLEAN DEFAULT 0,
                replaced_by TEXT,
                created_date TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doid TEXT NOT NULL,
                synonym TEXT NOT NULL,
                FOREIGN KEY (doid) REFERENCES diseases(doid)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE xrefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doid TEXT NOT NULL,
                source TEXT NOT NULL,
                identifier TEXT NOT NULL,
                FOREIGN KEY (doid) REFERENCES diseases(doid)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE disease_search (
                term TEXT PRIMARY KEY,
                doid TEXT NOT NULL,
                term_type TEXT NOT NULL,  -- 'name' or 'synonym'
                normalized_term TEXT NOT NULL,
                FOREIGN KEY (doid) REFERENCES diseases(doid)
            )
        ''')
        
        # Insert data
        created_date = datetime.now().isoformat()
        
        for doid, info in tqdm(self.diseases.items(), desc="Inserting diseases"):
            # Insert disease
            cursor.execute('''
                INSERT INTO diseases (doid, name, definition, is_obsolete, replaced_by, created_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                doid,
                info['name'],
                info.get('definition'),
                info.get('is_obsolete', False),
                info.get('replaced_by'),
                created_date
            ))
            
            # Insert name into search table
            cursor.execute('''
                INSERT INTO disease_search (term, doid, term_type, normalized_term)
                VALUES (?, ?, ?, ?)
            ''', (
                info['name'],
                doid,
                'name',
                info['name'].lower().strip()
            ))
            
            # Insert synonyms
            for synonym in info.get('synonyms', []):
                cursor.execute('''
                    INSERT INTO synonyms (doid, synonym) VALUES (?, ?)
                ''', (doid, synonym))
                
                # Also add to search table
                cursor.execute('''
                    INSERT OR IGNORE INTO disease_search (term, doid, term_type, normalized_term)
                    VALUES (?, ?, ?, ?)
                ''', (
                    synonym,
                    doid,
                    'synonym',
                    synonym.lower().strip()
                ))
            
            # Insert cross-references
            for xref in info.get('xrefs', []):
                if ':' in xref:
                    source, identifier = xref.split(':', 1)
                    cursor.execute('''
                        INSERT INTO xrefs (doid, source, identifier) VALUES (?, ?, ?)
                    ''', (doid, source, identifier))
        
        # Create indices for fast searching
        cursor.execute('CREATE INDEX idx_diseases_name ON diseases(name)')
        cursor.execute('CREATE INDEX idx_synonyms_doid ON synonyms(doid)')
        cursor.execute('CREATE INDEX idx_synonyms_text ON synonyms(synonym)')
        cursor.execute('CREATE INDEX idx_xrefs_doid ON xrefs(doid)')
        cursor.execute('CREATE INDEX idx_xrefs_source ON xrefs(source)')
        cursor.execute('CREATE INDEX idx_search_normalized ON disease_search(normalized_term)')
        cursor.execute('CREATE INDEX idx_search_doid ON disease_search(doid)')
        
        # Add metadata table
        cursor.execute('''
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        cursor.execute('''
            INSERT INTO metadata (key, value) VALUES
            ('version', 'Disease Ontology Database v1.0'),
            ('source', 'https://disease-ontology.org/'),
            ('created_date', ?),
            ('total_diseases', ?),
            ('total_synonyms', ?),
            ('total_xrefs', ?)
        ''', (
            created_date,
            len(self.diseases),
            sum(len(d.get('synonyms', [])) for d in self.diseases.values()),
            sum(len(d.get('xrefs', [])) for d in self.diseases.values())
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Database created successfully at {self.db_path}")
    
    def print_statistics(self):
        """Print database statistics"""
        print("\n📊 Disease Ontology Database Statistics:")
        print("=" * 50)
        print(f"Total diseases: {len(self.diseases):,}")
        print(f"Total name mappings: {len(self.name_to_doid):,}")
        print(f"Obsolete diseases: {self.stats['obsolete']:,}")
        
        # Sample diseases
        print("\n🔍 Sample diseases:")
        for i, (doid, info) in enumerate(list(self.diseases.items())[:5]):
            print(f"  - {info['name']} ({doid})")
            if info.get('synonyms'):
                print(f"    Synonyms: {', '.join(info['synonyms'][:3])}")
    
    def run(self):
        """Run the complete pipeline"""
        print("\n🚀 Disease Ontology Database Builder")
        print("=" * 50)
        
        # Download if needed
        if not self.obo_path.exists():
            if not self.download_ontology():
                return False
        else:
            logger.info(f"Using existing OBO file: {self.obo_path}")
        
        # Parse OBO file
        self.parse_obo_file()
        
        # Create database
        self.create_database()
        
        # Show statistics
        self.print_statistics()
        
        print(f"\n✅ Complete! Database created at: {self.db_path}")
        print("\n📝 Next steps:")
        print("1. Update your configuration to include this database")
        print("2. Modify your disease detector to use both Orphanet and DO")
        print("3. Test the enhanced detection on your documents")
        
        return True


if __name__ == "__main__":
    builder = DiseaseOntologyBuilder()
    builder.run()