#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/rare_disease_renamer.py
#

"""
Rare Disease Renamer
===================

Intelligent document renaming for rare disease pharmaceutical documents.
Uses Claude AI to generate compliant, descriptive filenames.
"""

import re
import os
import sqlite3
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import logging

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

logger = logging.getLogger(__name__)

class RareDiseaseDocumentRenamer:
    """
    Intelligent document renamer specialized for rare disease pharmaceutical documents
    """
    
    def __init__(self, 
                 claude_api_key: Optional[str] = None,
                 claude_client: Optional[Any] = None,
                 model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize the renamer
        
        Args:
            claude_api_key: Claude API key
            claude_client: Pre-initialized Claude client
            model: Claude model to use
        """
        self.model = model
        self.client = None
        
        # Initialize Claude client
        if claude_client:
            self.client = claude_client
            logger.info("Using provided Claude client")
        elif CLAUDE_AVAILABLE and claude_api_key:
            try:
                self.client = Anthropic(api_key=claude_api_key)
                logger.info("Initialized Claude client")
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
        else:
            logger.warning("No Claude client available - will use fallback naming")
        
        # Medical/Pharmaceutical abbreviations
        self.medical_abbreviations = {
            # Regulatory
            'fda': 'FDA', 'ema': 'EMA', 'pmda': 'PMDA', 'mhra': 'MHRA',
            'ich': 'ICH', 'gcp': 'GCP', 'glp': 'GLP', 'gmp': 'GMP',
            
            # Clinical Trial
            'rct': 'RCT', 'dsmb': 'DSMB', 'idmc': 'IDMC', 'irb': 'IRB',
            'ec': 'EC', 'pi': 'PI', 'cro': 'CRO', 'cra': 'CRA',
            
            # Documents
            'csr': 'CSR', 'sap': 'SAP', 'ib': 'IB', 'icf': 'ICF',
            'crf': 'CRF', 'tmf': 'TMF', 'ctd': 'CTD', 'psur': 'PSUR',
            
            # Medical
            'ae': 'AE', 'sae': 'SAE', 'adr': 'ADR', 'susar': 'SUSAR',
            'teae': 'TEAE', 'moa': 'MOA', 'pk': 'PK', 'pd': 'PD',
            
            # Rare Disease Specific
            'ord': 'ORD', 'comp': 'COMP', 'pdco': 'PDCO', 'prime': 'PRIME',
            'rmat': 'RMAT', 'btd': 'BTD', 'fdd': 'FDD', 'rpd': 'RPD',
            
            # Other
            'hta': 'HTA', 'rwe': 'RWE', 'pro': 'PRO', 'qol': 'QoL',
            'heor': 'HEOR', 'icer': 'ICER', 'qaly': 'QALY'
        }
        
        # Document type mappings for intelligent naming
        self.doc_type_prefixes = {
            'PRO': 'Protocol',
            'PCS': 'Protocol Concept',
            'AME': 'Amendment',
            'IBR': 'Investigator Brochure',
            'ICF': 'Informed Consent',
            'CRF': 'CRF',
            'SAP': 'SAP',
            'CSR': 'Clinical Study Report',
            'SAE': 'Safety Report',
            'ADV': 'Advisory Board',
            'MOA': 'MOA',
            'EFF': 'Efficacy Analysis',
            'PUB': 'Publication',
            'REG': 'Regulatory Submission',
            'CI': 'CI Update',
            'SIT': 'Site Qualification',
            'TMP': 'Management Plan',
            'RCE': 'Recruitment',
            'TRA': 'Training',
            'GOV': 'Governance',
            'DSC': 'Disease Landscape',
            'HTA': 'HTA',
            'PJM': 'Patient Journey',
            'RWE': 'RWE',
            'NH': 'Natural History',
            'REG': 'Registry'
        }
    
    def process_file(self, filepath: str, document_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single file for renaming
        
        Args:
            filepath: Path to the file to rename
            document_data: Optional document metadata including:
                - short_desc: Short description of document
                - doc_type: Document classification type
                - doc_date: Document date
                - content: Document content (for analysis)
                
        Returns:
            Dictionary with results:
                - success: Whether renaming was successful
                - original_path: Original file path
                - new_path: New file path (if renamed)
                - original_name: Original filename
                - new_name: New filename (if renamed)
                - error: Error message (if failed)
        """
        file_path = Path(filepath)
        
        # Validate file exists
        if not file_path.exists():
            return {
                'success': False,
                'error': f"File not found: {filepath}",
                'original_path': str(file_path),
                'original_name': file_path.name
            }
        
        # Extract index from filename
        index = self._extract_index(file_path.name)
        if index is None:
            return {
                'success': False,
                'error': f"No index found in filename: {file_path.name}",
                'original_path': str(file_path),
                'original_name': file_path.name
            }
        
        # Prepare document data
        if document_data is None:
            document_data = {}
        
        # Generate new filename
        try:
            new_base_name = self._generate_filename(
                short_desc=document_data.get('short_desc', ''),
                doc_type=document_data.get('doc_type', 'UNKNOWN'),
                doc_date=document_data.get('doc_date', ''),
                content=document_data.get('content', ''),
                original_filename=file_path.name
            )
            
            # Construct new filename with preserved index and extension
            new_filename = f"{index:05d}_{new_base_name}{file_path.suffix}"
            new_path = file_path.parent / new_filename
            
            # Handle conflicts
            if new_path.exists() and new_path != file_path:
                counter = 1
                while new_path.exists():
                    conflict_name = f"{index:05d}_{new_base_name}_{counter}{file_path.suffix}"
                    new_path = file_path.parent / conflict_name
                    counter += 1
                new_filename = new_path.name
            
            # Check if renaming is needed
            if new_path == file_path:
                return {
                    'success': True,
                    'no_change': True,
                    'original_path': str(file_path),
                    'original_name': file_path.name,
                    'message': 'No renaming needed - filename already optimal'
                }
            
            # Perform rename
            try:
                shutil.move(str(file_path), str(new_path))
                logger.info(f"Renamed: {file_path.name} â†’ {new_filename}")
                
                return {
                    'success': True,
                    'original_path': str(file_path),
                    'new_path': str(new_path),
                    'original_name': file_path.name,
                    'new_name': new_filename,
                    'base_name': new_base_name
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Failed to rename file: {str(e)}",
                    'original_path': str(file_path),
                    'original_name': file_path.name,
                    'suggested_name': new_filename
                }
                
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            return {
                'success': False,
                'error': f"Error generating filename: {str(e)}",
                'original_path': str(file_path),
                'original_name': file_path.name
            }
    
    def _extract_index(self, filename: str) -> Optional[int]:
        """Extract numeric index from filename"""
        match = re.match(r'^(\d{5})[_-]', filename)
        if match:
            return int(match.group(1))
        return None
    
    def _generate_filename(self, short_desc: str, doc_type: str, 
                          doc_date: str, content: str, original_filename: str) -> str:
        """Generate intelligent filename using Claude or fallback"""
        
        # Use Claude if available
        if self.client and short_desc:
            try:
                filename = self._generate_claude_filename(
                    short_desc, doc_type, doc_date, content, original_filename
                )
                if filename:
                    return self._sanitize_filename(filename)
            except Exception as e:
                logger.warning(f"Claude generation failed: {e}")
        
        # Fallback to rule-based generation
        return self._generate_fallback_filename(
            short_desc, doc_type, doc_date, content, original_filename
        )
    
    def _generate_claude_filename(self, short_desc: str, doc_type: str,
                                 doc_date: str, content: str, original_filename: str) -> str:
        """Generate filename using Claude AI"""
        
        prompt = f"""You are a medical documentation specialist for rare disease pharmaceuticals.
Generate a precise, compliant filename for this document.

DOCUMENT DESCRIPTION:
{short_desc}

METADATA:
- Document Type: {doc_type}
- Date: {doc_date}
- Original Name: {original_filename}

REQUIREMENTS:
1. Use 3-10 words maximum
2. Include key identifiers (study ID, drug name, disease if mentioned)
3. Follow pharmaceutical naming conventions
4. Use proper medical abbreviations (FDA, EMA, SAE, etc.)
5. Include document version/date indicators when relevant
6. Use Title Case with spaces between words
7. Maximum 150 characters

RARE DISEASE SPECIFIC:
- Include orphan drug designation indicators if applicable
- Highlight if it's a natural history study, registry, or expanded access
- Include patient population identifiers if mentioned

Generate ONLY the filename, no explanations:"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        filename = response.content[0].text.strip()
        
        # Clean response
        filename = filename.strip('"\'')
        filename = re.sub(r'^[Ff]ilename:\s*', '', filename)
        filename = filename.split('\n')[0]
        
        return filename
    
    def _generate_fallback_filename(self, short_desc: str, doc_type: str,
                                   doc_date: str, content: str, original_filename: str) -> str:
        """Generate filename using rules when Claude is unavailable"""
        
        parts = []
        
        # Add document type prefix
        if doc_type in self.doc_type_prefixes:
            parts.append(self.doc_type_prefixes[doc_type])
        elif doc_type and doc_type != 'UNKNOWN':
            parts.append(doc_type.replace('_', ' ').title())
        
        # Extract key information from short description
        if short_desc:
            # Look for drug/compound names
            drug_match = re.search(r'(?:ALXN|ACT|SOBI|BIO|BMRN|VTXR|RARE)-?\d+', 
                                 short_desc, re.IGNORECASE)
            if drug_match:
                parts.append(drug_match.group(0).upper())
            
            # Look for disease mentions
            disease_keywords = {
                'hemophilia': 'Hemophilia',
                'pompe': 'Pompe',
                'gaucher': 'Gaucher',
                'fabry': 'Fabry',
                'hunter': 'Hunter',
                'duchenne': 'Duchenne',
                'sma': 'SMA',
                'pnh': 'PNH',
                'ahus': 'aHUS',
                'myasthenia': 'Myasthenia Gravis'
            }
            
            desc_lower = short_desc.lower()
            for keyword, display_name in disease_keywords.items():
                if keyword in desc_lower:
                    if display_name not in parts:
                        parts.append(display_name)
                    break
            
            # Extract phase information
            phase_match = re.search(r'phase\s*([IVX123]{1,3})', short_desc, re.IGNORECASE)
            if phase_match:
                parts.append(f"Phase {phase_match.group(1).upper()}")
        
        # Add date if available and not already in parts
        if doc_date:
            try:
                date_obj = datetime.strptime(doc_date[:10], '%Y-%m-%d')
                date_str = date_obj.strftime('%Y%m')
                if date_str not in ' '.join(parts):
                    parts.append(date_str)
            except:
                pass
        
        # If we have no meaningful parts, use cleaned original name
        if not parts:
            base_name = Path(original_filename).stem
            base_name = re.sub(r'^\d{5}[_-]', '', base_name)
            base_name = base_name.replace('_', ' ').replace('-', ' ')
            parts = [base_name]
        
        filename = ' '.join(parts[:6])  # Limit number of parts
        return self._sanitize_filename(filename)
    
    def _sanitize_filename(self, filename: str, max_length: int = 150) -> str:
        """Sanitize filename for cross-platform compatibility"""
        
        # Remove invalid characters
        invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
        clean_name = re.sub(invalid_chars, '', filename)
        
        # Normalize whitespace
        clean_name = re.sub(r'[\s_-]+', ' ', clean_name)
        clean_name = clean_name.strip()
        
        # Apply medical abbreviations
        words = clean_name.split()
        normalized_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.medical_abbreviations:
                normalized_words.append(self.medical_abbreviations[word_lower])
            else:
                normalized_words.append(word)
        clean_name = ' '.join(normalized_words)
        
        # Handle Windows reserved words
        reserved_words = {
            'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
            'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3',
            'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
        }
        if clean_name.lower() in reserved_words:
            clean_name = f"{clean_name}_doc"
        
        # Truncate if needed
        if len(clean_name) > max_length:
            clean_name = clean_name[:max_length].strip()
        
        # Remove trailing periods
        clean_name = clean_name.rstrip('.')
        
        # Default if empty
        if not clean_name:
            clean_name = "Rare Disease Document"
        
        return clean_name
    
    def preview_rename(self, filepath: str, document_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Preview what the rename would be without actually renaming
        
        Args:
            filepath: Path to the file
            document_data: Optional document metadata
            
        Returns:
            Dictionary with preview information
        """
        file_path = Path(filepath)
        
        if not file_path.exists():
            return {
                'success': False,
                'error': f"File not found: {filepath}",
                'original_name': file_path.name
            }
        
        index = self._extract_index(file_path.name)
        if index is None:
            return {
                'success': False,
                'error': f"No index found in filename",
                'original_name': file_path.name
            }
        
        if document_data is None:
            document_data = {}
        
        try:
            new_base_name = self._generate_filename(
                short_desc=document_data.get('short_desc', ''),
                doc_type=document_data.get('doc_type', 'UNKNOWN'),
                doc_date=document_data.get('doc_date', ''),
                content=document_data.get('content', ''),
                original_filename=file_path.name
            )
            
            new_filename = f"{index:05d}_{new_base_name}{file_path.suffix}"
            
            return {
                'success': True,
                'original_name': file_path.name,
                'suggested_name': new_filename,
                'base_name': new_base_name,
                'index_preserved': f"{index:05d}",
                'extension_preserved': file_path.suffix,
                'would_change': new_filename != file_path.name
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error generating preview: {str(e)}",
                'original_name': file_path.name
            }


# Example usage
if __name__ == "__main__":
    # This is just for testing - normally this would be imported
    import os
    
    # Initialize renamer
    api_key = os.getenv('CLAUDE_API_KEY')
    renamer = RareDiseaseDocumentRenamer(claude_api_key=api_key)
    
    # Example document data
    doc_data = {
        'short_desc': 'Phase III clinical trial protocol for ALXN1234 in patients with PNH',
        'doc_type': 'PRO',
        'doc_date': '2024-01-15',
        'content': 'This protocol describes the phase III study of ALXN1234...'
    }
    
    # Preview rename
    preview = renamer.preview_rename('./00001_document.pdf', doc_data)
    print("Preview:", preview)
    
    # Actual rename
    # result = renamer.process_file('./00001_document.pdf', doc_data)
    # print("Result:", result)