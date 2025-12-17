#!/usr/bin/env python3
"""
Document Intelligent Renamer - Core Module
==========================================
Location: corpus_metadata/document_intelligent_renamer.py

Provides intelligent filename generation for documents using AI and rule-based approaches.
Integrated with the centralized logging system.
"""

import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import centralized logging
from corpus_metadata.document_utils.metadata_logging_config import get_logger

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

class IntelligentDocumentRenamer:
    """
    Generate intelligent filenames for documents based on content and classification.
    Preserves index numbers and file extensions.
    """
    
    def __init__(self):
        """Initialize the intelligent document renamer"""
        
        # Use centralized logging system
        self.logger = get_logger('intelligent_renamer')
        
        # Model configuration
        self.model = "claude-sonnet-4-5-20250929"
        
        # Initialize Claude client
        self._initialize_claude_client()
        
        # Clinical trial specific reserved terms and abbreviations
        self.clinical_abbreviations = {
            'moa': 'MOA', 'fda': 'FDA', 'ema': 'EMA', 'ich': 'ICH',
            'gcp': 'GCP', 'crf': 'CRF', 'sae': 'SAE', 'ae': 'AE',
            'ctd': 'CTD', 'nda': 'NDA', 'bla': 'BLA', 'ind': 'IND',
            'cta': 'CTA', 'dsmb': 'DSMB', 'idmc': 'IDMC', 'irb': 'IRB',
            'ec': 'EC', 'pi': 'PI', 'cro': 'CRO', 'cra': 'CRA',
            'tm': 'TM', 'qc': 'QC', 'qa': 'QA'
        }
        
        self.invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
        
    def _initialize_claude_client(self):
        """Initialize Claude client from environment variable"""
        if not CLAUDE_AVAILABLE:
            self.logger.warning("Anthropic module not available. AI-based renaming disabled.")
            self.client = None
            return
            
        try:
            env_key = os.environ.get("CLAUDE_API_KEY")
            if env_key:
                self.client = Anthropic(api_key=env_key)
                self.logger.debug("Claude client initialized from environment")
            else:
                self.logger.warning("No CLAUDE_API_KEY found. AI-based renaming disabled.")
                self.client = None
                
        except Exception as e:
            self.logger.error(f"Error initializing Claude client: {e}")
            self.client = None
    
    def propose_filename(self, text_content: str, original_filename: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose a new filename based on document content and context
        
        Args:
            text_content: Document text content (first 5000 chars typically)
            original_filename: Original filename
            context: Dictionary containing document_type, title, short_description, document_date
            
        Returns:
            Dictionary with proposed_filename, alternatives, reasoning, confidence
        """
        try:
            # Extract document information from context
            doc_type = context.get('document_type', 'UNKNOWN')
            title = context.get('title', '')
            short_description = context.get('short_description', '')
            document_date = context.get('document_date', '')
            
            # Clean the short description
            if short_description:
                short_description = short_description.strip('"').strip(',').strip()
            
            self.logger.debug(f"Generating filename proposal for {original_filename} (type: {doc_type})")
            
            # Generate intelligent filename
            if short_description:
                proposed_name = self.generate_intelligent_filename(
                    short_description=short_description,
                    original_filename=original_filename,
                    doc_type=doc_type,
                    doc_date=document_date
                )
            else:
                proposed_name = self._generate_fallback_filename(
                    short_description=title or "",
                    original_filename=original_filename,
                    doc_type=doc_type
                )
            
            # Extract components from original filename
            index = self.extract_index_from_filename(original_filename)
            extension = self.extract_extension_from_filename(original_filename)
            
            # Construct full proposed filename
            if index is not None:
                full_proposed = f"{index:05d}_{proposed_name}{extension}"
            else:
                full_proposed = f"{proposed_name}{extension}"
            
            # Generate alternatives
            alternatives = self._generate_alternatives(
                proposed_name, doc_type, document_date, index, extension
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(short_description, doc_type, document_date)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                short_description, doc_type, document_date, proposed_name
            )
            
            result = {
                'proposed_filename': full_proposed,
                'alternatives': alternatives,
                'reasoning': reasoning,
                'confidence': confidence
            }
            
            self.logger.info(f"Proposed filename: {full_proposed} (confidence: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error proposing filename: {e}")
            return {
                'proposed_filename': original_filename,
                'alternatives': [],
                'reasoning': f"Error during proposal: {str(e)}",
                'confidence': 0.0
            }
    
    def extract_index_from_filename(self, filename: str) -> Optional[int]:
        """Extract the numeric index from a filename like '00096_document.pdf'"""
        match = re.match(r'^(\d{5})[_-]', filename)
        if match:
            return int(match.group(1))
        return None
    
    def extract_extension_from_filename(self, filename: str) -> str:
        """Extract file extension"""
        return Path(filename).suffix
    
    def sanitize_filename(self, filename: str, max_length: int = 250) -> str:
        """Sanitize filename for cross-platform compatibility"""
        # Remove invalid characters
        clean_name = re.sub(self.invalid_chars, '', filename)
        
        # Replace multiple spaces/underscores/hyphens with single space
        clean_name = re.sub(r'[\s_-]+', ' ', clean_name)
        
        # Trim whitespace
        clean_name = clean_name.strip()
        
        # Handle clinical abbreviations
        words = clean_name.split()
        normalized_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.clinical_abbreviations:
                normalized_words.append(self.clinical_abbreviations[word_lower])
            else:
                normalized_words.append(word)
        clean_name = ' '.join(normalized_words)
        
        # Handle Windows reserved words
        reserved_words = {
            'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 'com5',
            'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2', 'lpt3', 'lpt4',
            'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
        }
        if clean_name.lower() in reserved_words:
            clean_name = f"{clean_name}_doc"
        
        # Truncate if too long
        if len(clean_name) > max_length:
            clean_name = clean_name[:max_length].strip()
        
        # Ensure it doesn't end with a period
        clean_name = clean_name.rstrip('.')
        
        # If empty after cleaning, use default
        if not clean_name:
            clean_name = "clinical document"
            
        return clean_name
    
    def generate_intelligent_filename(self, short_description: str, original_filename: str, 
                                    doc_type: str = "", doc_date: str = "") -> str:
        """Generate filename using AI if available, otherwise use fallback"""
        if not self.client:
            return self._generate_fallback_filename(short_description, original_filename, doc_type)
        
        try:
            # Prepare context
            context_info = f"Original filename: {original_filename}\n"
            context_info += f"Document type: {doc_type}\n"
            if doc_date:
                context_info += f"Document date: {doc_date}\n"
            
            # Create prompt
            prompt = self._create_specialized_prompt(doc_type, short_description, context_info)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            suggested_name = response.content[0].text.strip()
            
            # Clean the response
            suggested_name = suggested_name.strip('"\'')
            suggested_name = re.sub(r'^[Ff]ilename:\s*', '', suggested_name)
            suggested_name = suggested_name.split('\n')[0]
            
            # Sanitize
            clean_name = self.sanitize_filename(suggested_name)
            
            self.logger.debug(f"AI suggested: '{suggested_name}' -> cleaned: '{clean_name}'")
            
            return clean_name
            
        except Exception as e:
            self.logger.warning(f"AI generation failed, using fallback: {e}")
            return self._generate_fallback_filename(short_description, original_filename, doc_type)
    
    def _create_specialized_prompt(self, doc_type: str, short_description: str, context_info: str) -> str:
        """Create prompt for AI filename generation"""
        
        base_requirements = """
Generate a precise filename for this clinical document.
Requirements:
- 3-10 words maximum  
- Use proper capitalization (Title Case)
- Use spaces between words
- Focus on key identifying information
- Do not include file extensions or index numbers

Context:
{context_info}

Description:
{short_description}

Generate ONLY the filename:
"""
        
        # Special handling for publication documents
        if doc_type == "PUB":
            return f"""
Generate a filename for a PUBLICATION document.

Format examples:
- "Pediatric ANCA Vasculitis Treatment Review"
- "Complement Inhibition Therapy Analysis"
- "Clinical Trial Results Summary"

{base_requirements.format(context_info=context_info, short_description=short_description)}
"""
        
        # Generic prompt for other types
        return f"""
Generate a filename for a {doc_type} document.

{base_requirements.format(context_info=context_info, short_description=short_description)}
"""
    
    def _generate_fallback_filename(self, short_description: str, original_filename: str, doc_type: str = "") -> str:
        """Generate filename without AI"""
        
        if short_description and len(short_description.strip()) > 0:
            short_description = short_description.strip('"').strip(',').strip()
            desc_words = short_description.split()[:12]
            fallback_name = ' '.join(desc_words)
        else:
            fallback_name = Path(original_filename).stem
            fallback_name = re.sub(r'^\d{5}[_-]', '', fallback_name)
            fallback_name = fallback_name.replace('_', ' ').replace('-', ' ')
        
        # Add document type prefix if appropriate
        if doc_type and doc_type != "UNKNOWN":
            doc_type_mapping = {
                "PUB": "Publication",
                "PRO": "Protocol",
                "SAE": "Safety Report",
                "CSR": "Clinical Study Report",
                "REG": "Regulatory",
                "CI": "Competitive Intelligence"
            }
            
            prefix = doc_type_mapping.get(doc_type, doc_type)
            if prefix.lower() not in fallback_name.lower():
                fallback_name = f"{prefix} {fallback_name}"
        
        return self.sanitize_filename(fallback_name)
    
    def _generate_alternatives(self, base_name: str, doc_type: str, doc_date: str, 
                              index: Optional[int], extension: str) -> List[str]:
        """Generate alternative filename suggestions"""
        alternatives = []
        
        try:
            # Alternative with date prefix
            if doc_date:
                date_obj = datetime.fromisoformat(doc_date)
                date_prefix = date_obj.strftime("%Y_%m")
                alt1 = f"{date_prefix}_{doc_type}_{base_name}"
                if index is not None:
                    alt1 = f"{index:05d}_{alt1}{extension}"
                else:
                    alt1 = f"{alt1}{extension}"
                alternatives.append(alt1)
            
            # Shorter version
            words = base_name.split()[:5]
            short_name = ' '.join(words)
            if index is not None:
                alt2 = f"{index:05d}_{short_name}{extension}"
            else:
                alt2 = f"{short_name}{extension}"
            alternatives.append(alt2)
            
            # Type-focused version
            if doc_type and doc_type != "UNKNOWN":
                type_focused = f"{doc_type} {' '.join(base_name.split()[:3])}"
                if index is not None:
                    alt3 = f"{index:05d}_{type_focused}{extension}"
                else:
                    alt3 = f"{type_focused}{extension}"
                alternatives.append(alt3)
            
        except Exception as e:
            self.logger.debug(f"Error generating alternatives: {e}")
        
        # Remove duplicates
        seen = set()
        unique_alts = []
        for alt in alternatives:
            if alt not in seen:
                seen.add(alt)
                unique_alts.append(alt)
        
        return unique_alts[:3]
    
    def _calculate_confidence(self, description: str, doc_type: str, doc_date: str) -> float:
        """Calculate confidence score"""
        confidence = 0.0
        
        if description and len(description.strip()) > 20:
            confidence += 0.5
        
        if doc_type and doc_type != "UNKNOWN":
            confidence += 0.3
        
        if doc_date:
            confidence += 0.1
        
        if self.client:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, description: str, doc_type: str, 
                           doc_date: str, proposed_name: str) -> str:
        """Generate reasoning for the proposed filename"""
        reasons = []
        
        if description:
            reasons.append("Based on document description")
        
        if doc_type and doc_type != "UNKNOWN":
            reasons.append(f"Document type: {doc_type}")
        
        if doc_date:
            reasons.append(f"Document dated {doc_date}")
        
        if self.client:
            reasons.append("Enhanced with AI analysis")
        else:
            reasons.append("Rule-based generation")
        
        return "; ".join(reasons) if reasons else "Default naming applied"