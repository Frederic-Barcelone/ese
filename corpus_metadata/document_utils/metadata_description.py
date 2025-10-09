#!/usr/bin/env python3
"""
Document Description Generator Module
======================================
Location: corpus_metadata/document_utils/metadata_description.py
Version: 3.0.0

Purpose:
    Generate descriptions for documents using LLM or fallback methods.
    Extract dates and create titles from document content.

Changes in 3.0.0:
    - Moved to document_utils subdirectory
    - Removed example usage
    - Graceful API handling with automatic fallback
"""

import re
import os
import logging
from pathlib import Path

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Use unified logging
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    desc_logger = get_logger("metadata.description")
except ImportError:
    log_dir = Path("./corpus_logs")
    log_dir.mkdir(exist_ok=True)
    desc_logger = logging.getLogger("metadata.description")
    if not desc_logger.handlers:
        file_handler = logging.FileHandler(log_dir / "corpus.log", mode='a')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        desc_logger.addHandler(file_handler)
        desc_logger.setLevel(logging.INFO)


class DescriptionExtractor:
    """Generate short and long descriptions for documents and extract relevant dates"""

    def __init__(self, api_key=None, silent=True, model=None, client=None):
        """
        Initialize the description generator
        
        Args:
            api_key: Anthropic API key (optional, uses env variable)
            silent: If True, suppress logging output
            model: Model name to use
            client: Pre-initialized Anthropic client (optional)
        """
        self.silent = silent
        self.client = None
        
        if model is None:
            model = "claude-3-5-sonnet-20241022"
            self._log("Using default model: " + model)
        self.model = model

        if client:
            self.client = client
            self._log("Using provided Claude client")
            return

        if not CLAUDE_AVAILABLE:
            self._log("Anthropic module not available - using fallback descriptions")
            return

        if api_key is None:
            api_key = os.environ.get("CLAUDE_API_KEY")
            if not api_key:
                self._log("No API key available - using fallback descriptions")
                return

        try:
            self.client = Anthropic(api_key=api_key)
            
            try:
                self.client.messages.create(
                    model=self.model,
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Test"}]
                )
                self._log("Claude API connection successful")
            except Exception as test_error:
                error_str = str(test_error)
                if "401" in error_str or "Unauthorized" in error_str:
                    self._log("API key invalid (401 Unauthorized) - using fallback descriptions")
                    self.client = None
                elif "404" in error_str:
                    self._log(f"Model {self.model} not found - using fallback descriptions")
                    self.client = None
                else:
                    self._log(f"API test failed ({error_str}) - using fallback descriptions")
                    self.client = None
                    
        except Exception as e:
            self._log(f"Could not initialize Claude client: {e} - using fallback descriptions")
            self.client = None

    def _log(self, message):
        """Internal logging method"""
        if not self.silent:
            desc_logger.info(message)

    def _clean_title(self, title):
        """Clean up title text by removing common artifacts"""
        if not title:
            return ""
        
        cleaned = title.strip()
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)
        
        prefixes = [
            "Here's my analysis:", "Here is my analysis:", "Based on my review", 
            "TITLE:", "Title:", "My analysis:", "Analysis:", "Summary:",
            "Here's a", "Here is a", "This is a", "This document is"
        ]
        
        for p in prefixes:
            if cleaned.lower().startswith(p.lower()):
                cleaned = cleaned[len(p):].strip()
        
        cleaned = cleaned.strip('"\'')
        cleaned = re.sub(r'^[\.\…\s]+|[\.\…\s]+$', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if cleaned and cleaned.islower():
            cleaned = cleaned.capitalize()
        
        return cleaned

    def generate_descriptions(self, text, filename=None, doc_classification=None):
        """
        Generate descriptions, preferring LLM with fallback
        
        Args:
            text: Document text content
            filename: Original filename (optional)
            doc_classification: Document classification info (optional)
            
        Returns:
            Dict with title, short_description, long_description, and dates
        """
        if not text or len(text.strip()) < 50:
            return self._fallback_descriptions(text or "", filename, doc_classification)

        if self.client:
            try:
                llm_desc = self._generate_content_with_llm(text, filename, doc_classification)
                
                if not llm_desc.get('title'):
                    fallback = self._fallback_descriptions(text, filename, doc_classification)
                    llm_desc['title'] = fallback['title']
                
                if not llm_desc.get('short_description'):
                    fallback = self._fallback_descriptions(text, filename, doc_classification)
                    llm_desc['short_description'] = fallback['short_description']
                
                return llm_desc
                
            except Exception as e:
                self._log(f"LLM generation failed: {e} - using fallback")
                return self._fallback_descriptions(text, filename, doc_classification)
        else:
            return self._fallback_descriptions(text, filename, doc_classification)

    def _generate_content_with_llm(self, text, filename=None, doc_classification=None):
        """Generate content using LLM"""
        if not self.client:
            raise ValueError("No LLM client available")
        
        truncated_text = text[:8000] if len(text) > 8000 else text
        
        prompt = f"""Analyze this document and provide:
1. A concise title (max 100 chars)
2. A short description (1-2 sentences, max 200 chars)
3. A long description (3-5 sentences, max 500 chars)
4. Extract any dates mentioned (publication date, study dates, etc.)

Document text:
{truncated_text}

{f"Filename: {filename}" if filename else ""}
{f"Classification: {doc_classification}" if doc_classification else ""}

Respond in JSON format:
{{
    "title": "...",
    "short_description": "...",
    "long_description": "...",
    "dates": ["YYYY-MM-DD", ...]
}}
"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            import json
            try:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = self._parse_text_response(response_text)
            except json.JSONDecodeError:
                result = self._parse_text_response(response_text)
            
            if result.get('title'):
                result['title'] = self._clean_title(result['title'])
            
            self._log("Successfully generated LLM descriptions")
            return result
            
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                self._log("API authentication failed during generation")
                self.client = None
            else:
                self._log(f"Error in LLM generation: {e}")
            raise

    def _parse_text_response(self, response_text):
        """Parse non-JSON LLM response"""
        result = {
            'title': '',
            'short_description': '',
            'long_description': '',
            'dates': []
        }
        
        lines = response_text.split('\n')
        for line in lines:
            lower_line = line.lower()
            if 'title:' in lower_line:
                result['title'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'short' in lower_line and 'description' in lower_line:
                result['short_description'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'long' in lower_line and 'description' in lower_line:
                result['long_description'] = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'date' in lower_line:
                date_matches = re.findall(r'\d{4}-\d{2}-\d{2}', line)
                result['dates'].extend(date_matches)
        
        return result

    def _fallback_descriptions(self, text, filename=None, doc_classification=None):
        """Generate fallback descriptions without LLM"""
        text = text.strip() if text else ""
        
        # Generate title
        if filename:
            title = Path(filename).stem.replace('_', ' ').replace('-', ' ')
            title = ' '.join(word.capitalize() if len(word) > 3 else word 
                           for word in title.split())
        elif text:
            lines = text.split('\n')
            first_line = lines[0][:100] if lines else "Document"
            title = self._clean_title(first_line)
            if not title or len(title) < 5:
                title = "Extracted Document"
        else:
            title = "Untitled Document"
        
        # Generate short description
        doc_type = "Document"
        if doc_classification:
            if isinstance(doc_classification, dict):
                doc_type = doc_classification.get('document_type', 'Document')
            else:
                doc_type = str(doc_classification)
        
        if text and len(text) > 50:
            sentences = re.split(r'[.!?]\s+', text[:500])
            if sentences and len(sentences[0]) > 20:
                short_desc = sentences[0][:200].strip()
                if not short_desc.endswith('.'):
                    short_desc += '.'
            else:
                short_desc = f"{doc_type} containing {len(text):,} characters of text"
        else:
            short_desc = f"{doc_type} document{' from ' + Path(filename).name if filename else ''}"
        
        # Generate long description
        if text and len(text) > 100:
            paragraphs = text.split('\n\n')
            first_para = paragraphs[0][:500].strip() if paragraphs else text[:500].strip()
            first_para = re.sub(r'\s+', ' ', first_para)
            
            if len(first_para) > 100:
                long_desc = first_para
                if not long_desc.endswith('.'):
                    sentences = re.split(r'[.!?]\s+', long_desc)
                    if len(sentences) > 1:
                        long_desc = '. '.join(sentences[:-1]) + '.'
                    else:
                        long_desc += '.'
            else:
                long_desc = short_desc + f" The document contains {len(text):,} characters."
        else:
            long_desc = short_desc
        
        # Extract dates
        dates = []
        if text:
            search_text = text[:2000]
            date_patterns = [
                (r'\d{4}-\d{2}-\d{2}', 'ISO'),
                (r'\d{1,2}/\d{1,2}/\d{4}', 'US'),
                (r'\d{1,2}-\d{1,2}-\d{4}', 'Dash'),
                (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}', 'Text'),
            ]
            
            seen_dates = set()
            for pattern, pattern_type in date_patterns:
                matches = re.findall(pattern, search_text, re.IGNORECASE)
                for match in matches[:3]:
                    if match not in seen_dates:
                        dates.append(match)
                        seen_dates.add(match)
                        if len(dates) >= 5:
                            break
                if len(dates) >= 5:
                    break
        
        self._log("Generated fallback descriptions successfully")
        
        return {
            'title': title[:100],
            'short_description': short_desc[:200],
            'long_description': long_desc[:500],
            'dates': dates
        }

    def extract_dates(self, text):
        """Extract dates from text using regex"""
        if not text:
            return []
        
        dates = []
        seen = set()
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
        ]
        
        search_text = text[:5000] if len(text) > 5000 else text
        
        for pattern in date_patterns:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            for date in matches:
                if date not in seen:
                    seen.add(date)
                    dates.append(date)
                    if len(dates) >= 10:
                        return dates
        
        return dates