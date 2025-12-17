#!/usr/bin/env python3
"""
Document Metadata Classifier - WITH CENTRALIZED LOGGING
========================================================
/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/metadata_classifier.py

Purpose:
--------
Classify documents using a comprehensive approach combining:
1. LLM-based classification with JSON schema context (highest priority)
2. Long-term description analysis
3. Filename pattern recognition
4. Short description analysis
5. Content pattern recognition
6. Entity extraction
7. Fuzzy matching against reference terms

NOW USES CENTRALIZED LOGGING SYSTEM - all logs go to corpus.log file

FIXED: Removed nonsense CorpusConfig import - configuration comes from YAML file
"""

import json
import re
import os
import time
import traceback
from anthropic import Anthropic
from rapidfuzz import fuzz
from collections import Counter
from datetime import datetime
from typing import Dict, Any, Optional, List

# ============================================================================
# USE CENTRALIZED LOGGING SYSTEM
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import (
        get_logger,
        log_separator,
        log_metric
    )
    logger = get_logger('DocumentClassifier')
    LOGGING_AVAILABLE = True
except ImportError:
    # Fallback if centralized logging not available
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Keep console quiet
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('DocumentClassifier')
    LOGGING_AVAILABLE = False
    
    def log_separator(logger, style='minor'):
        pass
    
    def log_metric(logger, name, value, unit=''):
        logger.debug(f"{name}: {value}{unit}")



def load_document_types(filename='document_types.json', json_path=None, silent=False):
    """Load document type definitions from JSON file"""
    
    # Use the specific path for document_types.json
    filepath = "/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_dictionaries/output_datasources/2025_08_document_types.json"
    
    if not os.path.exists(filepath):
        logger.error(f"Document types file not found: {filepath}")
        return []
    
    if not silent:
        logger.info(f"Loading document types from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not silent:
                logger.info(f"Loaded document types: {len(data)} groups")
            return data
    except FileNotFoundError:
        logger.error(f"Document types file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error loading document types: {e}")
        return []
    


def safe_json_loads(text):
    """Safely parse JSON from text, with fallback strategies"""
    # First, strip markdown code fences if present (```json ... ``` or ``` ... ```)
    cleaned = text.strip()
    if cleaned.startswith('```'):
        # Remove opening fence (with optional language identifier)
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        # Remove closing fence
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON, attempting to extract from text: {text[:100]}...")
        match = re.search(r'({.*?})', text, re.DOTALL)
        if match:
            try:
                extracted = match.group(1)
                logger.info(f"Extracted JSON-like content: {extracted[:100]}...")
                return json.loads(extracted)
            except json.JSONDecodeError as e:
                logger.error(f"Still failed to parse JSON after extraction: {e}")
                return None
        # Try another pattern with looser matching
        match = re.search(r'"doc_type"\s*:\s*"([^"]+)".*?"confidence"\s*:\s*"([^"]+)"', text, re.DOTALL)
        if match:
            return {
                "doc_type": match.group(1),
                "confidence": match.group(2),
                "suggested_doc_type": ""
            }
    return None


def call_claude_with_retry(client, prompt, retries=3, backoff=2, max_tokens=500, silent=False):
    """Call Claude API with retry mechanism"""
    for attempt in range(1, retries + 1):
        try:
            if not silent:
                logger.info(f"Calling Claude (attempt {attempt}/{retries})")
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            if not silent:
                logger.info(f"Claude responded with {len(response.content[0].text)} characters")
            return response
        except Exception as e:
            if not silent:
                logger.error(f"Claude API error (attempt {attempt}/{retries}): {str(e)}")
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
    return None


def normalize_label(label):
    """Strip punctuation/whitespace and uppercase for consistent matching"""
    if not label:
        return ""
    return re.sub(r'[^A-Z0-9]', '', label.upper())


def extract_first_doc_type(text):
    """Try to extract doc_type using various patterns when JSON parsing fails"""
    patterns = [
        r'"doc_type"\s*:\s*"([^"]+)"',
        r'doc_type\s*:\s*"([^"]+)"',
        r'doc_type\s*:\s*(\w+)',
        r'type[: ]+(\w+)',
        r'classify.*?as.*?(\w+)',
        r'document.*?(?:is|as).*?(\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def extract_keywords_from_text(text, max_keywords=20):
    """Extract key terms from document content"""
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = re.findall(r'\b\w{3,}\b', cleaned_text)
    
    stopwords = {
        'the', 'and', 'for', 'was', 'with', 'are', 'that', 'this', 'were', 'have',
        'from', 'not', 'but', 'they', 'what', 'which', 'when', 'who', 'will', 'more',
        'also', 'all', 'has', 'can', 'been', 'than', 'their', 'its', 'may', 'these'
    }
    filtered_words = [w for w in words if w not in stopwords]
    
    counter = Counter(filtered_words)
    return [word for word, count in counter.most_common(max_keywords)]


def extract_entities(text, entity_types):
    """Extract specific entity mentions from text"""
    entities = {}
    for entity_type, patterns in entity_types.items():
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        if matches:
            entities[entity_type] = list(set(matches))
    return entities


class DocumentClassifier:
    """
    Classify documents using a comprehensive approach combining multiple techniques.
    Uses centralized logging system for all output.
    """
    
    def __init__(self, api_key=None, client=None, json_path=None, silent=False, **kwargs):
        """
        Initialize the document classifier
        
        Args:
            api_key: Claude API key
            client: Anthropic client instance  
            json_path: Path to JSON files
            silent: If True, suppress info logging
            **kwargs: Catch any additional arguments
        """
        self.client = client or Anthropic(api_key=api_key or os.getenv('CLAUDE_API_KEY'))
        self.silent = silent
        
        # Load document types from JSON
        self.document_types = load_document_types(
            json_path=json_path,
            silent=silent
        )
        
        if not silent:
            logger.info(f"Loaded {len(self.document_types)} document type groups")
        
        # Build lookups
        self.type_lookup = {}
        self.alias_lookup = {}
        self.all_aliases = {}
        self.doc_type_patterns = {}
        self.confidence_boosters = {}
        
        for group in self.document_types:
            for t in group.get('types', []):
                code = normalize_label(t['code'])
                self.type_lookup[code] = {
                    'id': t.get('id'),
                    'name': t.get('name'),
                    'desc': t.get('desc'),
                    'group': group.get('group')
                }
                
                # Store all aliases for this code
                self.all_aliases[code] = [normalize_label(alias) for alias in t.get('aliases', [])]
                
                # Register aliases for lookup
                for alias in t.get('aliases', []):
                    norm = normalize_label(alias)
                    self.alias_lookup[norm] = code
                
                # Store patterns for this code
                self.doc_type_patterns[code] = t.get('patterns', [])
                
                # Store confidence boosters
                self.confidence_boosters[code] = t.get('confidence_boosters', {})
        
        # Log stats only if not silent
        if not silent:
            logger.info(f"Registered {len(self.type_lookup)} primary document types")
            logger.info(f"Registered {len(self.alias_lookup)} document type aliases")
            logger.info(f"Registered {sum(len(p) for p in self.doc_type_patterns.values())} pattern rules")
            logger.debug(f"Primary codes: {', '.join(sorted(self.type_lookup.keys()))}")
        
        # Combined set of all labels
        self.valid_labels = set(self.type_lookup.keys()) | set(self.alias_lookup.keys())
        
        # Matching thresholds
        self.fuzzy_threshold = 65
        self.high_threshold = 85
        self.medium_threshold = 75
        
        # Entity patterns for domain-specific extraction
        self.entity_patterns = {
            'disease': [
                r'\b(?:ANCA|AAV|GPA|MPA|EGPA|vasculitis|hemolysis|PNH|aHUS|gMG|NMOSD|IgAN|AKI|CKD|DCM|IgA nephropathy|ATTR-CM)\b',
                r'\b(?:renal|kidney|nephropathy|complement|amyloidosis|cardiomyopathy|transplant|delayed graft function|DGF)\b',
                r'\b(?:myasthenia gravis|neuromyelitis optica|paroxysmal nocturnal hemoglobinuria|dilated cardiomyopathy|lupus nephritis)\b'
            ],
            'drug': [
                r'\b(?:ravulizumab|eculizumab|rituximab|ALXN\d+|avacopan|pexelizumab|crovalimab|efgartigimod|zilucoplan)\b',
                r'\b(?:cyclophosphamide|patisiran|VYVGART|UPLIZNA|BENLYSTA|LUPKYNIS|IZERVAY|SYFOVRE|RYSTIGGO|PiaSky)\b'
            ],
            'study_type': [
                r'\b(?:protocol|amendment|trial|phase \d+|phase \d+/\d+|study|RCT|open[-\s]label|double-blind|multicenter)\b',
                r'\b(?:efficacy|safety|interim|outcomes|observational|registry|placebo-controlled|randomized|adaptive design)\b'
            ],
            'document_type': [
                r'\b(?:poster|abstract|presentation|manuscript|protocol|amendment|brochure|weekly update|global update)\b',
                r'\b(?:report|synopsis|correspondence|slides|analysis|submission|CRF|ICF|eCRF|informed consent)\b'
            ],
            'regulatory': [
                r'\b(?:FDA|EMA|CHMP|IND|CTA|EC approval|priority review|orphan|breakthrough)\b',
                r'\b(?:ICH|GCP|Declaration of Helsinki|regulatory|submission|approval|filing)\b'
            ],
            'company': [
                r'\b(?:Alexion|AstraZeneca|Roche|Novartis|UCB|argenx|Alnylam|Apellis|Astellas)\b',
                r'\b(?:Pharmaceuticals|Rare Disease|Division|Subsidiary|Corporation|sponsor)\b'
            ]
        }

    def classify_document(self, doc):
        """
        Main entry point for document classification
        """
        if not self.silent:
            logger.info(f"Classifying document: {doc.get('name', 'Unknown')}")
            logger.debug(f"Type patterns available: {len(self.doc_type_patterns)}")
        
        # 1. LLM-based classification
        llm_result = self._classify_using_llm(doc)
        if llm_result:
            if not self.silent:
                logger.info(f"Classification via LLM: {llm_result['doc_type']} ({llm_result['confidence']})")
            return self._post_process_confidence(llm_result, doc)
        
        # 2. Long-term description
        if 'long_term_desc' in doc or 'long_desc' in doc:
            long_desc = doc.get('long_term_desc', '') or doc.get('long_desc', '')
            if long_desc and len(long_desc) > 10:
                if not self.silent:
                    logger.info(f"Attempting classification via long description ({len(long_desc)} chars)")
                result = self._classify_by_long_term_desc(long_desc)
                if result:
                    if not self.silent:
                        logger.info(f"Classification via long description: {result['doc_type']} ({result['confidence']})")
                    return self._post_process_confidence(result, doc)
        
        # 3. Content patterns
        if 'content' in doc and doc['content']:
            content_sample = doc['content'][:5000]
            filename = doc.get('name', '')
            
            scores = self._score_text_patterns(content_sample, filename)
            ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            logger.debug(f"Top content pattern scores: {ranked_scores[:3]}")
            
            if ranked_scores and ranked_scores[0][1] >= 30:
                best_code = ranked_scores[0][0]
                score_gap = ranked_scores[0][1] - (ranked_scores[1][1] if len(ranked_scores) > 1 else 0)
                
                confidence = 'Low'
                if ranked_scores[0][1] >= 70 and score_gap >= 30:
                    confidence = 'High'
                elif ranked_scores[0][1] >= 50 and score_gap >= 20:
                    confidence = 'Medium'
                
                info = self.type_lookup.get(best_code, {})
                result = {
                    'doc_type': best_code,
                    'doc_id': info.get('id', 0),
                    'doc_name': info.get('name', ''),
                    'group': info.get('group', 'Unknown'),
                    'desc': info.get('desc', ''),
                    'confidence': confidence,
                    'method': 'content_pattern'
                }
                logger.info(f"Classification via content patterns: {result['doc_type']} ({result['confidence']})")
                return self._post_process_confidence(result, doc)
        
        # 4. Short description
        if 'short_desc' in doc and doc['short_desc']:
            logger.info(f"Attempting classification via short description")
            result = self._classify_by_short_desc(doc['short_desc'])
            if result:
                logger.info(f"Classification via short description: {result['doc_type']} ({result['confidence']})")
                return self._post_process_confidence(result, doc)
        
        # 5. Filename patterns
        filename = doc.get('name', '')
        if filename:
            result = self._classify_by_filename(filename)
            if result:
                logger.info(f"Classification via filename pattern: {result['doc_type']} ({result['confidence']})")
                return self._post_process_confidence(result, doc)
        
        # 6. Entity extraction
        if 'content' in doc and doc['content']:
            content_sample = doc['content'][:10000]
            entities = extract_entities(content_sample, self.entity_patterns)
            
            if 'document_type' in entities:
                result = self._classify_by_entities(entities)
                if result:
                    logger.info(f"Classification via entity extraction: {result['doc_type']} ({result['confidence']})")
                    return self._post_process_confidence(result, doc)
        
        # 7. Default to UNKNOWN
        logger.info(f"No classification method succeeded for: {doc.get('name', 'Unknown')}")
        return {
            'doc_type': 'UNKNOWN',
            'doc_id': 0,
            'doc_name': '',
            'group': 'Unknown',
            'desc': 'No classification method succeeded',
            'confidence': 'Low',
            'method': 'default'
        }
    
    def _classify_by_filename(self, filename):
        """Extract classification from filename patterns"""
        filename_lower = filename.lower()
        
        # Map of patterns to document types
        pattern_mappings = [
            (r'ALXN\d+.*protocol', 'PRO'),
            (r'protocol.*amendment', 'PRO'),
            (r'amendment', 'AME'),
            (r'weekly.*update', 'CI'),
            (r'global.*update', 'CI'),
            (r'poster', 'PUB'),
            (r'abstract', 'PUB'),
            (r'ASN\s+\d{4}', 'PUB'),
        ]
        
        for pattern, doc_type in pattern_mappings:
            if re.search(pattern, filename, re.IGNORECASE):
                info = self.type_lookup.get(doc_type, {})
                return {
                    'doc_type': doc_type,
                    'doc_id': info.get('id', 0),
                    'doc_name': info.get('name', ''),
                    'group': info.get('group', 'Unknown'),
                    'desc': info.get('desc', ''),
                    'confidence': 'Medium' if doc_type != 'CI' else 'High',
                    'method': 'filename_pattern'
                }
        return None
    
    def _classify_by_entities(self, entities):
        """Extract classification from entity mentions"""
        type_mappings = {
            'poster': 'PUB',
            'abstract': 'PUB',
            'presentation': 'PUB',
            'manuscript': 'PUB',
            'protocol': 'PRO',
            'amendment': 'AME',
            'brochure': 'IBR',
            'weekly update': 'CI',
            'global update': 'CI'
        }
        
        for doc_type_mention in entities.get('document_type', []):
            for key, code in type_mappings.items():
                if key.lower() in doc_type_mention.lower():
                    info = self.type_lookup.get(code, {})
                    return {
                        'doc_type': code,
                        'doc_id': info.get('id', 0),
                        'doc_name': info.get('name', ''),
                        'group': info.get('group', 'Unknown'),
                        'desc': info.get('desc', ''),
                        'confidence': 'Low',
                        'method': 'entity_extraction'
                    }
        return None
    
    def _classify_using_llm(self, doc):
        """Classify document using LLM with comprehensive document type schema"""
        filename = doc.get('name', 'Unknown')
        short_desc = doc.get('short_desc', '')
        long_desc = doc.get('long_term_desc', '') or doc.get('long_desc', '')
        content_sample = doc.get('content', '')[:2000] if 'content' in doc else ''
        
        # Build type descriptions
        type_desc = ""
        prioritized_types = [
            'PRO', 'AME', 'IBR', 'ICF', 'CRF', 'SAP', 'CSR', 'SAE',
            'PUB', 'MOA', 'EFF', 'ADV', 'COR', 'REG', 'HTA', 'CI',
            'DSC', 'DLA', 'REC'
        ]
        
        all_type_codes = list(self.type_lookup.keys())
        
        for code in prioritized_types:
            if code in self.type_lookup:
                info = self.type_lookup[code]
                type_desc += f"{code}: {info.get('name', '')} - {info.get('desc', '')}\n"
                if code in all_type_codes:
                    all_type_codes.remove(code)
        
        if all_type_codes:
            type_desc += "\nAdditional document types:\n"
            for code in all_type_codes:
                info = self.type_lookup[code]
                type_desc += f"{code}: {info.get('name', '')} - {info.get('desc', '')}\n"
        
        # Extract patterns
        patterns_found = []
        lower_filename = filename.lower()
        if "protocol" in lower_filename and "amendment" not in lower_filename:
            patterns_found.append("Protocol document pattern in filename")
        elif "amendment" in lower_filename:
            patterns_found.append("Protocol amendment pattern in filename")
        elif "poster" in lower_filename or "abstract" in lower_filename:
            patterns_found.append("Publication pattern in filename")
        elif "update" in lower_filename and ("weekly" in lower_filename or "global" in lower_filename):
            patterns_found.append("Competitive intelligence pattern in filename")
        
        # Create prompt
        prompt = f"""As a document classification expert for clinical and medical documents, please classify the following document into one of the document types defined below.

DOCUMENT INFORMATION:
Filename: {filename}
Short Description: {short_desc}
Long Description: {long_desc[:1200]}...
Content Sample: {content_sample[:1500]}...

Patterns detected: {'; '.join(patterns_found) if patterns_found else 'No specific patterns detected'}

DOCUMENT TYPES FOR CLASSIFICATION:
{type_desc}

Please analyze the document information carefully and determine the most appropriate document type code from the list above.
Your answer should be in JSON format with the following fields:
- doc_type: The document type code (e.g., PRO, AME, PUB)
- confidence: How confident you are in this classification (High, Medium, or Low)
- reasoning: A brief explanation of why you chose this classification

JSON Response:
"""
        
        try:
            response = call_claude_with_retry(self.client, prompt, retries=2, max_tokens=400, silent=self.silent)
            if not response or not response.content:
                logger.warning("No response from Claude or empty content")
                return None
            
            raw_response = response.content[0].text
            logger.debug(f"Claude's raw response: {raw_response[:200]}...")
            
            result_json = safe_json_loads(raw_response)
            if not result_json:
                doc_type = extract_first_doc_type(raw_response)
                if doc_type:
                    result_json = {
                        "doc_type": doc_type,
                        "confidence": "Low",
                        "reasoning": "Extracted from non-JSON response"
                    }
                else:
                    if not self.silent:
                        logger.warning(f"Failed to parse Claude response as JSON")
                    return None
            
            doc_type = result_json.get('doc_type', 'UNKNOWN')

            # Map common invalid responses to UNKNOWN
            invalid_responses = ['UNK', 'N/A', 'NA', 'None', 'null', '', 'UNKNOWN']
            if doc_type in invalid_responses:
                doc_type = 'UNKNOWN'
                # Don't return None - let it fall through to create UNKNOWN result

            doc_type = doc_type.upper()
            if doc_type != 'UNKNOWN' and doc_type not in self.type_lookup:
                norm_type = normalize_label(doc_type)
                if norm_type in self.alias_lookup:
                    doc_type = self.alias_lookup[norm_type]
                else:
                    if not self.silent:
                        logger.warning(f"LLM returned invalid doc_type: {doc_type}, mapping to UNKNOWN")
                    doc_type = 'UNKNOWN'  # Map to UNKNOWN instead of returning None
            
            info = self.type_lookup.get(doc_type, {})
            
            return {
                'doc_type': doc_type,
                'doc_id': info.get('id', 0),
                'doc_name': info.get('name', ''),
                'group': info.get('group', 'Unknown'),
                'desc': info.get('desc', ''),
                'confidence': result_json.get('confidence', 'Medium'),
                'method': 'llm_classification',
                'reasoning': result_json.get('reasoning', '')
            }
        
        except Exception as e:
            if not self.silent:
                logger.error(f"Error in LLM classification: {str(e)}")
                logger.debug(traceback.format_exc())
            return None
    
    def _post_process_confidence(self, classification, doc):
        """Apply enhanced document-specific confidence boosting rules"""
        doc_type = classification.get('doc_type')
        confidence = classification.get('confidence')
        
        if confidence == 'High' or doc_type == 'UNKNOWN':
            return classification
        
        boosters = self.confidence_boosters.get(doc_type, {})
        if not boosters:
            return classification
        
        filename = doc.get('name', '').lower()
        content = doc.get('content', '')[:2000].lower()
        long_desc = (doc.get('long_term_desc', '') or doc.get('long_desc', '')).lower()
        
        # Check patterns
        for pattern_type, patterns in [
            ('filename_patterns', filename),
            ('content_patterns', content),
            ('description_patterns', long_desc)
        ]:
            for pattern in boosters.get(pattern_type, []):
                try:
                    if re.search(pattern, patterns if isinstance(patterns, str) else filename, re.IGNORECASE):
                        classification['confidence'] = 'High'
                        logger.debug(f"Boosted confidence to High based on {pattern_type}: {pattern}")
                        return classification
                except re.error:
                    logger.warning(f"Invalid {pattern_type} for {doc_type}: {pattern}")
        
        return classification
    
    def _score_text_patterns(self, text, filename):
        """Score document content against pattern libraries"""
        scores = {code: 0 for code in self.doc_type_patterns.keys()}
        
        lower_text = text.lower()
        lower_filename = filename.lower()
        
        for doc_type, patterns in self.doc_type_patterns.items():
            # Filename scoring
            if doc_type == 'PRO' and ('protocol' in lower_filename and 'amendment' not in lower_filename):
                scores[doc_type] += 30
            elif doc_type == 'AME' and 'amendment' in lower_filename:
                scores[doc_type] += 30
            elif doc_type == 'CI' and ('weekly' in lower_filename and 'update' in lower_filename):
                scores[doc_type] += 40
            elif doc_type == 'PUB' and ('poster' in lower_filename or 'abstract' in lower_filename):
                scores[doc_type] += 30
            
            # Content patterns
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    match_count = len(matches)
                    
                    if match_count > 0:
                        base_score = 5
                        pattern_score = min(match_count * base_score, 25)
                        scores[doc_type] += pattern_score
                except re.error:
                    logger.warning(f"Invalid pattern in document type {doc_type}: {pattern}")
        
        return scores
    
    def _classify_by_long_term_desc(self, long_term_desc):
        """Classify document based on long-term description"""
        logger.debug(f"Attempting long-term description classification")
        
        norm_desc = long_term_desc.lower()
        
        # Direct alias matching
        for code, aliases in self.all_aliases.items():
            for alias in aliases:
                if alias.lower() in norm_desc:
                    info = self.type_lookup.get(code, {})
                    confidence = 'High' if len(alias) > 3 else 'Medium'
                    logger.debug(f"Long-term desc alias match: {alias} -> {code} ({confidence})")
                    return {
                        'doc_type': code,
                        'doc_id': info.get('id', 0),
                        'doc_name': info.get('name', ''),
                        'group': info.get('group', 'Unknown'),
                        'desc': info.get('desc', ''),
                        'confidence': confidence,
                        'method': 'long_term_desc_alias'
                    }
        
        # Pattern matching
        scores = self._score_text_patterns(norm_desc, "")
        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Top long-term desc pattern scores: {ranked_scores[:3]}")
        
        if ranked_scores and ranked_scores[0][1] >= 20:
            best_code = ranked_scores[0][0]
            score_gap = ranked_scores[0][1] - (ranked_scores[1][1] if len(ranked_scores) > 1 else 0)
            
            confidence = 'Low'
            if ranked_scores[0][1] >= 50 and score_gap >= 20:
                confidence = 'High'
            elif ranked_scores[0][1] >= 30 and score_gap >= 10:
                confidence = 'Medium'
            
            info = self.type_lookup.get(best_code, {})
            logger.debug(f"Long-term desc pattern match: {best_code} (confidence: {confidence})")
            
            return {
                'doc_type': best_code,
                'doc_id': info.get('id', 0),
                'doc_name': info.get('name', ''),
                'group': info.get('group', 'Unknown'),
                'desc': info.get('desc', ''),
                'confidence': confidence,
                'method': 'long_term_desc_pattern'
            }
        
        # Fuzzy matching
        best_match = None
        best_score = 0
        best_code = None
        
        for code, aliases in self.all_aliases.items():
            for alias in aliases:
                if len(alias) < 4:
                    continue
                
                score = fuzz.token_set_ratio(alias.lower(), norm_desc)
                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_match = alias
                    best_code = code
        
        if best_code:
            info = self.type_lookup.get(best_code, {})
            confidence = 'Low'
            if best_score >= self.high_threshold:
                confidence = 'High'
            elif best_score >= self.medium_threshold:
                confidence = 'Medium'
            
            logger.debug(f"Long-term desc fuzzy match: {best_match} -> {best_code} (score: {best_score}, confidence: {confidence})")
            
            return {
                'doc_type': best_code,
                'doc_id': info.get('id', 0),
                'doc_name': info.get('name', ''),
                'group': info.get('group', 'Unknown'),
                'desc': info.get('desc', ''),
                'confidence': confidence,
                'method': 'long_term_desc_fuzzy'
            }
        
        return None
    
    def _classify_by_short_desc(self, short_desc):
        """Classify document based on short description"""
        logger.debug(f"Attempting short description classification")
        
        norm_desc = short_desc.lower()
        
        # Direct alias matching
        for code, aliases in self.all_aliases.items():
            for alias in aliases:
                if alias.lower() in norm_desc:
                    info = self.type_lookup.get(code, {})
                    confidence = 'Medium'
                    logger.debug(f"Short desc alias match: {alias} -> {code} ({confidence})")
                    return {
                        'doc_type': code,
                        'doc_id': info.get('id', 0),
                        'doc_name': info.get('name', ''),
                        'group': info.get('group', 'Unknown'),
                        'desc': info.get('desc', ''),
                        'confidence': confidence,
                        'method': 'short_desc_alias'
                    }
        
        # Pattern matching
        scores = self._score_text_patterns(norm_desc, "")
        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Top short-desc pattern scores: {ranked_scores[:3]}")
        
        if ranked_scores and ranked_scores[0][1] >= 15:
            best_code = ranked_scores[0][0]
            score_gap = ranked_scores[0][1] - (ranked_scores[1][1] if len(ranked_scores) > 1 else 0)
            
            confidence = 'Low'
            if ranked_scores[0][1] >= 40 and score_gap >= 15:
                confidence = 'Medium'
            
            info = self.type_lookup.get(best_code, {})
            logger.debug(f"Short desc pattern match: {best_code} (confidence: {confidence})")
            
            return {
                'doc_type': best_code,
                'doc_id': info.get('id', 0),
                'doc_name': info.get('name', ''),
                'group': info.get('group', 'Unknown'),
                'desc': info.get('desc', ''),
                'confidence': confidence,
                'method': 'short_desc_pattern'
            }
        
        # Fuzzy matching
        best_match = None
        best_score = 0
        best_code = None
        
        for code, aliases in self.all_aliases.items():
            for alias in aliases:
                if len(alias) < 4:
                    continue
                
                score = fuzz.token_set_ratio(alias.lower(), norm_desc)
                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_match = alias
                    best_code = code
        
        if best_code:
            info = self.type_lookup.get(best_code, {})
            confidence = 'Low'
            if best_score >= self.high_threshold:
                confidence = 'Medium'
            elif best_score >= self.medium_threshold:
                confidence = 'Low'
            
            logger.debug(f"Short desc fuzzy match: {best_match} -> {best_code} (score: {best_score}, confidence: {confidence})")
            
            return {
                'doc_type': best_code,
                'doc_id': info.get('id', 0),
                'doc_name': info.get('name', ''),
                'group': info.get('group', 'Unknown'),
                'desc': info.get('desc', ''),
                'confidence': confidence,
                'method': 'short_desc_fuzzy'
            }
        
        return None