#!/usr/bin/env python3
"""
Claude Enhancement Module for Abbreviation Extraction
======================================================
Location: corpus_metadata/document_utils/abbreviation_claude.py
Version: 1.2.0

Handles Claude-based abbreviation resolution for difficult cases.

FIXES IN VERSION 1.2.0:
- Fixed index misalignment in _extract_context by using single-byte character replacement
- Maintained character position integrity when protecting abbreviations
- Added _protect_abbrev_dots and _restore_abbrev_dots helper methods
- Improved relative position calculation within extracted context
- Enhanced word boundary detection for context trimming

FIXES IN VERSION 1.1.0:
- Fixed sentence segmentation to handle common abbreviations
- Removed hardcoded model name (now from config only)
- Added rate limiting and retry logic
- Improved context extraction around abbreviation position
- Added centralized logging integration
- Enhanced response parsing robustness
- Added audit trail for confidence changes
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Character used to protect dots in abbreviations (single byte to maintain length)
PROTECT_DOT = '§'

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    try:
        # Try relative import
        from .metadata_logging_config import get_logger
        logger = get_logger(__name__)
    except ImportError:
        # Fallback to basic logging if centralized config not available
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        logger.debug("Using fallback logging configuration")

# ============================================================================
# SHARED DATA CLASSES (Should be imported from abbreviation_types.py)
# ============================================================================

# Try to import from shared types module
try:
    from .abbreviation_types import AbbreviationCandidate
except ImportError:
    # Define locally if shared module doesn't exist yet
    @dataclass
    class AbbreviationCandidate:
        """Represents a potential abbreviation found in text"""
        abbreviation: str
        expansion: str
        confidence: float = 0.0
        source: str = 'document'
        detection_source: str = ''
        dictionary_sources: List[str] = field(default_factory=list)
        context: str = ''
        position: int = -1
        page_number: int = -1
        validation_status: str = 'unvalidated'
        disambiguation_needed: bool = False
        alternative_expansions: List[str] = field(default_factory=list)
        context_type: str = 'general'
        local_expansion: Optional[str] = None
        context_score: float = 0.0
        domain_context: str = ''
        claude_resolved: bool = False
        # Audit fields
        old_confidence: Optional[float] = None
        resolution_reason: Optional[str] = None


class ClaudeEnhancer:
    """Handles Claude-based abbreviation resolution"""
    
    # Common abbreviations that shouldn't break sentences
    SENTENCE_BREAK_EXCEPTIONS = {
        'al', 'et', 'Fig', 'Dr', 'vs', 'Mr', 'Mrs', 'Ms', 
        'Prof', 'Inc', 'Ltd', 'Co', 'Corp', 'Jr', 'Sr',
        'Ph', 'M', 'D', 'Ed', 'i.e', 'e.g', 'cf', 'viz',
        'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 
        'Sep', 'Sept', 'Oct', 'Nov', 'Dec'
    }
    
    def __init__(self, claude_client, config: Dict):
        """
        Initialize Claude enhancer.
        
        Args:
            claude_client: Anthropic Claude client instance
            config: Configuration dictionary
        """
        self.claude_client = claude_client
        self.config = config
        
        # Get model from api_configuration.claude (matching your config.yaml structure)
        api_config = config.get('api_configuration', {})
        claude_config = api_config.get('claude', {})
        
        # Get the model - this matches where it's defined in your config.yaml
        self.model = claude_config.get('model')
        
        if self.model:
            logger.info(f"Using Claude model: {self.model}")
        else:
            # Use the same default as rare_disease_drug_validator.py
            self.model = 'claude-3-5-sonnet-20241022'
            logger.warning(f"No Claude model specified in config. Using fallback model: {self.model}")
        
        # Get temperature and max_tokens from api_configuration.claude
        self.temperature = claude_config.get('temperature', 0)
        self.max_tokens = claude_config.get('max_tokens', 1500)
        
        # Get abbreviation-specific settings (if they exist)
        abbrev_config = config.get('abbreviation_extraction', {})
        
        # These might be abbreviation-specific
        self.confidence_threshold = abbrev_config.get('confidence_threshold', 0.7)
        self.max_batch_size = abbrev_config.get('max_candidates_per_batch', 10)
        
        # Context settings (with sensible defaults)
        self.context_sentences_before = abbrev_config.get('context_sentences_before', 1)
        self.context_sentences_after = abbrev_config.get('context_sentences_after', 0)
        self.context_char_radius = abbrev_config.get('context_char_radius', 350)
        self.max_context_length = abbrev_config.get('max_context_length', 500)
        
        # Rate limiting (with defaults)
        self.max_retries = abbrev_config.get('max_retries', 3)
        self.retry_delay = abbrev_config.get('retry_delay', 1.0)
        self.retry_backoff = abbrev_config.get('retry_backoff', 2.0)
        
        # Load custom sentence break exceptions if provided
        custom_exceptions = abbrev_config.get('sentence_break_exceptions', [])
        if hasattr(self, 'SENTENCE_BREAK_EXCEPTIONS'):
            self.sentence_exceptions = self.SENTENCE_BREAK_EXCEPTIONS.union(set(custom_exceptions))
        else:
            self.sentence_exceptions = set(custom_exceptions)
    
    def _protect_abbrev_dots(self, text: str) -> str:
        """
        Protect dots after known abbreviations by replacing with a special character.
        This maintains the same text length to preserve character positions.
        
        Args:
            text: Original text
            
        Returns:
            Text with protected abbreviation dots
        """
        protected = text
        for abbr in self.sentence_exceptions:
            # Replace only the dot after the abbreviation, keeping length constant
            pattern = rf'\b{re.escape(abbr)}\.'
            replacement = f'{abbr}{PROTECT_DOT}'
            protected = re.sub(pattern, replacement, protected)
        return protected
    
    def _restore_abbrev_dots(self, text: str) -> str:
        """
        Restore protected dots back to original.
        
        Args:
            text: Text with protected dots
            
        Returns:
            Text with original dots restored
        """
        return text.replace(PROTECT_DOT, '.')
    
    def enhance_candidates(self, candidates: List[AbbreviationCandidate], 
                          text: str) -> List[AbbreviationCandidate]:
        """
        Enhance difficult candidates with Claude.
        
        Args:
            candidates: List of abbreviation candidates
            text: Full document text
            
        Returns:
            Enhanced list of candidates
        """
        if not self.claude_client:
            logger.debug("No Claude client available, skipping enhancement")
            return candidates
        
        # Find difficult cases
        difficult = [c for c in candidates 
                    if (not c.expansion or c.confidence < self.confidence_threshold)
                    and not c.claude_resolved]
        
        if not difficult:
            logger.debug("No difficult abbreviations to process")
            return candidates
        
        logger.info(f"Processing {len(difficult)} difficult abbreviations with Claude")
        
        # Process in batches
        for i in range(0, len(difficult), self.max_batch_size):
            batch = difficult[i:i + self.max_batch_size]
            
            # Extract contexts
            contexts = []
            for cand in batch:
                context = self._extract_context(text, cand.position)
                contexts.append({
                    'abbreviation': cand.abbreviation,
                    'context': context,
                    'position': cand.position
                })
            
            # Query Claude with retry logic
            expansions = self._query_claude_with_retry(contexts)
            
            # Update candidates
            for j, cand in enumerate(batch):
                if j < len(expansions) and expansions[j]:
                    # Store old confidence for audit trail
                    cand.old_confidence = cand.confidence
                    cand.expansion = expansions[j]
                    cand.confidence = 0.85
                    cand.source = 'claude_inference'
                    cand.validation_status = 'claude_context'
                    cand.claude_resolved = True
                    cand.resolution_reason = 'claude_context_resolution'
                    logger.debug(f"Claude resolved: {cand.abbreviation} -> {cand.expansion} "
                               f"(confidence: {cand.old_confidence:.2f} -> {cand.confidence:.2f})")
        
        return candidates
    
    def _extract_context(self, text: str, position: int) -> str:
        """
        Extract sentence context around position with improved segmentation.
        Maintains character position alignment when handling abbreviations.
        
        Args:
            text: Full text
            position: Character position of abbreviation
            
        Returns:
            Context string
        """
        # Protect abbreviation dots without changing text length
        protected_text = self._protect_abbrev_dots(text)
        
        # Now split sentences on remaining periods, exclamations, and questions
        # The positions remain aligned since we only replaced characters 1:1
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)
        
        # Find target sentence containing the position
        char_count = 0
        target_idx = 0
        
        for i, sent in enumerate(sentences):
            sent_len = len(sent)
            if char_count <= position < char_count + sent_len:
                target_idx = i
                break
            # Account for the space between sentences (if not last sentence)
            char_count += sent_len
            if i < len(sentences) - 1:
                char_count += 1  # Add 1 for the space between sentences
        
        # Extract surrounding sentences based on configuration
        start_idx = max(0, target_idx - self.context_sentences_before)
        end_idx = min(len(sentences), target_idx + self.context_sentences_after + 1)
        
        # Join the selected sentences and restore dots
        context = ' '.join(sentences[start_idx:end_idx])
        context = self._restore_abbrev_dots(context)
        
        # Trim if context is too long
        if len(context) > self.max_context_length:
            # Calculate the abbreviation's position within the selected context
            # Account for sentences before the target
            rel_pos = position
            for j in range(start_idx):
                rel_pos -= len(sentences[j])
                if j < len(sentences) - 1:
                    rel_pos -= 1  # Subtract space between sentences
            
            # Ensure rel_pos is within bounds
            if rel_pos < 0 or rel_pos >= len(context):
                # Fallback to center if position calculation is off
                rel_pos = len(context) // 2
            
            # Calculate window around the abbreviation
            window_start = max(0, rel_pos - self.context_char_radius)
            window_end = min(len(context), rel_pos + self.context_char_radius)
            
            # Adjust to word boundaries
            while window_start > 0 and context[window_start - 1] not in ' \n\t.,;:!?':
                window_start -= 1
            while window_end < len(context) and context[window_end] not in ' \n\t.,;:!?':
                window_end += 1
            
            # Extract the window
            windowed_context = context[window_start:window_end].strip()
            
            # Add ellipsis for truncated context
            if window_start > 0:
                windowed_context = "..." + windowed_context
            if window_end < len(context):
                windowed_context = windowed_context + "..."
            
            context = windowed_context
        
        return context.strip()
    
    def _extract_context_alternative(self, text: str, position: int) -> str:
        """
        Alternative approach: Extract context using a simple character window
        without sentence segmentation. This is more robust for edge cases.
        
        Args:
            text: Full text
            position: Character position of abbreviation
            
        Returns:
            Context string
        """
        # Simple character-based window extraction
        start = max(0, position - self.context_char_radius)
        end = min(len(text), position + self.context_char_radius)
        
        # Adjust to word boundaries
        while start > 0 and text[start - 1] not in ' \n\t':
            start -= 1
        while end < len(text) and text[end] not in ' \n\t':
            end += 1
        
        context = text[start:end].strip()
        
        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _query_claude_with_retry(self, contexts: List[Dict]) -> List[Optional[str]]:
        """
        Query Claude with retry logic for rate limiting.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            List of expansions (or None for unknown)
        """
        retry_count = 0
        delay = self.retry_delay
        
        while retry_count < self.max_retries:
            try:
                return self._query_claude(contexts)
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limiting
                if '429' in error_str or 'rate' in error_str:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {retry_count}/{self.max_retries})")
                        time.sleep(delay)
                        delay *= self.retry_backoff
                    else:
                        logger.error(f"Max retries reached for Claude query")
                        break
                
                # Check for server errors
                elif any(code in error_str for code in ['500', '502', '503', '504']):
                    retry_count += 1
                    if retry_count < self.max_retries:
                        logger.warning(f"Server error, retrying in {delay:.1f}s")
                        time.sleep(delay)
                        delay *= self.retry_backoff
                    else:
                        logger.error(f"Max retries reached for Claude query")
                        break
                else:
                    # Non-retryable error
                    logger.error(f"Claude query failed: {e}")
                    break
        
        # Return None for all contexts if all retries failed
        return [None] * len(contexts)
    
    def _query_claude(self, contexts: List[Dict]) -> List[Optional[str]]:
        """
        Query Claude for expansions.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            List of expansions (or None for unknown)
        """
        if not contexts:
            return []
        
        prompt = self._build_prompt(contexts)
        
        try:
            # Validate model is available
            if not self.model:
                raise ValueError("No Claude model configured")
            
            # Use the Claude API
            response = self.claude_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text from response
            response_text = response.content[0].text
            return self._parse_response(response_text, len(contexts))
            
        except Exception as e:
            # Re-raise for retry logic to handle
            raise
    
    def _build_prompt(self, contexts: List[Dict]) -> str:
        """
        Build prompt for Claude.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are a biomedical abbreviation expert. For each abbreviation below, provide ONLY the full expansion based on the context. Focus on medical, pharmaceutical, and scientific terminology.

Rules:
- If uncertain, respond with "UNKNOWN"
- Provide one expansion per line, nothing else
- No explanations or additional text
- Consider the biomedical/pharmaceutical context
- Pay attention to similar abbreviations that may have different meanings

"""
        
        for i, ctx in enumerate(contexts):
            # Ensure context isn't too long
            context_text = ctx['context'][:self.max_context_length] if len(ctx['context']) > self.max_context_length else ctx['context']
            prompt += f"{i+1}. Abbreviation: {ctx['abbreviation']}\n"
            prompt += f"   Context: {context_text}\n\n"
        
        prompt += "\nExpansions (one per line):"
        
        return prompt
    
    def _parse_response(self, response_text: str, expected_count: int) -> List[Optional[str]]:
        """
        Parse Claude's response into expansions with improved robustness.
        
        Args:
            response_text: Raw response from Claude
            expected_count: Number of expansions expected
            
        Returns:
            List of parsed expansions
        """
        if not response_text:
            logger.warning("Empty response from Claude")
            return [None] * expected_count
        
        lines = response_text.strip().split('\n')
        expansions = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Remove various formatting artifacts
            line = line.strip()
            # Remove numbering (1., 2., etc.)
            line = re.sub(r'^\d+\.\s*', '', line)
            # Remove bullet points (-, *, •)
            line = re.sub(r'^[-*•]\s*', '', line)
            # Remove quotes
            line = line.strip('"\'')
            
            # Check for UNKNOWN or similar indicators
            if line.upper() in ["UNKNOWN", "N/A", "NA", "NOT FOUND", "UNCLEAR"]:
                expansions.append(None)
            elif line:
                # Clean up the expansion
                expansion = line.strip()
                
                # Handle wrapped lines (if next line doesn't start with number/bullet)
                # This is handled by the line processing above
                
                expansions.append(expansion)
        
        # Handle mismatch in count
        if len(expansions) < expected_count:
            logger.warning(f"Got {len(expansions)} expansions but expected {expected_count}")
            # Pad with None
            while len(expansions) < expected_count:
                expansions.append(None)
        elif len(expansions) > expected_count:
            logger.warning(f"Got {len(expansions)} expansions but expected {expected_count}")
            # Truncate
            expansions = expansions[:expected_count]
        
        return expansions
    
    def resolve_single_abbreviation(self, abbreviation: str, context: str) -> Optional[str]:
        """
        Resolve a single abbreviation using Claude.
        
        Args:
            abbreviation: The abbreviation to resolve
            context: Context in which it appears
            
        Returns:
            Expansion or None if unknown
        """
        contexts = [{
            'abbreviation': abbreviation,
            'context': context,
            'position': 0
        }]
        
        expansions = self._query_claude_with_retry(contexts)
        return expansions[0] if expansions else None
    
    def validate_model(self) -> bool:
        """
        Validate that the configured model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        if not self.claude_client or not self.model:
            return False
        
        try:
            # Try a minimal query to validate model
            response = self.claude_client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Model validation failed for {self.model}: {e}")
            return False