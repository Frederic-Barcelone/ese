#!/usr/bin/env python3
"""
Entity Validations - AI-Powered Reference Validation
=====================================================
Location: corpus_metadata/document_utils/entity_validations.py
Version: 1.0.0
Last Updated: 2025-10-13

AI validation for extracted references using Claude to detect misclassifications,
particularly for PMID vs INSPIREHEP confusion and other identifier ambiguities.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ValidationIssue:
    """Represents a validation issue found by AI."""
    reference_id: str
    issue_type: str  # 'misclassification', 'invalid_format', 'missing_info', 'suspicious'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    suggested_fix: Optional[Dict] = None
    confidence: float = 1.0


@dataclass
class BatchValidationResult:
    """Results from batch validation of references."""
    
    # Counts
    total_references: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    misclassified_count: int = 0
    
    # Issues by reference
    issues_by_reference: Dict[str, List[ValidationIssue]] = field(default_factory=dict)
    
    # Corrections to apply
    corrections: Dict[str, Dict] = field(default_factory=dict)
    
    # Statistics
    issue_type_counts: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'summary': {
                'total_references': self.total_references,
                'valid_count': self.valid_count,
                'invalid_count': self.invalid_count,
                'misclassified_count': self.misclassified_count,
                'issue_type_counts': self.issue_type_counts,
                'severity_counts': self.severity_counts
            },
            'issues': {
                ref_id: [
                    {
                        'issue_type': issue.issue_type,
                        'severity': issue.severity,
                        'description': issue.description,
                        'suggested_fix': issue.suggested_fix,
                        'confidence': issue.confidence
                    }
                    for issue in issues
                ]
                for ref_id, issues in self.issues_by_reference.items()
            },
            'corrections': self.corrections,
            'validation_timestamp': self.validation_timestamp,
            'processing_time_seconds': round(self.processing_time_seconds, 3)
        }


# ============================================================================
# REFERENCE VALIDATOR
# ============================================================================

class ReferenceValidator:
    """
    AI-powered reference validator using pattern matching and heuristics.
    
    Detects common issues like:
    - PMID misclassified as INSPIREHEP
    - Invalid identifier formats
    - Missing or inconsistent metadata
    - Suspicious patterns
    """
    
    def __init__(self, batch_size: int = 20):
        """Initialize validator."""
        self.batch_size = batch_size
        
        # Compile validation patterns
        self._compile_patterns()
        
        logger.info("ReferenceValidator initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for validation."""
        
        # PMID indicators in context
        self.pmid_indicators = [
            re.compile(r'\bpubmed\b', re.IGNORECASE),
            re.compile(r'\bPMID:?\s*\d+', re.IGNORECASE),
            re.compile(r'ncbi\.nlm\.nih\.gov', re.IGNORECASE),
            re.compile(r'/pubmed/', re.IGNORECASE),
            re.compile(r'Available from:.*pubmed', re.IGNORECASE)
        ]
        
        # INSPIREHEP indicators
        self.inspirehep_indicators = [
            re.compile(r'\bINSPIRE(?:-HEP)?\b', re.IGNORECASE),
            re.compile(r'inspirehep\.net', re.IGNORECASE),
            re.compile(r'\bhigh energy physics\b', re.IGNORECASE),
            re.compile(r'\barXiv:hep-', re.IGNORECASE)
        ]
        
        # DOI pattern
        self.doi_pattern = re.compile(r'^10\.\d{4,9}/[^\s]+$')
        
        # PMID pattern
        self.pmid_pattern = re.compile(r'^\d{6,8}$')
        
        # NCT pattern
        self.nct_pattern = re.compile(r'^NCT\d{8}$')
    
    def validate_references(
        self,
        references: List[Dict],
        context_text: Optional[str] = None
    ) -> BatchValidationResult:
        """
        Validate a batch of references.
        
        Args:
            references: List of reference dictionaries
            context_text: Optional full document text for context analysis
            
        Returns:
            BatchValidationResult with issues and corrections
        """
        start_time = datetime.now()
        
        result = BatchValidationResult()
        result.total_references = len(references)
        
        logger.info("=" * 80)
        logger.info("REFERENCE VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Validating {len(references)} references...")
        
        # Count reference types
        type_counts = defaultdict(int)
        for ref in references:
            type_counts[ref.get('reference_type', 'unknown')] += 1
        
        logger.info(f"Reference types breakdown:")
        for ref_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {ref_type}: {count}")
        
        for idx, ref in enumerate(references, 1):
            ref_type = ref.get('reference_type', 'unknown')
            normalized = ref.get('normalized_value', '')
            
            logger.debug(f"[{idx}/{len(references)}] Validating {ref_type}: {normalized}")
            
            issues = self._validate_single_reference(ref, context_text)
            
            if issues:
                result.issues_by_reference[ref['reference_id']] = issues
                result.invalid_count += 1
                
                # Log issues found
                for issue in issues:
                    severity_symbol = "🔴" if issue.severity == "critical" else "⚠️" if issue.severity == "warning" else "ℹ️"
                    logger.info(f"  {severity_symbol} {ref_type} '{normalized}': {issue.description}")
                
                # Check if misclassification issue exists
                if any(issue.issue_type == 'misclassification' for issue in issues):
                    result.misclassified_count += 1
                    
                    # Generate correction
                    correction = self._generate_correction(ref, issues)
                    if correction:
                        result.corrections[ref['reference_id']] = correction
                        logger.info(f"    ✓ Correction: {correction}")
                
                # Count issue types and severities
                for issue in issues:
                    result.issue_type_counts[issue.issue_type] = \
                        result.issue_type_counts.get(issue.issue_type, 0) + 1
                    result.severity_counts[issue.severity] = \
                        result.severity_counts.get(issue.severity, 0) + 1
            else:
                result.valid_count += 1
                logger.debug(f"  ✓ Valid: {ref_type} '{normalized}'")
        
        # Calculate processing time
        end_time = datetime.now()
        result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        # Log summary
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total references: {result.total_references}")
        logger.info(f"Valid: {result.valid_count} ({result.valid_count/result.total_references*100:.1f}%)")
        logger.info(f"Invalid: {result.invalid_count} ({result.invalid_count/result.total_references*100:.1f}%)")
        logger.info(f"Misclassified: {result.misclassified_count}")
        logger.info(f"Corrections applied: {len(result.corrections)}")
        
        if result.issue_type_counts:
            logger.info(f"\nIssue types:")
            for issue_type, count in sorted(result.issue_type_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {issue_type}: {count}")
        
        if result.severity_counts:
            logger.info(f"\nSeverity breakdown:")
            for severity, count in sorted(result.severity_counts.items()):
                logger.info(f"  - {severity}: {count}")
        
        logger.info(f"\nProcessing time: {result.processing_time_seconds:.2f}s")
        logger.info("=" * 80)
        
        return result
    
    def _validate_single_reference(
        self,
        ref: Dict,
        context_text: Optional[str]
    ) -> List[ValidationIssue]:
        """Validate a single reference."""
        issues = []
        
        ref_type = ref.get('reference_type', '')
        ref_id = ref.get('reference_id', '')
        normalized = ref.get('normalized_value', '')
        
        # Check for PMID misclassified as INSPIREHEP
        if ref_type == 'inspirehep':
            issue = self._check_pmid_misclassification(ref)
            if issue:
                issues.append(issue)
        
        # Check for invalid format
        format_issue = self._check_format(ref)
        if format_issue:
            issues.append(format_issue)
        
        # Check for missing metadata
        metadata_issue = self._check_metadata(ref)
        if metadata_issue:
            issues.append(metadata_issue)
        
        # Check for suspicious patterns
        suspicious_issue = self._check_suspicious_patterns(ref)
        if suspicious_issue:
            issues.append(suspicious_issue)
        
        return issues
    
    def _check_pmid_misclassification(self, ref: Dict) -> Optional[ValidationIssue]:
        """Check if an INSPIREHEP reference is actually a PMID."""
        
        # Get context
        context = ""
        if ref.get('preceding_context'):
            context += ref['preceding_context'] + " "
        if ref.get('following_context'):
            context += ref['following_context']
        if ref.get('sentence'):
            context += " " + ref['sentence']
        
        context_lower = context.lower()
        
        # Count PMID indicators
        pmid_score = sum(
            1 for pattern in self.pmid_indicators
            if pattern.search(context)
        )
        
        # Count INSPIREHEP indicators
        inspirehep_score = sum(
            1 for pattern in self.inspirehep_indicators
            if pattern.search(context)
        )
        
        # Check if PMID pattern matches
        normalized = ref.get('normalized_value', '')
        is_pmid_format = self.pmid_pattern.match(normalized) is not None
        
        # Log detailed analysis for INSPIREHEP references
        if pmid_score > 0 or inspirehep_score > 0:
            logger.debug(f"  Context analysis for {normalized}:")
            logger.debug(f"    - PMID indicators: {pmid_score}")
            logger.debug(f"    - INSPIREHEP indicators: {inspirehep_score}")
            logger.debug(f"    - PMID format match: {is_pmid_format}")
        
        # Decision: likely PMID if:
        # 1. Has PMID indicators AND
        # 2. No INSPIREHEP indicators AND
        # 3. Matches PMID format
        if pmid_score > 0 and inspirehep_score == 0 and is_pmid_format:
            logger.info(f"  🔄 Misclassification detected: {normalized} (INSPIREHEP → PMID)")
            return ValidationIssue(
                reference_id=ref['reference_id'],
                issue_type='misclassification',
                severity='critical',
                description=f"Reference classified as INSPIREHEP but context indicates PMID (score: {pmid_score})",
                suggested_fix={
                    'reference_type': 'pmid',
                    'category': 'literature',
                    'source': 'PubMed',
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{normalized}/'
                },
                confidence=min(1.0, pmid_score * 0.3 + 0.5)
            )
        
        return None
    
    def _check_format(self, ref: Dict) -> Optional[ValidationIssue]:
        """Check if reference format is valid."""
        
        ref_type = ref.get('reference_type', '')
        normalized = ref.get('normalized_value', '')
        
        # Check DOI format
        if ref_type == 'doi':
            if not self.doi_pattern.match(normalized):
                logger.debug(f"  ⚠️  Invalid DOI format: {normalized}")
                return ValidationIssue(
                    reference_id=ref['reference_id'],
                    issue_type='invalid_format',
                    severity='warning',
                    description=f"Invalid DOI format: {normalized}",
                    confidence=0.9
                )
            else:
                logger.debug(f"  ✓ Valid DOI format: {normalized}")
        
        # Check PMID format
        elif ref_type == 'pmid':
            if not self.pmid_pattern.match(normalized):
                logger.debug(f"  ⚠️  Invalid PMID format: {normalized}")
                return ValidationIssue(
                    reference_id=ref['reference_id'],
                    issue_type='invalid_format',
                    severity='warning',
                    description=f"Invalid PMID format: {normalized}",
                    confidence=0.9
                )
            else:
                logger.debug(f"  ✓ Valid PMID format: {normalized}")
        
        # Check NCT format
        elif ref_type == 'clinicaltrials_gov':
            if not self.nct_pattern.match(normalized):
                logger.debug(f"  ⚠️  Invalid NCT format: {normalized}")
                return ValidationIssue(
                    reference_id=ref['reference_id'],
                    issue_type='invalid_format',
                    severity='warning',
                    description=f"Invalid NCT format: {normalized}",
                    confidence=0.9
                )
            else:
                logger.debug(f"  ✓ Valid NCT format: {normalized}")
        
        return None
    
    def _check_metadata(self, ref: Dict) -> Optional[ValidationIssue]:
        """Check if reference has expected metadata."""
        
        # Check for missing URL when expected
        ref_type = ref.get('reference_type', '')
        url = ref.get('url')
        
        if ref_type in ['doi', 'pmid', 'clinicaltrials_gov'] and not url:
            return ValidationIssue(
                reference_id=ref['reference_id'],
                issue_type='missing_info',
                severity='info',
                description=f"Missing URL for {ref_type}",
                confidence=0.7
            )
        
        return None
    
    def _check_suspicious_patterns(self, ref: Dict) -> Optional[ValidationIssue]:
        """Check for suspicious patterns in reference."""
        
        normalized = ref.get('normalized_value', '')
        raw_text = ref.get('raw_text', '')
        
        # Check for very short normalized values
        if len(normalized) < 3:
            return ValidationIssue(
                reference_id=ref['reference_id'],
                issue_type='suspicious',
                severity='warning',
                description=f"Suspicious short normalized value: '{normalized}'",
                confidence=0.8
            )
        
        # Check for significant mismatch between raw and normalized
        if len(normalized) > len(raw_text) * 2:
            return ValidationIssue(
                reference_id=ref['reference_id'],
                issue_type='suspicious',
                severity='info',
                description="Normalized value much longer than raw text",
                confidence=0.6
            )
        
        return None
    
    def _generate_correction(
        self,
        ref: Dict,
        issues: List[ValidationIssue]
    ) -> Optional[Dict]:
        """Generate correction dictionary from issues."""
        
        # Find the highest confidence suggested fix
        fixes = [
            issue.suggested_fix
            for issue in issues
            if issue.suggested_fix and issue.confidence >= 0.7
        ]
        
        if not fixes:
            return None
        
        # Return the first (highest priority) fix
        return fixes[0]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_validation_corrections(
    references: List[Dict],
    validation_result: BatchValidationResult
) -> List[Dict]:
    """
    Apply validation corrections to references.
    
    Args:
        references: List of reference dictionaries
        validation_result: Validation result with corrections
        
    Returns:
        List of corrected references
    """
    corrected_refs = []
    corrections_applied = 0
    
    logger.info("=" * 80)
    logger.info("APPLYING VALIDATION CORRECTIONS")
    logger.info("=" * 80)
    
    for ref in references:
        ref_id = ref['reference_id']
        
        if ref_id in validation_result.corrections:
            correction = validation_result.corrections[ref_id]
            
            # Log what's being corrected
            old_type = ref.get('reference_type', 'unknown')
            new_type = correction.get('reference_type', old_type)
            normalized = ref.get('normalized_value', '')
            
            logger.info(f"✓ Correcting: {normalized}")
            logger.info(f"  Type: {old_type} → {new_type}")
            if 'url' in correction:
                logger.info(f"  URL: {correction['url']}")
            
            # Apply corrections
            for key, value in correction.items():
                ref[key] = value
            
            # Mark as corrected
            ref['validation_corrected'] = True
            corrections_applied += 1
            
            # Add validation info
            issues = validation_result.issues_by_reference.get(ref_id, [])
            ref['validation'] = {
                'corrected': True,
                'issues': [
                    {
                        'type': issue.issue_type,
                        'severity': issue.severity,
                        'description': issue.description
                    }
                    for issue in issues
                ]
            }
            
            logger.debug(f"Applied correction to {ref_id}: {correction}")
        else:
            # No corrections needed
            ref['validation_corrected'] = False
        
        corrected_refs.append(ref)
    
    logger.info("=" * 80)
    logger.info(f"Applied {corrections_applied} correction(s) to references")
    logger.info("=" * 80)
    
    return corrected_refs


def validate_and_correct_references(
    references: List[Dict],
    context_text: Optional[str] = None,
    batch_size: int = 20
) -> Tuple[List[Dict], BatchValidationResult]:
    """
    Convenience function to validate and correct references in one step.
    
    Args:
        references: List of reference dictionaries
        context_text: Optional full document text
        batch_size: Batch size for validation
        
    Returns:
        Tuple of (corrected_references, validation_result)
    """
    validator = ReferenceValidator(batch_size=batch_size)
    
    # Validate
    validation_result = validator.validate_references(references, context_text)
    
    # Apply corrections
    corrected_refs = apply_validation_corrections(references, validation_result)
    
    return corrected_refs, validation_result