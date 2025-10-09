#!/usr/bin/env python3
"""
Entity Processing Utilities - UPDATED VERSION 3.0
============================
Location: corpus_metadata/document_utils/entity_utils.py
Version: 3.0.0
Updated: 2025-01-19

Description:
Enhanced utility functions with soft domain priors and uncertainty gating.
Avoids overfitting through evidence-based disambiguation.
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
import math
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Context-Aware Disambiguation with Soft Priors
# ============================================================================

class ContextAwareDisambiguator:
    """Context-aware disambiguation using soft priors and local evidence"""
    
    # Constants for uncertainty handling
    MARGIN_THRESHOLD = 0.12      # Min gap between top-1 and top-2 scores
    EVIDENCE_MIN = 0.25          # Min local evidence to make decision
    SOFT_PRIOR_WEIGHT = 0.15     # Weight for domain hints
    
    # Semantic type groups (general categories)
    SEMANTIC_TYPE_GROUPS = {
        'disorders': {
            'T047',  # Disease or Syndrome
            'T048',  # Mental or Behavioral Dysfunction
            'T191',  # Neoplastic Process
            'T046',  # Pathologic Function
        },
        'chemicals': {
            'T121',  # Pharmacologic Substance
            'T109',  # Organic Chemical
            'T195',  # Antibiotic
            'T200',  # Clinical Drug
        },
        'proteins': {
            'T116',  # Amino Acid, Peptide, or Protein
            'T126',  # Enzyme
            'T129',  # Immunologic Factor
        },
        'genes': {
            'T028',  # Gene or Genome
        }
    }
    
    def __init__(self, cooccurrence_matrix: Dict = None):
        """Initialize with optional co-occurrence data"""
        self.cooccurrence_matrix = cooccurrence_matrix or {}
        
    def score_expansion(self, 
                       abbr: str, 
                       expansion: str, 
                       context: str,
                       domain_hint: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Score expansion with soft priors and local evidence requirement
        
        Args:
            abbr: The abbreviation
            expansion: Candidate expansion
            context: Local context window
            domain_hint: Optional domain suggestion
            
        Returns:
            Tuple of (total_score, evidence_components)
        """
        evidence_components = {}
        
        # 1. Soft domain prior (small bump if matches)
        prior = 0.0
        if domain_hint and self._matches_domain_pattern(expansion, domain_hint):
            prior = self.SOFT_PRIOR_WEIGHT
        evidence_components['domain_prior'] = prior
        
        # 2. Local evidence signals
        exp_tokens = self._tokenize(expansion)
        ctx_tokens = self._tokenize(context)
        
        # Direct appearance of expansion in context
        longform_hit = 0.3 if expansion.lower() in context.lower() else 0.0
        evidence_components['longform'] = longform_hit
        
        # Token overlap (Jaccard similarity)
        overlap = self._jaccard(exp_tokens, ctx_tokens)
        overlap_score = 0.2 * overlap
        evidence_components['overlap'] = overlap_score
        
        # 3. Statistical co-occurrence
        cooc_score = self._calculate_cooccurrence(abbr, context)
        evidence_components['cooccurrence'] = cooc_score
        
        # 4. Semantic coherence
        coherence = self._semantic_coherence(expansion, context)
        evidence_components['coherence'] = coherence
        
        # 5. Definitional patterns
        definition_score = self._check_definition_pattern(abbr, expansion, context)
        evidence_components['definition'] = definition_score
        
        # Calculate total local evidence
        local_evidence = longform_hit + overlap_score + cooc_score
        evidence_components['local_evidence'] = local_evidence
        
        # Base score from multiple signals
        base = (
            0.25 * coherence +
            0.20 * cooc_score +
            0.15 * definition_score +
            0.10 * overlap_score
        )
        
        total = min(1.0, base + prior + local_evidence)
        
        return total, evidence_components
    
    def disambiguate_with_context(self, 
                                 abbr: str,
                                 candidates: List[str],
                                 context: str,
                                 domain_hint: str = None) -> Dict[str, Any]:
        """
        Disambiguate with uncertainty handling
        
        Args:
            abbr: The abbreviation
            candidates: List of candidate expansions
            context: Context window
            domain_hint: Optional domain
            
        Returns:
            Disambiguation result with confidence and alternatives
        """
        if not candidates:
            return {
                'expansion': None,
                'confidence': 0.0,
                'status': 'no_candidates',
                'alternatives': []
            }
        
        # Score all candidates
        scored = []
        for exp in candidates:
            score, components = self.score_expansion(abbr, exp, context, domain_hint)
            scored.append({
                'expansion': exp,
                'score': score,
                'components': components
            })
        
        # Sort by score
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        
        # Calculate margin and check evidence
        margin = best['score'] - (second['score'] if second else 0.0)
        local_evidence = best['components'].get('local_evidence', 0.0)
        
        # Uncertainty gating
        if margin < self.MARGIN_THRESHOLD and local_evidence < self.EVIDENCE_MIN:
            # Ambiguous - return with low confidence
            return {
                'expansion': best['expansion'],
                'confidence': min(0.65, best['score']),
                'status': 'ambiguous',
                'alternatives': [s['expansion'] for s in scored[1:3]],
                'margin': margin,
                'local_evidence': local_evidence
            }
        
        # Clear winner with evidence
        return {
            'expansion': best['expansion'],
            'confidence': best['score'],
            'status': 'disambiguated',
            'alternatives': [s['expansion'] for s in scored[1:3] if s['score'] > 0.3],
            'margin': margin,
            'local_evidence': local_evidence
        }
    
    def _tokenize(self, text: str) -> Set[str]:
        """Extract meaningful tokens"""
        return set(re.findall(r'[a-z]+', text.lower()))
    
    def _jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _matches_domain_pattern(self, expansion: str, domain: str) -> bool:
        """Check if expansion matches domain expectations"""
        exp_lower = expansion.lower()
        domain_lower = domain.lower()
        
        # General patterns, not specific terms
        domain_patterns = {
            'clinical': ['disease', 'disorder', 'syndrome', 'condition'],
            'pharmacological': ['drug', 'inhibitor', 'agonist', 'antagonist'],
            'molecular': ['protein', 'enzyme', 'receptor', 'kinase'],
            'genetic': ['gene', 'mutation', 'variant', 'allele']
        }
        
        for domain_type, terms in domain_patterns.items():
            if domain_type in domain_lower:
                return any(term in exp_lower for term in terms)
        
        return False
    
    def _calculate_cooccurrence(self, abbr: str, context: str) -> float:
        """Calculate co-occurrence score from matrix"""
        if not self.cooccurrence_matrix or abbr not in self.cooccurrence_matrix:
            return 0.0
        
        # Find other abbreviations in context
        context_abbrevs = re.findall(r'\b[A-Z]{2,10}\b', context)
        
        score = 0.0
        cooc_data = self.cooccurrence_matrix.get(abbr, {})
        
        for other_abbr in context_abbrevs:
            if other_abbr in cooc_data:
                score += cooc_data[other_abbr] * 0.1
        
        return min(score, 0.3)  # Cap contribution
    
    def _semantic_coherence(self, expansion: str, context: str) -> float:
        """Calculate semantic coherence between expansion and context"""
        # Extract semantic indicators from context
        indicators = {
            'clinical': ['patient', 'treatment', 'diagnosis', 'therapy'],
            'research': ['study', 'analysis', 'method', 'protocol'],
            'molecular': ['expression', 'pathway', 'binding', 'interaction']
        }
        
        context_lower = context.lower()
        exp_lower = expansion.lower()
        
        coherence = 0.0
        for category, terms in indicators.items():
            context_matches = sum(1 for t in terms if t in context_lower)
            if context_matches > 0:
                # Check if expansion fits this category
                if category == 'clinical' and any(
                    t in exp_lower for t in ['disease', 'syndrome', 'disorder']):
                    coherence += 0.2
                elif category == 'molecular' and any(
                    t in exp_lower for t in ['protein', 'enzyme', 'receptor']):
                    coherence += 0.2
        
        return min(coherence, 0.4)
    
    def _check_definition_pattern(self, abbr: str, expansion: str, context: str) -> float:
        """Check for definitional patterns"""
        patterns = [
            f"{expansion} ({abbr})",
            f"{expansion}, {abbr}",
            f"{abbr} ({expansion})",
            f"{abbr}, {expansion}"
        ]
        
        for pattern in patterns:
            if pattern in context:
                return 0.5  # Strong signal
        
        # Check proximity
        if abbr in context and expansion.lower() in context.lower():
            abbr_pos = context.find(abbr)
            exp_pos = context.lower().find(expansion.lower())
            distance = abs(abbr_pos - exp_pos)
            if distance < 50:  # Within 50 chars
                return 0.3
        
        return 0.0

# ============================================================================
# Token Normalization and Noise Filtering
# ============================================================================

class TokenNormalizer:
    """Token normalization with noise filtering"""
    
    @classmethod
    def normalize_tokens(cls, tokens: List[str]) -> List[str]:
        """Normalize and filter tokens"""
        normalized = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Canonicalize dashes
            token = re.sub(r'[‐‑‒–—―−]', '-', token)
            
            # Handle line-break hyphenation
            if token.endswith('-') and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token and next_token[0].islower():
                    merged = token[:-1] + next_token
                    normalized.append(merged)
                    i += 2
                    continue
            
            # Filter noise
            if cls._should_keep_token(token, normalized[-1] if normalized else None):
                normalized.append(token)
            
            i += 1
        
        return normalized
    
    @classmethod
    def _should_keep_token(cls, token: str, prev_token: str = None) -> bool:
        """Check if token should be kept"""
        
        # Noise patterns
        noise_patterns = [
            r'^\d{4,6}-[A-Za-z0-9]+$',  # DOI fragments
            r'^10\.\d{4,9}/',  # DOIs
            r'^(Fig|Figure|Table)[\s\-]?[S\d]+[A-Za-z]?$',  # References
            r'^v\d+\.\d+$',  # Versions
            r'^\d+[A-Z]$',  # Page markers
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, token, re.IGNORECASE):
                return False
        
        # Check for fragments
        if prev_token and re.match(r'^[A-Z]-\d+$', token) and prev_token.endswith('-'):
            return False
        
        return cls._is_valid_biomedical_token(token)
    
    @classmethod
    def _is_valid_biomedical_token(cls, token: str) -> bool:
        """Check if valid biomedical token"""
        if not token or len(token) < 2:
            return False
        
        if not any(c.isalpha() for c in token):
            return False
        
        # Biomedical compounds
        bio_patterns = [
            r'^[A-Z]{2,}(?:-[A-Z0-9]+)+$',  # ABC-123
            r'^[A-Z]\d+[a-z]?(?:-\d+)?$',  # C5b-9
            r'^COVID-\d+$',  # COVID-19
        ]
        
        for pattern in bio_patterns:
            if re.match(pattern, token):
                return True
        
        # Standard tokens
        if re.match(r'^[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]$', token):
            return True
        
        return False

# ============================================================================
# Cross-Entity Deduplication with Uncertainty
# ============================================================================

def cross_deduplicate_entities_enhanced(
    drugs: List[Dict], 
    diseases: List[Dict],
    abbreviations: List[Dict],
    doc_text: str = None,
    cooccurrence_matrix: Dict = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Enhanced cross-deduplication with uncertainty handling
    
    Args:
        drugs: Drug entities
        diseases: Disease entities
        abbreviations: Abbreviation entities
        doc_text: Full document text
        cooccurrence_matrix: Co-occurrence data
        
    Returns:
        Tuple of deduplicated entities
    """
    # Initialize disambiguator
    disambiguator = ContextAwareDisambiguator(cooccurrence_matrix)
    
    # Build abbreviation map
    abbrev_map = {a.get('abbreviation', '').upper(): a for a in abbreviations}
    
    # Infer document context
    doc_context = _infer_document_context(abbreviations, doc_text)
    
    # Process abbreviations with uncertainty handling
    corrected_abbreviations = []
    ambiguous_count = 0
    
    for abbrev in abbreviations:
        abbr = abbrev.get('abbreviation', '').upper()
        
        # Get context window
        context = _get_context_window(abbrev.get('position', 0), doc_text, window=300)
        
        # Get candidates
        candidates = _get_expansion_candidates(abbr, abbrev_map, doc_context)
        
        if candidates and len(candidates) > 1:
            # Disambiguate with uncertainty handling
            result = disambiguator.disambiguate_with_context(
                abbr, candidates, context, doc_context.get('primary_domain')
            )
            
            if result['status'] == 'ambiguous':
                abbrev['confidence'] = result['confidence']
                abbrev['disambiguation_status'] = 'ambiguous'
                abbrev['alternatives'] = result['alternatives']
                ambiguous_count += 1
            else:
                abbrev['expansion'] = result['expansion']
                abbrev['confidence'] = result['confidence']
                abbrev['disambiguation_status'] = result['status']
        
        corrected_abbreviations.append(abbrev)
    
    if ambiguous_count > 0:
        logger.info(f"Found {ambiguous_count} ambiguous abbreviations requiring manual review")
    
    # Process drugs with context filtering
    filtered_drugs = []
    for drug in drugs:
        # Check if drug makes sense in context
        drug_score = _score_entity_in_context(drug, doc_context, 'drug')
        
        if drug_score > 0.3:  # Threshold for keeping
            drug['context_score'] = drug_score
            filtered_drugs.append(drug)
        else:
            logger.debug(f"Filtering unlikely drug: {drug.get('name')}")
    
    # Process diseases with context enhancement
    enhanced_diseases = []
    seen_diseases = set()
    
    for disease in diseases:
        disease_name = disease.get('name', '')
        
        # Score disease in context
        disease_score = _score_entity_in_context(disease, doc_context, 'disease')
        disease['context_score'] = disease_score
        
        # Adjust confidence based on context
        if disease_score > 0.7:
            disease['confidence'] = min(1.0, disease.get('confidence', 0.5) + 0.1)
        
        # Deduplicate
        norm_name = normalize_entity_name(disease_name)
        if norm_name and norm_name not in seen_diseases:
            enhanced_diseases.append(disease)
            seen_diseases.add(norm_name)
    
    # Add diseases from disambiguated abbreviations
    for abbrev in corrected_abbreviations:
        if abbrev.get('disambiguation_status') == 'disambiguated':
            expansion = abbrev.get('expansion', '')
            if _is_disease_expansion(expansion):
                norm_expansion = normalize_entity_name(expansion)
                if norm_expansion not in seen_diseases:
                    enhanced_diseases.append({
                        'name': expansion,
                        'normalized': expansion,
                        'confidence': abbrev.get('confidence', 0.8),
                        'source': 'abbreviation_disambiguation',
                        'from_abbreviation': abbrev.get('abbreviation'),
                        'occurrences': abbrev.get('occurrences', 1)
                    })
                    seen_diseases.add(norm_expansion)
    
    return filtered_drugs, enhanced_diseases, corrected_abbreviations

def _get_context_window(position: int, text: str, window: int = 300) -> str:
    """Extract context window around position"""
    if not text:
        return ""
    start = max(0, position - window)
    end = min(len(text), position + window)
    return text[start:end]

def _get_expansion_candidates(abbr: str, abbrev_map: Dict, doc_context: Dict) -> List[str]:
    """Get candidate expansions for abbreviation"""
    candidates = []
    
    # From abbreviation data
    if abbr in abbrev_map:
        abbrev_data = abbrev_map[abbr]
        if abbrev_data.get('expansion'):
            candidates.append(abbrev_data['expansion'])
        if abbrev_data.get('alternatives'):
            candidates.extend(abbrev_data['alternatives'])
    
    # Could add KB lookups here
    
    return list(dict.fromkeys(candidates))  # Remove duplicates

def _infer_document_context(abbreviations: List[Dict], doc_text: str = None) -> Dict[str, Any]:
    """Infer document context from content"""
    context = {
        'primary_domain': None,
        'domains': defaultdict(float),
        'entity_types': defaultdict(float)
    }
    
    # Analyze abbreviations
    for abbrev in abbreviations:
        expansion = abbrev.get('expansion', '').lower()
        
        if any(term in expansion for term in ['disease', 'syndrome', 'disorder']):
            context['domains']['clinical'] += 1
            context['entity_types']['disease'] += 1
        elif any(term in expansion for term in ['drug', 'inhibitor', 'therapy']):
            context['domains']['pharmacological'] += 1
            context['entity_types']['drug'] += 1
        elif any(term in expansion for term in ['protein', 'enzyme', 'receptor']):
            context['domains']['molecular'] += 1
            context['entity_types']['protein'] += 1
    
    # Normalize
    total = sum(context['domains'].values())
    if total > 0:
        for domain in context['domains']:
            context['domains'][domain] /= total
        
        # Set primary domain
        context['primary_domain'] = max(context['domains'], key=context['domains'].get)
    
    return context

def _score_entity_in_context(entity: Dict, doc_context: Dict, entity_type: str) -> float:
    """Score how well entity fits document context"""
    score = 0.5  # Neutral baseline
    
    # Check if entity type matches document profile
    type_weight = doc_context['entity_types'].get(entity_type, 0)
    score += type_weight * 0.3
    
    # Additional scoring based on entity attributes
    if entity.get('confidence', 0) > 0.8:
        score += 0.1
    
    if entity.get('occurrences', 1) > 2:
        score += 0.1
    
    return min(1.0, score)

def _is_disease_expansion(expansion: str) -> bool:
    """Check if expansion represents a disease"""
    disease_terms = ['disease', 'syndrome', 'disorder', 'condition', 
                     'vasculitis', 'itis', 'osis', 'emia', 'pathy']
    exp_lower = expansion.lower()
    return any(term in exp_lower for term in disease_terms)

# ============================================================================
# Document-Level Consistency Pass
# ============================================================================

def apply_document_consistency(entities: List[Dict], entity_type: str = 'abbreviation') -> List[Dict]:
    """
    Apply document-level consistency for ambiguous entities
    
    Args:
        entities: List of entities
        entity_type: Type of entities
        
    Returns:
        Entities with consistency applied
    """
    if entity_type != 'abbreviation':
        return entities
    
    # Group by abbreviation
    abbrev_groups = defaultdict(list)
    for entity in entities:
        abbr = entity.get('abbreviation', '').upper()
        if abbr:
            abbrev_groups[abbr].append(entity)
    
    # Check consistency for each group
    for abbr, group in abbrev_groups.items():
        if len(group) < 2:
            continue
        
        # Count confident disambiguations
        confident_expansions = defaultdict(int)
        ambiguous_count = 0
        
        for entity in group:
            if entity.get('disambiguation_status') == 'disambiguated' and entity.get('confidence', 0) > 0.7:
                expansion = entity.get('expansion')
                if expansion:
                    confident_expansions[expansion] += 1
            elif entity.get('disambiguation_status') == 'ambiguous':
                ambiguous_count += 1
        
        # If 70% agree on one expansion, upgrade ambiguous ones
        if confident_expansions:
            total_confident = sum(confident_expansions.values())
            best_expansion = max(confident_expansions, key=confident_expansions.get)
            best_count = confident_expansions[best_expansion]
            
            if best_count / (total_confident + ambiguous_count) >= 0.7:
                # Apply to ambiguous instances
                for entity in group:
                    if entity.get('disambiguation_status') == 'ambiguous':
                        entity['expansion'] = best_expansion
                        entity['confidence'] = 0.75
                        entity['disambiguation_status'] = 'document_consistency'
                        logger.debug(f"Resolved {abbr} to {best_expansion} via document consistency")
    
    return entities

# ============================================================================
# Core Functions (kept from previous version)
# ============================================================================

def deduplicate_entities(entities: List[Dict], key_field: str = 'name') -> List[Dict]:
    """Remove duplicate entities keeping highest confidence"""
    if not entities:
        return []
    
    normalizer = TokenNormalizer()
    seen = {}
    
    for entity in entities:
        key = entity.get(key_field, '')
        if isinstance(key, str):
            tokens = key.split()
            normalized_tokens = normalizer.normalize_tokens(tokens)
            key = ' '.join(normalized_tokens).lower().strip()
        
        if not key:
            continue
            
        if key not in seen or entity.get('confidence', 0) > seen[key].get('confidence', 0):
            seen[key] = entity
    
    return list(seen.values())

def normalize_entity_name(name: str) -> str:
    """Normalize entity name for comparison"""
    if not name:
        return ""
    
    normalizer = TokenNormalizer()
    tokens = name.split()
    normalized_tokens = normalizer.normalize_tokens(tokens)
    normalized = ' '.join(normalized_tokens).lower()
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    # Remove common suffixes
    suffixes = [
        ' disease', ' syndrome', ' disorder',
        ' drug', ' medication', ' therapy'
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    
    return normalized.strip()

def consolidate_drug_entities(drugs: List[Dict]) -> List[Dict]:
    """Consolidate drugs by ingredient"""
    groups = defaultdict(list)
    
    for drug in drugs:
        canonical = drug.get('canonical_name') or drug.get('normalized') or drug.get('name')
        if canonical:
            key = normalize_entity_name(canonical)
            groups[key].append(drug)
    
    consolidated = []
    for key, group in groups.items():
        if len(group) == 1:
            consolidated.append(group[0])
        else:
            merged = {
                'name': group[0].get('name'),
                'normalized': key,
                'confidence': max(d.get('confidence', 0) for d in group),
                'occurrences': sum(d.get('occurrences', 1) for d in group),
                'aliases': list(set(d.get('name') for d in group)),
                'identifiers': {}
            }
            
            for drug in group:
                for id_type, id_value in drug.get('identifiers', {}).items():
                    if id_type not in merged['identifiers']:
                        merged['identifiers'][id_type] = id_value
            
            consolidated.append(merged)
    
    return consolidated

def calculate_entity_statistics(entities: List[Dict]) -> Dict[str, Any]:
    """Calculate entity statistics"""
    if not entities:
        return {
            'total': 0,
            'unique': 0,
            'avg_confidence': 0,
            'by_source': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
    
    by_source = defaultdict(int)
    for entity in entities:
        source = entity.get('source') or entity.get('detection_method', 'unknown')
        by_source[source] += 1
    
    high = sum(1 for e in entities if e.get('confidence', 0) >= 0.8)
    medium = sum(1 for e in entities if 0.5 <= e.get('confidence', 0) < 0.8)
    low = sum(1 for e in entities if e.get('confidence', 0) < 0.5)
    
    confidences = [e.get('confidence', 0) for e in entities]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total': len(entities),
        'unique': len(set(normalize_entity_name(e.get('name', '')) for e in entities)),
        'avg_confidence': round(avg_confidence, 3),
        'by_source': dict(by_source),
        'confidence_distribution': {'high': high, 'medium': medium, 'low': low},
        'total_occurrences': sum(e.get('occurrences', 1) for e in entities),
        'ambiguous_count': sum(1 for e in entities if e.get('disambiguation_status') == 'ambiguous')
    }

# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    'ContextAwareDisambiguator',
    'TokenNormalizer',
    'cross_deduplicate_entities_enhanced',
    'apply_document_consistency',
    'deduplicate_entities',
    'normalize_entity_name',
    'consolidate_drug_entities',
    'calculate_entity_statistics'
]