#!/usr/bin/env python3
"""
Reference Format Catalog - Citation Style Recognition & Parsing
================================================================
Location: corpus_metadata/document_utils/entity_citation_patterns.py
Version: 1.1.0 - IMPROVED PATTERNS & VALIDATION
Last Updated: 2025-10-08

CHANGELOG v1.1.0:
-----------------
✓ IMPROVED: More robust citation patterns with better capture groups
✓ IMPROVED: Enhanced journal abbreviation mappings
✓ IMPROVED: Better inline citation detection
✓ ADDED: Pattern compilation validation
✓ ADDED: Citation style confidence scoring
✓ FIXED: Regex anchoring for better matching
✓ ADDED: Support for more citation formats (PLoS, Frontiers, MDPI)
"""

import re
import logging
from typing import Dict, List, Optional, Pattern, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# REFERENCE SECTION PATTERNS - FINDING WHERE REFERENCES ARE
# ============================================================================

REFERENCE_SECTION_MARKERS = {
    'references_header': {
        'patterns': [
            r'^\s*(?:REFERENCES?|Bibliography|Works?\s+Cited|Literature\s+Cited)\s*$',
            r'^\s*\d+\.\s*REFERENCES?\s*$',
            r'^\s*References?\s*[:\-]?\s*$',
        ],
        'confidence': 0.98,
        'description': 'Standard reference section headers'
    },
    
    'numbered_list_start': {
        'patterns': [
            r'^\s*(?:1\.|1\)|\[1\])\s+[A-Z][a-z]+\s+[A-Z]{1,3}(?:,|\s)',
        ],
        'confidence': 0.90,
        'description': 'Start of numbered reference list'
    },
    
    'acknowledgments_boundary': {
        'patterns': [
            r'^\s*(?:ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS?)\s*$',
            r'^\s*(?:Funding|Financial\s+Disclosure)\s*$',
        ],
        'confidence': 0.85,
        'description': 'Section boundaries (references typically follow)'
    }
}

# ============================================================================
# CITATION STYLE PATTERNS - STRUCTURED REFERENCE FORMATS
# ============================================================================

CITATION_STYLES = {
    
    # ========================================================================
    # VANCOUVER STYLE (Numbered, Medical Journals)
    # ========================================================================
    
    'vancouver_full': {
        'pattern': r'(?P<number>\d+)\.\s+(?P<authors>[A-Z][a-z]+(?:\s+[A-Z]{1,3})?(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*(?:,?\s+et\s+al\.?)?)\.\s+(?P<title>[^.]+?)\.\s+(?P<journal>[^.]+?)\.\s+(?P<year>\d{4});?\s*(?P<volume>\d+)?(?:\((?P<issue>\d+)\))?:?\s*(?P<pages>[\d\-]+)?\.\s*(?:(?:doi:|PMID:|PMC)\s*[^\s.]+)?',
        'style': 'vancouver',
        'confidence': 0.95,
        'description': 'Full Vancouver format',
        'example': '1. Smith JA, Jones BC. Title of article. Journal Name. 2021;45(3):123-45. doi:10.1234/journal.2021.123',
        'components': ['number', 'authors', 'title', 'journal', 'year', 'volume', 'issue', 'pages']
    },
    
    'vancouver_simple': {
        'pattern': r'(?P<number>\d+)\.\s+(?P<authors>[A-Z][^\n.]+?)\.\s+(?P<journal>[^\n.]+?)\.\s+(?P<year>\d{4})',
        'style': 'vancouver',
        'confidence': 0.85,
        'description': 'Simplified Vancouver',
        'example': '1. Smith JA et al. Journal Name. 2021'
    },
    
    # ========================================================================
    # AMA STYLE (American Medical Association)
    # ========================================================================
    
    'ama_full': {
        'pattern': r'(?P<number>\d+)\.\s+(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\.\s+(?P<title>[^.]+?)\.\s+(?P<journal>[^.]+?)\.\s+(?P<year>\d{4});(?P<volume>\d+)\((?P<issue>\d+)\):(?P<pages>[\d\-]+)',
        'style': 'ama',
        'confidence': 0.95,
        'description': 'AMA citation format',
        'example': '1. Smith JA, Jones BC. Article title. JAMA. 2021;324(15):1234-1245',
        'components': ['number', 'authors', 'title', 'journal', 'year', 'volume', 'issue', 'pages']
    },
    
    # ========================================================================
    # NATURE/SCIENCE STYLE
    # ========================================================================
    
    'nature_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+,?\s+[A-Z]\.(?:\s*,?\s*[A-Z][a-z]+,?\s+[A-Z]\.)*(?:\s+et\s+al\.)?)\s+(?P<title>[^.]+?)\.\s+(?P<journal>[^\d]+?)\s+(?P<volume>\d+),\s+(?P<pages>[\d\-]+)\s+\((?P<year>\d{4})\)',
        'style': 'nature',
        'confidence': 0.93,
        'description': 'Nature journal format',
        'example': 'Smith, J. A., Jones, B. C. et al. Article title. Nature 589, 123-128 (2021)',
        'components': ['authors', 'title', 'journal', 'volume', 'pages', 'year']
    },
    
    'science_style': {
        'pattern': r'(?P<authors>[A-Z]\.\s+[A-Z][a-z]+(?:,\s+[A-Z]\.\s+[A-Z][a-z]+)*)\s*,\s*(?P<journal>[^,]+?)\s+(?P<volume>\d+),\s+(?P<pages>[\d\-]+)\s+\((?P<year>\d{4})\)',
        'style': 'science',
        'confidence': 0.93,
        'description': 'Science journal format',
        'example': 'J. A. Smith, B. C. Jones, Science 371, 123-128 (2021)',
        'components': ['authors', 'journal', 'volume', 'pages', 'year']
    },
    
    # ========================================================================
    # HARVARD STYLE (Author-Year)
    # ========================================================================
    
    'harvard_full': {
        'pattern': r'(?P<authors>[A-Z][a-z]+(?:,\s+[A-Z]\.[A-Z]?\.?)?(?:,\s+[A-Z][a-z]+(?:,\s+[A-Z]\.[A-Z]?\.?)?)*(?:\s+and\s+[A-Z][a-z]+)?)\s+\((?P<year>\d{4})\)\s+[\'"]?(?P<title>[^\'"]+?)[\'"]?,?\s+(?P<journal>[^,]+?),\s+(?:vol\.\s*)?(?P<volume>\d+)(?:\((?P<issue>\d+)\))?,\s+pp\.\s*(?P<pages>[\d\-]+)',
        'style': 'harvard',
        'confidence': 0.90,
        'description': 'Harvard author-year format',
        'example': "Smith, J.A., Jones, B.C. and Williams, C. (2021) 'Article title', Journal Name, vol. 45(3), pp. 123-145",
        'components': ['authors', 'year', 'title', 'journal', 'volume', 'issue', 'pages']
    },
    
    # ========================================================================
    # APA STYLE (7th Edition)
    # ========================================================================
    
    'apa_journal': {
        'pattern': r'(?P<authors>[A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)?(?:,\s+(?:&\s+)?[A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)?)*)\s+\((?P<year>\d{4})\)\.\s+(?P<title>[^.]+?)\.\s+(?P<journal>[^,]+?),\s+(?P<volume>\d+)(?:\((?P<issue>\d+)\))?,\s+(?P<pages>[\d\-]+)\.\s*(?:https?://doi\.org/(?P<doi>[\w./-]+))?',
        'style': 'apa',
        'confidence': 0.93,
        'description': 'APA 7th edition',
        'example': 'Smith, J. A., & Jones, B. C. (2021). Article title. Journal Name, 45(3), 123-145. https://doi.org/10.1234/journal.2021.123',
        'components': ['authors', 'year', 'title', 'journal', 'volume', 'issue', 'pages', 'doi']
    },
    
    # ========================================================================
    # CHICAGO STYLE
    # ========================================================================
    
    'chicago_author_date': {
        'pattern': r'(?P<authors>[A-Z][a-z]+,\s+[A-Z][a-z]+(?:,\s+and\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)\.\s+(?P<year>\d{4})\.\s+"(?P<title>[^"]+?)"\.\s+(?P<journal>[^,]+?)\s+(?P<volume>\d+)(?:\s+\((?P<issue>\d+)\))?\s*:\s*(?P<pages>[\d\-]+)',
        'style': 'chicago',
        'confidence': 0.88,
        'description': 'Chicago author-date',
        'example': 'Smith, John, and Barbara Jones. 2021. "Article title." Journal Name 45 (3): 123-145',
        'components': ['authors', 'year', 'title', 'journal', 'volume', 'issue', 'pages']
    },
    
    # ========================================================================
    # JOURNAL-SPECIFIC STYLES
    # ========================================================================
    
    'lancet_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\.\s+(?P<title>[^.]+?)\.\s+(?P<journal>(?:The\s+)?Lancet[^.]*?)\s+(?P<year>\d{4});\s+(?P<volume>\d+):\s+(?P<pages>[\d\-]+)',
        'style': 'lancet',
        'confidence': 0.95,
        'description': 'The Lancet format',
        'example': 'Smith JA, Jones BC. Article title. Lancet 2021; 397: 1234-45',
        'components': ['authors', 'title', 'journal', 'year', 'volume', 'pages']
    },
    
    'nejm_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\.\s+(?P<title>[^.]+?)\.\s+N\s+Engl\s+J\s+Med\s+(?P<year>\d{4});(?P<volume>\d+):(?P<pages>[\d\-]+)',
        'style': 'nejm',
        'confidence': 0.98,
        'description': 'NEJM format',
        'example': 'Smith JA, Jones BC. Article title. N Engl J Med 2021;384:1234-45',
        'components': ['authors', 'title', 'year', 'volume', 'pages']
    },
    
    'jama_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\.\s+(?P<title>[^.]+?)\.\s+JAMA\.\s+(?P<year>\d{4});(?P<volume>\d+)\((?P<issue>\d+)\):(?P<pages>[\d\-]+)',
        'style': 'jama',
        'confidence': 0.98,
        'description': 'JAMA format',
        'example': 'Smith JA, Jones BC. Article title. JAMA. 2021;325(15):1234-1245',
        'components': ['authors', 'title', 'year', 'volume', 'issue', 'pages']
    },
    
    'bmj_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\.\s+(?P<title>[^.]+?)\.\s+BMJ\s+(?P<year>\d{4});(?P<volume>\d+):(?P<article_id>[a-z]\d+)',
        'style': 'bmj',
        'confidence': 0.95,
        'description': 'BMJ format',
        'example': 'Smith JA, Jones BC. Article title. BMJ 2021;374:n1234',
        'components': ['authors', 'title', 'year', 'volume', 'article_id']
    },
    
    # ========================================================================
    # OPEN ACCESS STYLES
    # ========================================================================
    
    'plos_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\s+\((?P<year>\d{4})\)\s+(?P<title>[^.]+?)\.\s+PLoS\s+(?P<journal>\w+)\s+(?P<volume>\d+)\((?P<issue>\d+)\):\s*(?P<article_id>e\d+)',
        'style': 'plos',
        'confidence': 0.95,
        'description': 'PLoS format',
        'example': 'Smith JA, Jones BC (2021) Article title. PLoS ONE 16(5): e0123456',
        'components': ['authors', 'year', 'title', 'journal', 'volume', 'issue', 'article_id']
    },
    
    'frontiers_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*)\s+\((?P<year>\d{4})\)\.\s+(?P<title>[^.]+?)\.\s+Front\.\s+(?P<journal>[^.]+?)\s+(?P<volume>\d+):(?P<article_id>\d+)',
        'style': 'frontiers',
        'confidence': 0.93,
        'description': 'Frontiers format',
        'example': 'Smith JA, Jones BC (2021). Article title. Front. Immunol. 12:654321',
        'components': ['authors', 'year', 'title', 'journal', 'volume', 'article_id']
    },
    
    'mdpi_style': {
        'pattern': r'(?P<authors>[A-Z][a-z]+,\s+[A-Z]\.(?:;?\s+[A-Z][a-z]+,\s+[A-Z]\.)*)\s+(?P<title>[^.]+?)\.\s+(?P<journal>[^\d]+?)\s+(?P<year>\d{4}),\s+(?P<volume>\d+),\s+(?P<article_id>\d+)',
        'style': 'mdpi',
        'confidence': 0.90,
        'description': 'MDPI format',
        'example': 'Smith, J.A.; Jones, B.C. Article Title. Int. J. Mol. Sci. 2021, 22, 12345',
        'components': ['authors', 'title', 'journal', 'year', 'volume', 'article_id']
    }
}

# ============================================================================
# INLINE CITATION PATTERNS - REFERENCES WITHIN TEXT
# ============================================================================

INLINE_CITATION_PATTERNS = {
    'numbered_square': {
        'pattern': r'\[(?P<numbers>\d+(?:\s*[-–,]\s*\d+)*)\]',
        'style': 'vancouver',
        'confidence': 0.98,
        'description': 'Square bracket numbered',
        'examples': ['[1]', '[1-3]', '[1,2,5]', '[1-5,8,10-12]']
    },
    
    'numbered_superscript_marker': {
        'pattern': r'\^(?P<numbers>\d+(?:[-–,]\s*\d+)*)\^',
        'style': 'nature',
        'confidence': 0.90,
        'description': 'Superscript marker format',
        'examples': ['^1^', '^1-3^', '^1,2,5^']
    },
    
    'author_year_parenthesis': {
        'pattern': r'\((?P<authors>[A-Z][a-z]+(?:\s+et\s+al\.?)?),?\s+(?P<year>\d{4}[a-z]?)\)',
        'style': 'harvard',
        'confidence': 0.95,
        'description': 'Author-year in parentheses',
        'examples': ['(Smith, 2021)', '(Smith et al., 2021)', '(Smith 2021a)']
    },
    
    'author_year_narrative': {
        'pattern': r'(?P<authors>[A-Z][a-z]+(?:\s+et\s+al\.?)?)\s+\((?P<year>\d{4}[a-z]?)\)',
        'style': 'apa',
        'confidence': 0.93,
        'description': 'Narrative citation',
        'examples': ['Smith (2021)', 'Smith et al. (2021)', 'Jones (2020a)']
    },
    
    'multiple_authors_year': {
        'pattern': r'\((?P<citations>(?:[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?;\s*)+[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?)\)',
        'style': 'apa',
        'confidence': 0.90,
        'description': 'Multiple author-year',
        'examples': ['(Smith, 2021; Jones, 2020; Williams et al., 2019)']
    },
    
    'author_year_semicolon': {
        'pattern': r'\((?P<author1>[A-Z][a-z]+)(?:\s+et\s+al\.)?\s+(?P<year1>\d{4}[a-z]?);(?:\s+(?P<author2>[A-Z][a-z]+)(?:\s+et\s+al\.)?\s+(?P<year2>\d{4}[a-z]?))+\)',
        'style': 'apa',
        'confidence': 0.92,
        'description': 'Semicolon-separated citations',
        'examples': ['(Smith 2021; Jones 2020)']
    }
}

# ============================================================================
# REFERENCE COMPONENT EXTRACTORS
# ============================================================================

REFERENCE_COMPONENTS = {
    'authors': {
        'patterns': [
            r'^(?P<authors>[A-Z][a-z]+\s+[A-Z]{1,3}(?:,\s+[A-Z][a-z]+\s+[A-Z]{1,3})*(?:,?\s+et\s+al\.?)?)',
            r'^(?P<authors>[A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)?(?:,\s+(?:&\s+)?[A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)?)*)',
        ],
        'description': 'Extract author list'
    },
    
    'title': {
        'patterns': [
            r'(?P<title>["\'"][^"\']+["\'])',
            r'(?P<title>[A-Z][^.]+)\.\s+[A-Z]',
        ],
        'description': 'Extract article title'
    },
    
    'journal': {
        'patterns': [
            r'(?P<journal>(?:N\s+Engl\s+J\s+Med|JAMA|Lancet|Nature|Science|BMJ|Cell|Neuron)[^.]*)',
            r'(?P<journal>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\d{4}',
        ],
        'description': 'Extract journal name'
    },
    
    'year': {
        'patterns': [
            r'(?P<year>(?:19|20)\d{2})',
        ],
        'description': 'Extract publication year'
    },
    
    'volume_issue_pages': {
        'patterns': [
            r'(?P<volume>\d+)\((?P<issue>\d+)\):(?P<pages>[\d\-]+)',
            r'(?P<volume>\d+):(?P<pages>[\d\-]+)',
            r';(?P<volume>\d+):(?P<pages>[\d\-]+)',
        ],
        'description': 'Extract volume, issue, pages'
    },
    
    'doi': {
        'patterns': [
            r'(?:doi:|DOI:)?\s*(10\.\d{4,}/[^\s]+)',
        ],
        'description': 'Extract DOI'
    },
    
    'pmid': {
        'patterns': [
            r'PMID:?\s*(\d{7,8})',
        ],
        'description': 'Extract PMID'
    },
    
    'pmc': {
        'patterns': [
            r'(?:PMCID:?\s*)?PMC(\d{6,8})',
        ],
        'description': 'Extract PMC ID'
    }
}

# ============================================================================
# JOURNAL ABBREVIATIONS - EXPANDED
# ============================================================================

JOURNAL_ABBREVIATIONS = {
    # High-Impact General
    'N Engl J Med': 'New England Journal of Medicine',
    'NEJM': 'New England Journal of Medicine',
    'Lancet': 'The Lancet',
    'JAMA': 'Journal of the American Medical Association',
    'BMJ': 'British Medical Journal',
    'Ann Intern Med': 'Annals of Internal Medicine',
    
    # Nature Portfolio
    'Nat Med': 'Nature Medicine',
    'Nat Genet': 'Nature Genetics',
    'Nat Biotechnol': 'Nature Biotechnology',
    'Nat Immunol': 'Nature Immunology',
    'Nat Rev Drug Discov': 'Nature Reviews Drug Discovery',
    'Nat Commun': 'Nature Communications',
    
    # Cell Press
    'Cell': 'Cell',
    'Neuron': 'Neuron',
    'Immunity': 'Immunity',
    'Cancer Cell': 'Cancer Cell',
    'Mol Cell': 'Molecular Cell',
    'Cell Rep': 'Cell Reports',
    
    # Specialty
    'J Clin Oncol': 'Journal of Clinical Oncology',
    'Blood': 'Blood',
    'Circulation': 'Circulation',
    'Diabetes': 'Diabetes',
    'Kidney Int': 'Kidney International',
    'Am J Respir Crit Care Med': 'American Journal of Respiratory and Critical Care Medicine',
    'Clin Pharmacol Ther': 'Clinical Pharmacology & Therapeutics',
    'Rheumatology': 'Rheumatology',
    'Ann Rheum Dis': 'Annals of the Rheumatic Diseases',
    
    # Open Access
    'PLoS One': 'PLOS ONE',
    'PLoS Med': 'PLOS Medicine',
    'Sci Rep': 'Scientific Reports',
    'Front Immunol': 'Frontiers in Immunology',
    'Front Med': 'Frontiers in Medicine',
    'Int J Mol Sci': 'International Journal of Molecular Sciences',
    'Cells': 'Cells',
    'Cancers': 'Cancers',
    'Biomolecules': 'Biomolecules'
}

# ============================================================================
# CONFIDENCE SCORING RULES
# ============================================================================

CONFIDENCE_SCORING = {
    'has_doi': +0.15,
    'has_pmid': +0.15,
    'has_pmc': +0.10,
    'has_volume_pages': +0.10,
    'recognized_journal': +0.10,
    'complete_author_list': +0.08,
    'has_title': +0.08,
    'in_reference_section': +0.10,
    'numbered_format': +0.05,
    'has_year': +0.05
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_citation_style(text: str, sample_size: int = 10000) -> Dict[str, float]:
    """
    Detect which citation style is used in the text
    
    Args:
        text: Input text
        sample_size: Number of characters to analyze
        
    Returns:
        Dictionary of style -> confidence scores
    """
    style_scores: defaultdict = defaultdict(float)
    sample = text[:sample_size]
    
    for style_name, style_config in CITATION_STYLES.items():
        pattern = style_config['pattern']
        
        try:
            compiled_pattern = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            matches = compiled_pattern.findall(sample)
            
            if matches:
                base_confidence = style_config['confidence']
                match_score = len(matches) * base_confidence
                style_scores[style_config['style']] += match_score
        except re.error as e:
            logger.warning(f"Pattern compilation failed for {style_name}: {e}")
            continue
    
    return dict(style_scores)

def find_reference_section(text: str) -> Optional[Tuple[int, int]]:
    """
    Find the reference section in document
    
    Args:
        text: Input text
        
    Returns:
        (start_pos, end_pos) tuple or None
    """
    for marker_name, marker_config in REFERENCE_SECTION_MARKERS.items():
        for pattern in marker_config['patterns']:
            try:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    start_pos = match.end()
                    
                    # Find end
                    end_patterns = [
                        r'^\s*(?:SUPPLEMENTARY|APPENDIX|SUPPORTING\s+INFORMATION)\s*$',
                        r'^\s*Figure\s+\d+\.',
                        r'^\s*Table\s+\d+\.',
                    ]
                    
                    end_pos = len(text)
                    for end_pattern in end_patterns:
                        end_match = re.search(end_pattern, text[start_pos:], re.MULTILINE | re.IGNORECASE)
                        if end_match:
                            end_pos = start_pos + end_match.start()
                            break
                    
                    return (start_pos, end_pos)
            except re.error as e:
                logger.warning(f"Pattern search failed for {marker_name}: {e}")
                continue
    
    return None

def extract_inline_citations(text: str) -> List[Dict]:
    """
    Extract all inline citations from text
    
    Args:
        text: Input text
        
    Returns:
        List of citation dictionaries
    """
    citations = []
    
    for cite_type, cite_config in INLINE_CITATION_PATTERNS.items():
        pattern = cite_config['pattern']
        
        try:
            for match in re.finditer(pattern, text):
                citation = {
                    'type': cite_type,
                    'text': match.group(0),
                    'style': cite_config['style'],
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': cite_config['confidence']
                }
                
                # Extract captured groups
                citation.update(match.groupdict())
                citations.append(citation)
        except re.error as e:
            logger.warning(f"Citation extraction failed for {cite_type}: {e}")
            continue
    
    return citations

def normalize_journal_name(abbrev: str) -> Optional[str]:
    """
    Normalize journal abbreviation to full name
    
    Args:
        abbrev: Journal abbreviation
        
    Returns:
        Full journal name or None
    """
    # Direct lookup
    full_name = JOURNAL_ABBREVIATIONS.get(abbrev)
    if full_name:
        return full_name
    
    # Try case-insensitive
    for key, value in JOURNAL_ABBREVIATIONS.items():
        if key.lower() == abbrev.lower():
            return value
    
    return None

def get_citation_style_stats() -> Dict[str, int]:
    """Get count of patterns by style"""
    styles: defaultdict = defaultdict(int)
    for config in CITATION_STYLES.values():
        styles[config['style']] += 1
    return dict(styles)

def validate_citation_patterns() -> Dict[str, List[str]]:
    """
    Validate all citation patterns
    
    Returns:
        Dictionary of pattern_name -> list of errors
    """
    validation_errors = {}
    
    for pattern_name, pattern_config in CITATION_STYLES.items():
        errors = []
        pattern = pattern_config['pattern']
        example = pattern_config.get('example')
        
        # Try to compile
        try:
            compiled = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        except re.error as e:
            errors.append(f"Pattern compilation failed: {e}")
            validation_errors[pattern_name] = errors
            continue
        
        # Test against example
        if example:
            if not compiled.search(example):
                errors.append(f"Example does not match pattern: {example}")
        
        if errors:
            validation_errors[pattern_name] = errors
    
    return validation_errors

def parse_reference(reference_text: str) -> Optional[Dict]:
    """
    Parse a reference string into components
    
    Args:
        reference_text: Reference citation text
        
    Returns:
        Dictionary of parsed components or None
    """
    for style_name, style_config in CITATION_STYLES.items():
        pattern = style_config['pattern']
        
        try:
            match = re.search(pattern, reference_text, re.MULTILINE | re.IGNORECASE)
            if match:
                return {
                    'style': style_config['style'],
                    'confidence': style_config['confidence'],
                    'components': match.groupdict()
                }
        except re.error:
            continue
    
    return None

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = '1.1.0'
__author__ = 'Biomedical Entity Extraction System'
__all__ = [
    'REFERENCE_SECTION_MARKERS',
    'CITATION_STYLES',
    'INLINE_CITATION_PATTERNS',
    'REFERENCE_COMPONENTS',
    'JOURNAL_ABBREVIATIONS',
    'CONFIDENCE_SCORING',
    'detect_citation_style',
    'find_reference_section',
    'extract_inline_citations',
    'normalize_journal_name',
    'get_citation_style_stats',
    'validate_citation_patterns',
    'parse_reference'
]

if __name__ == "__main__":
    print("=" * 80)
    print("REFERENCE FORMAT CATALOG - v1.1.0 (IMPROVED)")
    print("=" * 80)
    print(f"Citation styles: {len(CITATION_STYLES)}")
    print(f"Inline patterns: {len(INLINE_CITATION_PATTERNS)}")
    print(f"Journal abbreviations: {len(JOURNAL_ABBREVIATIONS)}")
    
    print("\nStyles by type:")
    for style, count in sorted(get_citation_style_stats().items()):
        print(f"  {style:15s}: {count:2d}")
    
    # Validate patterns
    print("\n" + "=" * 80)
    print("PATTERN VALIDATION")
    print("=" * 80)
    validation_errors = validate_citation_patterns()
    
    if not validation_errors:
        print("✅ All citation patterns validated successfully!")
    else:
        print(f"❌ Found {len(validation_errors)} patterns with issues:\n")
        for key, errors in validation_errors.items():
            print(f"  {key}:")
            for error in errors:
                print(f"    - {error}")
    
    print("=" * 80)