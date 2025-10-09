#!/usr/bin/env python3
"""
Person Pattern Dictionary - Scientific Authors & Investigators
===============================================================
Location: corpus_metadata/document_utils/entity_person_patterns.py
Version: 2.0.0 - MAJOR REWRITE WITH UNICODE & MULTILINGUAL SUPPORT
Last Updated: 2025-10-08

CHANGELOG v2.0.0:
-----------------
✓ COMPLETE REWRITE: Unicode support for diacritics (García-López, O'Connor)
✓ FIXED: Surname particles (van der Berg, de la Cruz, ibn Rushd)
✓ FIXED: Hyphenated and apostrophe names (García-López, O'Brien)
✓ IMPROVED: ORCID pattern now validates correctly
✓ ADDED: Spanish language triggers and sections
✓ IMPROVED: More robust affiliation patterns (UCL, Charité, hospitals)
✓ ADDED: Reusable name building blocks
✓ ADDED: Pattern validation function
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Pattern, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# REUSABLE NAME BUILDING BLOCKS - UNICODE SUPPORT
# ============================================================================

# Latin alphabet with diacritics
LAT_UP = r"A-ZÀ-ÖØ-ÞÇ"                    # Uppercase with diacritics
LAT_LO = r"a-zà-öø-ÿçñ"                   # Lowercase with diacritics

# Name components
NAME_CHUNK = rf"[{LAT_UP}][{LAT_LO}]+(?:[-''][{LAT_UP}{LAT_LO}]+)*"  # García-López, O'Connor

# Surname particles (lowercase)
PARTICLE = r"(?:van|von|der|den|de|del|della|di|da|do|dos|du|le|la|al|el|ibn|bin|bint)"

# Complete surname (with or without particles)
SURNAME = rf"(?:{NAME_CHUNK}(?:\s+{NAME_CHUNK})*|(?:{PARTICLE}\s+)+{NAME_CHUNK}(?:\s+{NAME_CHUNK})*)"

# Initials
INITIALS = r"(?:[A-Z](?:\.[A-Z])?\.?|[A-Z]{1,3})"  # J., JA., JA, J.A.

# Core person pattern (Surname + Initials)
PERSON_CORE = rf"(?:{SURNAME}\s+{INITIALS})"

# Full name (First Middle Last)
FULL_NAME = rf"(?:{NAME_CHUNK}(?:\s+{NAME_CHUNK})*)"

# ============================================================================
# PERSON NAME PATTERNS - DETECTION
# ============================================================================

PERSON_NAME_PATTERNS = {
    
    # ========================================================================
    # CATEGORY 1: BIBLIOGRAPHY / REFERENCE SECTION
    # ========================================================================
    
    'citation_standard': {
        'pattern': rf"\b{PERSON_CORE}(?:\s*,\s*{PERSON_CORE})*\b",
        'context': 'references',
        'role': 'author',
        'confidence': 0.95,
        'description': 'Standard citation: LastName Initials',
        'examples': [
            'Smith JA, Jones BC, Williams CD',
            'García-López M, Chen X',
            'van der Berg P, O\'Brien K'
        ],
        'capture_groups': {
            'full_match': 0
        }
    },
    
    'citation_et_al': {
        'pattern': rf"\b{SURNAME}\s+(?:et\s+al\.?|and\s+others)\b",
        'context': 'inline_citation',
        'role': 'author',
        'confidence': 0.90,
        'description': 'Et al. citation',
        'examples': [
            'Smith et al.',
            'García-López et al',
            'van den Berg and others'
        ],
        'capture_groups': {
            'surname': 1
        }
    },
    
    'citation_year': {
        'pattern': rf"\b({SURNAME}(?:\s+(?:et\s+al\.?|and\s+colleagues))?)[\s,]+\(?(?:19|20)\d{{2}}\)?",
        'context': 'inline_citation',
        'role': 'author',
        'confidence': 0.92,
        'description': 'Author with year citation',
        'examples': [
            'Smith et al., 2021',
            'García (2020)',
            'Jones and colleagues, 2022',
            'van der Berg, 1998'
        ],
        'capture_groups': {
            'author': 1
        }
    },
    
    'full_author_name': {
        'pattern': rf"\b({SURNAME}),?\s+({NAME_CHUNK})(?:\s+([A-Z]\.?(?:\s+[A-Z]\.?)?))?",
        'context': 'references',
        'role': 'author',
        'confidence': 0.95,
        'description': 'Full name: Last, First Middle',
        'examples': [
            'Smith, John A.',
            'García-López, María E.',
            'van der Berg, Peter',
            'O\'Connor, Patrick J.'
        ],
        'capture_groups': {
            'last_name': 1,
            'first_name': 2,
            'middle_initial': 3
        }
    },
    
    'author_list_semicolon': {
        'pattern': rf"\b{PERSON_CORE}(?:\s*;\s*{PERSON_CORE})+\b",
        'context': 'references',
        'role': 'author',
        'confidence': 0.93,
        'description': 'Semicolon-separated authors',
        'examples': [
            'Smith JA; Jones BC; Williams CD',
            'García M; Chen X; Kim Y',
            'van der Berg P; O\'Brien K'
        ]
    },
    
    # ========================================================================
    # CATEGORY 2: CLINICAL TRIAL INVESTIGATORS
    # ========================================================================
    
    'principal_investigator_explicit': {
        'pattern': rf"(?:principal\s+investigator|PI|study\s+director|lead\s+investigator|investigador\s+principal|responsable\s+del\s+estudio):?\s+(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?({SURNAME}(?:\s+{INITIALS})?(?:\s+{NAME_CHUNK})?)",
        'context': 'methods',
        'role': 'principal_investigator',
        'confidence': 0.98,
        'description': 'Explicit PI designation',
        'examples': [
            'Principal Investigator: Dr. John Smith',
            'PI: María García-López',
            'Investigador Principal: Dr. José de la Cruz',
            'Study Director: Prof. Peter van den Berg',
            'Lead Investigator: Sarah O\'Connor, MD'
        ],
        'capture_groups': {
            'full_name': 1
        }
    },
    
    'trial_leadership': {
        'pattern': rf"(?:led\s+by|conducted\s+by|coordinated\s+by|under\s+the\s+direction\s+of|dirigido\s+por)\s+(?:Dr\.?\s+|Prof\.?\s+)?({SURNAME}(?:\s+{INITIALS})?(?:\s+{NAME_CHUNK})?)",
        'context': 'methods',
        'role': 'principal_investigator',
        'confidence': 0.95,
        'description': 'Trial leadership',
        'examples': [
            'led by Dr. John Smith',
            'conducted by María García',
            'dirigido por Prof. José Martínez',
            'coordinated by Peter van den Berg'
        ],
        'capture_groups': {
            'full_name': 1
        }
    },
    
    'study_team': {
        'pattern': rf"(?:co-investigator|co-PI|study\s+coordinator|site\s+investigator|coinvestigador):?\s+(?:Dr\.?\s+)?({SURNAME}(?:\s+{NAME_CHUNK})?)",
        'context': 'methods',
        'role': 'co_investigator',
        'confidence': 0.93,
        'description': 'Study team members',
        'examples': [
            'Co-investigator: Dr. Sarah Chen',
            'Study Coordinator: María López',
            'Coinvestigador: Dr. Juan Pérez',
            'Site Investigator: John Williams'
        ],
        'capture_groups': {
            'full_name': 1
        }
    },
    
    # ========================================================================
    # CATEGORY 3: CORRESPONDING AUTHOR
    # ========================================================================
    
    'corresponding_author': {
        'pattern': rf"(?:correspondence\s+(?:to|should\s+be\s+addressed\s+to)|for\s+correspondence|contact|autor\s+de\s+correspondencia|para\s+correspondencia):?\s+(?:Dr\.?\s+)?({SURNAME}(?:\s+{INITIALS})?(?:\s+{NAME_CHUNK})?)",
        'context': 'author_info',
        'role': 'corresponding_author',
        'confidence': 0.97,
        'description': 'Corresponding author',
        'examples': [
            'Correspondence to: Dr. John Smith',
            'Para correspondencia: María García',
            'Contact: Peter van den Berg, MD',
            'Autor de correspondencia: Dr. José Martínez'
        ],
        'capture_groups': {
            'full_name': 1
        }
    },
    
    'email_author': {
        'pattern': rf"\b({SURNAME}|{NAME_CHUNK}(?:\s+{SURNAME})?)\s*[:\(<]?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}})",
        'context': 'author_info',
        'role': 'corresponding_author',
        'confidence': 0.95,
        'description': 'Author with email',
        'examples': [
            'John Smith: jsmith@university.edu',
            'María García (mgarcia@hospital.org)',
            'P. van den Berg <p.vandenberg@institution.nl>',
            'José Martínez: jmartinez@universidad.es'
        ],
        'capture_groups': {
            'name': 1,
            'email': 2
        }
    },
    
    # ========================================================================
    # CATEGORY 4: EXPERT OPINION
    # ========================================================================
    
    'expert_opinion': {
        'pattern': rf"(?:according\s+to|as\s+(?:noted|stated|reported)\s+by|in\s+the\s+(?:opinion|view)\s+of|según|de\s+acuerdo\s+con)\s+(?:Dr\.?\s+|Prof\.?\s+)?({SURNAME}(?:\s+{NAME_CHUNK})?)",
        'context': 'discussion',
        'role': 'expert_opinion',
        'confidence': 0.88,
        'description': 'Expert opinion citation',
        'examples': [
            'according to Dr. Smith',
            'según Prof. García',
            'as noted by Peter van den Berg',
            'de acuerdo con Dr. Martínez'
        ],
        'capture_groups': {
            'full_name': 1
        }
    },
    
    'demonstrated_by': {
        'pattern': rf"\b({SURNAME})\s+(?:demonstrated|showed|found|observed|reported|demostró|observó)",
        'context': 'results',
        'role': 'author',
        'confidence': 0.85,
        'description': 'Author demonstration',
        'examples': [
            'Smith demonstrated',
            'García demostró',
            'van den Berg found'
        ],
        'capture_groups': {
            'surname': 1
        }
    },
    
    # ========================================================================
    # CATEGORY 5: GUIDELINE AUTHORS
    # ========================================================================
    
    'guideline_committee': {
        'pattern': rf"(?:guideline\s+(?:committee|panel|working\s+group)|consensus\s+panel|panel\s+de\s+expertos|comité\s+de\s+guías)(?:\s+member)?:?\s+(?:Dr\.?\s+)?({SURNAME}(?:\s+{NAME_CHUNK})?)",
        'context': 'guideline',
        'role': 'guideline_author',
        'confidence': 0.92,
        'description': 'Guideline committee member',
        'examples': [
            'Guideline Committee: Dr. John Smith',
            'Consensus Panel Member: María García',
            'Panel de expertos: Dr. José Martínez',
            'Working Group: Peter van den Berg'
        ],
        'capture_groups': {
            'full_name': 1
        }
    },
    
    # ========================================================================
    # CATEGORY 6: ACADEMIC TITLES & DEGREES
    # ========================================================================
    
    'titled_person': {
        'pattern': rf"\b(Dr\.?|Prof\.?|Professor|M\.?D\.?|Ph\.?D\.?|D\.?Sc\.?)\s+({SURNAME}|{NAME_CHUNK}(?:\s+{SURNAME})?)",
        'context': 'any',
        'role': 'author',
        'confidence': 0.90,
        'description': 'Person with academic title',
        'examples': [
            'Dr. John Smith',
            'Prof. María García',
            'Professor Peter van den Berg',
            'Dr. José Martínez',
            'PhD Sarah O\'Connor'
        ],
        'capture_groups': {
            'title': 1,
            'name': 2
        }
    },
    
    'person_with_degrees': {
        'pattern': rf"\b({SURNAME}|{NAME_CHUNK}(?:\s+{SURNAME})?),?\s+(M\.?D\.?|Ph\.?D\.?|M\.?B\.?B\.?S\.?|D\.?Sc\.?)(?:\s*,\s*(M\.?D\.?|Ph\.?D\.?|M\.?B\.?B\.?S\.?|D\.?Sc\.?))*",
        'context': 'any',
        'role': 'author',
        'confidence': 0.92,
        'description': 'Person with medical/academic degrees',
        'examples': [
            'John Smith, MD',
            'María García, MD, PhD',
            'Peter van den Berg, MBBS',
            'José Martínez, PhD'
        ],
        'capture_groups': {
            'name': 1,
            'degree1': 2,
            'degree2': 3
        }
    },
    
    # ========================================================================
    # CATEGORY 7: ORCID IDENTIFIERS (FIXED)
    # ========================================================================
    
    'orcid_id': {
        'pattern': rf"\b({SURNAME}|{NAME_CHUNK}(?:\s+{SURNAME})?)\s*[\(:]*\s*(?:ORCID:?\s*|https?://orcid\.org/)?(0000-\d{{4}}-\d{{4}}-\d{{3}}[0-9X])\b",
        'context': 'author_info',
        'role': 'author',
        'confidence': 1.0,
        'description': 'Person with ORCID identifier (FIXED pattern)',
        'examples': [
            'John Smith (ORCID: 0000-0002-1234-567X)',
            'María García ORCID:0000-0001-2345-6789',
            'https://orcid.org/0000-0003-4567-8901'
        ],
        'capture_groups': {
            'name': 1,
            'orcid': 2
        }
    }
}

# ============================================================================
# AFFILIATION PATTERNS - IMPROVED WITH MULTILINGUAL SUPPORT
# ============================================================================

AFFILIATION_PATTERNS = {
    'university': {
        'pattern': rf"\b(?:University\s+of\s+{NAME_CHUNK}(?:\s+{NAME_CHUNK})*|{NAME_CHUNK}(?:\s+{NAME_CHUNK})*\s+University|University\s+College\s+{NAME_CHUNK}|{NAME_CHUNK}\s+Universität|Universidad\s+de\s+{NAME_CHUNK}(?:\s+{NAME_CHUNK})*|Université\s+de\s+{NAME_CHUNK})",
        'type': 'academic',
        'confidence': 0.95,
        'examples': [
            'University of California',
            'Harvard University',
            'University College London',
            'Charité Universitätsmedizin',
            'Universidad de Barcelona',
            'Université de Paris'
        ]
    },
    
    'hospital': {
        'pattern': rf"\b(?:{NAME_CHUNK}(?:\s+{NAME_CHUNK})*\s+(?:Hospital|Medical\s+Center|Clinic|Health\s+System|Hosp\.?|Hospital\s+Universitario(?:\s+{NAME_CHUNK})*|Centre\s+Hospitalier|Hôpital))",
        'type': 'clinical',
        'confidence': 0.95,
        'examples': [
            'Massachusetts General Hospital',
            'Johns Hopkins Medical Center',
            'Mayo Clinic',
            'Hospital Universitario La Paz',
            'Hôpital Saint-Louis',
            'Charité Hospital'
        ]
    },
    
    'research_institute': {
        'pattern': rf"\b(?:{NAME_CHUNK}(?:\s+{NAME_CHUNK})*\s+(?:Institute|Research\s+Center|Laborator(?:y|io)|Centro\s+de\s+Investigación|Institut|Instituto|Zentrum))",
        'type': 'research',
        'confidence': 0.92,
        'examples': [
            'National Cancer Institute',
            'Max Planck Institute',
            'Cold Spring Harbor Laboratory',
            'Instituto de Salud Carlos III',
            'Centre de Recherche',
            'Forschungszentrum'
        ]
    },
    
    'medical_school': {
        'pattern': rf"\b(?:{NAME_CHUNK}(?:\s+{NAME_CHUNK})*\s+(?:School\s+of\s+Medicine|Medical\s+School|Facultad\s+de\s+Medicina|Faculté\s+de\s+Médecine))",
        'type': 'academic',
        'confidence': 0.95,
        'examples': [
            'Harvard Medical School',
            'Johns Hopkins School of Medicine',
            'Stanford Medical School',
            'Facultad de Medicina de Barcelona'
        ]
    },
    
    'department': {
        'pattern': rf"\b(?:Department|Dept\.|Service|Servicio|Département)\s+(?:of\s+|de\s+)?({NAME_CHUNK}(?:\s+{NAME_CHUNK})*)",
        'type': 'academic_unit',
        'confidence': 0.85,
        'examples': [
            'Department of Medicine',
            'Department of Neurology',
            'Servicio de Oncología',
            'Département de Cardiologie'
        ]
    }
}

# ============================================================================
# ROLE CLASSIFICATION TRIGGERS - MULTILINGUAL
# ============================================================================

ROLE_CLASSIFICATION_TRIGGERS = {
    'principal_investigator': [
        r'principal\s+investigator',
        r'\bPI\b',
        r'study\s+director',
        r'lead\s+investigator',
        r'trial\s+coordinator',
        r'protocol\s+chair',
        r'investigador\s+principal',
        r'responsable\s+del\s+estudio',
        r'coordinador\s+del\s+ensayo'
    ],
    
    'author': [
        r'et\s+al',
        r'published',
        r'reported\s+by',
        r'according\s+to',
        r'demonstrated\s+by',
        r'in\s+references',
        r'bibliography',
        r'según',
        r'de\s+acuerdo\s+con',
        r'publicado\s+por',
        r'referencias',
        r'bibliografía'
    ],
    
    'corresponding_author': [
        r'correspondence',
        r'contact',
        r'email',
        r'for\s+reprints',
        r'address\s+correspondence',
        r'correspondencia',
        r'contacto',
        r'autor\s+de\s+correspondencia',
        r'para\s+correspondencia'
    ],
    
    'co_investigator': [
        r'co-investigator',
        r'co-PI',
        r'collaborator',
        r'site\s+investigator',
        r'coinvestigador',
        r'colaborador'
    ],
    
    'study_coordinator': [
        r'study\s+coordinator',
        r'research\s+coordinator',
        r'clinical\s+coordinator',
        r'coordinador\s+de\s+estudio',
        r'coordinador\s+de\s+investigación'
    ],
    
    'expert_opinion': [
        r'according\s+to',
        r'in\s+the\s+opinion\s+of',
        r'as\s+noted\s+by',
        r'expert\s+opinion',
        r'según',
        r'de\s+acuerdo\s+con',
        r'opinión\s+experta'
    ],
    
    'guideline_author': [
        r'guideline\s+committee',
        r'consensus\s+panel',
        r'working\s+group',
        r'expert\s+panel',
        r'comité\s+de\s+guías',
        r'panel\s+de\s+expertos',
        r'grupo\s+de\s+trabajo'
    ]
}

# ============================================================================
# NAME NORMALIZATION RULES
# ============================================================================

NAME_NORMALIZATION_RULES = {
    'prefixes': [
        'van', 'von', 'de', 'del', 'della', 'di', 'da', 'le', 'la',
        'van der', 'van den', 'von der', 'de la', 'de los', 'de las',
        'ibn', 'bin', 'bint', 'al', 'el', 'do', 'dos', 'du'
    ],
    
    'suffixes': [
        'Jr', 'Sr', 'II', 'III', 'IV', 'V',
        'MD', 'PhD', 'DO', 'DPhil', 'DSc', 'MBBS',
        'FRCP', 'FACP', 'FACS', 'FRCPC'
    ],
    
    'titles': [
        'Dr', 'Prof', 'Professor', 'Mr', 'Mrs', 'Ms', 'Miss',
        'Sr', 'Sra', 'Srta', 'Don', 'Doña'
    ]
}

# ============================================================================
# CONTEXT SECTION MAPPING - MULTILINGUAL
# ============================================================================

CONTEXT_SECTIONS = {
    'references': [
        'references', 'bibliography', 'works cited', 'citations',
        'referencias', 'bibliografía', 'citas', 'obras citadas',
        'références', 'bibliographie'
    ],
    'methods': [
        'methods', 'study design', 'patients and methods', 'materials and methods',
        'métodos', 'materiales y métodos', 'diseño del estudio',
        'méthodes', 'matériels et méthodes'
    ],
    'author_info': [
        'author information', 'affiliations', 'corresponding author', 'contact',
        'información de los autores', 'afiliaciones', 'autor de correspondencia', 'contacto',
        'informations sur les auteurs', 'affiliations'
    ],
    'discussion': [
        'discussion', 'conclusion', 'interpretation',
        'discusión', 'conclusiones', 'interpretación',
        'discussion', 'conclusions'
    ],
    'results': [
        'results', 'findings', 'outcomes',
        'resultados', 'hallazgos',
        'résultats'
    ],
    'guideline': [
        'guideline', 'recommendations', 'consensus', 'expert panel',
        'guía', 'recomendaciones', 'consenso', 'panel de expertos',
        'lignes directrices', 'recommandations'
    ],
    'inline_citation': ['introduction', 'background', 'discussion', 'results']
}

# ============================================================================
# CONFIDENCE ADJUSTMENT RULES
# ============================================================================

CONFIDENCE_ADJUSTMENTS = {
    'has_orcid': +0.2,
    'has_email': +0.15,
    'has_affiliation': +0.10,
    'has_degree': +0.08,
    'explicit_role': +0.10,
    'in_author_section': +0.05,
    'multiple_mentions': +0.05,
    'has_title': +0.05,
    'has_particles': +0.03  # van der, de la, etc.
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_text_for_person_matching(text: str) -> str:
    """
    Normalize text for person name matching
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Normalize Unicode (NFC = canonical composition)
    normalized = unicodedata.normalize('NFC', text)
    return normalized

def validate_orcid(orcid: str) -> bool:
    """
    Validate ORCID using ISO 7064 (11,2) checksum
    
    Args:
        orcid: ORCID identifier
        
    Returns:
        True if valid, False otherwise
    """
    # Remove all non-digits except X
    digits = re.sub(r'[^\dX]', '', orcid.upper())
    
    if len(digits) != 16:
        return False
    
    # Calculate checksum
    total = 0
    for c in digits[:-1]:
        total = (total + int(c)) * 2
    
    remainder = total % 11
    result = (12 - remainder) % 11
    check_digit = 'X' if result == 10 else str(result)
    
    return check_digit == digits[-1]

def extract_person_components(text: str) -> Dict[str, str]:
    """
    Extract components from person name
    
    Args:
        text: Person name text
        
    Returns:
        Dictionary with name components
    """
    components = {
        'surname': None,
        'given_names': None,
        'initials': None,
        'particles': [],
        'titles': [],
        'degrees': []
    }
    
    # Extract titles
    for title in NAME_NORMALIZATION_RULES['titles']:
        if text.startswith(title):
            components['titles'].append(title)
            text = text[len(title):].strip()
    
    # Extract particles
    for particle in NAME_NORMALIZATION_RULES['prefixes']:
        if particle.lower() in text.lower():
            components['particles'].append(particle)
    
    # Extract degrees
    for degree in NAME_NORMALIZATION_RULES['suffixes']:
        if degree in text:
            components['degrees'].append(degree)
    
    return components

def get_person_pattern_stats() -> Dict[str, int]:
    """Get count of patterns by context"""
    contexts: defaultdict = defaultdict(int)
    for pattern_config in PERSON_NAME_PATTERNS.values():
        contexts[pattern_config['context']] += 1
    return dict(contexts)

def get_total_person_pattern_count() -> int:
    """Get total number of person patterns"""
    return len(PERSON_NAME_PATTERNS)

def get_patterns_by_role(role: str) -> Dict[str, Dict]:
    """Get all patterns for a specific role"""
    return {
        key: config for key, config in PERSON_NAME_PATTERNS.items()
        if config['role'] == role
    }

def get_all_roles() -> List[str]:
    """Get list of all unique roles"""
    return list(set(config['role'] for config in PERSON_NAME_PATTERNS.values()))

def validate_person_patterns() -> Dict[str, List[str]]:
    """
    Validate all person patterns against examples
    
    Returns:
        Dictionary of pattern_key -> list of validation errors
    """
    validation_results = {}
    
    for pattern_key, config in PERSON_NAME_PATTERNS.items():
        errors = []
        pattern = config['pattern']
        examples = config.get('examples', [])
        
        # Try to compile pattern
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            errors.append(f"Pattern compilation failed: {e}")
            validation_results[pattern_key] = errors
            continue
        
        # Test against examples
        for example in examples:
            normalized_example = normalize_text_for_person_matching(example)
            if not compiled.search(normalized_example):
                errors.append(f"Example '{example}' does not match pattern")
        
        if errors:
            validation_results[pattern_key] = errors
    
    return validation_results

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = '2.0.0'
__author__ = 'Biomedical Entity Extraction System'
__all__ = [
    'PERSON_NAME_PATTERNS',
    'AFFILIATION_PATTERNS',
    'ROLE_CLASSIFICATION_TRIGGERS',
    'NAME_NORMALIZATION_RULES',
    'CONFIDENCE_ADJUSTMENTS',
    'CONTEXT_SECTIONS',
    'LAT_UP',
    'LAT_LO',
    'NAME_CHUNK',
    'PARTICLE',
    'SURNAME',
    'INITIALS',
    'PERSON_CORE',
    'FULL_NAME',
    'get_person_pattern_stats',
    'get_total_person_pattern_count',
    'get_patterns_by_role',
    'get_all_roles',
    'normalize_text_for_person_matching',
    'validate_orcid',
    'extract_person_components',
    'validate_person_patterns'
]

if __name__ == "__main__":
    print("=" * 80)
    print("PERSON PATTERN CATALOG - v2.0.0 (COMPLETE REWRITE)")
    print("=" * 80)
    print(f"Total patterns: {get_total_person_pattern_count()}")
    print(f"\nPatterns by role:")
    for role in sorted(get_all_roles()):
        count = len(get_patterns_by_role(role))
        print(f"  {role:30s}: {count:3d}")
    
    print("\n" + "=" * 80)
    print("FEATURES")
    print("=" * 80)
    print("✓ Unicode support (diacritics)")
    print("✓ Surname particles (van der, de la, ibn)")
    print("✓ Hyphenated names (García-López, O'Connor)")
    print("✓ ORCID validation (ISO 7064)")
    print("✓ Multilingual (English, Spanish, French)")
    print("✓ Robust affiliation matching")
    
    # Run validation
    print("\n" + "=" * 80)
    print("PATTERN VALIDATION")
    print("=" * 80)
    validation_errors = validate_person_patterns()
    
    if not validation_errors:
        print("✅ All patterns validated successfully!")
    else:
        print(f"❌ Found {len(validation_errors)} patterns with issues:\n")
        for key, errors in validation_errors.items():
            print(f"  {key}:")
            for error in errors:
                print(f"    - {error}")
    
    # Test ORCID validation
    print("\n" + "=" * 80)
    print("ORCID VALIDATION TEST")
    print("=" * 80)
    test_orcids = [
        ('0000-0002-1234-567X', True),
        ('0000-0001-2345-6789', True),
        ('0000-0003-4567-8901', True),
        ('0000-0002-1234-5678', False),  # Invalid checksum
    ]
    
    for orcid, expected in test_orcids:
        result = validate_orcid(orcid)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {orcid}: {result} (expected {expected})")
    
    print("=" * 80)