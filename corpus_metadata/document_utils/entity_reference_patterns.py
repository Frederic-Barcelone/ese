#!/usr/bin/env python3
"""
Reference Pattern Dictionary - Comprehensive Biomedical & Scientific Literature Catalog
========================================================================================
Location: corpus_metadata/document_utils/entity_reference_patterns.py
Version: 1.2.0 - BUG FIXES AND IMPROVEMENTS
Last Updated: 2025-10-08

CHANGELOG v1.2.0:
-----------------
✓ FIXED: DOI pattern - removed extra ] in character class
✓ FIXED: MeSH pattern - now accepts both tree codes and UI (Dxxxxxx)
✓ FIXED: HGNC normalization - no longer duplicates prefix in URL
✓ IMPROVED: ClinVar normalization - strips leading zeros
✓ IMPROVED: Patent patterns - more robust formats
✓ IMPROVED: ICD-10 pattern - handles subcategories correctly
✓ IMPROVED: Added flags=re.I where appropriate
✓ ADDED: Pattern validation utility function
"""

import re
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# REFERENCE PATTERN DICTIONARY - CORRECTED VERSION
# ============================================================================

REFERENCE_PATTERNS = {
    
    # ========================================================================
    # CATEGORY 1: UNIVERSAL IDENTIFIERS
    # ========================================================================
    
    'doi': {
        'pattern': r'\b(?:doi:?\s*)?(10\.\d{4,9}/[^\s"<>]+)\b',  # FIXED: removed extra ]
        'url_template': 'https://doi.org/{id}',
        'normalize': lambda x: re.sub(r'\.$', '', re.sub(r'^https?://(dx\.)?doi\.org/', '', re.sub(r'^doi:\s*', '', x.strip(), flags=re.I), flags=re.I)),
        'confidence': 1.0,
        'category': 'universal',
        'source': 'Crossref/Publisher',
        'description': 'Digital Object Identifier (DOI) - Universal standard',
        'examples': ['10.1038/s41586-020-2649-2', 'DOI:10.1101/2020.12.31.123456', 'https://doi.org/10.1038/nphys1170']
    },
    
    # ========================================================================
    # CATEGORY 2: PUBMED & NCBI ECOSYSTEM
    # ========================================================================
    
    'pubmed': {
        'pattern': r'\bPMID:?\s*\d{1,9}\b',
        'url_template': 'https://pubmed.ncbi.nlm.nih.gov/{id}/',
        'normalize': lambda x: re.sub(r'\D', '', x),
        'confidence': 1.0,
        'category': 'literature',
        'source': 'PubMed',
        'description': 'PubMed unique identifier',
        'examples': ['PMID:34567890', 'PMID 12345678', 'PMID:1234567']
    },
    
    'pmcid': {
        'pattern': r'\b(?:PMCID:?\s*)?PMC\d{5,9}\b',
        'url_template': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{id}/',
        'normalize': lambda x: re.sub(r'\D', '', x),
        'confidence': 1.0,
        'category': 'repository',
        'source': 'PubMed Central',
        'description': 'PubMed Central identifier',
        'examples': ['PMCID: PMC7654321', 'PMC1234567', 'PMC9260009']
    },
    
    'ncbi_bookshelf': {
        'pattern': r'\bNBK\d{5,7}\b',
        'url_template': 'https://www.ncbi.nlm.nih.gov/books/NBK{id}/',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 1.0,
        'category': 'literature',
        'source': 'NCBI Bookshelf',
        'description': 'NCBI Bookshelf resource identifier',
        'examples': ['NBK554372', 'NBK123456']
    },
    
    'ncbi_gene': {
        'pattern': r'\b(?:Gene\s*ID|GeneID):?\s*\d{4,8}\b',  # IMPROVED: added \b boundaries
        'url_template': 'https://www.ncbi.nlm.nih.gov/gene/{id}',
        'normalize': lambda x: re.sub(r'\D', '', x),
        'confidence': 1.0,
        'category': 'database',
        'source': 'NCBI Gene',
        'description': 'NCBI Gene database identifier',
        'examples': ['Gene ID: 6606', 'GeneID:5460']
    },
    
    # ========================================================================
    # CATEGORY 3: PREPRINT SERVERS
    # ========================================================================
    
    'arxiv': {
        'pattern': r'\barXiv:?\s*(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?\b',
        'url_template': 'https://arxiv.org/abs/{id}',
        'normalize': lambda x: re.sub(r'^arXiv:\s*', '', x.strip(), flags=re.I),
        'confidence': 1.0,
        'category': 'preprint',
        'source': 'arXiv',
        'description': 'arXiv identifier (new and legacy formats)',
        'examples': ['arXiv:2101.01234', 'arXiv:2101.01234v2', 'arXiv:hep-th/9901001']
    },
    
    'biorxiv': {
        'pattern': r'\bbioRxiv:?\s*(\d{4}\.\d{2}\.\d{2}\.\d{5,7}(?:v\d+)?)\b',
        'url_template': 'https://www.biorxiv.org/content/10.1101/{id}',
        'normalize': lambda x: re.sub(r'^(?i)biorxiv:\s*', '', x).strip(),
        'confidence': 0.95,
        'category': 'preprint',
        'source': 'bioRxiv',
        'description': 'bioRxiv DOI suffix (biology preprints)',
        'examples': ['bioRxiv: 2020.12.31.123456', 'bioRxiv: 2023.05.10.567890v1'],
        'is_preprint': True
    },
    
    'medrxiv': {
        'pattern': r'\bmedRxiv:?\s*(\d{4}\.\d{2}\.\d{2}\.\d{5,7}(?:v\d+)?)\b',
        'url_template': 'https://www.medrxiv.org/content/10.1101/{id}',
        'normalize': lambda x: re.sub(r'^(?i)medrxiv:\s*', '', x).strip(),
        'confidence': 0.95,
        'category': 'preprint',
        'source': 'medRxiv',
        'description': 'medRxiv DOI suffix (medical preprints)',
        'examples': ['medRxiv: 2021.04.01.123456', 'medRxiv: 2022.08.15.765432v2'],
        'is_preprint': True
    },
    
    'ssrn': {
        'pattern': r'\bSSRN(?:[-\s]?id)?:?\s*\d{4,9}\b',
        'url_template': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id={id}',
        'normalize': lambda x: re.sub(r'\D', '', x),
        'confidence': 0.9,
        'category': 'preprint',
        'source': 'SSRN',
        'description': 'Social Science Research Network',
        'examples': ['SSRN: 1234567', 'SSRN-id: 9876543']
    },
    
    'europe_pmc_ppr': {
        'pattern': r'\bPPR\d{6,}\b',
        'url_template': 'https://europepmc.org/abstract/ppr/ppr{id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 0.85,
        'category': 'preprint',
        'source': 'Europe PMC',
        'description': 'Europe PMC preprint ID',
        'examples': ['PPR123456', 'PPR9876543']
    },
    
    # ========================================================================
    # CATEGORY 4: ACADEMIC INDEXES
    # ========================================================================
    
    'scopus_eid': {
        'pattern': r'\b(?:SCOPUS_ID:?\s*|EID:?\s*|eid=)?(2-s2\.0-\d{7,20})\b',
        'url_template': 'https://www.scopus.com/record/display.uri?eid={id}',
        'normalize': lambda x: re.search(r'2-s2\.0-\d{7,20}', x).group(0) if re.search(r'2-s2\.0-\d{7,20}', x) else x.strip(),
        'confidence': 0.95,
        'category': 'index',
        'source': 'Scopus (Elsevier)',
        'description': 'Scopus EID',
        'examples': ['2-s2.0-85012345678', 'SCOPUS_ID: 2-s2.0-84976543210']
    },
    
    'wos': {
        'pattern': r'\bWOS:\d{14,20}\b',
        'url_template': 'https://www.webofscience.com/wos/woscc/full-record/{id}',
        'normalize': lambda x: x.strip().upper(),
        'confidence': 0.9,
        'category': 'index',
        'source': 'Web of Science',
        'description': 'Web of Science Accession Number',
        'examples': ['WOS:000123456700012']
    },
    
    'openalex': {
        'pattern': r'\bW\d{6,13}\b',  # IMPROVED: increased to 13 digits
        'url_template': 'https://openalex.org/{id}',
        'normalize': lambda x: x.strip().upper(),
        'confidence': 0.95,
        'category': 'index',
        'source': 'OpenAlex',
        'description': 'OpenAlex Work ID',
        'examples': ['W2755950973']
    },
    
    'semanticscholar': {
        'pattern': r'\b(?:CorpusID:?\s*|S2PaperId:?\s*)(\d{6,12})\b',
        'url_template': None,  # FIXED: unstable endpoint, use API instead
        'normalize': lambda x: re.sub(r'\D', '', x),
        'confidence': 0.85,
        'category': 'index',
        'source': 'Semantic Scholar',
        'description': 'Semantic Scholar CorpusID',
        'examples': ['CorpusID: 12345678', 'S2PaperId: 9876543210']
    },
    
    'dblp': {
        'pattern': r'\bDBLP:[^\s,;]+',
        'url_template': 'https://dblp.org/rec/{id}.html',
        'normalize': lambda x: x.strip(),
        'confidence': 0.8,
        'category': 'index',
        'source': 'DBLP',
        'description': 'DBLP computer science bibliography',
        'examples': ['DBLP:journals/tods/Codd70']
    },
    
    'ads_bibcode': {
        'pattern': r'\b\d{4}[A-Za-z.&]{5}\d{4}[A-Za-z.][A-Za-z0-9.]{4}[A-Z]\b',
        'url_template': 'https://ui.adsabs.harvard.edu/abs/{id}/abstract',
        'normalize': lambda x: x.strip(),
        'confidence': 0.9,
        'category': 'index',
        'source': 'NASA ADS',
        'description': 'ADS bibcode (astronomy/physics)',
        'examples': ['1998ApJ...500..525S']
    },
    
    'inspirehep': {
        'pattern': r'\b(?:INSPIRE(?:-HEP)?:?\s*)?\d{6,8}\b',
        'url_template': 'https://inspirehep.net/literature/{id}',
        'normalize': lambda x: re.sub(r'\D', '', x),
        'confidence': 0.85,
        'category': 'index',
        'source': 'INSPIRE-HEP',
        'description': 'InspireHEP literature record',
        'examples': ['INSPIRE: 1234567']
    },
    
    # ========================================================================
    # CATEGORY 5: INSTITUTIONAL REPOSITORIES
    # ========================================================================
    
    'hal': {
        'pattern': r'\bhal-\d{8}\b',
        'url_template': 'https://hal.science/{id}',
        'normalize': lambda x: x.strip().lower(),
        'confidence': 0.95,
        'category': 'repository',
        'source': 'HAL (France)',
        'description': 'HAL open archive identifier',
        'examples': ['hal-01234567']
    },
    
    'repec': {
        'pattern': r'\bRePEc:[A-Za-z0-9:.\-_/]+\b',
        'url_template': None,  # Variable endpoint structure
        'normalize': lambda x: x.strip(),
        'confidence': 0.8,
        'category': 'repository',
        'source': 'RePEc',
        'description': 'RePEc handle (economics)',
        'examples': ['RePEc:nbr:nberwo:12345']
    },
    
    'oai': {
        'pattern': r'\boai:[^\s<>"]+\b',
        'url_template': None,
        'normalize': lambda x: x.strip(),
        'confidence': 0.7,
        'category': 'repository',
        'source': 'OAI-PMH',
        'description': 'OAI-PMH identifier',
        'examples': ['oai:arXiv.org:2101.01234']
    },
    
    # ========================================================================
    # CATEGORY 6: CLINICAL TRIAL REGISTRIES
    # ========================================================================
    
    'clinicaltrials_gov': {
        'pattern': r'\bNCT\d{8}\b',
        'url_template': 'https://clinicaltrials.gov/study/{id}',
        'normalize': lambda x: x.upper(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'ClinicalTrials.gov',
        'description': 'US Clinical Trials Registry',
        'examples': ['NCT03777891', 'NCT02994927']
    },
    
    'eudract': {
        'pattern': r'EudraCT\s*\d{4}-\d{6}-\d{2}',
        'url_template': None,  # FIXED: search endpoint more stable
        'normalize': lambda x: re.sub(r'EudraCT\s*', '', x, flags=re.IGNORECASE).strip(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'EU Clinical Trials Register',
        'description': 'European Clinical Trials',
        'examples': ['EudraCT 2018-001234-56']
    },
    
    'isrctn': {
        'pattern': r'\bISRCTN\d{8}\b',
        'url_template': 'https://www.isrctn.com/{id}',
        'normalize': lambda x: x.upper(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'ISRCTN Registry',
        'description': 'International Standard RCT Number',
        'examples': ['ISRCTN12345678']
    },
    
    'jprn_umin': {
        'pattern': r'(?:JPRN-)?UMIN\d{9}',
        'url_template': None,  # FIXED: requires search
        'normalize': lambda x: re.sub(r'JPRN-', '', x, flags=re.IGNORECASE).upper(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'UMIN-CTR (Japan)',
        'description': 'UMIN Clinical Trials Registry',
        'examples': ['UMIN000012345']
    },
    
    'chictr': {
        'pattern': r'\bChiCTR\d{10,12}\b',
        'url_template': None,  # FIXED: requires search
        'normalize': lambda x: x.upper(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'ChiCTR (China)',
        'description': 'Chinese Clinical Trial Registry',
        'examples': ['ChiCTR1800015678']
    },
    
    'ctri_india': {
        'pattern': r'CTRI/\d{4}/\d{2}/\d{6}',
        'url_template': None,  # FIXED: requires search
        'normalize': lambda x: x.upper(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'CTRI (India)',
        'description': 'Clinical Trials Registry - India',
        'examples': ['CTRI/2020/01/022345']
    },
    
    'actrn_australia': {
        'pattern': r'\bACTRN\d{14}\b',
        'url_template': None,  # FIXED: requires search
        'normalize': lambda x: x.upper(),
        'confidence': 1.0,
        'category': 'clinical_trial',
        'source': 'ANZCTR',
        'description': 'Australian NZ Clinical Trials Registry',
        'examples': ['ACTRN12618000123456']
    },
    
    # ========================================================================
    # CATEGORY 7: REGULATORY AGENCIES
    # ========================================================================
    
    'fda_document': {
        'pattern': r'(?:FDA|BLA|NDA|ANDA)[-\s]?\d{6,7}',
        'url_template': None,
        'normalize': lambda x: x.upper().replace(' ', '-'),
        'confidence': 0.95,
        'category': 'regulatory',
        'source': 'FDA (US)',
        'description': 'US FDA document identifier',
        'examples': ['BLA-125662', 'NDA 212819']
    },
    
    'ema_document': {
        'pattern': r'EMA/\d{5,7}/\d{4}',
        'url_template': None,
        'normalize': lambda x: x.upper(),
        'confidence': 0.95,
        'category': 'regulatory',
        'source': 'EMA (EU)',
        'description': 'European Medicines Agency',
        'examples': ['EMA/123456/2023']
    },
    
    'ema_epar': {
        'pattern': r'EMEA/H/C/\d{6}',
        'url_template': None,
        'normalize': lambda x: x.upper(),
        'confidence': 1.0,
        'category': 'regulatory',
        'source': 'EMA (EU)',
        'description': 'EMA Public Assessment Report',
        'examples': ['EMEA/H/C/004338']
    },
    
    # ========================================================================
    # CATEGORY 8: PATENTS
    # ========================================================================
    
    'us_patent': {
        'pattern': r'\bUS(?:RE|D|PP)?\s*\d{1,2}[,\d]{6,}\s*[A-Z]\d?\b',  # IMPROVED: more formats
        'url_template': 'https://patents.google.com/patent/{id}',
        'normalize': lambda x: re.sub(r'[,\s]', '', x).upper(),
        'confidence': 1.0,
        'category': 'patent',
        'source': 'USPTO',
        'description': 'United States Patent',
        'examples': ['US 9,876,543 B2', 'US10234567A1']
    },
    
    'ep_patent': {
        'pattern': r'\bEP\s*\d{7,9}\s*[ABC]\d?\b',  # IMPROVED: 7-9 digits
        'url_template': 'https://patents.google.com/patent/{id}',
        'normalize': lambda x: re.sub(r'\s+', '', x).upper(),
        'confidence': 1.0,
        'category': 'patent',
        'source': 'EPO',
        'description': 'European Patent',
        'examples': ['EP 1234567 A1']
    },
    
    'wo_patent': {
        'pattern': r'\bWO\s*\d{4}/\d{6}\s*A\d?\b',
        'url_template': 'https://patents.google.com/patent/{id}',
        'normalize': lambda x: re.sub(r'\s+', '', x).upper(),
        'confidence': 1.0,
        'category': 'patent',
        'source': 'WIPO',
        'description': 'WIPO Patent',
        'examples': ['WO 2020/123456 A1']
    },
    
    # ========================================================================
    # CATEGORY 9: CLINICAL GUIDELINES
    # ========================================================================
    
    'kdigo_guideline': {
        'pattern': r'\bKDIGO\s+(?:20\d{2})\s+(?:CKD|AKI|GN|Transplant|Hepatitis\s*C|Glomerulonephritis|Anemia)\b',  # IMPROVED: stricter
        'url_template': None,
        'normalize': lambda x: x.upper(),
        'confidence': 0.85,  # LOWERED: permissive pattern
        'category': 'guideline',
        'source': 'KDIGO',
        'description': 'KDIGO Guidelines',
        'examples': ['KDIGO 2021 CKD', 'KDIGO 2012 Glomerulonephritis']
    },
    
    'nice_guideline': {
        'pattern': r'\bNICE\s*(?:TA|NG|CG|MTG|DG|IPG|QS|SG)\d{1,4}\b',
        'url_template': 'https://www.nice.org.uk/guidance/{id}',
        'normalize': lambda x: re.sub(r'\s+', '', x).upper(),
        'confidence': 0.95,
        'category': 'guideline',
        'source': 'NICE (UK)',
        'description': 'NICE Guidelines',
        'examples': ['NICE TA123', 'NICE NG45']
    },
    
    'ich_guideline': {
        'pattern': r'\bICH\s*[EQM]\d{1,2}[A-Z]?\(?R\d\)?\b',
        'url_template': 'https://www.ich.org/page/ich-guidelines',
        'normalize': lambda x: re.sub(r'\s+', '', x).upper(),
        'confidence': 0.95,
        'category': 'guideline',
        'source': 'ICH',
        'description': 'ICH Guidelines',
        'examples': ['ICH E6(R2)', 'ICH Q1A(R2)']
    },
    
    # ========================================================================
    # CATEGORY 10: DRUG & DISEASE DATABASES
    # ========================================================================
    
    'rxnorm': {
        'pattern': r'\bRxCUI:?\s*\d{4,8}\b',
        'url_template': 'https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 1.0,
        'category': 'database',
        'source': 'RxNorm',
        'description': 'RxNorm Concept Unique Identifier',
        'examples': ['RxCUI: 123456', 'RxCUI 2574890']
    },
    
    'atc_code': {
        'pattern': r'\bATC:?\s*[A-Z]\d{2}[A-Z]{2}\d{2}\b',
        'url_template': 'https://www.whocc.no/atc_ddd_index/?code={id}',
        'normalize': lambda x: re.sub(r'ATC:?\s*', '', x, flags=re.IGNORECASE).upper(),
        'confidence': 1.0,
        'category': 'database',
        'source': 'WHO ATC',
        'description': 'ATC Classification',
        'examples': ['ATC: L01XE21', 'ATC C09AA01']
    },
    
    'orphanet': {
        'pattern': r'\bORPHA:?\s*\d{3,6}\b',
        'url_template': 'https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=en&Expert={id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 1.0,
        'category': 'database',
        'source': 'Orphanet',
        'description': 'Orphanet Rare Disease Ontology',
        'examples': ['ORPHA:324', 'ORPHA 558']
    },
    
    'omim': {
        'pattern': r'\bOMIM:?\s*[*#%^+]?\d{6}\b',
        'url_template': 'https://www.omim.org/entry/{id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 1.0,
        'category': 'database',
        'source': 'OMIM',
        'description': 'Online Mendelian Inheritance in Man',
        'examples': ['OMIM: 301500', 'OMIM *146045']
    },
    
    'snomed_ct': {
        'pattern': r'\bSNOMED\s*CT:?\s*\d{6,18}\b',
        'url_template': None,
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 0.95,
        'category': 'database',
        'source': 'SNOMED CT',
        'description': 'SNOMED Clinical Terms',
        'examples': ['SNOMED CT: 123456789012']
    },
    
    'icd10': {
        'pattern': r'\bICD-?10:?\s*[A-TV-Z]\d{2}(?:\.\d{1,2})?\b',  # IMPROVED: subcategories
        'url_template': None,
        'normalize': lambda x: re.sub(r'ICD-?10:?\s*', '', x, flags=re.IGNORECASE).upper(),
        'confidence': 0.95,
        'category': 'database',
        'source': 'ICD-10',
        'description': 'ICD-10 Classification',
        'examples': ['ICD-10: E75.2', 'ICD10 C50.9']
    },
    
    'mesh': {
        'pattern': r'\bMeSH:?\s*(?:D\d{6}|[A-Z]\d{2}(?:\.\d{3})+)\b',  # FIXED: accepts UI and tree
        'url_template': 'https://meshb.nlm.nih.gov/record/ui?ui={id}',
        'normalize': lambda x: re.sub(r'MeSH:?\s*', '', x, flags=re.IGNORECASE).upper(),
        'confidence': 0.95,
        'category': 'database',
        'source': 'MeSH',
        'description': 'Medical Subject Headings',
        'examples': ['MeSH: D005199', 'MeSH C10.228.140.300']
    },
    
    'uniprot': {
        'pattern': r'\bUniProt:?\s*[A-Z]\d[A-Z0-9]{3}\d\b',
        'url_template': 'https://www.uniprot.org/uniprotkb/{id}',
        'normalize': lambda x: re.sub(r'UniProt:?\s*', '', x, flags=re.IGNORECASE).upper(),
        'confidence': 0.95,
        'category': 'database',
        'source': 'UniProt',
        'description': 'Universal Protein Resource',
        'examples': ['UniProt: P12345']
    },
    
    'drugbank': {
        'pattern': r'\bDrugBank:?\s*DB\d{5}\b',
        'url_template': 'https://go.drugbank.com/drugs/{id}',
        'normalize': lambda x: re.sub(r'DrugBank:?\s*', '', x, flags=re.IGNORECASE).upper(),
        'confidence': 0.95,
        'category': 'database',
        'source': 'DrugBank',
        'description': 'DrugBank Database',
        'examples': ['DrugBank: DB00001']
    },
    
    'chembl': {
        'pattern': r'\bCHEMBL\d{5,7}\b',
        'url_template': 'https://www.ebi.ac.uk/chembl/compound_report_card/{id}',
        'normalize': lambda x: x.upper(),
        'confidence': 0.95,
        'category': 'database',
        'source': 'ChEMBL',
        'description': 'ChEMBL Bioactivity Database',
        'examples': ['CHEMBL123456']
    },
    
    # ========================================================================
    # CATEGORY 11: GENETIC & GENOMIC DATABASES
    # ========================================================================
    
    'clinvar': {
        'pattern': r'\b(?:ClinVar|VCV)0*\d{1,9}\b',
        'url_template': 'https://www.ncbi.nlm.nih.gov/clinvar/variation/{id}/',
        'normalize': lambda x: re.sub(r'^0+', '', re.sub(r'[^\d]', '', x)),  # FIXED: strip leading zeros
        'confidence': 0.95,
        'category': 'database',
        'source': 'ClinVar',
        'description': 'ClinVar Genetic Variant Database',
        'examples': ['VCV000012345', 'ClinVar 000123456']
    },
    
    'dbsnp': {
        'pattern': r'\brs\d{5,10}\b',
        'url_template': 'https://www.ncbi.nlm.nih.gov/snp/{id}',
        'normalize': lambda x: x.lower(),
        'confidence': 0.9,
        'category': 'database',
        'source': 'dbSNP',
        'description': 'SNP Database',
        'examples': ['rs123456789']
    },
    
    'ensembl': {
        'pattern': r'\bENS[A-Z]{0,3}\d{11}\b',
        'url_template': 'https://www.ensembl.org/id/{id}',
        'normalize': lambda x: x.upper(),
        'confidence': 0.95,
        'category': 'database',
        'source': 'Ensembl',
        'description': 'Ensembl Genome Database',
        'examples': ['ENSG00000139618']
    },
    
    'hgnc': {
        'pattern': r'\bHGNC:?\s*\d{4,6}\b',
        'url_template': 'https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/HGNC:{id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),  # FIXED: no prefix duplication
        'confidence': 0.95,
        'category': 'database',
        'source': 'HGNC',
        'description': 'HUGO Gene Nomenclature Committee',
        'examples': ['HGNC:5', 'HGNC 12345']
    },
    
    # ========================================================================
    # CATEGORY 12: SYSTEMATIC REVIEWS
    # ========================================================================
    
    'cochrane_review': {
        'pattern': r'\bCD\d{6}\b',
        'url_template': 'https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.{id}/full',
        'normalize': lambda x: x.upper(),
        'confidence': 0.95,
        'category': 'literature',
        'source': 'Cochrane Library',
        'description': 'Cochrane Systematic Reviews',
        'examples': ['CD001234']
    },
    
    'prospero': {
        'pattern': r'\bCRD\d{11}\b',
        'url_template': 'https://www.crd.york.ac.uk/prospero/display_record.php?RecordID={id}',
        'normalize': lambda x: x.upper(),
        'confidence': 0.95,
        'category': 'literature',
        'source': 'PROSPERO',
        'description': 'Prospective Register of Systematic Reviews',
        'examples': ['CRD42021234567']
    },
    
    # ========================================================================
    # CATEGORY 13: RARE DISEASE DATABASES
    # ========================================================================
    
    'gard': {
        'pattern': r'\bGARD:?\s*\d{4,5}\b',
        'url_template': 'https://rarediseases.info.nih.gov/diseases/{id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 0.95,
        'category': 'database',
        'source': 'GARD (NIH)',
        'description': 'Genetic and Rare Diseases Information Center',
        'examples': ['GARD: 12345']
    },
    
    'mondo': {
        'pattern': r'\bMONDO:?\s*\d{7}\b',
        'url_template': 'https://monarchinitiative.org/disease/MONDO:{id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 0.95,
        'category': 'database',
        'source': 'MONDO',
        'description': 'Monarch Disease Ontology',
        'examples': ['MONDO: 0007915']
    },
    
    'hpo': {
        'pattern': r'\bHP:?\s*\d{7}\b',
        'url_template': 'https://hpo.jax.org/app/browse/term/HP:{id}',
        'normalize': lambda x: re.sub(r'[^\d]', '', x),
        'confidence': 0.95,
        'category': 'database',
        'source': 'HPO',
        'description': 'Human Phenotype Ontology',
        'examples': ['HP: 0001250']
    },
    
    # ========================================================================
    # CATEGORY 14: PUBLISHER-SPECIFIC
    # ========================================================================
    
    'pii': {
        'pattern': r'\bS\d{4}-\d{3}[0-9X]\(\d{2}\)\d{5}-\d\b|\bS\d{15}\b',
        'url_template': None,
        'normalize': lambda x: x.strip(),
        'confidence': 0.75,
        'category': 'publisher',
        'source': 'Publisher',
        'description': 'Publisher Item Identifier',
        'examples': ['S0167-6423(00)00100-7']
    }
}

# ============================================================================
# CONTEXT ROLE PATTERNS
# ============================================================================

CONTEXT_ROLE_PATTERNS = {
    'supporting_evidence': [
        r'demonstrated in',
        r'shown (?:in|by)',
        r'reported (?:in|by)',
        r'observed in',
        r'found in',
        r'evidence from',
        r'according to'
    ],
    
    'methodology': [
        r'following',
        r'using (?:the )?method',
        r'per protocol',
        r'adapted from',
        r'as described in'
    ],
    
    'guideline_reference': [
        r'per (?:the )?guideline',
        r'according to (?:the )?recommendation',
        r'recommended by'
    ],
    
    'comparative': [
        r'compared (?:with|to)',
        r'versus',
        r'in contrast',
        r'differ(?:s|ed) from'
    ],
    
    'safety_data': [
        r'adverse (?:event|effect)',
        r'safety profile',
        r'tolerability'
    ],
    
    'trial_registry': [
        r'registered (?:as|at)',
        r'trial registration',
        r'clinicaltrials\.gov'
    ],
    
    'background': [
        r'previous(?:ly)? stud(?:y|ies)',
        r'literature review',
        r'prior research'
    ]
}

# ============================================================================
# PUBLISHER METADATA
# ============================================================================

PUBLISHER_METADATA = {
    'elsevier': {
        'name': 'Elsevier',
        'quality_tier': 'high',
        'major_journals': ['The Lancet', 'Cell', 'Neuron']
    },
    'springer_nature': {
        'name': 'Springer Nature',
        'quality_tier': 'high',
        'major_journals': ['Nature', 'Nature Medicine']
    },
    'nejm': {
        'name': 'New England Journal of Medicine',
        'quality_tier': 'high',
        'major_journals': ['NEJM']
    }
}

# ============================================================================
# CONFIDENCE BOOSTERS
# ============================================================================

CONFIDENCE_BOOSTERS = {
    'in_title': +0.1,
    'in_methods': +0.05,
    'explicit_citation': +0.1,
    'multiple_mentions': +0.05,
    'peer_reviewed': +0.1
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_reference_category_stats() -> Dict[str, int]:
    """Get count of patterns by category"""
    categories: defaultdict = defaultdict(int)
    for ref_config in REFERENCE_PATTERNS.values():
        categories[ref_config['category']] += 1
    return dict(categories)

def get_total_pattern_count() -> int:
    """Get total number of reference patterns"""
    return len(REFERENCE_PATTERNS)

def validate_pattern(pattern_key: str) -> bool:
    """Check if a pattern key exists"""
    return pattern_key in REFERENCE_PATTERNS

def get_patterns_by_category(category: str) -> Dict[str, Dict]:
    """Get all patterns for a specific category"""
    return {
        key: config for key, config in REFERENCE_PATTERNS.items()
        if config['category'] == category
    }

def get_all_categories() -> List[str]:
    """Get list of all unique categories"""
    return list(set(config['category'] for config in REFERENCE_PATTERNS.values()))

def validate_all_patterns() -> Dict[str, List[str]]:
    """
    Validate all patterns against their examples
    
    Returns dictionary of pattern_key -> list of validation errors
    """
    validation_results = {}
    
    for pattern_key, config in REFERENCE_PATTERNS.items():
        errors = []
        pattern = config['pattern']
        examples = config.get('examples', [])
        
        # Try to compile pattern
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            errors.append(f"Pattern compilation failed: {e}")
            validation_results[pattern_key] = errors
            continue
        
        # Test against examples
        for example in examples:
            if not compiled.search(example):
                errors.append(f"Example '{example}' does not match pattern")
        
        # Test normalization if present
        if errors == [] and examples and config.get('normalize'):
            try:
                for example in examples[:1]:  # Test first example
                    match = compiled.search(example)
                    if match:
                        normalized = config['normalize'](match.group(0))
                        if not normalized:
                            errors.append(f"Normalization returned empty for '{example}'")
            except Exception as e:
                errors.append(f"Normalization failed: {e}")
        
        if errors:
            validation_results[pattern_key] = errors
    
    return validation_results

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = '1.2.0'
__author__ = 'Biomedical Entity Extraction System'
__all__ = [
    'REFERENCE_PATTERNS',
    'CONTEXT_ROLE_PATTERNS',
    'PUBLISHER_METADATA',
    'CONFIDENCE_BOOSTERS',
    'get_reference_category_stats',
    'get_total_pattern_count',
    'validate_pattern',
    'get_patterns_by_category',
    'get_all_categories',
    'validate_all_patterns'
]

if __name__ == "__main__":
    print("=" * 80)
    print("REFERENCE PATTERN CATALOG - v1.2.0 (FIXED)")
    print("=" * 80)
    print(f"Total patterns: {get_total_pattern_count()}")
    print(f"\nPatterns by category:")
    for cat in sorted(get_all_categories()):
        count = len(get_patterns_by_category(cat))
        print(f"  {cat:25s}: {count:3d}")
    
    # Run validation
    print("\n" + "=" * 80)
    print("PATTERN VALIDATION")
    print("=" * 80)
    validation_errors = validate_all_patterns()
    
    if not validation_errors:
        print("✅ All patterns validated successfully!")
    else:
        print(f"❌ Found {len(validation_errors)} patterns with issues:\n")
        for key, errors in validation_errors.items():
            print(f"  {key}:")
            for error in errors:
                print(f"    - {error}")
    
    print("=" * 80)