# Entity Detection -- How It Works

> **Date**: February 2026
> **Pipeline version**: v0.8

A deep dive into how the ESE pipeline detects, validates, and normalizes every entity type -- from abbreviations and diseases to drugs, genes, authors, citations, feasibility data, and clinical recommendations.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Abbreviations](#2-abbreviations)
3. [Diseases](#3-diseases)
4. [Drugs](#4-drugs)
5. [Genes](#5-genes)
6. [Authors](#6-authors)
7. [Citations](#7-citations)
8. [Pharma Companies](#8-pharma-companies)
9. [Document Metadata](#9-document-metadata)
10. [Feasibility](#10-feasibility)
11. [Recommendations](#11-recommendations)
12. [Care Pathways](#12-care-pathways)
13. [Validation Layer](#13-validation-layer)
14. [Normalization Layer](#14-normalization-layer)

---

## 1. Overview

### Three-Layer Architecture

Every entity type follows the same three-layer philosophy:

```
PDF  -->  Generators (C_generators/)  -->  Validation (D_validation/)  -->  Normalization (E_normalization/)
            HIGH RECALL                      HIGH PRECISION                  STANDARDIZATION
            "Find everything"                "Filter noise"                  "Map to ontologies"
```

**Generators** cast a wide net using multiple complementary strategies -- syntax parsing, regex patterns, FlashText lexicon matching, scispacy NER, and LLM/VLM extraction. False positives are acceptable at this stage.

**Validation** applies precision filtering through LLM verification, heuristic rules (PASO for abbreviations), false-positive filter lists, and confidence scoring. The goal is to reject noise while preserving true detections.

**Normalization** maps validated entities to standard ontologies (MONDO, RxNorm, HGNC), deduplicates by canonical ID, and enriches with external data from PubTator3, ClinicalTrials.gov, and other APIs.

### Generator Interface

All generators implement a common interface:

```python
class BaseCandidateGenerator(ABC):
    @property
    def generator_type(self) -> GeneratorType: ...

    def extract(self, doc_structure: DocumentModel) -> List[Candidate]:
        """Extract candidates from document."""
```

Generator types include: `SYNTAX` (algorithmic parsing), `LEXICON_MATCH` (FlashText keyword matching), `GLOSSARY` (structured glossary extraction), `REGEX` (pattern matching), `NER` (scispacy named entity recognition), and `LLM` / `VLM` (language and vision model extraction).

---

## 2. Abbreviations

Abbreviation detection is the most mature pipeline, with four generators, PASO heuristics, and LLM validation.

### Generator 1: Schwartz-Hearst Algorithm (C01)

The primary generator implements a modified Schwartz-Hearst algorithm for extracting abbreviation-definition pairs from parenthetical patterns:

- Scans for patterns like `long form (SF)` and `SF (long form)`
- Character-matching heuristic: verifies that initials of the long form match the short form
- Handles bidirectional patterns (SF before or after definition)
- Filters by short form length (2-12 characters) and validates form structure
- Produces `DEFINITION_PAIR` type candidates with both short and long forms

### Generator 2: FlashText Lexicon Matching (C04)

Matches document text against 600K+ known abbreviation-definition pairs from six lexicon sources:

| Source | Terms | Description |
|--------|-------|-------------|
| Meta-Inventory | 65K | Clinical abbreviations |
| Trial acronyms | 125K | ClinicalTrials.gov study acronyms |
| Statistical terms | ~100 | HR, CI, OR, SD, etc. |
| Clinical terms | ~200 | Common medical abbreviations |
| Gene symbols | ~500 | HGNC-approved gene abbreviations |
| Country codes | ~200 | ISO country code abbreviations |

FlashText provides O(n) matching regardless of vocabulary size, making it practical to search 600K+ terms in a single pass. Both straight and curly apostrophe variants are loaded to handle PDF extraction artifacts.

### Generator 3: Glossary Extraction (C05)

Detects structured glossary sections within documents:

- Identifies glossary headers ("List of Abbreviations", "Glossary", "Abbreviations Used")
- Parses tabular and list-based glossary formats
- Extracts SF-LF pairs from structured layouts (e.g., `AE    Adverse Event`)
- Produces `GLOSSARY_ENTRY` type candidates with high confidence

### Generator 4: Regex Pattern Matching

Applies five syntax patterns for abbreviation detection:

1. **Parenthetical forward**: `long form (SF)` -- most common pattern
2. **Parenthetical reverse**: `SF (long form)` -- less common but valid
3. **Dash/colon separator**: `SF - long form` or `SF: long form`
4. **Equals sign**: `SF = long form`
5. **Inline definition**: `SF, also known as long form`

### PASO Heuristics

Four heuristic rules that auto-approve or auto-reject candidates without LLM:

| Rule | Action | Examples |
|------|--------|---------|
| **PASO A** | Auto-approve statistical abbreviations | CI, HR, SD, OR, RR, BMI |
| **PASO B** | Auto-reject country codes and blacklisted terms | US, UK, EU, DNA, RNA |
| **PASO C** | Auto-enrich hyphenated abbreviations from NCT data | ANCA-PR3 from ClinicalTrials.gov |
| **PASO D** | LLM short-form-only extraction for missing definitions | Extracts definitions from context |

### LLM Validation

Candidates not handled by PASO rules go through LLM validation:

1. **Fast-reject pre-screening** (Haiku): Quick filter with 0.85 confidence threshold -- rejects obvious false positives cheaply
2. **Batch validation** (Haiku): Groups of 15-20 candidates verified in a single LLM call -- checks if each SF-LF pair is a genuine abbreviation in context
3. **Single validation** (Haiku): Fallback for candidates that fail batch processing
4. **In-memory caching**: Results cached by (SF, LF) to avoid re-validating duplicates across documents

---

## 3. Diseases

Disease detection combines lexicon matching across four sources with scispacy NER and aggressive false-positive filtering.

### FlashText Lexicon Matching (C06)

Four lexicon layers loaded into FlashText for O(n) matching:

| Source | Terms | Purpose |
|--------|-------|---------|
| General disease lexicon | 29K | Broad disease vocabulary |
| Orphanet | 9.5K | Rare disease names and synonyms |
| MONDO | 97K | Unified disease ontology terms |
| Rare disease acronyms | 1,640 | Short-form rare disease names (ALD, PKU) |

**Apostrophe normalization**: PDFs often contain curly quotes (U+2019) while lexicons use straight quotes (0x27). Both variants are loaded into FlashText so terms like "Crohn's disease" match regardless of quote style.

### scispacy NER with UMLS Linking

A biomedical NER model identifies disease mentions not in the lexicons:

- Uses `en_cia_biomedical` model for named entity recognition
- Links detected entities to UMLS concepts for ontology mapping
- Catches novel disease names, rare variants, and non-standard nomenclature

### False Positive Filter (C24)

Two-tier filtering to remove non-disease terms that match lexicon entries:

**COMMON_ENGLISH_FP_TERMS** (single-word hard filter):
- Removes generic words: "syndrome", "disease", "disorder" when standalone
- Filters common English words that appear in disease names: "cold", "depression", "shock"
- Threshold: Immediate rejection, no confidence scoring

**GENERIC_MULTIWORD_FP_TERMS** (multi-word hard filter):
- Removes clinical descriptions that aren't diagnoses: "associated with", "family history of"
- Filters symptom descriptions: "loss of consciousness", "difficulty breathing"

**Confidence adjustment system**:
- `MIN_ADJUSTMENT_FLOOR = -0.45`: Maximum negative adjustment to confidence
- C06 skip threshold = -0.5: Candidates below this are dropped
- Citation override = -0.55: Known citation patterns get extra penalty

### 6-Step Evaluation Matching

For benchmarking against gold standards, disease matching uses a 6-step cascade:

1. **Exact match**: Case-insensitive string equality
2. **Substring match**: One term contains the other
3. **Token overlap**: Shared word tokens exceed threshold
4. **Synonym group**: Both terms in same MONDO/Orphanet synonym group
5. **Synonym normalization**: Normalize both through synonym lookup, then compare
6. **Fuzzy match**: String similarity >= 0.8 (Levenshtein-based)

### PubTator3 Enrichment

Validated diseases are enriched via PubTator3 API:

- Maps to MeSH codes and concept identifiers
- Retrieves disease aliases and cross-references
- Links to MONDO, Orphanet, SNOMED, ICD-10, ICD-11 codes

---

## 4. Drugs

Drug detection uses five FlashText lexicon layers, compound ID regex, scispacy NER, and extensive false-positive filtering.

### FlashText Lexicon Matching (C07)

Five lexicon layers loaded for comprehensive drug detection:

| Source | Terms | Purpose |
|--------|-------|---------|
| Alexion drug list | ~50 | Sponsor-specific drug names |
| Investigational drugs | ~200 | Pipeline compounds and candidates |
| FDA/ChEMBL approved | 23K | Approved drug names and generics |
| RxNorm | 132K | Comprehensive drug vocabulary |
| Consumer variants | ~500 | Brand name variants (Tylenol, Advil) |

Consumer drug variants (C26) handle informal drug references found in social media and patient forums -- maps terms like "tylenol" to acetaminophen, handles common misspellings, and recognizes supplement names.

### Compound ID Regex Patterns

Detects development-stage compounds by their identifier patterns:

- **Trial compound IDs**: `ALXN1210`, `LY3009120`, `ABT-199`
- **Chemical identifiers**: CAS numbers, UNII codes
- **Development codes**: Phase I/II/III compound naming conventions

### scispacy Chemical NER

The biomedical NER model also identifies chemical/drug entities:

- Detects drug names not in lexicons (novel compounds, experimental drugs)
- Links to UMLS chemical concepts
- Catches brand names, generic names, and chemical formulas

### False Positive Filter (C25)

Extensive filtering across multiple categories:

| Category | Examples | Count |
|----------|----------|-------|
| **Body parts** | hip, disc, arm, knee | ~30 |
| **Biological entities** | C3, C4, C5, MAC, complement | ~50 |
| **Equipment/procedures** | blood test, MRI, CT scan | ~20 |
| **Common words** | sleep, cold, cough, statin | ~40 |
| **Credentials** | MD, PhD, MPH, MBBS, FRCP | ~25 |
| **Vaccine terms** | meningococcal, conjugate vaccine | ~20 |
| **Disease terms** | dengue, zika, ebola, RSV | ~15 |

**Author pattern filtering**: Detects drug names appearing in author name patterns (e.g., "by [name]," with comma/period) and suppresses false positives from bibliography sections.

### PubTator/RxNorm/DrugBank Enrichment

Validated drugs are enriched with:

- **RxNorm**: RxCUI codes, drug class, dosage forms, routes
- **DrugBank**: DrugBank IDs, mechanism of action, targets
- **PubTator3**: MeSH codes, cross-references
- **Marketing status**: Approved, investigational, withdrawn

---

## 5. Genes

Gene detection focuses on HGNC-approved symbols with aggressive filtering for the high ambiguity of short gene names.

### HGNC Lexicon Matching (C16)

Two lexicon sources loaded into FlashText:

| Source | Terms | Purpose |
|--------|-------|---------|
| HGNC symbols + aliases | 42K+ | Official gene symbols and approved aliases |
| Orphadata gene-disease | ~5K | Gene-disease association mappings |

### Pattern Matching

Regex patterns for common gene symbol formats:

- **Standard symbols**: 2-6 uppercase letters followed by optional digits (e.g., BRCA1, TP53, HER2)
- **Gene families**: Patterns like "MAPK family", "SMADs"
- **Variant notation**: p.V600E, c.1234A>G mutation patterns

### scispacy Gene NER

Biomedical NER identifies gene/protein mentions:

- Links to UMLS gene/protein concepts
- Catches full gene names alongside symbols

### False Positive Filter (C34)

Gene symbols are highly ambiguous -- many 2-3 letter symbols clash with common abbreviations. The filter maintains multiple exclusion sets:

| Category | Examples | Purpose |
|----------|----------|---------|
| **Statistical terms** | OR, HR, CI, SD, SE, RR | Odds ratio, hazard ratio, etc. |
| **Units** | mm, cm, kg, mg, ml, ng | Measurement units |
| **Clinical terms** | IV, ER, ICU, CT, MRI, GFR | Medical abbreviations |
| **Countries** | US, UK, EU, CA, AU, DE | ISO country codes |
| **Credentials** | MD, PhD, MPH, DO, RN | Academic/medical titles |
| **Drug terms** | ACE (inhibitor), ARB | Drug class abbreviations |

**Context-aware filtering**: The filter examines surrounding text to distinguish between gene symbols and their non-gene homonyms (e.g., "OR" as odds ratio vs. "OR" as olfactory receptor).

### PubTator3 Enrichment

Validated genes are enriched with:

- **Entrez Gene IDs**: NCBI Gene database identifiers
- **Ensembl IDs**: Ensembl genome database identifiers
- **OMIM numbers**: Online Mendelian Inheritance in Man
- **Disease associations**: Gene-disease linkages from Orphadata

---

## 6. Authors

Author detection uses multiple strategies to find investigators, affiliations, and identifiers.

### Detection Strategies (C08)

1. **Header patterns**: Author names from document title pages and headers
2. **Affiliation blocks**: Institutional affiliations with superscript numbering
3. **Contribution sections**: "Author Contributions", "Writing Committee" sections
4. **Investigator lists**: "Study Investigators", "APPENDIX" investigator tables
5. **Corresponding author**: Contact information blocks with email/phone

### Identifier Extraction

- **ORCID**: Regex patterns for `0000-0000-0000-000X` format
- **Email**: Standard email regex with domain validation
- **Role classification**: 7 role types (Lead Author, Co-Author, Investigator, Medical Writer, Statistician, Study Chair, Contributor)

### Author Name Parsing

- Handles "First Last", "Last, First", and "F. Last" formats
- Detects and filters credentials (MD, PhD, etc.) from names
- Groups authors by affiliation using superscript markers

---

## 7. Citations

Citation detection extracts structured references and inline citation markers.

### Identifier Extraction

Regex patterns for each identifier type:

| Type | Pattern | Example |
|------|---------|---------|
| **PMID** | `PMID:?\s*\d{7,8}` | PMID: 12345678 |
| **DOI** | `10\.\d{4,}/[^\s]+` | 10.1056/NEJMoa2023370 |
| **NCT** | `NCT\d{8}` | NCT04195139 |
| **PMCID** | `PMC\d{7}` | PMC7654321 |
| **URL** | Standard URL regex | https://doi.org/... |

### Reference Section Parsing

- Detects "References", "Bibliography", "Works Cited" section headers
- Parses numbered reference lists (1., [1], etc.)
- Extracts structured fields: authors, title, journal, year, volume, pages

### Inline Citation Markers

- Matches superscript numbers, bracketed numbers `[1,2,3]`, and author-year citations
- Links inline markers to full references in the reference section

### API Validation

- **PubMed**: Validates PMIDs and retrieves full citation metadata
- **ClinicalTrials.gov**: Validates NCT numbers and retrieves study information

---

## 8. Pharma Companies

### FlashText Lexicon Matching

- Loads a curated list of pharmaceutical company names
- Matches both full names and common abbreviations (e.g., "AstraZeneca", "AZ")
- **Canonical name resolution**: Maps variants to official company names (e.g., "Pfizer Inc.", "Pfizer, Inc." -> "Pfizer")

---

## 9. Document Metadata

Document metadata extraction combines file system data, PDF properties, and LLM classification.

### Extraction Chain (C09)

1. **File system metadata**: Filename, file size, creation/modification dates
2. **PDF properties**: Title, author, creator, producer, creation date from PDF metadata dictionary
3. **LLM classification** (Haiku): Classifies document type (clinical trial, guideline, marketing material, regulatory, scientific article, etc.)
4. **LLM description** (Haiku): Generates a 2-4 sentence document description
5. **Date extraction**: Priority fallback chain across multiple date sources

### Date Extraction Priority

Dates are extracted with a priority fallback chain:

1. PDF metadata dates (creation, modification)
2. Document text dates (publication date, copyright year)
3. Reference section dates (most recent reference year)
4. File system dates (last resort)

---

## 10. Feasibility

Feasibility extraction uses Claude Sonnet for complex clinical reasoning.

### LLM-Driven Extraction (C11)

A single Sonnet call extracts structured feasibility data from clinical trial documents:

- **Eligibility criteria**: Inclusion/exclusion criteria with lab values, diagnosis confirmation requirements, severity grades, and logical expressions (AND/OR grouping)
- **Screening flow**: Expected screen failure rates, randomization ratios, enrollment targets
- **Study design**: Phase, arms, blinding, comparator, primary endpoint
- **Epidemiology**: Prevalence, incidence, geographic distribution
- **Operational burden**: Visit frequency, procedures per visit, blood draw volumes
- **Visit schedules**: Screening, treatment, and follow-up visit timelines
- **Background therapy**: Permitted and prohibited concomitant medications

### NCT Enrichment

When NCT numbers are detected in the document:

- ClinicalTrials.gov API provides structured study data
- Enriches eligibility criteria, study design, and enrollment information
- Cross-references document text with official trial registration

---

## 11. Recommendations

Clinical guideline recommendations are extracted through both text-based LLM and visual VLM analysis.

### LLM Text Extraction (C32)

- Claude Sonnet processes recommendation sections
- Extracts: population, action, evidence level, strength, dosing constraints, duration, alternatives
- Infers evidence level from language cues ("strong evidence suggests" -> HIGH)
- Infers recommendation strength from hedging language ("may consider" -> CONDITIONAL)

### VLM Table Extraction (C33)

- Renders recommendation tables as page images
- Claude Sonnet vision extracts structured recommendation data from table images
- Correlates VLM-extracted evidence levels/strengths with text-extracted recommendations
- Handles garbled PDF table parsing by falling back to visual analysis

### Evidence Level and Strength Parsing

| Evidence Level | Language Cues |
|---------------|---------------|
| HIGH | "high-quality evidence", "level 1", "grade A" |
| MODERATE | "moderate evidence", "level 2", "grade B" |
| LOW | "low-quality evidence", "level 3", "grade C" |
| VERY_LOW | "very low", "level 4", "grade D" |
| EXPERT_OPINION | "expert consensus", "good practice" |

| Strength | Language Cues |
|----------|---------------|
| STRONG | "we recommend", "should be used", "is recommended" |
| CONDITIONAL | "we suggest", "may be considered", "can be used" |
| WEAK | "insufficient evidence", "no recommendation" |

---

## 12. Care Pathways

Care pathway extraction analyzes clinical decision trees from guideline flowcharts.

### Flowchart Analysis (C17)

- Detects flowchart/algorithm figures in the document
- Uses VLM (Sonnet) to analyze flowchart structure
- Extracts nodes (actions, decisions, assessments) and edges (transitions with conditions)
- Identifies treatment phases (induction, maintenance, relapse)

### Node/Edge Extraction

- **Action nodes**: Treatment steps (prescribe drug, perform procedure)
- **Decision nodes**: Clinical decision points (response assessment, severity check)
- **Edge conditions**: Transition criteria ("Yes/No", "organ-threatening", "relapse")

### VLM Enrichment (C19)

- Renders flowchart figures as high-resolution images
- Claude Vision extracts structured pathway data
- Maps drugs, doses, durations to individual pathway nodes
- Links pathway steps to corresponding recommendations

---

## 13. Validation Layer

The validation layer (D_validation/) provides LLM-based verification for all entity types.

### D02 LLM Engine

The central validation engine coordinates all LLM-based verification:

- **Model tier routing**: Routes each call to the appropriate model (Haiku vs Sonnet) based on task complexity
- **17 distinct call types**: Each with a configured model tier in config.yaml
- **Prompt registry**: Versioned prompts for each validation task
- **Response parsing**: Structured JSON extraction from LLM responses

### Fast-Reject Pre-Screening

Before expensive Sonnet validation, Haiku performs quick filtering:

- Confidence threshold: 0.85 -- only rejects if very confident
- Reduces Sonnet call volume by filtering obvious false positives
- Cost savings: ~3-5x cheaper than running everything through Sonnet

### Batch Validation

For abbreviations and other high-volume entities:

- Groups 15-20 candidates per LLM call
- In-memory caching by (entity_key) to avoid re-validation
- 100ms delay between batches to respect rate limits
- Fallback to single validation on batch errors

### Confidence Scoring

All validated entities receive a confidence score (0.0 - 1.0):

- Combined from generator confidence, LLM confidence, and heuristic adjustments
- Threshold-based filtering: entities below minimum confidence are dropped
- Context-dependent: same entity may score differently in different documents

---

## 14. Normalization Layer

The normalization layer (E_normalization/) maps validated entities to standard ontologies and deduplicates.

### Deduplication (E07)

- Groups entities by canonical identifier (not string matching)
- Merges evidence from multiple generators
- Preserves the highest-confidence mention as primary
- Tracks all source locations for provenance

### Ontology Mapping

| Entity Type | Target Ontologies |
|-------------|-------------------|
| Diseases | MONDO, Orphanet, ICD-10, ICD-11, SNOMED, UMLS, MeSH |
| Drugs | RxNorm (RxCUI), MeSH, DrugBank, NDC, UNII |
| Genes | HGNC, Entrez Gene, Ensembl, OMIM, UniProt |
| Abbreviations | Meta-Inventory, UMLS |

### External API Enrichment

| API | Purpose | Entity Types |
|-----|---------|--------------|
| PubTator3 | MeSH codes, aliases, cross-references | Diseases, Drugs, Genes |
| ClinicalTrials.gov | NCT metadata, study information | Feasibility, Citations |
| RxNorm API | Drug codes, classifications | Drugs |
| HGNC REST API | Gene symbols, identifiers | Genes |

### Term Mapping (E01)

- Normalizes surface forms to canonical representations
- Handles synonym resolution across ontologies
- Maps informal names to standard nomenclature (e.g., "lupus" -> "systemic lupus erythematosus")

### Disambiguation (E03)

- Resolves ambiguous terms using document context
- Example: "ALS" could be amyotrophic lateral sclerosis or antiphospholipid syndrome
- Uses co-occurring entities and document classification to disambiguate
