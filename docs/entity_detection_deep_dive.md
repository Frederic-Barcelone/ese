# Entity Detection -- How It Works

> **Pipeline version**: v0.8

---

## 1. Overview

Every entity type follows three layers:

```
PDF  -->  Generators (C_generators/)  -->  Validation (D_validation/)  -->  Normalization (E_normalization/)
            HIGH RECALL                      HIGH PRECISION                  STANDARDIZATION
```

**Generators** (high recall): syntax parsing, regex, FlashText (617K+ terms), scispacy NER, LLM/VLM. FPs acceptable.

**Validation** (high precision): LLM verification, PASO heuristics, FP filter lists, confidence scoring.

**Normalization**: maps to ontologies (MONDO, RxNorm, HGNC), deduplicates by canonical ID, enriches via PubTator3/ClinicalTrials.gov.

All generators implement `BaseCandidateGenerator` with `generator_type` property and `extract(doc_structure) -> List[Candidate]`.

---

## 2. Abbreviations

Four generators, PASO heuristics, and LLM validation.

### Generator 1: Schwartz-Hearst (C01)

Modified Schwartz-Hearst for parenthetical abbreviation-definition pairs. `long form (SF)` and `SF (long form)` patterns, initial verification, bidirectional. SF length: 2-12 chars.

### Generator 2: FlashText Lexicon (C04)

600K+ known pairs from six sources:

| Source | Terms |
|--------|-------|
| Meta-Inventory | 65K |
| Trial acronyms | 125K |
| Statistical terms | ~100 |
| Clinical terms | ~200 |
| Gene symbols | ~500 |
| Country codes | ~200 |

Both straight and curly apostrophe variants loaded for PDF extraction artifacts.

### Generator 3: Glossary Extraction (C05)

Detects glossary sections ("List of Abbreviations"), parses tabular/list formats.

### Generator 4: Regex Patterns

Five patterns: parenthetical forward/reverse, dash/colon, equals, inline ("also known as").

### PASO Heuristics

| Rule | Action | Examples |
|------|--------|---------|
| **PASO A** | Auto-approve statistical abbreviations | CI, HR, SD, OR, RR, BMI |
| **PASO B** | Auto-reject country codes/blacklist | US, UK, EU, DNA, RNA |
| **PASO C** | Auto-enrich hyphenated from NCT data | ANCA-PR3 |
| **PASO D** | LLM SF-only extraction for missing defs | Context-based extraction |

### LLM Validation

1. **Fast-reject** (Haiku): 0.85 confidence threshold, rejects obvious FPs cheaply
2. **Batch validation** (Haiku): 15-20 candidates per call, checks SF-LF genuineness
3. **Single validation** (Haiku): Fallback for batch failures
4. **Caching**: Results cached by (SF, LF) across documents

---

## 3. Diseases

Lexicon matching across four sources with scispacy NER and aggressive FP filtering.

### FlashText Lexicon (C06)

| Source | Terms |
|--------|-------|
| General disease lexicon | 29K |
| Orphanet | 9.5K |
| MONDO | 97K |
| Rare disease acronyms | 1,640 |

Apostrophe normalization: curly (U+2019) and straight (0x27) variants loaded for PDF quote style differences.

### scispacy NER

Uses `en_core_sci_lg` model (fallback: `en_core_sci_sm`) with UMLS linking. Catches novel disease names, rare variants, and non-standard nomenclature not in lexicons.

### False Positive Filter (C24)

- **COMMON_ENGLISH_FP_TERMS**: Single-word hard filter (standalone "syndrome", "disease", "cold", "depression")
- **GENERIC_MULTIWORD_FP_TERMS**: Multi-word hard filter ("associated with", "family history of", "loss of consciousness")
- **Confidence adjustments**: `MIN_ADJUSTMENT_FLOOR = -0.45`, skip threshold = -0.5, citation override = -0.55

### PubTator3 Enrichment

Maps to MeSH codes, retrieves aliases, links to MONDO, Orphanet, SNOMED, ICD-10, ICD-11.

---

## 4. Drugs

Five FlashText layers, compound ID regex, scispacy NER, FP filtering.

### FlashText Lexicon (C07)

| Source | Terms |
|--------|-------|
| Alexion drug list | ~50 |
| Investigational drugs | ~200 |
| FDA/ChEMBL approved | 23K |
| RxNorm | 132K |
| Consumer variants | ~500 |

Consumer variants (C26) handle informal references -- maps "tylenol" to acetaminophen, handles misspellings.

### Compound ID Regex

Trial IDs (ALXN1210, ABT-199), CAS numbers, UNII codes.

### scispacy Chemical NER

Catches novel compounds not in lexicons, links to UMLS.

### False Positive Filter (C25)

Categories: body parts, biological entities, equipment/procedures, common words, credentials, vaccine terms, disease terms. Author pattern filtering suppresses bibliography-section FPs.

### Enrichment

RxNorm (RxCUI, drug class), DrugBank (IDs, mechanism), PubTator3 (MeSH), marketing status.

---

## 5. Genes

HGNC-approved symbols with aggressive filtering for short gene name ambiguity.

### HGNC Lexicon (C16)

| Source | Terms |
|--------|-------|
| HGNC symbols + aliases | 42K+ |
| Orphadata gene-disease | ~5K |

### Pattern Matching

Standard symbols (BRCA1, TP53), gene families ("MAPK family"), variant notation (p.V600E, c.1234A>G).

### False Positive Filter (C34)

Exclusion sets: statistical terms (OR, HR, CI), units (mm, kg), clinical terms (IV, ER, ICU), countries (US, UK), credentials (MD, PhD), drug terms (ACE, ARB). Context-aware filtering for homonyms.

### Enrichment

Entrez Gene IDs, Ensembl IDs, OMIM numbers, disease associations from Orphadata.

---

## 6. Authors

### Detection (C08)

Five strategies: header patterns, affiliation blocks, contribution sections, investigator lists, corresponding author blocks.

### Identifiers

ORCID, email, 8 role types (Author, Principal Investigator, Co-Investigator, Corresponding Author, Steering Committee, Study Chair, Data Safety Board, Unknown). Handles "First Last", "Last, First", "F. Last" formats.

---

## 7. Citations

### Identifiers

| Type | Pattern |
|------|---------|
| PMID | `PMID:?\s*\d{7,8}` |
| DOI | `10\.\d{4,}/[^\s]+` |
| NCT | `NCT\d{8}` |
| PMCID | `PMC\d{7}` |

### Reference Parsing

Parses numbered reference lists, extracts authors/title/journal/year. Matches inline markers (superscripts, brackets, author-year) to full references. PubMed validates PMIDs; ClinicalTrials.gov validates NCTs.

---

## 8. Pharma Companies

FlashText lexicon with canonical name resolution ("Pfizer Inc." -> "Pfizer").

---

## 9. Document Metadata

### Extraction Chain (C09)

1. File system metadata (filename, size, dates)
2. PDF properties (title, author, creator)
3. LLM classification (Haiku): document type
4. LLM description (Haiku): 2-4 sentence summary
5. Date extraction: PDF metadata > document text > reference dates > file system

---

## 10. Feasibility

### LLM Extraction (C11)

Sonnet extracts: eligibility criteria (lab values, severity grades, logical expressions), screening flow, study design (phase, arms, blinding, endpoints), epidemiology, operational burden, visit schedules.

### NCT Enrichment

ClinicalTrials.gov API enriches eligibility, design, and enrollment when NCT numbers are detected.

---

## 11. Recommendations

### LLM Text Extraction (C32)

Sonnet extracts: population, action, evidence level, strength, dosing, duration, alternatives.

### VLM Table Extraction (C33)

Renders recommendation tables as images for Sonnet Vision when PDF table parsing is garbled.

### Evidence/Strength Mapping

Evidence: HIGH/MODERATE/LOW/VERY_LOW/EXPERT_OPINION. Strength: STRONG/CONDITIONAL/WEAK. Inferred from language cues ("we recommend" -> STRONG, "may be considered" -> CONDITIONAL).

---

## 12. Care Pathways

### Flowchart Analysis (C17)

VLM (Sonnet) extracts nodes (actions, decisions, assessments) and edges (transitions with conditions). Identifies treatment phases.

### VLM Enrichment (C19)

Claude Vision extracts structured pathway data, maps drugs/doses/durations to nodes.

---

## 13. Validation Layer (D02)

**Fast-reject**: Haiku pre-screens at 0.85 confidence threshold. **Batch validation**: 15-20 candidates/call, in-memory cache, 100ms delay, single-validation fallback. **Confidence scoring**: 0.0-1.0, combining generator + LLM confidence + heuristic adjustments.

---

## 14. Normalization Layer

### Deduplication (E07)

Groups by canonical ID, merges evidence, preserves highest-confidence mention.

### Ontology Mapping

| Entity Type | Ontologies |
|-------------|------------|
| Diseases | MONDO, Orphanet, ICD-10, ICD-11, SNOMED, UMLS, MeSH |
| Drugs | RxNorm, MeSH, DrugBank, NDC, UNII |
| Genes | HGNC, Entrez, Ensembl, OMIM, UniProt |
| Abbreviations | Meta-Inventory, UMLS |

### External APIs

PubTator3 (MeSH), ClinicalTrials.gov (NCT), RxNorm API, HGNC REST.

### Term Mapping (E01) and Disambiguation (E03)

Normalizes surface forms to canonical representations. Disambiguates using document context (e.g., "ALS" as amyotrophic lateral sclerosis vs antiphospholipid syndrome).
