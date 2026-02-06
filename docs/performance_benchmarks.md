# ESE Pipeline v0.8 -- Performance & Benchmarks

> **Date**: February 2026
> **Pipeline version**: v0.8

This is the authoritative reference for all ESE pipeline performance data: test suite health, gold standard benchmark results, pipeline throughput, cost profile, and comparison with state-of-the-art approaches.

---

## Table of Contents

1. [Test Suite Health](#1-test-suite-health)
2. [Gold Standard Benchmarks](#2-gold-standard-benchmarks)
3. [Pipeline Throughput & Cost](#3-pipeline-throughput--cost)
4. [SOTA Comparison](#4-sota-comparison)
5. [Measurement Infrastructure](#5-measurement-infrastructure)

---

## 1. Test Suite Health

### 1.1 Overview

| Metric | Value |
|--------|-------|
| **Total test functions** | **1,474** |
| **Total test files** | 60 (in `K_tests/`) |
| Test framework | pytest |
| Python version | 3.12+ |
| Type checker | mypy (strict) |
| Linter | ruff |

### 1.2 Running the Full Suite

```bash
# All tests (1,474 tests, ~1 second)
cd corpus_metadata && python -m pytest K_tests/ -v

# Type checking
cd corpus_metadata && python -m mypy .

# Linting
cd corpus_metadata && python -m ruff check .
```

All three must pass before any change is considered complete.

### 1.3 Test Distribution by Pipeline Layer

```
Layer                   Files   Tests   Coverage Level
------------------------------------------------------------
A_core (Domain Models)     11     300   Comprehensive (56% behavioral)
B_parsing (PDF Parsing)    13     248   Comprehensive (41% behavioral)
C_generators (Extraction)   9     271   Good (17% behavioral, 77% import)
D_validation (LLM)          5      99   Complete (100% behavioral)
E_normalization              4      67   Good (22% behavioral)
F_evaluation                 2      43   Good (67% behavioral)
G_config                     1      35   Complete (100% behavioral)
H_pipeline                   2      36   Good (40% behavioral)
I_extraction                 2      24   Moderate (67% behavioral)
J_export                     2      24   Moderate (40% behavioral)
Z_utils                      4     112   Comprehensive (33% behavioral)
Cross-cutting/imports        5      61   Complete (all layers)
Import health checks         -      54   All modules importable
------------------------------------------------------------
TOTAL                       60   1,474
```

### 1.4 Test Categories

| Type | Count | % |
|------|-------|---|
| Pure unit tests (no I/O, no mocks) | ~780 | 53% |
| Unit tests with filesystem I/O | ~150 | 10% |
| Unit tests with mocking | ~180 | 12% |
| Integration tests (cross-layer) | ~50 | 3% |
| Import smoke tests | ~100 | 7% |
| False-positive filter tests | 124 | 8% |
| Pattern extraction tests | 86 | 6% |
| Deduplication tests | 52 | 4% |

### 1.5 Key Testing Strengths

- **124 false-positive filter tests** across diseases (K26: 43), drugs (K27: 38), and genes (K29: 43) -- directly protect extraction precision
- **86 pattern extraction tests** for abbreviations (K23: 42) and feasibility (K28: 44) -- protect recall
- **47 unicode normalization tests** (K54) covering all dash variants, mojibake detection, and combined transformations
- **All 4 D_validation modules** have behavioral test coverage (100%)
- **Import health checks** verify all 98+ modules are importable with valid `__all__` exports

---

## 2. Gold Standard Benchmarks

The pipeline is evaluated against four gold standard corpora covering three entity types: drugs, genes, diseases, and abbreviations.

### 2.1 Cross-Benchmark Summary

| Benchmark | Entity | Docs | Precision | Recall | F1 | Perfect Docs | Status |
|-----------|--------|------|-----------|--------|-----|-------------|--------|
| **CADEC** | Drugs | 311 | **93.5%** | **92.9%** | **93.2%** | 284/311 (91.3%) | Production-ready |
| **BC2GM** | Genes | 100 | **90.3%** | 12.3% | 21.7% | 4/100 (4.0%) | Validated methodology |
| **NLP4RARE** | Diseases | 1,040 | **77.0%** | **74.4%** | **75.7%** | 315/1040 (30.3%) | Active improvement |
| **NLP4RARE** | Abbreviations | 1,040 | 46.8% | **88.0%** | **61.1%** | -- | Active improvement |

### 2.2 CADEC Drug Detection -- Production-Ready (F1=93.2%)

**Corpus**: CADEC (CSIRO Adverse Drug Event Corpus) -- social media posts from AskaPatient forums
**Scope**: 1,248 documents (937 train, 311 test), 1,198 drug annotations
**Challenge level**: High -- informal text, misspellings, consumer brand names, supplements, noisy gold annotations

#### Results (Test Split: 311 documents)

| Metric | Value |
|--------|-------|
| True Positives | 273 |
| False Positives | 19 |
| False Negatives | 21 |
| **Precision** | **93.5%** |
| **Recall** | **92.9%** |
| **F1 Score** | **93.2%** |
| Perfect documents | 284/311 (91.3%) |

#### Detection Architecture

```
C07 DrugDetector
  ├── FlashText layer 1: ChEMBL approved drugs (23K terms)
  ├── FlashText layer 2: RxNorm drug vocabulary (132K terms)
  ├── FlashText layer 3: Consumer drug variants (~200 terms)
  └── scispaCy NER: Biomedical named entity recognition
       ↓
C25 DrugFilterer + C26 FP Constants
  ├── BIOLOGICAL_ENTITIES: cholesterol, ldl, hdl, triglycerides
  ├── BODY_PARTS: hip, disc, spine
  ├── EQUIPMENT_PROCEDURES: blood test, catheter
  ├── COMMON_WORDS: sleep, cold, cough, statin
  └── NER_FALSE_POSITIVES: expanded false positive set
       ↓
Drug matching: exact → substring → brand/generic → fuzzy (0.8)
```

#### Error Analysis

**False Positives (19)**: Mostly correct detections not present in gold standard
- Lipitor, Zocor, ezetimibe, CoQ10 -- legitimate drugs the gold missed
- Supplements flagged correctly but not in gold
- Analysis: Pipeline is *more correct* than the gold for many FPs

**False Negatives (21)**: Mostly gold noise and edge cases
- Noisy gold: "Stopped", "SATIN" (misspelling of statin)
- Dedup issues: same drug counted multiple times in gold
- Spacing variants: "Gas - X", "CoQ 10"
- Class-level terms: "statin" (drug class, not specific drug)

#### Improvement Trajectory

| Version | Precision | Recall | F1 | Change |
|---------|-----------|--------|-----|--------|
| Baseline | 45.0% | 71.6% | 55.2% | -- |
| + FP filter expansion | 89.4% | 71.8% | 79.6% | +24.4 F1 |
| + Author pattern fix | -- | -- | -- | Reduced FPs from author initials |
| + Consumer variants | 93.5% | 92.9% | 93.2% | +13.6 F1 |

Key fixes applied:
1. **FP filter**: Added biological entities, body parts, equipment/procedures, common words
2. **Author pattern bug**: Removed `re.IGNORECASE` on author initials regex (matched lowercase words "to", "i", "at")
3. **Consumer variants**: CONSUMER_DRUG_VARIANTS + CONSUMER_DRUG_PATTERNS in C26, loaded as FlashText layer
4. **Brand/generic equivalences**: 30+ mapping pairs (acetaminophen ↔ Tylenol, atorvastatin ↔ Lipitor, etc.)
5. **Restored legitimate drugs**: Removed morphine/fentanyl from NER_FALSE_POSITIVES, potassium/magnesium from BIOLOGICAL_ENTITIES

#### Running CADEC Evaluation

```bash
cd corpus_metadata && python ../gold_data/CADEC/evaluate_cadec_drugs.py --split=test
# Options: --split=test|train|all, --max-docs=N, --seqeval
```

---

### 2.3 BC2GM Gene Detection -- Validated Methodology (P=90.3%)

**Corpus**: BioCreative II Gene Mention -- PubMed sentences with gene/protein annotations
**Scope**: 5,000 test sentences, 6,331 gene annotations (pipe-delimited text format)
**Challenge level**: Schema mismatch -- BC2GM annotates protein names; pipeline extracts HGNC symbols

#### Results (100 documents)

| Metric | Value |
|--------|-------|
| True Positives | 28 |
| False Positives | 3 |
| False Negatives | 199 |
| **Precision** | **90.3%** |
| **Recall** | **12.3%** |
| **F1 Score** | **21.7%** |
| Perfect documents | 4/100 (4.0%) |

#### Why Low Recall Is Expected

The ESE pipeline is designed to extract **HGNC gene symbols** (BRCA1, TP53, BMPR2), not broad gene/protein names. BC2GM annotates protein names like "hemoglobin", "DNA-PK", "procollagen" which are outside the pipeline's scope.

**FN breakdown** (199 missed):
- ~60% protein names (hemoglobin, insulin, collagen) -- not HGNC symbols
- ~15% gene family names (MAPK, SMADs, cadherins) -- ambiguous without context
- ~15% receptor/enzyme names (GnRH, kinase, protease) -- functional names, not symbols
- ~10% short symbols (IL, GM, AP) -- too ambiguous, filtered by gene FP filter

**FP analysis** (3 false positives):
- CAT, BCR, C3 -- all legitimate HGNC symbols used in non-gene contexts
- These are common abbreviations that happen to also be gene symbols

#### Detection Architecture

```
C16 GeneDetector
  ├── FlashText: HGNC symbols + aliases (40K terms)
  └── FlashText: Orphadata gene-disease associations
       ↓
C34 GeneFalsePositiveFilter
  ├── Common abbreviation filter
  ├── Short symbol filter (2-char)
  └── Context validation
       ↓
Gene matching: exact symbol → matched text → substring (3+ chars) → name-based
```

#### Running BC2GM Evaluation

```bash
# Generate gold data (first time only)
cd gold_data/bc2gm && python generate_bc2gm_gold.py --max 100

# Run evaluation
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
# Configure: RUN_BC2GM=True
```

See [Gene Evaluation Guide](guides/06_gene_evaluation.md) for full setup and analysis.

---

### 2.4 NLP4RARE Disease Detection -- Active Improvement (F1=75.7%)

**Corpus**: NLP4RARE-CM-UC3M -- rare disease documents with BRAT standoff annotations
**Scope**: 2,311 PDFs (317 dev, 536 test, 1,458 train), 4,123 disease annotations
**Challenge level**: High -- rare disease nomenclature, abbreviation-as-disease, redundant gold annotations

#### Results (1,040 documents, all splits)

| Metric | Value |
|--------|-------|
| True Positives | 2,994 |
| False Positives | 894 |
| False Negatives | 1,032 |
| **Precision** | **77.0%** |
| **Recall** | **74.4%** |
| **F1 Score** | **75.7%** |
| Perfect documents | 315/1,040 (30.3%) |

#### Detection Architecture

```
C06 DiseaseDetector
  ├── FlashText: General disease lexicon (29K terms)
  ├── FlashText: Orphanet rare diseases (9.5K terms)
  ├── FlashText: MONDO unified ontology (97K terms)
  ├── FlashText: Rare disease acronyms (1,640 terms)
  └── scispaCy NER: Biomedical named entity recognition
       ↓
C24 DiseaseFalsePositiveFilter
  ├── COMMON_ENGLISH_FP_TERMS: single-word hard filter
  ├── GENERIC_MULTIWORD_FP_TERMS: multi-word hard filter
  └── Confidence adjustments (MIN_ADJUSTMENT_FLOOR = -0.45)
       ↓
Disease matching: exact → substring → token overlap (65%) → synonym group → normalization → fuzzy (0.8)
```

#### Error Analysis

**False Positives (894)**:
- Symptoms classified as diseases: ataxia, hypocalcemia, nystagmus
- Clinical signs and related terms
- Gold annotation inconsistencies (both "lupus" and "systemic lupus erythematosus" annotated separately)

**False Negatives (1,032)**:
- Abbreviation-as-disease: AVM, BGS, CPEO -- abbreviations used as disease names not in lexicons
- Generic terms: "skin condition", "metabolic disorder" -- too vague for lexicon matching
- Qualified names: "Secondary APS", "Familial Mediterranean Fever" -- full qualified forms not in lexicons
- Redundant gold: gold annotates both the abbreviation and the full form as separate entities

#### 6-Step Matching Algorithm

The disease comparison in F03 uses progressive matching:

1. **Exact match**: Lowercased, quote-normalized string equality
2. **Substring match**: Gold text is a substring of extracted text (or vice versa)
3. **Token overlap**: At least 2 shared tokens and 65% overlap ratio
4. **Synonym group match**: 14 synonym groups (stroke = cerebrovascular accident, MI = myocardial infarction, etc.)
5. **Synonym normalization**: Adjectival forms (autistic → autism), suffix equivalence (syndrome ↔ disorder)
6. **Fuzzy match**: SequenceMatcher with 0.8 threshold

#### Running NLP4RARE Evaluation

```bash
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
# Default: RUN_NLP4RARE=True, NLP4RARE_SPLITS=["dev","test","train"], MAX_DOCS=None
```

---

### 2.5 NLP4RARE Abbreviation Detection (F1=61.1%)

#### Results (1,040 documents, all splits)

| Metric | Value |
|--------|-------|
| True Positives | 213 |
| False Positives | 242 |
| False Negatives | 29 |
| **Precision** | **46.8%** |
| **Recall** | **88.0%** |
| **F1 Score** | **61.1%** |

#### Error Analysis

**False Positives (242)**:
- Gene symbols detected as abbreviations: JAG1, NOTCH2, COL4A3
- Common acronyms not in gold: DNA, OMIM, MRI, PCR
- Many FPs are legitimate abbreviations the gold simply doesn't annotate

**False Negatives (29)**:
- Complex patterns: CdLS (mixed case), LD-HIV (hyphenated compound)
- Single-letter abbreviations: AI (too short for pipeline's 2-char minimum)
- Non-standard formatting in source PDFs

#### Detection Architecture

```
Abbreviation Pipeline (H02)
  ├── C01 AbbrevSyntaxCandidateGenerator: Parenthetical patterns
  ├── C04 RegexLexiconGenerator: FlashText lexicon matching (617K terms)
  ├── C05 GlossaryTableCandidateGenerator: Glossary/table extraction
  ├── C23 InlineDefinitionDetector: Non-parenthetical definitions
  └── PASO Heuristics: A (stats whitelist), B (blacklist), C (hyphenated), D (SF-only LLM)
       ↓
  D02 LLM Validation: Claude batch verification
       ↓
  E01 Term Mapper + E02 Disambiguator: Normalization
```

---

## 3. Pipeline Throughput & Cost

All measurements below are from actual pipeline logs, not estimates.

### 3.1 Per-Stage Processing Times

| Stage | Typical Range | % of Total | Notes |
|-------|-------------|------------|-------|
| PDF Parsing | 20--1,005s | 6--31% | Unstructured.io + Docling TableFormer |
| Candidate Generation | 35--393s | 7--18% | FlashText (617K terms), syntax, scispaCy NER |
| LLM Validation | 51--516s | 9--18% | Claude Sonnet 4 (batched, 100ms delay) |
| Feasibility Extraction | 18--72s | 1--6% | 6 NLP enrichers incl. EpiExtract4GARD (~25s) |
| Visual Extraction (VLM) | 22--444s | 5--14% | Vision LLM for figures, tables, flowcharts |
| Normalization | ~0s | 0% | PubTator3 + NCT enrichment (cached) |
| Export | 1--8s | <1% | JSON serialization |
| **Total per document** | **189--3,358s** | -- | **Median ~801s (13.4 min)** |

Document-level variance is extreme. A short marketing PDF (6 pages, few figures) processes in ~3 min. A dense clinical trial protocol (50+ pages, many figures/tables) takes 28--56 min. The dominant factor is document complexity.

### 3.2 Per-Document Cost Breakdown (10-document batch)

| Document | Time | API Calls | Input Tokens | Output Tokens | Cost |
|----------|------|-----------|-------------|--------------|------|
| Article -- Iptacopan C3G Trial (12pp) | 516s | 40 | 94K | 17K | $0.54 |
| Article -- AAV Guidelines | 1,460s | 90 | 218K | 49K | $1.40 |
| Marketing -- WINREVAIR EU | 189s | 25 | 56K | 9K | $0.30 |
| Protocol -- ALXN1720-oMG-303 | 1,679s | 138 | 287K | 55K | $1.69 |
| Article -- Radiomics for PH | 1,647s | 57 | 109K | 23K | $0.67 |
| Article -- TRPC Channels in PAH | 1,060s | 64 | 160K | 27K | $0.89 |
| **Batch Total (10 docs)** | **8,015s** | **713** | **1,570K** | **289K** | **$9.04** |

### 3.3 Cost Profile Summary

| Metric | Value |
|--------|-------|
| Batch cost (10 PDFs) | $9.04 |
| **Average cost per document** | **$0.90** |
| Range per document | $0.23--$1.69 |
| Average API calls per document | 71.3 |
| Average input tokens per doc | 157K |
| Average output tokens per doc | 28.9K |
| Model used in measured run | 100% Claude Sonnet 4 ($3/$15 per MTok) |

### 3.4 Cost by Call Type

| Call Type | % of Calls | % of Input Tokens | Configured Tier |
|-----------|-----------|-------------------|-----------------|
| feasibility_extraction | 35% | 28% | Sonnet |
| abbreviation_batch_validation | 13% | 21% | Haiku |
| vlm_visual_enrichment | 14% | 17% | Haiku |
| fast_reject | 8% | 10% | Haiku |
| recommendation_extraction | 6% | 6% | Sonnet |
| document_classification | 6% | 6.5% | Haiku |
| description_extraction | 6% | 2% | Haiku |
| sf_only_extraction | 9% | 6% | Haiku |

**Cost optimization opportunity**: The measured batch ran 100% on Sonnet despite config specifying Haiku for 10 of 17 call types. Correct model tier routing would yield an estimated **60--70% cost reduction** (average per-document cost: **$0.25--$0.35**).

### 3.5 Model Pricing Reference

| Model | Input (per MTok) | Output (per MTok) | Cache Read | Cache Create |
|-------|-------------------|-------------------|------------|--------------|
| Claude Sonnet 4 | $3.00 | $15.00 | $0.30 (10%) | $3.75 (125%) |
| Claude Haiku 4.5 | $1.00 | $5.00 | $0.10 (10%) | $1.25 (125%) |

### 3.6 Throughput

| Configuration | Throughput |
|---------------|-----------|
| Current (sequential, single machine) | ~4--5 docs/hour (complex) to ~20/hour (simple) |
| 10-doc PAPERS batch | 10 docs in 2h14m = ~4.5 docs/hour |
| Estimated daily capacity (24h) | ~108--120 documents |

### 3.7 Bottleneck Ranking

1. **PDF Parsing** (Unstructured.io): Up to 1,005s for complex PDFs with many tables. CPU-bound with no parallelism.
2. **LLM Validation**: Up to 516s per document. Sequential synchronous API calls with 100ms inter-batch delay.
3. **Visual Extraction**: Up to 444s. One VLM call per figure/table, sequential.
4. **No document-level parallelism**: `process_folder()` loops sequentially. No `asyncio`, `concurrent.futures`, or multiprocessing.

---

## 4. SOTA Comparison

### 4.1 Component-Level Comparison

| Dimension | ESE Pipeline v0.8 | SOTA / Achievable | Gap |
|-----------|-------------------|-------------------|-----|
| **NER speed** | 35--393s (FlashText + scispaCy + LLM) | <1s with BiomedBERT/PubMedBERT (HunFlair2, BERN2) | 100--400x slower |
| **NER accuracy (F1)** | 75.7% diseases, 93.2% drugs | Encoder NER: 82--93 F1 | At parity for drugs, below for diseases |
| **LLM validation latency** | 51--516s (synchronous, Sonnet) | Haiku 2--4x faster; async batching 5--10x throughput | 10--40x improvable |
| **PDF parsing** | 20--1,005s (Unstructured.io + Docling) | PyMuPDF native: <1s; Docling alone: 2--15s | 10--100x improvable |
| **Cost per document** | $0.90 avg (all Sonnet) | $0.25--0.35 with tier routing; $0.05--0.10 encoder-only | 3--18x reducible |
| **Parallelism** | None (sequential) | Async + multiprocessing: ~5--10x speedup | 5--10x improvable |

### 4.2 Key Context

ESE is not a standard NER pipeline. It extracts **14+ entity types**, links to **6+ ontologies**, runs clinical **feasibility analysis**, **VLM figure/table extraction**, and **care pathway mapping**. No single SOTA benchmark covers this scope. The comparison reflects what's achievable per-component, but no existing system provides equivalent breadth.

### 4.3 Projected Optimized Performance

| Metric | Current ESE | Optimized ESE | Full SOTA Rewrite |
|--------|------------|---------------|-------------------|
| Time per doc | 13.4 min avg | ~2--4 min | ~10--30s |
| Throughput | 4.5 docs/hour | 15--30 docs/hour | 120--360 docs/hour |
| Cost per doc | $0.90 | $0.25--0.35 | $0.03--0.08 |
| Entity coverage | 14+ entity types | 14+ entity types | Typically 3--5 types |
| Ontology linking | 6+ ontologies | 6+ ontologies | Usually 1--2 |

---

## 5. Measurement Infrastructure

The pipeline includes built-in performance tracking at multiple levels.

### 5.1 Stage Timing

**StageTimer** (in `orchestrator_utils.py`) provides wall-clock timing for all 16 pipeline stages. Output includes a visual bar chart showing relative time per stage.

### 5.2 LLM Usage Tracking

**LLMUsageTracker** (in `D02_llm_engine.py`) tracks per-call metrics:
- Model used, call_type
- Input, output, cache read tokens
- Cost calculation with cache pricing
- Cumulative session totals

### 5.3 Persistent Storage

**SQLite database** (`corpus_log/usage_stats.db`) with three tables:

| Table | Tracks |
|-------|--------|
| `llm_usage` | Every LLM API call: model, call_type, tokens, cost, timestamp |
| `lexicon_usage` | Lexicon loading: source, term count, load time |
| `datasource_usage` | External API calls: service, endpoint, response time |

### 5.4 Console and File Logs

- **Console logs**: Tee'd to `corpus_log/pipeline_run_*.log` with full stage breakdowns
- **Validation logs**: Per-candidate JSONL in `corpus_log/RUN_*.jsonl`
- **Per-document summaries**: Cost, entity counts, timing printed at end of each document

### 5.5 Querying Performance Data

```bash
# View LLM usage by call type
sqlite3 corpus_log/usage_stats.db "SELECT call_type, COUNT(*), SUM(cost) FROM llm_usage GROUP BY call_type"

# View average cost per document
sqlite3 corpus_log/usage_stats.db "SELECT AVG(cost) FROM llm_usage GROUP BY document_id"
```

---

## References

- [Do LLMs Surpass Encoders for Biomedical NER? (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12335919/)
- [HunFlair2 Cross-corpus NER Evaluation (2024)](https://academic.oup.com/bioinformatics/article/40/10/btae564/7762634)
- [Benchmarking LLM-based IE for Medical Documents (Jan 2026)](https://www.medrxiv.org/content/10.64898/2026.01.19.26344287v1.full)
- [Claude Haiku 4.5 Announcement](https://www.anthropic.com/news/claude-haiku-4-5)
- [PubTator 3.0 (2024)](https://academic.oup.com/nar/article/52/W1/W540/7640526)
- [Clinical Pipeline Speed-Accuracy Trade-offs (2025)](https://www.johnsnowlabs.com/clinical-de-identification-at-scale-pipeline-design-and-speed-accuracy-trade-offs-across-infrastructures/)
