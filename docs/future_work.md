# ESE Pipeline -- Future Work & Development Roadmap

> **Date**: February 2026
> **Current version**: v0.8

This document outlines proposed improvements, development directions, and research opportunities for the ESE pipeline. Items are organized by timeline and priority.

---

## Table of Contents

1. [Short-Term Improvements (Next Release)](#1-short-term-improvements)
2. [Medium-Term Development (v0.9)](#2-medium-term-development-v09)
3. [Long-Term Vision (v1.0+)](#3-long-term-vision-v10)
4. [New Entity Types & Capabilities](#4-new-entity-types--capabilities)
5. [Research Directions](#5-research-directions)

---

## 1. Short-Term Improvements

Targeted fixes that improve accuracy, cost, and reliability with minimal architectural changes.

### 1.1 Extraction Accuracy

#### Disease Detection (F1: 75.7% → target 80%+)

**Reduce False Negatives (1,032 FNs):**
- **Abbreviation-as-disease resolution**: AVM, BGS, CPEO are abbreviations that the gold standard treats as diseases. Cross-reference abbreviation pipeline output: if an abbreviation expands to a known disease, add the abbreviation as a disease entity.
- **Qualified disease names**: "Secondary APS", "Familial Mediterranean Fever" -- add qualified form generation to lexicon loaders (prepend common qualifiers to existing disease entries).
- **Generic term handling**: "skin condition", "metabolic disorder" -- these are deliberately excluded by the FP filter. Some may be worth adding back with low confidence scores.

**Reduce False Positives (894 FPs):**
- **Symptom vs disease disambiguation**: ataxia, hypocalcemia, nystagmus -- use context-aware classification (is the term describing a diagnosis or a symptom?). Could leverage scispaCy entity type or add a symptom lexicon for exclusion.
- **Gold annotation artifacts**: Both "lupus" and "systemic lupus erythematosus" are separately annotated. Account for this in matching by checking if a gold FN is a substring of an existing TP.

#### Abbreviation Detection (F1: 61.1% → target 70%+)

**Reduce False Positives (242 FPs):**
- **Gene symbol filtering**: JAG1, NOTCH2, COL4A3 detected as abbreviations -- cross-reference against HGNC gene list and suppress abbreviations that are known gene symbols (unless they have an explicit long form definition in the document).
- **Common non-abbreviation acronyms**: DNA, OMIM, MRI, PCR -- expand the SF blacklist in `heuristics.sf_blacklist` with standard scientific/medical acronyms that are not document-specific abbreviations.

**Reduce False Negatives (29 FNs):**
- **Mixed-case patterns**: CdLS, LD-HIV -- relax the abbreviation pattern matcher to accept mixed-case forms where the first character is uppercase.
- **Single-letter expansion**: Some gold abbreviations use single characters (AI) -- consider allowing 1-character SFs for specific patterns.

#### Drug Detection (F1: 93.2% -- maintenance mode)

- **Spacing variants**: "Gas - X", "CoQ 10" -- add whitespace normalization before FlashText matching.
- **Extreme misspellings**: Out of scope for lexicon matching; would require fuzzy matching at detection time (performance tradeoff).

### 1.2 Cost Optimization

**Fix model tier routing** (estimated 60-70% cost reduction):
- Current issue: All 17 call types route to Sonnet despite 10 being configured for Haiku.
- Fix: Verify `resolve_model_tier()` in D02_llm_engine.py correctly reads config.yaml model_tiers.
- Expected outcome: Average per-document cost drops from $0.90 to $0.25-0.35.

**Improve prompt caching** (estimated 40% token cost reduction):
- Current cache hit rate: 5.2%.
- Fix: Restructure LLM prompts to share system prompt prefixes across call types, enabling the Anthropic prompt caching mechanism.
- Expected outcome: Cache hit rate 50-80%, reducing effective token costs.

### 1.3 Error Handling & Reliability

**Critical silent failure fixes:**
- **scispaCy NER catch-all** (`C04_strategy_flashtext.py:553`): Replace `except Exception: pass` with proper logging. This is the highest-severity silent failure -- entire biomedical NER silently disabled on any error.
- **UMLS KB lookup** (`C04_strategy_flashtext.py:497`): Same pattern, add logging.

**Enrichment failure visibility:**
- Flag entities that failed PubTator/NCT enrichment (currently indistinguishable from "enrichment not needed").
- Add `enrichment_status` field to entity models: `enriched`, `failed`, `skipped`.

**Per-stage error isolation:**
- Currently only 4 of 16 orchestrator stages have try/except. A failure in disease detection prevents all downstream processing (genes, drugs, authors, citations, feasibility, export).
- Wrap stages 1-11 with individual try/except blocks so partial results can be exported even when some stages fail.

**Structured error tracking:**
- Add `pipeline_errors` table to existing SQLite database (`usage_stats.db`).
- Log every caught exception with document_id, stage, error_type, recovery_action.
- Enable failure rate monitoring and identification of flaky stages.

---

## 2. Medium-Term Development (v0.9)

Architectural improvements that require multi-file changes but stay within the current design.

### 2.1 Performance & Scalability

**Async LLM calls** (3-5x validation speedup):
- Replace synchronous `anthropic.Anthropic()` with `anthropic.AsyncAnthropic()`.
- Use `asyncio.gather()` to parallelize validation batches.
- Impact: LLM validation stage drops from 51-516s to ~15-150s per document.

**Document-level parallelism** (5-10x batch throughput):
- Use `concurrent.futures.ProcessPoolExecutor` for PDF parsing (CPU-bound).
- Use async event loop for LLM API calls (I/O-bound).
- Impact: Batch throughput increases from 4.5 docs/hour to 20-45 docs/hour.

**Smarter PDF parsing**:
- For simple layouts (marketing materials, newsletters), use PyMuPDF native extraction (10-100x faster than Unstructured.io).
- Auto-detect document complexity: page count, figure count, table count.
- Route simple documents to fast path, complex documents to full Unstructured.io + Docling pipeline.
- Expected impact: Median processing time drops from 13.4 min to ~4-6 min per document.

**Pipeline checkpointing**:
- Save intermediate state after each pipeline stage to disk.
- On failure, resume from last successful stage instead of reprocessing from scratch.
- Critical for expensive documents: a 56-minute protocol that fails at export (stage 16/16) currently requires full reprocessing. Checkpointing recovers in seconds.

### 2.2 Testing Infrastructure

**End-to-end pipeline tests**:
- Create a test that runs `process_pdf()` with mocked LLM, a small reference PDF, and mocked external APIs.
- Verify output structure, entity counts, and export file creation.
- Critical gap: currently no test exercises the inter-layer interfaces.

**Snapshot testing** (Syrupy):
- Capture full JSON output from processing a reference document.
- Compare against saved snapshot on every test run.
- Catches export format regressions, entity count changes, unintentional behavior changes.

**CI/CD integration** (GitHub Actions):
- Run `pytest`, `mypy`, `ruff` on every push and PR.
- Run gold standard evaluation as a separate workflow (triggered on relevant file changes).
- Block merges on failing checks.

**Coverage measurement**:
- Enable pytest-cov with minimum threshold (start at 40%, increase over time).
- Focus coverage efforts on critical untested paths: orchestrator, abbreviation pipeline, enrichers.

**Behavioral NER tests**:
- **Invariance tests**: Same entity in different contexts must be extracted.
- **Directional tests**: Adding definition context must maintain or improve confidence.
- **Minimum functionality tests**: Curated set of 50-100 critical entity + context pairs that must always pass.

### 2.3 New Gold Standards

**Cross-corpus disease validation:**
- Add NCBI Disease Corpus (793 PubMed abstracts, 6,892 disease mentions) for cross-corpus validation.
- Add BC5CDR disease track (1,500 abstracts, 4,182 disease mentions) for additional coverage.
- Validates that improvements on NLP4RARE generalize to other medical text domains.

**Author & citation gold:**
- Manually annotate 50+ documents with author names, affiliations, ORCIDs.
- Annotate citation PMID/DOI/NCT references for validation.
- Currently no evaluation exists for these entity types.

**Feasibility gold:**
- Create structured extraction benchmarks from clinical trial protocols.
- Annotate eligibility criteria, study design, epidemiology data.
- Enables evaluation of the most complex extraction task in the pipeline.

### 2.4 Evaluation Enhancements

**Slice-based analysis:**
- Break down performance by document type (article, protocol, guideline, marketing).
- Analyze performance by entity length (short vs long disease names).
- Identify document categories where the pipeline underperforms.

**Error categorization:**
- Classify extraction errors by type: boundary errors (partial match), type confusion (disease detected as symptom), context errors (correct entity, wrong section).
- Guides targeted improvement: boundary errors suggest lexicon gaps, type confusion suggests filter issues.

**Automated evaluation in CI:**
- Run gold standard evaluation on every PR that modifies C_generators/, C24-C34 FP filters, or E_normalization/.
- Report F1 changes as PR comments.
- Prevent merges that decrease F1 by more than 1%.

---

## 3. Long-Term Vision (v1.0+)

Major architectural evolution and platform capabilities.

### 3.1 Architecture Evolution

These items build on existing design plans in `docs/plans/`:

**A_core refactoring** (per `2026-02-02-a-core-refactoring.md`):
- Unified entity base class reducing per-entity-type code duplication.
- Stronger type invariants with Pydantic validators.
- Cleaner serialization boundaries.

**B_parsing refactoring** (per `2026-02-02-b-parsing-refactoring.md`):
- Pluggable parser backends (PyMuPDF, Unstructured, Docling, marker-pdf).
- Layout-aware extraction using page geometry.
- Faster table detection pipeline.

**Visual extraction redesign** (per `2026-01-31-visual-extraction-redesign.md`):
- 4-stage pipeline: Detection + FAST Structure → Rendering + Caption → Triage + VLM → Document-Level Resolution.
- FAST/ACCURATE escalation: most visuals processed cheaply, VLM reserved for complex cases.
- Unified ExtractedVisual model with rich metadata.

**Z_utils refactoring** (per `2026-02-02-z-utils-refactor-design.md`):
- Cleaner separation of concerns between API client, caching, and tracking.
- Consistent error handling patterns across all utility modules.

### 3.2 API & Model Resilience

**Circuit breaker for external APIs:**
- Track failure rate for Claude, PubTator3, ClinicalTrials.gov.
- After N failures in a time window, stop calling the degraded service.
- Auto-recover with probe calls after timeout.
- Impact: Prevents accumulated retry delays during API outages (currently 60s sleep per 429 response).

**Model fallback chain:**
```
feasibility_extraction: Sonnet 4 → Haiku 4.5 → cached/heuristic fallback
abbreviation_validation: Haiku 4.5 → local encoder → skip validation
vlm_table_extraction: Sonnet 4 → Haiku 4.5 → Docling-only extraction
```
Eliminates single-provider dependency. During Sonnet outages, extraction degrades gracefully instead of failing.

**Pre-flight health checks:**
- Probe all external dependencies before starting batch processing.
- Test Claude API, PubTator3, ClinicalTrials.gov connectivity.
- Check disk space, verify lexicon files, validate config.
- Fail fast before spending 84-1,005s on PDF parsing.

### 3.3 Local Model Integration

**BiomedBERT/PubMedBERT for fast NER pre-screening:**
- Use a local transformer model for initial entity detection.
- Only send ambiguous cases to Claude for validation.
- Expected impact: 10-50x reduction in LLM validation calls.

**Fine-tuned BERT for entity classification:**
- Train a small classifier on the pipeline's own TP/FP data.
- Replace or supplement the rule-based FP filters (C24, C25, C34).
- Data source: accumulated evaluation results provide labeled training data.

**Active learning loop:**
- Use FP/FN analysis from gold standard evaluation to identify systematic gaps.
- Automatically expand lexicons with confirmed FN terms.
- Automatically add confirmed FP terms to filter lists.
- Requires human-in-the-loop approval for safety.

### 3.4 Production Operations

**Observability:**
- OpenTelemetry traces for end-to-end request tracking.
- Prometheus metrics for throughput, latency, error rates.
- Grafana dashboards for real-time monitoring.

**Dead letter queue:**
- Failed documents queued for review and reprocessing.
- Distinguishes "document failed" from "document had no entities."
- Enables targeted debugging of problematic PDFs.

**Idempotent reprocessing:**
- Safe re-runs without entity duplication.
- Content-based deduplication using document fingerprints.
- Incremental updates: only process changed or new documents.

**Web dashboard:**
- Monitor extraction quality (P/R/F1 trends over time).
- Track cost per document, per entity type.
- Review individual document results with entity highlighting.
- One-click reprocessing of failed documents.

---

## 4. New Entity Types & Capabilities

### 4.1 Biomarker Extraction

**Motivation**: Biomarkers are increasingly central to clinical trial eligibility and rare disease diagnosis. Currently captured incidentally through disease/gene extraction but not as a first-class entity.

**Scope**: Extract biomarker names, associated conditions, threshold values, and measurement methods.

**Implementation path**: New A_core model → C_generators lexicon (UniProt, biomarker databases) → FP filter → E_normalization mapping.

### 4.2 Study Outcome Extraction

**Motivation**: Efficacy endpoints, p-values, effect sizes, and confidence intervals are critical for systematic reviews and meta-analyses.

**Scope**: Extract primary/secondary endpoints, statistical results (hazard ratios, odds ratios, p-values), and study conclusions.

**Implementation path**: Pattern-based detection (regex for p-values, CIs) → LLM extraction for structured endpoint data → table parsing for results tables.

### 4.3 Patient Population Extraction

**Motivation**: Demographics, comorbidities, and inclusion/exclusion criteria are essential for trial feasibility and patient matching.

**Scope**: Extract age ranges, gender distributions, disease severity, comorbidity profiles, and geographic distributions.

**Implementation path**: Extend feasibility extraction (C11) with structured population models. Use table extraction (VLM) for demographics tables.

### 4.4 Temporal Event Extraction

**Motivation**: Treatment timelines, disease progression milestones, and study visit schedules are critical for understanding clinical workflows.

**Scope**: Extract temporal expressions, event sequences, treatment duration, and follow-up periods.

**Implementation path**: TimeML-inspired annotation model → pattern detection for temporal expressions → LLM extraction for event ordering.

### 4.5 Multi-Document Synthesis

**Motivation**: Cross-referencing entities across document sets enables systematic review capabilities and therapeutic area intelligence.

**Scope**: Entity linking across documents, contradiction detection, evidence aggregation, knowledge graph construction.

**Implementation path**: Post-processing layer on top of per-document extraction → entity resolution across documents → graph database storage.

---

## 5. Research Directions

### 5.1 Retrieval-Augmented Extraction

Use extracted entities to query external knowledge bases in real-time during extraction. For example:
- Extract "Alagille syndrome" → query MONDO → confirm as rare disease (MONDO:0008497) → auto-enrich with inheritance pattern, prevalence, gene associations.
- Could replace or supplement PubTator3 enrichment with more targeted queries.

### 5.2 Multi-Lingual Support

Extend extraction beyond English clinical documents:
- Key target languages: French, German, Spanish, Japanese (largest clinical trial output after English).
- Approach: Multilingual BERT models for NER, language-specific lexicons, Claude's multilingual capabilities for validation.
- Challenge: Ontology linking (MONDO, RxNorm) is primarily English-based.

### 5.3 Ontology-Guided Extraction

Use MONDO/ORPHA ontology hierarchy to improve rare disease recognition:
- If "Duchenne muscular dystrophy" is detected, also recognize its parent terms ("hereditary motor neuron disease") and child terms.
- Use ontology relationships to validate extraction: if both "DMD" and "dystrophin" are found in the same document, increase confidence in both.
- Could improve recall for rare diseases that have many synonyms in the ontology.

### 5.4 Confidence Calibration

Ensure reported confidence scores are well-calibrated against actual accuracy:
- Current confidence scores are assigned by heuristic rules and LLM responses.
- Calibration study: compare confidence distributions against gold standard TP/FP labels.
- Apply Platt scaling or isotonic regression to calibrate scores.
- Goal: a confidence score of 0.9 should mean ~90% probability of being correct.

### 5.5 LLM-as-Judge Evaluation

Use a separate LLM to evaluate extraction quality without human annotation:
- Present extracted entities alongside source text to a judge model.
- Judge rates each entity as correct/incorrect/uncertain.
- Could enable continuous evaluation on unlabeled documents.
- Validated against gold standard agreement rates before deployment.

### 5.6 Agentic Extraction Pipeline

Replace the fixed 16-stage pipeline with an agent-based architecture:
- Agent decides which extraction strategies to apply based on document type.
- Agent can request additional LLM calls for ambiguous entities.
- Agent can route documents to specialized sub-pipelines (e.g., clinical trial protocol pipeline vs guideline pipeline).
- Enables adaptive extraction that improves with each document processed.

---

## Priority Matrix

| Initiative | Impact | Effort | Priority |
|-----------|--------|--------|----------|
| Fix model tier routing | High (60-70% cost reduction) | Low | **P0** |
| scispaCy silent failure fix | High (reliability) | Low | **P0** |
| Per-stage error isolation | High (reliability) | Medium | **P0** |
| Disease FN reduction (abbrev-as-disease) | Medium (F1 +3-5%) | Medium | **P1** |
| Abbreviation FP reduction (gene symbols) | Medium (F1 +5-10%) | Low | **P1** |
| Prompt caching improvement | Medium (40% cost saving) | Medium | **P1** |
| Async LLM calls | High (3-5x speed) | Medium | **P1** |
| CI/CD setup | High (development velocity) | Medium | **P1** |
| Document-level parallelism | High (5-10x throughput) | High | **P2** |
| Pipeline checkpointing | Medium (failure recovery) | High | **P2** |
| End-to-end tests | High (regression prevention) | High | **P2** |
| New gold standards | Medium (validation breadth) | High | **P2** |
| Local model integration | High (cost + speed) | Very High | **P3** |
| Visual extraction redesign | Medium (quality) | Very High | **P3** |
| Multi-document synthesis | High (new capability) | Very High | **P3** |
| Agentic pipeline | Transformative | Very High | **P4** |

---

*Generated: February 2026*
*Pipeline version: v0.8*
