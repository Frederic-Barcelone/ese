# ESE Pipeline -- Future Work

> **Date**: February 2026 | **Pipeline version**: v0.8

---

## 1. Short-Term Improvements

### 1.1 Extraction Accuracy

#### Disease Detection (F1: 74.6% -> target 80%+)

**Reduce FNs:**
- Abbreviation-as-disease (AVM, BGS, CPEO): cross-reference abbreviation output against disease lexicons.
- Qualified disease names ("Secondary APS"): prepend common qualifiers to lexicon entries.
- Generic terms ("skin condition"): consider re-adding with low confidence.

**Reduce FPs:**
- Symptom vs disease (ataxia, hypocalcemia): context-aware classification or symptom exclusion lexicon.

#### Abbreviation Detection (F1: 61.9% -> target 70%+)

**Reduce FPs:** Cross-reference against HGNC gene list to suppress gene symbols (JAG1, NOTCH2). Expand SF blacklist with standard acronyms (DNA, OMIM, MRI).

**Reduce FNs:** Relax pattern matcher for mixed-case forms (CdLS, LD-HIV).

#### Drug Detection (F1: 93.2% -- maintenance)

- Whitespace normalization for spacing variants ("Gas - X", "CoQ 10").

### 1.2 Cost Optimization

- **Model tier routing**: Fixed 2026-02-06 (3 bugs: missing config_path, wrong Haiku model ID, temp/top_p conflict). Per-doc cost reduced from $0.90 to ~$0.25-0.35.
- **Prompt caching** (est. 40% token reduction): Restructure prompts to share system prefixes. Target cache hit rate: 50-80% (currently 5.2%).

### 1.3 Error Handling

- Replace `except Exception: pass` in C04 scispaCy NER with proper logging.
- Add `enrichment_status` field (enriched/failed/skipped) to entity models.
- Per-stage try/except in orchestrator so partial results export on failure.
- Add `pipeline_errors` table to SQLite.

---

## 2. Medium-Term Development (v0.9)

### 2.1 Performance

- **Async LLM calls** (3-5x speedup): `AsyncAnthropic()` + `asyncio.gather()`.
- **Document parallelism** (5-10x throughput): `ProcessPoolExecutor` for PDF parsing, async for LLM.
- **Smart PDF routing**: Auto-detect complexity, route simple docs to PyMuPDF. Median 13.4 min -> ~4-6 min.
- **Checkpointing**: Per-stage intermediate state. Resume on failure.

### 2.2 Testing

- **E2E pipeline tests**: `process_pdf()` with mocked LLM/APIs and reference PDF.
- **Snapshot testing** (Syrupy): Compare full JSON output against saved snapshots.
- **CI/CD** (GitHub Actions): pytest/mypy/ruff on every push.
- **Coverage**: pytest-cov starting at 40% threshold.

### 2.3 New Gold Standards

- **Cross-corpus diseases**: NCBI Disease Corpus (6,892 mentions), BC5CDR (4,182 mentions).
- **Authors & citations**: Manually annotate 50+ documents.
- **Feasibility**: Structured benchmarks from clinical trial protocols.

### 2.4 Evaluation Enhancements

- Slice-based analysis by document type and entity length.
- Automated CI evaluation: block merges if F1 drops >1%.

---

## 3. Long-Term Vision (v1.0+)

### 3.1 Architecture

Planned refactoring (details in `docs/plans/`):
- **A_core**: Unified entity base class, stronger type invariants
- **B_parsing**: Pluggable parser backends (PyMuPDF, Unstructured, Docling, marker-pdf), layout-aware extraction
- **Visual extraction**: 4-stage pipeline with FAST/ACCURATE escalation
- **Z_utils**: Cleaner separation of concerns

### 3.2 API Resilience

- Circuit breaker with auto-recovery probes.
- Model fallback chain: `Sonnet 4 -> Haiku 4.5 -> heuristic fallback`.
- Pre-flight dependency checks before batch processing.

### 3.3 Local Models

- BiomedBERT pre-screening: local NER, send only ambiguous cases to Claude (10-50x fewer LLM calls).
- Fine-tuned classifier trained on pipeline TP/FP data to supplement FP filters.

### 3.4 Production Operations

- OpenTelemetry traces, Prometheus metrics, Grafana dashboards.
- Dead letter queue for failed documents.
- Web dashboard for P/R/F1 trends and cost tracking.

---

## 4. New Entity Types

| Entity | Motivation | Implementation |
|--------|-----------|----------------|
| **Biomarkers** | Central to trial eligibility and rare disease diagnosis | Lexicon (UniProt, biomarker DBs) -> FP filter -> normalization |
| **Study outcomes** | Endpoints, p-values, effect sizes for systematic reviews | Regex (p-values, CIs) + LLM extraction + table parsing |
| **Patient populations** | Demographics, comorbidities for trial feasibility | Extend C11 feasibility + VLM for demographics tables |
| **Temporal events** | Treatment timelines, disease progression milestones | TimeML-inspired model + pattern detection + LLM ordering |
| **Multi-document synthesis** | Cross-document entity linking, knowledge graphs | Post-processing layer + entity resolution + graph DB |

---

## 5. Research Directions

- **Retrieval-augmented extraction**: Query MONDO/Orphanet in real-time for auto-enrichment.
- **Multi-lingual**: Multilingual BERT + language-specific lexicons + Claude multilingual validation.
- **Ontology-guided extraction**: Use MONDO/ORPHA hierarchy for parent/child detection and confidence boosting.
- **Confidence calibration**: Platt scaling against gold TP/FP labels.
- **LLM-as-judge**: Continuous evaluation on unlabeled documents.
- **Agentic pipeline**: Agent chooses strategies per document type instead of fixed 16-stage pipeline.

---

## Priority Matrix

| Initiative | Impact | Effort | Priority |
|-----------|--------|--------|----------|
| ~~Fix model tier routing~~ | ~~High (60-70% cost cut)~~ | ~~Low~~ | **Done** |
| scispaCy silent failure fix | High (reliability) | Low | **P0** |
| Per-stage error isolation | High (reliability) | Medium | **P0** |
| Disease FN reduction | Medium (F1 +3-5%) | Medium | **P1** |
| Abbreviation FP reduction | Medium (F1 +5-10%) | Low | **P1** |
| Prompt caching | Medium (40% cost saving) | Medium | **P1** |
| Async LLM calls | High (3-5x speed) | Medium | **P1** |
| CI/CD setup | High (dev velocity) | Medium | **P1** |
| Document parallelism | High (5-10x throughput) | High | **P2** |
| Pipeline checkpointing | Medium (failure recovery) | High | **P2** |
| E2E tests | High (regression prevention) | High | **P2** |
| New gold standards | Medium (validation breadth) | High | **P2** |
| Local model integration | High (cost + speed) | Very High | **P3** |
| Visual extraction redesign | Medium (quality) | Very High | **P3** |
| Multi-document synthesis | High (new capability) | Very High | **P3** |
| Agentic pipeline | Transformative | Very High | **P4** |

---

*Generated: February 2026 | Pipeline v0.8*
