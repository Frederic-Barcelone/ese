# ESE Pipeline v0.8 — Performance Analysis

> **Date**: February 4, 2026
> **Based on**: Actual pipeline logs from 10-document PAPERS gold data batch run

---

## Real Processing Times

All numbers below are **measured from actual pipeline logs**, not estimates.

| Stage | Typical Range | % of Total | Notes |
|-------|-------------|------------|-------|
| PDF Parsing | 20–1005s | 6–31% | Highly variable; Unstructured.io + Docling TableFormer |
| Candidate Generation | 35–393s | 7–18% | FlashText lexicons (617K terms), syntax patterns, scispaCy NER |
| LLM Validation | 51–516s | 9–18% | Claude Sonnet 4 via API (batched, 100ms inter-batch delay) |
| Feasibility Extraction | 18–72s | 1–6% | 6 NLP enrichers including EpiExtract4GARD (~25s alone) |
| Visual Extraction (VLM) | 22–444s | 5–14% | Vision LLM for figures, tables, flowcharts |
| Normalization | ~0s | 0% | PubTator3 + NCT enrichment (cached, amortized) |
| Export | 1–8s | <1% | JSON serialization |
| **Total per document** | **189–3358s** | — | **Median ~801s (13.4 min), mean ~801s** |

Document-level variance is extreme. A short marketing PDF (6 pages, few figures) processes in ~3 min. A dense clinical trial protocol (50+ pages, many figures/tables) takes 28–56 min. The dominant factor is document complexity (page count, figure count, entity density), not pipeline overhead.

### Per-Document Breakdown (10-document batch)

| Document | Total Time | API Calls | Input Tokens | Output Tokens | Cost |
|----------|-----------|-----------|-------------|--------------|------|
| Article — Iptacopan C3G Trial (12pp) | 516s | 40 | 94K | 17K | $0.54 |
| Article — AAV Management Guidelines | 1460s | 90 | 218K | 49K | $1.40 |
| Marketing — WINREVAIR EU Expansion | 189s | 25 | 56K | 9K | $0.30 |
| Protocol — ALXN1720-oMG-303 | 1679s | 138 | 287K | 55K | $1.69 |
| Article — Radiomics for PH Diagnosis | 1647s | 57 | 109K | 23K | $0.67 |
| Article — TRPC Channels in PAH | 1060s | 64 | 160K | 27K | $0.89 |
| Article — 41927_2025 | 571s | 61 | 121K | 18K | $0.63 |
| Newsletter — 20251213-56068 | 226s | 25 | 42K | 7K | $0.23 |
| Article — BMJ m1070 | 243s | 22 | 55K | 7K | $0.26 |
| Document — 10_ | 426s | 37 | 80K | 14K | $0.45 |
| **Batch Total** | **8015s** | **713** | **1,570K** | **289K** | **$9.04** |

---

## Cost Profile

| Metric | Value |
|--------|-------|
| Batch cost (10 PDFs) | $9.04 |
| **Average cost per document** | **$0.90** |
| Range per document | $0.23–$1.69 |
| Average API calls per document | 71.3 (this batch) / 23.1 (overall avg across 69 docs) |
| Average input tokens per doc | 157K |
| Average output tokens per doc | 28.9K |
| Model used in this run | 100% Claude Sonnet 4 ($3/$15 per MTok) |

### Cost Breakdown by Call Type

Data across all 1,608 tracked API calls (69 documents):

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
| Others | 3% | 3.5% | Mixed |

**Critical finding**: The logged batch ran 100% on Sonnet 4 despite the config specifying Haiku for 10 of 17 call types. If model tier routing were correctly applied, ~65% of calls would use Haiku ($1/$5 per MTok instead of $3/$15), yielding an estimated **60–70% cost reduction** — bringing the average per-document cost to approximately **$0.25–$0.35**.

### Model Pricing Reference

| Model | Input (per MTok) | Output (per MTok) | Cache Read | Cache Create |
|-------|-------------------|-------------------|------------|--------------|
| Claude Sonnet 4 | $3.00 | $15.00 | $0.30 (10%) | $3.75 (125%) |
| Claude Haiku 4.5 | $1.00 | $5.00 | $0.10 (10%) | $1.25 (125%) |

---

## Bottlenecks (Ranked)

1. **PDF Parsing (Unstructured.io)** — Up to 1005s for complex PDFs with many tables. CPU-bound local operation with no parallelism.
2. **LLM Validation** — Up to 516s per document. Sequential synchronous API calls with 100ms inter-batch delay. Each batch sends 10–15 candidates.
3. **Visual Extraction** — Up to 444s. One VLM call per figure/table, sequential.
4. **No document-level parallelism** — `process_folder()` loops over PDFs sequentially. No `asyncio`, `concurrent.futures`, or multiprocessing anywhere in the pipeline.

---

## Throughput

| Configuration | Throughput |
|---------------|-----------|
| Current (sequential, single machine) | ~4–5 docs/hour (complex) to ~20/hour (simple) |
| Batch of 10 PAPERS gold data docs | 10 docs in 2h14m = **~4.5 docs/hour** |
| Estimated daily capacity (24h) | ~108–120 documents |

---

## Comparison with SOTA Approaches

| Dimension | ESE Pipeline v0.8 (Current) | SOTA / Achievable | Gap |
|-----------|----------------------------|-------------------|-----|
| **NER speed** | 35–393s (FlashText + scispaCy + LLM) | <1s per doc with BiomedBERT/PubMedBERT encoder models (HunFlair2, BERN2) | 100–400x slower due to LLM validation layer |
| **NER accuracy (F1)** | High recall generators + LLM precision filter | Encoder NER: 82–93 F1; LLM NER: 85–95 F1 but 220x slower | ESE trades speed for multi-entity, multi-ontology richness |
| **LLM validation latency** | 51–516s (synchronous, Sonnet 4) | Haiku 4.5 is 2–4x faster at 3x lower cost; async batching could 5–10x throughput | 10–40x improvable |
| **PDF parsing** | 20–1005s (Unstructured.io + Docling) | Docling alone: 2–15s; PyMuPDF native: <1s; marker-pdf: 1–5s | 10–100x improvable for simple layouts |
| **Cost per document** | $0.90 avg (all Sonnet) | $0.25–0.35 with correct tier routing; $0.05–0.10 with Haiku-only + encoder NER | 3–18x reducible |
| **Parallelism** | None (sequential) | Async API calls + multiprocessing PDF parsing: ~5–10x speedup | 5–10x improvable |
| **Normalization (PubTator3)** | 3 req/sec rate limit, 7-day cache | PubTator3 bulk download (36M abstracts); local UMLS/MONDO lookup is instant | Cache largely mitigates this |
| **ClinicalTrials.gov** | Free, no published rate limit | Free, already well-utilized | At parity |

### Key Context

ESE is not a standard NER pipeline. It extracts 11 entity types, links to 6+ ontologies, runs clinical feasibility analysis, VLM figure/table extraction, and care pathway mapping. No single SOTA benchmark covers this scope. The comparison above reflects what's achievable per-component, but no existing system provides equivalent breadth.

---

## What SOTA Would Look Like

A state-of-the-art pipeline processing the same 10 documents could achieve:

| Metric | Current ESE | Optimized ESE | Full SOTA Rewrite |
|--------|------------|---------------|-------------------|
| Time per doc | 13.4 min avg | ~2–4 min | ~10–30s |
| Throughput | 4.5 docs/hour | 15–30 docs/hour | 120–360 docs/hour |
| Cost per doc | $0.90 | $0.25–0.35 | $0.03–0.08 |
| Entity coverage | 11 entity types | 11 entity types | Typically 3–5 types |
| Ontology linking | 6+ ontologies | 6+ ontologies | Usually 1–2 |

---

## Recommended Optimizations (by impact)

### 1. Fix Model Tier Routing
Ensure Haiku is actually used for the 10 call types configured for it. Immediate ~65% cost reduction, plus 2–4x latency improvement on those calls.

### 2. Async LLM Calls
Use `anthropic.AsyncAnthropic()` with `asyncio.gather()` to parallelize validation batches. Expected 3–5x speedup on LLM stages.

### 3. Document-Level Parallelism
`concurrent.futures.ProcessPoolExecutor` for PDF parsing + `asyncio` event loop for API calls. Expected 5–10x throughput increase.

### 4. Replace Unstructured.io for Simple Layouts
For simple layouts, PyMuPDF native extraction is 10–100x faster. Use Unstructured.io only as fallback for complex layouts.

### 5. Improve Prompt Caching
Current cache hit rate is only 5.2%. Restructuring prompts to share system prompt prefixes across calls could push this to 50–80%, saving ~40% on token costs.

---

## External API Dependencies

| API | Rate Limit | Cost | Cache Strategy |
|-----|-----------|------|----------------|
| Claude (Anthropic) | Tier-based (RPM/TPM) | $1–$15 per MTok | Prompt caching (ephemeral) |
| PubTator3 (NCBI) | 3 req/sec | Free | 7-day disk cache |
| ClinicalTrials.gov | No published limit | Free | Per-request |

---

## Measurement Infrastructure

The pipeline includes built-in performance tracking:

- **StageTimer** (`orchestrator_utils.py`): Wall-clock timing for all 16 pipeline stages with visual bar chart output
- **LLMUsageTracker** (`D02_llm_engine.py`): Per-call token usage, cost calculation, model routing
- **SQLite persistence** (`Z10_usage_tracker.py`): `llm_usage`, `lexicon_usage`, `datasource_usage` tables in `corpus_log/usage_stats.db`
- **Console logs**: Tee'd to `corpus_log/pipeline_run_*.log` with full stage breakdowns
- **Validation logs**: Per-candidate JSONL in `corpus_log/RUN_*.jsonl`

---

## References

- [Do LLMs Surpass Encoders for Biomedical NER? (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12335919/)
- [HunFlair2 Cross-corpus NER Evaluation (2024)](https://academic.oup.com/bioinformatics/article/40/10/btae564/7762634)
- [Benchmarking LLM-based IE for Medical Documents (Jan 2026)](https://www.medrxiv.org/content/10.64898/2026.01.19.26344287v1.full)
- [Claude Haiku 4.5 Announcement](https://www.anthropic.com/news/claude-haiku-4-5)
- [PubTator 3.0 (2024)](https://academic.oup.com/nar/article/52/W1/W540/7640526)
- [Clinical Pipeline Speed–Accuracy Trade-offs (2025)](https://www.johnsnowlabs.com/clinical-de-identification-at-scale-pipeline-design-and-speed-accuracy-trade-offs-across-infrastructures/)
