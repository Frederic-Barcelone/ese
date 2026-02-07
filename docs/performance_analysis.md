# ESE Pipeline v0.8 -- Performance Analysis (Supplemental)

> **Date**: February 4, 2026
> **Based on**: Actual pipeline logs from 10-document PAPERS batch run

This file contains data points supplemental to [`performance_benchmarks.md`](performance_benchmarks.md), which is the authoritative performance reference. Only unique content is kept here.

---

## Extended Per-Document Breakdown

Full 10-document breakdown (4 additional docs not in benchmarks file):

| Document | Time | Calls | In Tok | Out Tok | Cost |
|----------|------|-------|--------|---------|------|
| 41927_2025 | 571s | 61 | 121K | 18K | $0.63 |
| Newsletter 20251213-56068 | 226s | 25 | 42K | 7K | $0.23 |
| BMJ m1070 | 243s | 22 | 55K | 7K | $0.26 |
| Document 10_ | 426s | 37 | 80K | 14K | $0.45 |

Overall average API calls per doc: 23.1 (across 69 docs) vs 71.3 (this batch of 10).

---

## Extended SOTA Comparison

Additional dimensions not covered in benchmarks:

| Dimension | ESE v0.8 | SOTA / Achievable | Gap |
|-----------|----------|-------------------|-----|
| PubTator3 normalization | 3 req/sec, 7-day cache | Bulk download (36M abstracts); local UMLS/MONDO | Cache mitigates |
| ClinicalTrials.gov | Free, no published limit | Already well-utilized | At parity |

---

## Recommended Optimizations (Ranked)

1. **~~Fix model tier routing~~** -- DONE (2026-02-06). Haiku now correctly routes 10 call types.
2. **Async LLM calls** -- `anthropic.AsyncAnthropic()` with `asyncio.gather()`. Expected 3-5x LLM stage speedup.
3. **Document-level parallelism** -- `ProcessPoolExecutor` for PDF parsing + async event loop for API. 5-10x throughput.
4. **Replace Unstructured.io for simple layouts** -- PyMuPDF native is 10-100x faster. Use Unstructured.io as fallback only.
5. **Improve prompt caching** -- Current hit rate 5.2%. Restructure shared system prompts to reach 50-80%, saving ~40% on tokens.

---

## External API Dependencies

| API | Rate Limit | Cost | Cache |
|-----|-----------|------|-------|
| Claude (Anthropic) | Tier-based RPM/TPM | $1-$15/MTok | Prompt caching (ephemeral) |
| PubTator3 (NCBI) | 3 req/sec | Free | 7-day disk cache |
| ClinicalTrials.gov | No published limit | Free | Per-request |
