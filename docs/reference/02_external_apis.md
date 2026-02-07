# External API Reference

Remote APIs and local NER models used for validation, enrichment, and vision analysis. Remote clients share `Z01_api_client.py` with disk caching, rate limiting, and retry.

## Claude API (Anthropic)

| Property | Value |
|----------|-------|
| Purpose | LLM validation, classification, vision analysis, feasibility, recommendations |
| Auth | `ANTHROPIC_API_KEY` environment variable |

**Model tier routing** via `call_type` in `config.yaml`:

| Tier | Model | Cost (I/O per MTok) | Usage |
|------|-------|---------------------|-------|
| Haiku | `claude-haiku-4-5-20251001` | $1 / $5 | Classification, layout, abbreviation validation |
| Sonnet | `claude-sonnet-4-20250514` | $3 / $15 | Feasibility, flowcharts, tables, recommendations |

**17 call_types across 12 files:**

| Module | call_type | Tier |
|--------|-----------|------|
| `D02_llm_engine.py` | `abbreviation_batch_validation`, `abbreviation_single_validation`, `fast_reject` | Haiku |
| `C09_strategy_document_metadata.py` | `document_classification`, `description_extraction` | Haiku |
| `C10_vision_image_analysis.py` | `image_classification`, `ocr_text_fallback` | Haiku |
| `C10_vision_image_analysis.py` | `flowchart_analysis`, `chart_analysis`, `vlm_table_extraction` | Sonnet |
| `C11_llm_feasibility.py` | `feasibility_extraction` | Sonnet |
| `C17_flowchart_graph_extractor.py` | `flowchart_analysis` | Sonnet |
| `C19_vlm_visual_enrichment.py` | `vlm_visual_enrichment` | Haiku |
| `C32_recommendation_llm.py` | `recommendation_extraction` | Sonnet |
| `C33_recommendation_vlm.py` | `recommendation_vlm` | Sonnet |
| `B19_layout_analyzer.py` | `layout_analysis` | Haiku |
| `B22_doclayout_detector.py` | `vlm_visual_enrichment` | Haiku |
| `B31_vlm_detector.py` | `vlm_detection` | Sonnet |
| `H02_abbreviation_pipeline.py` | `sf_only_extraction` | Haiku |

Prompt caching on system prompts: cache reads 10%, creation 125% of input price.

Usage tracking: `LLMUsageTracker` (in-memory) + `llm_usage` SQLite table.

Error handling: `ClaudeClient` catches rate limit, connection, status, and timeout errors. Batch failures fall back to individual verification.

---

## PubTator3 (NCBI)

| Property | Value |
|----------|-------|
| Purpose | Disease, drug, gene enrichment (MeSH codes, aliases) |
| Base URL | `https://www.ncbi.nlm.nih.gov/research/pubtator3-api` |
| Rate Limit | 3 req/sec |
| Cache TTL | 7 days |

Endpoints: `/entity/autocomplete/` (primary), `/search/` (fallback).

Used by `E04` (diseases), `E05` (drugs), `E18` (genes) via `PubTator3Client`.

---

## ClinicalTrials.gov

| Property | Value |
|----------|-------|
| Purpose | NCT enrichment (trial title, conditions, interventions, phase, status) |
| Base URL | `https://clinicaltrials.gov/api/v2/studies` |
| Rate Limit | 1 req/sec |
| Cache TTL | 30 days |

Used by `E06_nct_enricher.py`: `NCTEnricher` and `TrialAcronymEnricher`.

---

## Local NER Models

Lazy-loaded, no API calls.

| Model | Purpose | Module | Toggle |
|-------|---------|--------|--------|
| **scispacy** (`en_core_sci_sm`) | Biomedical NER with UMLS linking | C04, C06, C07, C16 | Always on |
| **EpiExtract4GARD-v2** | Rare disease epidemiology | E08 | `use_epi_enricher` |
| **ZeroShotBioNER** | ADE, dosage, frequency, route | E09 | `use_zeroshot_bioner` |
| **d4data/biomedical-ner-all** | 84 biomedical entity types | E10 | `use_biomedical_ner` |

---

## Shared API Client Infrastructure

`BaseAPIClient` provides:
- **Disk caching** -- Thread-safe JSON file cache with TTL
- **Rate limiting** -- Token-bucket with minimum request interval
- **Connection pooling** -- `requests.Session` with `ESE-Pipeline/1.0` User-Agent
- **Error handling** -- `APIError`, `RateLimitError`; HTTP 429 waits 60s and retries

| Retry Policy | Retries | Base Delay | Max Delay |
|--------------|---------|-----------|-----------|
| `DEFAULT_RETRY` | 3 | 1.0s | 30s |
| `API_RETRY` | 3 | 1.0s | 60s |
| `AGGRESSIVE_RETRY` | 5 | 0.5s | 30s |

Cache structure: `cache/{service}/{key}.json` with `_cached_at` timestamp.
