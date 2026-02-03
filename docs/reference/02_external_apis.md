# External API Reference

ESE integrates with several external APIs and local NER models for entity validation, enrichment, and vision analysis. All remote API clients share a common base class (`Z_utils/Z01_api_client.py`) providing disk caching, rate limiting, retry logic, and connection pooling.

## Remote APIs

### Claude API (Anthropic)

| Property | Value |
|----------|-------|
| Purpose | LLM validation, document classification, vision analysis, feasibility extraction, recommendation extraction |
| SDK | `anthropic` Python package |
| Auth | `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` environment variable |
| Caching | None (live API calls) |

**Models used:**

| Tier | Model | Usage |
|------|-------|-------|
| Validation | `claude-sonnet-4-20250514` | Abbreviation, drug, disease validation (D_validation) |
| Fast | `claude-sonnet-4-5-20250929` | Basic tasks, cheaper/faster |
| Vision | `claude-sonnet-4-20250514` | Figure/flowchart/table analysis |

Models are configurable in `G_config/config.yaml` under `api.claude.validation.model` and `api.claude.fast.model`.

**Usage across the pipeline:**

| Module | Purpose |
|--------|---------|
| `D_validation/D02_llm_engine.py` | `ClaudeClient` -- abbreviation candidate validation (single + batch), fast-reject via the fast-tier model (`claude-sonnet-4-5-20250929`) |
| `C_generators/C10_vision_image_analysis.py` | Vision LLM analysis of figures, charts, flowcharts |
| `C_generators/C11_llm_feasibility.py` | LLM-based feasibility data extraction from protocols |
| `C_generators/C32_recommendation_llm.py` | Guideline recommendation extraction |

**Rate limiting:**

Configured per tier in `config.yaml` via `api.claude.batch_delay_ms` (default: 100ms between calls). The `LLMEngine` in `D02_llm_engine.py` handles batch validation with configurable batch sizes and inter-batch delays.

**Error handling:**

The `ClaudeClient` catches `AnthropicRateLimitError`, `AnthropicConnectionError`, `AnthropicStatusError`, and `AnthropicTimeoutError`. On batch failures, it falls back to individual candidate verification.

---

### PubTator3 (NCBI)

| Property | Value |
|----------|-------|
| Purpose | Disease, drug, and gene enrichment with MeSH codes and aliases |
| Base URL | `https://www.ncbi.nlm.nih.gov/research/pubtator3-api` |
| Rate Limit | 3 requests/second (per NCBI guidelines) |
| Cache TTL | 7 days (168 hours) |
| Cache Dir | `cache/pubtator/` |

**Endpoints used:**

| Endpoint | Purpose |
|----------|---------|
| `/entity/autocomplete/` | Entity normalization (primary) -- returns MeSH IDs, normalized names, biotype |
| `/search/` | Comprehensive entity search (fallback) |

**Usage across the pipeline:**

| Module | Entity Type | Enrichment |
|--------|-------------|------------|
| `E_normalization/E04_pubtator_enricher.py` | Diseases | MeSH IDs, normalized names, aliases |
| `E_normalization/E05_drug_enricher.py` | Drugs/Chemicals | MeSH IDs, normalized names, aliases |
| `E_normalization/E18_gene_enricher.py` | Genes | Entrez/NCBI Gene IDs, normalized names, aliases |

All three enrichers reuse the `PubTator3Client` class from `E04_pubtator_enricher.py`, which extends `BaseAPIClient`.

**Response format:**

```json
[
  {
    "name": "IgA nephropathy",
    "db_id": "D005922",
    "db": "ncbi_mesh",
    "biotype": "disease",
    "_id": "..."
  }
]
```

**Configuration** (in `config.yaml`):

```yaml
api:
  pubtator:
    enabled: true
    base_url: "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    timeout_seconds: 30
    rate_limit_per_second: 3
    cache:
      enabled: true
      directory: "cache/pubtator"
      ttl_hours: 168
    enrichment:
      enrich_missing_mesh: true
      add_aliases: true
```

---

### ClinicalTrials.gov API

| Property | Value |
|----------|-------|
| Purpose | NCT identifier enrichment (trial titles, conditions, interventions, phase, status) |
| Base URL | `https://clinicaltrials.gov/api/v2/studies` |
| Rate Limit | 1 request/second (conservative) |
| Cache TTL | 30 days (720 hours) |
| Cache Dir | `cache/clinicaltrials/` |

**Usage across the pipeline:**

| Module | Class | Purpose |
|--------|-------|---------|
| `E_normalization/E06_nct_enricher.py` | `NCTEnricher` | Enrich NCT IDs with official trial metadata |
| `E_normalization/E06_nct_enricher.py` | `TrialAcronymEnricher` | Search by trial acronym (e.g., "APPEAR-C3G") to find official titles |

**Returned data** (`NCTTrialInfo`):

| Field | Description |
|-------|-------------|
| `nct_id` | Normalized NCT identifier |
| `official_title` | Full official trial title |
| `brief_title` | Short trial title |
| `acronym` | Trial acronym |
| `conditions` | List of conditions studied |
| `interventions` | List of drug/intervention names |
| `status` | Overall study status |
| `phase` | Trial phase(s) |

**Configuration** (in `config.yaml`):

```yaml
nct_enricher:
  enabled: true
  cache:
    enabled: true
    directory: "cache/clinicaltrials"
    ttl_days: 30
```

**Convenience functions:**

```python
from E_normalization.E06_nct_enricher import enrich_nct_id, enrich_trial_acronym

info = enrich_nct_id("NCT04817618")
title = enrich_trial_acronym("APPEAR-C3G")
```

---

## Local NER Models

These models run locally without API calls. They are lazy-loaded to minimize startup time and memory usage.

### scispacy

| Property | Value |
|----------|-------|
| Purpose | Biomedical named entity recognition with UMLS concept linking |
| Models | `en_core_sci_sm`, `en_ner_bc5cdr_md` |
| Type | Local (no API calls) |
| Loading | Lazy (loaded on first use) |

**Usage:**

| Module | Entity Types |
|--------|-------------|
| `C_generators/C04_strategy_flashtext.py` | General biomedical NER |
| `C_generators/C06_strategy_disease.py` | Disease NER (DISEASE type) |
| `C_generators/C07_strategy_drug.py` | Chemical/drug NER (CHEMICAL type) |
| `C_generators/C16_strategy_gene.py` | Gene NER (GENE type) |

The `en_ner_bc5cdr_md` model specializes in disease and chemical entity recognition. UMLS concept linking is available but limited by the `umls_max_blocks` config setting (default: 500) to avoid O(n^2) scaling on large documents.

---

### EpiExtract4GARD-v2

| Property | Value |
|----------|-------|
| Purpose | Rare disease epidemiology NER |
| Entity Types | LOC (locations), EPI (epidemiology terms), STAT (statistics) |
| Module | `E_normalization/E08_epi_extract_enricher.py` |
| Loading | Lazy |
| Toggle | `extraction_pipeline.options.use_epi_enricher` |

---

### ZeroShotBioNER

| Property | Value |
|----------|-------|
| Purpose | Flexible zero-shot entity extraction |
| Entity Types | ADE (adverse events), dosage, frequency, route |
| Module | `E_normalization/E09_zeroshot_bioner.py` |
| Loading | Lazy |
| Toggle | `extraction_pipeline.options.use_zeroshot_bioner` |

---

### d4data/biomedical-ner-all

| Property | Value |
|----------|-------|
| Purpose | Broad biomedical NER (84 entity types) |
| Entity Types | Symptoms, procedures, lab values, anatomy, and more |
| Module | `E_normalization/E10_biomedical_ner_all.py` |
| Loading | Lazy |
| Toggle | `extraction_pipeline.options.use_biomedical_ner` |

---

## Shared API Client Infrastructure

All remote API clients extend `BaseAPIClient` from `Z_utils/Z01_api_client.py`.

### BaseAPIClient

Provides:

- **Disk caching** (`DiskCache`) -- Thread-safe JSON file cache with configurable TTL. Cache keys are MD5-hashed. Cache files stored as `{key}.json` with `_cached_at` timestamps.
- **Rate limiting** (`RateLimiter`) -- Thread-safe, enforces minimum interval between requests.
- **Connection pooling** -- `requests.Session` with persistent connections and custom User-Agent (`ESE-Pipeline/1.0`).
- **Error handling** -- Catches timeouts, connection errors, JSON decode errors. Raises `APIError`, `RateLimitError` with status code context.
- **HTTP 429 handling** -- Waits 60 seconds and retries once on rate limit responses.

### Retry Logic

The `retry_with_backoff` decorator provides exponential backoff with jitter:

- Default: 3 retries, 1s base delay, 30s max delay
- Retryable HTTP codes: 429, 500, 502, 503, 504
- Retryable exceptions: `Timeout`, `ConnectionError`

Pre-configured retry policies:

| Policy | Retries | Base Delay | Max Delay |
|--------|---------|-----------|-----------|
| `DEFAULT_RETRY` | 3 | 1.0s | 30s |
| `API_RETRY` | 3 | 1.0s | 60s |
| `AGGRESSIVE_RETRY` | 5 | 0.5s | 30s |

### Cache Structure

```
cache/
  pubtator/
    autocomplete_disease_a1b2c3d4e5f6.json
    search_gene_f6e5d4c3b2a1.json
  clinicaltrials/
    nct_NCT04817618.json
    acronym_APPEAR-C3G.json
```

Each cache file:

```json
{
  "_cached_at": 1706900000.123,
  "result": { ... }
}
```
