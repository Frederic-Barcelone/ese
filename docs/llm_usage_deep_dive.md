# LLM Usage Strategy -- Model Routing & Cost Optimization

> **Pipeline version**: v0.8

---

## 1. Overview

Every LLM call is routed to the optimal model tier, tracked per-call, and persisted to SQLite.

| Metric | Value |
|--------|-------|
| LLM call sites | 17 (10 Haiku, 7 Sonnet) |
| Haiku pricing | $1.0 / $5.0 per MTok (in/out) |
| Sonnet pricing | $3.0 / $15.0 per MTok (in/out) |

---

## 2. Model Tier Architecture

**Haiku** ($1/$5) -- classification, quick filtering, OCR, batch processing.

**Sonnet** ($3/$15) -- multi-field extraction, visual understanding, nuanced analysis, structured output.

### Routing

`resolve_model_tier()` in `D02_llm_engine.py` loads `model_tiers` from `config.yaml`, looks up the `call_type`, and returns the configured model (or the default).

```yaml
api:
  claude:
    validation:
      model: "claude-sonnet-4-20250514"  # Default fallback
    model_tiers:
      abbreviation_batch_validation: "claude-haiku-4-5-20251001"
      feasibility_extraction: "claude-sonnet-4-20250514"
      # ... 17 call_types total
```

`ClaudeClient.resolve_model(call_type)` checks `_model_tiers` dict, falls back to `self.default_model`. Every `complete_json()` and `complete_vision_json()` call accepts a `call_type` parameter.

---

## 3. Call Site Inventory

### Haiku Tier (10 Sites)

| call_type | Script | Purpose |
|-----------|--------|---------|
| `abbreviation_batch_validation` | D02 | Batch-verify 15-20 abbreviation candidates |
| `abbreviation_single_validation` | D02 | Single abbreviation fallback |
| `fast_reject` | D02 | Pre-screen obvious FPs |
| `document_classification` | C09 | Classify document type |
| `description_extraction` | C09 | 2-4 sentence document summary |
| `image_classification` | C10 | Classify visual type |
| `sf_only_extraction` | H02 | PASO D: find definitions for short-form-only abbreviations |
| `layout_analysis` | B19 | Single vs two-column detection |
| `vlm_visual_enrichment` | C19, B22 | Visual titles and descriptions |
| `ocr_text_fallback` | C10 | Text extraction from images |

### Sonnet Tier (7 Sites)

| call_type | Script | Purpose |
|-----------|--------|---------|
| `feasibility_extraction` | C11 | Eligibility, study design, endpoints |
| `recommendation_extraction` | C32 | Clinical recommendations |
| `recommendation_vlm` | C33 | LoE/SoR from table images |
| `flowchart_analysis` | C10, C17 | Node/edge graphs from flowcharts |
| `chart_analysis` | C10 | Data extraction from charts |
| `vlm_table_extraction` | C10 | Complex tables Docling cannot parse |
| `vlm_detection` | B31 | Table/figure bounding box detection |

---

## 4. VLM (Vision Language Model)

### Image Processing Pipeline

1. PDF pages rendered at 150 DPI (PyMuPDF)
2. Scaled to max 1568px on longest dimension
3. Base64-encoded; auto-compressed if >5MB
4. Media type auto-detected from header bytes

### Visual Type Routing

| Visual Type | Model | Output |
|-------------|-------|--------|
| Simple figure | Haiku | Title + description |
| Complex chart | Sonnet | Data series, axes, values |
| Data table | Sonnet | Headers, rows, cells |
| Flowchart | Sonnet | Nodes, edges, conditions, drugs |
| Recommendation table | Sonnet | LoE/SoR codes |

Sequential per-page processing; results cached (e.g., `_vlm_loe_sor_cache`).

---

## 5. Validation Engine (D02)

**Batch verification**: 15-20 candidates/call, dynamic `max_tokens`, in-memory cache by `(sf_upper, lf_lower)`, 100ms inter-batch delay, single-verification fallback.

**Fast-reject**: Haiku batch of 20, rejects only if `decision == "REJECT"` AND `confidence >= 0.85`. Survivors proceed to full batch validation.

Prompts versioned in registry: `VERIFY_BATCH`, `VERIFY_SINGLE`, `FAST_REJECT`.

---

## 6. Prompt Caching

System messages marked with `"cache_control": {"type": "ephemeral"}`: first call 125% cost (creation), subsequent calls 10% input cost, 5-minute TTL. Stable instructions in system message, variable content in user message.

---

## 7. Cost Tracking

### Recording

- **ClaudeClient calls**: Automatic via `_record_usage()` after every response
- **Raw API calls**: Manual `record_api_usage(response, model, call_type)`

### Cost Calculation

```
total = (input_tokens * price/1M) + (cache_read * price * 0.10/1M)
      + (cache_create * price * 1.25/1M) + (output_tokens * out_price/1M)
```

### MODEL_PRICING

| Model ID | In $/MTok | Out $/MTok |
|----------|-----------|------------|
| `claude-sonnet-4-20250514` | $3.00 | $15.00 |
| `claude-sonnet-4-5-20250929` | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 |
| `claude-haiku-4-5-20251001` | $1.00 | $5.00 |

### LLMUsageTracker

Singleton via `get_usage_tracker()`. Methods: `record()`, `total_input_tokens`, `total_output_tokens`, `estimated_cost()`, `summary_by_model()`, `summary_by_call_type()`, `reset()`.

---

## 8. SQLite Persistence (Z06)

`UsageTracker` persists to `usage_stats.db` with tables: `documents`, `llm_usage`, `lexicon_usage`, `datasource_usage`. Indexed on document_id, model, lexicon_name, datasource_name.

Key methods: `log_llm_usage()`, `log_llm_usage_batch()`, `get_llm_stats()`, `get_llm_stats_by_call_type()`, `print_summary()`. Orchestrator provides per-document and batch cost summaries.

---

## 9. Cost Optimization Strategies

**1. Model tier routing (biggest lever):** Haiku for 10/17 call types gives 3-5x savings. ~67% reduction on abbreviation batches, classifications, visual enrichment.

**2. Fast-reject pre-screening:** Haiku eliminates 30-50% of candidates before full validation. ~$0.01 per batch of 20.

**3. Batch processing:** 15-20 items per call. System prompt amortized, cache creation paid once.

**4. Prompt caching:** 10% input cost on cache hits. 20-40% savings on input tokens.

**5. Extraction presets:** Skip unneeded entity types (e.g., `preset: "diseases_only"` instead of `"all"`).

---

## 10. Future Work

- **Local models**: Ollama/vLLM for classification; fine-tuned abbreviation validation
- **Async LLM calls**: Parallel API calls, streaming
- **Dynamic routing**: Per-doc complexity scoring, cost budgets, auto tier downgrade
