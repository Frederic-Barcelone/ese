# LLM Cost Optimization Guide

This guide covers the ESE pipeline's LLM cost optimization system: model tier routing, prompt caching, fast-reject pre-screening, usage tracking, and cost reporting.

## Overview

The pipeline makes 17 unique LLM call_types across 12 files per document. Without optimization, all calls use Sonnet 4 ($3/$15 per MTok). The cost optimization system routes simple tasks to Haiku ($1/$5 per MTok), applies prompt caching (90% savings on repeated system prompts), and pre-screens candidates with cheap fast-reject calls.

## Model Tier Routing

Every LLM call site passes a `call_type` string that identifies the task. The model is resolved at runtime from `config.yaml`:

```yaml
api:
  claude:
    model_tiers:
      abbreviation_batch_validation: "claude-haiku-4-5-20250901"
      feasibility_extraction: "claude-sonnet-4-20250514"
      # ...
```

### call_type Reference

| call_type | Tier | Model | Used By | Purpose |
|-----------|------|-------|---------|---------|
| `abbreviation_batch_validation` | Haiku | `claude-haiku-4-5-20250901` | D02 LLMEngine | Batch abbreviation validation |
| `abbreviation_single_validation` | Haiku | `claude-haiku-4-5-20250901` | D02 LLMEngine | Single abbreviation validation |
| `fast_reject` | Haiku | `claude-haiku-4-5-20250901` | D02 LLMEngine | Fast-reject pre-screening |
| `document_classification` | Haiku | `claude-haiku-4-5-20250901` | C09 document metadata | Document type classification |
| `description_extraction` | Haiku | `claude-haiku-4-5-20250901` | C09 document metadata | Title/description generation |
| `image_classification` | Haiku | `claude-haiku-4-5-20250901` | C10 vision analysis | Image type classification |
| `sf_only_extraction` | Haiku | `claude-haiku-4-5-20250901` | H02 abbreviation pipeline | PASO D SF-only extraction |
| `layout_analysis` | Haiku | `claude-haiku-4-5-20250901` | B19 layout analyzer | Page layout spatial analysis |
| `vlm_visual_enrichment` | Haiku | `claude-haiku-4-5-20250901` | C19, B22 | Visual title/description |
| `ocr_text_fallback` | Haiku | `claude-haiku-4-5-20250901` | C10 vision analysis | OCR text extraction fallback |
| `feasibility_extraction` | Sonnet | `claude-sonnet-4-20250514` | C11 feasibility | Clinical trial feasibility |
| `recommendation_extraction` | Sonnet | `claude-sonnet-4-20250514` | C32 recommendation LLM | Guideline recommendations |
| `recommendation_vlm` | Sonnet | `claude-sonnet-4-20250514` | C33 recommendation VLM | Visual recommendations |
| `vlm_table_extraction` | Sonnet | `claude-sonnet-4-20250514` | C10 vision analysis | Table structure extraction |
| `flowchart_analysis` | Sonnet | `claude-sonnet-4-20250514` | C10, C17 | Flowchart/CONSORT analysis |
| `chart_analysis` | Sonnet | `claude-sonnet-4-20250514` | C10 vision analysis | Chart data extraction |
| `vlm_detection` | Sonnet | `claude-sonnet-4-20250514` | B31 VLM detector | Visual element detection |

### Tier Selection Criteria

**Haiku** (simple, classification-like tasks):
- Binary or categorical decisions (is this an abbreviation? what type is this image?)
- Tasks with well-defined schemas and limited reasoning depth
- Layout and spatial analysis
- Tasks where Haiku quality is equivalent to Sonnet

**Sonnet** (complex reasoning tasks):
- Multi-step extraction requiring document understanding (feasibility, recommendations)
- Structured data extraction from complex visuals (flowcharts, charts, tables)
- Tasks where reasoning quality directly affects extraction accuracy

## Architecture

### Key Files

| File | Role |
|------|------|
| `D_validation/D02_llm_engine.py` | Core infrastructure: `resolve_model_tier()`, `record_api_usage()`, `calc_record_cost()`, `LLMUsageTracker`, `MODEL_PRICING` |
| `G_config/config.yaml` | `model_tiers` section maps `call_type` to model ID |
| `Z_utils/Z06_usage_tracker.py` | `llm_usage` SQLite table for persistent tracking across runs |
| `orchestrator.py` | Per-document and batch cost summaries using `calc_record_cost()` |

### Two Categories of API Calls

**Category A: ClaudeClient-based calls** (D02, C09, C10, C11)

These go through `ClaudeClient._call_claude()` or `complete_vision_json()`, which automatically:
- Resolves the model via `self._model_tiers.get(call_type, ...)`
- Records usage in the global `LLMUsageTracker`
- Applies prompt caching on system prompts

Usage: pass `call_type=` parameter to `complete_json()`, `complete_json_any()`, or `complete_vision_json()`.

**Category B: Raw `anthropic.Anthropic()` calls** (B19, B22, B31, C17, C19, C32, C33)

These bypass `ClaudeClient` and call the Anthropic SDK directly. They must:
1. Call `resolve_model_tier(call_type)` to get the correct model
2. Call `record_api_usage(response, model, call_type)` after each API call

```python
from D_validation.D02_llm_engine import resolve_model_tier, record_api_usage

model = resolve_model_tier("layout_analysis")
response = client.messages.create(model=model, ...)
record_api_usage(response, model, "layout_analysis")
```

## Prompt Caching

`ClaudeClient._call_claude()` applies `cache_control: {"type": "ephemeral"}` on system prompts:

```python
messages_payload = {
    "system": [{"type": "text", "text": system_prompt,
                "cache_control": {"type": "ephemeral"}}],
    ...
}
```

**Cost impact:**
- Cache reads: 10% of input token price
- Cache creation: 125% of input token price (first call only)
- Net savings: ~90% on repeated system prompts across batch validation calls

## Fast-Reject Pre-Screening

`H02_abbreviation_pipeline.py` pre-screens abbreviation candidates with a cheap Haiku call before full validation. Candidates that are obviously invalid are rejected without consuming a Sonnet API call.

## Usage Tracking

### In-Memory Tracking (LLMUsageTracker)

Every API call records an `LLMUsageRecord` in the global `LLMUsageTracker`:

```python
tracker = get_usage_tracker()
tracker.record(model, input_tokens, output_tokens, call_type, cache_read_tokens)
```

The tracker provides:
- `total_input_tokens`, `total_output_tokens`, `total_cache_read_tokens`, `total_calls`
- `estimated_cost()` -- Total estimated cost in USD
- `summary_by_model()` -- Usage grouped by model
- `summary_by_call_type()` -- Usage grouped by call_type

### Persistent Tracking (SQLite)

The orchestrator flushes `LLMUsageTracker.records` to the `llm_usage` SQLite table via `UsageTracker.log_llm_usage_batch()` after each document:

```sql
CREATE TABLE llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    model TEXT NOT NULL,
    call_type TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0.0,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Query with:
- `get_llm_stats()` -- Aggregated by model
- `get_llm_stats_by_call_type()` -- Aggregated by call_type and model
- `print_summary()` -- Full console report

### Cost Reporting

The orchestrator prints per-document cost summaries:

```
LLM Usage: 45 calls, 125,432 input tokens, 8,201 output tokens
  Estimated cost: $0.42
```

And batch summaries across all documents in `process_folder()`.

## Cost Calculation

`calc_record_cost()` computes estimated cost accounting for prompt caching:

```python
def calc_record_cost(model, input_tokens, output_tokens,
                     cache_read_tokens=0, cache_creation_tokens=0) -> float:
    input_price, output_price = MODEL_PRICING.get(model, (3.0, 15.0))
    base_input = input_tokens - cache_read_tokens - cache_creation_tokens
    cache_read_cost = cache_read_tokens * (input_price * 0.1) / 1_000_000
    cache_create_cost = cache_creation_tokens * (input_price * 1.25) / 1_000_000
    input_cost = max(0, base_input) * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    return input_cost + cache_read_cost + cache_create_cost + output_cost
```

### MODEL_PRICING

| Model | Input ($/MTok) | Output ($/MTok) |
|-------|---------------|----------------|
| `claude-sonnet-4-20250514` | $3.00 | $15.00 |
| `claude-sonnet-4-5-20250929` | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 |
| `claude-haiku-4-5-20250901` | $1.00 | $5.00 |

## Adding a New LLM Call Site

1. **Choose a descriptive `call_type` string** (e.g., `"entity_validation"`)
2. **Add it to `config.yaml`** under `api.claude.model_tiers` with the appropriate model
3. **For ClaudeClient calls**: pass `call_type=` to `complete_json()`, `complete_json_any()`, or `complete_vision_json()`
4. **For raw API calls**:
   ```python
   from D_validation.D02_llm_engine import resolve_model_tier, record_api_usage
   model = resolve_model_tier("entity_validation")
   response = client.messages.create(model=model, ...)
   record_api_usage(response, model, "entity_validation")
   ```
5. **Add the model to `MODEL_PRICING`** in `D02_llm_engine.py` if it's a new model not already listed

## Changing a Task's Model

To move a task between tiers, update its entry in `config.yaml`:

```yaml
# Move layout_analysis from Haiku to Sonnet (if quality needs improvement)
model_tiers:
  layout_analysis: "claude-sonnet-4-20250514"
```

No code changes required -- `resolve_model_tier()` reads from config at runtime.

## Related Documentation

- [Configuration Guide](03_configuration.md) -- `model_tiers` config section
- [D_validation Layer](../layers/D_validation.md) -- Cost optimization components
- [External APIs Reference](../reference/02_external_apis.md) -- Full call_type table by module
- [Z_utils Layer](../layers/Z_utils.md) -- SQLite usage tracking
