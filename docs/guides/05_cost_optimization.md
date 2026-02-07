# LLM Cost Optimization

## Overview

17 LLM call_types across 12 files. Cost system routes simple tasks to Haiku ($1/$5 per MTok), complex reasoning to Sonnet ($3/$15 per MTok), applies prompt caching (90% savings on repeated system prompts), and pre-screens candidates with fast-reject calls.

## Model Tier Routing

Every LLM call site passes a `call_type` string. Model resolved at runtime from `config.yaml`:

```yaml
api:
  claude:
    model_tiers:
      abbreviation_batch_validation: "claude-haiku-4-5-20251001"
      feasibility_extraction: "claude-sonnet-4-20250514"
      # ...
```

### call_type Reference

| call_type | Tier | Used By |
|-----------|------|---------|
| `abbreviation_batch_validation` | Haiku | D02 LLMEngine |
| `abbreviation_single_validation` | Haiku | D02 LLMEngine |
| `fast_reject` | Haiku | D02 LLMEngine |
| `document_classification` | Haiku | C09 document metadata |
| `description_extraction` | Haiku | C09 document metadata |
| `image_classification` | Haiku | C10 vision analysis |
| `sf_only_extraction` | Haiku | H02 abbreviation pipeline |
| `layout_analysis` | Haiku | B19 layout analyzer |
| `vlm_visual_enrichment` | Haiku | C19, B22 |
| `ocr_text_fallback` | Haiku | C10 vision analysis |
| `feasibility_extraction` | Sonnet | C11 feasibility |
| `recommendation_extraction` | Sonnet | C32 recommendation LLM |
| `recommendation_vlm` | Sonnet | C33 recommendation VLM |
| `vlm_table_extraction` | Sonnet | C10 vision analysis |
| `flowchart_analysis` | Sonnet | C10, C17 |
| `chart_analysis` | Sonnet | C10 vision analysis |
| `vlm_detection` | Sonnet | B31 VLM detector |

**Haiku** -- Binary/categorical decisions, layout analysis, abbreviation validation.

**Sonnet** -- Multi-step extraction, structured data from complex visuals, reasoning-dependent accuracy.

## Key Files

| File | Role |
|------|------|
| `D02_llm_engine.py` | `resolve_model_tier()`, `record_api_usage()`, `calc_record_cost()`, `LLMUsageTracker`, `MODEL_PRICING` |
| `config.yaml` | `model_tiers` maps `call_type` to model ID |
| `Z06_usage_tracker.py` | `llm_usage` SQLite table for persistent tracking |
| `orchestrator.py` | Per-document and batch cost summaries |

## Two API Call Patterns

**ClaudeClient-based** (D02, C09, C10, C11) -- Automatically resolves models, records usage, applies prompt caching. Pass `call_type=` to `complete_json()` / `complete_json_any()` / `complete_vision_json()`.

**Raw `anthropic.Anthropic()`** (B19, B22, B31, C17, C19, C32, C33) -- Must manually resolve and record:

```python
from D_validation.D02_llm_engine import resolve_model_tier, record_api_usage
model = resolve_model_tier("layout_analysis")
response = client.messages.create(model=model, ...)
record_api_usage(response, model, "layout_analysis")
```

## Prompt Caching

`ClaudeClient._call_claude()` applies `cache_control: {"type": "ephemeral"}` on system prompts. Cache reads cost 10% of input price; creation costs 125% (first call only). Net savings ~90% on repeated prompts.

## Fast-Reject Pre-Screening

`H02_abbreviation_pipeline.py` pre-screens abbreviation candidates with a Haiku call before full validation, rejecting obviously invalid candidates cheaply.

## Usage Tracking

**In-memory:** `LLMUsageTracker` -- `total_input_tokens`, `total_output_tokens`, `total_cache_read_tokens`, `estimated_cost()`, `summary_by_model()`, `summary_by_call_type()`.

**Persistent:** SQLite `llm_usage` table in `corpus_log/usage_stats.db`. Flushed after each document via `UsageTracker.log_llm_usage_batch()`.

## MODEL_PRICING

| Model | Input ($/MTok) | Output ($/MTok) |
|-------|---------------|----------------|
| `claude-sonnet-4-20250514` | $3.00 | $15.00 |
| `claude-sonnet-4-5-20250929` | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 |
| `claude-haiku-4-5-20251001` | $1.00 | $5.00 |

## Adding a New LLM Call Site

1. Choose a descriptive `call_type` string
2. Add to `config.yaml` under `api.claude.model_tiers`
3. **ClaudeClient calls**: pass `call_type=`
4. **Raw API calls**: use `resolve_model_tier()` + `record_api_usage()`
5. Add model to `MODEL_PRICING` in `D02_llm_engine.py` if new

## Changing a Task's Model

Update `config.yaml` only -- no code changes needed:

```yaml
model_tiers:
  layout_analysis: "claude-sonnet-4-20250514"  # Upgrade from Haiku
```

## Related

- [Configuration Guide](03_configuration.md)
- [External APIs Reference](../reference/02_external_apis.md)
