# LLM Usage Strategy -- Model Routing & Cost Optimization

> **Date**: February 2026
> **Pipeline version**: v0.8

Comprehensive explanation of how the ESE pipeline uses Claude LLM and Vision LLM across 17 call sites, with two-tier model routing and integrated cost tracking.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Tier Architecture](#2-model-tier-architecture)
3. [Complete Call Site Inventory](#3-complete-call-site-inventory)
4. [VLM (Vision Language Model)](#4-vlm-vision-language-model)
5. [Validation Engine](#5-validation-engine)
6. [Prompt Caching](#6-prompt-caching)
7. [Cost Tracking Infrastructure](#7-cost-tracking-infrastructure)
8. [Usage Tracking & Reporting](#8-usage-tracking--reporting)
9. [Cost Optimization Strategies](#9-cost-optimization-strategies)
10. [Future Work](#10-future-work)

---

## 1. Overview

The ESE pipeline makes extensive use of Claude LLM and Vision LLM across its extraction workflow. Every LLM call is:

- **Routed** to the optimal model tier (Haiku for simple tasks, Sonnet for complex reasoning)
- **Tracked** with per-call token counts, cache statistics, and USD cost estimates
- **Persisted** to SQLite for analysis and cost reporting

### Key Numbers

| Metric | Value |
|--------|-------|
| Total LLM call sites | 17 |
| Model tiers | 2 (Haiku, Sonnet) |
| Haiku call sites | 10 |
| Sonnet call sites | 7 |
| Haiku pricing | $1.0 / $5.0 per MTok (input/output) |
| Sonnet pricing | $3.0 / $15.0 per MTok (input/output) |

---

## 2. Model Tier Architecture

### Two-Tier Strategy

The pipeline routes every LLM call to one of two model tiers based on task complexity:

**Haiku Tier** ($1/$5 per MTok) -- Fast, inexpensive tasks:
- Classification (document type, image type, layout pattern)
- Quick filtering (fast-reject, abbreviation validation)
- Simple extraction (OCR text, short descriptions)
- Batch processing (multiple items per call)

**Sonnet Tier** ($3/$15 per MTok) -- Complex reasoning tasks:
- Multi-field extraction (feasibility, recommendations)
- Visual understanding (flowcharts, charts, complex tables)
- Nuanced analysis (evidence levels, care pathways)
- Structured output with many fields

### Routing Logic

Model routing is implemented in `D02_llm_engine.py` via `resolve_model_tier()`:

1. Loads `model_tiers` from `config.yaml` on first call (cached globally)
2. Looks up the `call_type` string in the mapping
3. Returns the configured model ID or falls back to the default model

Configuration in `config.yaml`:

```yaml
api:
  claude:
    validation:
      model: "claude-sonnet-4-20250514"  # Default fallback
    model_tiers:
      abbreviation_batch_validation: "claude-haiku-4-5-20250901"
      feasibility_extraction: "claude-sonnet-4-20250514"
      # ... 17 call_types total
```

### ClaudeClient Model Resolution

The `ClaudeClient` class wraps model routing:

- `resolve_model(call_type)` checks `_model_tiers` dict from config
- Falls back to `self.default_model` if no tier mapping exists
- Every `complete_json()` and `complete_vision_json()` call accepts a `call_type` parameter

---

## 3. Complete Call Site Inventory

### Haiku Tier (10 Call Sites)

| call_type | Script | Purpose | Typical Use |
|-----------|--------|---------|-------------|
| `abbreviation_batch_validation` | D02 LLMEngine | Batch-verify 15-20 abbreviation candidates | Is "CI = confidence interval" a genuine abbreviation? |
| `abbreviation_single_validation` | D02 LLMEngine | Verify single abbreviation candidate | Fallback when batch processing fails |
| `fast_reject` | D02 LLMEngine | Quick pre-screening to filter obvious FPs | Reject "US = United States" before Sonnet |
| `document_classification` | C09 | Classify document type | "clinical trial protocol" vs "marketing material" |
| `description_extraction` | C09 | Generate 2-4 sentence document description | Document summary for metadata |
| `image_classification` | C10 | Classify visual type (table/figure/chart) | Route to appropriate extraction logic |
| `sf_only_extraction` | H02 | Extract definitions for short-form-only abbreviations | PASO D: find definition for "ALD" in context |
| `layout_analysis` | B19 | Detect page layout pattern | Single column vs two-column detection |
| `vlm_visual_enrichment` | C19, B22 | Generate visual titles and descriptions | "Figure showing Kaplan-Meier survival curves" |
| `ocr_text_fallback` | C10 | Extract text from images via OCR | Simple text extraction from screenshots |

### Sonnet Tier (7 Call Sites)

| call_type | Script | Purpose | Typical Use |
|-----------|--------|---------|-------------|
| `feasibility_extraction` | C11 | Extract structured feasibility data | Eligibility criteria, study design, endpoints |
| `recommendation_extraction` | C32 | Extract clinical recommendations | "Recommend RTX over CYC for relapsing GPA" |
| `recommendation_vlm` | C33 | Extract LoE/SoR from table images | Visual extraction of evidence level codes |
| `flowchart_analysis` | C10, C17 | Analyze clinical decision flowcharts | Node/edge graph from treatment algorithms |
| `chart_analysis` | C10 | Extract data from charts | Bar chart values, Kaplan-Meier curves |
| `vlm_table_extraction` | C10 | Extract table structure from images | Complex tables that Docling cannot parse |
| `vlm_detection` | B31 | Detect tables/figures on pages | Bounding box detection via vision |

---

## 4. VLM (Vision Language Model)

### Image Input Handling

All VLM calls follow a consistent image processing pipeline:

1. **Rendering**: PDF pages rendered at 150 DPI via PyMuPDF
2. **Scaling**: Images scaled to max 1568px on longest dimension
3. **Encoding**: Base64 encoding for API transmission
4. **Compression**: Automatic JPEG compression for images > 5MB
5. **Media type detection**: Auto-detected from image header bytes
   - `/9j/` -> JPEG
   - `R0lGOD` -> GIF
   - `UklGR` -> WebP
   - Default -> PNG

### Vision-Specific Prompts

VLM prompts are structured differently from text prompts:

- Image content block with base64-encoded image data
- Task-specific system prompt describing what to look for
- Structured JSON response format specification
- Confidence thresholds in the prompt itself

### Figure/Table/Flowchart Analysis

Each visual type has specialized VLM processing:

| Visual Type | Model | Output |
|-------------|-------|--------|
| Simple figure | Haiku | Title + 2-4 sentence description |
| Complex chart | Sonnet | Data series, axis labels, values |
| Data table | Sonnet | Headers, rows, cell content |
| Flowchart | Sonnet | Nodes, edges, conditions, drugs |
| Recommendation table | Sonnet | LoE/SoR codes per recommendation |

### Multi-Image Batching

When multiple images need processing:

- Batch similar images together where possible
- Process one page at a time for VLM detection
- Cache VLM results to avoid re-processing (e.g., `_vlm_loe_sor_cache`)
- Sequential processing with results aggregated post-hoc

---

## 5. Validation Engine

### D02 LLM Engine Architecture

The `LLMEngine` class in D02 is the central abbreviation validation coordinator:

**Single Candidate Verification:**
- Calls `ClaudeClient.complete_json()` with `call_type="abbreviation_single_validation"`
- Receives structured response: status (APPROVE/REJECT), confidence, evidence
- Handles LLM errors and schema validation failures
- Returns `ExtractedEntity` with all validation metadata

**Batch Verification:**
- Groups 15-20 candidates per LLM call
- Dynamic `max_tokens`: `max(self.max_tokens * 2, len(batch) * 100 + 200)`
- In-memory cache: `(sf_upper, lf_lower) -> validation result`
- 100ms delay between batches (rate limit management)
- Falls back to single verification on batch errors

### Fast-Reject Pre-Screening

A two-stage approach minimizes expensive Sonnet calls:

1. **Stage 1 (Haiku)**: Quick filter with `call_type="fast_reject"`
   - Batch of 20 candidates sent to Haiku
   - Rejects only if `decision == "REJECT"` AND `confidence >= 0.85`
   - Returns tuple: `(needs_review, rejected)`

2. **Stage 2 (Haiku)**: Full validation for surviving candidates
   - `call_type="abbreviation_batch_validation"` for batch processing
   - More detailed analysis with context-aware prompts

### Prompt Registry

All validation prompts are versioned and managed through a prompt registry:

- Task-specific prompts: `VERIFY_BATCH`, `VERIFY_SINGLE`, `FAST_REJECT`
- Each prompt includes:
  - System context (document type, domain)
  - Task description with examples
  - Required output JSON schema
  - Edge case handling instructions

---

## 6. Prompt Caching

### Anthropic Prompt Caching

The pipeline leverages Anthropic's prompt caching to reduce costs on repeated calls:

**How it works:**
- System messages are marked with `"cache_control": {"type": "ephemeral"}`
- First call pays full input token cost (125% for cache creation)
- Subsequent calls with identical system prompt pay only 10% of input cost
- Cache has a 5-minute TTL

**Cache-friendly prompt structuring:**
- Large, stable system prompts (task instructions, schema definitions) go in the system message
- Variable content (document text, candidate data) goes in the user message
- Batch processing benefits most (same system prompt across all batches)

### Cache Hit Rate Optimization

The pipeline structures calls to maximize cache hits:

- All abbreviation batch validation calls share the same system prompt
- Document-level calls (classification, description) share a common system prompt
- Visual analysis calls share the VLM system prompt
- Estimated cache hit rate: 60-80% for batch processing within a single document

---

## 7. Cost Tracking Infrastructure

### Recording API Usage

Every LLM call is recorded via one of two mechanisms:

**For ClaudeClient calls** (automatic):
```python
# Internal _record_usage() called after every API response
# Extracts: input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
```

**For raw Anthropic API calls** (manual):
```python
from D_validation.D02_llm_engine import record_api_usage
response = client.messages.create(...)
record_api_usage(response, model, "recommendation_extraction")
```

### Cost Calculation

`calc_record_cost()` computes USD cost for each API call:

```
base_input_cost     = input_tokens * input_price / 1M
cache_read_cost     = cache_read_tokens * input_price * 0.10 / 1M
cache_creation_cost = cache_creation_tokens * input_price * 1.25 / 1M
output_cost         = output_tokens * output_price / 1M
total_cost          = base_input_cost + cache_read_cost + cache_creation_cost + output_cost
```

### MODEL_PRICING Dictionary

All known model prices are maintained in D02:

| Model ID | Input ($/MTok) | Output ($/MTok) |
|----------|---------------|-----------------|
| `claude-sonnet-4-20250514` | $3.00 | $15.00 |
| `claude-sonnet-4-5-20250929` | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 |
| `claude-haiku-4-5-20250901` | $1.00 | $5.00 |

### LLMUsageTracker Class

Accumulates token usage across all API calls in a pipeline run:

- `record()` -- Add single usage record with tokens, cache info, and call_type
- `total_input_tokens` / `total_output_tokens` -- Running totals
- `estimated_cost()` -- Total USD cost across all calls
- `summary_by_model()` -- Aggregated statistics per model
- `summary_by_call_type()` -- Aggregated statistics per call_type
- `reset()` -- Clears all records (used between documents)

Global access: `get_usage_tracker()` returns the singleton `LLMUsageTracker`.

---

## 8. Usage Tracking & Reporting

### SQLite Persistence (Z06)

The `UsageTracker` class in Z06 persists all usage data to SQLite (`usage_stats.db`):

**Database Schema:**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `documents` | Document processing metadata | filename, started_at, finished_at, status |
| `llm_usage` | Per-call token tracking | model, call_type, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, estimated_cost_usd |
| `lexicon_usage` | Lexicon matching statistics | lexicon_name, matches_count, candidates_generated, validated_count |
| `datasource_usage` | External API query statistics | datasource_name, queries_count, results_count, errors_count |

**Indexes** on (document_id), (model), (lexicon_name), (datasource_name) for fast queries.

### Reporting Methods

| Method | Output |
|--------|--------|
| `log_llm_usage()` | Record single API call |
| `log_llm_usage_batch()` | Bulk insert from LLMUsageTracker records |
| `get_llm_stats()` | Group by model: calls, tokens, cost |
| `get_llm_stats_by_call_type()` | Group by call_type and model |
| `print_summary()` | Console output of all statistics |

### Orchestrator Cost Summaries

The orchestrator provides cost visibility at two levels:

**Per-document summary** (after each PDF is processed):
- Total LLM calls, input tokens, output tokens
- Cost breakdown by model tier
- Cache hit rate and savings

**Batch summary** (after all PDFs are processed):
- Aggregate costs across all documents
- Average cost per document
- Top cost drivers by call_type

---

## 9. Cost Optimization Strategies

### Strategy 1: Model Tier Routing (Biggest Lever)

Routing simple tasks to Haiku instead of Sonnet provides 3-5x cost savings:

| Scenario | Sonnet Cost | Haiku Cost | Savings |
|----------|-------------|------------|---------|
| 100 abbreviation batches | ~$0.90 | ~$0.30 | 67% |
| 50 document classifications | ~$0.15 | ~$0.05 | 67% |
| 20 visual enrichments | ~$0.60 | ~$0.20 | 67% |

### Strategy 2: Fast-Reject Pre-Screening

Filtering obvious false positives with Haiku before detailed validation:

- Haiku fast-reject: ~$0.01 per batch of 20 candidates
- Eliminates 30-50% of candidates before full validation
- Net savings: Avoids ~30-50% of batch validation calls

### Strategy 3: Batch Processing

Grouping multiple items per LLM call reduces per-item overhead:

- Abbreviation batches: 15-20 candidates per call (vs. 1 per call)
- System prompt tokens amortized across batch items
- Cache creation cost paid once per batch, not per item
- Dynamic batch sizing based on content length

### Strategy 4: Prompt Caching

Leveraging Anthropic's prompt caching for repeated system prompts:

- System prompt cached at 125% cost on first call
- Subsequent calls pay only 10% of input cost for cached portion
- Within a single document, most calls reuse the same system prompt
- Estimated savings: 20-40% on input token costs

### Strategy 5: Extraction Presets

Skipping unnecessary entity types reduces total LLM calls:

```yaml
extraction_pipeline:
  preset: "diseases_only"  # Only runs disease extraction
  # vs. preset: "all" which runs all 14+ entity types
```

Available presets: `standard`, `all`, `minimal`, `drugs_only`, `diseases_only`, `genes_only`, `abbreviations_only`, `feasibility_only`, `entities_only`, `clinical_entities`, `metadata_only`, `images_only`, `tables_only`.

---

## 10. Future Work

### Local Model Integration

- **Ollama / vLLM**: Run small models locally for classification tasks currently on Haiku
- **Specialized models**: Fine-tuned models for specific tasks (abbreviation validation, entity classification)
- **Hybrid routing**: Local models for simple tasks, Claude API for complex reasoning

### Async LLM Calls

- **Parallel processing**: Multiple LLM calls in flight simultaneously
- **Batch pipelines**: Process multiple documents with shared LLM batches
- **Streaming responses**: Start processing before full response arrives

### Dynamic Model Routing

- **Document complexity scoring**: Route entire documents to appropriate model tiers
- **Adaptive routing**: Learn which documents need Sonnet vs. Haiku
- **Cost budgets**: Per-document cost limits with automatic tier downgrade

### Cost Target Projections

| Scenario | Current | Target | Approach |
|----------|---------|--------|----------|
| Simple document (10 pages) | ~$0.15 | ~$0.05 | Local models for classification |
| Complex guideline (50 pages) | ~$1.50 | ~$0.80 | Async + caching optimization |
| Large trial protocol (200 pages) | ~$5.00 | ~$2.50 | Batch processing + presets |
