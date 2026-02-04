# Layer D: LLM Validation

## Purpose

`D_validation/` is the high-precision filtering layer. It uses Claude API calls (and deterministic PASO heuristics) to verify candidates produced by `C_generators/`, rejecting false positives and assigning validation status. The layer is designed to minimize API costs through heuristic pre-filtering while maintaining high precision on the candidates that do reach the LLM.

This layer contains 4 focused modules.

See also: [Data Flow](../architecture/02_data_flow.md) | [Pipeline Overview](../architecture/01_overview.md)

---

## Key Modules

### D01_prompt_registry.py -- Versioned Prompt Store

Centralized registry for all LLM validation prompts. Every prompt is versioned with a deterministic hash for reproducibility.

| Component | Description |
|-----------|-------------|
| `PromptTask` | Enum of validation task types: `VERIFY_DEFINITION_PAIR`, `VERIFY_SHORT_FORM_ONLY`, `VERIFY_BATCH`, `FAST_REJECT`, `VERIFY_DISEASE`, `VERIFY_DISEASE_BATCH`, `VERIFY_AUTHOR_BATCH`, `VERIFY_CITATION_BATCH`. |
| `PromptBundle` | Frozen Pydantic model containing `system_prompt`, `user_template`, `output_schema`, `version`, and `prompt_bundle_hash`. |
| `PromptRegistry` | Central store mapping `(PromptTask, version)` to `PromptBundle`. Tracks latest version per task. |

### D02_llm_engine.py -- Claude Validation Engine & Cost Optimization

Core validation engine using the Anthropic Claude API. Also houses the LLM cost optimization infrastructure: model tier routing, usage tracking, prompt caching, and cost calculation.

**Validation Components:**

| Component | Description |
|-----------|-------------|
| `LLMClient` | Protocol (interface) for LLM backends, allowing different implementations. |
| `ClaudeClient` | Anthropic SDK client implementation with model tier routing and prompt caching. |
| `LLMEngine` | Main verifier: retry logic with exponential backoff, rate limiting, batch processing, structured JSON output parsing, fast-reject pre-screening. |
| `VerificationResult` | Pydantic model for structured validation responses (status, confidence, reasoning, evidence). |

**Cost Optimization Components:**

| Component | Description |
|-----------|-------------|
| `MODEL_PRICING` | Dict mapping model IDs to `(input_cost, output_cost)` per million tokens. Covers Sonnet 4, Sonnet 4.5, Haiku 3.5, and Haiku 4.5. |
| `calc_record_cost()` | Shared cost calculation accounting for base input, cache reads (10% of input price), and cache creation (125% of input price). |
| `LLMUsageRecord` | Dataclass for a single API call: model, input/output tokens, call_type, cache tokens. |
| `LLMUsageTracker` | Accumulates records across all API calls in a run. Provides `summary_by_model()`, `summary_by_call_type()`, and `estimated_cost()`. |
| `get_usage_tracker()` | Returns the global `LLMUsageTracker` singleton shared across all `ClaudeClient` instances. |
| `resolve_model_tier()` | Resolves the model to use for a given `call_type` from `config.yaml` `model_tiers`. Lazy-loads config with caching. Falls back to Sonnet 4 if no mapping exists. |
| `record_api_usage()` | Standalone function for raw `anthropic.Anthropic()` calls (B_parsing, C_generators) to record token usage into the global tracker. |

### D03_validation_logger.py -- Structured JSONL Logging

| Component | Description |
|-----------|-------------|
| `ValidationLogger` | Writes structured JSONL logs to `corpus_log/` directory. One log file per run. In-memory buffering for batch writes. Tracks statistics per entity type (total, validated, rejected, ambiguous, errors). |

### D04_quote_verifier.py -- Anti-Hallucination Verification

Post-LLM verification that checks whether quoted text and numbers in LLM responses actually exist in the source document.

| Component | Description |
|-----------|-------------|
| `QuoteVerifier` | Exact and fuzzy text matching against source document. Configurable `min_match_ratio` (default 0.85). Returns `QuoteVerificationResult` with match ratio and position. |
| `NumericalVerifier` | Verifies that numerical values referenced by the LLM exist in the source text. Returns `NumericalVerificationResult` with found positions. |
| `ExtractionVerifier` | Combined verifier for full extraction validation (quotes + numbers). |

---

## Public Interfaces

### PromptRegistry

```python
from D_validation.D01_prompt_registry import PromptRegistry, PromptTask

registry = PromptRegistry()
bundle = registry.get(PromptTask.VERIFY_DISEASE_BATCH)
# bundle.system_prompt, bundle.user_template, bundle.version, bundle.prompt_bundle_hash
```

### LLMEngine

```python
from D_validation.D02_llm_engine import LLMEngine, ClaudeClient

client = ClaudeClient(model="claude-sonnet-4-20250514")
engine = LLMEngine(client=client)
results = engine.verify_batch(candidates, doc_context)
# Each result: VerificationResult with status, confidence, reasoning
```

### Cost Optimization Functions

```python
from D_validation.D02_llm_engine import (
    resolve_model_tier, record_api_usage, get_usage_tracker, calc_record_cost,
)

# Resolve model for a call type from config.yaml
model = resolve_model_tier("layout_analysis")  # Returns Haiku or Sonnet

# Record usage from raw anthropic.Anthropic() calls
response = client.messages.create(model=model, ...)
record_api_usage(response, model, "layout_analysis")

# Calculate cost for a single call
cost = calc_record_cost("claude-haiku-4-5-20250901", input_tokens=1000, output_tokens=200,
                        cache_read_tokens=500)

# Get accumulated usage across the run
tracker = get_usage_tracker()
print(tracker.estimated_cost())
print(tracker.summary_by_call_type())
```

### QuoteVerifier

```python
from D_validation.D04_quote_verifier import QuoteVerifier

verifier = QuoteVerifier(min_match_ratio=0.85)
result = verifier.verify("patients were randomized", source_text)
# result.verified, result.match_ratio, result.position
```

### ValidationLogger

```python
from D_validation.D03_validation_logger import ValidationLogger

logger = ValidationLogger(log_dir="corpus_log", run_id="VAL_20240115")
logger.log_validation(candidate, entity, status="VALIDATED")
logger.flush()
# logger.stats: {'total': 100, 'validated': 85, 'rejected': 10, ...}
```

---

## Usage Patterns

### PASO Heuristic Pre-Filtering

Before any LLM call, abbreviation candidates pass through deterministic PASO heuristic rules (configured in `A_core/A04_heuristics_config.py`):

| Rule | Action | Example |
|------|--------|---------|
| **PASO A** | Auto-approve known statistical abbreviations | CI, HR, SD, OR, RR, IQR, BMI |
| **PASO B** | Auto-reject country codes in geographic context | US, UK, EU |
| **PASO C** | Auto-enrich hyphenated abbreviations via ClinicalTrials.gov | anti-C5, IL-6 |
| **PASO D** | LLM SF-only extraction for candidates without definitions | Short forms with no long form in document |

Only candidates that survive PASO rules are sent to the LLM, reducing API costs.

### Model Tier Routing

Every LLM call site passes a `call_type` string identifying the task. `ClaudeClient` and raw API callers use `resolve_model_tier(call_type)` to route simple tasks to Haiku (cheaper) and complex reasoning to Sonnet (higher quality):

| Tier | Models | Example call_types |
|------|--------|-------------------|
| **Haiku** ($1/$5 per MTok) | `claude-haiku-4-5-20250901` | `abbreviation_batch_validation`, `document_classification`, `layout_analysis`, `image_classification` |
| **Sonnet** ($3/$15 per MTok) | `claude-sonnet-4-20250514` | `feasibility_extraction`, `flowchart_analysis`, `vlm_table_extraction`, `recommendation_extraction` |

Tier mappings are configured in `config.yaml` under `api.claude.model_tiers`. See [Cost Optimization Guide](../guides/05_cost_optimization.md) for the full call_type table.

### Prompt Caching

`ClaudeClient._call_claude()` applies `cache_control: {"type": "ephemeral"}` on system prompts, enabling Anthropic's prompt caching. Cache reads cost 10% of the input token price; cache creation costs 125%. This saves significantly on repeated system prompts across batch validation calls.

### Validation Flow

```
Candidates from C_generators
  |
  v
PASO heuristics (auto-approve / auto-reject)
  |
  v
LLMEngine.verify_batch() --> Claude API
  |
  v
VerificationResult (VALIDATED / REJECTED / AMBIGUOUS)
  |
  v
QuoteVerifier (anti-hallucination check)
  |
  v
ValidationLogger (JSONL audit trail)
```

### Prompt Versioning

All prompts are versioned and hashed for reproducibility. The hash is stored in `ProvenanceMetadata.prompt_bundle_hash`, enabling exact reproduction of any validation decision.

```python
bundle = registry.get(PromptTask.VERIFY_BATCH)
# bundle.version = "v2.0"
# bundle.prompt_bundle_hash = "a1b2c3d4..."  (deterministic)
```

### Batch Processing

The `LLMEngine` supports batch validation to reduce API round trips. Multiple candidates are grouped into a single prompt with structured JSON output, parsed back into individual `VerificationResult` objects.

### Error Handling

- **Rate limiting**: Built-in retry with exponential backoff for `RateLimitError`.
- **API errors**: Caught and logged; candidates receive `AMBIGUOUS` status on failure.
- **Timeout**: Configurable timeout per API call.
- **Hallucination detection**: `QuoteVerifier` downgrades confidence when LLM-quoted text cannot be found in the source document.
