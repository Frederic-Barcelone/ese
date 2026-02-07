# Layer D: LLM Validation

## Purpose

`D_validation/` is the high-precision filtering layer. Uses Claude API and PASO heuristics to verify candidates from `C_generators/`, rejecting false positives. Minimizes API costs through heuristic pre-filtering. 4 modules.

See also: [Data Flow](../architecture/02_data_flow.md) | [Pipeline Overview](../architecture/01_overview.md)

---

## Key Modules

### D01_prompt_registry.py -- Versioned Prompt Store

| Component | Description |
|-----------|-------------|
| `PromptTask` | Enum: `VERIFY_DEFINITION_PAIR`, `VERIFY_SHORT_FORM_ONLY`, `VERIFY_BATCH`, `FAST_REJECT`, `VERIFY_DISEASE`, etc. |
| `PromptBundle` | Frozen model: `system_prompt`, `user_template`, `output_schema`, `version`, `prompt_bundle_hash`. |
| `PromptRegistry` | Maps `(PromptTask, version)` to `PromptBundle`. |

### D02_llm_engine.py -- Validation Engine and Cost Optimization

**Validation:**

| Component | Description |
|-----------|-------------|
| `ClaudeClient` | Anthropic SDK client with model tier routing and prompt caching. |
| `LLMEngine` | Retry with backoff, rate limiting, batch processing, fast-reject pre-screening. |
| `VerificationResult` | Pydantic model: status, confidence, reasoning, evidence. |

**Cost Optimization:**

| Component | Description |
|-----------|-------------|
| `MODEL_PRICING` | Model IDs to `(input_cost, output_cost)` per MTok. |
| `calc_record_cost()` | Cost calculation with cache read (10%) and creation (125%) accounting. |
| `LLMUsageTracker` | Accumulates records. Provides `summary_by_model()`, `summary_by_call_type()`, `estimated_cost()`. |
| `resolve_model_tier()` | Resolves model for `call_type` from `config.yaml`. Falls back to Sonnet 4. |
| `record_api_usage()` | For raw `anthropic.Anthropic()` calls (B/C layers) to record into global tracker. |

### D03_validation_logger.py -- JSONL Logging

Structured JSONL logs to `corpus_log/`. One file per run. Tracks per-entity-type stats (total, validated, rejected, ambiguous, errors).

### D04_quote_verifier.py -- Anti-Hallucination

| Component | Description |
|-----------|-------------|
| `QuoteVerifier` | Checks quoted text exists in source document. Configurable `min_match_ratio` (0.85). |
| `NumericalVerifier` | Verifies numerical values exist in source text. |
| `ExtractionVerifier` | Combined quotes + numbers verification. |

---

## Public Interfaces

```python
# Prompt Registry
registry = PromptRegistry()
bundle = registry.get(PromptTask.VERIFY_DISEASE_BATCH)

# Validation
client = ClaudeClient(model="claude-sonnet-4-20250514")
engine = LLMEngine(client=client)
results = engine.verify_candidates_batch(candidates, batch_size=15, delay_ms=100)

# Cost Optimization
model = resolve_model_tier("layout_analysis")
record_api_usage(response, model, "layout_analysis")
cost = calc_record_cost("claude-haiku-4-5-20251001", input_tokens=1000, output_tokens=200,
                        cache_read_tokens=500)
```

---

## Usage Patterns

### PASO Heuristic Pre-Filtering

| Rule | Action | Example |
|------|--------|---------|
| **PASO A** | Auto-approve statistical abbreviations | CI, HR, SD, OR, RR, BMI |
| **PASO B** | Auto-reject country codes in geographic context | US, UK, EU |
| **PASO C** | Auto-enrich hyphenated abbreviations via ClinicalTrials.gov | anti-C5, IL-6 |
| **PASO D** | LLM SF-only extraction for candidates without definitions | Short forms with no long form |

Only candidates surviving PASO rules reach the LLM.

### Model Tier Routing

Every call site passes a `call_type` mapped to a model in `config.yaml`:

| Tier | Models | Example call_types |
|------|--------|-------------------|
| **Haiku** ($1/$5 per MTok) | `claude-haiku-4-5-20251001` | `abbreviation_batch_validation`, `document_classification`, `layout_analysis` |
| **Sonnet** ($3/$15 per MTok) | `claude-sonnet-4-20250514` | `feasibility_extraction`, `flowchart_analysis`, `vlm_table_extraction` |

### Prompt Caching

`ClaudeClient._call_claude()` applies `cache_control: {"type": "ephemeral"}` on system prompts. Cache reads cost 10% of input price; creation costs 125%.

### Validation Flow

```
Candidates from C_generators
  --> PASO heuristics (auto-approve / auto-reject)
  --> LLMEngine.verify_batch() --> Claude API
  --> VerificationResult (VALIDATED / REJECTED / AMBIGUOUS)
  --> QuoteVerifier (anti-hallucination)
  --> ValidationLogger (JSONL audit trail)
```
