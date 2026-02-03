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

### D02_llm_engine.py -- Claude Validation Engine

Core validation engine using the Anthropic Claude API.

| Component | Description |
|-----------|-------------|
| `LLMClient` | Protocol (interface) for LLM backends, allowing different implementations. |
| `ClaudeClient` | Anthropic SDK client implementation with model selection. |
| `LLMEngine` | Main verifier: retry logic with exponential backoff, rate limiting, batch processing, structured JSON output parsing. |
| `VerificationResult` | Pydantic model for structured validation responses (status, confidence, reasoning, evidence). |

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
