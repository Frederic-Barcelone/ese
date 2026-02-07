# ESE Pipeline v0.8 -- Error Handling & Recovery Analysis

> **Date**: February 4, 2026

---

## 1. Exception Architecture

### 1.1 Custom Exception Hierarchy

Defined in `A_core/A12_exceptions.py`:

```
ESEPipelineError (base)
+-- ConfigurationError          -- config_key, expected_type, actual_value
+-- ParsingError                -- file_path, page_number
+-- ExtractionError             -- extractor_name, entity_type, input_text (truncated 100 chars)
+-- EnrichmentError             -- enricher_name, entity_id, entity_type
|   +-- APIError                -- status_code, response_body (truncated 200 chars), api_name, endpoint
|       +-- RateLimitError      -- retry_after (hardcodes status_code=429)
+-- ValidationError             -- entity_id, field_name, expected_value, actual_value
+-- CacheError                  -- cache_key, operation
+-- EvaluationError             -- metric_name, file_path
```

All exceptions store `context: Dict[str, Any]` and render it in `__str__` as `[k=v, ...]`.

### 1.2 Adoption Gap

These custom exceptions are **barely used**. The codebase catches generic `Exception` instead. `Z_utils/Z01_api_client.py` defines **duplicate** `CacheError`, `APIError`, and `RateLimitError` classes (simpler, no context dict), creating import ambiguity.

---

## 2. Layer-by-Layer Error Handling

### 2.1 API Client (`Z01_api_client.py`)

**`retry_with_backoff`**: 4 attempts, retries on 429/5xx/Timeout/ConnectionError, exponential backoff + jitter. No traceback logged -- uses `str(e)` only. Line 150: silent `return None` safety net.

**`DiskCache`**: `get()`/`set()` catch at DEBUG level. `clear()` silently swallows `OSError`. `delete()` has no try/except.

**`BaseAPIClient._request()`**: 429 sleeps 60s + retries once, then `RateLimitError`. Timeout/RequestException raise `APIError` with traceback via `from e`.

**Resource leak**: `requests.Session` context manager exists but no caller uses it. PubTator3Client, ClinicalTrialsGovClient, TrialAcronymEnricher never close sessions.

### 2.2 LLM Engine (`D02_llm_engine.py`)

**Anthropic import fallback**: When not installed, all exception types become bare `Exception` -- catches everything silently.

**Config loading**: Both `resolve_model_tier()` and `ClaudeClient._load_config()` catch broad exceptions, log WARNING, fall back to defaults.

**`complete_vision_json()`**: Compression failures return `None`. The `messages.create()` call has **no try/except** -- Anthropic exceptions propagate.

**JSON parsing** (`_extract_json_any()`): Three-level fallback (markdown block, balanced brackets, raw parse). First two silently `pass` on JSONDecodeError. If all fail, returns hardcoded AMBIGUOUS.

**`verify_candidate()`**: Catches `(..., Exception)` -- bare `Exception` makes specific catches redundant. Errors become AMBIGUOUS (confidence=0.0). Pydantic `ValidationError` caught separately with `"llm_schema_error"` flag.

**Batch validation** (`_verify_batch()`): Graceful degradation -- batch failure falls back to per-candidate `verify_candidate()`.

**Fast reject**: Same `Exception` catch-all. Returns all candidates to review (conservative).

### 2.3 PDF Parsing (`B_parsing/`)

**`B01_pdf_to_docgraph.py`**: Helper methods catch `Exception` broadly, return `None` at DEBUG level. PyMuPDF properly closed via `finally`. **Critical gap**: `parse()` and `_call_partition_pdf()` have no try/except -- corrupted PDFs crash `process_pdf()`, caught only at batch level.

**`B28_docling_backend.py`**: Import catches `ImportError` -> `DOCLING_AVAILABLE = False`. Converter and table extraction have multi-level fallbacks, ultimately returning `[]` or `None` on errors.

### 2.4 Generators (`C_generators/`)

**`C04_strategy_flashtext.py`** (scispaCy): lg->sm model fallback; if both fail, `scispacy_nlp = None` with `print()` warning. UMLS KB lookup: `except Exception: pass` (silent). **Line 553**: Full scispaCy NER wrapped in `except Exception: pass` -- the **most dangerous silent swallow** in the codebase.

**`C10_vision_image_analysis.py`**: VLM calls catch exceptions, fall back to OCR text. All methods return `None` on failure with WARNING log.

### 2.5 Normalization (`E_normalization/`)

**`E04_pubtator_enricher.py`**: Individual methods catch Timeout/RequestException/JSONDecodeError -> return `None`. **Critical gap**: `enrich_batch()` has no try/except -- one `KeyError` aborts the entire batch.

**`E06_nct_enricher.py`**: 429 sleeps 60s + retries once. Other errors return `None` with WARNING.

### 2.6 Entity Processors (`I01_entity_processors.py`)

**Zero try/except blocks.** All processing methods propagate to orchestrator. Disease enrichment failure prevents all downstream processing.

### 2.7 Export (`J01_export_handlers.py`)

Per-image/table exports have try/except. **Critical gap**: `export_results()` and entity JSON writes have no try/except -- I/O failure loses all extraction work.

### 2.8 Orchestrator (`orchestrator.py`)

`process_pdf()` per-stage error isolation:

| Stages | Try/Except? | On Failure |
|--------|-------------|------------|
| 1-11 (Parsing through Citations) | **No** | Crashes process_pdf() |
| 12-13 (Care Pathways, Recommendations) | Yes (per-item) | WARNING, continues |
| 14-15 (Visual, Metadata) | Yes | WARNING, continues |
| 16 (Export) | **No** | Crashes process_pdf() |

**Only 4 of 16 stages have error isolation.**

**`process_folder()`**: Batch safety net catches `Exception`, prints traceback, assigns empty `ExtractionResult`, continues.

### 2.9 Component Factory (`H01_component_factory.py`)

Only `create_table_extractor()` and `load_rare_disease_lookup()` have error handling. All other `create_*` methods propagate -- missing lexicons crash initialization.

---

## 3. Error Propagation Traces

### 3.1 Claude API 429 During Abbreviation Validation

```
ClaudeClient._call_claude() -> AnthropicRateLimitError
  -> propagates to verify_candidate() line 754 -> _handle_llm_error()
  -> AMBIGUOUS entity (confidence=0.0, flag="llm_rate_limit_error")
  -> excluded from validated output
```

**Outcome**: Abbreviation not exported. Pipeline reports SUCCESS. No retry at LLM engine level.

### 3.2 PubTator3 Timeout During Disease Enrichment

```
PubTator3Client.autocomplete() catches Timeout -> returns None
DiseaseEnricher.enrich() -> returns disease unmodified
-> exported to JSON without MeSH ID, aliases, or normalized name
```

**Outcome**: Disease exported without enrichment data. **No flag set.** Indistinguishable from "enrichment unnecessary."

### 3.3 Corrupted PDF

```
fitz.open() or partition_pdf() raises exception
  -> propagates through parse() (no try/except)
  -> propagates through process_pdf() (no try/except)
  -> caught by process_folder() -> empty ExtractionResult, batch continues
```

**Outcome**: Full traceback printed. Document gets empty result. Batch continues.

### 3.4 VLM Call Failure

```
_call_vision_llm() catches Exception -> WARNING log
  -> OCR text fallback if available, else returns None
  -> figure exported without vision_analysis data
  -> processing_errors array + failed counter track it
```

**Outcome**: Well-instrumented -- tracked in `processing_errors` array and `analysis_stats["failed"]` counter.

### 3.5 Pydantic Validation Failure on LLM Response

```
VerificationResult.model_validate(raw) raises ValidationError
  -> AMBIGUOUS entity (confidence=0.0, flag="llm_schema_error", raw response preserved)
```

### 3.6 scispaCy Model Fails to Load

```
spacy.load("en_core_sci_lg") fails -> fallback to sm -> fails
  -> print() warning, scispacy_nlp = None
  -> pipeline runs with reduced recall (only FlashText + regex)
```

**Outcome**: Pipeline reports SUCCESS. scispaCy NER silently disabled. Only indicator is a `print()` line in console.

### 3.7 Export JSON Write Failure

```
json.dump() or open() fails -> propagates to process_folder() -> empty ExtractionResult
```

**Outcome**: ALL extraction work lost. `export_extracted_text()` has its own try/except but entity exports do not.

### 3.8 Malformed Config YAML

```
yaml.safe_load() fails -> returns {}
```

**Impact**: Wrong paths, degraded tier routing, relaxed filtering. Pipeline runs but degraded.

---

## 4. Silent Data Loss Scenarios

| Scenario | Data Lost | Visibility | Severity |
|----------|-----------|------------|----------|
| PubTator timeout | MeSH ID, aliases, normalized name | **INVISIBLE** -- no flag | **Critical** |
| scispaCy load failure | All biomedical NER entities | Console `print()` only | **Critical** |
| scispaCy NER exception (line 553) | Same as above, mid-processing | **FULLY SILENT** (`except: pass`) | **Critical** |
| UMLS KB lookup failure (line 497) | UMLS codes | **FULLY SILENT** (`except: pass`) | Medium |
| Claude 429 rate limit | Validation downgraded to AMBIGUOUS | Entity flag (not in exported JSON) | Medium |
| Malformed config | 3x cost increase, relaxed filtering | Console `[WARN]` only | Medium |
| Pydantic validation failure | Same as 429 | Entity flag `llm_schema_error` | Low |
| VLM failure | Figure analysis data | `processing_errors` array (well-tracked) | Low |
| DiskCache corruption | Cached API responses | DEBUG log (invisible) | Low |

---

## 5. Code Quality Issues

### 5.1 Overly Broad Exception Catches

| File | Line(s) | Issue |
|------|---------|-------|
| `D02_llm_engine.py` | 754, 991-993, 1193 | `except (..., Exception)` catches everything including bugs |
| `orchestrator.py` | 232 | Hides config errors |
| `B01_pdf_to_docgraph.py` | 758 | Hides parsing bugs |
| `B28_docling_backend.py` | 124 | Hides converter bugs |
| `C04_strategy_flashtext.py` | 277 | Hides UMLS linker bugs |

### 5.2 Silent Exception Swallowing

| File | Line | Impact |
|------|------|--------|
| `Z01_api_client.py` | 412 | Low -- cache cleanup |
| `D02_llm_engine.py` | 620-621, 629-630 | Low -- JSON fallback chain |
| `C04_strategy_flashtext.py` | 497 | **Medium** -- lost UMLS enrichment |
| **`C04_strategy_flashtext.py`** | **553** | **HIGH** -- entire NER silently fails |

### 5.3 Missing Error Handling

| File | Location | Impact |
|------|----------|--------|
| `I01_entity_processors.py` | All 6 `process_*` methods | **Critical** -- cascading failure |
| `J01_export_handlers.py` | `export_results()` | **Critical** -- work lost |
| `orchestrator.py` | `process_pdf()` stages 1-11 | High -- one stage kills document |
| `E04_pubtator_enricher.py` | `enrich_batch()` | High -- single failure aborts batch |
| `B01_pdf_to_docgraph.py` | `parse()`, `_call_partition_pdf()` | High -- caught at batch level |
| `H01_component_factory.py` | All `create_*` except table | Medium -- fails fast |

### 5.4 Error Context Loss

Most WARNING/ERROR logs use `str(e)` rather than `logger.exception()`. Only `process_folder()` at line 1751 properly logs a full traceback.

### 5.5 Resource Leaks

| File | Resource | Issue |
|------|----------|-------|
| `Z01_api_client.py` | `requests.Session` | Context manager exists but never used |
| `E04_pubtator_enricher.py` | `_session` | Never closed |
| `E06_nct_enricher.py` | `_session` (2 clients) | Never closed |
| `J01_export_handlers.py` | `fitz.Document` | May leak on exception |

### 5.6 Inconsistent Error Reporting

Errors reported through 4 mechanisms with no unified tracking:

| Mechanism | Queryable? | Preserved? |
|-----------|-----------|------------|
| `logger.warning/error()` | Log files only | Until rotation |
| `print()` / `batch_printer` | Console log | Until deleted |
| Entity flags | Per-document JSON (if exported) | Per-document |
| `processing_errors` array | Figures JSON | Per-document |

No errors are persisted to SQLite.

---

## 6. Gaps vs. SOTA

| Capability | Status |
|------------|--------|
| Retry with backoff | **At parity** -- 3 profiles, exponential + jitter |
| Document isolation | **At parity** -- per-document try/except in batch |
| Schema validation | **At parity** -- Pydantic models + invariants |
| Provenance tracking | **Ahead** -- full provenance on every entity |
| Graceful degradation | **Partial** -- 4/16 stages only |
| Circuit breaker | **Missing** -- would prevent accumulated retry delays during outages |
| Model fallback chain | **Missing** -- Sonnet -> Haiku -> cached/heuristic |
| Dead letter queue | **Missing** |
| Structured error tracking | **Missing** -- errors should go to SQLite |
| Checkpointing | **Missing** -- resume from last successful stage |
| Health checks | **Missing** -- probe dependencies before batch |
| Idempotency | **Missing** |
| Observability | **Gap** -- timing bars + token tracker vs. OpenTelemetry/Prometheus |
| LLM-level retry | **Missing** -- relies on SDK |
| Enrichment failure tracking | **Missing** -- no flags on failed entities |

### Key SOTA Patterns

**Circuit breaker**: After N failures in window, fast-fail with fallback. Currently a 10-min outage wastes ~10 min on retries.

**Checkpointing**: Save intermediate state per stage. A document failing at export (stage 16/16) currently requires full reprocessing.

**Structured error tracking**: Add `pipeline_errors` table to `usage_stats.db` (document_id, stage, error_type, recovery_action, data_loss_risk).

**Pre-flight health checks**: Probe Claude API, PubTator3, disk space before batch.

---

## 7. Recommendations

### Priority 1 -- Fix Critical Silent Failures

1. **Log scispaCy NER catch-all** (`C04:553`): Replace `except Exception: pass` with `except Exception as e: logger.error("scispaCy NER failed: %s", e, exc_info=True)`.
2. **Flag enrichment failures** (`E04`): Set `enrichment_failed=True` on entities when PubTator returns `None`.
3. **Per-stage try/except in `process_pdf()`**: Wrap stages 1-11 individually so disease failure doesn't prevent gene/drug/author extraction.

### Priority 2 -- Error Visibility

4. **Structured error tracking in SQLite**: `pipeline_errors` table in `usage_stats.db`.
5. **Replace `print()` with `logger`** in orchestrator, component factory, export handlers.
6. **Add `exc_info=True`** to WARNING/ERROR log calls.

### Priority 3 -- Resilience

7. **Pre-flight health checks**: Probe Claude API, PubTator3, disk space.
8. **LLM-level retry**: 1 retry with backoff in `verify_candidate()` before AMBIGUOUS.
9. **Circuit breaker for Claude API**.
10. **Model fallback chain**: Try secondary model before AMBIGUOUS.

### Priority 4 -- Architectural

11. **Adopt custom exception hierarchy**: Replace generic catches with `ESEPipelineError` subclasses. Remove Z01 duplicates.
12. **Checkpointing**: Save per-stage state; resume on failure.
13. **`enrich_batch()` isolation**: Per-entity try/except.
14. **Close HTTP sessions**: Context managers or `__del__` cleanup.

---

## References

- [Retries, Fallbacks, and Circuit Breakers in LLM Apps](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [Building Bulletproof LLM Applications: SRE Best Practices (Google Cloud, 2025)](https://medium.com/google-cloud/building-bulletproof-llm-applications-a-guide-to-applying-sre-best-practices-1564b72fd22e)
- [Error Handling Best Practices for Production LLM Applications](https://markaicode.com/llm-error-handling-production-guide/)
- [Circuit Breaker for LLM with Retry and Backoff -- Anthropic API Example](https://medium.com/@spacholski99/circuit-breaker-for-llm-with-retry-and-backoff-anthropic-api-example-typescript-1f99a0a0cf87)
- [The Spectrum of Failure Models in AI Agentic Systems](https://cenrax.substack.com/p/the-spectrum-of-failure-models-in)
- [Fault-Tolerant Data Pipeline Design](https://www.linkedin.com/advice/3/what-techniques-can-you-use-design-fault-tolerant-i5ref)
