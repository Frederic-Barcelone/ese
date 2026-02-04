# ESE Pipeline v0.8 — Error Handling & Recovery Analysis

> **Date**: February 4, 2026
> **Based on**: Exhaustive code analysis of 68 files, 13 key modules read in full

---

## Table of Contents

1. [Exception Architecture](#1-exception-architecture)
2. [Layer-by-Layer Error Handling Inventory](#2-layer-by-layer-error-handling-inventory)
3. [Error Propagation Traces](#3-error-propagation-traces)
4. [Silent Data Loss Scenarios](#4-silent-data-loss-scenarios)
5. [Code Quality Issues](#5-code-quality-issues)
6. [Gaps vs. SOTA](#6-gaps-vs-sota)
7. [Recommendations](#7-recommendations)

---

## 1. Exception Architecture

### 1.1 Custom Exception Hierarchy

Defined in `A_core/A12_exceptions.py` (lines 40–421):

```
ESEPipelineError (base)
├── ConfigurationError          — config_key, expected_type, actual_value
├── ParsingError                — file_path, page_number
├── ExtractionError             — extractor_name, entity_type, input_text (truncated 100 chars)
├── EnrichmentError             — enricher_name, entity_id, entity_type
│   ├── APIError                — status_code, response_body (truncated 200 chars), api_name, endpoint
│   │   └── RateLimitError      — retry_after (hardcodes status_code=429)
├── ValidationError             — entity_id, field_name, expected_value, actual_value
├── CacheError                  — cache_key, operation
└── EvaluationError             — metric_name, file_path
```

All exceptions inherit from `ESEPipelineError`, which stores a `context: Dict[str, Any]` and renders it in `__str__` as `[k=v, ...]`.

### 1.2 Adoption Gap

These custom exceptions are **barely used** in the actual pipeline code. The codebase predominantly catches generic Python exceptions (`Exception`, `requests.RequestException`, Anthropic SDK exceptions). The structured hierarchy in A12 is largely theoretical infrastructure that is not wired into most error paths.

Additionally, `Z_utils/Z01_api_client.py` (lines 235–256) defines **duplicate** `CacheError`, `APIError`, and `RateLimitError` classes — simpler versions without the context dict. This creates import ambiguity: callers could accidentally import the wrong one.

---

## 2. Layer-by-Layer Error Handling Inventory

### 2.1 API Client Layer (`Z_utils/Z01_api_client.py`)

#### `retry_with_backoff` Decorator (lines 64–153)

| Aspect | Detail |
|--------|--------|
| Loop | `for attempt in range(max_retries + 1)` — `max_retries=3` means 4 total attempts |
| Retryable HTTP codes | 429, 500, 502, 503, 504 |
| Retryable exceptions | `Timeout`, `ConnectionError` |
| Backoff formula | `base_delay * 2^attempt + jitter(0–50%)` |
| Final failure | Re-raises original exception (traceback preserved) |
| Logging | WARNING on retry, ERROR on final failure — uses `str(e)` only, **no traceback logged** |
| Edge case | Line 150: fallback `return None` if loop exits without returning — silent safety net |

#### `DiskCache` Error Handling

| Method | Lines | Exception | Recovery | Log Level |
|--------|-------|-----------|----------|-----------|
| `get()` | 342 | `json.JSONDecodeError`, `OSError` | Returns `None` | DEBUG (invisible in production) |
| `set()` | 371 | `TypeError`, `OSError` | Returns `False` | DEBUG (invisible in production) |
| `clear()` | 412 | `OSError` | `pass` — **SILENT SWALLOW** | None |
| `delete()` | 375–394 | None | **No try/except at all** — `OSError` propagates | N/A |

#### `BaseAPIClient._request()` (lines 543–614)

| Lines | Exception | Recovery | Context Preserved? |
|-------|-----------|----------|-------------------|
| 585–594 | HTTP 429 | Sleep 60s, retry once; if second 429, raises `RateLimitError` | Yes (`from e`) |
| 603–605 | `requests.Timeout` | Raises `APIError` | Yes (`from e`) |
| 607–610 | `requests.RequestException` | Raises `APIError`, extracts `status_code` | Yes (`from e`) |
| 612–614 | `json.JSONDecodeError` | Raises `APIError` | Yes (`from e`) |

**Resource management**: Creates `self._session = requests.Session()` (line 535). Context manager (`__enter__`/`__exit__`) exists but **not used by callers** — PubTator3Client, ClinicalTrialsGovClient, and TrialAcronymEnricher all instantiate the client without `with` blocks. Sessions are never explicitly closed.

---

### 2.2 LLM Engine Layer (`D_validation/D02_llm_engine.py`)

#### Anthropic Import Fallback (lines 255–268)

```python
try:
    import anthropic
    from anthropic import APIConnectionError as AnthropicConnectionError
    # ... 4 more imports
except ImportError:
    anthropic = None
    AnthropicAPIError = Exception  # Fallback to bare Exception
```

**Problem**: When anthropic is not installed, all `except (AnthropicRateLimitError, ...)` clauses become `except (Exception, Exception, ...)` — catching **everything**, including `TypeError`, `ValueError`, even `KeyboardInterrupt`-derived exceptions. All errors silently convert to "LLM API failures."

#### Config Loading (lines 210–230, 381–403)

| Method | Lines | Exception | Recovery |
|--------|-------|-----------|----------|
| `resolve_model_tier()` | 227 | `OSError`, `yaml.YAMLError`, `TypeError`, `KeyError` | WARNING log, `_model_tier_cache = {}`, falls back to default model |
| `ClaudeClient._load_config()` | 401 | `OSError`, `IOError`, `yaml.YAMLError`, `KeyError`, `TypeError` | WARNING log, returns `{}` |

#### `complete_vision_json()` (lines 445–520)

| Lines | Condition | Recovery |
|-------|-----------|----------|
| 484–486 | Compression fails | WARNING log, returns `None` |
| 487–489 | Image oversized, `auto_compress=False` | WARNING log, returns `None` |
| 494 | `self.client.messages.create()` | **NO try/except** — Anthropic API exceptions propagate uncaught |

#### JSON Parsing Chain — `_extract_json_any()` (lines 609–637)

Three-level fallback:

| Level | Lines | Strategy | On Failure |
|-------|-------|----------|------------|
| 1 | 618–621 | Markdown code block extraction | `except JSONDecodeError: pass` — **SILENT** |
| 2 | 627–630 | Balanced bracket extraction (loop) | `except JSONDecodeError: pass` — **SILENT** |
| 3 | 633–637 | Parse entire text | `except JSONDecodeError:` DEBUG log, returns `None` |

If all three fail, `_extract_json_object()` (lines 639–652) returns a hardcoded AMBIGUOUS response. This is the critical conversion point for malformed LLM output.

#### `verify_candidate()` (lines 709–770)

```python
except (AnthropicRateLimitError, AnthropicConnectionError,
        AnthropicStatusError, AnthropicAPIError, Exception) as e:
```

**Problem**: The tuple ends with bare `Exception`, making the specific Anthropic catches redundant. Every possible error — including genuine bugs — gets caught and converted to AMBIGUOUS status via `_handle_llm_error()`.

**Recovery**: Creates AMBIGUOUS entity with:
- `confidence_score = 0.0`
- `flags = ["llm_rate_limit_error" | "llm_connection_error" | "llm_status_error" | "llm_api_error"]`
- `raw_llm_response = {"error": str(error), "error_type": "..."}`

Pydantic `ValidationError` on LLM response caught separately (lines 757–759) → AMBIGUOUS with `"llm_schema_error"` flag, raw response preserved.

#### Batch Validation Fallback — `_verify_batch()` (lines 963–995)

Two-level exception handling:

| Lines | Exception | Recovery |
|-------|-----------|----------|
| 988–990 | Anthropic-specific | WARNING log, falls back to per-candidate `verify_candidate()` |
| 991–993 | `Exception` | ERROR log, also falls back to per-candidate |

This is a genuine **graceful degradation** pattern: batch failure degrades to individual validation, preserving all candidates.

#### Fast Reject (lines 1165–1214)

Line 1193: `except (AnthropicRateLimitError, ..., Exception)` — same `Exception` catch-all. Recovery: returns all candidates to Sonnet review (no rejection applied). This is conservative — false negatives are preferred over false positives in the fast-reject stage.

---

### 2.3 PDF Parsing Layer (`B_parsing/`)

#### `B01_pdf_to_docgraph.py`

| Method | Lines | Exception | Recovery | Resource Cleanup |
|--------|-------|-----------|----------|-----------------|
| `_get_element_page_num()` | 758 | `Exception` (too broad) | DEBUG log, returns `None` | N/A |
| `_bbox_from_element()` | 777 | `Exception` (too broad) | DEBUG log, returns `None`; falls back to zeroed BoundingBox | N/A |
| `_extract_with_pymupdf()` | 705 | — | — | `finally: doc.close()` (proper) |
| `_is_scanned_pdf()` | 743 | — | — | `finally: doc.close()` (proper) |
| `_get_page_dimensions()` | 959 | — | — | `finally: doc.close()` (proper) |
| **`parse()`** | 327–618 | **NONE** | Exception propagates | No cleanup needed |
| **`_call_partition_pdf()`** | 660 | **NONE** | Exception propagates | No cleanup needed |

**Critical gap**: `parse()` and `_call_partition_pdf()` have no try/except. A corrupted PDF that causes `partition_pdf()` or `fitz.open()` to throw will crash the entire `process_pdf()` method. This is caught only at the `process_folder()` batch level.

#### `B28_docling_backend.py`

| Method | Lines | Exception | Recovery |
|--------|-------|-----------|----------|
| Module import | 55 | `ImportError` | `DOCLING_AVAILABLE = False`, WARNING log |
| `_create_converter()` | 124 | `NameError`, `ImportError`, `Exception` | WARNING log, falls back to basic pipeline options |
| `extract_tables()` | 183 | `Exception` | ERROR log (no traceback), returns `[]` |
| `_convert_table()` inner | 231–247 | `Exception` | WARNING log, falls back to `_extract_from_table_data()` |
| `_convert_table()` outer | 268–270 | `Exception` | WARNING log, returns `None` (table skipped) |
| `_extract_from_table_data()` | 297 | `Exception` | DEBUG log, returns empty `([], [])` |

---

### 2.4 Generator Layer (`C_generators/`)

#### `C04_strategy_flashtext.py` — scispaCy Integration

| Location | Lines | Exception | Recovery | Severity |
|----------|-------|-----------|----------|----------|
| Module import | 40–57 | `ImportError` | `SCISPACY_AVAILABLE = False` | INFO-level impact |
| Model load (lg) | 258 | `OSError` | Falls back to sm model | Low |
| Model load (sm) | 261–282 | `OSError` | `scispacy_nlp = None`, `print()` warning | Medium |
| UMLS linker | 277 | `Exception` (too broad) | `print()` warning, continues without linker | Medium |
| UMLS KB lookup | 497 | `Exception` | `pass` — **SILENT SWALLOW** | Medium |
| **Full scispaCy NER** | **553** | **`Exception`** | **`pass` — SILENT SWALLOW** | **HIGH** |

Line 553 is the **most dangerous silent swallow** in the entire codebase. If the full-document scispaCy NLP analysis fails for any reason, the error is completely lost — no logging, no warning, no flag. Processing continues silently without scispaCy NER results.

#### `C10_vision_image_analysis.py` — VLM Calls

`_call_vision_llm()` (lines 703–755):

| Lines | Exception | Recovery |
|-------|-----------|----------|
| 737–741 | `AttributeError` | Vision unavailable → OCR text fallback |
| 743–755 | `Exception` | Inspects `str(e)` for "5 MB"/"size"/"exceeds" to classify; OCR fallback if available |

Each analysis method follows the same pattern:

| Method | Parse Error Line | Recovery |
|--------|-----------------|----------|
| `analyze_flowchart` | 405–407 | Returns `None` |
| `analyze_chart` | 495–497 | Returns `None` |
| `analyze_table` | 548–550 | Returns `None` |
| `analyze_glossary_table` | 596–598 | Returns `None` |
| `classify_image` | 645–647 | Returns default dict |

All analysis methods: WARNING log + return `None`. Callers check for `None` and skip the figure analysis.

---

### 2.5 Normalization Layer (`E_normalization/`)

#### `E04_pubtator_enricher.py`

| Method | Lines | Exception | Recovery | Log Level |
|--------|-------|-----------|----------|-----------|
| `autocomplete()` | 134 | `requests.Timeout` | Returns `None` | WARNING |
| `autocomplete()` | 137 | `requests.RequestException` | Returns `None` | WARNING |
| `autocomplete()` | 140 | `json.JSONDecodeError` | Returns `None` | WARNING |
| `search_entity()` | 196 | `requests.RequestException` | Returns `None` | WARNING |
| `search_entity()` | 199 | `json.JSONDecodeError` | Returns `None` | WARNING |
| **`enrich()`** | — | **NONE** | Propagates | — |
| **`enrich_batch()`** | — | **NONE** | Propagates — single failure aborts batch | — |

**Critical gap**: `enrich_batch()` iterates over diseases calling `enrich()` for each. If `enrich()` raises an unexpected exception (not from the HTTP client, but e.g., a `KeyError` while parsing the PubTator response), the entire batch fails — all subsequent diseases in the batch are not enriched.

#### `E06_nct_enricher.py`

| Method | Lines | Exception | Recovery |
|--------|-------|-----------|----------|
| `get_trial_info()` | 146–149 | HTTP 429 | Sleep 60s, retry once |
| `get_trial_info()` | 154 | `requests.RequestException` | Returns `None`, WARNING |
| `get_trial_info()` | 158 | `json.JSONDecodeError` | Returns `None`, WARNING |
| `_parse_study()` | 217 | `KeyError`, `TypeError` | Returns `None`, WARNING |
| `_parse_study()` | 221 | `ValueError` | Returns `None`, WARNING |
| `search_by_acronym()` primary | 422 | `RequestException`, `JSONDecodeError` | Returns `None`, WARNING |
| `search_by_acronym()` fallback | 440 | Same | Returns `None`, DEBUG (deliberately lower) |

---

### 2.6 Entity Processors (`I_extraction/I01_entity_processors.py`)

**This file has ZERO try/except blocks.** All 6 processing methods — `process_diseases`, `process_genes`, `process_drugs`, `process_pharma`, `process_authors`, `process_citations` — have no error handling.

If any detector, normalizer, enricher, or deduplicator call fails, the exception propagates directly to `orchestrator.process_pdf()`, which also lacks per-stage error isolation for entity processing stages (stages 5–10).

**Impact**: A failure in disease PubTator enrichment cascades — it prevents all downstream entity processing (genes, drugs, pharma, authors, citations), feasibility extraction, care pathways, recommendations, and export.

---

### 2.7 Export Layer (`J_export/J01_export_handlers.py`)

| Method | Lines | Exception | Recovery |
|--------|-------|-----------|----------|
| `export_extracted_text()` | 305 | `Exception` | `print()` warning, continues |
| `render_figure_with_padding()` | 173 | `ImportError` | Returns `None` |
| `render_figure_with_padding()` | 276 | `Exception` | `print()` warning, returns `None`; **may leak `fitz.Document`** |
| **`export_results()`** | **308–422** | **NONE** | `json.dump()` failure propagates |
| `export_feasibility_results()` | 529 | `Exception` | `print()` warning on trial ID extraction, continues |
| **`export_feasibility_results()` file write** | **679–681** | **NONE** | Write failure propagates |
| `export_images()` per-image | 752, 817, 884, 974, 1047 | `Exception` | Increments error counter, `print()` warning, continues |
| `export_tables()` per-table | 1163 | `Exception` | `print()` warning, continues |

**Critical gap**: The main entity export methods (`export_results()`, entity JSON exports in `J01a_entity_exporters.py`) have **no try/except** on file writes. An I/O failure at export crashes `process_pdf()`, losing all extraction work for that document.

---

### 2.8 Orchestrator (`orchestrator.py`)

#### `process_pdf()` (lines 653–1140) — Per-Stage Error Isolation

| Stage | Lines | Try/Except? | On Failure |
|-------|-------|-------------|------------|
| 1. PDF Parsing | 689 | **No** | Crashes process_pdf() |
| 2. Candidate Generation | 698 | **No** | Crashes process_pdf() |
| 3. LLM Validation | 750–828 | **No** (internal try in H02) | H02 has partial isolation |
| 4. Normalization | — | **No** | Crashes process_pdf() |
| 5. Disease Detection | 882 | **No** | Crashes process_pdf() |
| 6. Gene Detection | 896 | **No** | Crashes process_pdf() |
| 7. Drug Detection | 910 | **No** | Crashes process_pdf() |
| 8. Pharma Detection | 928 | **No** | Crashes process_pdf() |
| 9. Author Detection | 940 | **No** | Crashes process_pdf() |
| 10. Citation Detection | 953 | **No** | Crashes process_pdf() |
| 11. Feasibility | 966 | **No** | Crashes process_pdf() |
| 12. Care Pathways | 1220–1257 | **Yes** (per-figure) | WARNING, continues |
| 13. Recommendations | 1259–1324 | **Yes** (per-table) | WARNING, continues |
| 14. Visual Extraction | 972–981 | **Yes** | WARNING, continues |
| 15. Document Metadata | 1150–1159 | **Yes** | WARNING, returns `None` |
| 16. Export | 1006 | **No** | Crashes process_pdf() |

**Only 4 of 16 stages have error isolation.** The other 12 stages propagate any exception directly to `process_folder()`'s batch-level catch.

#### `process_folder()` (lines 1716–1811) — Batch Safety Net

```python
try:
    all_results[pdf_path.name] = self.process_pdf(...)
except Exception as e:
    batch_printer.error(f"Processing failed: {e}")
    traceback.print_exc()
    all_results[pdf_path.name] = ExtractionResult()  # empty
```

- Full traceback printed (good)
- Failed documents get empty `ExtractionResult` (batch continues)
- `import traceback` is inline inside the except block (minor style issue)
- **No distinction** between "document failed" and "document had no results" in the final output

#### Configuration (lines 222–234)

```python
except Exception as e:
    print(f"[WARN] Failed to load config from {config_path}: {e}")
    return {}
```

- Too-broad `except Exception`
- Uses `print()` not logger (logger may not be configured yet)
- Returns empty dict — all pipeline behavior falls back to hardcoded defaults

---

### 2.9 Component Factory (`H_pipeline/H01_component_factory.py`)

| Method | Lines | Exception | Recovery |
|--------|-------|-----------|----------|
| `create_table_extractor()` | 145–146 | Checks `DOCLING_AVAILABLE` | Returns `None`, `print()` warning |
| `create_table_extractor()` | 154 | `ImportError` | Returns `None`, `print()` warning |
| `load_rare_disease_lookup()` | 573 | `Exception` (too broad) | Returns `{}`, `print()` warning |
| **All other `create_*` methods** | — | **NONE** | Propagates — crashes pipeline initialization |

**Impact**: If a lexicon file is missing or a model fails to load during `create_generators()`, `create_disease_detector()`, etc., the entire `Orchestrator.__init__()` fails. No documents are processed.

---

## 3. Error Propagation Traces

### 3.1 Claude API 429 During Abbreviation Validation

```
ClaudeClient._call_claude() → self.client.messages.create()
  ↓ raises AnthropicRateLimitError
  ↓ propagates uncaught through complete_json()
LLMEngine.verify_candidate() line 754 catches Exception
  → _handle_llm_error() → creates AMBIGUOUS entity
    flags=["llm_rate_limit_error"], confidence=0.0
  ↓ returned to caller
H02_abbreviation_pipeline → adds to results
orchestrator.process_pdf() → AMBIGUOUS entities excluded from validated output
```

**Outcome**: Abbreviation not validated, not exported as validated. Pipeline reports SUCCESS. The only indicator is the `llm_rate_limit_error` flag on the entity object (not visible in the final JSON export unless AMBIGUOUS entities are included).

**No retry** at the LLM engine level. The Anthropic SDK may retry internally, but if the SDK gives up, the entity is permanently degraded.

For **batch validation**: the batch falls back to per-candidate validation (one level of retry via degradation). If individual calls also hit 429, each becomes AMBIGUOUS independently.

---

### 3.2 PubTator3 Timeout During Disease Enrichment

```
DiseaseEnricher.enrich() → self.client.autocomplete(disease.preferred_label)
  ↓ HTTP request
PubTator3Client.autocomplete() line 134 catches Timeout
  → logger.warning("PubTator timeout for 'X'")
  → returns None
DiseaseEnricher.enrich() → if not results: return disease  (unmodified)
DiseaseEnricher.enrich_batch() → appends unenriched disease to results
EntityProcessor.process_diseases() → exports unenriched disease
orchestrator → exports to JSON
```

**Outcome**: Disease entity IS exported but **missing MeSH ID, aliases, and normalized name**. Pipeline reports SUCCESS. **No flag is set on the entity.** The absence of `pubtator_enriched` is indistinguishable from "enrichment was not needed."

**This is the most invisible failure mode in the pipeline.**

---

### 3.3 Corrupted PDF (Complete Parsing Failure)

```
orchestrator.process_pdf() → abbreviation_pipeline.parse_pdf()
  → PDFToDocGraphParser.parse() → fitz.open() or partition_pdf()
    ↓ raises fitz.FileDataError (or similar)
    ↓ NO try/except in parse(), parse_pdf()
    ↓ propagates to process_pdf() — NO try/except
    ↓ propagates to process_folder()
process_folder() line 1748 catches Exception
  → batch_printer.error() + traceback.print_exc()
  → all_results[name] = ExtractionResult()  (empty)
  → batch continues with next PDF
```

**Outcome**: Error printed with full traceback. Document gets empty result. Batch continues. The final "BATCH COMPLETE" message still says "PDFs processed: 10" even if 3 failed — potentially misleading.

---

### 3.4 VLM Call Failure for a Figure

```
VisionImageAnalyzer._call_vision_llm() → llm_client.complete_vision_json()
  ↓ raises Exception (API error, timeout, image too large)
  line 743 catches Exception
  → logger.warning("Vision LLM call failed: %s")
  → checks str(e) for size-related keywords
  → attempts OCR text fallback if available
  → returns None if no fallback
Calling method (e.g., analyze_flowchart) → if not response: return None
J01_export_handlers.export_images() → per-image try/except
  → analysis_stats["failed"] += 1
  → processing_errors.append({...})
  → figure still exported to JSON without vision_analysis data
```

**Outcome**: Figure appears in output but without structured analysis. The `processing_errors` array and `analysis_stats["failed"]` counter track the failure — this is **well-instrumented**.

---

### 3.5 Pydantic Validation Failure on LLM Response

```
LLMEngine.verify_candidate() → VerificationResult.model_validate(raw)
  ↓ raises pydantic.ValidationError
  line 757 catches ValidationError
  → returns _entity_ambiguous(
      reason="LLM response schema invalid: {ve}",
      flags=["llm_schema_error"],
      raw_llm=raw  # raw response preserved for debugging
    )
```

**Outcome**: Entity becomes AMBIGUOUS with 0.0 confidence. Raw LLM response preserved in entity for debugging. Not exported as validated. Same as 429 scenario.

---

### 3.6 scispaCy Model Fails to Load

```
RegexLexiconGenerator.__init__() line 258
  → spacy.load("en_core_sci_lg") raises OSError
  → except OSError: spacy.load("en_core_sci_sm") raises OSError
  → except OSError: print("Warning: Could not load scispacy model")
  → self.scispacy_nlp = None (implicitly — never assigned)
Generator initializes successfully (FlashText + regex still work)
orchestrator → candidate generation proceeds with reduced recall
```

**Outcome**: Pipeline reports SUCCESS. scispaCy NER is silently disabled. Entities that would only be found by biomedical NER (not in lexicons) are **permanently missed**. The only indicator is a `print()` line in console output — no flag in output JSON, no metric tracking, no error in the database.

---

### 3.7 Export JSON Write Failure

```
J01_export_handlers.export_results() → json.dump() or open() fails
  ↓ raises IOError/PermissionError
  ↓ NO try/except in export_results()
  ↓ propagates to orchestrator.process_pdf() — NO try/except around export stage
  ↓ propagates to process_folder()
  line 1748 catches Exception → empty ExtractionResult, batch continues
```

**Outcome**: ALL extraction work for the document is lost. The error IS visible (traceback printed), but partial results (e.g., valid abbreviations, diseases, genes already extracted) cannot be recovered.

Exception: `export_extracted_text()` DOES have its own try/except (line 305) — text export failures are non-fatal.

---

### 3.8 Malformed Config YAML

```
Orchestrator._load_config() → yaml.safe_load() raises yaml.YAMLError
  line 232 catches Exception
  → print("[WARN] Failed to load config...")
  → returns {}
```

**Impact cascade**:
- `paths` defaults: PDF dir = `"Pdfs"`, logs = `"corpus_log"` — may not point to correct location
- `model` defaults: `"claude-sonnet-4-20250514"` — correct but no Haiku tier routing
- `model_tiers` (loaded independently in D02): also fails → `_model_tier_cache = {}` → all calls route to Sonnet (3x cost increase)
- `heuristics` defaults: blacklists/whitelists may be empty → more false positives
- `extractors` defaults: all enabled
- Pipeline runs but with wrong paths, no cost optimization, and relaxed filtering

---

## 4. Silent Data Loss Scenarios

These scenarios result in the pipeline reporting **SUCCESS** while data has been silently lost or degraded:

| Scenario | Data Lost | Visibility | Severity |
|----------|-----------|------------|----------|
| PubTator timeout | Disease missing MeSH ID, aliases, normalized name | **INVISIBLE** — no flag, no metric | **Critical** |
| scispaCy load failure | All biomedical NER entities (diseases, genes not in lexicons) | Console `print()` only — not in output | **Critical** |
| scispaCy NER exception (line 553) | Same as above, but mid-processing | **FULLY SILENT** (`except: pass`) | **Critical** |
| UMLS KB lookup failure (line 497) | UMLS codes for detected entities | **FULLY SILENT** (`except: pass`) | Medium |
| Claude 429 rate limit | Abbreviation validation downgraded to AMBIGUOUS | Entity flag `llm_rate_limit_error` (not in exported JSON) | Medium |
| Pydantic validation failure | Same as above | Entity flag `llm_schema_error` | Low |
| VLM failure | Figure analysis data | `processing_errors` array + counter (well-tracked) | Low |
| Malformed config | No direct data loss but 3x cost increase, relaxed filtering | Console `[WARN]` only | Medium |
| DiskCache corruption | Cached API responses lost, causes re-fetch | DEBUG log (invisible) | Low |

---

## 5. Code Quality Issues

### 5.1 Overly Broad Exception Catches

| File | Line(s) | Pattern | Issue |
|------|---------|---------|-------|
| `D02_llm_engine.py` | 754 | `except (..., Exception) as e:` | Catches everything including bugs |
| `D02_llm_engine.py` | 991–993 | `except Exception as e:` | Same |
| `D02_llm_engine.py` | 1193 | `except (..., Exception) as e:` | Same |
| `orchestrator.py` | 232 | `except Exception as e:` | Hides config errors |
| `B01_pdf_to_docgraph.py` | 758 | `except Exception as e:` | Hides parsing bugs |
| `B28_docling_backend.py` | 124 | `except (..., Exception) as e:` | Hides converter bugs |
| `C04_strategy_flashtext.py` | 277 | `except Exception as e:` | Hides UMLS linker bugs |

### 5.2 Silent Exception Swallowing

| File | Line | Pattern | Impact |
|------|------|---------|--------|
| `Z01_api_client.py` | 412 | `except OSError: pass` (DiskCache.clear) | Low — cache cleanup |
| `D02_llm_engine.py` | 620–621 | `except JSONDecodeError: pass` | Low — JSON fallback chain |
| `D02_llm_engine.py` | 629–630 | `except JSONDecodeError: pass` | Low — JSON fallback chain |
| `C04_strategy_flashtext.py` | 497 | `except Exception: pass` (UMLS lookup) | **Medium** — lost enrichment |
| **`C04_strategy_flashtext.py`** | **553** | **`except Exception: pass`** (full scispaCy NER) | **HIGH** — entire NER silently fails |

### 5.3 Missing Error Handling

| File | Location | Risk | Impact |
|------|----------|------|--------|
| `B01_pdf_to_docgraph.py` | `parse()` method | PDF parsing crash kills document | High — caught at batch level |
| `B01_pdf_to_docgraph.py` | `_call_partition_pdf()` | Unstructured library failures | High — caught at batch level |
| `I01_entity_processors.py` | All 6 `process_*` methods | Any detector/enricher failure crashes pipeline | **Critical** — cascading failure |
| `orchestrator.py` | `process_pdf()` stages 1–11 | No per-stage isolation | High — one stage failure kills document |
| `E04_pubtator_enricher.py` | `enrich_batch()` | Single disease failure aborts batch | High — data loss |
| `H01_component_factory.py` | All `create_*` except table | Component init failure crashes pipeline | Medium — fails fast |
| `J01_export_handlers.py` | `export_results()` | File write failure loses all extraction | **Critical** — work lost |

### 5.4 Error Context Loss

| File | Line(s) | Issue |
|------|---------|-------|
| `Z01_api_client.py` | 134, 137, 143 | `str(e)` without traceback in retry logging |
| `B28_docling_backend.py` | 184 | `logger.error("...%s", e)` without traceback |
| `orchestrator.py` | 233 | `print(f"...{e}")` without traceback |
| Most WARNING logs | Various | Consistently use `str(e)` rather than `logger.exception()` |

Only `process_folder()` at line 1751 properly logs a full traceback via `traceback.print_exc()`.

### 5.5 Resource Leaks

| File | Resource | Issue |
|------|----------|-------|
| `Z01_api_client.py` | `requests.Session` in `BaseAPIClient` | Context manager exists but never used by callers |
| `E04_pubtator_enricher.py` | `PubTator3Client._session` | Session never closed |
| `E06_nct_enricher.py` | `ClinicalTrialsGovClient._session` | Session never closed |
| `E06_nct_enricher.py` | `TrialAcronymEnricher._session` | Session never closed |
| `J01_export_handlers.py` | `fitz.Document` in `render_figure_with_padding()` | May leak on exception between open and close |

### 5.6 Duplicate Exception Classes

`Z_utils/Z01_api_client.py` defines `CacheError`, `APIError`, and `RateLimitError` (lines 235–256) that shadow the identically-named classes in `A_core/A12_exceptions.py`. The Z01 versions are simpler (no context dict). Callers could import the wrong one.

### 5.7 Inconsistent Error Reporting

Errors are reported through 4 different mechanisms with no unified tracking:

| Mechanism | Used By | Queryable? | Preserved? |
|-----------|---------|-----------|------------|
| `logger.warning/error()` | API clients, LLM engine, Docling | In log files only | Until rotation |
| `print()` / `batch_printer` | Orchestrator, component factory, exports | In tee'd console log | Until deleted |
| Entity flags | LLM validation errors | In entity objects | Per-document JSON (if exported) |
| `processing_errors` array | Image export | In figures JSON | Per-document |
| SQLite `usage_stats.db` | Token usage only | Yes | Permanent |

No errors are persisted to the SQLite database.

---

## 6. Gaps vs. SOTA

| Capability | ESE Current | SOTA Best Practice | Status |
|------------|------------|-------------------|--------|
| **Retry with backoff** | 3 profiles, exponential + jitter | Standard pattern | **At parity** |
| **Document isolation** | Per-document try/except in batch | Standard pattern | **At parity** |
| **Schema validation** | Pydantic models + invariant checks | Standard pattern | **At parity** |
| **Provenance tracking** | Full provenance on every entity | Exceeds most pipelines | **Ahead** |
| **Rate limiting** | Thread-safe RateLimiter class | Standard pattern | **At parity** |
| **Graceful degradation** | Skip-and-continue per stage (partial) | Per-stage isolation for ALL stages | **Partial** (4/16 stages) |
| **Circuit breaker** | Not implemented | Open/half-open/closed pattern; stop calling degraded services | **Missing** |
| **Model fallback chain** | Not implemented | Sonnet → Haiku → local model → cached | **Missing** |
| **Dead letter queue** | Not implemented | Failed items queued for review/reprocessing | **Missing** |
| **Structured error tracking** | Console logs only | Errors in database with type, context, timestamps | **Missing** |
| **Checkpointing** | Not implemented | Save state per stage; resume from last | **Missing** |
| **Health checks** | Not implemented | Probe dependencies before batch start | **Missing** |
| **Chaos testing** | Not implemented | Inject failures in CI to validate recovery | **Missing** |
| **Idempotency** | Not implemented | Safe reprocessing without duplication | **Missing** |
| **Observability** | Timing bars + token tracker | OpenTelemetry traces, Prometheus metrics | **Gap** |
| **LLM-level retry** | Not implemented (relies on SDK) | Explicit retry with fallback model | **Missing** |
| **Enrichment failure tracking** | No flags on failed entities | Flag entities that failed enrichment | **Missing** |

### SOTA Patterns in Detail

#### Circuit Breaker

```
CLOSED (normal) → N failures in window → OPEN (fast-fail, return fallback)
OPEN → timeout expires → HALF-OPEN (probe with single call)
HALF-OPEN → probe succeeds → CLOSED
HALF-OPEN → probe fails → OPEN
```

**ESE impact**: Would prevent accumulated 60s+ retry delays when Claude API is degraded. A 10-minute outage currently causes ~10 calls × 60s sleep = 10 minutes of wasted time. With a circuit breaker, the second failure would trip the circuit and all subsequent calls would immediately return AMBIGUOUS (~0s).

#### Model Fallback Chain

```
feasibility_extraction: Sonnet 4 → Haiku 4.5 → cached/heuristic fallback
abbreviation_validation: Haiku 4.5 → local encoder → skip validation
```

**ESE impact**: Eliminates single-provider dependency. During Sonnet outages, feasibility extraction degrades to Haiku (lower quality but non-zero) instead of failing entirely.

#### Checkpointing

Save intermediate state after each stage:
```python
checkpoint = {
    "document_id": doc_id,
    "stage": 5,  # disease detection
    "parsed_data": doc_graph,
    "candidates": candidates,
    "validated": validated_entities,
    "timestamp": now()
}
```

**ESE impact**: A 56-minute protocol document that fails during export (stage 16/16) currently requires full reprocessing. Checkpointing would recover in seconds.

#### Structured Error Tracking

Add `pipeline_errors` table to existing SQLite database:

```sql
CREATE TABLE pipeline_errors (
    id INTEGER PRIMARY KEY,
    document_id TEXT,
    stage TEXT,
    error_type TEXT,
    error_message TEXT,
    call_type TEXT,
    recovery_action TEXT,  -- retried, skipped, fallback, failed, ambiguous
    data_loss_risk TEXT,   -- none, partial, full
    logged_at TIMESTAMP
);
```

**ESE impact**: Enables failure rate dashboards, identifies flaky stages, detects silent data loss patterns, and informs model tier routing decisions.

#### Pre-flight Health Checks

```python
def preflight_checks(self) -> List[str]:
    errors = []
    # Test Claude API
    try:
        self.claude.complete_json("test", call_type="health_check")
    except Exception as e:
        errors.append(f"Claude API unavailable: {e}")
    # Test PubTator
    try:
        self.pubtator.autocomplete("aspirin", "chemical")
    except Exception as e:
        errors.append(f"PubTator3 unavailable: {e}")
    # Check disk space
    if shutil.disk_usage(output_dir).free < 100_000_000:
        errors.append("Less than 100MB disk space available")
    return errors
```

**ESE impact**: Fail fast before spending 84–1005s on PDF parsing. Currently, API issues are discovered only at the first LLM validation call.

---

## 7. Recommendations

### Priority 1 — Fix Critical Silent Failures

1. **Add logging to scispaCy NER catch-all** (`C04_strategy_flashtext.py:553`): Replace `except Exception: pass` with `except Exception as e: logger.error("scispaCy NER failed: %s", e, exc_info=True)`. This is the highest-severity silent swallow in the codebase.

2. **Flag enrichment failures on entities** (`E04_pubtator_enricher.py`): When PubTator returns `None`, set a flag on the entity (e.g., `enrichment_failed=True`) so downstream consumers can distinguish "not enriched" from "enrichment unnecessary."

3. **Add per-stage try/except in `process_pdf()`**: Wrap stages 1–11 with individual try/except blocks so a failure in disease detection doesn't prevent gene/drug/author extraction and export.

### Priority 2 — Improve Error Visibility

4. **Structured error tracking in SQLite**: Add a `pipeline_errors` table to the existing `usage_stats.db`. Log every caught exception with document_id, stage, error_type, and recovery_action.

5. **Replace `print()` with `logger`** in orchestrator, component factory, and export handlers. Ensure consistent log levels (ERROR for data loss, WARNING for degradation, INFO for fallbacks).

6. **Add `exc_info=True`** to WARNING/ERROR log calls to preserve tracebacks.

### Priority 3 — Production Resilience

7. **Pre-flight health checks**: Probe Claude API, PubTator3, disk space before starting batch.

8. **LLM-level retry**: Add 1 retry with backoff in `verify_candidate()` before converting to AMBIGUOUS.

9. **Circuit breaker for Claude API**: Track failure rate; stop calling after N failures in a window.

10. **Model fallback chain**: If primary model fails, try secondary model before defaulting to AMBIGUOUS.

### Priority 4 — Architectural

11. **Adopt the custom exception hierarchy**: Replace generic `Exception` catches with specific `ESEPipelineError` subclasses. Remove duplicate exception classes from `Z01_api_client.py`.

12. **Checkpointing**: Save intermediate state per stage; resume from last successful stage on failure.

13. **Add `enrich_batch()` error isolation**: Wrap per-entity enrichment calls so a single failure doesn't abort the batch.

14. **Close HTTP sessions**: Use context managers for all API client instantiations, or add `__del__` cleanup.

---

## References

- [Retries, Fallbacks, and Circuit Breakers in LLM Apps](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [Building Bulletproof LLM Applications: SRE Best Practices (Google Cloud, 2025)](https://medium.com/google-cloud/building-bulletproof-llm-applications-a-guide-to-applying-sre-best-practices-1564b72fd22e)
- [Error Handling Best Practices for Production LLM Applications](https://markaicode.com/llm-error-handling-production-guide/)
- [Circuit Breaker for LLM with Retry and Backoff — Anthropic API Example](https://medium.com/@spacholski99/circuit-breaker-for-llm-with-retry-and-backoff-anthropic-api-example-typescript-1f99a0a0cf87)
- [The Spectrum of Failure Models in AI Agentic Systems](https://cenrax.substack.com/p/the-spectrum-of-failure-models-in)
- [Deploying NLP Pipelines in Production (2025)](https://medium.com/@akshaybhargavkulakarni50/natural-language-processing-nlp-has-advanced-rapidly-over-the-last-few-years-moving-from-b0caa733a3b6)
- [Fault-Tolerant Data Pipeline Design](https://www.linkedin.com/advice/3/what-techniques-can-you-use-design-fault-tolerant-i5ref)
- [Clinical De-Identification Pipeline Speed–Accuracy Trade-offs (2025)](https://www.johnsnowlabs.com/clinical-de-identification-at-scale-pipeline-design-and-speed-accuracy-trade-offs-across-infrastructures/)
