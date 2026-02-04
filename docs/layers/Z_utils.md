# Z_utils -- Utilities

## Purpose

Layer Z provides shared infrastructure used across all pipeline layers: API client base classes, text processing helpers, path resolution, usage tracking, console output formatting, lexicon downloading, and entity manipulation utilities.

## Modules

### Z01_api_client.py

Shared API client base class with disk caching, rate limiting, and retry logic.

**Classes:**

- `DiskCache` -- Thread-safe TTL-based disk cache. Stores JSON-serialized responses as files with hash-based keys.
  - `make_key(prefix, *parts)` -- Generate cache key from components
  - `get(key)` -- Retrieve cached value (None if expired or missing)
  - `set(key, value)` -- Store value with timestamp
- `RateLimiter` -- Token-bucket rate limiter.
  - `wait()` -- Block until a request slot is available
- `BaseAPIClient(ABC)` -- Abstract base class for HTTP API clients.
  - Constructor: `config`, `service_name`, `default_rate_limit`, `default_cache_ttl_hours`
  - `_request(method, path, **kwargs)` -- Make HTTP request with error handling
  - Subclasses: `PubTator3Client` (direct subclass). Note: `NCTEnricher` extends `BaseEnricher` (not `BaseAPIClient`), and `CitationValidator` uses API clients internally but does not extend `BaseAPIClient`.

**Usage pattern:**

```python
class MyClient(BaseAPIClient):
    def __init__(self, config=None):
        super().__init__(config=config, service_name="myservice",
                        default_rate_limit=5, default_cache_ttl_hours=24)

    def fetch(self, query):
        key = self.cache.make_key("query", query)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.rate_limiter.wait()
        result = self._request("GET", "/endpoint", params={"q": query})
        self.cache.set(key, result)
        return result
```

### Z02_text_helpers.py

Text processing helpers for the extraction pipeline.

**Functions:**

| Function | Purpose |
|----------|---------|
| `extract_context_snippet(full_text, match_start, match_end, window=100)` | Extract text around a match position |
| `normalize_lf_for_dedup(lf)` | Normalize long form for deduplication: lowercased, dashes standardized, whitespace collapsed |
| `has_numeric_evidence(context, short_form)` | Check if context contains numeric evidence for statistical abbreviations |
| `is_valid_sf_form(short_form, context, allowed_2letter, allowed_mixed)` | Validate short form structure (length, case, allowed exceptions) |
| `score_lf_quality(candidate, full_text, full_text_lower)` | Score long form quality for candidate ranking |

### Z03_text_normalization.py

Text normalization for PDF-extracted content. Used across multiple generators.

**Functions:**

| Function | Purpose |
|----------|---------|
| `clean_whitespace(s)` | Collapse all whitespace to single spaces and strip |
| `dehyphenate_long_form(lf)` | Remove line-break hyphens from long forms. Preserves intentional hyphens in compound words (e.g., "anti-inflammatory"). Fixes PDF artifacts like "gastroin-testinal" to "gastrointestinal" |

### Z04_image_utils.py

Image utilities for Vision API integration.

**Functions:**

| Function | Purpose |
|----------|---------|
| `get_image_size_bytes(base64_str)` | Calculate decoded image size from base64 string |
| `is_image_oversized(base64_str)` | Check if image exceeds Vision API 5MB limit |
| `compress_image_for_vision(base64_str, max_size_bytes, quality, max_dimension)` | Compress image for Vision API (JPEG quality, dimension limits) |
| `extract_ocr_text_from_base64(base64_str, lang, config)` | Extract text from image via Tesseract OCR |

**Constants:** `PYTESSERACT_AVAILABLE` -- Boolean flag indicating Tesseract availability.

### Z05_path_utils.py

Base path resolution for the project.

**Functions:**

- `get_base_path(config=None)` -- Resolves project root from `CORPUS_BASE_PATH` environment variable, or auto-detects by walking up from `corpus_metadata/` directory.

### Z06_usage_tracker.py

SQLite-based tracking of lexicon, data source, and LLM token usage during extraction.

**Class: `UsageTracker`**

**Tables managed:**

| Table | Purpose |
|-------|---------|
| `documents` | Document processing status (started_at, finished_at, status) |
| `lexicon_usage` | Per-document lexicon matches, candidates, validated counts |
| `datasource_usage` | Per-document API query counts, results, errors |
| `llm_usage` | Per-call LLM token usage: model, call_type, input/output/cache tokens, estimated cost |

**Lexicon & Data Source Methods:**

- `start_document(document_id, filename)` -- Register document as processing
- `finish_document(document_id, status)` -- Mark document complete/failed
- `log_lexicon_usage(document_id, lexicon_name, matches, candidates, validated)` -- Record lexicon contribution
- `log_datasource_usage(document_id, datasource_name, queries, results, errors)` -- Record API call statistics
- `get_lexicon_stats()` -- Aggregated stats by lexicon (total matches, candidates, validated, avg per doc)
- `get_datasource_stats()` -- Aggregated stats by data source
- `get_unused_lexicons(min_documents)` -- Find lexicons with zero matches across N+ documents
- `get_document_usage(document_id)` -- Full usage details for a specific document

**LLM Usage Methods:**

- `log_llm_usage(document_id, model, call_type, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, estimated_cost_usd)` -- Record a single LLM API call
- `log_llm_usage_batch(document_id, records)` -- Batch insert from `LLMUsageTracker.records` (from `D02_llm_engine`)
- `get_llm_stats()` -- Aggregated LLM stats by model (total calls, tokens, cost, documents)
- `get_llm_stats_by_call_type()` -- Aggregated LLM stats by call_type and model
- `print_summary()` -- Print full summary to console (documents, LLM tokens, lexicons, data sources)

### Z07_console_output.py

Rich console output formatting with color support.

**Classes/Functions:**

- `Colors` / `C` -- ANSI color constants. `Colors.supports_color()` for terminal detection.
- `get_printer(total_steps)` -- Returns a `StepPrinter` with styled output:
  - `step(message, step_num)` -- Print numbered step header
  - `detail(message)` -- Print indented detail line
  - `detail_highlight(label, value)` -- Print label: value with highlighting
  - `time(elapsed)` -- Print timing information
  - `skip(step_name, reason)` -- Print skip notification

### Z08_download_utils.py

HTTP download utilities for lexicon files.

**Features:** HTTP client with progress bars for downloading lexicon files from external sources.

### Z09_download_gene_lexicon.py

Gene lexicon builder from HGNC REST API.

**Purpose:** Downloads HGNC gene data and builds a gene synonym lexicon file for FlashText matching. Produces JSON with HGNC symbols, aliases, Entrez IDs, Ensembl IDs, and disease associations.

### Z10_download_lexicons.py

Batch lexicon download script.

**Purpose:** Downloads all required lexicons from external sources (MONDO, ChEMBL, RxNorm, Orphanet, Meta-Inventory, etc.) and saves them in the configured dictionaries path.

### Z11_entity_helpers.py

Shared entity creation helpers used by both `AbbreviationPipeline` (H02) and `EntityProcessor` (I01).

**Functions:**

- `create_entity_from_candidate(candidate, status, confidence, reason, flags, raw_response, long_form_override)` -- Convert a `Candidate` to `ExtractedEntity`. Preserves provenance, evidence spans, and field type from the candidate.
- `create_entity_from_search(doc_id, full_text, match, long_form, field_type, confidence, flags, rule_version, lexicon_source, pipeline_version, run_id)` -- Create `ExtractedEntity` from a regex `re.Match` object. Builds context snippet and evidence span from match position.

## Usage Patterns

```python
# API client
from Z_utils.Z01_api_client import BaseAPIClient

# Text processing
from Z_utils.Z02_text_helpers import extract_context_snippet, normalize_lf_for_dedup
from Z_utils.Z03_text_normalization import clean_whitespace, dehyphenate_long_form

# Path resolution
from Z_utils.Z05_path_utils import get_base_path
base = get_base_path(config)

# Console output
from Z_utils.Z07_console_output import get_printer
printer = get_printer(total_steps=12)
printer.step("Processing diseases...", step_num=5)

# Entity creation
from Z_utils.Z11_entity_helpers import create_entity_from_candidate
entity = create_entity_from_candidate(
    candidate, ValidationStatus.VALIDATED, 0.9,
    "Auto-approved stats", ["auto_approved_stats"], {"auto": "stats_whitelist"},
)
```
