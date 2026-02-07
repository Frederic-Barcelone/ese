# Layer Z: Utilities

## Purpose

Shared infrastructure across all layers: API client base, text processing, path resolution, usage tracking, console output, lexicon downloading, entity helpers. 11 modules.

---

## Modules

### Z01_api_client.py

API client base with disk caching, rate limiting, and retry.

- `DiskCache` -- Thread-safe TTL-based JSON file cache with hash-based keys.
- `RateLimiter` -- Token-bucket rate limiter.
- `BaseAPIClient(ABC)` -- Abstract HTTP client. Subclassed by `PubTator3Client` in `E04_pubtator_enricher.py`.

### Z02_text_helpers.py

| Function | Purpose |
|----------|---------|
| `extract_context_snippet(full_text, start, end, window)` | Text around a match position |
| `normalize_lf_for_dedup(lf)` | Lowercase, dash-standardize, whitespace-collapse |
| `has_numeric_evidence(context, sf)` | Check for numeric evidence of statistical abbreviations |
| `is_valid_sf_form(sf, context, ...)` | Validate short form structure |
| `score_lf_quality(candidate, ...)` | Score long form quality for ranking |

### Z03_text_normalization.py

| Function | Purpose |
|----------|---------|
| `clean_whitespace(s)` | Collapse whitespace, strip |
| `dehyphenate_long_form(lf)` | Remove PDF line-break hyphens, preserve intentional hyphens |

### Z04_image_utils.py

Vision API image utilities: `get_image_size_bytes()`, `is_image_oversized()`, `compress_image_for_vision()`, `extract_ocr_text_from_base64()`.

### Z05_path_utils.py

`get_base_path(config)` -- Resolves project root from `CORPUS_BASE_PATH` or auto-detects from `corpus_metadata/`.

### Z06_usage_tracker.py

`UsageTracker`: SQLite-based tracking.

| Table | Purpose |
|-------|---------|
| `documents` | Processing status |
| `lexicon_usage` | Per-document lexicon matches |
| `datasource_usage` | API query counts |
| `llm_usage` | Per-call LLM tokens, model, call_type, cost |

### Z07_console_output.py

`get_printer(total_steps)` returns `StepPrinter` with `step()`, `detail()`, `detail_highlight()`, `time()`, `skip()`.

### Z08-Z10: Download Utilities

- `Z08_download_utils.py` -- HTTP download with progress bars.
- `Z09_download_gene_lexicon.py` -- Build gene lexicon from HGNC REST API.
- `Z10_download_lexicons.py` -- Batch download all lexicons (MONDO, ChEMBL, RxNorm, etc.).

### Z11_entity_helpers.py

- `create_entity_from_candidate(candidate, status, confidence, ...)` -- Convert `Candidate` to `ExtractedEntity`.
- `create_entity_from_search(doc_id, full_text, match, ...)` -- Create entity from regex match.

---

## Usage

```python
from Z_utils.Z05_path_utils import get_base_path
from Z_utils.Z07_console_output import get_printer
from Z_utils.Z11_entity_helpers import create_entity_from_candidate

base = get_base_path(config)
printer = get_printer(total_steps=12)
printer.step("Processing diseases...", step_num=5)
```
