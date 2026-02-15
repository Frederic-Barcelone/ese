# Code Quality Audit Report

**Last updated:** 2026-02-15
**Scope:** `corpus_metadata/` — ~154 modules, ~75K lines
**Baseline:** 1758 tests, all passing
**Current:** 1756 tests, all passing (2 removed with deleted shim modules)

---

## Summary

Two rounds of refactoring were applied. Round 1 addressed interface consistency, initial DRY consolidation, monster function decomposition, docstrings, and type:ignore cleanup. Round 2 completed context extraction DRY, further orchestrator decomposition, removed backward-compatibility shims, and improved error handling.

| Category | Before | After | Round |
|----------|--------|-------|-------|
| Context extraction DRY | 6 private wrappers | 0 (all use Z02_text_helpers) | R1+R2 |
| Monster functions (>100 lines) | 4 in orchestrator | 0 | R1+R2 |
| Backward-compat re-export shims | 3 (B02, B18, D04) | 0 (deleted) | R2 |
| Interface non-compliance | 5 detector classes | 0 | R1 |
| `type: ignore` (bare) | 12 | 0 | R1 |
| Silent exception handling | 2 in C04 | 0 (logged) | R2 |
| Error handling inconsistency | 5 print-based | 2 remaining | R2 |
| Module docstrings | 100% coverage | 100% | — |

---

## Round 2 Changes (2026-02-15)

### 1. Context Extraction Consolidation

Deleted 5 private wrapper methods across C_generators, replacing all call sites with canonical functions from `Z_utils.Z02_text_helpers`:

| File | Deleted Method | Replacement |
|------|---------------|-------------|
| C04_strategy_flashtext.py | `_make_context()` | `extract_context_snippet()` + `.replace("\n", " ").strip()` |
| C06_strategy_disease.py | `_make_context()` | `extract_context_snippet()` + `.replace("\n", " ").strip()` |
| C07_strategy_drug.py | `_extract_context()` | `extract_context_window()` (direct call) |
| C08_strategy_feasibility.py | `_get_context()` | `extract_context_window()` (direct call) |
| C16_strategy_gene.py | `_extract_context()` | `extract_context_window()` (direct call) |

C20_abbrev_patterns.py `_context_window()` was refactored to call `extract_context_snippet()` internally while keeping its public name (used by C01).

**Canonical functions:**
- `extract_context_snippet(text, start, end, window)` — window extends full distance each side
- `extract_context_window(text, start, end, window)` — window is total size, split half each side

### 2. Orchestrator Decomposition

Extracted `_EXTRACTION_PRESETS` to module-level constant (was nested inside `_load_extraction_settings`). Split `_load_extraction_settings` (~126 lines) into:
- `_apply_entity_toggles()` — preset or individual entity enable/disable
- `_load_processing_options()` — processing options from config

Net result: 52 insertions, 117 deletions across orchestrator.py.

### 3. Backward-Compatibility Shim Removal

Deleted 3 re-export modules after migrating all consumers to canonical imports:

| Deleted File | Canonical Location | Files Updated |
|-------------|-------------------|---------------|
| `B_parsing/B02_doc_graph.py` | `A_core/A23_doc_graph_models` | 0 (no consumers) |
| `B_parsing/B18_layout_models.py` | `A_core/A24_layout_models` | 7 (6 tests + B20) |
| `D_validation/D04_quote_verifier.py` | `Z_utils/Z14_quote_verifier` | 2 (K42, K64) |

Total: 3 files deleted, 12 files updated (37 insertions, 96 deletions).

### 4. Error Handling Improvements

- Added `logging` import and logger to `C04_strategy_flashtext.py`
- Replaced 2 silent `except: pass` blocks with `logger.debug()`/`logger.warning()` (UMLS KB lookup, scispacy NER)
- Added try/except wrapper around `process_pdf()` core to ensure `usage_tracker.close()` cleanup
- Added structured `logger.error()` for visual extraction and metadata extraction failures

### 5. Test Updates

- Updated `K35_test_gene_detector.py` `TestExtractContext` to test canonical `extract_context_window()` directly (deleted method no longer exists)
- Removed `D04_quote_verifier` from `K42_test_d_e_f_imports.py` parametrized module list

---

## Round 1 Changes (2026-02-15, earlier session)

### Interface Consistency
Added `extract()` alias methods to 5 detector classes (C07, C13, C14, C16, C18).

### Initial DRY: Context Window
Extracted `extract_context_window()` to `Z_utils/Z02_text_helpers.py`.

### Monster Function Decomposition
| Function | File | Before | After |
|----------|------|--------|-------|
| `process_pdf()` | `orchestrator.py` | 537 lines | ~50 lines + 8 sub-methods |
| `extract()` | `C01_strategy_abbrev.py` | 399 lines | ~35 lines + 5 strategy methods |
| `extract()` | `C04_strategy_flashtext.py` | 322 lines | ~100 lines + 2 extracted methods |
| `__init__()` | `E02_disambiguator.py` | 289 lines | ~30 lines (YAML-loaded) |
| `parse()` | `B01_pdf_to_docgraph.py` | 293 lines | ~40 lines + 3 sub-methods |

### Docstrings
Added/enriched docstrings for orchestrator, C01, C04, C06, C07.

### `type: ignore` Cleanup
Removed 1 bare ignore, narrowed 12 bare comments to specific codes.

---

## Verification Results

```
pytest K_tests/ -v             → 1756 passed in ~56s
ruff check .                   → All checks passed
mypy .                         → 11 pre-existing errors (none introduced)
```

## What Was NOT Changed

- No public method signatures changed
- No new pip dependencies
- No performance optimizations
- No type hints added to untouched functions
- F03 evaluation runner left as-is (test/eval tool, not production)
