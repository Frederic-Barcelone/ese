# Z_utils Refactoring Design

**Date:** 2026-02-02
**Goal:** Clean structure - fix numbering, consolidate overlapping modules, remove duplication

## Current State

| File | Lines | Purpose | Used By |
|------|-------|---------|---------|
| Z01_api_client.py | 627 | API client, cache, rate limiter | E04, E06 enrichers |
| Z02_text_helpers.py | 283 | Abbreviation text helpers | I01, H02 |
| Z03_text_normalization.py | 166 | Text normalization | **Unused - duplicated in C_generators** |
| Z04_image_utils.py | 449 | Image utils, OCR | D02, J01 |
| Z05_path_utils.py | 49 | Path resolution | orchestrator, H01, K16 |
| Z09_download_gene_lexicon.py | 353 | Gene lexicon script | standalone |
| Z10_download_lexicons.py | 431 | Lexicon download scripts | standalone |
| Z10_usage_tracker.py | 430 | Usage tracking DB | orchestrator |
| Z11_console_output.py | 203 | Console printing | orchestrator, F03, H02 |

### Problems Identified

1. **Numbering gap:** Z06, Z07, Z08 missing
2. **Duplicate Z10 prefix:** Both `Z10_download_lexicons.py` and `Z10_usage_tracker.py`
3. **DRY violation - download_file():** Duplicated in Z09 and Z10_download_lexicons
4. **DRY violation - text normalization:** Z03 functions duplicated in C03, C05, C20
5. **Hardcoded paths:** Download scripts have hardcoded output directory

## Target State

| File | Content | Action |
|------|---------|--------|
| Z01_api_client.py | API client, cache, rate limiter | Keep unchanged |
| Z02_text_helpers.py | Abbreviation text helpers | Keep unchanged |
| Z03_text_normalization.py | Text normalization | Keep - fix C_generators to use it |
| Z04_image_utils.py | Image utils, OCR | Keep unchanged |
| Z05_path_utils.py | Path resolution | Keep unchanged |
| Z06_usage_tracker.py | Usage tracking DB | **Rename** from Z10_usage_tracker |
| Z07_console_output.py | Console printing | **Rename** from Z11_console_output |
| Z08_download_utils.py | Shared download helpers | **NEW** - extract shared code |
| Z09_download_gene_lexicon.py | Gene lexicon script | Refactor to use Z08 |
| Z10_download_lexicons.py | Other lexicons script | Refactor to use Z08 |

## Implementation Steps

### Phase 1: Rename files to fix numbering

1. **Rename Z10_usage_tracker.py → Z06_usage_tracker.py**
   - Update imports in: `orchestrator.py`

2. **Rename Z11_console_output.py → Z07_console_output.py**
   - Update imports in: `orchestrator.py`, `F03_evaluation_runner.py`, `H02_abbreviation_pipeline.py`

### Phase 2: Create shared download utilities

3. **Create Z08_download_utils.py** with:
   ```python
   def download_file(url: str, dest: Path, timeout: int = 60) -> bool
   def get_default_output_dir() -> Path
   ```

4. **Refactor Z09_download_gene_lexicon.py**
   - Import `download_file` from Z08
   - Remove duplicate function
   - Make OUTPUT_DIR configurable via function parameter

5. **Refactor Z10_download_lexicons.py**
   - Import `download_file` from Z08
   - Remove duplicate function
   - Make OUTPUT_DIR configurable via function parameter

### Phase 3: Fix C_generators DRY violations

6. **Update C03_strategy_layout.py**
   - Remove local `_dehyphenate_long_form` function
   - Import from `Z_utils.Z03_text_normalization`

7. **Update C05_strategy_glossary.py**
   - Remove local `_dehyphenate_long_form` function
   - Import from `Z_utils.Z03_text_normalization`

8. **Update C20_abbrev_patterns.py**
   - Remove local `_dehyphenate_long_form`, `_normalize_long_form`, `_clean_ws`, `_truncate_at_clause_breaks`
   - Import from `Z_utils.Z03_text_normalization`
   - Keep internal aliases with underscore prefix for backwards compatibility in tests

### Phase 4: Update __init__.py and verify

9. **Update Z_utils/__init__.py** with new structure

10. **Run verification:**
    ```bash
    python -m pytest corpus_metadata/K_tests/ -v
    mypy corpus_metadata
    ruff check corpus_metadata
    ```

## Files Modified

### Z_utils (renamed/new)
- `Z06_usage_tracker.py` (renamed from Z10)
- `Z07_console_output.py` (renamed from Z11)
- `Z08_download_utils.py` (new)
- `Z09_download_gene_lexicon.py` (refactored)
- `Z10_download_lexicons.py` (refactored)
- `__init__.py` (updated)

### Import updates
- `orchestrator.py`
- `F_evaluation/F03_evaluation_runner.py`
- `H_pipeline/H02_abbreviation_pipeline.py`

### DRY fixes in C_generators
- `C03_strategy_layout.py`
- `C05_strategy_glossary.py`
- `C20_abbrev_patterns.py`

## Risk Assessment

- **Low risk:** File renames with import updates are mechanical
- **Medium risk:** C_generators refactoring - functions may have subtle differences
  - Mitigation: Compare implementations before removing, run existing tests

## Verification Checklist

- [ ] All tests pass: `pytest K_tests/ -v`
- [ ] Type checking passes: `mypy corpus_metadata`
- [ ] Linting passes: `ruff check corpus_metadata`
- [ ] No duplicate `download_file` functions
- [ ] No duplicate text normalization functions in C_generators
- [ ] Z_utils numbering is sequential (Z01-Z10)
