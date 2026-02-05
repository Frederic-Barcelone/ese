# ESE Pipeline v0.8 — Testing Strategy Analysis

> **Date**: February 4, 2026
> **Based on**: Exhaustive analysis of 60 test files (16,073 lines, 1,266 test functions), 13 key modules, and SOTA research

---

## Table of Contents

1. [Test Suite Overview](#1-test-suite-overview)
2. [Layer-by-Layer Coverage Inventory](#2-layer-by-layer-coverage-inventory)
3. [Test Classification & Patterns](#3-test-classification--patterns)
4. [Coverage Gap Analysis](#4-coverage-gap-analysis)
5. [Test Quality Assessment](#5-test-quality-assessment)
6. [Gold Standard Evaluation Framework](#6-gold-standard-evaluation-framework)
7. [Infrastructure & CI/CD](#7-infrastructure--cicd)
8. [Gaps vs. SOTA](#8-gaps-vs-sota)
9. [Recommendations](#9-recommendations)

---

## 1. Test Suite Overview

### 1.1 Inventory Summary

| Metric | Value |
|--------|-------|
| Total test files | 60 (59 in K_tests/ + conftest.py) |
| Total lines of test code | 16,073 |
| Total test functions | 1,266 |
| Total production files | ~140 |
| Files with behavioral tests | 54 (39%) |
| Files with import-only tests | 44 (31%) |
| Files with zero coverage | ~40 (29%) |
| Test-to-production line ratio | ~0.35:1 |
| Pytest markers defined | 3 (slow, integration, requires_gpu) |
| Pytest markers actually used | 0 |
| Parametrized test files | 6 |
| Stub/placeholder tests (pass body) | 15 |
| CI/CD pipelines | 0 |

### 1.2 Test Distribution by Layer

| Layer | Test Files | Test Functions | Lines | Top-Tested Module |
|-------|-----------|---------------|-------|-------------------|
| A_core (Domain Models) | 11 | 300 | 3,398 | A20 unicode_utils (47 tests) |
| B_parsing (PDF Parsing) | 13 | 248 | 3,654 | B15 caption_extractor (37 tests) |
| C_generators (Extraction) | 9 | 271 | 2,399 | C27 feasibility_patterns (44 tests) |
| D_validation (LLM) | 5 | 99 | 1,029 | D04 quote_verifier (33 tests) |
| E_normalization (Enrichment) | 4 | 67 | 1,091 | E11 span_deduplicator (20 tests) |
| F_evaluation (Scoring) | 2 | 43 | 658 | F02 scorer (25 tests) |
| G_config (Configuration) | 1 | 35 | 249 | G01 config_keys (35 tests) |
| H_pipeline (Orchestration) | 2 | 36 | 576 | H01 component_factory (19 tests) |
| I_extraction (Processors) | 2 | 24 | 328 | I01 entity_processors (14 tests) |
| J_export (Output) | 2 | 24 | 379 | J02 visual_export (15 tests) |
| Z_utils (Utilities) | 4 | 112 | 808 | Z03 text_normalization (48 tests) |
| Cross-cutting | 5 | 61 | 1,504 | A14 extraction_result (26 tests) |

### 1.3 Test Infrastructure

**conftest.py** (357 lines) provides:
- Shared configuration fixtures (`test_config`, `temp_cache_dir`, `cache_enabled_config`)
- Mock API response fixtures (PubTator3 disease/chemical, ClinicalTrials.gov NCT metadata)
- Mock client fixtures (patched PubTator3Client, ClinicalTrialsClient)
- Sample entity factories (`sample_disease_dict`, `sample_drug_dict`, `sample_genetic_text`)
- HTTP mock fixtures (patched `requests.get`, `requests.Session`)
- Utility fixtures (`assert_no_warnings`, `capture_logs`)
- Three registered markers (`slow`, `integration`, `requires_gpu`) — none used by any test

**pytest.ini** configuration:
- Two naming conventions: `test_*.py` and `K*.py`
- `--strict-markers` enforced
- `--tb=short` for concise tracebacks
- Deprecation warnings globally suppressed
- No pytest plugins installed (no pytest-cov, pytest-xdist, pytest-mock, pytest-timeout)

---

## 2. Layer-by-Layer Coverage Inventory

### 2.1 Files with Dedicated Behavioral Tests

54 production files have dedicated test files with functional assertions that exercise logic, validate outputs, and test edge cases. These are grouped below by layer.

**A_core** — 14 of 25 files tested (56%)

| Production File | Test File | Tests | Key Coverage |
|----------------|-----------|-------|-------------|
| A03_provenance.py | K51 | 32 | Hashing, fingerprinting, run ID generation |
| A05_disease_models.py | K02 | 18 | DiseaseProvenanceMetadata migration |
| A10_author_models.py | K52 | 15 | AuthorCandidate, ExtractedAuthor |
| A11_citation_models.py | K53 | 18 | CitationCandidate, ExtractedCitation |
| A12_exceptions.py | K04 | 19 | Full exception hierarchy |
| A13_ner_models.py | K06 | 29 | NEREntity, NERResult |
| A13_visual_models.py | K07, K22 | 50 | PageLocation, VisualReference, TableStructure |
| A14_extraction_result.py | K05 | 26 | Cross-layer integration (A14+B06+H04) |
| A15_domain_profile.py | K55 | 32 | DomainProfile, YAML loading |
| A16_pipeline_metrics.py | K56 | 33 | Metrics invariants |
| A20_unicode_utils.py | K54 | 47 | All dash variants, mojibake, combined normalization |
| A21_clinical_criteria.py | K57 | 31 | LabCriterion operators, SeverityGrade |

**B_parsing** — 13 of 32 files tested (41%)

| Production File | Test File | Tests | Key Coverage |
|----------------|-----------|-------|-------------|
| B12_visual_pipeline.py | K13 | 7 | Detection mode integration |
| B13_visual_detector.py | K18 | 13 | DetectorConfig, bbox overlap |
| B14_visual_renderer.py | K20 | 18 | Coordinate conversion, adaptive DPI |
| B15_caption_extractor.py | K17 | 37 | Caption patterns, geometry helpers |
| B16_triage.py | K21 | 34 | Visual triage logic, batch triage |
| B18_layout_models.py | K12 | 24 | LayoutPattern, VisualZone, PageLayout |
| B19_layout_analyzer.py | K10 | 15 | Layout response parsing |
| B20_zone_expander.py | K14 | 17 | Column boundaries, zone expansion |
| B21_filename_generator.py | K09 | 16 | Filename generation |
| B28_docling_backend.py | K08 | 9 | Docling availability, table conversion |

**C_generators** — 6 of 35 files tested (17%)

| Production File | Test File | Tests | Key Coverage |
|----------------|-----------|-------|-------------|
| C20_abbrev_patterns.py | K23 | 42 | Abbreviation regex patterns |
| C21_noise_filters.py | K24 | 24 | OBVIOUS_NOISE set, validation |
| C22_lexicon_loaders.py | K30 | 15 | MockLexiconLoader |
| C23_inline_definition_detector.py | K25 | 20 | InlineDefinitionDetectorMixin |
| C24_disease_fp_filter.py | K26 | 43 | DiseaseFalsePositiveFilter |
| C25_drug_fp_filter.py | K27 | 38 (15 stubs) | DrugFalsePositiveFilter (partially implemented) |
| C27_feasibility_patterns.py | K28 | 44 | Feasibility regex patterns |
| C34_gene_fp_filter.py | K29 | 43 | GeneFalsePositiveFilter |

**D_validation** — 4 of 4 files tested (100%)

| Production File | Test File | Tests | Key Coverage |
|----------------|-----------|-------|-------------|
| D01_prompt_registry.py | K32 | 18 | PromptTask, PromptBundle, PromptRegistry |
| D02_llm_engine.py | K33 | 22 | ClaudeClient JSON parsing, offset inference |
| D03_validation_logger.py | K34 | 10 | ValidationLogger stats and output |
| D04_quote_verifier.py | K35 | 33 | QuoteVerifier, NumericalVerifier |

**E_normalization** — 4 of 18 files tested (22%)

| Production File | Test File | Tests | Key Coverage |
|----------------|-----------|-------|-------------|
| E01_term_mapper.py | K36 | 15 | TermMapper normalization, fuzzy matching |
| E07_deduplicator.py | K37 | 17 | Scoring, selection |
| E11_span_deduplicator.py | K38 | 20 | NER span deduplication |
| E17_entity_deduplicator.py | K39 | 15 | Cross-entity deduplication |

**F_evaluation** — 2 of 3 files tested (67%)

| Production File | Test File | Tests | Key Coverage |
|----------------|-----------|-------|-------------|
| F01_gold_loader.py | K40 | 18 | JSON/CSV loading, normalization |
| F02_scorer.py | K41 | 25 | Precision, recall, F1, fuzzy matching |

**Other layers**: G_config (1/1 = 100%), H_pipeline (2/5 = 40%), I_extraction (2/3 = 67%), J_export (2/5 = 40%), Z_utils (4/12 = 33%)

### 2.2 Files with Import-Only Coverage

44 production files are tested only via import smoke tests (K31, K42, K50). These tests verify that the module can be imported and that `__all__` exports are valid, but they exercise **zero functional logic**. This includes all 20 strategy modules (C00–C19), all enricher modules (E02–E16, E18), and the abbreviation pipeline (H02).

### 2.3 Files with Zero Coverage

~40 production files have no test of any kind — not even import validation. The most critical untested files:

| File | Lines | Risk | Description |
|------|-------|------|-------------|
| `orchestrator.py` | ~1,850 | **Critical** | Main pipeline entry point, all 16 stages |
| `B01_pdf_to_docgraph.py` | ~960 | **Critical** | Core PDF parser |
| `B02_doc_graph.py` | ~300 | High | Document graph data model |
| `H02_abbreviation_pipeline.py` | ~800 | **Critical** | Abbreviation validation pipeline (import-only) |
| `E04_pubtator_enricher.py` | ~350 | High | PubTator3 API integration |
| `E06_nct_enricher.py` | ~520 | High | ClinicalTrials.gov API integration |
| `A06_drug_models.py` | ~200 | Medium | Drug domain models |
| `A07_feasibility_models.py` | ~250 | Medium | Feasibility domain models |
| `Z06_usage_tracker.py` | ~575 | Medium | LLM usage SQLite tracker |
| `F03_evaluation_runner.py` | ~1,323 | Medium | Evaluation runner (IS the evaluation, not unit-tested) |

---

## 3. Test Classification & Patterns

### 3.1 Test Type Distribution

| Type | Count | % | Description |
|------|-------|---|-------------|
| Pure unit tests | ~780 | 62% | No mocking, no I/O; Pydantic models, regex, math |
| Unit tests with filesystem I/O | ~150 | 12% | tempfile-based isolated I/O |
| Unit tests with mocking | ~180 | 14% | MagicMock, monkeypatch, patch() |
| Integration tests | ~50 | 4% | Cross-layer (A14+B06+H04), layout E2E |
| Import smoke tests | ~100 | 8% | Module importability + __all__ validation |
| End-to-end tests | 0 | 0% | None exist |
| Performance/benchmark tests | 0 | 0% | None exist |
| Property-based tests | 0 | 0% | None exist |
| Snapshot/golden file tests | 0 | 0% | None exist |
| Regression tests (bug-anchored) | 0 | 0% | None exist |

### 3.2 Mocking Strategy

**What is mocked:**

| Dependency | Mock Method | Used In |
|-----------|------------|---------|
| Anthropic SDK | `patch("D_validation.D02_llm_engine.anthropic")` | K33 |
| PubTator3 Client | conftest fixture `mock_pubtator_client` | Available but unused |
| ClinicalTrials.gov | conftest fixture `mock_clinicaltrials_client` | Available but unused |
| HTTP requests | conftest fixtures `mock_requests_get`, `mock_requests_session` | K03 |
| Docling | Conditional skip via `DOCLING_AVAILABLE` | K08 |
| os.environ | `mock.patch.dict("os.environ")` | K16, K33 |
| builtins.print | `patch("builtins.print")` | K15 |
| Entity models | `MagicMock()` with configured attributes | K39, K60 |

**What runs real:**
- All Pydantic model validation
- All regex pattern matching
- FlashText keyword processor (in conftest but not extensively exercised)
- File I/O via tempfile (isolated)
- Scoring/evaluation logic

**What is NOT mocked but should be for unit tests:**
- `ComponentFactory` in K45 instantiates real objects including lexicon loading, which makes the test slow and dependent on lexicon files being present
- K03 uses `time.sleep(1.5)` for cache expiration instead of mocking time

### 3.3 Assertion Patterns

| Pattern | Usage | Quality |
|---------|-------|---------|
| `assert result == expected` | Most common | Strong |
| `assert "keyword" in result` | String containment | Medium |
| `pytest.raises(ValidationError)` | Pydantic rejection | Strong |
| `pytest.raises(ExceptionType)` | Error conditions | Strong |
| `pytest.approx()` | Float comparison | Strong (K14, K41) |
| `assert X is not None` | Existence check | **Weak** — 76 instances |
| `assert result is None or result is not None` | Tautology | **Useless** — 1 instance (K45:128) |
| Custom assertion messages | `assert is_fp, f"{word} should be filtered"` | Strong |

### 3.4 Parametrization

Used effectively in 6 files:
- K26: False positive terms for disease filter (expanded vocabulary)
- K31: 34 C_generators module imports
- K42: D/E/F module imports (6 parametrize decorators)
- K50: G/H/I/J module imports (8 parametrize decorators)

Not used where it would help most:
- K23 (abbreviation patterns): 42 individually written test functions that could be parametrized
- K28 (feasibility patterns): 44 individually written test functions

---

## 4. Coverage Gap Analysis

### 4.1 Coverage by Pipeline Layer

```
Layer               Behavioral   Import-Only   Zero    Coverage
─────────────────────────────────────────────────────────────────
A_core (Models)     ██████████░░░░░░░   56%    28%     16%
B_parsing (PDF)     ████████░░░░░░░░░   41%     0%     59%
C_generators        ███░░░░░░░░░░░░░░   17%    77%      6%
D_validation (LLM)  ████████████████░  100%     0%      0%
E_normalization     ████░░░░░░░░░░░░░   22%    67%     11%
F_evaluation        █████████████░░░░   67%     0%     33%
G_config            ████████████████░  100%     0%      0%
H_pipeline          ████████░░░░░░░░░   40%    40%     20%
I_extraction        █████████████░░░░   67%     0%     33%
J_export            ████████░░░░░░░░░   40%    40%     20%
Z_utils             ██████░░░░░░░░░░░   33%     0%     67%
orchestrator        ████████░░░░░░░░░   50%*    0%     50%

* orchestrator_utils.py tested; orchestrator.py not tested
```

### 4.2 The Critical Untested Path

The pipeline's primary execution path — from PDF input to JSON output — has **no end-to-end test coverage**. Tracing the critical path:

```
orchestrator.process_pdf()          → NOT TESTED
  → H02_abbreviation_pipeline      → IMPORT ONLY
    → B01_pdf_to_docgraph.parse()   → NOT TESTED
    → C01–C04 strategy.extract()    → IMPORT ONLY
    → D02_llm_engine.verify()       → JSON parsing tested, API path not tested
  → I01_entity_processors           → Entity creation tested, full flow not tested
    → E04_pubtator_enricher         → IMPORT ONLY
    → E06_nct_enricher              → IMPORT ONLY
    → E07_deduplicator              → Unit tested
  → J01_export_handlers             → Init tested, file write not tested
```

Every layer is tested in isolation (to varying degrees), but the **inter-layer interfaces are never exercised** in a test environment. A breaking change in one layer's output format would not be caught until a real pipeline run.

### 4.3 What the Tests Would Miss

Based on the coverage gaps, the following classes of bugs would go undetected:

| Bug Type | Example | Would Be Caught? |
|----------|---------|-------------------|
| Regex pattern regression in C20 | Pattern stops matching valid abbreviations | Yes (K23 tests patterns) |
| FP filter regression in C24/C25/C34 | Valid disease name gets filtered | Yes (K26/K27/K29 test filters) |
| Pydantic model field rename in A05 | Disease model field changes name | Partially (K02 tests A05, but not all fields) |
| LLM response format change | Claude returns new JSON structure | No (K33 mocks the response) |
| PubTator API response change | API returns different JSON schema | No (conftest mock is static) |
| PDF parsing quality regression | Unstructured.io update breaks parsing | No (B01 untested) |
| Orchestrator stage ordering bug | Disease detection runs before parsing | No (orchestrator untested) |
| Export file format change | JSON output structure changes | No (no snapshot tests) |
| Entity enrichment data loss | PubTator timeout silently loses MeSH IDs | No (E04 untested) |
| Cross-entity contamination | Drug entity appears in disease output | Partially (K60 tests cross-entity filter) |
| Performance regression | Lexicon loading takes 10x longer | No (no performance tests) |
| Config key typo | Model tier key misspelled | No (config integration untested) |

---

## 5. Test Quality Assessment

### 5.1 Strengths

**Strong domain model testing**: The A_core tests (K01–K57) thoroughly exercise Pydantic validation, immutability, serialization roundtrips, and enum values. Unicode normalization (K54) is particularly comprehensive with all dash variants, mojibake detection, and combined transformations.

**Thorough false-positive filter coverage**: K26 (disease, 43 tests), K27 (drug, 38 tests — though 15 are stubs), and K29 (gene, 43 tests) collectively validate 124+ filter rules. These tests use descriptive assertion messages (`assert is_fp, f"{word} should be filtered"`) that aid debugging.

**Well-structured mock infrastructure**: conftest.py provides realistic API response mocks that match actual PubTator3 and ClinicalTrials.gov response structures. HTTP mocking covers both `requests.get` and `requests.Session` paths.

**Good test isolation**: Tests consistently use `tempfile.TemporaryDirectory()` and pytest's `tmp_path` fixture. No test makes real API calls. Fixtures use function scope (default) preventing cross-test contamination.

**Import smoke tests prevent module-level crashes**: K31, K42, K50 verify all 98+ modules are importable and have valid `__all__` exports. This catches import-time errors, circular dependencies, and missing optional dependencies.

### 5.2 Weaknesses

**15 stub tests in K27**: The drug false-positive filter test file has 15 test methods with empty `pass` bodies covering bacteria filtering, vaccine terms, company names, organization names, biological entities, trial status terms, edge cases, and integration. These represent planned-but-unimplemented tests.

**76 weak assertions**: Tests across the suite use `assert X is not None` without verifying the actual content. The worst offenders are import tests (K31, K42, K50) and ComponentFactory tests (K45), where factory outputs are verified to exist but not to be correctly configured.

**1 tautology assertion**: K45 line 128 contains `assert result is None or result is not None` — a statement that can never fail and provides zero verification.

**Missing negative tests**: Several test files only test the happy path:
- K46 (entity processors): Tests valid entity creation but never tests invalid status transitions, out-of-range confidence scores, or missing required fields
- K47 (feasibility processor): Only tests "returns empty when enricher is None" — never tests enricher exceptions, malformed data, or concurrent access
- K48 (export handlers): Tests export to file but never tests invalid filenames, disk-full conditions, or non-serializable types

**Missing edge cases in existing tests**:
- No test passes `None` inputs to FP filters (K26/K27/K29 test empty strings but not `None`)
- K36 (term mapper) never tests maximum-length strings or empty lexicons
- K41 (scorer) never tests division-by-zero edge case for F1 when both precision and recall are zero
- K37 (deduplicator) never tests tiebreaking behavior when entities have identical scores
- No test verifies behavior with malformed YAML configuration
- No test verifies behavior with corrupted PDF files

### 5.3 Flaky Test Indicators

**Real `time.sleep()` in K03**: Line 49 uses `time.sleep(1.5)` for a cache expiration test. This is timing-dependent — on slow CI machines the sleep may be insufficient; on fast machines it wastes time. Line 220 in the same file correctly patches `time.sleep` for a different test, showing awareness of the problem.

**Real `time.sleep()` in K15**: Lines 27, 38, 63, 67, 89 use `time.sleep(0.005-0.01)` for StageTimer tests with a 0.001-second tolerance assertion. While unlikely to flake, this is fragile on heavily loaded systems.

**Global state mutation in K15**: Lines 131–143 modify `os.environ` (PYTHONWARNINGS, TRANSFORMERS_VERBOSITY, HF_HUB_DISABLE_PROGRESS_BARS) without guaranteed cleanup on test failure. Lines 193–217 replace `sys.stdout` and `sys.stderr` — any test failure between activate and deactivate leaves I/O corrupted for subsequent tests.

### 5.4 Mock Fidelity Concerns

**LLM response mocks in K33**: Tests verify JSON extraction from raw text strings but never validate against the actual `anthropic.types.Message` response structure. If the Anthropic SDK changes its response format, tests would still pass but production would break.

**MagicMock entities in K39 and K60**: The EntityDeduplicator and cross-entity filter tests use `MagicMock()` instead of real Pydantic model instances. Any future Pydantic validator changes could break production but not tests, since MagicMock bypasses validation entirely.

**Static conftest mocks**: API response mocks in conftest.py are hardcoded dictionaries that represent a single point-in-time snapshot of the PubTator3 and ClinicalTrials.gov response formats. These mocks are never validated against the actual API and could become stale.

---

## 6. Gold Standard Evaluation Framework

### 6.1 Architecture

The evaluation framework consists of three files in `F_evaluation/`:

| File | Lines | Purpose |
|------|-------|---------|
| `F01_gold_loader.py` | 264 | Loads gold annotations from JSON/CSV, normalizes, deduplicates |
| `F02_scorer.py` | 639 | Set-based matching with fuzzy/substring support, P/R/F1 |
| `F03_evaluation_runner.py` | 1,323 | End-to-end evaluation against gold PDFs |

### 6.2 Gold Standard Datasets

| Dataset | Documents | Entity Types | Purpose |
|---------|-----------|-------------|---------|
| NLP4RARE | ~95 | Abbreviations | Rare disease medical documents |
| PAPERS | 10 | Abbreviations, diseases, genes, drugs | Research papers (gold v2) |

### 6.3 Scoring Methodology

The `Scorer` class uses set-based matching on `(SF_upper, lf_lower)` tuples with configurable matching modes:

| Mode | Method |
|------|--------|
| Exact match | Tuple equality after normalization |
| Substring match | Gold LF is substring of predicted LF (or vice versa) |
| Fuzzy match | `difflib.SequenceMatcher` with configurable threshold |

Metrics reported:
- **Per-document**: Precision, recall, F1, TP/FP/FN counts
- **Corpus-level**: Micro (summed TP/FP/FN) and macro (averaged per-document) scores
- **Per-entity-type**: Separate scores for abbreviations, diseases, genes, drugs

### 6.4 Evaluation Runner

`F03_evaluation_runner.py` runs the actual orchestrator pipeline against gold PDFs and compares output to gold annotations. It targets 100% precision and recall. Disease evaluation includes synonym group handling, fuzzy text matching, and plural deduplication.

This file has **no unit tests** — it IS the evaluation tool, not a component that gets unit-tested. However, this means the evaluation logic itself (synonym grouping, plural deduplication, fuzzy matching thresholds) is untested for correctness.

### 6.5 Evaluation Gaps vs. SOTA

| Aspect | ESE Current | SOTA Practice |
|--------|------------|---------------|
| **Annotator agreement** | Not measured | Multi-expert annotation with IAA > 0.9 (Krippendorff's Alpha) |
| **Match types** | Exact + substring + fuzzy | Exact + relaxed (left-match, right-match) + token-level |
| **Cross-corpus evaluation** | 2 datasets (NLP4RARE, PAPERS) | Test on 3+ datasets including unseen corpora |
| **Zero-shot evaluation** | Not implemented | Test NER on entity types never seen in training |
| **Error categorization** | Not implemented | Classify errors by type (boundary, type confusion, context) |
| **Behavioral testing** | Not implemented | Test invariance, directional expectations, minimum functionality |
| **Slice-based analysis** | Not implemented | Performance by document type, entity length, context complexity |
| **Automated evaluation in CI** | Not implemented | Gold evaluation runs on every PR |

---

## 7. Infrastructure & CI/CD

### 7.1 Current State

| Component | Status |
|-----------|--------|
| pytest.ini | Configured (testpaths, markers, verbosity) |
| conftest.py | 357 lines (fixtures, mocks, markers) |
| GitHub Actions | **Not configured** |
| Makefile | **Does not exist** |
| tox.ini | **Does not exist** |
| Coverage measurement | **Commented out** in pytest.ini |
| Coverage threshold | **Not set** |
| Test parallelization | **Not configured** (no pytest-xdist) |
| Test timeout | **Not configured** (no pytest-timeout) |
| Pre-commit hooks | **Not configured** |

Tests are run manually:
```
cd corpus_metadata && python -m pytest K_tests/ -v
```

### 7.2 Verification Commands (from CLAUDE.md)

| Command | Purpose |
|---------|---------|
| `python -m pytest K_tests/ -v` | Run all tests |
| `python -m mypy .` | Type checking |
| `python -m ruff check .` | Linting |

All three must pass before any task is considered complete, but this is enforced by convention (CLAUDE.md instructions), not by automation.

### 7.3 Test Execution Characteristics

| Characteristic | Status |
|---------------|--------|
| API key required for tests | No — all external APIs mocked |
| Network access required | No |
| Test ordering dependencies | None detected |
| Heavy resource tests | K45 loads full lexicons (~617K terms) |
| Conditional skips | K08 skips if Docling unavailable |
| Test data location | All inline or tempfile-created; gold data in `/gold_data/` |

---

## 8. Gaps vs. SOTA

### 8.1 Testing Practices Comparison

| Practice | ESE Current | SOTA | Status |
|----------|------------|------|--------|
| **Unit tests for logic** | 780+ tests for models, patterns, filters | Standard practice | **At parity** |
| **Import smoke tests** | 100+ tests across all modules | Good practice | **At parity** |
| **Mock isolation** | API mocks via conftest.py | Standard practice | **At parity** |
| **Pydantic validation tests** | Thorough (edge cases, immutability) | Standard for typed models | **At parity** |
| **CI/CD integration** | Not configured | GitHub Actions/GitLab CI on every PR | **Missing** |
| **Coverage measurement** | Commented out | pytest-cov with minimum threshold (80%+) | **Missing** |
| **End-to-end pipeline tests** | None | Run pipeline with mock LLM, verify output format | **Missing** |
| **Snapshot/golden file tests** | None | Syrupy: compare output JSON against saved snapshots | **Missing** |
| **Regression tests** | None | Test anchored to specific past bugs | **Missing** |
| **Performance tests** | None | Benchmark critical paths, alert on regression | **Missing** |
| **Property-based testing** | None | Hypothesis for data validation, roundtrips, fuzzing | **Missing** |
| **Mutation testing** | None | MutMut/Cosmic-Ray for critical components | **Missing** |
| **LLM output evaluation** | Gold standard F1 scoring | DeepEval DAG/QAG metrics, LLM-as-Judge, tiered thresholds | **Partial** |
| **LLM response caching** | None | Record real responses, replay in tests | **Missing** |
| **Behavioral NER tests** | None | Invariance, directional, minimum functionality tests | **Missing** |
| **Cross-corpus evaluation** | 2 datasets | 3+ datasets including unseen corpora | **Partial** |
| **Data contract testing** | None | Great Expectations at pipeline boundaries | **Missing** |
| **Test parallelization** | Not configured | pytest-xdist -n auto | **Missing** |
| **Test markers/selective execution** | Markers defined but unused | `@pytest.mark.slow` to skip in dev | **Partial** (infrastructure exists) |

### 8.2 SOTA Patterns Not Yet Adopted

#### LLM Response Caching for Tests

Record real Claude API responses during a designated "recording" test run. Store them as JSON fixtures. Replay them in subsequent test runs. This provides realistic test data without API costs or non-determinism.

**ESE impact**: Would enable testing the full LLM validation path (D02 → verify_candidate → _extract_json_object → Pydantic validation) with real response shapes rather than string mocks.

#### Behavioral NER Tests

Three test categories from the ML testing literature:

**Invariance tests**: Changing the context around a known entity should not change extraction. E.g., "Patient has **Duchenne muscular dystrophy**" and "The study focuses on **Duchenne muscular dystrophy** treatment" should both extract the same disease.

**Directional tests**: Adding more context should improve or maintain extraction. E.g., adding "DMD (Duchenne muscular dystrophy)" should increase confidence over "DMD" alone.

**Minimum functionality tests**: Known entities in clean, unambiguous sentences must always be extracted. A curated set of 50–100 critical entity+context pairs that must pass on every pipeline version.

**ESE impact**: Would catch regressions in extraction quality that pure unit tests miss. Particularly valuable for FP filter changes, where a new filter rule might accidentally block valid entities.

#### Snapshot Testing

Use Syrupy (pytest snapshot plugin) to capture full JSON output from processing a reference document. On every test run, compare the new output against the saved snapshot. Any change (added field, missing entity, reordered output) triggers a failure that must be explicitly approved.

**ESE impact**: Would catch export format regressions, entity count changes, and unintentional behavior changes from dependency updates. The gold data in `gold_data/PAPERS/` already contains reference outputs from commit `575fa8b` — these could serve as initial snapshots.

#### Data Contract Testing

Define expected schemas and value constraints at each pipeline boundary:
- Parser output → Candidate generator input: DocumentModel with blocks, tables, pages
- Generator output → Validator input: List of Candidates with required fields
- Validator output → Normalizer input: ExtractedEntities with validation status
- Normalizer output → Export input: Enriched entities with ontology codes

**ESE impact**: Would catch interface mismatches between pipeline layers. Currently, a change in one layer's output format is only caught at runtime during a real pipeline run.

#### Test Pyramid for ML/NLP Systems

The recommended test distribution for a production NLP pipeline:

```
                ╱╲
               ╱  ╲         Acceptance (human eval)
              ╱    ╲
             ╱──────╲       System/E2E (full pipeline, mock LLM)
            ╱        ╲
           ╱──────────╲     Evaluation (gold standard P/R/F1)
          ╱            ╲
         ╱──────────────╲   Behavioral (invariance, directional, min func)
        ╱                ╲
       ╱──────────────────╲  Integration (layer-to-layer data flow)
      ╱                    ╲
     ╱──────────────────────╲ Unit (models, patterns, filters, helpers)
```

**ESE current state**: Only the bottom layer (unit tests) and part of the evaluation layer (gold standard scoring) exist. All other layers are missing.

---

## 9. Recommendations

### Priority 1 — Fix Existing Test Issues

1. **Implement the 15 stub tests in K27** (drug FP filter). These represent known gaps in bacteria, vaccine, company name, and trial status filtering that were planned but never written.

2. **Replace the tautology assertion** in K45 line 128 (`assert result is None or result is not None`) with a meaningful type and configuration check.

3. **Remove real `time.sleep(1.5)` from K03** line 49. Mock `time.time()` to simulate cache expiration instead of waiting real time.

4. **Fix global state mutation in K15**. Use `monkeypatch.setenv()` and `monkeypatch.setattr()` instead of directly modifying `os.environ` and `sys.stdout/sys.stderr`, ensuring cleanup on test failure.

5. **Strengthen 76 weak `is not None` assertions**. At minimum, add type checks (`assert isinstance(result, ExpectedType)`). Ideally, verify specific attribute values.

### Priority 2 — Add Critical Missing Coverage

6. **Add orchestrator integration test**. Create a test that runs `process_pdf()` with mocked LLM, a small test PDF, and mocked external APIs. Verify the output structure, entity counts, and export file creation.

7. **Add enricher unit tests for E04/E06**. Test PubTator3 and ClinicalTrials.gov enrichers with the mock fixtures already defined in conftest.py (currently unused). Cover timeout handling, rate limiting, and response parsing.

8. **Add strategy functional tests for top 5 generators**. Test `extract()` on C01 (abbreviation), C06 (disease), C07 (drug), C16 (gene), C04 (FlashText) with sample text inputs and verify candidate output structure.

9. **Add export round-trip test**. Create entities, export to JSON via J01, read back, verify the deserialized data matches the original entities.

### Priority 3 — Adopt SOTA Practices

10. **Enable coverage measurement**. Uncomment the coverage line in pytest.ini. Set a minimum threshold (start at 40%, increase over time). Install pytest-cov.

11. **Add snapshot tests** using Syrupy. Use the existing gold data pipeline outputs as initial snapshots. Run against a reference document on every test execution.

12. **Add behavioral NER tests**. Create a curated set of 50+ minimum-functionality test cases: known entities in clean sentences that must always be extracted. Run through the full candidate generation + FP filter path.

13. **Set up CI/CD**. Create a GitHub Actions workflow that runs `pytest`, `mypy`, and `ruff` on every push and PR. Add gold standard evaluation as a separate workflow (runs longer, triggered on relevant file changes).

### Priority 4 — Advanced Testing

14. **Add property-based tests** using Hypothesis for unicode normalization (idempotence), text helpers (no consecutive spaces after cleaning), FP filters (valid drug names never filtered), and deduplication (idempotence, subset property).

15. **Add LLM response caching**. Record real Claude responses for 5–10 representative documents. Store as JSON fixtures. Test the full validation path (JSON parsing → Pydantic validation → entity creation) with real response shapes.

16. **Implement data contracts** at pipeline boundaries. Define expected schemas for parser→generator, generator→validator, validator→normalizer, and normalizer→exporter interfaces. Fail tests if any interface changes.

17. **Add performance benchmarks**. Measure lexicon loading time, FlashText matching throughput, and FP filter performance on a reference corpus. Alert if any metric regresses by more than 20%.

18. **Use test markers**. Apply the already-defined `@pytest.mark.slow` to K45 (lexicon loading) and any future integration/E2E tests. Configure pytest to skip slow tests by default in development (`-m "not slow"`).

---

## References

- [LLM Testing in 2025: Top Methods and Strategies — Confident AI](https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies)
- [Testing LLM Applications: A Practical Guide — Langfuse](https://langfuse.com/blog/2025-10-21-testing-llm-applications)
- [DeepEval: The Open-Source LLM Evaluation Framework](https://deepeval.com/)
- [Effective Practices for Mocking LLM Responses — MLOps Community](https://home.mlops.community/public/blogs/effective-practices-for-mocking-llm-responses-during-the-software-development-lifecycle)
- [Automated Prompt Regression Testing with LLM-as-a-Judge — Traceloop](https://www.traceloop.com/blog/automated-prompt-regression-testing-with-llm-as-a-judge-and-ci-cd)
- [LLM-as-a-Judge: A Complete Guide — Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Various Criteria in Biomedical NER Evaluation — BMC Bioinformatics](https://pmc.ncbi.nlm.nih.gov/articles/PMC1402329/)
- [Inter-Annotator Agreement and Upper Limit on Machine Performance — ResearchGate](https://www.researchgate.net/publication/322252759_Inter-Annotator_Agreement_and_the_Upper_Limit_on_Machine_Performance_Evidence_from_Biomedical_Natural_Language_Processing)
- [Improving LLMs for Clinical NER via Prompt Engineering — JAMIA](https://academic.oup.com/jamia/article/31/9/1812/7590607)
- [Effective Testing for Machine Learning Systems — Jeremy Jordan](https://www.jeremyjordan.me/testing-ml/)
- [Testing through the ML Pipeline — MLOps Playbook](https://playbooks.equalexperts.com/mlops-playbook/practices/testing-through-the-ml-pipeline)
- [Great Expectations: Data Validation Framework](https://greatexpectations.io/)
- [Mutation Testing Tools for Python — SBQS 2024](https://dl.acm.org/doi/10.1145/3701625.3701659)
- [pytest-cov 7.0 Documentation](https://pytest-cov.readthedocs.io/en/latest/readme.html)
- [Modern Python CI with Coverage in 2025 — Daniel Nouri](https://danielnouri.org/notes/2025/11/03/modern-python-ci-with-coverage-in-2025/)
- [Hypothesis: Property-Based Testing for Python](https://hypothesis.readthedocs.io/)
- [Agentic Property-Based Testing — arXiv 2025](https://arxiv.org/html/2510.09907v1)
- [Syrupy: The Sweeter Pytest Snapshot Plugin](https://github.com/syrupy-project/syrupy)
- [Annotation Metrics — Prodigy](https://prodi.gy/docs/metrics)
