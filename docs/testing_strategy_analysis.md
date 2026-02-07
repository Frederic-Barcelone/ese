# ESE Pipeline v0.8 -- Testing Strategy Analysis

> **Date**: February 4, 2026

---

## 1. Test Suite Overview

| Metric | Value |
|--------|-------|
| Test files | 62 (60 K-files + conftest.py + \_\_init\_\_.py) |
| Test functions | ~1,350 |
| Lines of test code | ~16,300 |
| Production files | ~147 |
| Parametrized test files | 6 |
| Stub/placeholder tests | 15 |
| CI/CD pipelines | 0 |

### Test Distribution by Layer

| Layer | Files | Functions | Top Module |
|-------|-------|-----------|------------|
| A_core | 11 | 300 | A20 unicode_utils (47) |
| B_parsing | 13 | 248 | B15 caption_extractor (37) |
| C_generators | 9 | 271 | C27 feasibility_patterns (44) |
| D_validation | 4 | 99 | D04 quote_verifier (33) |
| E_normalization | 4 | 67 | E11 span_deduplicator (20) |
| F_evaluation | 2 | 43 | F02 scorer (25) |
| G_config | 1 | 35 | G01 config_keys (35) |
| H_pipeline | 2 | 36 | H01 component_factory (19) |
| I_extraction | 2 | 24 | I01 entity_processors (14) |
| J_export | 2 | 24 | J02 visual_export (15) |
| Z_utils | 4 | 112 | Z03 text_normalization (48) |
| Cross-cutting | 5 | 61 | A14 extraction_result (26) |

### Infrastructure

**conftest.py** (357 lines): Shared fixtures, mock API responses, entity factories. Three markers defined (`slow`, `integration`, `requires_gpu`) -- none used.

**pytest.ini**: `test_*.py` and `K*.py` naming, `--strict-markers`, `--tb=short`. No plugins (no pytest-cov, xdist, timeout).

---

## 2. Layer-by-Layer Coverage

### Files with Behavioral Tests (54 of ~140)

**A_core** -- 14/25 (56%): Provenance hashing (K51, 32 tests), disease models (K02, 18), author/citation models (K52/K53), exceptions (K04, 19), NER models (K06, 29), visual models (K07/K22, 50), extraction result (K05, 26), domain profile (K55, 32), pipeline metrics (K56, 33), unicode utils (K54, 47), clinical criteria (K57, 31).

**B_parsing** -- 13/32 (41%): Visual pipeline (K13), detector (K18), renderer (K20), caption extractor (K17, 37), triage (K21, 34), layout models (K12, 24), layout analyzer (K10), zone expander (K14), filename generator (K09), docling backend (K08).

**C_generators** -- 6/35 (17%): Abbreviation patterns (K23, 42), noise filters (K24, 24), lexicon loaders (K30, 15), inline definition (K25, 20), disease FP filter (K26, 43), drug FP filter (K27, 38 -- 15 stubs), feasibility patterns (K28, 44), gene FP filter (K29, 43).

**D_validation** -- 4/4 (100%): Prompt registry (K32), LLM engine (K33, 22), validation logger (K34), quote verifier (K35, 33).

**E_normalization** -- 4/18 (22%): Term mapper (K36), deduplicator (K37), span deduplicator (K38), entity deduplicator (K39).

**Other**: F_evaluation 2/3 (67%), G_config 1/1 (100%), H_pipeline 2/5 (40%), I_extraction 2/3 (67%), J_export 2/5 (40%), Z_utils 4/12 (33%).

### Import-Only Coverage

All strategy modules (C00-C19), enricher modules (E02-E16, E18), and H02 have only import smoke tests via K31/K42/K50. These verify importability and `__all__` exports but exercise zero logic.

### Critical Untested Files

| File | Lines | Risk |
|------|-------|------|
| `orchestrator.py` | ~1,840 | **Critical** -- main entry point |
| `B01_pdf_to_docgraph.py` | ~960 | **Critical** -- core PDF parser |
| `H02_abbreviation_pipeline.py` | ~800 | **Critical** -- import-only |
| `E04_pubtator_enricher.py` | ~350 | High -- API integration |
| `E06_nct_enricher.py` | ~520 | High -- API integration |
| `B02_doc_graph.py` | ~300 | High |
| `Z06_usage_tracker.py` | ~575 | Medium |
| `F03_evaluation_runner.py` | ~1,687 | Medium -- IS the evaluation tool |

---

## 3. Test Classification & Patterns

### Type Distribution

| Type | Count | % |
|------|-------|---|
| Pure unit tests | ~780 | 62% |
| Unit with filesystem I/O | ~150 | 12% |
| Unit with mocking | ~180 | 14% |
| Integration tests | ~50 | 4% |
| Import smoke tests | ~100 | 8% |
| E2E / performance / property-based / snapshot / regression | 0 | 0% |

### Mocking Strategy

**Mocked**: Anthropic SDK (K33), HTTP requests (conftest), Docling (conditional skip), os.environ. **Real**: Pydantic validation, regex, FlashText, file I/O via tempfile, scoring.

**Problems**: K45 loads real lexicons (~617K terms). K03 uses `time.sleep(1.5)` instead of mocking time.

### Assertion Quality

Strong: `assert result == expected`, `pytest.raises()`, `pytest.approx()`. Weak: 76 `assert X is not None` without content verification. Tautology: K45:128 (`assert result is None or result is not None`).

### Parametrization

Used well in K26/K31/K42/K50. K23 (42 tests) and K28 (44 tests) should be parametrized but are not.

---

## 4. Coverage Gap Analysis

### Coverage by Layer

```
Layer               Behavioral  Import-Only  Zero
A_core (Models)        56%         28%        16%
B_parsing (PDF)        41%          0%        59%
C_generators           17%         77%         6%
D_validation          100%          0%         0%
E_normalization        22%         67%        11%
F_evaluation           67%          0%        33%
G_config              100%          0%         0%
H_pipeline             40%         40%        20%
I_extraction           67%          0%        33%
J_export               40%         40%        20%
Z_utils                33%          0%        67%
orchestrator           50%*         0%        50%
* orchestrator_utils.py tested; orchestrator.py not
```

### The Critical Untested Path

The primary execution path (PDF input to JSON output) has **no end-to-end test coverage**:

```
orchestrator.process_pdf()         -> NOT TESTED
  -> H02_abbreviation_pipeline     -> IMPORT ONLY
    -> B01_pdf_to_docgraph.parse() -> NOT TESTED
    -> C01-C04 strategy.extract()  -> IMPORT ONLY
    -> D02_llm_engine.verify()     -> JSON parsing tested, API path not
  -> I01_entity_processors          -> Entity creation tested, full flow not
    -> E04/E06 enrichers           -> IMPORT ONLY
    -> E07_deduplicator            -> Unit tested
  -> J01_export_handlers            -> Init tested, file write not
```

Every layer is tested in isolation, but **inter-layer interfaces are never exercised**.

### What Tests Would Miss

| Bug Type | Caught? |
|----------|---------|
| Regex pattern regression in C20 | Yes |
| FP filter regression in C24/C25/C34 | Yes |
| LLM response format change | No (mocked) |
| PubTator API response change | No (static mock) |
| PDF parsing quality regression | No (B01 untested) |
| Orchestrator stage ordering bug | No (untested) |
| Export file format change | No (no snapshots) |
| Enrichment data loss (PubTator timeout) | No (E04 untested) |
| Performance regression | No (no benchmarks) |

---

## 5. Test Quality Assessment

### Strengths

- **Domain model testing**: A_core exercises Pydantic validation, immutability, serialization roundtrips. K54 (unicode) is comprehensive.
- **FP filter coverage**: K26/K27/K29 validate 124+ filter rules.
- **Mock infrastructure**: conftest.py provides realistic PubTator3/ClinicalTrials.gov response mocks.
- **Test isolation**: `tempfile`/`tmp_path`, no real API calls, function-scoped fixtures.
- **Import smoke tests**: K31/K42/K50 verify 98+ modules importable with valid `__all__`.

### Weaknesses

- **15 stub tests in K27**: Empty `pass` bodies.
- **76 weak assertions**: `assert X is not None` without content verification.
- **Missing negative tests**: K46/K47/K48 only test happy paths.
- **Missing edge cases**: No `None` inputs to FP filters, no division-by-zero for F1, no malformed YAML, no corrupted PDFs.

### Flaky Tests

- **K03**: `time.sleep(1.5)` for cache expiration.
- **K15**: `time.sleep(0.005-0.01)` with 0.001s tolerance; modifies `os.environ`/`sys.stdout` without guaranteed cleanup.

### Mock Fidelity

- **K33**: LLM mocks are raw strings, not `anthropic.types.Message` -- SDK changes would pass tests but break production.
- **K39/K60**: `MagicMock()` instead of real Pydantic models.
- **conftest.py**: Static API mocks never validated against actual APIs.

---

## 6. Gold Standard Evaluation

### Architecture

| File | Lines | Purpose |
|------|-------|---------|
| `F01_gold_loader.py` | 264 | Load gold annotations from JSON/CSV |
| `F02_scorer.py` | 639 | Set-based matching with fuzzy/substring, P/R/F1 |
| `F03_evaluation_runner.py` | ~1,765 | End-to-end evaluation against gold PDFs |

### Datasets

| Dataset | Documents | Entity Types |
|---------|-----------|-------------|
| NLP4RARE | 1,040+ | Diseases, abbreviations |
| CADEC | 1,248 | Drugs |
| NLM-Gene | 550 | Genes |
| RareDisGene | ~3,976 | Genes (gene-disease associations) |
| PAPERS | 10 | Abbreviations, diseases, genes, drugs |

### Scoring

Set-based matching: exact, substring, fuzzy (`difflib.SequenceMatcher`). Per-document and corpus-level (micro/macro) P/R/F1. F03 has 6-step disease matching including synonym groups and token overlap.

`F03_evaluation_runner.py` has **no unit tests** -- it IS the evaluation tool.

### Evaluation Gaps

| Aspect | Current | SOTA |
|--------|---------|------|
| Match types | Exact + substring + fuzzy + synonym | + token-level |
| Cross-corpus evaluation | 5 datasets | 6+ including unseen |
| Error categorization | Not implemented | By type (boundary, confusion) |
| Behavioral testing | Not implemented | Invariance, directional |
| Automated in CI | Not implemented | On every PR |

---

## 7. Infrastructure & CI/CD

| Component | Status |
|-----------|--------|
| pytest.ini | Configured |
| conftest.py | 357 lines |
| GitHub Actions | **Not configured** |
| Coverage measurement | **Commented out** |
| Coverage threshold | **Not set** |
| Test parallelization | **Not configured** |
| Test timeout | **Not configured** |
| Pre-commit hooks | **Not configured** |
| Makefile / tox.ini | **Not present** |

Tests run manually: `cd corpus_metadata && python -m pytest K_tests/ -v`. All three checks (`pytest`, `mypy`, `ruff`) enforced by convention (CLAUDE.md) not automation. No API key required. K45 loads full lexicons (~617K terms). K08 skips if Docling unavailable.

---

## 8. Gaps vs. SOTA

| Practice | Status |
|----------|--------|
| Unit tests for logic | **At parity** -- 780+ tests |
| Import smoke tests | **At parity** -- 100+ tests |
| Mock isolation | **At parity** |
| Pydantic validation tests | **At parity** |
| CI/CD integration | **Missing** |
| Coverage measurement | **Missing** |
| E2E pipeline tests | **Missing** |
| Snapshot/golden file tests | **Missing** |
| Regression tests (bug-anchored) | **Missing** |
| Performance benchmarks | **Missing** |
| Property-based testing | **Missing** |
| Mutation testing | **Missing** |
| LLM response caching for tests | **Missing** |
| Behavioral NER tests | **Missing** |
| Data contract testing | **Missing** |
| Test parallelization | **Missing** |
| Test markers (selective execution) | **Partial** -- defined but unused |
| LLM output evaluation | **Partial** -- gold F1 scoring |
| Cross-corpus evaluation | **Partial** -- 5 datasets |

### Key SOTA Patterns Not Yet Adopted

**LLM response caching**: Record real Claude responses as JSON fixtures, replay in tests. Enables testing full validation path.

**Behavioral NER tests**: (1) Invariance -- context changes should not affect extraction. (2) Directional -- more context should improve confidence. (3) Minimum functionality -- 50-100 critical entity+context pairs.

**Snapshot testing** (Syrupy): Capture JSON output from reference documents; changes trigger review.

**Data contract testing**: Schemas at each pipeline boundary (parser->generator->validator->normalizer->exporter).

**ESE current state**: Only unit tests and partial evaluation exist (bottom two layers of pyramid).

---

## 9. Recommendations

### Priority 1 -- Fix Existing Issues

1. **Implement 15 stub tests in K27** (drug FP filter: bacteria, vaccine, company names, trial status).
2. **Replace tautology** in K45:128 with meaningful type/configuration check.
3. **Remove `time.sleep(1.5)` from K03** -- mock `time.time()` instead.
4. **Fix global state mutation in K15** -- use `monkeypatch.setenv()` and `monkeypatch.setattr()`.
5. **Strengthen 76 weak assertions** -- add `isinstance` checks or verify specific values.

### Priority 2 -- Critical Missing Coverage

6. **Orchestrator integration test**: `process_pdf()` with mocked LLM/APIs, verify output structure.
7. **Enricher unit tests for E04/E06**: Use existing conftest mock fixtures. Cover timeouts, rate limits, parsing.
8. **Strategy functional tests**: `extract()` on C01/C06/C07/C16/C04 with sample text.
9. **Export round-trip test**: Create entities, export to JSON, read back, verify.

### Priority 3 -- SOTA Practices

10. **Coverage measurement** (pytest-cov, threshold 40%).
11. **Snapshot tests** (Syrupy) on gold data outputs.
12. **Behavioral NER tests**: 50+ minimum-functionality cases through candidate generation + FP filter.
13. **CI/CD** (GitHub Actions): pytest, mypy, ruff on push/PR.

### Priority 4 -- Advanced

14. **Property-based tests** (Hypothesis) for unicode normalization, FP filter invariants, dedup.
15. **LLM response caching**: Record Claude responses for 5-10 documents as fixtures.
16. **Data contracts** at pipeline boundaries.
17. **Performance benchmarks** for lexicon loading, FlashText throughput.
18. **Test markers**: `@pytest.mark.slow` on K45 and integration tests.

---

## References

- [LLM Testing in 2025: Top Methods and Strategies -- Confident AI](https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies)
- [Testing LLM Applications: A Practical Guide -- Langfuse](https://langfuse.com/blog/2025-10-21-testing-llm-applications)
- [DeepEval: The Open-Source LLM Evaluation Framework](https://deepeval.com/)
- [Effective Practices for Mocking LLM Responses -- MLOps Community](https://home.mlops.community/public/blogs/effective-practices-for-mocking-llm-responses-during-the-software-development-lifecycle)
- [Various Criteria in Biomedical NER Evaluation -- BMC Bioinformatics](https://pmc.ncbi.nlm.nih.gov/articles/PMC1402329/)
- [Effective Testing for Machine Learning Systems -- Jeremy Jordan](https://www.jeremyjordan.me/testing-ml/)
- [Great Expectations: Data Validation Framework](https://greatexpectations.io/)
- [Hypothesis: Property-Based Testing for Python](https://hypothesis.readthedocs.io/)
- [Syrupy: The Sweeter Pytest Snapshot Plugin](https://github.com/syrupy-project/syrupy)
