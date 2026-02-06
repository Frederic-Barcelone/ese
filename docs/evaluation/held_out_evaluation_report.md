# Held-Out Evaluation Report — ESE Pipeline v0.8

**Date:** 2026-02-06
**Pipeline version:** v0.8 (frozen — no code changes between dev and held-out runs)

---

## 1. Motivation: Why Held-Out Evaluation Matters

During development, we iteratively improved the pipeline by analyzing errors on the same data used to report metrics. This is standard practice for tuning, but it creates a methodological risk: **the reported metrics may overestimate generalization performance** due to implicit overfitting to the development set.

This concern applies even when no machine learning model is trained end-to-end, because:

- **FP filter lists** (C24, C25, C34) were curated by examining development-set errors
- **Synonym groups** in F03 were added to match specific gold annotations observed during development
- **Threshold tuning** (e.g., MIN_ADJUSTMENT_FLOOR, fuzzy match ratio) was calibrated on development-set F1
- **Evaluation matching logic** (substring, token overlap, synonym normalization) was refined against development-set edge cases

The standard approach in NLP evaluation is to **develop on a dev/train set** and **report final numbers on a held-out test set** never seen during development. This report follows that protocol.

---

## 2. Experimental Protocol

### 2.1 Design Principles

1. **Frozen pipeline**: No code changes between the development evaluation and the held-out evaluation. The exact same pipeline binary processes both splits.
2. **Non-overlapping splits**: Held-out data comes from portions of each corpus never used during development error analysis.
3. **Identical evaluation logic**: Same matching rules, synonym groups, and scoring in F03 for both dev and held-out runs.

### 2.2 Benchmarks and Splits

| Benchmark | Entity | Dev Split | Dev N | Held-Out Split | Held-Out N |
|-----------|--------|-----------|-------|----------------|------------|
| CADEC | Drugs | test (311 docs) | 294 annotations | train (937 docs) | 904 annotations |
| NLP4RARE | Diseases | all splits pooled (1040 docs) | 4,123 annotations | test only (208 docs) | 808 annotations |
| NLP4RARE | Abbreviations | all splits pooled (1040 docs) | 242 annotations | test only (208 docs) | 58 annotations |
| BC2GM | Genes | batch1 (100 docs, sentences 1-100) | 227 annotations | batch2 (100 docs, sentences 101-200) | 249 annotations |

**Notes on split design:**
- **CADEC**: The evaluation script's `--split=train` gives us 937 forum posts never analyzed during development. The test split (311 docs) was used for all tuning.
- **NLP4RARE**: Development work examined errors across all splits pooled. The held-out run uses only the `test` subfolder (208 docs). This is conservative — some test-split documents may have been seen during pooled error analysis, but they were never isolated for targeted fixes.
- **BC2GM**: The generation script (`--skip 100 --max 100`) produces 100 PDFs from sentence IDs 101-200 in the BioCreative II GM corpus. Batch1 (IDs 1-100) was used during development. Zero sentence ID overlap was verified programmatically.

---

## 3. Results

### 3.1 Summary Table

| Benchmark | Entity | Split | N Docs | N Gold | TP | FP | FN | P | R | F1 |
|-----------|--------|-------|--------|--------|----|----|-----|---|---|-----|
| CADEC | Drugs | test (dev) | 311 | 294 | 275 | 19 | 21 | 93.5% | 92.9% | 93.2% |
| CADEC | Drugs | **train (held-out)** | 937 | 904 | 743 | 152 | 161 | **83.0%** | **82.2%** | **82.6%** |
| NLP4RARE | Diseases | all (dev) | 1040 | 3,938 | 2,915 | 965 | 1,023 | 75.1% | 74.0% | 74.6% |
| NLP4RARE | Diseases | **test (held-out)** | 208 | 808 | 608 | 175 | 200 | **77.7%** | **75.2%** | **76.4%** |
| NLP4RARE | Abbrevs | all (dev) | 1040 | 242 | 215 | 238 | 27 | 47.5% | 88.8% | 61.9% |
| NLP4RARE | Abbrevs | **test (held-out)** | 208 | 58 | 53 | 61 | 5 | **46.5%** | **91.4%** | **61.6%** |
| BC2GM | Genes | batch1 (dev) | 100 | 227 | 28 | 3 | 199 | 90.3% | 12.3% | 21.7% |
| BC2GM | Genes | **batch2 (held-out)** | 100 | 249 | 21 | 7 | 228 | **75.0%** | **8.4%** | **15.2%** |

### 3.2 Delta Analysis (Held-Out minus Dev)

| Benchmark | Entity | Delta P | Delta R | Delta F1 | Interpretation |
|-----------|--------|---------|---------|----------|---------------|
| CADEC | Drugs | -10.5pp | -10.7pp | -10.6pp | Meaningful drop — see 4.1 |
| NLP4RARE | Diseases | +2.6pp | +1.2pp | +1.8pp | Stable or slightly better |
| NLP4RARE | Abbrevs | -1.0pp | +2.6pp | -0.3pp | Stable (small sample, see 4.2) |
| BC2GM | Genes | -15.3pp | -3.9pp | -6.5pp | Precision drop — see 4.3 |

---

## 4. Analysis

### 4.1 CADEC Drugs: -10.6pp F1 Drop

The CADEC drug benchmark shows the largest generalization gap. This was expected for several reasons:

**Why the train split is harder:**
- The CADEC train split (937 docs) covers more diverse drug names and consumer spelling variants than the test split (311 docs)
- The FP filter and consumer drug variant lists were explicitly tuned on the test split during development
- The train split contains more noisy gold annotations (misspellings like "ibruprofen", "vicodine", "cq10") that our exact-match pipeline doesn't handle
- False positives increased from 19 to 152 — the FP filter was tuned for test-split patterns

**Conclusion:** The -10.6pp gap indicates real overfitting to the test split. The held-out F1 of 82.6% is the more honest estimate of generalization performance on social media drug extraction.

### 4.2 NLP4RARE: Stable Performance

**Diseases (+1.8pp F1):** The held-out test split actually performs slightly *better* than the pooled development set. This suggests the pipeline's disease detection generalizes well and the development tuning was not overfit to specific documents. The test split may be slightly easier on average, or the larger pooled set includes more challenging edge cases from dev/train splits.

**Abbreviations (-0.3pp F1):** Nearly identical performance. However, the held-out sample is small — only **58 gold annotations across ~50 documents**. With N=58, the 95% confidence interval on F1 is approximately +/-13pp (Wilson interval). The -0.3pp delta is well within sampling noise and should not be interpreted as evidence of overfitting or improvement.

**Conclusion:** NLP4RARE shows no evidence of overfitting. The disease pipeline generalizes reliably, and the abbreviation result is consistent but imprecise due to small sample size.

### 4.3 BC2GM Genes: Expected Low Performance

**Precision dropped from 90.3% to 75.0%** on the held-out batch. The batch2 FPs include HAL, ASPM, GTF2I, ATR, IFNGR2, CAT, ASPM — HGNC gene symbols that happen to be present in the text but are not annotated in BC2GM's protein-name-oriented gold standard. This is a schema mismatch rather than pipeline error.

**Recall remained very low (8.4% vs 12.3%).** This is consistent with the known schema mismatch: BC2GM annotates broad protein/gene names (collagen, metalloproteinase, luciferase), while our pipeline extracts HGNC symbols only. The held-out batch has 249 annotations (vs 227 in batch1), with slightly more protein names and fewer short symbols, explaining the further recall drop.

**Conclusion:** The BC2GM benchmark confirms the known schema limitation. The -6.5pp F1 drop is driven by precision (more gene symbols flagged as FP in batch2 text), not by overfitting.

---

## 5. Validity of This Evaluation Approach

### 5.1 Why Train/Test Splits Are the Standard

The train/test (or dev/test) split methodology is the **universally accepted practice** in NLP evaluation, codified in shared tasks (SemEval, BioNLP, CoNLL) and recommended by survey papers on biomedical NER evaluation:

1. **Development set**: Used for all tuning — error analysis, threshold calibration, filter list curation, matching rule refinement. Metrics from this set are *optimistic* because the system was adapted to it.

2. **Held-out test set**: Touched only once, at the end, with the frozen system. Metrics from this set estimate *true generalization performance* on unseen data from the same distribution.

This applies even to rule-based and hybrid systems (like ESE) that don't have traditional ML training. Any human-in-the-loop decision that examined error outputs and adjusted the system constitutes implicit training on that data.

### 5.2 Limitations of This Protocol

- **NLP4RARE test split is not fully held-out**: During pooled development runs, test-split documents were processed and their errors were visible. However, targeted fixes were made by examining specific documents (usually from dev or train), not test specifically.
- **Small sample for abbreviations**: N=58 gold abbreviation annotations in the test split limits statistical power. Confidence intervals are wide.
- **BC2GM schema mismatch**: Both dev and held-out suffer from the same systematic issue (protein names vs HGNC symbols), so the delta mainly reflects batch-level variation rather than overfitting.
- **LLM non-determinism**: The abbreviation pipeline uses LLM calls (Haiku 4.5) whose outputs vary between runs. The ~1-2% F1 variation observed across repeat runs means small deltas may be noise rather than signal.

### 5.3 Recommendations for Future Evaluation

1. **Strict split discipline**: Designate a test split at the start and never examine its errors during development. Only run it for final reporting.
2. **Larger held-out sets**: For abbreviations, the 58-annotation test set is too small for precise estimates. Consider expanding the NLP4RARE gold standard or adding a second abbreviation benchmark.
3. **Multiple held-out batches**: For BC2GM, running 3-5 batches of 100 documents each and reporting mean +/- std would give more robust estimates.
4. **Cross-validation**: For small corpora like CADEC, k-fold cross-validation would give tighter confidence intervals than a single train/test split.

---

## 6. Summary

| Benchmark | Entity | Dev F1 | Held-Out F1 | Gap | Verdict |
|-----------|--------|--------|-------------|-----|---------|
| CADEC | Drugs | 93.2% | 82.6% | -10.6pp | Overfitting detected |
| NLP4RARE | Diseases | 74.6% | 76.4% | +1.8pp | Generalizes well |
| NLP4RARE | Abbreviations | 61.9% | 61.6% | -0.3pp | Stable (low N) |
| BC2GM | Genes | 21.7% | 15.2% | -6.5pp | Schema mismatch dominates |

**Key takeaway:** The NLP4RARE disease benchmark — the pipeline's primary use case — shows **no evidence of overfitting**, with the held-out test set actually scoring slightly higher than the development set. The CADEC drug benchmark reveals meaningful overfitting (-10.6pp), driven by FP filter lists tuned on the test split. The BC2GM gene benchmark's poor performance is dominated by schema mismatch rather than overfitting.

**Honest generalization estimates for the ESE pipeline:**
- **Drug extraction:** F1 ~83% on social media text (CADEC train)
- **Disease extraction:** F1 ~76% on rare disease medical documents (NLP4RARE test)
- **Abbreviation extraction:** F1 ~62% on rare disease documents (NLP4RARE test, N=58)
- **Gene extraction:** F1 ~15-22% on BC2GM (schema mismatch — pipeline extracts HGNC symbols, benchmark expects protein names)
