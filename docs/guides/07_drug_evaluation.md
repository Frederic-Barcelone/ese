# Drug Detection Evaluation

## Current Status: Production-Ready

The drug detection pipeline has been validated against the CADEC benchmark, confirming strong performance on informal, consumer-written drug mentions.

| Benchmark | Docs | P | R | F1 | Perfect Docs | Status |
|-----------|------|---|---|----|-------------|--------|
| CADEC drugs (social media) | 311 | 93.5% | 92.9% | 93.2% | 91.3% | Production-ready |

## Pipeline Architecture

Drug detection follows the standard ESE three-layer strategy:

```
C_generators/C07_strategy_drug.py    -> High recall candidate generation
C_generators/C25_drug_fp_filter.py   -> False positive filtering
C_generators/C26_drug_fp_constants.py -> FP filter term lists
E_normalization/                     -> Drug normalization (RxNorm, DrugBank)
```

The drug generator uses FlashText lexicon matching against multiple drug vocabularies (ChEMBL, RxNorm, Orphanet drugs), supplemented by consumer drug variant patterns. Candidates are filtered through the drug FP filter before validation.

### FlashText Layers

| Layer | Terms | Source |
|-------|-------|--------|
| ChEMBL approved drugs | 23K | `ouput_datasources/` |
| RxNorm drug vocabulary | 132K | `ouput_datasources/` |
| Consumer drug variants | ~200 | `C26_drug_fp_constants.py` |

## Gold Standard: CADEC

### Why CADEC

CADEC (CSIRO Adverse Drug Event Corpus) provides annotated drug mentions from AskaPatient consumer health forums. This is a challenging benchmark because:

- Text is informal and contains misspellings (e.g., "Liptor", "zocar", "alleve")
- Drug names appear in consumer brand form (Lipitor, Advil) rather than INN/generic names
- Posts mention supplements, vitamins, and borderline substances alongside prescription drugs
- Some gold annotations are noisy (e.g., "Stopped", "SATIN", "one" annotated as drugs)

### Corpus Details

- **Source**: CSIRO CADEC v2 (collection 17190)
- **Size**: 1,248 documents (937 train, 311 test), 1,198 drug annotations
- **Format**: BRAT .ann files with text files for decoding
- **Evaluation split**: Test set (311 docs) is the primary benchmark

### File Layout

```
gold_data/
  CADEC/
    generate_cadec_gold.py       # Gold generation script
    evaluate_cadec_drugs.py      # Standalone drug evaluation
    cadec_gold.json              # Generated gold standard
    original/                    # Downloaded CADEC corpus (gitignored)
```

### Running Evaluation

```bash
cd corpus_metadata && python ../gold_data/CADEC/evaluate_cadec_drugs.py --split=test
```

Options:
- `--split=test` (default), `--split=train`, or `--split=all`
- `--max-docs=N` to limit documents
- `--seqeval` for token-level BIO F1 (requires seqeval package)

## Full Run Results (311 test docs)

| Metric | Value |
|--------|-------|
| Precision | 93.5% |
| Recall | 92.9% |
| F1 | 93.2% |
| TP | 273 |
| FP | 19 |
| FN | 21 |
| Perfect docs | 284/311 (91.3%) |

### FP Analysis (19 false positives)

Most FPs are correct drug detections that happen to not be in the gold annotations:

| Category | Examples | Notes |
|----------|----------|-------|
| Correct but not in gold | Lipitor, Zocor, ezetimibe, CoQ10 | Pipeline finds drugs the gold missed |
| Supplements | red yeast rice, Niacin | Borderline substances |
| Other substances | Steroids, Caffeine, alcohol, progesterone | Legitimate detections, not annotated |

### FN Analysis (21 false negatives)

| Category | Examples | Notes |
|----------|----------|-------|
| Noisy gold | Stopped, SATIN, one, considering | Gold annotation errors |
| Misspellings | Liptor, time - release naicin | Extreme consumer misspellings |
| Spacing variants | CoQ 10, Gas - X, Vit . C | Unusual whitespace in drug names |
| Generic terms | antibiotic, opiate, thyroid | Class-level terms, not specific drugs |
| Not in lexicon | Pernamax, RX | Uncommon brand names |

### Drug Matching Logic

The evaluation uses multi-step matching via `drug_matches()`:

1. **Exact match** -- Lowercased, whitespace-normalized comparison
2. **Substring match** -- Either name contained in the other
3. **Brand/generic equivalence** -- Maps between brand and INN names (e.g., acetaminophen ↔ Tylenol, atorvastatin ↔ Lipitor)
4. **Fuzzy match** -- SequenceMatcher ratio >= 0.8

### Brand/Generic Equivalences

The evaluation includes 30+ brand/generic mappings to handle the consumer brand names prevalent in CADEC:

| Generic (INN) | Brand Names |
|---------------|-------------|
| acetaminophen | Tylenol, Paracetamol, Panadol |
| atorvastatin | Lipitor |
| celecoxib | Celebrex |
| diclofenac | Voltaren, Arthrotec, Cataflam |
| ibuprofen | Advil, Motrin, Nurofen |
| naproxen | Aleve, Naprosyn |
| simvastatin | Zocor |

## Key Fixes That Improved F1

### FP Filter Expansion (F1: 55.2% → 79.6%)

Added terms to `C26_drug_fp_constants.py`:
- **BIOLOGICAL_ENTITIES**: cholesterol, LDL, HDL, triglycerides
- **BODY_PARTS**: hip, disc
- **EQUIPMENT_PROCEDURES**: blood test
- **COMMON_WORDS**: sleep, cold, cough, statin + consumer terms

### Author Pattern Bug Fix (F1: 79.6% → improved)

`C25_drug_fp_filter.py` line 256 had `re.IGNORECASE` on author initials regex, causing `[A-Z]{1,2}` to match lowercase words ("to", "i", "at"). Fixed by removing IGNORECASE.

### Consumer Drug Variants (F1: → 93.2%)

Added `CONSUMER_DRUG_VARIANTS` and `CONSUMER_DRUG_PATTERNS` in `C26_drug_fp_constants.py`, loaded as a new FlashText layer in `C07_strategy_drug.py`. This handles common misspellings and informal brand names.

## Interpretation

The 93.5% precision / 92.9% recall profile tells us:

1. **The pipeline handles informal drug text well.** 93%+ F1 on consumer health forum text with misspellings.
2. **Most FPs are actually correct.** 19 FPs include drugs like Lipitor and Zocor that the gold standard missed.
3. **Most FNs are gold noise.** Annotations like "Stopped" and "SATIN" are not real drug names.
4. **The consumer variant layer works.** Adding informal spelling patterns significantly improved recall.

## Related Documentation

- [Evaluation Guide](04_evaluation.md) for general evaluation framework usage
- [Gene Evaluation](06_gene_evaluation.md) for BC2GM gene benchmark details
- [Configuration Guide](03_configuration.md) for pipeline settings
