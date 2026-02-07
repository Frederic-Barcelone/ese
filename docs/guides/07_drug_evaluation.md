# Drug Detection Evaluation

## Results (CADEC, 311 test docs)

| Metric | Value |
|--------|-------|
| Precision | 93.5% |
| Recall | 92.9% |
| F1 | 93.2% |
| TP / FP / FN | 273 / 19 / 21 |
| Perfect docs | 284/311 (91.3%) |

## Pipeline Architecture

```
C07_strategy_drug.py      -> High recall candidate generation (FlashText)
C25_drug_fp_filter.py     -> False positive filtering
C26_drug_fp_constants.py  -> FP filter term lists
E_normalization/          -> Drug normalization (RxNorm, DrugBank)
```

### FlashText Layers

| Layer | Terms | Source |
|-------|-------|--------|
| ChEMBL approved drugs | 23K | `ouput_datasources/` |
| RxNorm drug vocabulary | 133K | `ouput_datasources/` |
| Consumer drug variants | ~200 | `C26_drug_fp_constants.py` |

## Gold Standard: CADEC

Social media adverse drug event corpus (AskaPatient forums, CSIRO CADEC v2). 1,248 docs (937 train, 311 test), 1,198 drug annotations. Challenging due to misspellings, consumer brand names, supplements, and noisy gold annotations.

```
gold_data/CADEC/
  generate_cadec_gold.py       # Gold generation
  evaluate_cadec_drugs.py      # Standalone evaluator
  cadec_gold.json              # Generated gold standard
  original/                    # Downloaded corpus (gitignored)
```

### Running

```bash
cd corpus_metadata && python ../gold_data/CADEC/evaluate_cadec_drugs.py --split=test
# Options: --split=test|train|all, --max-docs=N, --seqeval
```

## Error Analysis

### FPs (19)

Mostly correct detections not in gold (Lipitor, Zocor, ezetimibe, CoQ10) and borderline substances (supplements, other medications).

### FNs (21)

| Category | Examples |
|----------|----------|
| Gold noise | Stopped, SATIN, one |
| Misspellings | Liptor, time-release naicin |
| Spacing variants | CoQ 10, Gas - X |
| Generic terms | antibiotic, opiate |

## Drug Matching Logic

1. **Exact** -- Lowercased, whitespace-normalized
2. **Substring** -- Either name contained in the other
3. **Brand/generic equivalence** -- 30+ mappings (e.g., acetaminophen <-> Tylenol, atorvastatin <-> Lipitor)
4. **Fuzzy** -- SequenceMatcher >= 0.8

## Key Fixes

1. **FP filter expansion** (F1: 55% -> 80%): Added biological entities (cholesterol, LDL), body parts, equipment, common words to `C26_drug_fp_constants.py`
2. **Author pattern bug** (improved precision): `C25` had `re.IGNORECASE` on author initials regex matching lowercase words. Removed.
3. **Consumer drug variants** (F1: -> 93%): `CONSUMER_DRUG_VARIANTS` + `CONSUMER_DRUG_PATTERNS` in `C26`, loaded as FlashText layer in `C07`

## Related

- [Evaluation Guide](04_evaluation.md)
- [Gene Evaluation](06_gene_evaluation.md)
- [Configuration Guide](03_configuration.md)
