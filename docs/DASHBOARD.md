# ESE Pipeline Accuracy Dashboard

> Last updated: 2026-02-12

| Entity | Benchmark | Docs | TP | FP | FN | Precision | Recall | F1 | Delta | Perfect |
|--------|-----------|------|----|----|----|-----------|--------|----|-------|---------|
| Disease | BC5CDR | 100 | 383 | 18 | 99 | 95.5% | 79.5% | 86.7% | +0.0 | 21/100 |
| Drug | BC5CDR | 100 | 272 | 25 | 69 | 91.6% | 79.8% | 85.3% | -0.9 | 21/100 |
| Drug | CADEC | 311 | 272 | 20 | 22 | 93.2% | 92.5% | 92.8% | -0.2 | 282/311 |
| Gene | NLM-Gene | 46 | 198 | 84 | 26 | 70.2% | 88.4% | 78.3% | +0.0 | 6/46 |
| Gene | RareDisGene | 100 | 93 | 90 | 8 | 50.8% | 92.1% | 65.5% | +0.0 | 44/100 |
| Disease | NLP4RARE | 100 | 280 | 29 | 42 | 90.6% | 87.0% | 88.7% | -0.2 | 52/100 |
| Abbreviation | NLP4RARE | 100 | 16 | 3 | 8 | 84.2% | 66.7% | 74.4% | — | 52/100 |
| Author | PubMed Authors | — | — | — | — | — | — | — | — | — |
| Citation | PubMed Authors | — | — | — | — | — | — | — | — | — |
| Disease | NLP4RARE dev | 20 | 79 | 9 | 10 | 89.8% | 88.8% | 89.3% | — | 10/20 |
| Abbreviation | NLP4RARE dev | 20 | 5 | 0 | 0 | 100.0% | 100.0% | 100.0% | — | 10/20 |
| Disease | NCBI Disease | 73 | — | — | — | — | — | 54.4% | — | 13/73 |
| Feasibility (epi) | Synthetic | 20 | 39 | 32 | 11 | 54.9% | 78.0% | 64.5% | — | — |
| Feasibility (screen) | Synthetic | 20 | — | — | — | — | — | — | — | 100% |
| Feasibility (design) | Synthetic | 20 | — | — | — | — | — | — | — | 97% |

> **Delta** = F1 change since last evaluation run.
> Rows with — need a fresh evaluation run to fill in.

## Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2026-02-12 | Full benchmark run: filled TP/FP/FN for CADEC, NLM-Gene, RareDisGene, NLP4RARE | CADEC Drug F1 92.8%, Gene F1s stable, NLP4RARE combined 100-doc Disease 88.7% |
| 2026-02-11 | Disease cross-ref FP filter, expanded drug/disease lexicons | Disease F1 86.7% stable, Drug F1 85.3% (LLM variance range 85.3-86.2%) |
| 2026-02-11 | Drug FP filter: compound ID, bio entities, author "et al" only | Drug P 87.5%->93.8%, F1 81.5%->86.2% |
| 2026-02-11 | Dataset-aware FP filter, acronym collision filters | Disease F1 83.4%->86.7%, Drug F1 79.0%->86.2% |
| 2026-02-10 | Drug lexicon variants, opioid FP filter, eval alt_name | Drug F1 79.1%->81.2% |
