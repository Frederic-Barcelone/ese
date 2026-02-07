# Gene Detection Evaluation

## Benchmarks

Two benchmarks evaluate HGNC symbol extraction:

| Benchmark | Source | Annotations | Docs | Domain |
|-----------|--------|-------------|------|--------|
| **NLM-Gene** | BioC XML corpus | 1,266 | 290 | General biomedical (PubMed) |
| **RareDisGene** | Figshare Gene-RD-Provenance V2 | 4,117 | 3,976 | Rare disease gene-disease associations |

Both use HGNC symbols as gold standard.

### NLM-Gene

550 BioC XML files from PubMed abstracts. Filtered to `type=Gene`, excludes `code=222` (family names) and `code=333` (non-specific), only annotations with NCBI Gene IDs mapping to HGNC via Orphadata. Splits: 100 test / 450 train.

```bash
python gold_data/nlm_gene/generate_nlm_gene_gold.py
```

### RareDisGene

Gene-RD-Provenance V2 from Figshare (CC0). 4,725 gene-disease associations with HGNC symbols linked to rare diseases via PubMed IDs, filtered to pipeline lexicon (~4,100 Orphadata genes). Splits: 80/20 train/test (seed=42), ~796 test / ~3,181 train.

```bash
python gold_data/raredis_gene/generate_raredis_gene_gold.py
```

## Running Evaluation

Edit `F03_evaluation_runner.py`:

```python
RUN_NLM_GENE = True
RUN_RAREDIS_GENE = True
MAX_DOCS = 10
```

```bash
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
```

Revert `RUN_NLM_GENE` and `RUN_RAREDIS_GENE` to `False` and `MAX_DOCS` to `None` after evaluation.

## Pipeline Architecture

```
C16_strategy_gene.py     -> High recall candidate generation (FlashText + Orphadata)
C34_gene_fp_filter.py    -> False positive filtering
E_normalization/         -> HGNC symbol normalization
```

### FlashText Layers

| Layer | Terms | Source |
|-------|-------|--------|
| Orphadata genes | ~4,100 | `2025_08_orphadata_genes.json` |
| HGNC aliases | ~9,000 | `2025_08_orphadata_genes.json` |

### Gene FP Filter (C34)

- **Always-filter**: Statistical (OR, HR, CI), units (MM, KG), clinical (IV, ICU, CT), countries (US, UK), credentials (MD, PhD), drug terms (ACE, NSAID)
- **Short gene context**: Genes <= 3 chars require gene context keywords (mutation, variants, genetic)
- **Special handling**: EGFR/eGFR disambiguation, antibody terms (ANA, ACPA), disease abbreviation overlap

## Gene Matching (F03)

1. **Exact symbol** -- Uppercase comparison
2. **Matched text vs gold** -- Raw extracted text against gold symbol
3. **Substring** -- Min 3 chars
4. **Name-based** -- Full gene name vs gold symbol

## Related

- [Evaluation Guide](04_evaluation.md)
- [Configuration Guide](03_configuration.md)
