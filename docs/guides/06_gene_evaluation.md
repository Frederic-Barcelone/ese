# Gene Detection Evaluation

## Benchmarks

Two gene benchmarks evaluate the pipeline's HGNC symbol extraction:

| Benchmark | Source | Annotations | Docs | Domain |
|-----------|--------|-------------|------|--------|
| **NLM-Gene** | NLM-Gene BioC XML corpus | 1,266 | 290 | General biomedical (PubMed) |
| **RareDisGene** | Figshare Gene-RD-Provenance V2 | 4,117 | 3,976 | Rare disease gene-disease associations |

Both benchmarks use HGNC symbols as the gold standard, matching the pipeline's extraction target. The previous benchmark (BioCreative II GM) was removed because it annotated broad gene/protein names rather than HGNC symbols (only 6.3% schema match).

### Benchmark A: NLM-Gene

**Source:** NLM-Gene corpus (550 BioC XML files from PubMed abstracts)

**Filtering:**
- Only `type=Gene` annotations (excludes GENERIF, STARGENE, Other, Domain)
- Excludes `code=222` (family names: "cytokine", "transcription factor")
- Excludes `code=333` (non-specific: "IFN")
- Only annotations with NCBI Gene IDs mapping to HGNC via Orphadata
- Deduplicated: unique (doc_id, HGNC symbol) pairs per document

**Splits:** 100 test / 450 train PMIDs (pre-defined by corpus)

**Gold generation:**
```bash
python gold_data/nlm_gene/generate_nlm_gene_gold.py
python gold_data/nlm_gene/generate_nlm_gene_gold.py --split=test --no-pdf
```

### Benchmark B: RareDisGene

**Source:** Gene-RD-Provenance V2 from Figshare (CC0 license)
- 4,725 gene-disease associations with HGNC symbols linked to rare diseases via PubMed IDs
- Filtered to symbols in pipeline lexicon (~4,100 Orphadata genes)

**Gold standard:** For PMID X, the pipeline should find gene Y (HGNC symbol from TSV).

**Splits:** 80/20 train/test (seed=42), ~796 test / ~3,181 train PMIDs

**Gold generation:**
```bash
python gold_data/raredis_gene/generate_raredis_gene_gold.py
python gold_data/raredis_gene/generate_raredis_gene_gold.py --max-docs=50 --no-pdf
```

## Running Evaluation

Edit `F_evaluation/F03_evaluation_runner.py` configuration:

```python
RUN_NLM_GENE = True       # Enable NLM-Gene benchmark
RUN_RAREDIS_GENE = True   # Enable RareDisGene benchmark
NLM_GENE_SPLITS = ["test"]
RAREDIS_GENE_SPLITS = ["test"]
MAX_DOCS = 10             # Small batch for testing
```

Then run:
```bash
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
```

**Important:** Revert `RUN_NLM_GENE` and `RUN_RAREDIS_GENE` to `False` and `MAX_DOCS` to `None` after evaluation.

## Pipeline Architecture

Gene detection follows the standard ESE three-layer strategy:

```
C_generators/C16_strategy_gene.py    -> High recall candidate generation
C_generators/C34_gene_fp_filter.py   -> False positive filtering
E_normalization/                     -> HGNC symbol normalization
```

The gene generator uses FlashText lexicon matching against HGNC-approved symbols and names from the Orphadata gene-disease association dataset. Candidates are filtered through the gene FP filter (context-aware short symbol handling, common abbreviation filtering) and validated via LLM before export.

### FlashText Layers

| Layer | Terms | Source |
|-------|-------|--------|
| Orphadata genes | ~4,100 | `ouput_datasources/2025_08_orphadata_genes.json` |
| HGNC aliases | ~9,000 | Same file (hgnc_alias entries) |

### Gene FP Filter (C34)

The filter handles gene symbol ambiguity:
- **Always-filter terms**: Statistical (OR, HR, CI), units (MM, KG), clinical (IV, ICU, CT), countries (US, UK), credentials (MD, PhD), drug terms (ACE, NSAID)
- **Short gene context**: Genes <= 3 characters require gene context keywords (mutation, variants, genetic)
- **Special handling**: EGFR/eGFR disambiguation, antibody terms (ANA, ACPA), disease abbreviation overlap

## Gene Matching in F03

The evaluation uses multi-step matching via `gene_matches()`:

1. **Exact symbol match** -- Uppercase comparison (e.g., "BRCA1" == "BRCA1")
2. **Matched text vs gold symbol** -- Raw extracted text against gold
3. **Substring match** -- Handles "BRCA1 gene" vs "BRCA1" (min 3 chars)
4. **Name-based match** -- Full gene name vs gold symbol

## Related Documentation

- [Evaluation Guide](04_evaluation.md) for general evaluation framework usage
- [Adding Entity Types](02_adding_entity_type.md) for the full entity pipeline pattern
- [Configuration Guide](03_configuration.md) for pipeline settings
