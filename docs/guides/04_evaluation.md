# Evaluation Guide

## Overview

`F_evaluation/` compares pipeline output against gold standard annotations, computing precision, recall, and F1 per entity type.

**Two evaluation paths:**

- **F03_evaluation_runner.py** (recommended) -- Self-contained runner for abbreviations, diseases, genes, and drugs.
- **F01_gold_loader.py + F02_scorer.py** -- Reusable API for **abbreviation-only** evaluation via `GoldLoader` and `Scorer` classes.

F03 does **not** use F01/F02 internally -- it has its own comparison functions.

> Other entity types (pharma, authors, citations, care pathways, recommendations) do not yet have gold standard evaluation.

## Gold Standard Data

```
gold_data/
  papers_gold_v2.json        # Papers dataset (abbreviations)
  nlp4rare_gold.json         # NLP4RARE (abbreviations, diseases, genes)
  nlm_gene_gold.json         # NLM-Gene (genes)
  raredis_gene_gold.json     # RareDisGene (genes)
  PAPERS/                    # PDF files for papers dataset
  NLP4RARE/                  # PDF files (dev/test/train splits)
  CADEC/                     # Drug adverse event corpus
    generate_cadec_gold.py
    evaluate_cadec_drugs.py
```

### Gold Format

**Abbreviations** (papers_gold_v2.json):
```json
{"defined_annotations": [{"doc_id": "doc.pdf", "short_form": "PAH", "long_form": "pulmonary arterial hypertension"}]}
```

**Multi-entity** (nlp4rare_gold.json):
```json
{
  "abbreviations": {"annotations": [{"doc_id": "doc.pdf", "short_form": "TNF", "long_form": "Tumor Necrosis Factor"}]},
  "diseases": {"annotations": [{"doc_id": "doc.pdf", "text": "pulmonary arterial hypertension", "type": "RAREDISEASE"}]},
  "genes": {"annotations": [{"doc_id": "doc.pdf", "symbol": "BMPR2"}]}
}
```

## Running Evaluation

```bash
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
```

### F03 Configuration

Edit the top of `F03_evaluation_runner.py`:

```python
RUN_NLP4RARE = False       # NLP4RARE rare disease corpus
RUN_PAPERS = False         # Papers in gold_data/PAPERS/
RUN_NLM_GENE = False       # NLM-Gene corpus (PubMed)
RUN_RAREDIS_GENE = False   # RareDisGene (rare disease genes)

NLP4RARE_SPLITS = ["dev", "test", "train"]
NLM_GENE_SPLITS = ["test"]
RAREDIS_GENE_SPLITS = ["test"]
MAX_DOCS = None            # None = all
```

**Important:** Revert config flags to defaults after evaluation runs. Exit code 1 is expected (means accuracy < 100%).

### Scorer API (Abbreviations Only)

```python
from F_evaluation.F02_scorer import Scorer, ScorerConfig
from F_evaluation.F01_gold_loader import GoldLoader

loader = GoldLoader(strict=False)
gold, by_doc = loader.load_json("gold_data/papers_gold_v2.json")
config = ScorerConfig(require_long_form_match=True, fuzzy_long_form_match=True, fuzzy_threshold=0.8)
scorer = Scorer(config)
report = scorer.evaluate_doc(extracted_entities, gold_annotations)
```

## Matching Logic

**Abbreviations**: `(short_form, long_form)` pairs. SFs uppercased and hyphen-normalized. LFs support exact, substring, and fuzzy matching.

**Diseases**: 6-step matching -- exact, substring, token overlap, synonym group, synonym normalization, fuzzy (0.8). Compares against both `matched_text` and `preferred_label`.

**Drugs**: Exact, substring, fuzzy (0.8), and brand/generic equivalence (e.g., acetaminophen <-> Tylenol).

**Genes**: Exact symbol (uppercase), matched text vs gold symbol, substring (min 3 chars), and name-based matching.

## Benchmark Results

| Benchmark | Entity | Docs | P | R | F1 |
|-----------|--------|------|---|---|----|
| CADEC | Drugs | 311 | 93.5% | 92.9% | 93.2% |
| NLP4RARE | Diseases | 1,040 | 75.1% | 74.0% | 74.6% |
| NLP4RARE | Abbreviations | 1,040 | 47.5% | 88.8% | 61.9% |
| NLM-Gene | Genes | 290 | -- | -- | -- |
| RareDisGene | Genes | 3,976 | -- | -- | -- |

### Running Each Benchmark

**NLP4RARE** (diseases + abbreviations + genes):
```bash
# Configure: RUN_NLP4RARE=True, NLP4RARE_SPLITS=["dev","test","train"]
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
```

**CADEC** (drugs -- standalone evaluator):
```bash
cd corpus_metadata && python ../gold_data/CADEC/evaluate_cadec_drugs.py --split=test
```

**NLM-Gene** / **RareDisGene** (genes):
```bash
# Configure: RUN_NLM_GENE=True or RUN_RAREDIS_GENE=True, MAX_DOCS=10
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
```

### Benchmark Details

**CADEC**: Social media adverse drug event corpus (AskaPatient). 1,248 docs (937 train, 311 test), 1,198 drug annotations.

**NLP4RARE**: Rare disease corpus with BRAT annotations. 2,311 PDFs across dev/test/train. Evaluates abbreviations, diseases (RAREDISEASE, DISEASE, SKINRAREDISEASE), and genes.

**NLM-Gene**: BioC XML corpus filtered to Entrez-HGNC mappings. 550 PubMed abstracts (100 test, 450 train), 1,266 gene annotations.

**RareDisGene**: Gene-RD-Provenance V2 from Figshare (CC0). 4,725 gene-disease associations filtered to pipeline lexicon. ~3,976 PMIDs.

Per-benchmark deep dives: [Drug Evaluation](07_drug_evaluation.md), [Gene Evaluation](06_gene_evaluation.md)

## Improving Results

**Low Precision (FPs):** Tighten FP filters (`C24`, `C25`, `C34`), adjust confidence thresholds, add blacklist terms.

**Low Recall (FNs):** Add lexicon terms, add generator strategies, lower confidence thresholds, check for over-aggressive FP filtering.

**Key files per entity:**
- Abbreviations: `C01_strategy_abbrev.py`, `C04_strategy_flashtext.py`, `D_validation/`
- Diseases: `C06_strategy_disease.py`, `C24_disease_fp_filter.py`, `E03_disease_normalizer.py`
- Genes: `C16_strategy_gene.py`, `C34_gene_fp_filter.py`
- Drugs: `C07_strategy_drug.py`, `C25_drug_fp_filter.py`, `C26_drug_fp_constants.py`

## Related

- [Drug Evaluation](07_drug_evaluation.md)
- [Gene Evaluation](06_gene_evaluation.md)
- [Configuration Guide](03_configuration.md)
