# Evaluation Guide

This guide covers how to evaluate the ESE pipeline's extraction accuracy against gold standard annotations.

## Overview

The evaluation framework in `F_evaluation/` compares pipeline output against human-annotated ground truth. It computes precision, recall, and F1 scores per entity type, per document, and across the full corpus.

**Important**: There are two separate evaluation paths:

- **F03_evaluation_runner.py** (recommended) -- Self-contained runner supporting abbreviations, diseases, and genes. Has its own loading, comparison, and reporting logic. This is the primary evaluation tool.
- **F01_gold_loader.py + F02_scorer.py** -- Reusable API for **abbreviation-only** evaluation. Provides `GoldLoader` and `Scorer` classes for programmatic use. Does not support disease or gene evaluation.

The F03 runner does **not** use F01/F02 internally — it has its own comparison functions optimized for multi-entity evaluation.

> **Note**: Evaluation is currently supported for abbreviations, diseases, genes, and drugs. Other entity types (pharma, authors, citations, care pathways, recommendations) do not yet have gold standard evaluation support.

## Gold Standard Data

### Location

Gold standard annotations are stored in the `gold_data/` directory:

```
gold_data/
  papers_gold_v2.json        # Papers dataset (abbreviations)
  nlp4rare_gold.json         # NLP4RARE dataset (abbreviations, diseases, genes)
  golden_bc2gm.json          # BioCreative II GM dataset (genes)
  PAPERS/                    # PDF files for the papers dataset
  NLP4RARE/                  # PDF files for NLP4RARE (dev/test/train splits)
    dev/
    test/
    train/
  bc2gm/                     # BioCreative II GM gene mention corpus
    generate_bc2gm_gold.py   # Gold generation script
    corpus/                  # Downloaded corpus (gitignored)
    pdfs/                    # Generated PDFs (gitignored)
  CADEC/                     # CADEC drug adverse event corpus
    generate_cadec_gold.py   # Gold generation script
    evaluate_cadec_drugs.py  # CADEC-specific drug evaluation
```

### Gold Standard Format

Gold annotations are stored as JSON. The format supports multiple entity types:

**Abbreviation annotations** (papers_gold_v2.json):

```json
{
  "defined_annotations": [
    {
      "doc_id": "document.pdf",
      "short_form": "PAH",
      "long_form": "pulmonary arterial hypertension",
      "category": "Disease"
    }
  ]
}
```

**Multi-entity annotations** (nlp4rare_gold.json):

```json
{
  "abbreviations": {
    "annotations": [
      {"doc_id": "doc.pdf", "short_form": "TNF", "long_form": "Tumor Necrosis Factor"}
    ]
  },
  "diseases": {
    "annotations": [
      {"doc_id": "doc.pdf", "text": "pulmonary arterial hypertension", "type": "RAREDISEASE"}
    ]
  },
  "genes": {
    "annotations": [
      {"doc_id": "doc.pdf", "symbol": "BMPR2", "name": "bone morphogenetic protein receptor type 2"}
    ]
  }
}
```

The `GoldLoader` class (`F01_gold_loader.py`) accepts flexible field names for compatibility:
- `doc_id` / `filename` / `doc` / `file`
- `short_form` / `short` / `sf` / `abbr`
- `long_form` / `long` / `lf` / `expansion`

## Running Evaluation

### Full Evaluation Runner

The primary evaluation entry point runs the pipeline on gold standard documents and compares results:

```bash
python corpus_metadata/F_evaluation/F03_evaluation_runner.py
```

This will:

1. Initialize the pipeline orchestrator.
2. Load gold standard annotations for all configured datasets.
3. Process each PDF through the full extraction pipeline.
4. Compare extracted entities against gold annotations.
5. Print per-document and aggregate metrics.
6. Exit with code 0 if targets are met, 1 otherwise.

### Configuration

Edit the configuration section at the top of `F03_evaluation_runner.py`:

```python
# Which datasets to run
RUN_NLP4RARE = True    # NLP4RARE annotated rare disease corpus
RUN_PAPERS = True      # Papers in gold_data/PAPERS/
RUN_BC2GM = True        # BioCreative II GM gene mention corpus

# Which entity types to evaluate
EVAL_ABBREVIATIONS = True
EVAL_DISEASES = True
EVAL_GENES = True
EVAL_DRUGS = True

# NLP4RARE splits to include
NLP4RARE_SPLITS = ["dev", "test", "train"]

# Limit documents (None = all)
MAX_DOCS = None

# Matching threshold
FUZZY_THRESHOLD = 0.8  # 80% similarity for fuzzy long form matching
```

### Programmatic Evaluation (Scorer API — Abbreviations Only)

For evaluating abbreviation extraction on individual documents or custom datasets, use the `Scorer` class directly. This API compares `(short_form, long_form)` pairs and does **not** support disease or gene evaluation.

```python
from F_evaluation.F02_scorer import Scorer, ScorerConfig
from F_evaluation.F01_gold_loader import GoldLoader

# Load gold standard
loader = GoldLoader(strict=False)
gold, by_doc = loader.load_json("gold_data/papers_gold_v2.json")

# Configure scorer
config = ScorerConfig(
    require_long_form_match=True,
    fuzzy_long_form_match=True,
    fuzzy_threshold=0.8,
    only_validated=True,
)
scorer = Scorer(config)

# Evaluate a single document
report = scorer.evaluate_doc(extracted_entities, gold_annotations)
scorer.print_summary(report)

# Evaluate full corpus
corpus_report = scorer.evaluate_corpus(all_extracted, all_gold)
scorer.print_corpus_summary(corpus_report)
```

## Scoring Metrics

### Definitions

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Fraction of extracted entities that are correct |
| **Recall** | TP / (TP + FN) | Fraction of gold entities that were found |
| **F1** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |

### Classification

- **True Positive (TP)**: System entity matches a gold entity.
- **False Positive (FP)**: System extracted an entity not in the gold standard.
- **False Negative (FN)**: Gold entity not found by the system.

### Matching Modes

**Abbreviation matching** compares `(short_form, long_form)` pairs:
- Short forms are uppercased and hyphen-normalized for comparison.
- Long forms support exact match, substring match, and fuzzy matching (configurable via `fuzzy_threshold`).
- When gold has no long form, matching is SF-only (if `allow_sf_only_gold=True`).

**Disease matching** compares against both `matched_text` (raw document text) and `preferred_label` (normalized ontology name) for better coverage. Supports exact, substring, and fuzzy matching.

**Drug matching** compares drug names via exact, substring, fuzzy (0.8 threshold), and brand/generic equivalence (e.g., acetaminophen ↔ Tylenol). CADEC uses a standalone evaluator (`evaluate_cadec_drugs.py`) with the same matching logic.

**Gene matching** uses multi-step comparison: exact symbol match (uppercase), matched text vs gold symbol, substring match (min 3 chars), and name-based matching. See [Gene Evaluation](06_gene_evaluation.md) for details.

## Score Report

The `ScoreReport` contains:

```python
class ScoreReport:
    is_scored: bool               # False when no gold annotations exist
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    true_positives: int
    false_positives: int
    false_negatives: int
    gold_count: int
    fp_examples: List[str]        # Up to 10 false positive examples
    fn_examples: List[str]        # Up to 10 false negative examples
```

For corpus-level evaluation, `CorpusScoreReport` provides:
- `micro`: Global TP/FP/FN computed across all documents (weighted by entity count).
- `macro`: Average of per-document metrics (equal weight per document).
- `per_doc`: Dictionary of per-document `ScoreReport` objects.

## Interpreting Results

### Low Precision (Too Many False Positives)

The system is extracting entities that should not be there.

**Actions:**
- Tighten false positive filters in `C_generators/` (e.g., `C24_disease_fp_filter.py`, `C25_drug_fp_filter.py`).
- Adjust confidence thresholds in `D_validation/` or `config.yaml` (`validation` section).
- Add terms to the blacklist (`heuristics.sf_blacklist` for abbreviations).
- Review the `fp_examples` in the score report for patterns.

### Low Recall (Missing Entities)

The system is failing to find entities that exist in the document.

**Actions:**
- Add missing terms to lexicon files in `ouput_datasources/`.
- Add new generator strategies in `C_generators/` (e.g., additional regex patterns, new lexicon sources).
- Lower the confidence threshold in `config.yaml` for the relevant entity type.
- Check if entities are being filtered too aggressively by FP filters.
- Review the `fn_examples` in the score report for patterns.

### Per-Entity-Type Breakdown

The evaluation runner reports metrics separately for abbreviations, diseases, and genes. This helps identify which extraction stages need improvement:

- **Abbreviation issues** likely involve `C_generators/C01_strategy_abbrev.py` (syntax), `C04_strategy_flashtext.py` (lexicon), or `D_validation/` (LLM validation).
- **Disease issues** likely involve `C_generators/C06_strategy_disease.py` (detection), `C24_disease_fp_filter.py` (filtering), or `E_normalization/E03_disease_normalizer.py` (normalization).
- **Gene issues** likely involve `C_generators/C16_strategy_gene.py` (detection) or `C34_gene_fp_filter.py` (filtering).
- **Drug issues** likely involve `C_generators/C07_strategy_drug.py` (detection), `C25_drug_fp_filter.py` (filtering), or `C26_drug_fp_constants.py` (filter term lists).

## Benchmark Results

| Benchmark | Entity | Docs | P | R | F1 | Perfect | Status |
|-----------|--------|------|---|---|----|---------|--------|
| CADEC (social media) | Drugs | 311 | 93.5% | 92.9% | 93.2% | 91.3% | Production-ready |
| BC2GM (PubMed) | Genes | 100 | 90.3% | 12.3% | 21.7% | 4.0% | Validated methodology |
| NLP4RARE (rare disease) | Diseases | 1,040 | 77.0% | 74.4% | 75.7% | 30.3% | Active improvement |
| NLP4RARE (rare disease) | Abbreviations | 1,040 | 46.8% | 88.0% | 61.1% | -- | Active improvement |

### Running Each Benchmark

**NLP4RARE** (diseases + abbreviations + genes):
```bash
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
# Configure: RUN_NLP4RARE=True, NLP4RARE_SPLITS=["dev","test","train"]
```

**BC2GM** (genes):
```bash
cd corpus_metadata && python F_evaluation/F03_evaluation_runner.py
# Configure: RUN_BC2GM=True
# Requires: gold_data/bc2gm/corpus/ and gold_data/bc2gm/pdfs/ (see Gene Evaluation guide)
```

**CADEC** (drugs — standalone evaluator):
```bash
cd corpus_metadata && python ../gold_data/CADEC/evaluate_cadec_drugs.py --split=test
# Options: --split=test|train|all, --max-docs=N, --seqeval
```

### Benchmark Details

**CADEC**: Social media adverse drug event corpus (AskaPatient forums). 1,248 documents (937 train, 311 test), 1,198 drug annotations. Evaluates drug detection with brand/generic equivalence mapping (30+ pairs: acetaminophen/Tylenol, atorvastatin/Lipitor, etc.).

**BC2GM**: BioCreative II Gene Mention benchmark. 5,000 PubMed sentences, 6,331 gene annotations. The 12.3% recall reflects deliberate scope — the pipeline targets HGNC symbols, not all gene/protein mentions. See [Gene Evaluation](06_gene_evaluation.md) for details.

**NLP4RARE**: Rare disease corpus with BRAT annotations (UC3M). 2,311 PDFs across dev/test/train splits. Evaluates abbreviations (242 gold pairs), diseases (4,123 gold annotations across RAREDISEASE, DISEASE, SKINRAREDISEASE types), and genes (not annotated in this corpus).

For detailed per-benchmark analysis, see:
- [Gene Evaluation](06_gene_evaluation.md) for BC2GM methodology and FP/FN analysis
- [Drug Evaluation](07_drug_evaluation.md) for CADEC improvement trajectory and error patterns

## Creating Gold Standard Annotations

To create gold annotations for new documents:

1. Place PDF files in a folder under `gold_data/`.
2. Manually annotate each document, recording entities with their types.
3. Create a JSON file following the format described above. Each annotation needs at minimum:
   - `doc_id`: The PDF filename.
   - For abbreviations: `short_form` and `long_form`.
   - For diseases: `text` and optionally `type` (DISEASE, RAREDISEASE).
   - For genes: `symbol` and optionally `name`.
4. Run the evaluation runner to measure pipeline performance against your new annotations.

## Related Documentation

- [Gene Evaluation](06_gene_evaluation.md) for BC2GM gene benchmark details and results
- [Drug Evaluation](07_drug_evaluation.md) for CADEC drug benchmark details and results
- [Architecture Data Flow](../architecture/02_data_flow.md) for understanding how entities flow through pipeline stages
- [Configuration Guide](03_configuration.md) for adjusting extraction parameters
