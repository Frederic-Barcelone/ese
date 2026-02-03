# F_evaluation -- Gold Standard Evaluation

## Purpose

Layer F provides end-to-end evaluation of the extraction pipeline against human-annotated gold standard corpora. It measures precision, recall, and F1 for abbreviations, diseases, and genes, targeting 100% accuracy on curated datasets.

## Modules

### F01_gold_loader.py

Loads gold standard annotations from JSON or CSV files, normalizing and indexing them for efficient lookup.

**Classes:**

- `GoldAnnotation(BaseModel)` -- Single annotation with `doc_id`, `short_form`, `long_form`, `category`. Pydantic model with `extra="ignore"`.
- `GoldStandard(BaseModel)` -- Collection of `GoldAnnotation` items.
- `GoldLoader` -- Main loader class. Supports strict mode (requires all fields) and lenient mode (allows missing `long_form`).

**Public API:**

```python
loader = GoldLoader(strict=False)
gold, by_doc = loader.load_json("gold_standard.json")
gold, by_doc = loader.load_csv("gold_standard.csv", delimiter=",", encoding="utf-8")
```

**Normalization applied:**

- `doc_id`: filename only (strips paths via `Path.name`)
- `short_form`: uppercased
- `long_form`: whitespace-normalized
- Duplicates removed by `(doc_id, SF, LF)` key

**Input formats accepted:**

- `list[dict]` -- flat list of annotations
- `dict` with `"annotations"` key -- v1 format
- `dict` with `"defined_annotations"` key -- v2 format

**Field name aliases:** `doc_id`/`filename`/`doc`/`file`, `short_form`/`short`/`sf`/`abbr`, `long_form`/`long`/`lf`/`expansion`.

### F02_scorer.py

Computes precision, recall, and F1 by comparing system output against gold annotations using set-based matching on `(short_form, long_form)` pairs.

**Classes:**

- `ScorerConfig(BaseModel)` -- Controls matching behavior:
  - `require_long_form_match: bool = True` -- Match by (SF, LF) pair vs SF only
  - `fuzzy_long_form_match: bool = True` -- Enable substring and SequenceMatcher matching
  - `fuzzy_threshold: float = 0.8` -- Minimum similarity ratio
  - `only_validated: bool = True` -- Only score validated entities
  - `allow_sf_only_gold: bool = True` -- Handle gold entries with missing LF
  - `include_field_types: Set[FieldType]` -- Filter by `DEFINITION_PAIR`, `GLOSSARY_ENTRY`
- `ScoreReport(BaseModel)` -- Result container: `precision`, `recall`, `f1`, `true_positives`, `false_positives`, `false_negatives`, `gold_count`, `fp_examples`, `fn_examples`. When `is_scored=False`, metrics are `None` (no gold data available).
- `CorpusScoreReport(BaseModel)` -- Corpus-level: `micro` (global set logic), `macro` (average per doc), `per_doc` dict.
- `Scorer` -- Main scorer class.

**Public API:**

```python
scorer = Scorer(ScorerConfig(require_long_form_match=True))
report = scorer.evaluate_doc(system_output, gold_truth, doc_id="doc.pdf")
corpus_report = scorer.evaluate_corpus(system_output, gold_truth)
scorer.print_summary(report, title="EVALUATION REPORT")
scorer.print_corpus_summary(corpus_report)
```

**Matching logic:**

- SF normalized: uppercased, hyphens/dashes removed (`SC5B-9` matches `SC5B9`)
- LF normalized: lowercased, whitespace-collapsed
- Fuzzy matching: substring containment, then SequenceMatcher ratio >= threshold

### F03_evaluation_runner.py

Unified evaluation runner that processes PDFs against gold standard corpora and reports per-entity-type metrics.

**Data classes (dataclass-based, not Pydantic):**

- `GoldAbbreviation`, `GoldDisease`, `GoldGene` -- Gold standard entries
- `ExtractedAbbreviation`, `ExtractedDisease`, `ExtractedGene` -- System output wrappers
- `EntityResult` -- Per-entity TP/FP/FN with computed `precision`, `recall`, `f1` properties
- `DocumentResult` -- Per-document results across all entity types, with `is_perfect` property
- `DatasetResult` -- Aggregate results for a dataset

**Key functions:**

- `load_nlp4rare_gold(gold_path)` -- Load NLP4RARE gold (abbreviations, diseases, genes)
- `load_papers_gold(gold_path)` -- Load papers gold (abbreviations only, v2 format)
- `compare_abbreviations(extracted, gold, doc_id)` -- Compare with fuzzy LF matching
- `compare_diseases(extracted, gold, doc_id)` -- Compare with substring/fuzzy matching
- `compare_genes(extracted, gold, doc_id)` -- Compare by normalized symbol
- `evaluate_dataset(name, pdf_folder, gold_data, orch, ...)` -- Full dataset evaluation
- `main()` -- Entry point, runs all configured datasets

**Configuration constants:** `RUN_NLP4RARE`, `RUN_PAPERS`, `EVAL_ABBREVIATIONS`, `EVAL_DISEASES`, `EVAL_GENES`, `FUZZY_THRESHOLD=0.8`, `TARGET_ACCURACY=1.0`, `MAX_DOCS`, `NLP4RARE_SPLITS`.

## Usage

```bash
# Run full evaluation
python corpus_metadata/F_evaluation/F03_evaluation_runner.py

# Programmatic usage
from F_evaluation.F01_gold_loader import GoldLoader
from F_evaluation.F02_scorer import Scorer, ScorerConfig

loader = GoldLoader(strict=False)
gold, by_doc = loader.load_json("gold_data/papers_gold_v2.json")

scorer = Scorer(ScorerConfig(fuzzy_long_form_match=True))
report = scorer.evaluate_doc(extracted_entities, by_doc["document.pdf"])
```

## Datasets

- **NLP4RARE** -- Rare disease medical documents with `dev`/`test`/`train` splits. Gold: `gold_data/nlp4rare_gold.json`. Covers abbreviations, diseases, genes.
- **PAPERS** -- Research papers with human-annotated abbreviations. Gold: `gold_data/papers_gold_v2.json`. Abbreviations only.
