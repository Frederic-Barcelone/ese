# Layer F: Gold Standard Evaluation

## Purpose

End-to-end evaluation against human-annotated gold standard corpora. Measures precision, recall, and F1 for abbreviations, diseases, drugs, and genes. 3 modules.

---

## Modules

### F01_gold_loader.py

Loads gold annotations from JSON or CSV.

- `GoldAnnotation`: `doc_id`, `short_form`, `long_form`, `category`
- `GoldStandard`: Collection of annotations
- `GoldLoader`: Supports strict and lenient modes

Normalization: `doc_id` stripped to filename, `short_form` uppercased, `long_form` whitespace-normalized, duplicates removed.

Field aliases: `doc_id`/`filename`/`doc`/`file`, `short_form`/`short`/`sf`/`abbr`, `long_form`/`long`/`lf`/`expansion`.

### F02_scorer.py

Set-based matching on `(short_form, long_form)` pairs.

- `ScorerConfig`: `require_long_form_match`, `fuzzy_long_form_match`, `fuzzy_threshold` (0.8), `only_validated`, `include_field_types`
- `ScoreReport`: `precision`, `recall`, `f1`, `true_positives`, `false_positives`, `false_negatives`
- `CorpusScoreReport`: `micro` (global), `macro` (per-doc average), `per_doc`
- `Scorer`: Matching: SF uppercased with hyphens removed; LF lowercased with whitespace collapsed; fuzzy via substring then SequenceMatcher.

### F03_evaluation_runner.py

Unified runner processing PDFs against gold corpora.

- Data classes: `GoldAbbreviation`, `GoldDisease`, `GoldDrug`, `GoldGene`, `EntityResult`, `DocumentResult`, `DatasetResult`
- Key functions: `load_nlp4rare_gold()`, `compare_abbreviations()`, `compare_diseases()`, `compare_drugs()`, `compare_genes()`, `evaluate_dataset()`, `main()`
- Config constants: `RUN_NLP4RARE`, `RUN_PAPERS`, `RUN_NLM_GENE`, `RUN_RAREDIS_GENE`, `EVAL_ABBREVIATIONS`, `EVAL_DISEASES`, `EVAL_DRUGS`, `EVAL_GENES`, `NLP4RARE_SPLITS`, `NLM_GENE_SPLITS`, `RAREDIS_GENE_SPLITS`, `FUZZY_THRESHOLD=0.8`, `MAX_DOCS`

## Usage

```bash
python corpus_metadata/F_evaluation/F03_evaluation_runner.py
```

## Datasets

- **NLP4RARE** -- Rare disease documents with dev/test/train splits. Covers abbreviations, diseases, genes.
- **NLM-Gene** -- PubMed abstracts with gene annotations (Entrez->HGNC mapped).
- **RareDisGene** -- Rare disease gene-disease associations with HGNC symbols.
- **PAPERS** -- Research papers with human-annotated abbreviations only.
