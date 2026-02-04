# Getting Started

This guide covers installation, configuration, and running the ESE pipeline for the first time.

## Prerequisites

- **Python 3.12+**
- **pip** (Python package manager)
- **git**
- An **Anthropic API key** (for Claude-based validation and vision analysis)

## Clone and Setup

```bash
git clone <repo-url>
cd ese
python -m venv .venv
source .venv/bin/activate
pip install -r corpus_metadata/requirements.txt
```

### scispacy Models

The pipeline uses scispacy for biomedical NER. After installing requirements, download the required model:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz
```

## API Key Setup

The pipeline requires an Anthropic API key for LLM-based validation, feasibility extraction, and vision analysis. Set it via environment variable:

```bash
export ANTHROPIC_API_KEY="your-key"
```

Or add it to a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key
```

The config file comments mention `CLAUDE_API_KEY` as an alternative, but the pipeline code uses `ANTHROPIC_API_KEY` exclusively.

## Lexicon Download

The pipeline uses FlashText for fast lexicon matching against 600K+ biomedical terms. Download the public lexicons before your first run:

```bash
python corpus_metadata/Z_utils/Z10_download_lexicons.py
```

This downloads and converts three public lexicon sets:

| Source | Terms | Purpose |
|--------|-------|---------|
| Meta-Inventory | 104K+ | Clinical abbreviations with 170K senses |
| MONDO | 97K | Unified disease ontology (OMIM, Orphanet, ICD-11 mappings) |
| ChEMBL | 23K | Open drug database with approved drugs |

Lexicon files are stored in the directory specified by `paths.dictionaries` in `config.yaml` (default: `ouput_datasources/`).

## First Pipeline Run

Process all PDFs in the configured input folder:

```bash
python corpus_metadata/orchestrator.py
```

The input folder is set in `G_config/config.yaml` under `paths.pdf_input` (default: `Pdfs/`). You can also override the base path with the `CORPUS_BASE_PATH` environment variable.

## Output

Processing creates a folder per document containing JSON files for each entity type:

```
document/
  abbreviations_document_YYYYMMDD_HHMMSS.json
  diseases_document_YYYYMMDD_HHMMSS.json
  drugs_document_YYYYMMDD_HHMMSS.json
  genes_document_YYYYMMDD_HHMMSS.json
  pharma_document_YYYYMMDD_HHMMSS.json
  authors_document_YYYYMMDD_HHMMSS.json
  citations_document_YYYYMMDD_HHMMSS.json
  feasibility_document_YYYYMMDD_HHMMSS.json
  care_pathways_document_YYYYMMDD_HHMMSS.json
  recommendations_document_YYYYMMDD_HHMMSS.json
  metadata_document_YYYYMMDD_HHMMSS.json
  tables_document_YYYYMMDD_HHMMSS.json
  figures_document_YYYYMMDD_HHMMSS.json
  document_extracted_text_YYYYMMDD_HHMMSS.txt
```

Which files are generated depends on the active extraction preset and the content of the document.

## Extraction Presets

The pipeline supports presets that control which extractors run. Set the preset in `config.yaml` under `extraction_pipeline.preset`:

| Preset | What It Extracts |
|--------|-----------------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | Everything enabled (all entity types, figures, tables, metadata) |
| `minimal` | Abbreviations only |
| `drugs_only` | Drug detection only |
| `diseases_only` | Disease detection only |
| `genes_only` | Gene detection only |
| `abbreviations_only` | Abbreviation extraction only |
| `feasibility_only` | Feasibility extraction only |
| `entities_only` | Drugs + diseases + genes + abbreviations |
| `clinical_entities` | Drugs + diseases only |
| `metadata_only` | Authors + citations + document metadata |
| `images_only` | Tables + figures/visuals |
| `tables_only` | Table extraction only (no figures) |

Example configuration:

```yaml
extraction_pipeline:
  preset: "standard"
```

## Running Tests

```bash
cd corpus_metadata && python -m pytest K_tests/ -v
```

## Type Checking

```bash
mypy corpus_metadata
```

## Linting

```bash
ruff check corpus_metadata
```

## Next Steps

- [Configuration Guide](03_configuration.md) for detailed settings and tuning options
- [Architecture Overview](../architecture/01_overview.md) for understanding the pipeline layers and data flow
