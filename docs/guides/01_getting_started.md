# Getting Started

## Prerequisites

- Python 3.12+, pip, git
- Anthropic API key (for LLM validation and vision analysis)

## Setup

```bash
git clone <repo-url>
cd ese
python -m venv .venv
source .venv/bin/activate
pip install -r corpus_metadata/requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_sm-0.5.5.tar.gz
```

## API Key

```bash
export ANTHROPIC_API_KEY="your-key"
# Or add to .env in project root: ANTHROPIC_API_KEY=your-key
```

## Lexicon Download

```bash
python corpus_metadata/Z_utils/Z10_download_lexicons.py
```

| Source | Terms | Purpose |
|--------|-------|---------|
| Meta-Inventory | 104K+ | Clinical abbreviations (170K senses) |
| MONDO | 97K | Unified disease ontology |
| ChEMBL | 23K | Open drug database |

Stored in `ouput_datasources/` (configurable via `paths.dictionaries` in `config.yaml`).

## Run

```bash
python corpus_metadata/orchestrator.py
```

Input folder: `G_config/config.yaml` under `paths.pdf_input` (default: `Pdfs/`). Override with `CORPUS_BASE_PATH` env var.

## Output

Per-document folder with timestamped JSON files per entity type. Which files appear depends on the active preset and document content.

## Extraction Presets

Set in `config.yaml` under `extraction_pipeline.preset`:

| Preset | Extracts |
|--------|----------|
| `standard` | Drugs, diseases, genes, abbreviations, feasibility, tables, figures, care pathways, recommendations |
| `all` | Everything |
| `minimal` | Abbreviations only |
| `drugs_only` / `diseases_only` / `genes_only` / `abbreviations_only` / `feasibility_only` | Single entity type |
| `entities_only` | Drugs + diseases + genes + abbreviations |
| `clinical_entities` | Drugs + diseases |
| `metadata_only` | Authors + citations + document metadata |
| `images_only` | Tables + figures/visuals |
| `tables_only` | Tables only |

## Verification

```bash
cd corpus_metadata && python -m pytest K_tests/ -v
cd corpus_metadata && python -m mypy .
cd corpus_metadata && python -m ruff check .
```

## Next Steps

- [Configuration Guide](03_configuration.md)
- [Architecture Overview](../architecture/01_overview.md)
