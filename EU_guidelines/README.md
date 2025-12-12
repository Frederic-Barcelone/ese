# EU Pharmaceutical Guidelines Scraper & RAG Content Extractor

## Complete Toolkit for Clinical Trials & Rare Disease Drug Development Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Components](#components)
   - [Main Scraper (v7)](#main-scraper-v7)
   - [RAG Content Extractor (v4)](#rag-content-extractor-v4)
6. [Output Structure](#output-structure)
7. [Configuration](#configuration)
8. [Sources Covered](#sources-covered)
9. [Regulatory Framework Reference](#regulatory-framework-reference)
10. [Troubleshooting](#troubleshooting)
11. [Version History](#version-history)

---

## Overview

This toolkit provides automated downloading and extraction of EU pharmaceutical regulatory documents for:

- **Clinical Trials**: Regulation (EU) No 536/2014 and supporting guidelines
- **Orphan Medicinal Products**: Regulation (EC) No 141/2000 and related guidance  
- **Good Clinical Practice (GCP)**: ICH E6(R2) and EU implementation
- **Scientific Guidelines**: EMA efficacy and safety guidelines for drug development

### Use Cases

- Building RAG (Retrieval-Augmented Generation) systems for regulatory Q&A
- Creating searchable document repositories
- Regulatory intelligence and compliance monitoring
- Training materials for regulatory affairs teams

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Source Scraping** | EC Health Portal, EMA, EUR-Lex |
| **PDF Downloads** | Direct download of guideline documents |
| **RAG-Ready Markdown** | Clean text extraction for LLM applications |
| **JavaScript Rendering** | Playwright integration for dynamic EMA pages |
| **Smart Resume** | Survives interruptions, skips existing files |
| **Content Deduplication** | Hash-based duplicate detection |
| **English-Only Filtering** | Excludes non-English language variants |
| **Comprehensive Logging** | Detailed progress and error tracking |

---

## Installation

### Requirements

- Python 3.8+
- pip (Python package manager)
- ~500MB disk space for full corpus

### Dependencies

```bash
# Core dependencies
pip install requests beautifulsoup4 lxml

# For RAG extractor with EMA support (recommended)
pip install playwright
playwright install chromium
```

### Quick Install

```bash
# Clone or download the scripts
cd /path/to/EU_guidelines

# Install dependencies
pip install requests beautifulsoup4 lxml playwright
playwright install chromium

# Verify installation
python eu_pharma_guidelines_scraper.py --help
```

---

## Quick Start

### Basic Usage

```bash
# Run full scraper (PDFs + RAG content)
python eu_pharma_guidelines_scraper.py

# Run RAG extractor only
python eu_pharma_rag_extractor.py
```

### Expected Output

```
============================================================
SCRAPE COMPLETE
============================================================
  Downloaded:      273
  Skipped:        2116
  Failed:           19
  Pages Crawled:    78
  Total Size:    171.31 MB
  Output:        /path/to/EU_guidelines_library
============================================================
```

---

## Components

### Main Scraper (v7)

**File**: `eu_pharma_guidelines_scraper.py`

The main scraper handles:
- PDF document downloads from all sources
- Link discovery and recursive crawling
- EUR-Lex legislation retrieval
- State persistence for resume capability
- Orchestration of RAG extraction

**Key Configuration** (edit `CONFIG` dict at top of file):

```python
CONFIG = {
    "output_dir": "EU_guidelines_library",
    "include_eudralex_vol10": True,
    "include_ec_orphan": True,
    "include_known_pdfs": True,        # ICH guidelines, etc.
    "include_eurlex": True,
    "include_ema_guidelines": True,
    "include_ema_orphan": True,
    "include_deep_crawl": True,
    "include_rag_extraction": True,    # Enable RAG markdown extraction
    "skip_existing": True,             # Skip already downloaded files
}
```

### RAG Content Extractor (v4)

**File**: `eu_pharma_rag_extractor.py`

Extracts clean markdown content from HTML pages for RAG systems.

**Key Features**:
- **Hybrid fetching**: HTTP for EC/EUR-Lex, Playwright for EMA
- **Content quality filters**: Minimum length, word count thresholds
- **YAML frontmatter**: Structured metadata in each file
- **Smart deduplication**: Content hash comparison

**Extraction Results** (typical run):

| Source | Pages Extracted |
|--------|-----------------|
| EC Health Portal (EudraLex) | ~40 |
| EC Clinical Trials | ~1 |
| EMA Orphan Designation | ~25 |
| EMA Scientific Guidelines | ~68 |
| EMA Clinical Trials | ~28 |
| EUR-Lex | ~7 |
| **Total** | **~169 pages** |

---

## Output Structure

```
EU_guidelines_library/
│
├── pdfs/                              # Downloaded PDF documents
│   ├── [flat structure with descriptive names]
│   ├── ICH_E6_R2_Good_Clinical_Practice.pdf
│   ├── Regulation_EU_536_2014_Clinical_Trials.pdf
│   └── ...
│
├── rag_content/                       # Markdown files for RAG
│   ├── metadata/
│   │   └── extractor_state_v4.json    # Extraction state
│   ├── EC_EudraLex_Volume_10_Clinical_trials_guidelines_f139e1.md
│   ├── EMA_Orphan_designation_Overview_704b0c.md
│   ├── EMA_Scientific_guidelines_ffef7c.md
│   ├── EURLEX_Regulation_ff2b99.md
│   └── ...
│
├── metadata/
│   ├── scraper_state.json             # Scraper state for resume
│   ├── full_catalog.json              # Complete document catalog
│   └── summary.json                   # Statistics
│
└── scraper.log                        # Detailed log file
```

### Markdown File Format

Each RAG markdown file includes YAML frontmatter:

```markdown
---
title: "Orphan designation Overview"
source: "ema_orphan"
url: "https://www.ema.europa.eu/en/human-regulatory-overview/orphan-designation-overview"
extracted_at: "2025-11-29T10:16:29"
word_count: 1250
---

# Orphan designation Overview

*Orphan designation is granted to medicines intended to diagnose, prevent or treat rare diseases...*

## What is orphan designation?

Orphan designation (also known as orphan status) is granted by the European Commission...
```

---

## Configuration

### Main Scraper Options

| Option | Default | Description |
|--------|---------|-------------|
| `output_dir` | `EU_guidelines_library` | Output directory path |
| `include_eudralex_vol10` | `True` | Scrape EudraLex Volume 10 |
| `include_ec_orphan` | `True` | Scrape EC orphan medicinal products |
| `include_known_pdfs` | `True` | Download ICH and key EMA PDFs |
| `include_eurlex` | `True` | Download EUR-Lex legislation |
| `include_ema_guidelines` | `True` | Scrape EMA scientific guidelines |
| `include_ema_orphan` | `True` | Scrape EMA orphan designation pages |
| `include_deep_crawl` | `True` | Follow links to discover more docs |
| `include_rag_extraction` | `True` | Extract markdown for RAG |
| `skip_existing` | `True` | Skip already downloaded files |
| `force_fresh_start` | `False` | Ignore previous state |

### RAG Extractor Options

| Option | Default | Description |
|--------|---------|-------------|
| `min_content_length` | `200` | Minimum characters to save |
| `min_word_count` | `50` | Minimum words to save |
| `english_only` | `True` | Filter non-English pages |
| `dedupe_by_content` | `True` | Skip duplicate content |
| `ema_delay` | `2.0` | Seconds between EMA requests |
| `eurlex_delay` | `3.0` | Seconds between EUR-Lex requests |
| `max_depth` | `2` | Link following depth |
| `max_pages_per_source` | `100` | Max pages per source |

---

## Sources Covered

### European Commission Health Portal

| Section | URL | Content |
|---------|-----|---------|
| EudraLex Volume 10 | health.ec.europa.eu/eudralex-volume-10 | Clinical trials guidelines |
| Orphan Medicinal Products | health.ec.europa.eu/orphan-medicinal-products | EC orphan drug info |
| Clinical Trials | health.ec.europa.eu/clinical-trials | CTR overview |

### European Medicines Agency (EMA)

| Section | Content | Method |
|---------|---------|--------|
| Orphan Designation | Process, incentives, COMP | Playwright |
| Scientific Guidelines | ICH, quality, clinical efficacy | Playwright |
| Clinical Trials | CTIS, CTR implementation | Playwright |
| GCP | Good Clinical Practice guidance | Playwright |

### EUR-Lex

| Document | Reference |
|----------|-----------|
| Clinical Trials Regulation | Regulation (EU) No 536/2014 |
| Orphan Regulation | Regulation (EC) No 141/2000 |
| GCP Directive | Directive 2005/28/EC |
| Paediatric Regulation | Regulation (EC) No 1901/2006 |

---

## Regulatory Framework Reference

### Clinical Trials Regulation (CTR)

**Regulation (EU) No 536/2014** - Full application since 31 January 2022

| Feature | Description |
|---------|-------------|
| Single Submission | One application via CTIS for multinational trials |
| Harmonized Assessment | Joint evaluation by participating Member States |
| Transparency | Public database of all EU clinical trials |
| Timeline | Part I: 45 days, Part II: varies by Member State |

### Orphan Medicinal Products

**Regulation (EC) No 141/2000** - In force since 28 April 2000

**Orphan Designation Criteria**:
1. **Rarity**: Prevalence ≤ 5 in 10,000 in EU
2. **Seriousness**: Life-threatening or chronically debilitating
3. **Unmet Need**: No satisfactory treatment OR significant benefit
4. **Medical Plausibility**: Scientific rationale for efficacy

**Incentives**:
- 10 years market exclusivity (12 if paediatric)
- Protocol assistance (free/reduced fee)
- Fee reductions for SMEs
- Centralized authorization

### Key Committees

| Committee | Abbreviation | Function |
|-----------|--------------|----------|
| Committee for Medicinal Products for Human Use | CHMP | Marketing authorization |
| Committee for Orphan Medicinal Products | COMP | Orphan designation |
| Paediatric Committee | PDCO | Paediatric investigation plans |
| Committee for Advanced Therapies | CAT | ATMPs |

---

## Troubleshooting

### Common Issues

#### Playwright not installed

```
ERROR - Playwright not installed! Run: pip install playwright && playwright install chromium
```

**Solution**:
```bash
pip install playwright
playwright install chromium
```

#### EMA pages returning empty content

**Symptom**: `Content too short (30 chars)` for EMA URLs

**Cause**: EMA uses JavaScript rendering - requires Playwright

**Solution**: Ensure Playwright is installed and `use_playwright: True` for EMA sources

#### XML parsing warnings

```
XMLParsedAsHTMLWarning: It looks like you're using an HTML parser to parse an XML document.
```

**Solution**: Update to latest `eu_pharma_rag_extractor.py` which suppresses this warning

#### Rate limiting / 403 errors

**Symptom**: Repeated 403 Forbidden errors

**Solution**: 
- Increase delay settings in CONFIG
- Wait and retry later
- Check if IP is temporarily blocked

### Log Files

- `scraper.log` - Main scraper detailed log
- `rag_content/rag_extractor_v4.log` - RAG extractor log

---

## Version History

| Version | Date | Component | Changes |
|---------|------|-----------|---------|
| v7.0 | Nov 2025 | Main Scraper | Recursive crawling, state persistence, RAG integration |
| v4.0 | Nov 2025 | RAG Extractor | Playwright for EMA, hybrid HTTP/browser approach |
| v3.0 | Nov 2025 | RAG Extractor | Improved EMA selectors (HTTP only) |
| v2.0 | Nov 2025 | RAG Extractor | English filtering, content deduplication |
| v1.0 | Nov 2025 | Initial | Basic scraping functionality |

---

## Useful Links

| Resource | URL |
|----------|-----|
| CTIS (Clinical Trials) | https://euclinicaltrials.eu |
| EMA | https://www.ema.europa.eu |
| EC Health | https://health.ec.europa.eu |
| EUR-Lex | https://eur-lex.europa.eu |
| EudraVigilance | https://eudravigilance.ema.europa.eu |
| Orphanet | https://www.orpha.net |

---

## Glossary

| Term | Definition |
|------|------------|
| **CTR** | Clinical Trials Regulation (EU) No 536/2014 |
| **CTIS** | Clinical Trials Information System |
| **COMP** | Committee for Orphan Medicinal Products |
| **GCP** | Good Clinical Practice |
| **ICH** | International Council for Harmonisation |
| **IMP** | Investigational Medicinal Product |
| **OMP** | Orphan Medicinal Product |
| **RAG** | Retrieval-Augmented Generation |

---

## Disclaimer

This toolkit is for informational purposes only. Always consult official sources for the most current regulatory information. Guidelines and regulations are subject to change.

**Official Sources**:
- European Commission: https://health.ec.europa.eu
- European Medicines Agency: https://www.ema.europa.eu  
- EUR-Lex: https://eur-lex.europa.eu

---

*Last updated: November 2025*  
*Main Scraper: v7.0 | RAG Extractor: v4.0*