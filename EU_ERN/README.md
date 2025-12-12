# ERN Resource Scraper v3.2

A Python tool for collecting clinical practice guidelines, resources, and documents from all 24 European Reference Networks (ERNs) - optimized for RAG (Retrieval-Augmented Generation) systems.

## Table of Contents

- [Overview](#overview)
- [What's New in v3.2](#whats-new-in-v32)
- [What are ERNs?](#what-are-erns)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [How Smart State Management Works](#how-smart-state-management-works)
- [How Crawling Works](#how-crawling-works)
- [The 24 ERN Networks](#the-24-ern-networks)
- [EC Methodological Handbooks](#ec-methodological-handbooks)
- [Troubleshooting](#troubleshooting)
- [Legal Considerations](#legal-considerations)
- [Using Output with RAG Systems](#using-output-with-rag-systems)
- [Changelog](#changelog)

---

## Overview

European Reference Networks (ERNs) produce valuable clinical practice guidelines, consensus statements, care pathways, and other clinical resources for rare diseases. However, these resources are scattered across 24 different network websites and are often in HTML format - not ideal for modern AI/RAG systems.

**This tool solves that problem by:**

1. Crawling all 24 ERN network websites
2. Extracting clean text content (no HTML noise)
3. Outputting RAG-ready markdown files with metadata
4. Downloading all linked PDF guidelines
5. Collecting EC methodological handbooks
6. **Smart state management** - automatically re-scrapes when you increase settings

---

## What's New in v3.2

### Smart State Management (Major Feature)

**The scraper now automatically detects when you change crawl settings and re-scrapes networks that need it!**

No more manually deleting state files when you want to crawl deeper.

| Scenario | v3.1 Behavior | v3.2 Behavior |
|----------|---------------|---------------|
| Increase `max_crawl_depth` | Skips all networks | Auto-detects, re-scrapes affected networks |
| Increase `max_pages_per_network` | Skips all networks | Auto-detects, re-scrapes affected networks |
| Upgrade from v3.1 state file | Must delete state manually | Auto-detects legacy state, re-scrapes all |

### Version Comparison

| Feature | v3.0 | v3.1 | v3.2 |
|---------|------|------|------|
| Clean markdown output | ✅ | ✅ | ✅ |
| Config-driven | ❌ | ✅ | ✅ |
| Resume interrupted scrapes | ✅ | ✅ | ✅ |
| Per-network crawl stats | ❌ | ❌ | ✅ |
| Smart settings detection | ❌ | ❌ | ✅ |
| Legacy state migration | ❌ | ❌ | ✅ |

### New Features in v3.2

- **Smart re-scrape detection** - Tracks which networks hit depth/page limits
- **Per-network statistics** - Records pages scraped, max depth reached, limit hits
- **Legacy state handling** - Automatically migrates from v3.1 state files
- **Better logging** - Clear output showing what will be re-scraped and why

---

## What are ERNs?

European Reference Networks are virtual networks of healthcare providers across the EU, designed to tackle rare, low-prevalence, and complex diseases. Established under **Directive 2011/24/EU**, the 24 ERNs were launched in March 2017.

### Key Statistics

| Metric | Value |
|--------|-------|
| Networks | 24 |
| Specialized Centres | 1,613+ |
| Hospitals | 382 |
| Countries | 27 EU + Norway |
| Patients Affected | ~30 million in EU |

### Core Functions

- Virtual multidisciplinary consultations via CPMS platform
- Development of clinical practice guidelines
- Training and education for healthcare professionals
- Registry and research coordination
- Knowledge sharing across borders

---

## Features

### RAG-Ready Output

- **Clean markdown files** with noise removed
- **YAML frontmatter** with metadata (URL, title, date)
- **Structured content** preserving headings, paragraphs, lists
- **PDF link extraction** included in markdown files

### Smart Crawling

- **Configurable depth** - Set `max_crawl_depth` in config (default: 10)
- **Page limits** - Set `max_pages_per_network` in config (default: 100)
- **Automatic sub-page discovery** - Finds guidelines on nested pages
- **Deduplication** - Avoids crawling the same page twice

### Smart State Management (New in v3.2)

- **Tracks crawl statistics** per network (pages scraped, depth reached)
- **Detects settings changes** - knows when you increase limits
- **Selective re-scraping** - only re-scrapes networks that hit previous limits
- **Legacy migration** - handles old state files automatically

### Rate Limiting and Reliability

- **Domain-specific delays** - 8s for EC domains, 1.5s for others
- **Exponential backoff** - Automatic retry on failures
- **Checkpoint recovery** - Resume interrupted scrapes
- **Failed URL tracking** - Avoids retrying known failures

### Comprehensive Collection

- All 24 ERN network websites
- 13 EC methodological handbooks
- Network factsheets
- PDF guidelines and resources

---

## Installation

### Requirements

- Python 3.7+
- Internet connection

### Dependencies

```bash
pip install requests beautifulsoup4 lxml
```

Or create a `requirements.txt`:

```
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
```

Then install:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Basic Usage

```bash
python ern_scraper.py
```

The script will:

1. Create an `EU_ERN_DATA/` directory
2. Download all 13 EC methodological handbooks
3. Crawl enabled ERN networks (7 by default)
4. Extract clean content to markdown files
5. Download PDF guidelines
6. Generate summary reports

### 2. Enable/Disable Networks

Edit `ern_config.json` and set `"scrape": true` or `"scrape": false` for each network:

```json
{
  "networks": {
    "ERN-BOND": {
      "scrape": true,
      ...
    },
    "ERN-CRANIO": {
      "scrape": false,
      ...
    }
  }
}
```

### 3. Change Crawl Settings

Edit `ern_config.json` to adjust crawl depth and page limits:

```json
{
  "scraper_settings": {
    "max_crawl_depth": 20,
    "max_pages_per_network": 500,
    ...
  }
}
```

### 4. View Networks List

Edit `main()` in the script and set:

```python
ACTION = "list"          # List all networks with status
ACTION = "list_enabled"  # List only enabled networks
```

### 5. Resume Interrupted Scrape

Just run the script again - it will skip already processed files:

```bash
python ern_scraper.py
```

---

## Configuration

All configuration is in `ern_config.json`:

### Scraper Settings

```json
{
  "scraper_settings": {
    "max_crawl_depth": 20,
    "max_pages_per_network": 500,
    "default_request_delay": 1.5,
    "request_timeout": 45,
    "max_retries": 3,
    "backoff_base": 5.0,
    "skip_existing": true,
    "verbose": true,
    "output_directory": "EU_ERN_DATA"
  }
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `max_crawl_depth` | Maximum link depth to follow | 10 |
| `max_pages_per_network` | Maximum pages per network | 100 |
| `default_request_delay` | Seconds between requests | 1.5 |
| `request_timeout` | Request timeout in seconds | 45 |
| `max_retries` | Retry attempts on failure | 3 |
| `backoff_base` | Base for exponential backoff | 5.0 |
| `skip_existing` | Skip already downloaded files | true |
| `verbose` | Show detailed logging | true |
| `output_directory` | Output folder name | EU_ERN_DATA |

### Domain-Specific Delays

```json
{
  "domain_delays": {
    "health.ec.europa.eu": 8.0,
    "ec.europa.eu": 8.0,
    "ern-net.eu": 2.0,
    "default": 1.5
  }
}
```

### Recommended Settings by Use Case

| Use Case | max_crawl_depth | max_pages_per_network |
|----------|-----------------|----------------------|
| Quick test | 2 | 20 |
| Standard | 10 | 100 |
| Comprehensive | 20 | 500 |
| Full crawl | 30 | 1000 |

---

## Output Structure

```
EU_ERN_DATA/
│
├── rag_content/              # Clean markdown files (for RAG)
│   ├── ERN-BOND_main.md
│   ├── ERN-BOND_guidelines__1.md
│   ├── ERN-BOND_healthcare_providers__2.md
│   ├── ERKNet_main.md
│   ├── ERKNet_guidelines_pathways_1.md
│   └── ...
│
├── guidelines/               # Downloaded PDF guidelines
│   ├── ERN-BOND_osteogenesis_imperfecta.pdf
│   ├── ERKNet_alport_syndrome_2025.pdf
│   └── ...
│
├── methodologies/            # EC methodological handbooks
│   ├── Handbook_00_Introduction.pdf
│   ├── Handbook_01_Prioritisation.pdf
│   └── ...
│
├── factsheets/               # Network factsheets
│   ├── ERN-BOND_factsheet.pdf
│   ├── ERKNet_factsheet.pdf
│   └── ...
│
├── ern_scraper.log           # Detailed log file
├── ern_scrape_results.json   # Full results in JSON
├── scraper_state.json        # Checkpoint state (for resume)
└── SUMMARY.md                # Human-readable summary
```

### Markdown File Format

Each file in `rag_content/` has this structure:

```markdown
---
url: https://ernbond.eu/guidelines/
title: ERN-BOND - Guidelines
scraped_date: 2025-11-27T14:30:00.000000
---

# ERN-BOND - Guidelines

> ERN BOND connects patients and healthcare providers across Europe...

ERN-BOND focuses on rare bone disorders including osteogenesis imperfecta,
achondroplasia, and other skeletal dysplasias.

## Clinical Practice Guidelines

Guidelines for the diagnosis and management of rare bone diseases...

## Related Documents

- [PDF] OI Guidelines 2024: https://ernbond.eu/wp-content/uploads/...
- [PDF] Achondroplasia Consensus: https://ernbond.eu/wp-content/uploads/...
```

### State File Format (v3.2)

The `scraper_state.json` now includes per-network statistics:

```json
{
  "downloaded_urls": ["https://...", "https://..."],
  "failed_urls": [],
  "scraped_networks": ["ERN-BOND", "ERKNet", "..."],
  "network_stats": {
    "ERN-BOND": {
      "pages_scraped": 127,
      "max_depth_reached": 8,
      "hit_page_limit": false,
      "hit_depth_limit": false,
      "pdfs_found": 45,
      "last_scraped": "2025-11-27T16:30:00",
      "settings_used": {
        "max_crawl_depth": 20,
        "max_pages_per_network": 500
      }
    }
  },
  "settings": {
    "max_crawl_depth": 20,
    "max_pages_per_network": 500
  },
  "last_run": "2025-11-27T16:30:00"
}
```

---

## How Smart State Management Works

### The Problem (v3.1 and earlier)

When you changed crawl settings (e.g., increased depth from 10 to 20), the scraper would skip all previously scraped networks - even if they had hit the old limits and would benefit from deeper crawling.

**Old workaround:** Manually delete `scraper_state.json` to force re-scraping.

### The Solution (v3.2)

The scraper now tracks:
1. **What settings were used** for each network
2. **Whether limits were hit** (page limit, depth limit)
3. **How many pages** were actually scraped

When you increase settings, the scraper automatically:
1. Detects the change
2. Checks which networks hit the old limits
3. Removes only those networks from the "completed" list
4. Re-scrapes them with the new settings

### Example Output

```
[OK] Loaded configuration from: ern_config.json

============================================================
ERN Resource Scraper v3.2 (Smart State Management)
============================================================
Config file: ern_config.json
Output directory: EU_ERN_DATA/
Skip existing: True
Max crawl depth: 20
Max pages/network: 500
Networks enabled: 7
============================================================

2025-11-27 16:01:14 - INFO - Loaded previous state: 458 URLs already processed
2025-11-27 16:01:14 - INFO - [SETTINGS CHANGE] Crawl depth increased: 10 -> 20
2025-11-27 16:01:14 - INFO - [SETTINGS CHANGE] Max pages increased: 100 -> 500
2025-11-27 16:01:14 - INFO - Checking which networks need re-scraping due to increased limits...
2025-11-27 16:01:14 - INFO -   -> ERN-BOND will be RE-SCRAPED: hit page limit (100/100)
2025-11-27 16:01:14 - INFO -   -> ERN-EuroBloodNet will be RE-SCRAPED: near page limit (95/100)
2025-11-27 16:01:14 - INFO -   -> ERKNet OK (didn't hit limits)
2025-11-27 16:01:14 - INFO -   -> ERN-EURO-NMD will be RE-SCRAPED: hit depth limit (10/10)
2025-11-27 16:01:14 - INFO - Marked 3 network(s) for re-scraping
```

### Legacy State Migration

If you upgrade from v3.1, your state file won't have `network_stats`. The scraper detects this and shows:

```
[LEGACY STATE] Detected old state file without network statistics
[LEGACY STATE] Settings increased or unknown - will re-scrape all 7 networks
[LEGACY STATE] Networks to re-scrape:
  -> ERN-BOND
  -> ERN-EuroBloodNet
  -> ERKNet
  -> ERN-EURO-NMD
  -> ERN-GUARD-HEART
  -> MetabERN
  -> ERN-RITA
```

After this first v3.2 run, future runs will have proper statistics.

---

## How Crawling Works

### Multi-Level Discovery

```
[L1] /guidelines/                        (defined in config)
     │
     ├── finds link to /guidelines/bone-fragility/
     ├── finds link to /guidelines/skeletal-dysplasia/
     └── finds link to /guidelines/metabolic-bone/
           │
[L2]       └── /guidelines/metabolic-bone/
           │     │
           │     ├── finds PDF: hypophosphatasia_2024.pdf  --> Downloaded!
           │     └── finds link to /guidelines/metabolic-bone/hpp/
           │
[L3]       └── /guidelines/metabolic-bone/hpp/
                 │
                 └── finds more PDFs...  --> Downloaded!
```

### Log Output Example

```
============================================================
Scraping ERN-BOND: European Reference Network on Rare Bone Diseases
============================================================
Will scrape ERN-BOND: not previously scraped
Fetching main page: https://ernbond.eu
Fetching [L1]: https://ernbond.eu/guidelines/
Fetching [L2]: https://ernbond.eu/guidelines/osteogenesis-imperfecta/
Fetching [L2]: https://ernbond.eu/guidelines/achondroplasia/
Fetching [L3]: https://ernbond.eu/guidelines/osteogenesis-imperfecta/diagnosis/
Crawled 127 pages total (max depth reached: 8)
Found 45 PDF links, 23 other guideline links
Downloaded: ERN-BOND_oi_guidelines_2024.pdf (2,456,789 bytes)
```

---

## The 24 ERN Networks

| # | Network ID | Disease Area | Default |
|---|------------|--------------|---------|
| 1 | ERN-BOND | Rare bone disorders | ✅ Enabled |
| 2 | ERN-CRANIO | Craniofacial anomalies and ENT disorders | ❌ |
| 3 | ERN-EuroBloodNet | Rare haematological diseases | ✅ Enabled |
| 4 | Endo-ERN | Rare endocrine conditions | ❌ |
| 5 | ERN-EpiCARE | Rare and complex epilepsies | ❌ |
| 6 | ERKNet | Rare kidney diseases | ✅ Enabled |
| 7 | ERNICA | Inherited and congenital anomalies | ❌ |
| 8 | ERN-LUNG | Rare respiratory diseases | ❌ |
| 9 | ERN-Skin | Rare skin disorders | ❌ |
| 10 | ERN-EURACAN | Adult solid tumours | ❌ |
| 11 | ERN-eUROGEN | Rare urogenital diseases | ❌ |
| 12 | ERN-EURO-NMD | Neuromuscular diseases | ✅ Enabled |
| 13 | ERN-EYE | Rare eye diseases | ❌ |
| 14 | ERN-GENTURIS | Genetic tumour risk syndromes | ❌ |
| 15 | ERN-GUARD-HEART | Rare cardiac diseases | ✅ Enabled |
| 16 | ERN-ITHACA | Congenital malformations, intellectual disability | ❌ |
| 17 | MetabERN | Hereditary metabolic disorders | ✅ Enabled |
| 18 | ERN-PaedCan | Paediatric haemato-oncology | ❌ |
| 19 | ERN-RARE-LIVER | Rare liver diseases | ❌ |
| 20 | ERN-ReCONNET | Connective tissue and musculoskeletal diseases | ❌ |
| 21 | ERN-RITA | Immunodeficiency, autoinflammatory, autoimmune | ✅ Enabled |
| 22 | ERN-TRANSPLANT-CHILD | Paediatric transplantation | ❌ |
| 23 | VASCERN | Rare vascular diseases | ❌ |
| 24 | ERN-RND | Rare neurological diseases | ❌ |

To enable/disable networks, edit `ern_config.json` and set `"scrape": true` or `"scrape": false`.

---

## EC Methodological Handbooks

The European Commission provides 14 standardized handbooks for clinical practice guideline development:

| # | Handbook | Description |
|---|----------|-------------|
| 0 | Introduction | Overview of ERN CPG/CDST framework |
| 1 | Prioritisation | Prioritizing conditions for guideline development |
| 2 | Appraisal | Evaluating quality of existing guidelines (AGREE II) |
| 3 | Adaptation | Adapting guidelines for ERN use |
| 4 | CPG Development | Full methodology for developing new CPGs |
| 5 | Consensus | Methodology for expert consensus statements |
| 6 | Evidence Reports | Systematic review methodology for rare diseases |
| 7 | Pathways | Diagnostic, monitoring and therapeutic pathways |
| 8 | Protocols | Evidence-based protocol development |
| 9 | Do's and Don'ts | Quick reference factsheets for clinicians |
| 10 | Quality Measures | Quality indicator development |
| 11 | Patient Info | Patient information material development |
| 12 | Implementation | Implementation and evaluation frameworks |
| 13 | Patient Involvement | Guide for patient engagement |

---

## Troubleshooting

### Rate Limiting (429 Errors)

```
429 Client Error: Too Many Requests
```

**Solution:** The scraper handles this automatically with:
- Domain-specific delays (8s for EC domains)
- Exponential backoff (waits 5s, 10s, 20s between retries)
- Up to 3 retry attempts

If errors persist, increase delays in `domain_delays` in the config.

### Missing PDFs

**Problem:** Expected PDF not downloaded.

**Solutions:**
1. Increase `max_crawl_depth` to find deeper pages
2. Check the network's `guidelines_paths` configuration
3. Review the log file for 404 errors

### Empty Markdown Files

**Problem:** Markdown file has little content.

**Cause:** Page content is loaded via JavaScript (not visible to scraper).

**Solution:** These pages cannot be scraped without a headless browser. Consider using Selenium or Playwright for JavaScript-heavy sites.

### Interrupted Downloads

**Solution:** Just re-run the script. It automatically:
- Loads previous state from `scraper_state.json`
- Skips already downloaded files
- Continues from where it stopped

### Networks Not Re-scraping After Settings Change

**Problem:** Increased settings but networks still skipped.

**Possible causes:**
1. **Legacy state file** - If upgrading from v3.1, the script will auto-detect and re-scrape all networks
2. **Networks didn't hit limits** - If a network scraped 50 pages with a limit of 100, increasing to 500 won't trigger re-scrape (it didn't need more)

**To force re-scrape all networks:**
```bash
rm EU_ERN_DATA/scraper_state.json
python ern_scraper.py
```

Or set `"skip_existing": false` in config (will re-download everything).

### Force Fresh Start

```bash
rm EU_ERN_DATA/scraper_state.json
```

Or delete the entire output directory:
```bash
rm -rf EU_ERN_DATA/
```

---

## Legal Considerations

### Copyright and Usage

- Most ERN materials are **publicly funded** and freely available
- Always **cite the source** ERN and publication details
- Check individual document licenses for specific terms

### Attribution Format

When using ERN materials, cite as:

> "This document was developed by [ERN Name] with support from the European Commission."

### Respectful Scraping

This tool is configured to:
- Add **domain-specific delays** between requests (1.5-8 seconds)
- Identify itself with a proper User-Agent
- Respect server responses with **exponential backoff**
- **Not overwhelm servers** with rapid requests

---

## Using Output with RAG Systems

### LangChain Example

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load markdown files
loader = DirectoryLoader('EU_ERN_DATA/rag_content/', glob="**/*.md")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Create vector store
vectorstore = Chroma.from_documents(chunks, embedding_function)
```

### LlamaIndex Example

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load documents
documents = SimpleDirectoryReader('EU_ERN_DATA/rag_content/').load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What are the guidelines for osteogenesis imperfecta?")
```

### Processing PDFs

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load PDF guidelines
pdf_loader = PyPDFDirectoryLoader('EU_ERN_DATA/guidelines/')
pdf_docs = pdf_loader.load()

# Combine with markdown content
all_docs = docs + pdf_docs
```

---

## Changelog

### v3.2.0 (November 2025)

- **NEW**: Smart state management - auto-detects settings changes
- **NEW**: Per-network crawl statistics tracking
- **NEW**: Legacy state file migration from v3.1
- **NEW**: Detailed logging of re-scrape decisions
- **IMPROVED**: Better handling of limit detection
- **FIXED**: Networks now properly re-scrape when limits increase

### v3.1.0 (November 2025)

- **NEW**: Config-driven network selection (`ern_config.json`)
- **NEW**: Enable/disable individual networks via config
- **CHANGED**: Settings moved from script to config file
- **IMPROVED**: Better organization of network metadata

### v3.0.0 (November 2025)

- **NEW**: Clean markdown output (RAG-ready)
- **NEW**: YAML frontmatter with metadata
- **NEW**: Configurable crawl depth
- **NEW**: Page limit per network
- **NEW**: Multi-level link discovery
- **CHANGED**: Output folder renamed to `rag_content/`
- **FIXED**: Corrupted UTF-8 emoji characters
- **IMPROVED**: Content extraction removes noise

### v2.0.0 (2025)

- Domain-specific rate limiting
- Retry logic with exponential backoff
- Updated handbook URLs
- Better network path discovery
- Improved error handling

### v1.0.0 (2025)

- Initial release
- Support for all 24 ERN networks
- EC methodological handbook downloads
- Caching and resume functionality

---

## License

This scraper tool is provided for research and educational purposes. ERN content is subject to individual network and European Commission licensing terms.

---

## Resources

### Official ERN Resources

| Resource | URL |
|----------|-----|
| EC ERN Portal | https://health.ec.europa.eu/rare-diseases-and-european-reference-networks/european-reference-networks_en |
| ERN Search Tool | https://webgate.ec.europa.eu/ernsd/cgi-bin/ern_public.cgi |
| Orphanet | https://www.orpha.net |
| EURORDIS | https://www.eurordis.org |

### Related Projects

| Project | Description |
|---------|-------------|
| ERICA | ERN Integration and Collaboration Activities |
| Solve-RD | Solving rare diseases through genomics |
| EJP RD | European Joint Programme on Rare Diseases |

---

## Files

| File | Description |
|------|-------------|
| `ern_scraper.py` | Main scraper script (v3.2) |
| `ern_config.json` | Configuration file (networks, settings) |
| `README.md` | This documentation |

---

*Updated: November 2025 (v3.2)*