# FDA Data Syncer v2.1

## Complete Documentation

A Python-based system for downloading and synchronizing FDA drug data for rare disease research in **Hematology** and **Nephrology** therapeutic areas.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Sources](#data-sources)
6. [Usage](#usage)
7. [Output Structure](#output-structure)
8. [Downloaders Reference](#downloaders-reference)
9. [Therapeutic Areas](#therapeutic-areas)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)
13. [Changelog](#changelog)

---

## Overview

### Purpose

The FDA Syncer downloads comprehensive drug data from the FDA's openFDA API and related sources, focusing on:

- **Drug Labels** - Package inserts, indications, warnings, dosing
- **Adverse Events** - Safety reports and patient outcomes
- **Enforcement Reports** - Recalls and safety alerts
- **Approval Packages** - Complete FDA review documentation (PDFs)

### Therapeutic Focus

| Area | Diseases | Drug Classes |
|------|----------|--------------|
| **Nephrology** | 32 rare kidney diseases | 14 drug classes |
| **Hematology** | 70 rare blood disorders | 20 drug classes |

### Key Features

- ✅ Parallel processing (2-4x speedup)
- ✅ Incremental saves (resume on interruption)
- ✅ Smart drug filtering (excludes cosmetics/OTC)
- ✅ Circuit breaker (prevents API abuse)
- ✅ Timeout protection (no infinite stalls)
- ✅ Connection pool recycling (prevents stale connections)

---

## Architecture

### Directory Structure

```
FDA/
├── sync.py                          # Entry point
├── syncher_keys.py                  # Configuration (API key, mode, settings)
├── syncher_therapeutic_areas.py     # Disease/drug definitions
├── fda_syncher/
│   ├── __init__.py                  # Package init (v2.1.0)
│   ├── orchestrator.py              # Main coordinator
│   ├── downloaders/
│   │   ├── __init__.py
│   │   ├── labels.py                # Drug labels downloader
│   │   ├── adverse_events.py        # Adverse events downloader
│   │   ├── enforcement.py           # Enforcement reports downloader
│   │   └── approval_packages.py     # Approval packages downloader
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py               # Utility functions
│       └── http_client.py           # HTTP client with retry logic
├── FDA_DATA/                        # Output directory
│   ├── labels/
│   ├── adverse_events/
│   ├── enforcement/
│   └── approval_packages/
└── fda_data_quality_check.py        # Data quality checker
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         sync.py                                  │
│                      (Entry Point)                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FDASyncOrchestrator                          │
│                    (orchestrator.py)                            │
│  - Coordinates all downloaders                                  │
│  - Parallel processing with ThreadPoolExecutor                  │
│  - Progress tracking and reporting                              │
└──────┬──────────┬──────────┬──────────┬────────────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Labels  │ │ Adverse  │ │ Enforce- │ │ Approval │
│Downloader│ │ Events   │ │  ment    │ │ Packages │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SimpleHTTPClient                            │
│                     (http_client.py)                            │
│  - Retry logic with exponential backoff                         │
│  - Connection pool recycling                                    │
│  - Adaptive rate limiting                                       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FDA openFDA API                            │
│              https://api.fda.gov/drug/...                       │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Labels Downloaded (sequential)
        │
        ▼
2. Drug Names Extracted & Filtered
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
3a. Adverse Events   3b. Approval Pkgs   3c. Enforcement
    (parallel)           (parallel)          (parallel)
        │                  │                  │
        └──────────────────┴──────────────────┘
                          │
                          ▼
4. Results Saved to FDA_DATA/
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Required Packages

```bash
pip install requests beautifulsoup4 --break-system-packages
```

### Setup Steps

```bash
# 1. Clone or copy the FDA directory
cd /path/to/your/project

# 2. Verify structure
ls FDA/
# Should show: sync.py, syncher_keys.py, syncher_therapeutic_areas.py, fda_syncher/

# 3. Get FDA API Key (optional but recommended)
# Visit: https://open.fda.gov/apis/authentication/
# Add to syncher_keys.py

# 4. Test configuration
python FDA/syncher_keys.py
# Should show: "✅ Configuration is valid!"

# 5. Run sync
cd FDA
python sync.py
```

---

## Configuration

### syncher_keys.py

The main configuration file controlling all sync behavior.

```python
# ============================================================================
# SYNC MODE
# ============================================================================
MODE = 'full'  # Options: 'test', 'daily', 'full'

# ============================================================================
# FDA API KEY
# ============================================================================
FDA_API_KEY = "your-api-key-here"  # Get from https://open.fda.gov/apis/authentication/
# Without key: 240 requests/min, 120,000/day
# With key:    240 requests/min, 240,000/day

# ============================================================================
# BEHAVIOR
# ============================================================================
FORCE_REDOWNLOAD = False  # True = re-download everything, False = skip existing

# ============================================================================
# OUTPUT
# ============================================================================
OUTPUT_DIR = "./FDA_DATA"

# ============================================================================
# THERAPEUTIC AREAS
# ============================================================================
SYNC_AREAS = ['hematology', 'nephrology']  # Only these two are supported
```

### Mode Comparison

| Setting | Test | Daily | Full |
|---------|------|-------|------|
| **Time** | ~15 min | ~1 hour | 6-10 hours |
| **Labels** | 2 diseases, 10 results | All | All |
| **Adverse Events** | Disabled | 90 days, 50 drugs | 90 days, 200 drugs |
| **Approval Packages** | Disabled | Disabled | All |
| **Enforcement** | 30 days, 10 results | 90 days | 365 days |

### Sync Parameters (Full Mode)

```python
SYNC_PARAMETERS = {
    'full': {
        'labels': {
            'enabled': True,
            'max_diseases': None,        # All diseases
            'max_results_per_disease': None
        },
        'integrated_reviews': {          # Approval packages
            'enabled': True,
            'max_drugs': None            # All drugs (capped at 200)
        },
        'adverse_events': {
            'enabled': True,
            'days_back': 90,             # Last 90 days
            'max_drugs': 200             # Top 200 drugs
        },
        'enforcement': {
            'enabled': True,
            'days_back': 365,            # Last year
            'max_results': None
        }
    }
}
```

---

## Data Sources

### 1. Drug Labels API

**Endpoint:** `https://api.fda.gov/drug/label.json`

**Content:**
- Package insert text
- Indications and usage
- Contraindications
- Warnings and precautions
- Adverse reactions
- Dosage and administration
- Drug interactions

**Search Fields:**
- `indications_and_usage`
- `description`
- `openfda.pharm_class_epc`

### 2. Adverse Events API

**Endpoint:** `https://api.fda.gov/drug/event.json`

**Content:**
- Patient demographics
- Drug information
- Adverse reaction descriptions
- Outcome (hospitalization, death, etc.)
- Reporter information

**Search Parameters:**
- `patient.drug.medicinalproduct` - Drug name
- `receivedate` - Report date range

### 3. Enforcement Reports API

**Endpoint:** `https://api.fda.gov/drug/enforcement.json`

**Content:**
- Recall classification (I, II, III)
- Product description
- Reason for recall
- Distribution pattern
- Firm information

**Search Parameters:**
- `report_date` - Date range

### 4. Approval Packages (Web Scraping)

**Source:** `https://www.accessdata.fda.gov/drugsatfda_docs/`

**Content:**
- Approval letters
- Medical reviews
- Clinical pharmacology reviews
- Statistical reviews
- Chemistry reviews
- Package labels

**Process:**
1. Find NDA/BLA number via labels API
2. Locate Table of Contents (TOC) page
3. Download all linked PDF documents

---

## Usage

### Basic Usage

```bash
cd FDA
python sync.py
```

### Interactive Prompts

For `full` mode, you'll be asked to confirm:
```
Continue with FULL sync? (yes/no): yes
```

### Non-Interactive Mode

```bash
# For automated/scheduled runs
echo "yes" | python sync.py
```

### Running Specific Modes

Edit `syncher_keys.py`:
```python
MODE = 'test'   # Quick test
MODE = 'daily'  # Daily update
MODE = 'full'   # Complete sync
```

### Resuming Interrupted Syncs

The syncer automatically resumes from where it left off:

1. **Labels**: Progress saved after each field search
2. **Adverse Events**: Progress saved every 10 drugs
3. **Approval Packages**: Skips existing packages

To force re-download:
```python
FORCE_REDOWNLOAD = True
```

---

## Output Structure

### FDA_DATA Directory

```
FDA_DATA/
├── labels/
│   ├── hematology_labels_20251126.json
│   ├── nephrology_labels_20251126.json
│   └── .progress/                       # Resume tracking
│
├── adverse_events/
│   ├── hematology_adverse_events_20251126.json
│   ├── nephrology_adverse_events_20251126.json
│   └── .progress/                       # Resume tracking
│
├── enforcement/
│   ├── hematology_enforcement_20251126.json
│   └── nephrology_enforcement_20251126.json
│
└── approval_packages/
    ├── hematology/
    │   ├── KEYTRUDA/
    │   │   ├── INDEX.md
    │   │   ├── approval_letter/
    │   │   │   └── approval_letter.pdf
    │   │   ├── label/
    │   │   │   └── package_insert.pdf
    │   │   ├── medical_review/
    │   │   │   └── medical_review.pdf
    │   │   └── other_review/
    │   │       └── statistical_review.pdf
    │   └── REVLIMID/
    │       └── ...
    └── nephrology/
        └── ...
```

### JSON File Formats

#### Labels JSON
```json
[
  {
    "id": "abc123",
    "set_id": "def456",
    "effective_time": "20230101",
    "indications_and_usage": ["For treatment of..."],
    "dosage_and_administration": ["Take 10mg daily..."],
    "warnings": ["Do not use if..."],
    "openfda": {
      "brand_name": ["KEYTRUDA"],
      "generic_name": ["pembrolizumab"],
      "application_number": ["BLA125514"],
      "product_type": ["HUMAN PRESCRIPTION DRUG"]
    }
  }
]
```

#### Adverse Events JSON
```json
[
  {
    "safetyreportid": "12345678",
    "receivedate": "20231015",
    "serious": "1",
    "patient": {
      "patientonsetage": "65",
      "patientsex": "1",
      "drug": [...],
      "reaction": [...]
    },
    "query_drug": "KEYTRUDA",
    "therapeutic_area": "hematology"
  }
]
```

#### Enforcement JSON
```json
[
  {
    "recall_number": "D-1234-2023",
    "classification": "Class II",
    "product_description": "Drug tablets...",
    "reason_for_recall": "Contamination...",
    "report_date": "20231201",
    "therapeutic_area": "hematology"
  }
]
```

---

## Downloaders Reference

### LabelsDownloader

**File:** `fda_syncher/downloaders/labels.py`

**Features:**
- Keyword batching (15 keywords per query) to avoid URI length errors
- Incremental saves every 100 results
- Deduplication by `set_id`
- Product type tracking (prescription vs OTC)

**Key Methods:**
```python
download(therapeutic_area) -> List[dict]
```

**Output Stats:**
```
[OK] Downloaded 4486 unique drug labels
📊 Breakdown:
   - Prescription drugs: 3200
   - OTC drugs: 1100
   - Other: 186
```

---

### AdverseEventsDownloader

**File:** `fda_syncher/downloaders/adverse_events.py`

**Features:**
- Drug name filtering (excludes cosmetics)
- Circuit breaker (pauses on 10 consecutive errors)
- Smart 404 handling (expected "no data" vs real errors)
- Incremental saves every 10 drugs

**Key Methods:**
```python
download(therapeutic_area, drug_names) -> List[dict]
```

**Circuit Breaker:**
```
⚠️  CIRCUIT BREAKER TRIGGERED!
    Consecutive errors: 10
    Pausing for 60 seconds to allow API recovery...
```

**Output Stats:**
```
[OK] Downloaded 14,156 adverse events
📊 Stats:
   - Drugs with events: 85/200 (42%)
   - API requests: 419
   - Real errors: 12 (2.9%)
   - No data (expected): 198
```

---

### ApprovalPackagesDownloader

**File:** `fda_syncher/downloaders/approval_packages.py`

**Features:**
- Timeout protection (20s per drug TOC search, 120s per package)
- Year range 2010-2025 for TOC lookup
- ANDA (generic) filtering
- Document categorization

**Key Methods:**
```python
download(therapeutic_area, drug_names) -> List[dict]
```

**Document Categories:**
- `approval_letter` - Official approval correspondence
- `label` - Package insert
- `medical_review` - Clinical review
- `clinical_pharm_review` - Clinical pharmacology
- `statistical_review` - Statistical analysis
- `chemistry_review` - CMC review
- `other_review` - Other documents

**Output Stats:**
```
✅ COMPLETE!
   Found & Downloaded: 30 packages
   Skipped (cached): 13
   Skipped (ANDA/generic): 45
   Not found: 106
   Timeouts: 2
   Errors: 4
   Total time: 85.3 minutes
   Avg docs/package: 12.5
```

---

### EnforcementDownloader

**File:** `fda_syncher/downloaders/enforcement.py`

**Features:**
- Date range filtering
- Classification breakdown in output

**Key Methods:**
```python
download(therapeutic_area) -> List[dict]
```

**Output Stats:**
```
[OK] Downloaded 730 enforcement reports
📊 By classification:
   - Class I: 45
   - Class II: 520
   - Class III: 165
```

---

## Therapeutic Areas

### syncher_therapeutic_areas.py

Defines diseases, drug classes, and aliases for comprehensive search coverage.

### Nephrology (32 Diseases)

| Category | Examples |
|----------|----------|
| **Glomerular** | IgA nephropathy, FSGS, membranous nephropathy |
| **Autoimmune** | Lupus nephritis, ANCA vasculitis, anti-GBM disease |
| **Complement** | aHUS, C3 glomerulopathy, PNH |
| **Genetic** | Alport syndrome, Fabry disease, cystinosis |
| **Polycystic** | ADPKD, ARPKD |
| **Tubular** | Bartter syndrome, Gitelman syndrome, RTA |

### Hematology (70 Diseases)

| Category | Examples |
|----------|----------|
| **Bleeding** | Hemophilia A/B, von Willebrand disease |
| **Hemoglobin** | Sickle cell disease, beta thalassemia |
| **Marrow Failure** | Aplastic anemia, Diamond-Blackfan anemia |
| **Myeloproliferative** | MDS, primary myelofibrosis, PV, ET |
| **Platelet** | ITP, TTP, HIT |
| **Leukemia** | AML, ALL, CML, CLL |
| **Lymphoma** | Hodgkin, DLBCL, follicular, mantle cell |
| **Plasma Cell** | Multiple myeloma, Waldenström's |

### Aliases

The system maps common abbreviations to canonical names:

```python
ALIASES = {
    "FSGS": "focal segmental glomerulosclerosis",
    "aHUS": "atypical hemolytic uremic syndrome",
    "AML": "acute myeloid leukemia",
    "SCD": "sickle cell disease",
    "TTP": "thrombotic thrombocytopenic purpura",
    # ... 100+ more
}
```

---

## Performance Optimization

### v2.1 Improvements

| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| Drug filtering | None | `filter_pharmaceutical_drugs()` | 50% fewer API calls |
| 404 handling | Counted as errors | Expected response | Accurate error rates |
| TOC timeout | None | 20 seconds | No stalls |
| Circuit breaker | 5 minutes | 60 seconds | Faster recovery |
| Connection recycling | Every 100 requests | Every 100 requests | Prevents stale connections |

### Parallel Processing

```python
# Workers by mode
'test':  2 workers  # Conservative
'daily': 4 workers  # Balanced
'full':  4 workers  # Reduced from 6 for stability
```

### Expected Performance

| Mode | Duration | Data Size |
|------|----------|-----------|
| Test | 15 min | ~50 MB |
| Daily | 1 hour | ~200 MB |
| Full | 6-10 hours | 2-3 GB |

### Memory Usage

- Labels: ~500 MB in memory during processing
- Adverse Events: Incremental saves keep memory low
- Approval Packages: Streaming downloads, minimal memory

---

## Troubleshooting

### Common Issues

#### 1. "URI Too Long" Error

**Cause:** Too many keywords in single query

**Solution:** Already fixed in v2.1 with keyword batching (15 per query)

#### 2. High Error Rate (~50%)

**Cause:** Cosmetics/OTC products in drug list returning 404s

**Solution:** v2.1 filters non-pharmaceutical products automatically

#### 3. Approval Packages Stalling

**Cause:** Infinite loop searching for non-existent TOC pages

**Solution:** v2.1 adds 20-second timeout per drug

#### 4. Circuit Breaker Triggering

**Cause:** API rate limiting or temporary outage

**Solution:** 
- Normal behavior - wait for 60-second pause
- If persistent, check your API key
- Reduce `max_drugs` in config

#### 5. SSL Certificate Errors

**Cause:** Corporate network SSL inspection

**Solution:** Already handled with `verify=False` in http_client.py

#### 6. Incomplete Data After Interruption

**Cause:** Sync stopped mid-process

**Solution:** Just re-run `python sync.py` - it auto-resumes

### Clearing Cache

```bash
# Clear today's labels (force re-download)
rm FDA_DATA/labels/*_labels_$(date +%Y%m%d).json
rm -rf FDA_DATA/labels/.progress/

# Clear today's adverse events
rm FDA_DATA/adverse_events/*_adverse_events_$(date +%Y%m%d).json
rm -rf FDA_DATA/adverse_events/.progress/

# Clear all approval packages
rm -rf FDA_DATA/approval_packages/

# Clear everything
rm -rf FDA_DATA/*
```

### Debug Mode

Add to http_client.py for verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## API Reference

### SimpleHTTPClient

```python
from fda_syncher.utils.http_client import SimpleHTTPClient

client = SimpleHTTPClient(max_retries=3, recycle_every=100)

# GET request with auto-retry
response = client.get(url, params={...})

# Download file with streaming
client.download_file(url, filepath)

# Cleanup
client.close()
```

### Helper Functions

```python
from fda_syncher.utils.helpers import (
    filter_pharmaceutical_drugs,
    extract_drug_names_from_labels,
    ensure_dir,
    get_today_file,
    check_existing_file
)

# Filter out cosmetics/OTC
pharma_drugs = filter_pharmaceutical_drugs(drug_list)

# Extract drugs from label data
drugs = extract_drug_names_from_labels(labels_json)

# Create directory if needed
ensure_dir("/path/to/dir")

# Check for today's file
existing = get_today_file("/path/to/dir", "pattern")
```

### Orchestrator

```python
from fda_syncher.orchestrator import FDASyncOrchestrator

# Create orchestrator with parallel workers
orchestrator = FDASyncOrchestrator(max_workers=4)

# Run full sync
results = orchestrator.run()

# Results structure
# {
#     'hematology': {
#         'labels': 4486,
#         'adverse_events': 14156,
#         'approval_packages': 30,
#         'enforcement': 730
#     },
#     'nephrology': {...}
# }
```

---

## Changelog

### v2.1.0 (Current)

**New Features:**
- Drug name filtering (`filter_pharmaceutical_drugs()`)
- Timeout protection for approval packages
- Better progress reporting with stats

**Bug Fixes:**
- 404s no longer counted as errors
- Circuit breaker reduced to 60 seconds
- TOC search limited to 20 seconds

**Performance:**
- ~50% reduction in wasted API calls
- No more multi-hour stalls
- Accurate error rate reporting

### v2.0.0

**New Features:**
- Parallel processing with ThreadPoolExecutor
- Incremental saves with resume capability
- Circuit breaker for API protection
- Connection pool recycling

**Bug Fixes:**
- Keyword batching for URI length
- SSL workarounds for corporate networks

### v1.0.0

- Initial release
- Sequential processing
- Basic retry logic

---

## License

Internal use only. FDA data is public domain but subject to FDA terms of use.

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review log output for specific errors
3. Run `fda_data_quality_check.py` to verify data integrity

---

## Appendix: FDA API Rate Limits

| Key Status | Requests/Minute | Requests/Day |
|------------|-----------------|--------------|
| No API Key | 240 | 120,000 |
| With API Key | 240 | 240,000 |

**Get your free API key:** https://open.fda.gov/apis/authentication/

---

*Documentation generated for FDA Syncer v2.1.0*
*Last updated: November 2025*