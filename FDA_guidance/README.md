# FDA Guidance Document Scraper (v5)

A comprehensive Python script to build a complete FDA regulatory library for rare disease drug development.

## What's New in v5

### 🔧 Bug Fixes
- **Fixed broken URLs**: Updated disease-specific guidances (Fabry, Huntington's, SMA)
- **Fixed gene therapy index page**: Updated URL for cellular/gene therapy guidances
- **Improved 404 handling**: No longer retries 404 errors (saves time on removed pages)
- **URL encoding**: Proper handling of special characters in URLs

### ✨ Improvements
- **Better error messages**: More informative failure reasons in progress tracking
- **Direct PDF links**: Added support for direct PDF/media URLs (e.g., Huntington's PFDD report)
- **New guidances added**: Added recent CGT guidances (CAR-T, genome editing, neurodegenerative diseases)
- **Main search disabled by default**: The FDA main search uses JavaScript tables - set to False to avoid wasted requests

## Quick Start

```bash
pip install requests beautifulsoup4
python fda_scraper_v5.py
```

That's it! The script will:
- Create `FDA_guidance_library/` folder with all PDFs
- Automatically resume if interrupted (just run again)
- Skip already-downloaded files
- Retry failed requests with exponential backoff (except 404s)
- Prevent duplicate downloads across categories
- Respect FDA's 30-second crawl delay

## Configuration

All settings are in the `CONFIG` dict at the top of the script:

```python
CONFIG = {
    # Output Settings
    "output_directory": "FDA_guidance_library",
    "log_file": "fda_scraper.log",
    
    # Network Settings
    "contact_email": "your-email@example.com",  # Update this!
    "request_delay": 30,           # FDA requires 30s between requests
    "request_timeout": 60,
    "download_timeout": 120,
    
    # Retry Settings
    "max_retries": 3,
    "backoff_base": 5.0,
    "backoff_max": 120.0,
    "skip_404_retry": True,        # NEW: Don't retry 404 errors
    
    # Crawl Settings
    "max_pages_per_index": 10,
    "max_documents_total": None,
    "skip_existing": True,
    "skip_index_pages": False,
    "crawl_main_search": False,    # CHANGED: Disabled by default (uses JS)
    
    # Logging Settings
    "verbose": True,
    "log_to_file": True,
    "log_to_console": True,
}
```

### Key Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_directory` | str | `"FDA_guidance_library"` | Where to save PDFs |
| `contact_email` | str | - | **Update this** - Used in User-Agent header |
| `request_delay` | int | `30` | Seconds between requests (FDA requires 30) |
| `max_retries` | int | `3` | Retry attempts per failed request |
| `skip_404_retry` | bool | `True` | **NEW** - Skip retries for 404 errors |
| `max_pages_per_index` | int/None | `10` | Limit pages crawled per index |
| `max_documents_total` | int/None | `None` | Stop after N documents |
| `skip_existing` | bool | `True` | Skip files already on disk |
| `crawl_main_search` | bool | `False` | **CHANGED** - Main search uses JS, disabled |

## Categories (13 Total)

| Folder | Category | Priority A Docs |
|--------|----------|-----------------|
| `01_rare_disease` | Rare Disease | Considerations, Natural History, Pre-IND |
| `02_evidence_benefit_risk` | Evidence Framework | Substantial Evidence, Benefit-Risk |
| `03_patient_focused` | PFDD | All 4 PFDD Guidances |
| `04_real_world_evidence` | RWD/RWE | Registries, External Controls, EHR |
| `05_pediatric` | Pediatric | E11(R1), E11A Extrapolation |
| `06_trial_design_statistics` | Trial Design | Adaptive, Enrichment, Endpoints |
| `07_safety_pharmacovigilance` | Safety | IND Safety, DSUR, REMS |
| `08_data_standards` | Data Standards | Study Data Submission |
| `09_expedited_programs` | Expedited | Fast Track, Breakthrough, AA, RMAT |
| `10_clinical_trials_gcp` | Clinical Trials | E6(R2), DCT, ICH |
| `11_disease_specific` | Disease-Specific | ALS, DMD, Fabry |
| `12_biomarkers` | Biomarkers | Biomarker Qualification |
| `13_gene_therapy` | Gene Therapy | Rare Diseases, CGT Design, CAR-T |

## Priority System

Files are prefixed by importance:

- **A_** = Must-have for any rare disease program (~30 documents)
- **B_** = Important reference documents (~40 documents)
- **C_** = Additional context from index pages (variable)

## Output Structure

```
FDA_guidance_library/
├── 01_rare_disease/
│   ├── A_Rare_Diseases_Considerations_for_Development.pdf
│   ├── A_Natural_History_Studies.pdf
│   └── ...
├── 02_evidence_benefit_risk/
│   ├── A_Demonstrating_Substantial_Evidence.pdf
│   └── ...
├── ...
├── 13_gene_therapy/
│   ├── A_Human_Gene_Therapy_for_Rare_Diseases.pdf
│   ├── B_CAR_T_Cell_Products_Development.pdf
│   └── ...
├── _progress.json      # Progress tracking (for resume)
└── manifest.json       # Final statistics & configuration
```

## Progress Tracking

The `_progress.json` file tracks everything:

```json
{
  "downloaded": [
    {
      "title": "Document Title",
      "url": "https://...",
      "path": "FDA_guidance_library/01_.../A_Document.pdf",
      "priority": "A",
      "category": "01_rare_disease"
    }
  ],
  "failed": [
    {
      "url": "https://...",
      "category": "11_disease_specific",
      "reason": "Could not find PDF on page"
    }
  ],
  "processed_urls": ["url1", "url2", ...],
  "last_category": "13_gene_therapy"
}
```

## Resume After Interruption

Just run the script again:

```bash
python fda_scraper_v5.py
```

It will automatically:
1. Load `_progress.json` to find previously processed URLs
2. Check which files already exist on disk
3. Skip all previously downloaded documents
4. Continue with remaining downloads

## Known Issues & Notes

### FDA Website Changes
The FDA periodically reorganizes their website. Some guidances that were previously available at specific URLs may have been:
- **Moved**: URL structure changed
- **Consolidated**: Multiple guidances merged into one
- **Withdrawn**: Draft guidance not finalized

The following disease-specific guidances currently return 404:
- Huntington's Disease (standalone guidance) - Use the PFDD report instead
- Spinal Muscular Atrophy (standalone guidance) - Covered by general rare disease guidance

### Main Search Page
The FDA's main guidance search (`/search-fda-guidance-documents`) uses a JavaScript-powered table that cannot be scraped with traditional methods. This is why `crawl_main_search` is set to `False` by default.

### Index Pages
Some index pages have been reorganized. The script now uses verified URLs:
- Gene therapy: `https://www.fda.gov/vaccines-blood-biologics/biologics-guidances/cellular-gene-therapy-guidances`

## Troubleshooting

### Many 404 Errors
If you see many 404 errors for explicitly defined documents:
1. Check the `_progress.json` file for failed URLs
2. Search for the document on [FDA Guidance Search](https://www.fda.gov/regulatory-information/search-fda-guidance-documents)
3. Update the URL in `GUIDANCE_LIBRARY` dict
4. Delete the failed entry from `_progress.json`
5. Re-run the script

### SSL/Certificate Errors
```python
# In create_session(), add:
session.verify = False  # Not recommended for production
```

### Timeout Errors
Increase timeouts in CONFIG:
```python
CONFIG = {
    "request_timeout": 120,
    "download_timeout": 300,
}
```

### Rate Limiting (429 errors)
```python
CONFIG = {
    "request_delay": 60,  # Increase from 30
}
```

### Fresh Start
To completely restart (keeping downloaded files):
```bash
rm FDA_guidance_library/_progress.json
python fda_scraper_v5.py
```

## Estimated Time

- **First run**: ~2-4 hours for ~80 documents (30-second delay per request)
- **Subsequent runs**: Minutes (skips existing files)

## Changes from v4 to v5

| Feature | v4 | v5 |
|---------|----|----|
| Fabry Disease URL | ❌ Broken | ✅ Fixed |
| Gene Therapy Index | ❌ Broken | ✅ Fixed |
| 404 Retry | Retries | ✅ Skips (faster) |
| URL Encoding | Basic | ✅ Improved |
| Main Search | Enabled | ✅ Disabled (JS) |
| Progress Tracking | Basic | ✅ With failure reasons |
| New CGT Guidances | ❌ Missing | ✅ Added (CAR-T, etc.) |

## Dependencies

```bash
pip install requests beautifulsoup4
```

- Python 3.7+
- requests
- beautifulsoup4

## Legal Notes

- FDA guidance documents are public domain
- Script respects robots.txt crawl delay (30 seconds)
- For bulk requests, consider contacting FDA: druginfo@fda.hhs.gov

## URL Maintenance

If you encounter 404 errors:

1. Check [FDA Guidance Documents](https://www.fda.gov/regulatory-information/search-fda-guidance-documents)
2. Search by document title
3. Update the URL in `GUIDANCE_LIBRARY`
4. Clear the entry from `_progress.json` failed list
5. Re-run

## Contributing

To add new guidances:

1. Add entry to appropriate category in `GUIDANCE_LIBRARY`:
```python
("B", "New Guidance Title",
 "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/new-guidance-slug"),
```

2. Use priority:
   - `A` = Critical for rare disease development
   - `B` = Important reference
   - `C` = Supplementary (usually discovered from index pages)

## Version History

- **v5.0** (2024): Fixed broken URLs, improved error handling, added new CGT guidances
- **v4.0**: Added deep pagination for index pages
- **v3.0**: Added comprehensive configuration, retry logic, progress tracking
- **v2.0**: Added resume capability, global URL tracking
- **v1.0**: Initial release

---

*Generated with assistance from Claude (Anthropic)*