CTIS Scraper with PDF Download Support
Updated CTIS (Clinical Trials Information System) data extraction scripts with PDF/document downloading capability.
Based on: https://hendrik.codes/post/scraping-the-clinical-trials-information-system
What's New
PDF/Document Download Feature
The scripts have been enhanced to automatically download PDF documents associated with each clinical trial from the CTIS portal. This includes:

Protocol documents
Informed consent forms
Clinical study reports
Lay summaries
Summary of results
Safety notifications
And more...

Key Features

Centralized PDF Storage: All PDFs saved to ctis-out/pdf/ folder
Consistent Naming: Files follow naming convention {TRIAL_ID}_{original_filename}
Version Control: Automatically checks for document updates
Smart Updates: Only downloads new or updated documents, skips unchanged files
File Replacement: Old versions are replaced when newer versions are available

Files Included

ctis_run.py - Main entry point with configuration
ctis_processor.py - Trial processing with PDF download integration
ctis_pdf_downloader.py - NEW PDF/document download module with version control
ctis_config.py - Configuration with PDF download settings
ctis_database.py - Database schema and operations
ctis_discovery.py - Trial discovery/search
ctis_extractors.py - Data extraction from JSON
ctis_http.py - HTTP utilities with rate limiting
ctis_utils.py - Utility functions
ctis_report_generator.py - Report generation
ctis_excel_gen.py - Excel export
ctis_qa.py - Quality assurance checks
ctis_metadata_extract.py - Metadata extraction
CTIS_Key_Mappings.json - Reference data mappings
CTIS_List_Values_All.json - CTIS list values

Configuration
Edit ctis_run.py to configure the extraction:
pythonclass CTISConfig:
    # ========== EXTRACTION MODE ============================
    SINGLE_TRIAL = None          # Set to CT number for single trial
    TRIAL_COUNT = None           # Set number to limit trials
    EXTRACT_ALL = True           # Set True to extract all
    
    # ========== RARE DISEASE FILTER ========================
    FILTER_RARE_DISEASE_ONLY = True
    
    # ========== PDF/DOCUMENT DOWNLOAD ======================
    DOWNLOAD_PDFS = False        # Set to True to enable PDF downloading
    DOWNLOAD_ONLY = False        # Set True to only download PDFs (skip extraction)
    DOWNLOAD_FILE_TYPES = ['.pdf']  # File types to download
    ONLY_FOR_PUBLICATION = True  # Only download public versions
    
    # ========== OUTPUT SETTINGS ============================
    OUTPUT_DIR = Path("ctis-out")
    RESET_DATABASE = False
Usage
Basic Usage (Trial Data Only)
bashpython ctis_run.py
With PDF Download
Set DOWNLOAD_PDFS = True in ctis_run.py, then:
bashpython ctis_run.py
Download PDFs for Existing Trials
If you've already extracted trial data and want to download PDFs:
pythonDOWNLOAD_PDFS = True
DOWNLOAD_ONLY = True
Then run:
bashpython ctis_run.py
Update PDFs (Check for New Versions)
Simply re-run the script with DOWNLOAD_PDFS = True. The downloader will:

Check each document's version against the database
Skip documents that are already current
Download and replace documents that have newer versions
Download new documents that don't exist yet

Output Structure
ctis-out/
├── ctis.db                     # SQLite database
├── ctis_full.ndjson            # Raw JSON data
├── ct_numbers.txt              # List of CT numbers
├── failed_ctnumbers.txt        # Failed extractions
└── pdf/                        # Centralized PDF folder
    ├── 2024-511234-12-00_Protocol.pdf
    ├── 2024-511234-12-00_InformedConsent_EN.pdf
    ├── 2024-512345-23-00_Protocol.pdf
    └── ...
Naming Convention
All PDFs follow this naming pattern:
{TRIAL_ID}_{original_filename}
Examples:

2024-511234-12-00_Protocol_v2.pdf
2024-511234-12-00_ICF_English.pdf
2024-512345-23-00_LayPersonSummary.pdf

Document Database Table
When PDF downloading is enabled, a trial_documents table is created in the database:
sqlCREATE TABLE trial_documents (
    id INTEGER PRIMARY KEY,
    ctNumber TEXT,
    doc_id TEXT,
    doc_type TEXT,
    filename TEXT,
    title TEXT,
    language TEXT,
    upload_date TEXT,
    version TEXT,
    country TEXT,
    for_publication INTEGER,
    source TEXT,
    downloaded INTEGER,
    file_path TEXT,
    file_size INTEGER,
    sha256 TEXT,
    downloaded_at TEXT,
    download_error TEXT
);
Version Control Logic
The downloader checks for updates using:

Version number comparison: If document has a version, compares numerically
Upload date comparison: If versions equal, checks upload dates
Content hash (SHA256): Ensures content is actually different before replacing
File existence: Always downloads if file doesn't exist on disk

Requirements
requests>=2.28.0
Notes

PDF downloading adds significant time to extraction
All documents for all trials go to single ctis-out/pdf folder
Filenames always start with TRIAL_ID for easy identification
The script respects rate limits to avoid overloading the CTIS server
Failed downloads are logged but don't stop the extraction process
Use DOWNLOAD_ONLY = True to retry failed PDF downloads or update existing PDFs

Document Types
The following document types may be available for download:

Protocol
Protocol Synopsis
Informed Consent Form
Investigator Brochure
SmPC (Summary of Product Characteristics)
PIL (Patient Information Leaflet)
Recruitment Materials
Subject Information
Clinical Study Report
Lay Summary
Summary of Results
Safety Notifications
Urgent Safety Measures
Corrective Measures

API Endpoints Used

Search: POST https://euclinicaltrials.eu/ctis-public-api/search
Trial Details: GET https://euclinicaltrials.eu/ctis-public-api/retrieve/{ct_number}
Document Download: GET https://euclinicaltrials.eu/ctis-public-api/retrieve/{ct}/document/{doc_id}

License
MIT License - See original repository for details.