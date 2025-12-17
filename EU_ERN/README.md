# ERN Resource Scraper v4.0 - Ultimate Edition

```
██████╗ ██████╗ ███╗   ██╗    ███████╗ ██████╗██████╗  █████╗ ██████╗ ███████╗██████╗ 
██╔════╝██╔══██╗████╗  ██║    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
█████╗  ██████╔╝██╔██╗ ██║    ███████╗██║     ██████╔╝███████║██████╔╝█████╗  ██████╔╝
██╔══╝  ██╔══██╗██║╚██╗██║    ╚════██║██║     ██╔══██╗██╔══██║██╔═══╝ ██╔══╝  ██╔══██╗
███████╗██║  ██║██║ ╚████║    ███████║╚██████╗██║  ██║██║  ██║██║     ███████╗██║  ██║
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
```

ERN-EuroBloodNet
European Reference Network on Rare Haematological Diseases – a network focusing on rare blood disorders, including complex and ultra-rare conditions requiring highly specialized expertise.
ERKNet
European Reference Network on Kidney Diseases – a collaborative network dedicated to rare and complex kidney disorders across all ages.
ERN RITA
European Reference Network on Immunodeficiency, Autoinflammatory and Autoimmune Diseases – a network for rare primary immunodeficiencies, autoinflammatory and systemic autoimmune diseases.
ERN EURO-NMD
European Reference Network on Neuromuscular Diseases – a network specializing in rare inherited and acquired neuromuscular disorders.
MetabERN
European Reference Network for Hereditary Metabolic Disorders – a network covering rare inherited metabolic diseases affecting multiple organs and systems.
ERN GUARD-HEART
European Reference Network on Rare and Complex Diseases of the Heart – a network that addresses rare cardiomyopathies, arrhythmia syndromes and other inherited cardiac conditions.
ERN BOND
European Reference Network on Rare Bone Diseases – a network focusing on rare skeletal disorders, bone fragility and metabolic bone diseases.
Enterprise-grade web scraper for European Reference Networks (ERNs) clinical practice guidelines and resources, optimized for RAG (Retrieval-Augmented Generation) systems.

Supported ERN Networks
NetworkFocus AreaERN-EuroBloodNetRare haematological diseases, complex and ultra-rare blood conditionsERKNetRare and complex kidney disorders across all agesERN-RITAPrimary immunodeficiencies, autoinflammatory and autoimmune diseasesERN-EURO-NMDRare inherited and acquired neuromuscular disordersMetabERNHereditary metabolic disorders affecting multiple organsERN-GUARD-HEARTRare cardiomyopathies, arrhythmia syndromes, inherited cardiac conditionsERN-BONDRare skeletal disorders, bone fragility, metabolic bone diseases

✨ What's New in v4.1 - Discovery-First
The Problem with v4.0
Config-based URL paths become outdated as websites change, resulting in:

60%+ 404 error rates on some networks
Wasted requests on non-existent pages
Missing content from new site sections

The Solution: Discovery-First Architecture
v4.1 discovers actual site structure BEFORE scraping:
[DISCOVERY] Systematic URL discovery: https://eurobloodnet.eu
============================================================

[Step 1/4] Checking sitemaps...
   [OK] Found 1672 URLs in sitemap

[Step 2/4] Analyzing navigation...
   [OK] Found 15 navigation links:
        - /best-practices/
        - /disease-groups/
        - /education/
        - /members/
        ...

[Step 3/4] Exploring depth 2 (following nav links)...
   [OK] Found 288 additional URLs at depth 2

[Step 4/4] Evaluating discovery...
   Sitemap:      0 URLs
   Navigation:  15 URLs
   Page links:  79 URLs
   Depth-2:    288 URLs
   -------------------------
   TOTAL:      382 URLs
   Threshold:    5 URLs

   [OK] Discovery SUCCESS! Config paths will be IGNORED.

[READY] 381 URLs queued for scraping
============================================================
Key Improvements
Featurev4.0v4.1URL DiscoveryConfig paths onlySitemap + Nav + Depth-2Config RolePrimary sourceFallback only404 Rate30-60%< 5%New ContentMissedAuto-discoveredERN-specific patternsNoneWorking groups, ePAG, CPMS

🔍 Discovery-First Pipeline
How It Works
┌─────────────────────────────────────────────────────────────┐
│                    DISCOVERY PHASE                          │
├─────────────────────────────────────────────────────────────┤
│  Step 1: SITEMAP                                            │
│  ├── /sitemap.xml                                           │
│  ├── /wp-sitemap.xml                                        │
│  └── /sitemap_index.xml → sub-sitemaps                      │
│                                                             │
│  Step 2: NAVIGATION                                         │
│  ├── <nav> elements                                         │
│  ├── <header> links                                         │
│  └── Menu structures                                        │
│                                                             │
│  Step 3: DEPTH-2 EXPLORATION                                │
│  ├── Follow top 15 nav links                                │
│  └── Extract all internal links from each                   │
│                                                             │
│  Step 4: EVALUATION                                         │
│  ├── Total >= 5 URLs? → SUCCESS (ignore config)             │
│  └── Total < 5 URLs?  → FALLBACK (use config paths)         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    SCRAPING PHASE                           │
│  • Only validated URLs from discovery                       │
│  • Near-zero 404 errors                                     │
│  • Discovers new content automatically                      │
└─────────────────────────────────────────────────────────────┘
Discovery Results by Network
NetworkSitemapNavDepth-2TotalConfig Used?ERKNet1672--1672NoEuroBloodNet015288382NoERN-EURO-NMD200+20150370+NoERN-BOND0000Yes (fallback)
ERN-Specific High-Value Patterns
The discovery engine prioritizes these medical/ERN patterns:
pythonHIGH_VALUE_PATTERNS = [
    # Clinical
    'guideline', 'protocol', 'pathway', 'recommendation',
    'clinical', 'patient', 'disease', 'disorder',
    'diagnosis', 'treatment', 'therapy', 'care',
    
    # Educational
    'education', 'training', 'webinar', 'course',
    'publication', 'research', 'registry', 'database',
    
    # ERN-specific
    'working-group', 'work-package', 'epag', 'e-pag',
    'governance', 'affiliated', 'partner', 'trial',
    'cpms', 'meeting', 'dissemination',
    'wg-leader', 'coordinator', 'board',
]

🚀 Quick Start
Installation
bash# Install dependencies
pip install requests beautifulsoup4 lxml

# Run with default config
python ern_scraper.py

# List enabled networks
python ern_scraper.py --list-enabled
Basic Usage
bash# Scrape all enabled networks (discovery-first)
python ern_scraper.py

# Scrape specific network only
python ern_scraper.py --network ERKNet

# Show statistics from last run
python ern_scraper.py --stats

# Reset state and start fresh
python ern_scraper.py --reset

📋 Configuration
ern_config.json
json{
  "scraper_settings": {
    "max_crawl_depth": 8,
    "max_pages_per_network": 3000,
    "default_request_delay": 2.5,
    "request_timeout": 90,
    "max_retries": 5,
    "skip_existing": true,
    "verbose": true,
    "output_directory": "EU_ERN_DATA",
    "use_sqlite_state": true,
    "detect_duplicates": true,
    "parse_sitemaps": true
  },
  "domain_delays": {
    "health.ec.europa.eu": 10.0,
    "ec.europa.eu": 10.0,
    "ern-net.eu": 3.0,
    "default": 2.5
  },
  "networks": {
    "ERKNet": {
      "scrape": true,
      "name": "European Rare Kidney Diseases Reference Network",
      "website": "https://www.erknet.org",
      "guidelines_paths": ["/guidelines/", "/patients/", ...]
    }
  }
}
Config Paths as Fallback
In v4.1, guidelines_paths are only used if discovery fails:
Discovery finds 382 URLs → Config paths IGNORED
Discovery finds 0 URLs   → Config paths USED as fallback
The scraper also outputs suggested paths based on what it discovered:
[CONFIG SUGGESTION] Update ern_config.json with these paths:
============================================================
"guidelines_paths": [
    "/best-practices/",
    "/cpms/",
    "/disease-groups/",
    "/education/",
    "/members/",
]

🎯 Smart URL Filtering
Automatically Skips
Binary Files:

Images: .png, .jpg, .jpeg, .gif, .webp, .svg, .ico
Media: .mp4, .mp3, .avi, .mov, .wav
Archives: .zip, .rar, .7z, .tar, .gz
Fonts: .woff, .woff2, .ttf, .eot

Low-Value Paths:

WordPress admin: /wp-admin/, /wp-login.php, /wp-includes/
Author/archive pages: /author/, /tag/, /category/
Pagination: /page/2/, ?paged=3
Login/register: /login, /sign-in, /account
Search results: /search/, ?s=query

Result: ~30-40% fewer unnecessary requests!

💾 SQLite State Management
Benefits

Faster: Direct queries vs. full file reads
Robust: ACID transactions prevent corruption
Scalable: Handles millions of URLs efficiently
Queryable: Direct SQL access for custom analysis

Querying State
bashsqlite3 EU_ERN_DATA/scraper_state.db

-- Show download progress
SELECT status, COUNT(*) FROM urls GROUP BY status;

-- Show network completion
SELECT network_id, pages_scraped, status FROM networks;

-- Find duplicates
SELECT * FROM content_hashes WHERE count > 1;

📁 Output Structure
EU_ERN_DATA/
├── rag_content/              # Clean markdown files (RAG-ready)
│   ├── ERKNet_main.md
│   ├── ERKNet_guidelines_pathways_1.md
│   └── ...
│
├── guidelines/               # Downloaded PDF guidelines
│   ├── ERKNet_alport_syndrome_2024.pdf
│   └── ...
│
├── methodologies/            # EC methodological handbooks
│   ├── Handbook_00_Introduction.pdf
│   └── ...
│
├── factsheets/               # Network factsheets
│   ├── ERKNet_factsheet.pdf
│   └── ...
│
├── logs/                     # Detailed log files
│   └── ern_scraper_20251217_*.log
│
├── exports/                  # CSV/JSON exports
│
├── scraper_state.db          # SQLite state database
├── ern_scrape_results.json   # Full results with discovery_info
└── SUMMARY.md                # Human-readable summary
Discovery Info in Results
Each network now includes discovery metadata:
json{
  "networks": {
    "ERN-EuroBloodNet": {
      "discovery_info": {
        "success": true,
        "sitemap_urls": 0,
        "nav_urls": 15,
        "depth2_urls": 288,
        "total_urls": 382,
        "suggested_paths": ["/best-practices/", "/cpms/", ...]
      },
      "pages_scraped": [...],
      "crawl_stats": {...}
    }
  }
}

🔧 Troubleshooting
Discovery Finds 0 URLs
Cause: Site uses JavaScript-rendered navigation (React, Vue, etc.)
Solution: The scraper falls back to config paths automatically. Update config with correct paths:
json"guidelines_paths": [
    "/guidelines/",
    "/patients/",
    "/publications/"
]
Rate Limiting (429 Errors)
Solution: Increase delays in config:
json"domain_delays": {
  "slow-domain.eu": 10.0
}
Reset and Start Fresh
bashpython ern_scraper.py --reset

📈 Performance Comparison
v4.0 vs v4.1 (ERN-EuroBloodNet)
Metricv4.0v4.1URLs attempted13 (from config)382 (discovered)404 errors5 (38%)~5 (<2%)Pages scraped8370+Content coveragePartialComprehensiveNew sections foundNone25+
Discovery Time Overhead
NetworkDiscovery TimeURLs FoundWorth It?ERKNet15s1672✅ YesEuroBloodNet7s382✅ YesERN-BOND2s0 (fallback)✅ Fast fail

📜 Changelog
v4.1 (December 2025) - Discovery-First Edition
New Features:

🔍 Discovery-First Architecture: Systematic URL discovery BEFORE scraping
🌐 Depth-2 Exploration: Follows navigation links to find more content
📋 Config as Fallback: Config paths only used if discovery fails
🏥 ERN-Specific Patterns: Working groups, ePAG, CPMS, governance
💡 Suggested Paths Output: Auto-generates config recommendations
📊 Discovery Metadata: Results include discovery success/failure info

Improvements:

90%+ reduction in 404 errors
Discovers new site sections automatically
No more outdated config maintenance
Better content coverage

Fixed:

_analyze_main_page() argument mismatch
UTF-8 encoding corruption in source file

v4.0 (December 2025) - Ultimate Edition
Features:

SQLite state management
Smart URL filtering (40+ skip patterns)
Content fingerprinting for duplicate detection
Automatic sitemap discovery
Robots.txt compliance support
Content quality scoring
Real-time progress tracking with ETA
CSV export functionality
Parallel PDF downloads
Colored terminal output

v3.x (November 2025)

Config-driven network selection
Clean markdown output (RAG-ready)
Multi-level link discovery


📄 License
This scraper tool is provided for research and educational purposes. ERN content is subject to individual network and European Commission licensing terms.

🔗 Resources

ERN Portal
ERN Search Tool
Orphanet


ERN Resource Scraper v4.1 - Discovery-First Edition
December 2025