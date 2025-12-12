"""
FDA Data Syncer - Entry Point v2.1
Replaces syncher.py

Uses configuration from:
- syncher_keys.py
- syncher_therapeutic_areas.py

UPDATES v2.1:
- Improved drug name filtering (removes cosmetics/OTC)
- Better timeout handling for approval packages
- Smarter 404 handling (expected vs error)
- Reduced circuit breaker pause time
- Better progress reporting

Usage:
    python sync.py
"""

from fda_syncher.orchestrator import main

if __name__ == "__main__":
    main()
