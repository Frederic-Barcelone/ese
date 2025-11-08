"""
FDA Data Syncer - Entry Point
Replaces syncher.py

Uses configuration from:
- syncher_keys.py
- syncher_therapeutic_areas.py
FDA/synch.py
Usage:
    python sync.py
"""

from fda_syncher.orchestrator import main

if __name__ == "__main__":
    main()