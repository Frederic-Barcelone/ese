"""
FDA Data Syncer Package - v2.1
Uses syncher_keys.py and syncher_therapeutic_areas.py for configuration

UPDATES v2.1:
- Improved drug name filtering (removes cosmetics/OTC)
- Better timeout handling for approval packages
- Smarter 404 handling (expected vs error)
- Reduced circuit breaker pause time
- Better progress reporting
"""

__version__ = "2.1.0"

from .orchestrator import FDASyncOrchestrator, main

__all__ = ['FDASyncOrchestrator', 'main']
