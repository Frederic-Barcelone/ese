"""
FDA Data Syncer Package
Uses syncher_keys.py and syncher_therapeutic_areas.py for configuration
"""

__version__ = "2.0.0"

from .orchestrator import FDASyncOrchestrator, main

__all__ = ['FDASyncOrchestrator', 'main']
