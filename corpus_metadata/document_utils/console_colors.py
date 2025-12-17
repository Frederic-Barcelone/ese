#!/usr/bin/env python3
"""
corpus_metadata/document_utils/console_colors.py
================================================
Centralized ANSI color codes for terminal output.

corpus_metadata/document_utils/console_colors.py

"""

import sys


class Colors:
    """ANSI color codes for terminal output"""
    
    # Standard colors
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    
    # Bright variants
    BRIGHT_WHITE = '\033[97m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_CYAN = '\033[96m'
    
    # Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output"""
        for attr in dir(cls):
            if attr.isupper():
                setattr(cls, attr, '')
    
    @classmethod
    def enable(cls):
        """Re-enable colors (restore defaults)"""
        cls.HEADER = '\033[95m'
        cls.BLUE = '\033[94m'
        cls.CYAN = '\033[96m'
        cls.GREEN = '\033[92m'
        cls.YELLOW = '\033[93m'
        cls.RED = '\033[91m'
        cls.BRIGHT_WHITE = '\033[97m'
        cls.BRIGHT_BLACK = '\033[90m'
        cls.BRIGHT_CYAN = '\033[96m'
        cls.BOLD = '\033[1m'
        cls.UNDERLINE = '\033[4m'
        cls.ENDC = '\033[0m'


# Auto-disable if not a terminal
if not sys.stdout.isatty():
    Colors.disable()