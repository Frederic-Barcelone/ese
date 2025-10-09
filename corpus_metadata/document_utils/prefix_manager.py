#!/usr/bin/env python3
"""
Document Prefix Manager - Auto-incrementing ID System
=====================================================
Adds sequential prefixes (01000_, 01001_, etc.) to files without existing IDs.

Location: corpus_metadata/document_utils/prefix_manager.py
"""

import re
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DocumentPrefixManager:
    """
    Manages auto-incrementing 5-digit prefixes for documents without IDs.
    
    Features:
    - Detects existing xxxxx_ prefixes (5 digits + underscore)
    - Starts at 01000 for new files
    - Persists counter to simple text file
    - Thread-safe file operations
    """
    
    COUNTER_FILENAME = ".document_counter.txt"
    DEFAULT_START = 1000
    PREFIX_PATTERN = re.compile(r'^\d{5}_')
    
    def __init__(self, 
                 counter_dir: Optional[Path] = None,
                 start_number: int = DEFAULT_START):
        """
        Initialize the prefix manager.
        
        Args:
            counter_dir: Directory to store counter file (default: current dir)
            start_number: Starting number for auto-increment (default: 1000)
        """
        self.counter_dir = Path(counter_dir) if counter_dir else Path.cwd()
        self.counter_file = self.counter_dir / self.COUNTER_FILENAME
        self.start_number = start_number
        self.current_number = self._load_counter()
        
        logger.debug(f"Initialized DocumentPrefixManager: counter={self.current_number}, "
                    f"file={self.counter_file}")
    
    def _load_counter(self) -> int:
        """Load counter from file or initialize to start number"""
        if self.counter_file.exists():
            try:
                with open(self.counter_file, 'r') as f:
                    counter = int(f.read().strip())
                    logger.info(f"Loaded counter: {counter} from {self.counter_file}")
                    return counter
            except (ValueError, IOError) as e:
                logger.warning(f"Could not load counter from {self.counter_file}: {e}. "
                             f"Using start number {self.start_number}")
                return self.start_number
        else:
            logger.info(f"No counter file found. Starting at {self.start_number}")
            return self.start_number
    
    def _save_counter(self):
        """Persist current counter to file"""
        try:
            self.counter_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.counter_file, 'w') as f:
                f.write(str(self.current_number))
            logger.debug(f"Saved counter: {self.current_number} to {self.counter_file}")
        except IOError as e:
            logger.error(f"Failed to save counter to {self.counter_file}: {e}")
    
    def has_prefix(self, filename: str) -> bool:
        """
        Check if filename already has a 5-digit prefix.
        
        Args:
            filename: Filename to check (can include path)
            
        Returns:
            True if filename starts with xxxxx_ pattern
            
        Examples:
            >>> manager.has_prefix("00985_document.pdf")
            True
            >>> manager.has_prefix("document.pdf")
            False
            >>> manager.has_prefix("12345_report.pdf")
            True
        """
        basename = Path(filename).name
        return bool(self.PREFIX_PATTERN.match(basename))
    
    def add_prefix_if_needed(self, filename: str, save_counter: bool = True) -> tuple[str, bool]:
        """
        Add auto-incrementing prefix if filename doesn't have one.
        
        Args:
            filename: Original filename (can include path)
            save_counter: Whether to persist counter after increment
            
        Returns:
            Tuple of (new_filename, was_modified)
            
        Examples:
            >>> manager.add_prefix_if_needed("document.pdf")
            ("01000_document.pdf", True)
            >>> manager.add_prefix_if_needed("00985_existing.pdf")
            ("00985_existing.pdf", False)
        """
        path = Path(filename)
        basename = path.name
        
        # Check if already has prefix
        if self.has_prefix(basename):
            logger.debug(f"File already has prefix: {basename}")
            return str(filename), False
        
        # Add new prefix
        new_basename = f"{self.current_number:05d}_{basename}"
        new_path = path.parent / new_basename
        
        logger.info(f"Added prefix {self.current_number:05d}_ to {basename}")
        
        # Increment counter
        self.current_number += 1
        
        # Persist counter
        if save_counter:
            self._save_counter()
        
        return str(new_path), True
    
    def get_next_prefix(self) -> str:
        """
        Get the next prefix without incrementing.
        
        Returns:
            Next prefix string (e.g., "01000_")
        """
        return f"{self.current_number:05d}_"
    
    def reset_counter(self, new_start: Optional[int] = None):
        """
        Reset counter to start number or specified value.
        
        Args:
            new_start: New starting number (default: use initial start_number)
        """
        self.current_number = new_start if new_start is not None else self.start_number
        self._save_counter()
        logger.info(f"Reset counter to {self.current_number}")
    
    def get_current_counter(self) -> int:
        """Get current counter value"""
        return self.current_number


# ============================================================================
# Integration Helper Functions
# ============================================================================

def integrate_prefix_manager_into_pipeline(documents_folder: Path) -> DocumentPrefixManager:
    """
    Create and initialize prefix manager for document processing pipeline.
    
    Args:
        documents_folder: Root folder for documents
        
    Returns:
        Initialized DocumentPrefixManager
    """
    manager = DocumentPrefixManager(
        counter_dir=documents_folder,
        start_number=1000  # Start at 01000
    )
    return manager


def apply_prefix_to_renamed_file(
    original_filename: str,
    enhanced_filename: str,
    prefix_manager: DocumentPrefixManager
) -> str:
    """
    Apply prefix to enhanced filename after intelligent renaming.
    
    Args:
        original_filename: Original PDF filename
        enhanced_filename: Enhanced name from IntelligentDocumentRenamer
        prefix_manager: Active DocumentPrefixManager instance
        
    Returns:
        Final filename with prefix applied if needed
        
    Usage in pipeline:
        enhanced_name = renamer.generate_filename(...)
        final_name = apply_prefix_to_renamed_file(
            original_name, enhanced_name, prefix_manager
        )
    """
    final_name, was_modified = prefix_manager.add_prefix_if_needed(enhanced_filename)
    
    if was_modified:
        logger.info(f"Applied auto-prefix: {enhanced_filename} -> {final_name}")
    
    return final_name

