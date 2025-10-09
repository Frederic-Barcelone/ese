#!/usr/bin/env python3
"""
Document Extraction Cache
========================
Provides caching functionality for document extraction results.
/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_extraction_cache.py
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import shutil

logger = logging.getLogger(__name__)


class VersionedExtractionCache:
    """
    A versioned cache for document extraction results.
    Stores extracted content and metadata to avoid re-processing documents.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = ".cache/extractions", 
                 max_size_mb: int = 500, 
                 ttl_days: int = 30):
        """
        Initialize the extraction cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
            ttl_days: Time to live for cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_days = ttl_days
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Version for cache invalidation
        self.cache_version = "1.0.0"
        
        logger.info(f"Initialized extraction cache at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            'version': '1.0.0',
            'entries': {},
            'total_size_bytes': 0
        }
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key for a file.
        
        Args:
            file_path: Path to the file
            options: Optional extraction options
            
        Returns:
            Cache key string
        """
        # Create a unique key based on file path, modification time, and options
        key_parts = [
            str(file_path.absolute()),
            str(file_path.stat().st_mtime) if file_path.exists() else "",
            self.cache_version
        ]
        
        if options:
            key_parts.append(json.dumps(options, sort_keys=True))
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached extraction results for a file.
        
        Args:
            file_path: Path to the file
            options: Optional extraction options
            
        Returns:
            Cached extraction results or None if not found/expired
        """
        cache_key = self._get_cache_key(file_path, options)
        
        # Check if entry exists in metadata
        if cache_key not in self.metadata['entries']:
            return None
        
        entry = self.metadata['entries'][cache_key]
        
        # Check if entry is expired
        entry_time = datetime.fromisoformat(entry['timestamp'])
        if datetime.now() - entry_time > timedelta(days=self.ttl_days):
            logger.debug(f"Cache entry expired for {file_path.name}")
            self._remove_entry(cache_key)
            return None
        
        # Load cached data
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            logger.warning(f"Cache file missing for {file_path.name}")
            self._remove_entry(cache_key)
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for {file_path.name}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cache for {file_path.name}: {e}")
            self._remove_entry(cache_key)
            return None
    
    def set(self, file_path: Path, data: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        """
        Cache extraction results for a file.
        
        Args:
            file_path: Path to the file
            data: Extraction results to cache
            options: Optional extraction options
        """
        # Check cache size limit
        if self._get_cache_size_mb() >= self.max_size_mb:
            self._cleanup_old_entries()
        
        cache_key = self._get_cache_key(file_path, options)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            self.metadata['entries'][cache_key] = {
                'file_path': str(file_path),
                'timestamp': datetime.now().isoformat(),
                'size_bytes': file_size,
                'options': options
            }
            
            self.metadata['total_size_bytes'] = sum(
                entry['size_bytes'] for entry in self.metadata['entries'].values()
            )
            
            self._save_metadata()
            logger.debug(f"Cached extraction for {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to cache extraction for {file_path.name}: {e}")
            if cache_file.exists():
                cache_file.unlink()
    
    def _remove_entry(self, cache_key: str):
        """Remove a cache entry"""
        if cache_key in self.metadata['entries']:
            # Remove file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            # Update metadata
            del self.metadata['entries'][cache_key]
            self.metadata['total_size_bytes'] = sum(
                entry['size_bytes'] for entry in self.metadata['entries'].values()
            )
            self._save_metadata()
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB"""
        return self.metadata['total_size_bytes'] / (1024 * 1024)
    
    def _cleanup_old_entries(self):
        """Remove old cache entries to free up space"""
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self.metadata['entries'].items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest entries until we're under 80% of max size
        target_size = self.max_size_mb * 0.8
        
        for cache_key, entry in sorted_entries:
            if self._get_cache_size_mb() <= target_size:
                break
            self._remove_entry(cache_key)
            logger.debug(f"Removed old cache entry: {entry['file_path']}")
    
    def clear(self):
        """Clear all cache entries"""
        logger.info("Clearing extraction cache")
        
        # Remove all cache files
        for cache_key in list(self.metadata['entries'].keys()):
            self._remove_entry(cache_key)
        
        # Reset metadata
        self.metadata = {
            'version': self.cache_version,
            'entries': {},
            'total_size_bytes': 0
        }
        self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'num_entries': len(self.metadata['entries']),
            'size_mb': self._get_cache_size_mb(),
            'max_size_mb': self.max_size_mb,
            'ttl_days': self.ttl_days,
            'cache_dir': str(self.cache_dir)
        }


# For backward compatibility
ExtractionCache = VersionedExtractionCache