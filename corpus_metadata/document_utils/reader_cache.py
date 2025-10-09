#!/usr/bin/env python3
"""
Document Cache Module
====================
Provides caching functionality for document reader.
/Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/reader_cache.py
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)


class DocumentCache:
    """Simple cache for document reading results"""
    
    def __init__(self, cache_config: Optional[Any] = None):
        """
        Initialize document cache
        
        Args:
            cache_config: Optional cache configuration object
        """
        if cache_config:
            self.enabled = getattr(cache_config, 'enabled', True)
            self.cache_dir = Path(getattr(cache_config, 'cache_dir', '.cache/documents'))
            self.ttl_seconds = getattr(cache_config, 'ttl_seconds', 3600)
            self.max_size_mb = getattr(cache_config, 'max_size_mb', 500)
        else:
            # Default settings
            self.enabled = True
            self.cache_dir = Path('.cache/documents')
            self.ttl_seconds = 3600
            self.max_size_mb = 500
        
        # Create cache directory
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_file = self.cache_dir / 'cache_metadata.json'
            self.metadata = self._load_metadata()
        else:
            self.metadata = {}
        
        logger.info(f"DocumentCache initialized: enabled={self.enabled}, dir={self.cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {'entries': {}}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a file"""
        # Use file path and modification time
        key_parts = [
            str(file_path.absolute()),
            str(file_path.stat().st_mtime) if file_path.exists() else "0",
            str(file_path.stat().st_size) if file_path.exists() else "0"
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached result for a file"""
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(file_path)
        
        # Check if in metadata
        if cache_key not in self.metadata.get('entries', {}):
            return None
        
        entry = self.metadata['entries'][cache_key]
        
        # Check TTL
        entry_time = datetime.fromisoformat(entry['timestamp'])
        if datetime.now() - entry_time > timedelta(seconds=self.ttl_seconds):
            logger.debug(f"Cache expired for {file_path.name}")
            self._remove_entry(cache_key)
            return None
        
        # Load cached data
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
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
    
    def set(self, file_path: Path, data: Dict[str, Any]):
        """Cache result for a file"""
        if not self.enabled:
            return
        
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.metadata.setdefault('entries', {})[cache_key] = {
                'file_path': str(file_path),
                'timestamp': datetime.now().isoformat(),
                'size_bytes': cache_file.stat().st_size
            }
            
            self._save_metadata()
            logger.debug(f"Cached result for {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to cache result for {file_path.name}: {e}")
            if cache_file.exists():
                cache_file.unlink()
    
    def _remove_entry(self, cache_key: str):
        """Remove a cache entry"""
        # Remove file
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        # Update metadata
        if 'entries' in self.metadata and cache_key in self.metadata['entries']:
            del self.metadata['entries'][cache_key]
            self._save_metadata()
    
    def clear(self):
        """Clear all cache entries"""
        if not self.enabled:
            return
        
        logger.info("Clearing document cache")
        
        # Remove all cache files
        for cache_key in list(self.metadata.get('entries', {}).keys()):
            self._remove_entry(cache_key)
        
        # Reset metadata
        self.metadata = {'entries': {}}
        self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        total_size = sum(
            entry.get('size_bytes', 0) 
            for entry in self.metadata.get('entries', {}).values()
        )
        
        return {
            'enabled': True,
            'num_entries': len(self.metadata.get('entries', {})),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }