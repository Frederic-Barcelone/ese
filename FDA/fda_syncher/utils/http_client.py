"""
Simple HTTP Client with retry - OPTIMIZED VERSION v2.0
FDA/fda_syncher/utils/http_client.py

FIXED: 
- Smarter retry logic that doesn't waste time on 404s
- Connection pool recycling to prevent stale connections
- Adaptive rate limiting based on error patterns
"""

import requests
import time
import urllib3

# Import from YOUR config file
from syncher_keys import REQUEST_TIMEOUT, RATE_LIMIT_DELAY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SimpleHTTPClient:
    """Simple HTTP client with intelligent retry and connection management"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.session = requests.Session()
        self.request_count = 0  # Track total requests
        self.consecutive_errors = 0  # Track error patterns
    
    def _recycle_session_if_needed(self):
        """Recycle connection pool every 100 requests to prevent stale connections"""
        self.request_count += 1
        if self.request_count % 100 == 0:
            print(f"    ♻️  Recycled connection pool (after {self.request_count} requests)")
            self.session.close()
            self.session = requests.Session()
    
    def _adaptive_delay(self):
        """Apply adaptive delay based on error rate"""
        base_delay = RATE_LIMIT_DELAY
        
        if self.consecutive_errors > 5:
            # Exponential backoff for repeated errors
            delay = min(base_delay * (2 ** (self.consecutive_errors - 5)), 30)
            print(f"    ⏸️  Adaptive cooling: {delay:.1f}s (consecutive errors: {self.consecutive_errors})")
            time.sleep(delay)
        else:
            time.sleep(base_delay)
    
    def get(self, url, **kwargs):
        """GET with intelligent retry
        
        - Network errors: Full retry with backoff
        - 404 Not Found: Single quick retry only
        - Other errors: Full retry with adaptive delays
        """
        kwargs.setdefault('timeout', REQUEST_TIMEOUT)
        kwargs.setdefault('verify', False)
        
        # Recycle connection pool periodically
        self._recycle_session_if_needed()
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, **kwargs)
                
                # If it's a 404, don't keep retrying - it's not going to appear
                if response.status_code == 404:
                    if attempt == 0:
                        # Try once more in case it's a transient issue
                        time.sleep(0.5)
                        continue
                    else:
                        # Give up on 404s after one retry
                        self.consecutive_errors = 0  # Reset - 404 is expected
                        response.raise_for_status()
                
                response.raise_for_status()
                
                # Success - reset error counter and apply normal delay
                self.consecutive_errors = 0
                self._adaptive_delay()
                return response
                
            except requests.exceptions.HTTPError as e:
                # For 404s, fail fast after one retry
                if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                    if attempt >= 1:  # Already tried twice
                        raise
                
                # For other HTTP errors, use full retry logic
                self.consecutive_errors += 1
                
                if attempt == self.max_retries - 1:
                    raise
                
                wait = (2 ** attempt)
                print(f"      Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
                
            except Exception as e:
                # For network errors, use full retry logic with adaptive delay
                self.consecutive_errors += 1
                
                if attempt == self.max_retries - 1:
                    raise
                
                wait = (2 ** attempt)
                print(f"      Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
    
    def download_file(self, url, filepath):
        """Download file with streaming"""
        response = self.get(url, stream=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return filepath