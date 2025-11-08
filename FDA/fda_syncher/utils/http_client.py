"""
Simple HTTP Client with retry - OPTIMIZED VERSION
FDA/fda_syncher/utils/http_client.py

FIXED: Smarter retry logic that doesn't waste time on 404s
"""

import requests
import time
import urllib3

# Import from YOUR config file
from syncher_keys import REQUEST_TIMEOUT, RATE_LIMIT_DELAY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SimpleHTTPClient:
    """Simple HTTP client with intelligent retry"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def get(self, url, **kwargs):
        """GET with intelligent retry
        
        - Network errors: Full retry with backoff
        - 404 Not Found: Single quick retry only
        - Other errors: Full retry
        """
        kwargs.setdefault('timeout', REQUEST_TIMEOUT)
        kwargs.setdefault('verify', False)
        
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
                        response.raise_for_status()
                
                response.raise_for_status()
                time.sleep(RATE_LIMIT_DELAY)  # Use config rate limit
                return response
                
            except requests.exceptions.HTTPError as e:
                # For 404s, fail fast after one retry
                if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                    if attempt >= 1:  # Already tried twice
                        raise
                
                # For other HTTP errors, use full retry logic
                if attempt == self.max_retries - 1:
                    raise
                wait = (2 ** attempt)
                print(f"      Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
                
            except Exception as e:
                # For network errors, use full retry logic
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