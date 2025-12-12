"""
Simple HTTP Client with retry - OPTIMIZED VERSION v2.1
FDA/fda_syncher/utils/http_client.py

FIXED: 
- Smarter retry logic that doesn't waste time on 404s
- Connection pool recycling to prevent stale connections
- Adaptive rate limiting based on error patterns
- Better timeout handling
"""

import requests
import time
import urllib3

# Import from YOUR config file
from syncher_keys import REQUEST_TIMEOUT, RATE_LIMIT_DELAY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SimpleHTTPClient:
    """Simple HTTP client with intelligent retry and connection management"""
    
    def __init__(self, max_retries=3, recycle_every=100):
        self.max_retries = max_retries
        self.recycle_every = recycle_every
        self.session = self._create_session()
        self.request_count = 0  # Track total requests
        self.consecutive_errors = 0  # Track error patterns
        self.last_request_time = 0  # For rate limiting
    
    def _create_session(self):
        """Create a new session with connection pooling"""
        session = requests.Session()
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0  # We handle retries ourselves
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def _recycle_session_if_needed(self):
        """Recycle connection pool periodically to prevent stale connections"""
        self.request_count += 1
        if self.request_count % self.recycle_every == 0:
            print(f"    ♻️  Recycled connection pool (after {self.request_count} requests)")
            self.session.close()
            self.session = self._create_session()
    
    def _adaptive_delay(self):
        """Apply adaptive delay based on error rate and rate limits"""
        base_delay = RATE_LIMIT_DELAY
        
        # Ensure minimum time between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < base_delay:
            time.sleep(base_delay - elapsed)
        
        # Extra delay if we're seeing errors
        if self.consecutive_errors > 5:
            # Exponential backoff for repeated errors (capped at 30s)
            extra_delay = min(base_delay * (2 ** (self.consecutive_errors - 5)), 30)
            print(f"    ⏸️  Adaptive cooling: {extra_delay:.1f}s (consecutive errors: {self.consecutive_errors})")
            time.sleep(extra_delay)
        
        self.last_request_time = time.time()
    
    def get(self, url, **kwargs):
        """GET with intelligent retry
        
        - Network errors: Full retry with backoff
        - 404 Not Found: Single quick retry only (expected for many queries)
        - Rate limit (429): Wait and retry
        - Other errors: Full retry with adaptive delays
        """
        kwargs.setdefault('timeout', REQUEST_TIMEOUT)
        kwargs.setdefault('verify', False)  # For corporate networks with SSL inspection
        
        # Recycle connection pool periodically
        self._recycle_session_if_needed()
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, **kwargs)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"      ⏳ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Handle 404 - don't keep retrying, it won't appear
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
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 404:
                        if attempt >= 1:  # Already tried twice
                            raise
                
                # For other HTTP errors, use full retry logic
                self.consecutive_errors += 1
                
                if attempt == self.max_retries - 1:
                    raise
                
                wait = (2 ** attempt)
                print(f"      Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
            
            except requests.exceptions.Timeout:
                # Timeout - retry with longer wait
                self.consecutive_errors += 1
                
                if attempt == self.max_retries - 1:
                    raise
                
                wait = (2 ** attempt) * 2  # Longer wait for timeouts
                print(f"      Timeout. Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
                
            except requests.exceptions.ConnectionError:
                # Connection error - recycle session and retry
                self.consecutive_errors += 1
                self.session.close()
                self.session = self._create_session()
                
                if attempt == self.max_retries - 1:
                    raise
                
                wait = (2 ** attempt) * 2
                print(f"      Connection error. Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
                
            except Exception as e:
                # For other errors, use full retry logic with adaptive delay
                self.consecutive_errors += 1
                
                if attempt == self.max_retries - 1:
                    raise
                
                wait = (2 ** attempt)
                print(f"      Retry {attempt+1}/{self.max_retries} in {wait}s...")
                time.sleep(wait)
    
    def download_file(self, url, filepath, chunk_size=8192):
        """Download file with streaming and progress"""
        response = self.get(url, stream=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        return filepath
    
    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
