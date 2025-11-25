#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS HTTP Module
Handles HTTP requests, rate limiting, and retries
ctis/ctis_http.py
"""

import time
import json
import threading
import requests
from typing import Dict, Any
from ctis_config import (
    BASE_HEADERS, MAX_RETRIES, REQUEST_TIMEOUT, PORTAL_URL
)
from ctis_utils import log, sleep_jitter, backoff

# ===================== Rate Limiter =====================

class RateLimiter:
    """Simple thread-safe leaky bucket (interval) limiter"""
    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / max(rate_per_sec, 0.001)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_for = self._next - now
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._next = now + self.interval


# Global rate limiter (set by caller)
GLOBAL_RATE_LIMITER = None


# ===================== HTTP Helpers =====================

def warm_up(session: requests.Session):
    """
    Warm up HTTP connection
    
    Note: For future stability improvements, consider using the "Download clinical trial" 
    feature from the portal (HTML file) instead of dynamic DOM parsing. This provides
    a more stable data source as documented in CTIS Full trial information guide.
    Implementation would require BeautifulSoup for HTML parsing.
    """
    try:
        session.get(PORTAL_URL, timeout=30)
        sleep_jitter()
    except Exception:
        pass


def _ensure_json_response(resp: requests.Response) -> Dict[str, Any]:
    """Validate and parse JSON response"""
    ctype = resp.headers.get("Content-Type", "")
    if "json" not in ctype and "text/plain" not in ctype:
        try:
            return resp.json()
        except Exception:
            raise ValueError(f"Unexpected Content-Type '{ctype}' and body is not valid JSON")
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON body: {e}") from e


# ===================== HTTP Request =====================

def req(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """
    Hardened HTTP with rate-limit, jitter, and backoff for common transient failures
    """
    headers = dict(session.headers)
    headers.update(kwargs.pop("headers", {}))
    timeout = float(kwargs.pop("timeout", REQUEST_TIMEOUT))

    for i in range(MAX_RETRIES):
        try:
            if GLOBAL_RATE_LIMITER:
                GLOBAL_RATE_LIMITER.wait()
            sleep_jitter()
            r = session.request(method, url, headers=headers, timeout=timeout, **kwargs)

            if r.status_code == 403:
                log("Received 403; warming up and backing off...", "WARN")
                warm_up(session)
                backoff(i)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                log(f"Transient HTTP {r.status_code} from {url}; retrying...", "WARN")
                backoff(i)
                continue

            r.raise_for_status()
            return r

        except (requests.Timeout, requests.ConnectionError) as e:
            log(f"Network error on {method} {url}: {e!r} - retrying...", "WARN")
            backoff(i)
        except requests.RequestException as e:
            log(f"HTTP error on {method} {url}: {e!r}", "ERROR")
            raise

    # Final retry with increased timeout
    warm_up(session)
    if GLOBAL_RATE_LIMITER:
        GLOBAL_RATE_LIMITER.wait()
    r = session.request(method, url, headers=headers, timeout=timeout * 1.5, **kwargs)
    r.raise_for_status()
    return r


# ===================== Session Setup =====================

def create_session() -> requests.Session:
    """Create configured HTTP session"""
    session = requests.Session()
    session.trust_env = True
    session.headers.update(BASE_HEADERS)
    
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=30,
        pool_maxsize=60,
        max_retries=0
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session