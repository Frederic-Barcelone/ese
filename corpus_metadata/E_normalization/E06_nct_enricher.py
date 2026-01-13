# corpus_metadata/corpus_metadata/E_normalization/E06_nct_enricher.py
"""
ClinicalTrials.gov API integration for NCT identifier enrichment.

Enriches extracted NCT identifiers with:
- Official trial title (long form expansion)
- Brief title
- Trial acronym (if available)
- Conditions being studied
- Interventions/drugs

API Reference: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


@dataclass
class NCTTrialInfo:
    """Information retrieved from ClinicalTrials.gov for an NCT ID."""

    nct_id: str
    official_title: Optional[str] = None
    brief_title: Optional[str] = None
    acronym: Optional[str] = None
    conditions: Optional[list[str]] = None
    interventions: Optional[list[str]] = None
    status: Optional[str] = None
    phase: Optional[str] = None

    @property
    def long_form(self) -> Optional[str]:
        """Get the best available title as long form."""
        return self.official_title or self.brief_title


class ClinicalTrialsGovClient:
    """
    ClinicalTrials.gov API v2 client with rate limiting and disk caching.

    Rate limit: 1 request/second (conservative, API allows more).
    Cache: Disk-based with 30-day TTL (trial data changes infrequently).
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.timeout = config.get("timeout_seconds", 30)
        self.rate_limit = config.get("rate_limit_per_second", 1)

        # Cache configuration
        cache_cfg = config.get("cache", {})
        self.cache_enabled = cache_cfg.get("enabled", True)
        self.cache_dir = Path(cache_cfg.get("directory", "cache/clinicaltrials"))
        self.cache_ttl = cache_cfg.get("ttl_days", 30) * 86400  # Default 30 days

        # Rate limiting state
        self._last_request_time = 0.0

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "ESE-Pipeline/1.0 (NCT Enrichment)"

        # Create cache directory if enabled
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        """Enforce rate limit."""
        if self.rate_limit <= 0:
            return
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, nct_id: str) -> str:
        """Generate cache key from NCT ID."""
        normalized = nct_id.upper().strip()
        # Ensure NCT prefix
        if not normalized.startswith("NCT"):
            normalized = f"NCT{normalized}"
        return f"nct_{normalized}"

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if exists and not expired."""
        if not self.cache_enabled:
            return None

        path = self._cache_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            timestamp = data.get("_cached_at", 0)
            if time.time() - timestamp > self.cache_ttl:
                path.unlink()  # Expired, remove
                return None
            return data.get("result")
        except Exception:
            return None

    def _cache_set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        if not self.cache_enabled:
            return

        path = self._cache_path(key)
        try:
            data = {"_cached_at": time.time(), "result": value}
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # Cache write failures are non-fatal

    def get_trial_info(self, nct_id: str) -> Optional[NCTTrialInfo]:
        """
        Fetch trial information from ClinicalTrials.gov API.

        Args:
            nct_id: NCT identifier (e.g., "NCT04817618" or "04817618")

        Returns:
            NCTTrialInfo with trial details, or None on error/not found.
        """
        # Normalize NCT ID
        nct_id = nct_id.upper().strip()
        if not nct_id.startswith("NCT"):
            nct_id = f"NCT{nct_id}"

        # Check cache
        cache_key = self._cache_key(nct_id)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return NCTTrialInfo(**cached) if cached else None

        # Rate limit
        self._rate_limit()

        # Build API request
        params = {
            "format": "json",
            "query.id": nct_id,
            "fields": "NCTId|OfficialTitle|BriefTitle|Acronym|Condition|InterventionName|OverallStatus|Phase",
        }

        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)

            # Handle rate limiting
            if resp.status_code == 429:
                print(f"[NCT] Rate limited, waiting 60s...")
                time.sleep(60)
                resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)

            resp.raise_for_status()
            data = resp.json()

        except requests.RequestException as e:
            print(f"[NCT] API error for {nct_id}: {e}")
            # Cache the failure (short TTL) to avoid repeated failed requests
            self._cache_set(cache_key, None)
            return None

        # Parse response
        studies = data.get("studies", [])
        if not studies:
            # Trial not found - cache this to avoid repeated lookups
            self._cache_set(cache_key, None)
            return None

        study = studies[0]
        trial_info = self._parse_study(study, nct_id)

        # Cache the result
        if trial_info:
            self._cache_set(cache_key, {
                "nct_id": trial_info.nct_id,
                "official_title": trial_info.official_title,
                "brief_title": trial_info.brief_title,
                "acronym": trial_info.acronym,
                "conditions": trial_info.conditions,
                "interventions": trial_info.interventions,
                "status": trial_info.status,
                "phase": trial_info.phase,
            })

        return trial_info

    def _parse_study(self, study: Dict[str, Any], nct_id: str) -> Optional[NCTTrialInfo]:
        """Parse study data from API response."""
        try:
            protocol = study.get("protocolSection", {})
            id_mod = protocol.get("identificationModule", {})
            cond_mod = protocol.get("conditionsModule", {})
            arms_mod = protocol.get("armsInterventionsModule", {})
            status_mod = protocol.get("statusModule", {})
            design_mod = protocol.get("designModule", {})

            # Extract intervention names
            interventions = []
            for intervention in arms_mod.get("interventions", []):
                name = (intervention.get("name") or "").strip()
                if name:
                    interventions.append(name)

            # Extract phase(s)
            phases = design_mod.get("phases", [])
            phase = ", ".join(phases) if phases else None

            return NCTTrialInfo(
                nct_id=id_mod.get("nctId") or nct_id,
                official_title=(id_mod.get("officialTitle") or "").strip() or None,
                brief_title=(id_mod.get("briefTitle") or "").strip() or None,
                acronym=(id_mod.get("acronym") or "").strip() or None,
                conditions=cond_mod.get("conditions") or None,
                interventions=interventions or None,
                status=status_mod.get("overallStatus") or None,
                phase=phase,
            )
        except Exception as e:
            print(f"[NCT] Parse error for {nct_id}: {e}")
            return None


class NCTEnricher:
    """
    Enriches NCT identifiers with trial information from ClinicalTrials.gov.

    Usage:
        enricher = NCTEnricher()
        info = enricher.enrich("NCT04817618")
        if info:
            print(f"Trial: {info.long_form}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.client = ClinicalTrialsGovClient(config)

    def enrich(self, nct_id: str) -> Optional[NCTTrialInfo]:
        """
        Enrich an NCT ID with trial information.

        Args:
            nct_id: NCT identifier (with or without "NCT" prefix)

        Returns:
            NCTTrialInfo with trial details, or None if not found.
        """
        return self.client.get_trial_info(nct_id)

    def get_expansion(self, nct_id: str) -> Optional[str]:
        """
        Get the long form expansion (official title) for an NCT ID.

        Args:
            nct_id: NCT identifier

        Returns:
            Official trial title, or None if not found.
        """
        info = self.enrich(nct_id)
        return info.long_form if info else None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Module-level enricher instance (lazy initialization)
_enricher: Optional[NCTEnricher] = None


def get_nct_enricher(config: Optional[Dict[str, Any]] = None) -> NCTEnricher:
    """Get or create the module-level NCT enricher instance."""
    global _enricher
    if _enricher is None:
        _enricher = NCTEnricher(config)
    return _enricher


def enrich_nct_id(nct_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[NCTTrialInfo]:
    """
    Convenience function to enrich an NCT ID.

    Args:
        nct_id: NCT identifier (e.g., "NCT04817618")
        config: Optional configuration dict

    Returns:
        NCTTrialInfo with trial details, or None if not found.

    Example:
        >>> info = enrich_nct_id("NCT04817618")
        >>> print(info.official_title)
        "A Phase 3, Randomized, Double-Blind, Placebo-Controlled Study..."
    """
    enricher = get_nct_enricher(config)
    return enricher.enrich(nct_id)


def get_nct_expansion(nct_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get the official title expansion for an NCT ID.

    Args:
        nct_id: NCT identifier
        config: Optional configuration dict

    Returns:
        Official trial title, or None if not found.

    Example:
        >>> title = get_nct_expansion("NCT04817618")
        >>> print(title)
        "A Phase 3, Randomized, Double-Blind, Placebo-Controlled Study..."
    """
    enricher = get_nct_enricher(config)
    return enricher.get_expansion(nct_id)
