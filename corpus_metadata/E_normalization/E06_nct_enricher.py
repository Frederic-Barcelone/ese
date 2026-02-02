# corpus_metadata/E_normalization/E06_nct_enricher.py
"""
ClinicalTrials.gov API integration for NCT identifier enrichment.

Enriches extracted NCT identifiers with:
- Official trial title (long form expansion)
- Brief title
- Trial acronym (if available)
- Conditions being studied
- Interventions/drugs

API Reference: https://clinicaltrials.gov/data-api/api

Example:
    >>> from E_normalization.E06_nct_enricher import NCTEnricher
    >>> enricher = NCTEnricher()
    >>> info = enricher.enrich("NCT04817618")
    >>> print(info.official_title)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from A_core.A00_logging import get_logger
from A_core.A02_interfaces import BaseEnricher
from Z_utils.Z01_api_client import BaseAPIClient

logger = get_logger(__name__)


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


class ClinicalTrialsGovClient(BaseAPIClient):
    """
    ClinicalTrials.gov API v2 client with rate limiting and disk caching.

    Extends BaseAPIClient to leverage shared cache and rate limiting.

    Rate limit: 1 request/second (conservative, API allows more).
    Cache: Disk-based with 30-day TTL (trial data changes infrequently).

    Attributes:
        base_url: ClinicalTrials.gov API base URL.
        cache: DiskCache instance for caching responses.
        rate_limiter: RateLimiter instance for API rate limiting.
    """

    DEFAULT_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Convert ttl_days to ttl_hours for BaseAPIClient
        config = config or {}
        cache_cfg = config.get("cache", {})
        if "ttl_days" in cache_cfg and "ttl_hours" not in cache_cfg:
            cache_cfg["ttl_hours"] = cache_cfg["ttl_days"] * 24

        super().__init__(
            config=config,
            service_name="clinicaltrials",
            default_base_url=self.DEFAULT_BASE_URL,
            default_rate_limit=1.0,  # Conservative rate limit
            default_cache_ttl_hours=720,  # 30 days
            default_cache_dir="cache/clinicaltrials",
        )

    def _make_cache_key(self, nct_id: str) -> str:
        """Generate cache key from NCT ID."""
        normalized = nct_id.upper().strip()
        if not normalized.startswith("NCT"):
            normalized = f"NCT{normalized}"
        return f"nct_{normalized}"

    def get_trial_info(self, nct_id: str) -> Optional[NCTTrialInfo]:
        """
        Fetch trial information from ClinicalTrials.gov API.

        Args:
            nct_id: NCT identifier (e.g., "NCT04817618" or "04817618").

        Returns:
            NCTTrialInfo with trial details, or None on error/not found.
        """
        # Normalize NCT ID
        nct_id = nct_id.upper().strip()
        if not nct_id.startswith("NCT"):
            nct_id = f"NCT{nct_id}"

        # Check cache
        cache_key = self._make_cache_key(nct_id)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return NCTTrialInfo(**cached) if cached else None

        # Rate limit and make request
        self.rate_limiter.wait()

        params = {
            "format": "json",
            "query.id": nct_id,
            "fields": "NCTId|OfficialTitle|BriefTitle|Acronym|Condition|InterventionName|OverallStatus|Phase",
        }

        try:
            resp = self._session.get(self.base_url, params=params, timeout=self.timeout)

            # Handle rate limiting
            if resp.status_code == 429:
                logger.warning("ClinicalTrials.gov rate limited, waiting 60s...")
                time.sleep(60)
                resp = self._session.get(self.base_url, params=params, timeout=self.timeout)

            resp.raise_for_status()
            data = resp.json()

        except requests.RequestException as e:
            logger.warning(f"ClinicalTrials.gov API error for {nct_id}: {e}")
            self.cache.set(cache_key, None)
            return None

        # Parse response
        studies = data.get("studies", [])
        if not studies:
            self.cache.set(cache_key, None)
            return None

        study = studies[0]
        trial_info = self._parse_study(study, nct_id)

        # Cache the result
        if trial_info:
            self.cache.set(cache_key, {
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
        except (KeyError, TypeError) as e:
            # JSON structure parsing error
            logger.warning(f"ClinicalTrials.gov parse error for {nct_id} (invalid structure): {e}")
            return None
        except ValueError as e:
            # Data conversion error
            logger.warning(f"ClinicalTrials.gov parse error for {nct_id} (invalid value): {e}")
            return None


class NCTEnricher(BaseEnricher[str, Optional[NCTTrialInfo]]):
    """
    Enriches NCT identifiers with trial information from ClinicalTrials.gov.

    Attributes:
        client: ClinicalTrialsGovClient for API access.

    Example:
        >>> enricher = NCTEnricher()
        >>> info = enricher.enrich("NCT04817618")
        >>> if info:
        ...     print(f"Trial: {info.long_form}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.client = ClinicalTrialsGovClient(self.config)
        logger.debug(f"NCTEnricher initialized: enabled={self.enabled}")

    @property
    def enricher_name(self) -> str:
        """Return the enricher identifier."""
        return "nct_enricher"

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


# =============================================================================
# TRIAL ACRONYM ENRICHMENT
# =============================================================================


class TrialAcronymEnricher(BaseAPIClient):
    """
    Enriches trial acronyms (e.g., APPEAR-C3G, NEPTUNE) with full trial descriptions.

    Extends BaseAPIClient to leverage shared cache and rate limiting.
    Searches ClinicalTrials.gov by acronym to find the official trial title.

    Attributes:
        base_url: ClinicalTrials.gov API base URL.
        cache: DiskCache instance for caching responses.
        rate_limiter: RateLimiter instance for API rate limiting.

    Example:
        >>> enricher = TrialAcronymEnricher()
        >>> info = enricher.search_by_acronym("APPEAR-C3G")
        >>> if info:
        ...     print(f"Trial: {info.official_title}")
    """

    DEFAULT_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Convert ttl_days to ttl_hours for BaseAPIClient
        config = config or {}
        cache_cfg = config.get("cache", {})
        if "ttl_days" in cache_cfg and "ttl_hours" not in cache_cfg:
            cache_cfg["ttl_hours"] = cache_cfg["ttl_days"] * 24

        super().__init__(
            config=config,
            service_name="trial_acronym",
            default_base_url=self.DEFAULT_BASE_URL,
            default_rate_limit=1.0,
            default_cache_ttl_hours=720,  # 30 days
            default_cache_dir="cache/clinicaltrials",
        )

    def _make_cache_key(self, acronym: str) -> str:
        """Generate cache key from acronym."""
        normalized = acronym.upper().strip().replace(" ", "_")
        return f"acronym_{normalized}"

    def search_by_acronym(self, acronym: str) -> Optional[NCTTrialInfo]:
        """
        Search ClinicalTrials.gov for a trial by its acronym.

        Args:
            acronym: Trial acronym (e.g., "APPEAR-C3G", "NEPTUNE", "FABHALTA").

        Returns:
            NCTTrialInfo with trial details, or None if not found.
        """
        if not acronym or len(acronym) < 2:
            return None

        acronym = acronym.strip()

        # Check cache
        cache_key = self._make_cache_key(acronym)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return NCTTrialInfo(**cached) if cached else None

        # Rate limit and make request
        self.rate_limiter.wait()

        params = {
            "format": "json",
            "query.term": f"AREA[Acronym]{acronym}",
            "fields": "NCTId|OfficialTitle|BriefTitle|Acronym|Condition|InterventionName|OverallStatus|Phase",
            "pageSize": 5,
        }

        try:
            resp = self._session.get(self.base_url, params=params, timeout=self.timeout)

            if resp.status_code == 429:
                logger.warning("TrialAcronym rate limited, waiting 60s...")
                time.sleep(60)
                resp = self._session.get(self.base_url, params=params, timeout=self.timeout)

            resp.raise_for_status()
            data = resp.json()

        except requests.RequestException as e:
            logger.warning(f"TrialAcronym API error for '{acronym}': {e}")
            self.cache.set(cache_key, None)
            return None

        # Find best match from results
        studies = data.get("studies", [])
        if not studies:
            # Try alternative search without AREA prefix
            params["query.term"] = acronym
            try:
                resp = self._session.get(self.base_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                studies = data.get("studies", [])
            except requests.RequestException as e:
                # Fallback search failed - log at debug level and continue
                logger.debug(f"TrialAcronym fallback search failed for '{acronym}': {e}")

        if not studies:
            self.cache.set(cache_key, None)
            return None

        # Find exact acronym match
        best_match = None
        for study in studies:
            protocol = study.get("protocolSection", {})
            id_mod = protocol.get("identificationModule", {})
            study_acronym = (id_mod.get("acronym") or "").strip()

            if study_acronym.upper() == acronym.upper():
                best_match = study
                break

        # Fall back to first result if no exact match
        if not best_match:
            best_match = studies[0]

        # Parse the study
        trial_info = self._parse_study(best_match, acronym)

        # Cache result
        if trial_info:
            self.cache.set(cache_key, {
                "nct_id": trial_info.nct_id,
                "official_title": trial_info.official_title,
                "brief_title": trial_info.brief_title,
                "acronym": trial_info.acronym,
                "conditions": trial_info.conditions,
                "interventions": trial_info.interventions,
                "status": trial_info.status,
                "phase": trial_info.phase,
            })
        else:
            self.cache.set(cache_key, None)

        return trial_info

    def _parse_study(self, study: Dict[str, Any], search_acronym: str) -> Optional[NCTTrialInfo]:
        """Parse study data from API response."""
        try:
            protocol = study.get("protocolSection", {})
            id_mod = protocol.get("identificationModule", {})
            cond_mod = protocol.get("conditionsModule", {})
            arms_mod = protocol.get("armsInterventionsModule", {})
            status_mod = protocol.get("statusModule", {})
            design_mod = protocol.get("designModule", {})

            interventions = []
            for intervention in arms_mod.get("interventions", []):
                name = (intervention.get("name") or "").strip()
                if name:
                    interventions.append(name)

            phases = design_mod.get("phases", [])
            phase = ", ".join(phases) if phases else None

            return NCTTrialInfo(
                nct_id=id_mod.get("nctId") or "",
                official_title=(id_mod.get("officialTitle") or "").strip() or None,
                brief_title=(id_mod.get("briefTitle") or "").strip() or None,
                acronym=(id_mod.get("acronym") or search_acronym).strip() or None,
                conditions=cond_mod.get("conditions") or None,
                interventions=interventions or None,
                status=status_mod.get("overallStatus") or None,
                phase=phase,
            )
        except (KeyError, TypeError) as e:
            # JSON structure parsing error
            logger.warning(f"TrialAcronym parse error for '{search_acronym}' (invalid structure): {e}")
            return None
        except ValueError as e:
            # Data conversion error
            logger.warning(f"TrialAcronym parse error for '{search_acronym}' (invalid value): {e}")
            return None

    def get_trial_description(self, acronym: str) -> Optional[str]:
        """
        Get the official trial title/description for a trial acronym.

        Args:
            acronym: Trial acronym (e.g., "APPEAR-C3G")

        Returns:
            Official trial title, or None if not found.

        Example:
            >>> desc = enricher.get_trial_description("APPEAR-C3G")
            >>> print(desc)
            "A Study of Iptacopan in Patients With C3 Glomerulopathy"
        """
        info = self.search_by_acronym(acronym)
        if info:
            return info.official_title or info.brief_title
        return None


# Module-level trial acronym enricher instance (lazy initialization)
_trial_acronym_enricher: Optional[TrialAcronymEnricher] = None


def get_trial_acronym_enricher(config: Optional[Dict[str, Any]] = None) -> TrialAcronymEnricher:
    """Get or create the module-level trial acronym enricher instance."""
    global _trial_acronym_enricher
    if _trial_acronym_enricher is None:
        _trial_acronym_enricher = TrialAcronymEnricher(config)
    return _trial_acronym_enricher


def enrich_trial_acronym(acronym: str, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get the official trial title for a trial acronym.

    Searches ClinicalTrials.gov to find the full trial description.

    Args:
        acronym: Trial acronym (e.g., "APPEAR-C3G", "NEPTUNE")
        config: Optional configuration dict

    Returns:
        Official trial title, or None if not found.

    Example:
        >>> title = enrich_trial_acronym("APPEAR-C3G")
        >>> print(title)
        "A Phase 3, Multicenter, Randomized Study of Iptacopan..."
    """
    enricher = get_trial_acronym_enricher(config)
    return enricher.get_trial_description(acronym)
