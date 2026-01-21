# corpus_metadata/E_normalization/E04_pubtator_enricher.py
"""
PubTator3 API integration for disease enrichment.

Enriches extracted diseases with:
- MeSH identifiers (if missing)
- Normalized disease names
- Aliases/synonyms from PubTator

API Reference: https://www.ncbi.nlm.nih.gov/research/pubtator3/api

Example:
    >>> from E_normalization.E04_pubtator_enricher import DiseaseEnricher
    >>> enricher = DiseaseEnricher()
    >>> enriched = enricher.enrich(disease_entity)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from A_core.A00_logging import get_logger
from A_core.A02_interfaces import BaseEnricher
from A_core.A05_disease_models import (
    DiseaseIdentifier,
    ExtractedDisease,
)
from Z_utils.Z01_api_client import BaseAPIClient

logger = get_logger(__name__)


class PubTator3Client(BaseAPIClient):
    """
    PubTator3 API client with rate limiting and disk caching.

    Extends BaseAPIClient to leverage shared cache and rate limiting.

    Rate limit: 3 requests/second max (per NCBI guidelines).
    Cache: Disk-based with 7-day TTL (configurable).

    Attributes:
        base_url: PubTator3 API base URL.
        cache: DiskCache instance for caching responses.
        rate_limiter: RateLimiter instance for API rate limiting.
    """

    DEFAULT_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            config=config,
            service_name="pubtator",
            default_base_url=self.DEFAULT_BASE_URL,
            default_rate_limit=3.0,  # NCBI guideline: 3 req/sec
            default_cache_ttl_hours=168,  # 7 days
            default_cache_dir="cache/pubtator",
        )

    def autocomplete(
        self, term: str, entity_type: str = "disease"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Query PubTator3 autocomplete endpoint for entity normalization.

        Args:
            term: Disease name to search.
            entity_type: Entity type ("disease", "chemical", "gene", etc.).

        Returns:
            List of matching entities with identifiers, or None on error.
            Each entity has:
              - name: Normalized name
              - db_id: MeSH database ID
              - db: Database source (ncbi_mesh)
              - biotype: Entity type
              - _id: PubTator ID
        """
        if not term or not term.strip():
            return None

        # Check cache
        cache_key = self.cache.make_key(f"autocomplete_{entity_type}", term)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached if isinstance(cached, list) else [cached]

        # Rate limit and make request
        self.rate_limiter.wait()

        url = f"{self.base_url}/entity/autocomplete/"
        params = {"query": term}

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json()

            # Filter by entity type (biotype field)
            if isinstance(results, list):
                filtered = [
                    r
                    for r in results
                    if r.get("biotype", "").lower() == entity_type.lower()
                ]
                self.cache.set(cache_key, filtered)
                return filtered

            # Handle non-list results (wrap dict in list or return None)
            if isinstance(results, dict):
                wrapped = [results]
                self.cache.set(cache_key, wrapped)
                return wrapped
            return None

        except requests.exceptions.Timeout:
            logger.warning(f"PubTator timeout for '{term}'")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"PubTator request failed for '{term}': {e}")
            return None
        except json.JSONDecodeError:
            logger.warning(f"PubTator invalid JSON for '{term}'")
            return None

    def search_entity(
        self, term: str, entity_type: str = "disease"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search PubTator3 for entities using the search endpoint.

        Alternative to autocomplete for more comprehensive results.

        Args:
            term: Entity name to search.
            entity_type: "disease", "chemical", "gene", "species", etc.

        Returns:
            List of matching entities or None on error.
        """
        if not term or not term.strip():
            return None

        # Check cache
        cache_key = self.cache.make_key(f"search_{entity_type}", term)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached if isinstance(cached, list) else [cached]

        # Rate limit and make request
        self.rate_limiter.wait()

        type_prefix = entity_type.upper()
        query = f"@{type_prefix}_{term}"

        url = f"{self.base_url}/search/"
        params = {"text": query}

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Extract results ensuring list type
            if isinstance(data, dict):
                results = data.get("results", [])
            elif isinstance(data, list):
                results = data
            else:
                return None

            if not isinstance(results, list):
                results = [results] if isinstance(results, dict) else []

            self.cache.set(cache_key, results)
            return results

        except requests.exceptions.RequestException as e:
            logger.warning(f"PubTator search failed for '{term}': {e}")
            return None


class DiseaseEnricher(BaseEnricher[ExtractedDisease, ExtractedDisease]):
    """
    Enriches ExtractedDisease entities with PubTator3 data.

    Enrichment includes:
    - MeSH ID (if missing)
    - Normalized name from PubTator
    - Aliases/synonyms

    Attributes:
        client: PubTator3Client for API access.
        enrich_missing_mesh: Add MeSH IDs to diseases missing them.
        add_aliases: Add aliases from PubTator.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.client = PubTator3Client(self.config)

        # Enrichment settings
        enrichment_cfg = self.config.get("enrichment", {})
        self.enrich_missing_mesh = enrichment_cfg.get("enrich_missing_mesh", True)
        self.add_aliases = enrichment_cfg.get("add_aliases", True)

        logger.debug(
            f"DiseaseEnricher initialized: mesh={self.enrich_missing_mesh}, "
            f"aliases={self.add_aliases}, enabled={self.enabled}"
        )

    @property
    def enricher_name(self) -> str:
        """Return the enricher identifier."""
        return "disease_enricher"

    def enrich(self, disease: ExtractedDisease) -> ExtractedDisease:
        """
        Enrich a single disease entity with PubTator data.

        Args:
            disease: Validated disease entity

        Returns:
            Enriched disease entity (or original if enrichment fails/unnecessary)
        """
        if not self.enabled:
            return disease

        # Skip if already has MeSH and we're not adding aliases
        if disease.mesh_id and not self.add_aliases:
            return disease

        # Query PubTator using preferred label
        results = self.client.autocomplete(disease.preferred_label, "disease")

        # Try abbreviation if no results
        if not results and disease.abbreviation:
            results = self.client.autocomplete(disease.abbreviation, "disease")

        if not results:
            return disease

        # Get best match (first result)
        best = results[0]

        # Prepare updates
        updates: Dict[str, Any] = {}
        new_identifiers = list(disease.identifiers)

        # Add MeSH ID if missing
        # PubTator3 returns db_id for MeSH ID (e.g., "D000081029")
        if self.enrich_missing_mesh and not disease.mesh_id:
            mesh_id = best.get("db_id", "")
            if mesh_id and best.get("db") == "ncbi_mesh":
                updates["mesh_id"] = mesh_id

                # Add to identifiers list
                new_identifiers.append(
                    DiseaseIdentifier(
                        system="MeSH",
                        code=mesh_id,
                        display=best.get("name"),
                    )
                )
                updates["identifiers"] = new_identifiers

        # Add normalized name from PubTator
        if self.add_aliases:
            normalized_name = best.get("name")
            if normalized_name:
                updates["pubtator_normalized_name"] = normalized_name

            # PubTator3 autocomplete doesn't return aliases directly,
            # but we could populate from similar matches if needed
            # For now, we store the PubTator ID which can be used for future lookups
            pubtator_id = best.get("_id", "")
            if pubtator_id:
                # Extract potential aliases from related entries
                other_names = [
                    r.get("name")
                    for r in results[1:4]  # Get up to 3 related names
                    if r.get("name") and r.get("name") != normalized_name
                ]
                if other_names:
                    updates["mesh_aliases"] = other_names

        # Mark as enriched
        if updates:
            updates["enrichment_source"] = "pubtator3"

            # Add flag
            current_flags = list(disease.validation_flags or [])
            if "pubtator_enriched" not in current_flags:
                current_flags.append("pubtator_enriched")
                updates["validation_flags"] = current_flags

            return disease.model_copy(update=updates)

        return disease

    def enrich_batch(
        self, diseases: List[ExtractedDisease], verbose: bool = True
    ) -> List[ExtractedDisease]:
        """
        Enrich a batch of disease entities.

        Args:
            diseases: List of validated disease entities.
            verbose: Log progress information.

        Returns:
            List of enriched disease entities.
        """
        if not self.enabled:
            return diseases

        enriched = []
        enriched_count = 0

        for disease in diseases:
            result = self.enrich(disease)
            enriched.append(result)

            if "pubtator_enriched" in (result.validation_flags or []):
                if "pubtator_enriched" not in (disease.validation_flags or []):
                    enriched_count += 1

        if verbose and enriched_count > 0:
            logger.info(f"PubTator enriched: {enriched_count}/{len(diseases)} diseases")

        return enriched
