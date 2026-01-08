# corpus_metadata/corpus_metadata/E_normalization/E04_pubtator_enricher.py
"""
PubTator3 API integration for disease enrichment.

Enriches extracted diseases with:
- MeSH identifiers (if missing)
- Normalized disease names
- Aliases/synonyms from PubTator

API Reference: https://www.ncbi.nlm.nih.gov/research/pubtator3/api
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from A_core.A05_disease_models import (
    DiseaseIdentifier,
    ExtractedDisease,
)


class PubTator3Client:
    """
    PubTator3 API client with rate limiting and disk caching.

    Rate limit: 3 requests/second max (per NCBI guidelines).
    Cache: Disk-based with configurable TTL.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.base_url = config.get(
            "base_url", "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
        )
        self.timeout = config.get("timeout_seconds", 30)
        self.rate_limit = config.get("rate_limit_per_second", 3)

        # Cache configuration
        cache_cfg = config.get("cache", {})
        self.cache_enabled = cache_cfg.get("enabled", True)
        self.cache_dir = Path(cache_cfg.get("directory", "cache/pubtator"))
        self.cache_ttl = cache_cfg.get("ttl_hours", 168) * 3600  # Default 7 days

        # Rate limiting state
        self._last_request_time = 0.0

        # Create cache directory if enabled
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        """Enforce rate limit (3 req/sec max per NCBI guidelines)."""
        if self.rate_limit <= 0:
            return
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, prefix: str, term: str) -> str:
        """Generate cache key from prefix and term."""
        normalized = term.lower().strip()
        hash_val = hashlib.md5(normalized.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}"

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
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # Cache write failures are non-fatal

    def autocomplete(
        self, term: str, entity_type: str = "disease"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Query PubTator3 autocomplete endpoint for entity normalization.

        Args:
            term: Disease name to search
            entity_type: Entity type ("disease", "chemical", "gene", etc.)

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
        cache_key = self._cache_key(f"autocomplete_{entity_type}", term)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Rate limit
        self._rate_limit()

        # Make request
        url = f"{self.base_url}/entity/autocomplete/"
        params = {"query": term}

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json()

            # Filter by entity type (biotype field)
            if isinstance(results, list):
                filtered = [
                    r
                    for r in results
                    if r.get("biotype", "").lower() == entity_type.lower()
                ]
                self._cache_set(cache_key, filtered)
                return filtered

            self._cache_set(cache_key, results)
            return results

        except requests.exceptions.Timeout:
            print(f"[WARN] PubTator timeout for '{term}'")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[WARN] PubTator request failed for '{term}': {e}")
            return None
        except json.JSONDecodeError:
            print(f"[WARN] PubTator invalid JSON for '{term}'")
            return None

    def search_entity(
        self, term: str, entity_type: str = "disease"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search PubTator3 for entities using the search endpoint.

        Alternative to autocomplete for more comprehensive results.

        Args:
            term: Entity name to search
            entity_type: "disease", "chemical", "gene", "species", etc.

        Returns:
            List of matching entities or None on error.
        """
        if not term or not term.strip():
            return None

        # Check cache
        cache_key = self._cache_key(f"search_{entity_type}", term)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Rate limit
        self._rate_limit()

        # Build search query with entity type prefix
        type_prefix = entity_type.upper()
        query = f"@{type_prefix}_{term}"

        url = f"{self.base_url}/search/"
        params = {"text": query}

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Extract results
            results = data.get("results", []) if isinstance(data, dict) else data
            self._cache_set(cache_key, results)
            return results

        except Exception as e:
            print(f"[WARN] PubTator search failed for '{term}': {e}")
            return None


class DiseaseEnricher:
    """
    Enriches ExtractedDisease entities with PubTator3 data.

    Enrichment includes:
    - MeSH ID (if missing)
    - Normalized name from PubTator
    - Aliases/synonyms
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.client = PubTator3Client(config)

        # Enrichment settings
        enrichment_cfg = config.get("enrichment", {})
        self.enrich_missing_mesh = enrichment_cfg.get("enrich_missing_mesh", True)
        self.add_aliases = enrichment_cfg.get("add_aliases", True)
        self.enabled = config.get("enabled", True)

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
            diseases: List of validated disease entities
            verbose: Print progress

        Returns:
            List of enriched disease entities
        """
        if not self.enabled:
            return diseases

        enriched = []
        enriched_count = 0

        for i, disease in enumerate(diseases):
            result = self.enrich(disease)
            enriched.append(result)

            if "pubtator_enriched" in (result.validation_flags or []):
                if "pubtator_enriched" not in (disease.validation_flags or []):
                    enriched_count += 1

        if verbose and enriched_count > 0:
            print(f"    PubTator enriched: {enriched_count}/{len(diseases)} diseases")

        return enriched
