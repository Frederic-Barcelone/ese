#!/usr/bin/env python3
"""
rare_disease_pubtator3.py
==========================
PubTator3 API integration for rare disease document processing.
- Robust HTTP with retries/backoff and light jitter (1)
- Canonicalize inputs & cache keys (2)
- Gazetteer pre-match for drugs (3)
- Autocomplete + search fallback for better recall (4)
- Smarter cache with size guard & TTL (5)
- Batch helpers return JSON-serializable dicts (6)
- Clear confidence policy for results (7)

UPDATED: Fixed _make_request_with_retry() call signatures
"""

import os
import re
import json
import yaml
import time
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random

# ============================================================================
# USE CENTRALIZED LOGGING CONFIGURATION
# ============================================================================
try:
    from corpus_metadata.document_utils.metadata_logging_config import get_logger
    logger = get_logger('rare_disease_pubtator3')
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('rare_disease_pubtator3')

# ---------------------------- Models ----------------------------

class EntityType(Enum):
    """PubTator3 entity types"""
    GENE = "gene"
    DISEASE = "disease"
    CHEMICAL = "chemical"
    VARIANT = "variant"
    SPECIES = "species"
    CELLLINE = "cellline"


@dataclass
class NormalizedEntity:
    """Normalized entity from PubTator3"""
    original_text: str
    normalized_name: str
    entity_type: str
    mesh_id: Optional[str] = None
    pubtator_id: Optional[str] = None
    ncbi_id: Optional[str] = None
    database: Optional[str] = None
    confidence: float = 1.0
    aliases: List[str] = field(default_factory=list)
    pmid_count: int = 0
    related_pmids: List[str] = field(default_factory=list)

# ---------------------------- Manager ----------------------------

class PubTator3Manager:
    """
    Main PubTator3 manager for rare disease document processing.
    Handles initialization, caching, and API interactions.
    """

    CONF_AUTOCOMPLETE = 1.0
    CONF_GAZETTEER    = 0.9
    CONF_SEARCH_HIT   = 0.7
    CONF_NOT_FOUND    = 0.0

    def __init__(self, config_path: Optional[str] = None):
        """Initialize PubTator3 manager with configuration"""
        self.config = self._load_config(config_path)
        
        # Support both config structures: api_configuration.pubtator3 (legacy) and api.pubtator (current)
        api_section = self.config.get('api_configuration', self.config.get('api', {}))
        self.api_config = api_section.get('pubtator3', api_section.get('pubtator', {}))
        self.enabled = self.api_config.get('enabled', True)

        if not self.enabled:
            logger.warning("PubTator3 is disabled in configuration")
            return

        self.base_url = self.api_config.get('base_url', 'https://www.ncbi.nlm.nih.gov/research/pubtator3-api')
        self.session = self._setup_session()
        
        self.cache_dir = Path(self.api_config.get('cache', {}).get('directory', './cache/pubtator'))
        self._setup_cache()

        self.last_request_time = 0.0
        self.min_interval = float(self.api_config.get('rate_limit', {}).get('min_interval_seconds', 2.0))

        self.drug_gazetteer = self._load_drug_gazetteer()

        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'drugs_normalized': 0,
            'diseases_normalized': 0,
            'errors': 0
        }

        logger.debug("PubTator3 Manager initialized")

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file, or defaults"""
        if not config_path:
            for path in (
                './corpus_metadata/document_config/config.yaml',
                '../corpus_metadata/document_config/config.yaml',
                '../../corpus_metadata/document_config/config.yaml',
                './corpus_config/config.yaml',
                '../corpus_config/config.yaml',
                './config/config.yaml',
                './config.yaml',
            ):
                if os.path.exists(path):
                    config_path = path
                    break

        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config file not found, using defaults")
            return self._get_default_config()

        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                logger.debug(f"Loaded configuration from {config_path}")
                return cfg or self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'api_configuration': {
                'pubtator3': {
                    'enabled': True,
                    'base_url': 'https://www.ncbi.nlm.nih.gov/research/pubtator3-api',
                    'rate_limit': {'requests_per_minute': 30, 'min_interval_seconds': 2.0},
                    'cache': {
                        'enabled': True,
                        'directory': './cache/pubtator',
                        'autocomplete_expiry_hours': 24,
                        'search_expiry_hours': 1,
                        'articles_expiry_hours': 24,
                        'max_entries_per_bucket': 50000
                    },
                    'timeout_seconds': 30,
                    'retry_attempts': 3,
                    'retry_delay_seconds': 2
                }
            },
            'drug_detection': {'gazetteer': {'enabled': True, 'sources': {'custom_drugs': {'drugs': {}}}}}
        }

    def _setup_session(self) -> requests.Session:
        """Setup requests session with headers + retry/backoff"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'RareDisease-PubTator3/1.0 (+contact: your-email@example.com)',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })

        retry = Retry(
            total=int(self.api_config.get('retry_attempts', 3)),
            connect=3,
            read=3,
            backoff_factor=float(self.api_config.get('retry_delay_seconds', 2)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(['GET', 'POST']),
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=10,
            pool_block=False
        )
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        return session

    def _make_request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic and better error handling"""
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (10, 30)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except requests.exceptions.ConnectionError as e:
                if attempt < max_attempts - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Connection error (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Connection failed after {max_attempts} attempts: {e}")
                    raise
                    
            except requests.exceptions.Timeout as e:
                if attempt < max_attempts - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Request timeout (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request timed out after {max_attempts} attempts: {e}")
                    raise
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                raise
        
        raise requests.exceptions.RequestException("Max retry attempts exceeded")

    def _canon(self, s: str) -> str:
        """Canonicalize a string to improve cache hits & query stability"""
        s = (s or "").strip()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[^\w\s\-\/+().]', '', s)
        return s.lower()

    def _get_cache_key(self, text: str, entity_type: Optional[str] = None) -> str:
        """Generate canonical cache key"""
        base = self._canon(text)
        key_str = f"{base}::{entity_type}" if entity_type else base
        return hashlib.md5(key_str.encode()).hexdigest()

    def _setup_cache(self):
        """Setup caching directory and load existing cache"""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created cache directory: {self.cache_dir}")

        self.cache = {'autocomplete': {}, 'search': {}, 'articles': {}}
        self._load_cache()

    def _load_cache(self):
        """Load existing cache from disk"""
        cache_file = self.cache_dir / 'pubtator_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    saved = json.load(f)
                    for k in self.cache.keys():
                        if k in saved:
                            for key, value in saved[k].items():
                                if self._is_cache_valid(k, value.get('timestamp')):
                                    self.cache[k][key] = value
                logger.debug(f"Loaded cache: {sum(len(c) for c in self.cache.values())} entries")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")

    def _is_cache_valid(self, bucket: str, timestamp: Optional[str]) -> bool:
        """Check TTL of a cache entry"""
        if not timestamp:
            return False
        try:
            cache_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - cache_time
            expiry_hours = {
                'autocomplete': self.api_config.get('cache', {}).get('autocomplete_expiry_hours', 24),
                'search': self.api_config.get('cache', {}).get('search_expiry_hours', 1),
                'articles': self.api_config.get('cache', {}).get('articles_expiry_hours', 24)
            }
            max_age = timedelta(hours=float(expiry_hours.get(bucket, 24)))
            return age < max_age
        except Exception:
            return False

    def _save_cache(self):
        """Save cache to disk with size guard"""
        cache_file = self.cache_dir / 'pubtator_cache.json'
        try:
            cap = int(self.api_config.get('cache', {}).get('max_entries_per_bucket', 50000))
            for bucket in ('autocomplete', 'search', 'articles'):
                if len(self.cache[bucket]) > cap:
                    items = sorted(self.cache[bucket].items(), key=lambda kv: kv[1].get('timestamp', ''))
                    keep = dict(items[int(len(items) * 0.2):])
                    self.cache[bucket] = keep

            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _rate_limit(self):
        """Enforce min interval + small jitter to avoid bursts"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = (self.min_interval - elapsed) + random.uniform(0, 0.25)
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _load_drug_gazetteer(self) -> Dict:
        """Load drug gazetteer from configuration"""
        gazetteer = {}
        drug_config = self.config.get('drug_detection', {}).get('gazetteer', {})
        if drug_config.get('enabled'):
            custom = drug_config.get('sources', {}).get('custom_drugs', {}).get('drugs', {})
            if isinstance(custom, dict):
                gazetteer.update(custom)

        alexion_path = self.config.get('resources', {}).get('alexion_drugs_path')
        if alexion_path and os.path.exists(alexion_path):
            try:
                with open(alexion_path, 'r') as f:
                    alexion = json.load(f)
                    if isinstance(alexion, dict):
                        gazetteer.update(alexion)
                        logger.debug(f"Loaded {len(alexion)} drugs from {alexion_path}")
            except Exception as e:
                logger.error(f"Error loading {alexion_path}: {e}")
        return gazetteer

    def _gazetteer_lookup(self, term: str) -> Optional[NormalizedEntity]:
        """Brand/generic/alias match for drugs (pre-API)"""
        if not self.drug_gazetteer:
            return None
        canon = self._canon(term)
        for canonical, info in self.drug_gazetteer.items():
            names = {self._canon(canonical)}
            names.update({self._canon(a) for a in info.get('aliases', [])})
            if canon in names:
                return NormalizedEntity(
                    original_text=term,
                    normalized_name=canonical,
                    entity_type='drug',
                    mesh_id=info.get('mesh_id'),
                    aliases=info.get('aliases', []),
                    confidence=self.CONF_GAZETTEER
                )
        return None

    def _pubtator_autocomplete(self, query: str, concept: str, limit: int = 1) -> Optional[dict]:
        """Autocomplete endpoint: best-effort single result"""
        self._rate_limit()
        url = f"{self.base_url}/entity/autocomplete/"
        resp = self._make_request_with_retry(
            'get',
            url,
            params={'query': query, 'concept': concept, 'limit': limit}
        )
        self.stats['api_calls'] += 1
        if resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, list) and data:
                    return data[0]
            except Exception:
                pass
        return None

    def _pubtator_search(self, query: str) -> Optional[str]:
        """Search endpoint: return a PMID (or _id) to signal evidence"""
        self._rate_limit()
        url = f"{self.base_url}/search/"
        resp = self._make_request_with_retry(
            'get',
            url,
            params={'text': query, 'page': 1}
        )
        self.stats['api_calls'] += 1
        if resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, dict) and data.get('results'):
                    pmid = str(data['results'][0].get('pmid', '') or data['results'][0].get('_id', ''))
                    return pmid or None
            except Exception:
                pass
        return None

    def normalize_drug(self, drug_text: str, use_cache: bool = True) -> Optional[NormalizedEntity]:
        """Normalize a drug name using Gazetteer -> Autocomplete -> Search fallback"""
        if not self.enabled:
            return None

        drug_text = self._canon(drug_text)
        if not drug_text:
            return None

        cache_key = self._get_cache_key(drug_text, 'chemical')
        if use_cache and cache_key in self.cache['autocomplete']:
            cached = self.cache['autocomplete'][cache_key]
            if self._is_cache_valid('autocomplete', cached.get('timestamp')):
                self.stats['cache_hits'] += 1
                return self._dict_to_normalized_entity(cached['data'])

        g = self._gazetteer_lookup(drug_text)
        if g:
            self.cache['autocomplete'][cache_key] = {'data': asdict(g), 'timestamp': datetime.now().isoformat()}
            self._save_cache()
            self.stats['drugs_normalized'] += 1
            logger.debug(f"[GAZ] {drug_text} -> {g.normalized_name}")
            return g

        try:
            result = self._pubtator_autocomplete(drug_text, 'chemical', limit=1)
            if result:
                normalized = NormalizedEntity(
                    original_text=drug_text,
                    normalized_name=result.get('name', drug_text),
                    entity_type='drug',
                    pubtator_id=result.get('_id'),
                    database=result.get('db', 'mesh'),
                    mesh_id=result.get('db_id'),
                    confidence=self.CONF_AUTOCOMPLETE
                )
                self.cache['autocomplete'][cache_key] = {'data': asdict(normalized), 'timestamp': datetime.now().isoformat()}
                self._save_cache()
                self.stats['drugs_normalized'] += 1
                logger.debug(f"[AC ] {drug_text} -> {normalized.normalized_name}")
                return normalized

            pmid = self._pubtator_search(drug_text)
            if pmid:
                normalized = NormalizedEntity(
                    original_text=drug_text,
                    normalized_name=drug_text,
                    entity_type='drug',
                    confidence=self.CONF_SEARCH_HIT,
                    related_pmids=[pmid],
                    pmid_count=1
                )
                self.cache['autocomplete'][cache_key] = {'data': asdict(normalized), 'timestamp': datetime.now().isoformat()}
                self._save_cache()
                self.stats['drugs_normalized'] += 1
                logger.debug(f"[SRCH] {drug_text} -> evidence PMID {pmid}")
                return normalized

        except Exception as e:
            logger.error(f"Error normalizing drug '{drug_text}': {e}")
            self.stats['errors'] += 1

        return None

    def normalize_drugs(self, drug_names: List[str], text: Optional[str] = None) -> List[Dict]:
        """Normalize multiple drug names using PubTator3 API"""
        if not self.enabled:
            return [None] * len(drug_names)
        
        batch_results = self.process_drug_batch(drug_names)
        
        results = []
        for drug_name in drug_names:
            canon_name = self._canon(drug_name)
            
            if canon_name in batch_results:
                entity_dict = batch_results[canon_name]
                results.append({
                    'normalized_name': entity_dict.get('normalized_name'),
                    'mesh_id': entity_dict.get('mesh_id'),
                    'identifier': entity_dict.get('pubtator_id'),
                    'pmid_count': entity_dict.get('pmid_count', 0),
                    'confidence': entity_dict.get('confidence', 1.0),
                    'database': entity_dict.get('database'),
                    'original_text': entity_dict.get('original_text'),
                    'entity_type': entity_dict.get('entity_type')
                })
            else:
                results.append(None)
        
        normalized_count = sum(1 for r in results if r is not None)
        logger.debug(f"Batch normalization: {normalized_count}/{len(drug_names)} drugs normalized")
        
        return results

    def normalize_disease(self, disease_text: str, use_cache: bool = True) -> Optional[NormalizedEntity]:
        """Normalize a disease name using Autocomplete -> Search fallback"""
        if not self.enabled:
            return None

        disease_text = self._canon(disease_text)
        if not disease_text:
            return None

        cache_key = self._get_cache_key(disease_text, 'disease')
        if use_cache and cache_key in self.cache['autocomplete']:
            cached = self.cache['autocomplete'][cache_key]
            if self._is_cache_valid('autocomplete', cached.get('timestamp')):
                self.stats['cache_hits'] += 1
                return self._dict_to_normalized_entity(cached['data'])

        try:
            result = self._pubtator_autocomplete(disease_text, 'disease', limit=1)
            if result:
                normalized = NormalizedEntity(
                    original_text=disease_text,
                    normalized_name=result.get('name', disease_text),
                    entity_type='disease',
                    pubtator_id=result.get('_id'),
                    database=result.get('db', 'mesh'),
                    mesh_id=result.get('db_id'),
                    confidence=self.CONF_AUTOCOMPLETE
                )
                self.cache['autocomplete'][cache_key] = {'data': asdict(normalized), 'timestamp': datetime.now().isoformat()}
                self._save_cache()
                self.stats['diseases_normalized'] += 1
                logger.debug(f"[AC ] {disease_text} -> {normalized.normalized_name}")
                return normalized

            pmid = self._pubtator_search(disease_text)
            if pmid:
                normalized = NormalizedEntity(
                    original_text=disease_text,
                    normalized_name=disease_text,
                    entity_type='disease',
                    confidence=self.CONF_SEARCH_HIT,
                    related_pmids=[pmid],
                    pmid_count=1
                )
                self.cache['autocomplete'][cache_key] = {'data': asdict(normalized), 'timestamp': datetime.now().isoformat()}
                self._save_cache()
                self.stats['diseases_normalized'] += 1
                logger.debug(f"[SRCH] {disease_text} -> evidence PMID {pmid}")
                return normalized

        except Exception as e:
            logger.error(f"Error normalizing disease '{disease_text}': {e}")
            self.stats['errors'] += 1

        return None

    def find_literature(self, entity_id: str, max_articles: int = 10) -> List[str]:
        """Fetch a few PMIDs for an entity id"""
        if not self.enabled or not entity_id:
            return []
        try:
            self._rate_limit()
            url = f"{self.base_url}/search/"
            resp = self._make_request_with_retry(
                'get',
                url,
                params={'text': entity_id, 'page': 1}
            )
            self.stats['api_calls'] += 1
            if resp.status_code == 200:
                data = resp.json()
                pmids = []
                for result in data.get('results', [])[:max_articles]:
                    pmid = str(result.get('pmid', result.get('_id', '')))
                    if pmid:
                        pmids.append(pmid)
                return pmids
        except Exception as e:
            logger.error(f"Error finding literature for '{entity_id}': {e}")
            self.stats['errors'] += 1
        return []

    def process_drug_batch(self, drug_list: List[str]) -> Dict[str, Dict]:
        """Normalize a batch of drugs; returns JSON-serializable dict"""
        results: Dict[str, Dict] = {}
        unique = sorted({self._canon(x) for x in drug_list if x and x.strip()})
        logger.debug(f"Processing {len(unique)} unique drugs")
        for i, drug in enumerate(unique, 1):
            if i % 25 == 0:
                logger.debug(f"  ... {i}/{len(unique)}")
            ent = self.normalize_drug(drug)
            if ent:
                results[drug] = asdict(ent)
        logger.debug(f"Normalized {len(results)}/{len(unique)} drugs")
        return results

    def process_disease_batch(self, disease_list: List[str]) -> Dict[str, Dict]:
        """Normalize a batch of diseases; returns JSON-serializable dict"""
        results: Dict[str, Dict] = {}
        unique = sorted({self._canon(x) for x in disease_list if x and x.strip()})
        logger.debug(f"Processing {len(unique)} unique diseases")
        for i, dis in enumerate(unique, 1):
            if i % 25 == 0:
                logger.debug(f"  ... {i}/{len(unique)}")
            ent = self.normalize_disease(dis)
            if ent:
                results[dis] = asdict(ent)
        logger.debug(f"Normalized {len(results)}/{len(unique)} diseases")
        return results

    def _dict_to_normalized_entity(self, data: Dict) -> NormalizedEntity:
        return NormalizedEntity(**data)

    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        return {
            **self.stats,
            'cache_size': sum(len(c) for c in self.cache.values()),
            'cache_info': {
                'autocomplete': len(self.cache['autocomplete']),
                'search': len(self.cache['search']),
                'articles': len(self.cache['articles'])
            }
        }

    def clear_cache(self):
        """Clear all cached data"""
        self.cache = {'autocomplete': {}, 'search': {}, 'articles': {}}
        self._save_cache()
        logger.debug("Cache cleared")

    def test_connection(self) -> bool:
        """Test connection to PubTator3 API"""
        try:
            url = f"{self.base_url}/entity/autocomplete/"
            resp = self._make_request_with_retry(
                'get',
                url,
                params={'query': 'test', 'limit': 1},
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

def setup_pubtator3(config_path: Optional[str] = None) -> PubTator3Manager:
    """Initialize manager and test connectivity"""
    manager = PubTator3Manager(config_path)
    if manager.enabled and manager.test_connection():
        logger.debug("[OK] PubTator3 API connection OK")
    elif manager.enabled:
        logger.warning("[WARN] Could not verify PubTator3 connectivity")
    return manager