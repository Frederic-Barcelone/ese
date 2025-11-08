#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Complete Extractor – Feasibility Enhanced Edition v7.1 CORRECTED
- Fixed to use ACTUAL CTIS field names (not documentation names)
- Real data extraction from authorizedPartI.trialDetails.trialInformation
- Handles age category codes (not specific ages)
- Complete feasibility study data extraction
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
import sqlite3
import re
import shutil
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from collections import defaultdict

# ===================== Configuration (defaults) =====================

DEFAULT_BASE = os.environ.get("CTIS_BASE", "https://euclinicaltrials.eu")
BASE = DEFAULT_BASE.rstrip("/")
SEARCH_URL = f"{BASE}/ctis-public-api/search"
DETAIL_URL = f"{BASE}/ctis-public-api/retrieve/{{ct}}"
PORTAL_URL = f"{BASE}/search-for-clinical-trials/?lang=en"

# These are set/overridden in main() once --out-dir is parsed
OUT_DIR = Path("ctis-out")
NDJSON_PATH = OUT_DIR / "ctis_full.ndjson"
DB_PATH = OUT_DIR / "ctis.db"
CTNUMBERS_PATH = OUT_DIR / "ct_numbers.txt"
FAILED_PATH = OUT_DIR / "failed_ctnumbers.txt"

PAGE_SIZE = 100
MAX_WORKERS = 3
MAX_RETRIES = 6
BASE_BACKOFF = 1.0
JITTER_RANGE = (0.15, 0.45)
FINAL_COOLDOWN = 1.0
REPORT_EVERY = 50
RATE_LIMIT_RPS = 2.0
REQUEST_TIMEOUT = 60.0

STATUS_SEGMENTS = [1, 2, 3, 4, 5, 6, 7, 8]
YEAR_START = 2019
CURRENT_YEAR = datetime.utcnow().year

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)
BASE_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": BASE,
    "Referer": PORTAL_URL,
    "Connection": "keep-alive",
}

# Global rate limiter (set in main)
GLOBAL_RATE_LIMITER = None  # type: Optional["RateLimiter"]

# Age category codes mapping
AGE_CATEGORY_MAP = {
    "1": "Preterm newborn",
    "2": "Newborns (0-27 days)",
    "3": "Infants and toddlers (28 days-23 months)",
    "4": "Children (2-11 years)",
    "5": "Adolescents (12-17 years)",
    "6": "Adults (18-64 years)",
    "7": "Elderly (65-84 years)",
    "8": "85 years and over"
}


# ===================== Logging =====================

def log(msg: str, level: str = "INFO"):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", file=sys.stderr, flush=True)


# ===================== Filesystem helpers =====================

def setup_output_dir(out_dir: Path, reset: bool = False):
    if out_dir.exists() and reset:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def safe_append_lines(path: Path, lines: List[str]):
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for line in lines:
            line = line.rstrip("\n")
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, records: List[Dict[str, Any]]):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ===================== Timing / Rate limit =====================

def sleep_jitter():
    time.sleep(random.uniform(*JITTER_RANGE))


def backoff(i: int, base: float = BASE_BACKOFF):
    time.sleep((base * (2 ** i)) + random.uniform(0, 0.6))


class RateLimiter:
    """Simple thread-safe leaky bucket (interval) limiter."""
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


# ===================== HTTP =====================

def warm_up(session: requests.Session):
    try:
        session.get(PORTAL_URL, timeout=30)
        sleep_jitter()
    except Exception:
        pass


def _ensure_json_response(resp: requests.Response) -> Dict[str, Any]:
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


def req(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """Hardened HTTP with rate-limit, jitter, and backoff for common transient failures."""
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
            log(f"Network error on {method} {url}: {e!r} – retrying...", "WARN")
            backoff(i)
        except requests.RequestException as e:
            log(f"HTTP error on {method} {url}: {e!r}", "ERROR")
            raise

    warm_up(session)
    if GLOBAL_RATE_LIMITER:
        GLOBAL_RATE_LIMITER.wait()
    r = session.request(method, url, headers=headers, timeout=timeout * 1.5, **kwargs)
    r.raise_for_status()
    return r


# ===================== Database =====================

DDL = """
CREATE TABLE IF NOT EXISTS trials (
    ctNumber TEXT PRIMARY KEY,
    ctStatus INTEGER,
    ctPublicStatusCode TEXT,
    title TEXT,
    shortTitle TEXT,
    sponsor TEXT,
    trialPhase TEXT,
    therapeuticAreas TEXT,
    medicalCondition TEXT,
    medicalConditionsList TEXT,
    isConditionRareDisease INTEGER,
    countries TEXT,
    decisionDate TEXT,
    publishDate TEXT,
    lastUpdated TEXT,
    
    -- FEASIBILITY FIELDS (using actual CTIS structure)
    ageCategories TEXT,  -- JSON array of age category codes
    gender TEXT,  -- "both", "male", or "female"
    isRandomised INTEGER,
    blindingType TEXT,  -- "open", "single-blind", "double-blind"
    trialScope TEXT,  -- JSON array of scope codes
    mainObjective TEXT,
    primaryEndpointsCount INTEGER,
    secondaryEndpointsCount INTEGER,
    estimatedRecruitmentStartDate TEXT,
    estimatedEndDate TEXT,
    
    data_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_lastUpdated ON trials(lastUpdated);
CREATE INDEX IF NOT EXISTS idx_trials_medicalCondition ON trials(medicalCondition);
CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(trialPhase);

CREATE TABLE IF NOT EXISTS trial_sites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    site_name TEXT,
    organisation TEXT,
    country TEXT,
    city TEXT,
    address TEXT,
    postal_code TEXT,
    path TEXT,
    raw_json TEXT,
    UNIQUE(ctNumber, site_name, organisation, country, city, address, postal_code, path)
);
CREATE INDEX IF NOT EXISTS idx_sites_ct ON trial_sites(ctNumber);
CREATE INDEX IF NOT EXISTS idx_sites_country ON trial_sites(country);

CREATE TABLE IF NOT EXISTS trial_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    name TEXT,
    role TEXT,
    email TEXT,
    phone TEXT,
    country TEXT,
    city TEXT,
    site_name TEXT,
    organisation TEXT,
    path TEXT,
    raw_json TEXT,
    UNIQUE(ctNumber, name, role, email, phone, country, city, site_name, organisation, path)
);
CREATE INDEX IF NOT EXISTS idx_people_ct ON trial_people(ctNumber);

CREATE TABLE IF NOT EXISTS inclusion_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    criterionNumber INTEGER,
    criterionText TEXT NOT NULL,
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_inclusion_ct ON inclusion_criteria(ctNumber);

CREATE TABLE IF NOT EXISTS exclusion_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    criterionNumber INTEGER,
    criterionText TEXT NOT NULL,
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_exclusion_ct ON exclusion_criteria(ctNumber);

CREATE TABLE IF NOT EXISTS endpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    endpointType TEXT NOT NULL,
    endpointNumber INTEGER,
    endpointText TEXT NOT NULL,
    timeFrame TEXT,
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_endpoints_ct ON endpoints(ctNumber);
CREATE INDEX IF NOT EXISTS idx_endpoints_type ON endpoints(endpointType);

CREATE TABLE IF NOT EXISTS trial_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    productRole TEXT,
    productName TEXT,
    activeSubstance TEXT,
    pharmaceuticalForm TEXT,
    route TEXT,
    maxDailyDose TEXT,
    maxDailyDoseUnit TEXT,
    maxTreatmentPeriod INTEGER,
    maxTreatmentPeriodUnit TEXT,
    isPaediatric INTEGER,
    isOrphanDrug INTEGER,
    authorizationStatus TEXT,
    raw_json TEXT,
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_products_ct ON trial_products(ctNumber);
"""


def init_db(db_path: Path, reset: bool = False) -> sqlite3.Connection:
    if reset and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(DDL)
    log(f"Database initialized: {db_path}")
    return conn


# ===================== Utilities =====================

def get(obj, *keys, default=""):
    cur = obj or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def coerce_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int,)):
            return v
        s = str(v).strip()
        return int(s) if s else None
    except Exception:
        return None


def parse_ts(v: Any) -> Optional[datetime]:
    """Parse many timestamp shapes into timezone-aware UTC datetime."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        sec = float(v) / (1000.0 if abs(float(v)) > 1e12 else 1.0)
        return datetime.fromtimestamp(sec, tz=timezone.utc)
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc)
    s = str(v).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if len(s) == 7 and s[4] == "-":
            s = s + "-01"
        if len(s) == 4 and s.isdigit():
            s = s + "-01-01"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    return None


def ts_is_newer(new_ts: Any, old_ts: Any) -> bool:
    """Return True if new_ts > old_ts (with robust parsing)."""
    new_dt = parse_ts(new_ts)
    old_dt = parse_ts(old_ts)
    if new_dt is None:
        return True
    if old_dt is None:
        return True
    return new_dt > old_dt


def normalize_country(node: Dict[str, Any]) -> Optional[str]:
    for k in ("country", "memberState", "countryName"):
        v = node.get(k)
        if isinstance(v, dict):
            for kk in ("label", "name", "text", "value"):
                val = v.get(kk)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            code = v.get("code") or v.get("id")
            if isinstance(code, str) and code.strip():
                return code.strip()
        elif isinstance(v, str) and v.strip():
            return v.strip()
    return None


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")


def extract_email_phone(d: Dict) -> Tuple[Optional[str], Optional[str]]:
    email = d.get("functionalEmailAddress") or d.get("email") or d.get("emailAddress")
    phone = d.get("telephone") or d.get("phone") or d.get("phoneNumber")

    if not email or not phone:
        s = json.dumps(d, ensure_ascii=False)
        if not email:
            e = EMAIL_RE.findall(s)
            email = e[0] if e else None
        if not phone:
            p = PHONE_RE.findall(s)
            phone = p[0].strip() if p else None

    return (email, phone)


# ===================== CORRECTED: Feasibility Data Extraction =====================

def extract_inclusion_criteria(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract inclusion criteria - ACTUAL CTIS structure."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    eligibility = trial_info.get("eligibilityCriteria", {}) or {}
    
    # ACTUAL field name: principalInclusionCriteria (not inclusionCriteria!)
    inclusion = eligibility.get("principalInclusionCriteria", []) or []
    
    result = []
    for criterion in inclusion:
        if isinstance(criterion, dict):
            # ACTUAL field name: principalInclusionCriteria (the value)
            text = criterion.get("principalInclusionCriteria", "")
            number = criterion.get("number", 0)
            if text:
                result.append({
                    "criterionNumber": number,
                    "criterionText": text
                })
    
    return result


def extract_exclusion_criteria(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract exclusion criteria - ACTUAL CTIS structure."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    eligibility = trial_info.get("eligibilityCriteria", {}) or {}
    
    # ACTUAL field name: principalExclusionCriteria (not exclusionCriteria!)
    exclusion = eligibility.get("principalExclusionCriteria", []) or []
    
    result = []
    for criterion in exclusion:
        if isinstance(criterion, dict):
            # ACTUAL field name: principalExclusionCriteria (the value)
            text = criterion.get("principalExclusionCriteria", "")
            number = criterion.get("number", 0)
            if text:
                result.append({
                    "criterionNumber": number,
                    "criterionText": text
                })
    
    return result


def extract_endpoints(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract primary and secondary endpoints - ACTUAL CTIS structure."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    
    # ACTUAL field name: endPoint (not trialObjectivesAndEndpoints!)
    endpoint_data = trial_info.get("endPoint", {}) or {}
    
    result = []
    
    # Primary endpoints
    primary = endpoint_data.get("primaryEndPoints", []) or []
    for ep in primary:
        if isinstance(ep, dict):
            # ACTUAL field name: endPoint (the value)
            text = ep.get("endPoint", "")
            number = ep.get("number", 0)
            if text:
                result.append({
                    "endpointType": "primary",
                    "endpointNumber": number,
                    "endpointText": text,
                    "timeFrame": ""  # Not provided in this structure
                })
    
    # Secondary endpoints
    secondary = endpoint_data.get("secondaryEndPoints", []) or []
    for ep in secondary:
        if isinstance(ep, dict):
            text = ep.get("endPoint", "")
            number = ep.get("number", 0)
            if text:
                result.append({
                    "endpointType": "secondary",
                    "endpointNumber": number,
                    "endpointText": text,
                    "timeFrame": ""
                })
    
    return result


def extract_trial_products(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract detailed product information."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    products = partI.get("products", []) or []
    
    role_map = {'1': 'test', '2': 'comparator', '3': 'placebo', '4': 'auxiliary'}
    
    result = []
    for prod in products:
        if not isinstance(prod, dict):
            continue
        
        prod_dict_info = prod.get("productDictionaryInfo", {}) or {}
        
        result.append({
            "productRole": role_map.get(prod.get("part1MpRoleTypeCode"), "unknown"),
            "productName": prod.get("productName", ""),
            "activeSubstance": prod_dict_info.get("activeSubstanceName", ""),
            "pharmaceuticalForm": prod.get("pharmaceuticalFormDisplay", ""),
            "route": ", ".join(prod.get("routes", [])),
            "maxDailyDose": str(prod.get("maxDailyDoseAmount", "")),
            "maxDailyDoseUnit": prod.get("doseUom", ""),
            "maxTreatmentPeriod": coerce_int(prod.get("maxTreatmentPeriod")),
            "maxTreatmentPeriodUnit": "months",  # timeUnitCode usually means months
            "isPaediatric": 1 if prod.get("isPaediatricFormulation") else 0,
            "isOrphanDrug": 1 if prod.get("orphanDrugEdit") else 0,
            "authorizationStatus": "authorized" if prod_dict_info.get("prodAuthStatus") == 2 else "not_authorized",
            "raw": prod
        })
    
    return result


def extract_population(js: Dict[str, Any]) -> Tuple[str, str]:
    """Extract patient population details - age categories and gender."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    
    # ACTUAL field name: populationOfTrialSubjects (not trialPopulation!)
    population = trial_info.get("populationOfTrialSubjects", {}) or {}
    
    # Extract age category CODES (not specific ages!)
    age_ranges = population.get("ageRanges", []) or []
    age_categories = []
    for age_range in age_ranges:
        if isinstance(age_range, dict):
            code = age_range.get("ageRangeCategoryCode") or age_range.get("ageRangeCategory")
            if code:
                age_categories.append(str(code))
    
    age_categories_json = json.dumps(sorted(set(age_categories)))
    
    # Gender
    is_female = population.get("isFemaleSubjects", False)
    is_male = population.get("isMaleSubjects", False)
    
    if is_female and is_male:
        gender = "both"
    elif is_female:
        gender = "female"
    elif is_male:
        gender = "male"
    else:
        gender = ""
    
    return (age_categories_json, gender)


def extract_trial_design(js: Dict[str, Any]) -> Tuple[int, str]:
    """Extract trial design details - ACTUAL CTIS structure."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    
    # ACTUAL location: protocolInformation.studyDesign
    protocol_info = trial_details.get("protocolInformation", {}) or {}
    study_design = protocol_info.get("studyDesign", {}) or {}
    
    period_details = study_design.get("periodDetails", []) or []
    
    is_randomised = 0
    blinding_type = ""
    
    if period_details and len(period_details) > 0:
        first_period = period_details[0]
        
        # Blinding method code
        blinding_code = first_period.get("blindingMethodCode")
        blinding_map = {
            "1": "open",
            "2": "single-blind",
            "3": "double-blind"
        }
        blinding_type = blinding_map.get(str(blinding_code), "")
        
        # Allocation method
        allocation_code = first_period.get("allocationMethod")
        if str(allocation_code) == "1":
            is_randomised = 1
    
    return (is_randomised, blinding_type)


def extract_objectives(js: Dict[str, Any]) -> Tuple[str, str, int, int]:
    """Extract objectives - ACTUAL CTIS structure."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    
    # ACTUAL field name: trialObjective (not trialObjectivesAndEndpoints!)
    trial_objective = trial_info.get("trialObjective", {}) or {}
    
    main_obj = trial_objective.get("mainObjective", "")
    
    # Trial scopes (as array of codes)
    scopes = trial_objective.get("trialScopes", []) or []
    scope_codes = [s.get("code") for s in scopes if isinstance(s, dict) and s.get("code")]
    scope_json = json.dumps(scope_codes)
    
    # Endpoint counts
    endpoint_data = trial_info.get("endPoint", {}) or {}
    primary_eps = endpoint_data.get("primaryEndPoints", []) or []
    secondary_eps = endpoint_data.get("secondaryEndPoints", []) or []
    
    primary_count = len(primary_eps) if isinstance(primary_eps, list) else 0
    secondary_count = len(secondary_eps) if isinstance(secondary_eps, list) else 0
    
    return (scope_json, main_obj, primary_count, secondary_count)


def extract_trial_duration(js: Dict[str, Any]) -> Tuple[str, str]:
    """Extract trial duration dates."""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    
    # ACTUAL field name: trialDuration
    duration = trial_info.get("trialDuration", {}) or {}
    
    start_date = duration.get("estimatedRecruitmentStartDate", "")
    end_date = duration.get("estimatedEndDate", "")
    
    return (start_date, end_date)


# ===================== Medical Condition Extraction =====================

def extract_medical_conditions(js: Dict[str, Any]) -> Tuple[str, str, int]:
    """Extract medical condition information from trial JSON."""
    partI = (js.get("authorizedApplication", {}).get("authorizedPartI") or
             js.get("authorisedApplication", {}).get("authorisedPartI") or {})
    
    # First try the new structure in trialInformation
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    med_cond_data = trial_info.get("medicalCondition", {}) or {}
    part_i_conditions = med_cond_data.get("partIMedicalConditions", []) or []
    
    if part_i_conditions:
        first_condition = part_i_conditions[0] if isinstance(part_i_conditions[0], dict) else {}
        primary_condition = first_condition.get("medicalCondition", "")
        is_rare_disease = 1 if first_condition.get("isConditionRareDisease") else 0
        
        condition_names = [c.get("medicalCondition") for c in part_i_conditions 
                          if isinstance(c, dict) and c.get("medicalCondition")]
        conditions_json = json.dumps(condition_names, ensure_ascii=False)
        
        return (primary_condition, conditions_json, is_rare_disease)
    
    # Fallback to old structure
    medical_conditions = partI.get("medicalConditions", []) or []
    if not medical_conditions or not isinstance(medical_conditions, list):
        return ("", "[]", 0)
    
    primary_condition = ""
    is_rare_disease = 0
    
    if len(medical_conditions) > 0 and isinstance(medical_conditions[0], dict):
        first_condition = medical_conditions[0]
        primary_condition = first_condition.get("medicalCondition", "")
        is_rare_disease = 1 if first_condition.get("isConditionRareDisease") else 0
    
    condition_names = []
    for cond in medical_conditions:
        if isinstance(cond, dict):
            name = cond.get("medicalCondition")
            if name:
                condition_names.append(name)
    
    conditions_json = json.dumps(condition_names, ensure_ascii=False)
    
    return (primary_condition, conditions_json, is_rare_disease)


# ===================== Site Extraction =====================

def extract_sites(js: Dict[str, Any], ct: str) -> List[Dict[str, Any]]:
    """Extract ALL sites from multiple locations (Part II variants included)."""
    sites: List[Dict[str, Any]] = []

    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}

    # authorizedPartsII (plural)
    parts_ii = auth_app.get("authorizedPartsII") or auth_app.get("authorisedPartsII") or []
    if isinstance(parts_ii, list):
        for part_idx, part_ii in enumerate(parts_ii or []):
            if not isinstance(part_ii, dict):
                continue

            trial_sites = part_ii.get("trialSites", [])
            if isinstance(trial_sites, list):
                for site_idx, site in enumerate(trial_sites or []):
                    if not isinstance(site, dict):
                        continue

                    org_addr_info = site.get("organisationAddressInfo", {}) or {}
                    org_obj = org_addr_info.get("organisation", {}) or {}
                    org = org_obj.get("name") if isinstance(org_obj, dict) else None

                    addr_obj = org_addr_info.get("address", {}) or {}
                    if isinstance(addr_obj, dict):
                        address = addr_obj.get("addressLine1") or addr_obj.get("street")
                        city = addr_obj.get("city") or addr_obj.get("town")
                        postal = addr_obj.get("postalCode") or addr_obj.get("postcode")
                        country = addr_obj.get("countryName") or addr_obj.get("country")
                    else:
                        address = city = postal = country = None

                    site_name = org
                    dept = site.get("departmentName")
                    if dept:
                        site_name = f"{org} - {dept}" if org else dept

                    sites.append({
                        "ctNumber": ct,
                        "site_name": site_name,
                        "organisation": org,
                        "country": country,
                        "city": city,
                        "address": address,
                        "postal_code": postal,
                        "path": f"authorizedPartsII[{part_idx}].trialSites[{site_idx}]",
                        "raw": site
                    })

    # Deduplicate
    def _norm(v):
        return v.strip() if isinstance(v, str) else (v if v is not None else "")

    seen = set()
    unique = []
    for s in sites:
        key = (_norm(s["ctNumber"]), _norm(s.get("site_name")), _norm(s.get("organisation")),
               _norm(s.get("country")), _norm(s.get("city")))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


# ===================== People Extraction =====================

def extract_people(js: Dict[str, Any], ct: str) -> List[Dict[str, Any]]:
    """Extract ALL people from sponsors and sites."""
    people: List[Dict[str, Any]] = []

    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})

    # Sponsors
    sponsors = partI.get("sponsors", []) or []
    for sp_idx, sponsor in enumerate(sponsors):
        if not isinstance(sponsor, dict):
            continue

        # Public contacts
        for idx, contact in enumerate(sponsor.get("publicContacts", []) or []):
            if not isinstance(contact, dict):
                continue
            name = contact.get("functionalName") or contact.get("name")
            email, phone = extract_email_phone(contact)
            org_info = contact.get("organisation", {}) or {}
            org = org_info.get("name") if isinstance(org_info, dict) else None
            addr = contact.get("address", {}) or {}
            country = addr.get("countryName") if isinstance(addr, dict) else None
            city = addr.get("city") if isinstance(addr, dict) else None

            people.append({
                "ctNumber": ct,
                "name": name,
                "role": "Public Contact (Sponsor)",
                "email": email,
                "phone": phone,
                "country": country,
                "city": city,
                "site_name": None,
                "organisation": org,
                "path": f"sponsors[{sp_idx}].publicContacts[{idx}]",
                "raw": contact
            })

        # Scientific contacts
        for idx, contact in enumerate(sponsor.get("scientificContacts", []) or []):
            if not isinstance(contact, dict):
                continue
            name = contact.get("functionalName") or contact.get("name")
            email, phone = extract_email_phone(contact)
            org_info = contact.get("organisation", {}) or {}
            org = org_info.get("name") if isinstance(org_info, dict) else None
            addr = contact.get("address", {}) or {}
            country = addr.get("countryName") if isinstance(addr, dict) else None
            city = addr.get("city") if isinstance(addr, dict) else None

            people.append({
                "ctNumber": ct,
                "name": name,
                "role": "Scientific Contact (Sponsor)",
                "email": email,
                "phone": phone,
                "country": country,
                "city": city,
                "site_name": None,
                "organisation": org,
                "path": f"sponsors[{sp_idx}].scientificContacts[{idx}]",
                "raw": contact
            })

    # Site investigators (Part II plural)
    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    parts_ii = auth_app.get("authorizedPartsII") or auth_app.get("authorisedPartsII") or []

    if isinstance(parts_ii, list):
        for part_idx, part_ii in enumerate(parts_ii):
            if not isinstance(part_ii, dict):
                continue
            trial_sites = part_ii.get("trialSites", []) or []
            for site_idx, site in enumerate(trial_sites):
                if not isinstance(site, dict):
                    continue
                org_addr_info = site.get("organisationAddressInfo", {}) or {}
                org_obj = org_addr_info.get("organisation", {}) or {}
                org = org_obj.get("name") if isinstance(org_obj, dict) else None
                addr_obj = org_addr_info.get("address", {}) or {}
                city = addr_obj.get("city") if isinstance(addr_obj, dict) else None
                country = addr_obj.get("countryName") if isinstance(addr_obj, dict) else None

                site_name = org
                dept = site.get("departmentName")
                if dept:
                    site_name = f"{org} - {dept}" if org else dept

                person_info = site.get("personInfo", {}) or {}
                if isinstance(person_info, dict) and person_info:
                    first_name = person_info.get("firstName")
                    last_name = person_info.get("lastName")
                    name = f"{(first_name or '').strip()} {(last_name or '').strip()}".strip() or None
                    email = person_info.get("email") or org_addr_info.get("email")
                    phone = person_info.get("telephone") or org_addr_info.get("phone")

                    if any([name, email, phone]):
                        people.append({
                            "ctNumber": ct,
                            "name": name,
                            "role": "Site Investigator",
                            "email": email,
                            "phone": phone,
                            "country": country,
                            "city": city,
                            "site_name": site_name,
                            "organisation": org,
                            "path": f"authorizedPartsII[{part_idx}].trialSites[{site_idx}].personInfo",
                            "raw": person_info
                        })

    # Deduplicate
    def _norm(v):
        return v.strip() if isinstance(v, str) else (v if v is not None else "")

    seen = set()
    unique = []
    for p in people:
        key = (_norm(p["ctNumber"]), _norm(p.get("name")), _norm(p.get("role")),
               _norm(p.get("email")), _norm(p.get("phone")))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ===================== Trial Field Extraction =====================

def extract_trial_fields(js: Dict[str, Any]) -> Dict[str, Any]:
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})

    root = js

    trial_details = get(partI, "trialDetails", "clinicalTrialIdentifiers")
    title = (get(trial_details, "fullTitle") or
             get(trial_details, "publicTitle") or
             get(root, "title"))

    short = get(trial_details, "shortTitle") or get(root, "shortTitle")

    # Sponsor
    sponsor = ""
    sponsors_list = partI.get("sponsors", []) or []
    if isinstance(sponsors_list, list) and sponsors_list:
        first_sponsor = sponsors_list[0]
        if isinstance(first_sponsor, dict):
            org = first_sponsor.get("organisation", {}) or {}
            if isinstance(org, dict):
                sponsor = org.get("name") or ""

    if not sponsor:
        trial_info_detail = get(partI, "trialDetails", "trialInformation")
        support_list = get(trial_info_detail, "sourceOfMonetarySupport") or []
        if isinstance(support_list, list) and support_list:
            sponsor = get(support_list[0], "organisationName") or ""

    # Therapeutic areas
    ta_list = get(partI, "therapeuticAreas") or []
    ta = "; ".join([t.get("name", "") for t in ta_list if isinstance(t, dict)]) if isinstance(ta_list, list) else ""

    # Medical conditions
    medical_condition, medical_conditions_json, is_rare = extract_medical_conditions(js)

    # Countries
    countries = ""
    root_msc = js.get("memberStatesConcerned") or []
    if isinstance(root_msc, list) and root_msc:
        country_names = []
        for msc in root_msc:
            if isinstance(msc, dict):
                name = msc.get("mscName") or msc.get("countryName")
                if name:
                    country_names.append(name)
        if country_names:
            countries = ";".join(sorted(set(country_names)))

    # Phase
    phase_code = get(partI, "trialDetails", "trialInformation", "trialCategory", "trialPhase")
    phase_map = {
        "1": "Phase I",
        "2": "Phase II",
        "3": "Phase III",
        "4": "Phase IV",
        "5": "Phase III",
        "6": "Expanded Access",
        "7": "Phase I/II",
        "8": "Phase II/III",
        "9": "Phase III/IV",
        "10": "Non-Interventional",
        "11": "Compassionate Use",
    }
    phase = phase_map.get(str(phase_code), str(phase_code) if phase_code else "")

    # Status
    ct_status_num = coerce_int(root.get("ctStatus"))
    ct_public_status = root.get("ctPublicStatusCode") or ""

    # Timestamps
    publish = root.get("publishDate")
    last_up = root.get("lastUpdated")
    decision = root.get("decisionDate")
    freshest = None
    for candidate in (publish, last_up, decision):
        if ts_is_newer(candidate, freshest):
            freshest = candidate

    # Feasibility fields
    age_categories, gender = extract_population(js)
    is_randomised, blinding_type = extract_trial_design(js)
    scope_json, main_obj, primary_count, secondary_count = extract_objectives(js)
    start_date, end_date = extract_trial_duration(js)

    return {
        "ctNumber": root.get("ctNumber"),
        "ctStatus": ct_status_num,
        "ctPublicStatusCode": str(ct_public_status) if ct_public_status is not None else "",
        "title": title,
        "shortTitle": short,
        "sponsor": sponsor,
        "trialPhase": phase,
        "therapeuticAreas": ta,
        "medicalCondition": medical_condition,
        "medicalConditionsList": medical_conditions_json,
        "isConditionRareDisease": is_rare,
        "countries": countries,
        "decisionDate": decision,
        "publishDate": publish,
        "lastUpdated": freshest,
        
        # Feasibility fields
        "ageCategories": age_categories,
        "gender": gender,
        "isRandomised": is_randomised,
        "blindingType": blinding_type,
        "trialScope": scope_json,
        "mainObjective": main_obj,
        "primaryEndpointsCount": primary_count,
        "secondaryEndpointsCount": secondary_count,
        "estimatedRecruitmentStartDate": start_date,
        "estimatedEndDate": end_date,
    }


def upsert_trial(conn: sqlite3.Connection, js: Dict[str, Any]):
    fields = extract_trial_fields(js)
    ct = fields["ctNumber"]
    if not ct:
        return

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_json = json.dumps(js, ensure_ascii=False)

    conn.execute("""
        INSERT INTO trials (
            ctNumber, ctStatus, ctPublicStatusCode, title, shortTitle, sponsor,
            trialPhase, therapeuticAreas, medicalCondition, medicalConditionsList,
            isConditionRareDisease, countries, decisionDate, publishDate,
            lastUpdated, ageCategories, gender, isRandomised, blindingType,
            trialScope, mainObjective, primaryEndpointsCount, secondaryEndpointsCount,
            estimatedRecruitmentStartDate, estimatedEndDate,
            data_json, updated_at_utc
        )
        VALUES (
            :ctNumber, :ctStatus, :ctPublicStatusCode, :title, :shortTitle, :sponsor,
            :trialPhase, :therapeuticAreas, :medicalCondition, :medicalConditionsList,
            :isConditionRareDisease, :countries, :decisionDate, :publishDate,
            :lastUpdated, :ageCategories, :gender, :isRandomised, :blindingType,
            :trialScope, :mainObjective, :primaryEndpointsCount, :secondaryEndpointsCount,
            :estimatedRecruitmentStartDate, :estimatedEndDate,
            :data_json, :updated_at_utc
        )
        ON CONFLICT(ctNumber) DO UPDATE SET
            ctStatus=excluded.ctStatus,
            ctPublicStatusCode=excluded.ctPublicStatusCode,
            title=excluded.title,
            shortTitle=excluded.shortTitle,
            sponsor=excluded.sponsor,
            trialPhase=excluded.trialPhase,
            therapeuticAreas=excluded.therapeuticAreas,
            medicalCondition=excluded.medicalCondition,
            medicalConditionsList=excluded.medicalConditionsList,
            isConditionRareDisease=excluded.isConditionRareDisease,
            countries=excluded.countries,
            decisionDate=excluded.decisionDate,
            publishDate=excluded.publishDate,
            lastUpdated=excluded.lastUpdated,
            ageCategories=excluded.ageCategories,
            gender=excluded.gender,
            isRandomised=excluded.isRandomised,
            blindingType=excluded.blindingType,
            trialScope=excluded.trialScope,
            mainObjective=excluded.mainObjective,
            primaryEndpointsCount=excluded.primaryEndpointsCount,
            secondaryEndpointsCount=excluded.secondaryEndpointsCount,
            estimatedRecruitmentStartDate=excluded.estimatedRecruitmentStartDate,
            estimatedEndDate=excluded.estimatedEndDate,
            data_json=excluded.data_json,
            updated_at_utc=excluded.updated_at_utc;
    """, {**fields, "data_json": data_json, "updated_at_utc": now})


def insert_criteria_endpoints_products(conn: sqlite3.Connection, ct: str, js: Dict[str, Any]):
    """Insert inclusion/exclusion criteria, endpoints, and products."""
    
    # Delete existing records for this trial (for updates)
    conn.execute("DELETE FROM inclusion_criteria WHERE ctNumber = ?", (ct,))
    conn.execute("DELETE FROM exclusion_criteria WHERE ctNumber = ?", (ct,))
    conn.execute("DELETE FROM endpoints WHERE ctNumber = ?", (ct,))
    conn.execute("DELETE FROM trial_products WHERE ctNumber = ?", (ct,))
    
    # Inclusion criteria
    inclusion = extract_inclusion_criteria(js)
    for inc in inclusion:
        conn.execute("""
            INSERT INTO inclusion_criteria (ctNumber, criterionNumber, criterionText)
            VALUES (?, ?, ?)
        """, (ct, inc["criterionNumber"], inc["criterionText"]))
    
    # Exclusion criteria
    exclusion = extract_exclusion_criteria(js)
    for exc in exclusion:
        conn.execute("""
            INSERT INTO exclusion_criteria (ctNumber, criterionNumber, criterionText)
            VALUES (?, ?, ?)
        """, (ct, exc["criterionNumber"], exc["criterionText"]))
    
    # Endpoints
    endpoints = extract_endpoints(js)
    for ep in endpoints:
        conn.execute("""
            INSERT INTO endpoints (ctNumber, endpointType, endpointNumber, endpointText, timeFrame)
            VALUES (?, ?, ?, ?, ?)
        """, (ct, ep["endpointType"], ep["endpointNumber"], ep["endpointText"], ep["timeFrame"]))
    
    # Products
    products = extract_trial_products(js)
    for prod in products:
        raw_json = json.dumps(prod.get("raw", {}), ensure_ascii=False)
        conn.execute("""
            INSERT INTO trial_products (
                ctNumber, productRole, productName, activeSubstance, pharmaceuticalForm,
                route, maxDailyDose, maxDailyDoseUnit, maxTreatmentPeriod, maxTreatmentPeriodUnit,
                isPaediatric, isOrphanDrug, authorizationStatus, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ct, prod["productRole"], prod["productName"], prod["activeSubstance"],
              prod["pharmaceuticalForm"], prod["route"], prod["maxDailyDose"],
              prod["maxDailyDoseUnit"], prod["maxTreatmentPeriod"], prod["maxTreatmentPeriodUnit"],
              prod["isPaediatric"], prod["isOrphanDrug"], prod["authorizationStatus"], raw_json))


def insert_sites_people(conn: sqlite3.Connection, ct: str,
                        sites: List[Dict], people: List[Dict]):
    for s in sites:
        raw_json = json.dumps(s.get("raw", {}), ensure_ascii=False)
        conn.execute("""
            INSERT OR IGNORE INTO trial_sites
            (ctNumber, site_name, organisation, country, city, address, postal_code, path, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ct, s.get("site_name"), s.get("organisation"), s.get("country"),
              s.get("city"), s.get("address"), s.get("postal_code"),
              s.get("path"), raw_json))

    for p in people:
        raw_json = json.dumps(p.get("raw", {}), ensure_ascii=False)
        conn.execute("""
            INSERT OR IGNORE INTO trial_people
            (ctNumber, name, role, email, phone, country, city, site_name, organisation, path, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ct, p.get("name"), p.get("role"), p.get("email"), p.get("phone"),
              p.get("country"), p.get("city"), p.get("site_name"), p.get("organisation"),
              p.get("path"), raw_json))


# ===================== Fetch Trial =====================

def fetch_trial(ct: str, session: requests.Session) -> Dict[str, Any]:
    log(f"Fetching trial: {ct}")
    r = req(session, "GET", DETAIL_URL.format(ct=ct))
    js = _ensure_json_response(r)
    js["_id"] = ct
    try:
        size = len(r.content)
    except Exception:
        size = len(json.dumps(js))
    log(f"Successfully fetched {ct}: ~{size} bytes")
    return js


# ===================== Checkpoint Management =====================

def load_ctnumbers_checkpoint(ct_numbers_path: Path) -> Set[str]:
    if ct_numbers_path.exists():
        with ct_numbers_path.open('r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def append_ctnumbers_checkpoint(ct_numbers_path: Path, cts: List[str]):
    if not cts:
        return
    safe_append_lines(ct_numbers_path, cts)


# ===================== Trial Discovery =====================

def load_processed_trials(db_path: Path) -> Set[str]:
    if not db_path.exists():
        return set()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT ctNumber FROM trials")
            processed = {row[0] for row in cur.fetchall()}
        log(f"Found {len(processed)} already processed trials in database")
        return processed
    except Exception as e:
        log(f"Error loading processed trials: {e}", "WARN")
        return set()


def get_trial_last_updated(ct: str, db_path: Path) -> Optional[str]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT lastUpdated FROM trials WHERE ctNumber = ?", (ct,))
            row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def trial_needs_update(ct: str, new_last_updated: Any, db_path: Path) -> bool:
    old_last_updated = get_trial_last_updated(ct, db_path)
    return ts_is_newer(new_last_updated, old_last_updated)


def _has_next_page(pag: Dict[str, Any]) -> bool:
    if not isinstance(pag, dict):
        return False
    if "nextPage" in pag:
        return bool(pag.get("nextPage"))
    try:
        cur = int(pag.get("pageNumber"))
        total = int(pag.get("totalPages"))
        return (cur + 1) < total
    except Exception:
        return False


def iter_ct_numbers_segmented(session: requests.Session,
                              limit: Optional[int],
                              check_updates: bool,
                              ct_numbers_path: Path,
                              db_path: Path,
                              page_size: int) -> Tuple[List[str], List[str]]:
    """
    Enumerate ctNumbers by (status × year) with checkpoint support.
    Returns: (all_trials, trials_to_update)
    """
    seen = load_ctnumbers_checkpoint(ct_numbers_path)
    initial_count = len(seen)
    years_desc = list(range(CURRENT_YEAR, YEAR_START - 1, -1))

    trial_timestamps: Dict[str, Any] = {}

    log(f"Starting trial discovery (already have {initial_count} CT numbers in checkpoint)")

    for status in STATUS_SEGMENTS:
        for year in years_desc:
            if limit and len(seen) >= limit:
                log(f"Reached limit of {limit} trials, stopping discovery")
                break

            page = 1
            seg_new = 0
            seg_cts: List[str] = []

            while True:
                if limit and len(seen) >= limit:
                    break

                payload = {
                    "pagination": {"page": page, "size": page_size},
                    "sort": {"property": "decisionDate", "direction": "DESC"},
                    "searchCriteria": {"status": [status], "number": f"{year}-"}
                }

                try:
                    r = req(session, "POST", SEARCH_URL, json=payload)
                    js = _ensure_json_response(r)
                except Exception as e:
                    log(f"Search failed for status={status} year={year} page={page}: {e}", "ERROR")
                    break

                data = js.get("data", []) or []
                if not isinstance(data, list):
                    log(f"Unexpected search payload shape at status={status}, year={year}, page={page}", "WARN")
                    break
                if not data:
                    break

                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    ct = rec.get("ctNumber")
                    last_updated = rec.get("publishDate") or rec.get("lastUpdated") or rec.get("decisionDate")

                    if ct:
                        trial_timestamps[ct] = last_updated
                        if ct not in seen:
                            seen.add(ct)
                            seg_cts.append(ct)
                            seg_new += 1
                            if limit and len(seen) >= limit:
                                break

                pag = js.get("pagination") or {}
                if not _has_next_page(pag):
                    break
                page += 1

            if seg_new > 0:
                append_ctnumbers_checkpoint(ct_numbers_path, seg_cts)
                log(f"[status={status} year={year}] Found +{seg_new} new trials | Total unique={len(seen)}")

        if limit and len(seen) >= limit:
            break

    new_count = len(seen) - initial_count
    log(f"Discovery complete: {new_count} new trials found (total: {len(seen)})")

    all_trials = list(seen)
    if limit:
        all_trials = all_trials[:limit]

    if check_updates:
        log("Checking which trials need updating...")
        trials_to_update = []
        for ct in all_trials:
            if trial_needs_update(ct, trial_timestamps.get(ct), db_path):
                trials_to_update.append(ct)

        already_current = len(all_trials) - len(trials_to_update)
        log(f"Update check: {len(trials_to_update)} need updating, {already_current} already current")
        return all_trials, trials_to_update
    else:
        return all_trials, all_trials


# ===================== Multi-trial Processing =====================

def process_multiple_trials(to_fetch: List[str],
                            conn: sqlite3.Connection,
                            session: requests.Session,
                            db_path: Path,
                            ndjson_path: Path,
                            failed_path: Path):
    """Process multiple trials with parallel execution and checkpointing."""
    if not to_fetch:
        log("No trials to process!")
        return

    total = len(to_fetch)
    batch_json: List[Dict[str, Any]] = []
    failed: List[str] = []
    counter = 0
    updated = 0
    start_time = time.time()

    log(f"Starting extraction of {total} trials with {MAX_WORKERS} workers...")

    def _was_in_db(ct: str) -> bool:
        return get_trial_last_updated(ct, db_path) is not None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_trial, ct, session): ct for ct in to_fetch}

        for fut in as_completed(futures):
            ct = futures[fut]
            counter += 1

            try:
                rec = fut.result()

                was_in_db = _was_in_db(ct)
                upsert_trial(conn, rec)

                try:
                    sites = extract_sites(rec, ct)
                    people = extract_people(rec, ct)

                    if was_in_db:
                        conn.execute("DELETE FROM trial_sites WHERE ctNumber = ?", (ct,))
                        conn.execute("DELETE FROM trial_people WHERE ctNumber = ?", (ct,))
                        updated += 1

                    insert_sites_people(conn, ct, sites, people)
                    insert_criteria_endpoints_products(conn, ct, rec)
                except Exception as ex:
                    log(f"[WARN] {ct}: extract secondary data error: {ex}", "WARN")

                batch_json.append(rec)

                if len(batch_json) >= 50:
                    append_jsonl(ndjson_path, batch_json)
                    conn.commit()
                    batch_json.clear()

            except Exception as e:
                log(f"Failed to process {ct}: {e}", "ERROR")
                failed.append(ct)

            if counter % REPORT_EVERY == 0 or counter == total:
                elapsed = time.time() - start_time
                rate = counter / elapsed if elapsed > 0 else 0
                remaining = (total - counter) / rate if rate > 0 else 0
                log(f"Progress: [{counter}/{total}] {rate:.2f} trials/s – ETA {remaining/60:.1f} min | Updated: {updated}")

    if batch_json:
        append_jsonl(ndjson_path, batch_json)
        conn.commit()

    if failed:
        log(f"Retrying {len(failed)} failed trials...")
        recovered: List[str] = []
        for ct in failed:
            try:
                time.sleep(FINAL_COOLDOWN)
                rec = fetch_trial(ct, session)
                upsert_trial(conn, rec)

                try:
                    sites = extract_sites(rec, ct)
                    people = extract_people(rec, ct)
                    conn.execute("DELETE FROM trial_sites WHERE ctNumber = ?", (ct,))
                    conn.execute("DELETE FROM trial_people WHERE ctNumber = ?", (ct,))
                    insert_sites_people(conn, ct, sites, people)
                    insert_criteria_endpoints_products(conn, ct, rec)
                except Exception:
                    pass

                append_jsonl(ndjson_path, [rec])
                conn.commit()
                recovered.append(ct)

            except Exception as e:
                log(f"Final retry failed for {ct}: {e}", "ERROR")

        still_failed = [x for x in failed if x not in set(recovered)]
        if still_failed:
            safe_append_lines(failed_path, still_failed)
            log(f"Still failed: {len(still_failed)} trials (saved to {failed_path})", "WARN")

    log(f"✅ Extraction complete! Processed {counter - len(failed)} trials ({updated} updates)")


# ===================== Process Single Trial (continued in next message due to length)

def process_single_trial(ct_number: str,
                         conn: sqlite3.Connection,
                         session: requests.Session,
                         out_dir: Path,
                         ndjson_path: Path) -> bool:
    log(f"=== Processing single trial: {ct_number} ===")

    try:
        trial_data = fetch_trial(ct_number, session)

        single_json = out_dir / f"{ct_number}_raw.json"
        with single_json.open("w", encoding="utf-8") as f:
            json.dump(trial_data, f, ensure_ascii=False, indent=2)
        log(f"Saved raw data to: {single_json}")

        upsert_trial(conn, trial_data)
        sites_extracted = extract_sites(trial_data, ct_number)
        people_extracted = extract_people(trial_data, ct_number)
        insert_sites_people(conn, ct_number, sites_extracted, people_extracted)
        insert_criteria_endpoints_products(conn, ct_number, trial_data)
        conn.commit()

        append_jsonl(ndjson_path, [trial_data])

        fields = extract_trial_fields(trial_data)
        inclusion = extract_inclusion_criteria(trial_data)
        exclusion = extract_exclusion_criteria(trial_data)
        endpoints = extract_endpoints(trial_data)
        products = extract_trial_products(trial_data)
        
        log(f"\n=== Trial Summary ===")
        log(f"CT Number: {fields['ctNumber']}")
        log(f"Title: {fields['title']}")
        log(f"Sponsor: {fields['sponsor']}")
        log(f"Phase: {fields['trialPhase']}")
        log(f"Status: {fields['ctPublicStatusCode']} ({fields['ctStatus']})")
        log(f"Medical Condition: {fields['medicalCondition']}")
        log(f"\n=== Feasibility Info ===")
        
        # Decode age categories
        age_cats = json.loads(fields['ageCategories'])
        age_names = [AGE_CATEGORY_MAP.get(code, f"Code {code}") for code in age_cats]
        log(f"Age Categories: {', '.join(age_names) if age_names else 'Not specified'}")
        
        log(f"Gender: {fields['gender']}")
        log(f"Randomized: {'Yes' if fields['isRandomised'] else 'No'}")
        log(f"Blinding: {fields['blindingType']}")
        log(f"Recruitment Start: {fields['estimatedRecruitmentStartDate']}")
        log(f"Est. End Date: {fields['estimatedEndDate']}")
        log(f"Inclusion Criteria: {len(inclusion)}")
        log(f"Exclusion Criteria: {len(exclusion)}")
        log(f"Primary Endpoints: {fields['primaryEndpointsCount']}")
        log(f"Secondary Endpoints: {fields['secondaryEndpointsCount']}")
        log(f"Products: {len(products)}")
        log(f"Sites: {len(sites_extracted)}")
        log(f"People: {len(people_extracted)}")
        log(f"===================\n")

        # Generate detailed report
        report_path = out_dir / f"{ct_number}_report.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"CTIS Trial Report: {ct_number}\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
            f.write("=" * 80 + "\n\n")

            f.write("TRIAL INFORMATION\n")
            f.write("-" * 80 + "\n")
            for key, value in fields.items():
                if key not in ("medicalConditionsList", "mainObjective", "trialScope", "ageCategories"):
                    if key in ("decisionDate", "publishDate", "lastUpdated") and value:
                        try:
                            dt = parse_ts(value)
                            if dt:
                                value = dt.strftime('%Y-%m-%d')
                        except:
                            pass
                    f.write(f"{key:30s}: {value}\n")
            
            # Decode age categories for report
            age_cats = json.loads(fields['ageCategories'])
            age_desc = [AGE_CATEGORY_MAP.get(code, f"Code {code}") for code in age_cats]
            f.write(f"{'ageCategories':30s}: {', '.join(age_desc) if age_desc else 'Not specified'}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            f.write(f"INCLUSION CRITERIA ({len(inclusion)})\n")
            f.write("-" * 80 + "\n")
            for inc in inclusion:
                f.write(f"{inc['criterionNumber']}. {inc['criterionText']}\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write(f"EXCLUSION CRITERIA ({len(exclusion)})\n")
            f.write("-" * 80 + "\n")
            for exc in exclusion:
                f.write(f"{exc['criterionNumber']}. {exc['criterionText']}\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write(f"ENDPOINTS\n")
            f.write("-" * 80 + "\n")
            primary_eps = [e for e in endpoints if e['endpointType'] == 'primary']
            secondary_eps = [e for e in endpoints if e['endpointType'] == 'secondary']
            
            f.write(f"\nPrimary Endpoints ({len(primary_eps)}):\n")
            for ep in primary_eps:
                f.write(f"{ep['endpointNumber']}. {ep['endpointText']}\n\n")
            
            f.write(f"\nSecondary Endpoints ({len(secondary_eps)}):\n")
            for ep in secondary_eps:
                f.write(f"{ep['endpointNumber']}. {ep['endpointText']}\n\n")

            f.write("=" * 80 + "\n\n")
            f.write(f"INVESTIGATIONAL PRODUCTS ({len(products)})\n")
            f.write("-" * 80 + "\n\n")
            for prod in products:
                f.write(f"Role: {prod['productRole'].upper()}\n")
                f.write(f"  Name: {prod['productName']}\n")
                f.write(f"  Active Substance: {prod['activeSubstance']}\n")
                f.write(f"  Form: {prod['pharmaceuticalForm']}\n")
                f.write(f"  Route: {prod['route']}\n")
                f.write(f"  Max Daily Dose: {prod['maxDailyDose']} {prod['maxDailyDoseUnit']}\n")
                f.write(f"  Treatment Period: {prod['maxTreatmentPeriod']} {prod['maxTreatmentPeriodUnit']}\n")
                f.write(f"  Paediatric: {'Yes' if prod['isPaediatric'] else 'No'}\n")
                f.write(f"  Orphan Drug: {'Yes' if prod['isOrphanDrug'] else 'No'}\n")
                f.write(f"  Authorization: {prod['authorizationStatus']}\n\n")

            f.write("=" * 80 + "\n\n")
            f.write(f"SITES ({len(sites_extracted)})\n")
            f.write("-" * 80 + "\n\n")
            for idx, site in enumerate(sites_extracted, 1):
                f.write(f"Site {idx}:\n")
                f.write(f"  Name: {site.get('site_name')}\n")
                f.write(f"  Organisation: {site.get('organisation')}\n")
                f.write(f"  Country: {site.get('country')}\n")
                f.write(f"  City: {site.get('city')}\n")
                f.write(f"  Address: {site.get('address')}\n")
                f.write(f"  Postal Code: {site.get('postal_code')}\n\n")

            f.write("=" * 80 + "\n\n")
            f.write(f"PEOPLE ({len(people_extracted)})\n")
            f.write("-" * 80 + "\n\n")

            by_role = defaultdict(list)
            for person in people_extracted:
                role = person.get('role') or 'Unknown'
                by_role[role].append(person)

            for role, persons in sorted(by_role.items()):
                f.write(f"{role} ({len(persons)}):\n")
                for person in persons:
                    if person.get('name'):
                        f.write(f"  - Name: {person.get('name')}\n")
                    if person.get('email'):
                        f.write(f"    Email: {person.get('email')}\n")
                    if person.get('phone'):
                        f.write(f"    Phone: {person.get('phone')}\n")
                    if person.get('organisation'):
                        f.write(f"    Organisation: {person.get('organisation')}\n")
                    if person.get('country'):
                        f.write(f"    Location: {person.get('city')}, {person.get('country')}\n")
                    f.write("\n")

        log(f"Saved detailed report to: {report_path}")
        log(f"✅ Single trial extraction complete!")
        return True

    except Exception as e:
        log(f"Error processing trial {ct_number}: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


# ===================== Main =====================

def main():
    global MAX_WORKERS, OUT_DIR, NDJSON_PATH, DB_PATH, CTNUMBERS_PATH, FAILED_PATH, PAGE_SIZE
    global BASE_BACKOFF, MAX_RETRIES, RATE_LIMIT_RPS, REQUEST_TIMEOUT, GLOBAL_RATE_LIMITER

    print("=" * 80, file=sys.stderr)
    print("CTIS EXTRACTOR v7.1 CORRECTED – ACTUAL CTIS STRUCTURE", file=sys.stderr)
    print("Fixed: Uses real CTIS field names from actual JSON", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    parser = argparse.ArgumentParser(description="CTIS Complete Extractor (v7.1 CORRECTED)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single", metavar="CT_NUMBER", help="Extract single trial")
    mode_group.add_argument("--count", type=int, metavar="N", help="Extract N trials")
    mode_group.add_argument("--full", action="store_true", help="Extract all trials")

    parser.add_argument("--reset", action="store_true", help="Reset database & output files")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max concurrent fetch workers")
    parser.add_argument("--rate", type=float, default=RATE_LIMIT_RPS, help="Max requests per second (global)")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE, help="Search page size")
    parser.add_argument("--no-check-updates", action="store_true", help="Process all discovered trials (skip update check)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory (db, ndjson, checkpoints)")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help="Per-request timeout (seconds)")
    parser.add_argument("--base-backoff", type=float, default=BASE_BACKOFF, help="Base backoff in seconds")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max per-request retries")

    args = parser.parse_args()

    # Apply CLI config
    MAX_WORKERS = max(1, int(args.workers))
    RATE_LIMIT_RPS = max(0.1, float(args.rate))
    PAGE_SIZE = max(1, int(args.page_size))
    REQUEST_TIMEOUT = max(5.0, float(args.timeout))
    BASE_BACKOFF = max(0.1, float(args.base_backoff))
    MAX_RETRIES = max(0, int(args.max_retries))

    OUT_DIR = args.out_dir
    NDJSON_PATH = OUT_DIR / "ctis_full.ndjson"
    DB_PATH = OUT_DIR / "ctis.db"
    CTNUMBERS_PATH = OUT_DIR / "ct_numbers.txt"
    FAILED_PATH = OUT_DIR / "failed_ctnumbers.txt"

    # IO setup
    setup_output_dir(OUT_DIR, reset=args.reset)
    if args.reset:
        for p in (NDJSON_PATH, CTNUMBERS_PATH, FAILED_PATH):
            if p.exists():
                p.unlink()

    conn = init_db(DB_PATH, reset=args.reset)

    session = requests.Session()
    session.trust_env = True
    session.headers.update(BASE_HEADERS)

    adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=60, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    warm_up(session)

    GLOBAL_RATE_LIMITER = RateLimiter(RATE_LIMIT_RPS)

    try:
        if args.single:
            success = process_single_trial(args.single, conn, session, OUT_DIR, NDJSON_PATH)
            sys.exit(0 if success else 1)

        elif args.count:
            log(f"Extracting {args.count} trials...")
            all_trials, trials_to_update = iter_ct_numbers_segmented(
                session=session,
                limit=args.count,
                check_updates=not args.no_check_updates,
                ct_numbers_path=CTNUMBERS_PATH,
                db_path=DB_PATH,
                page_size=PAGE_SIZE,
            )
            process_multiple_trials(trials_to_update, conn, session, DB_PATH, NDJSON_PATH, FAILED_PATH)

        elif args.full:
            log("Full database extraction mode")
            all_trials, trials_to_update = iter_ct_numbers_segmented(
                session=session,
                limit=None,
                check_updates=not args.no_check_updates,
                ct_numbers_path=CTNUMBERS_PATH,
                db_path=DB_PATH,
                page_size=PAGE_SIZE,
            )
            process_multiple_trials(trials_to_update, conn, session, DB_PATH, NDJSON_PATH, FAILED_PATH)

    except KeyboardInterrupt:
        log("\n⚠️  Interrupted by user. Progress saved! Run again without --reset to resume.", "WARN")
        conn.commit()
        sys.exit(0)
    finally:
        try:
            conn.commit()
        except Exception:
            pass
        conn.close()
        session.close()

    log(f"\n✅ Complete!")
    log(f"Database: {DB_PATH}")
    log(f"NDJSON: {NDJSON_PATH}")


if __name__ == "__main__":
    main()