#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Utility Functions
Common helper functions used across the extractor
ctis/ctis_utils.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Tuple
from ctis_config import EMAIL_RE, PHONE_RE

# ===================== Logging =====================

def log(msg: str, level: str = "INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", file=sys.stderr, flush=True)


# ===================== Filesystem Helpers =====================

def setup_output_dir(out_dir: Path, reset: bool = False):
    """Setup output directory, optionally resetting it"""
    if out_dir.exists() and reset:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def safe_append_lines(path: Path, lines: List[str]):
    """Safely append lines to a file with fsync"""
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
    """Append JSONL records to a file"""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def count_lines(path: Path) -> int:
    """Count lines in a file"""
    if not path.exists():
        return -1
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


# ===================== Data Extraction Helpers =====================

def get(obj, *keys, default=""):
    """Safely get nested dictionary value"""
    cur = obj or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def is_blank(x):
    """Check if value is blank/empty"""
    return x is None or (isinstance(x, str) and x.strip() == "")


def pct(n, d):
    """Calculate percentage safely"""
    return 0.0 if d == 0 else (100.0 * n / d)


def coerce_int(v: Any) -> Optional[int]:
    """Convert value to int safely"""
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


# ===================== Date/Time Parsing =====================

def parse_ts(v: Any) -> Optional[datetime]:
    """Parse many timestamp shapes into timezone-aware UTC datetime"""
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


def parse_year(date_str: Any) -> Optional[int]:
    """Extract year from various date formats"""
    if is_blank(date_str):
        return None
    s = str(date_str).strip()
    import re
    match = re.match(r'^(\d{4})', s)
    return int(match.group(1)) if match else None


def parse_date(date_str: Any) -> Optional[datetime]:
    """Parse date string to datetime object"""
    if is_blank(date_str):
        return None
    s = str(date_str).strip()
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def ts_is_newer(new_ts: Any, old_ts: Any) -> bool:
    """Return True if new_ts > old_ts (with robust parsing)"""
    new_dt = parse_ts(new_ts)
    old_dt = parse_ts(old_ts)
    if new_dt is None:
        return True
    if old_dt is None:
        return True
    return new_dt > old_dt


# ===================== Data Normalization =====================

def is_pediatric_trial(age_categories: List) -> bool:
    """
    Returns True if trial includes any pediatric age categories (codes 1-5)
    Pediatric = under 18 years old
    Handles both string and integer codes
    """
    if not age_categories:
        return False
    pediatric_codes = {1, 2, 3, 4, 5, "1", "2", "3", "4", "5"}
    return any(code in pediatric_codes for code in age_categories)


def is_adult_trial(age_categories: List) -> bool:
    """
    Returns True if trial includes any adult age categories (codes 6-8)
    Adult = 18 years and older
    Handles both string and integer codes
    """
    if not age_categories:
        return False
    adult_codes = {6, 7, 8, "6", "7", "8"}
    return any(code in adult_codes for code in age_categories)


def normalize_country(node: Dict[str, Any]) -> Optional[str]:
    """Extract country name from various node structures"""
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


def extract_email_phone(d: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract email and phone from contact dict"""
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


# ===================== Timing =====================

def sleep_jitter():
    """Sleep for a random jitter amount"""
    import random
    from ctis_config import JITTER_RANGE
    time.sleep(random.uniform(*JITTER_RANGE))


def backoff(i: int, base: float = None):
    """Exponential backoff with jitter"""
    import random
    if base is None:
        from ctis_config import BASE_BACKOFF
        base = BASE_BACKOFF
    time.sleep((base * (2 ** i)) + random.uniform(0, 0.6))