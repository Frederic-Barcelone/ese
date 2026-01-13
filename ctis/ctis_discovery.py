#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Discovery Module
Handles trial discovery, enumeration, and checkpoint management
ctis/ctis_discovery.py
"""

from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict, Any
from datetime import datetime
from ctis_config import (
    SEARCH_URL, STATUS_SEGMENTS, YEAR_START, PAGE_SIZE
)
from ctis_utils import log, safe_append_lines
from ctis_http import req, _ensure_json_response
from ctis_database import trial_needs_update
import requests

# ===================== Checkpoint Management =====================

def load_ctnumbers_checkpoint(ct_numbers_path: Path) -> Set[str]:
    """Load checkpoint of discovered trial numbers"""
    if ct_numbers_path.exists():
        with ct_numbers_path.open('r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def append_ctnumbers_checkpoint(ct_numbers_path: Path, cts: List[str]):
    """Append trial numbers to checkpoint file"""
    if not cts:
        return
    safe_append_lines(ct_numbers_path, cts)


# ===================== Pagination =====================

def _has_next_page(pag: Dict[str, Any]) -> bool:
    """Check if search results have next page"""
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


# ===================== Trial Discovery =====================

def iter_ct_numbers_segmented(session: requests.Session,
                              limit: Optional[int],
                              check_updates: bool,
                              ct_numbers_path: Path,
                              db_path: Path,
                              page_size: int = PAGE_SIZE,
                              filter_rare_disease: bool = False) -> Tuple[List[str], List[str]]:
    """
    Enumerate ctNumbers by (status × year) with checkpoint support.
    
    Args:
        session: HTTP session
        limit: Maximum number of trials to discover (None = no limit)
        check_updates: Whether to check if trials need updating
        ct_numbers_path: Path to checkpoint file
        db_path: Path to database
        page_size: Number of results per page
        filter_rare_disease: If True, only discover trials marked as rare diseases
    
    Returns: (all_trials, trials_to_update)
    """
    seen = load_ctnumbers_checkpoint(ct_numbers_path)
    initial_count = len(seen)
    
    current_year = datetime.utcnow().year
    years_desc = list(range(current_year, YEAR_START - 1, -1))

    trial_timestamps: Dict[str, Any] = {}

    if filter_rare_disease:
        log(f"Starting RARE DISEASE trial discovery (already have {initial_count} CT numbers in checkpoint)")
    else:
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

                # Build search payload
                search_criteria = {
                    "status": [status], 
                    "number": f"{year}-"
                }
                
                # Add rare disease filter if enabled
                if filter_rare_disease:
                    search_criteria["rareDisease"] = True
                
                payload = {
                    "pagination": {"page": page, "size": page_size},
                    "sort": {"property": "decisionDate", "direction": "DESC"},
                    "searchCriteria": search_criteria
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
                if filter_rare_disease:
                    log(f"[status={status} year={year}] Found +{seg_new} new RARE DISEASE trials | Total unique={len(seen)}")
                else:
                    log(f"[status={status} year={year}] Found +{seg_new} new trials | Total unique={len(seen)}")

        if limit and len(seen) >= limit:
            break

    new_count = len(seen) - initial_count
    if filter_rare_disease:
        log(f"Discovery complete: {new_count} new RARE DISEASE trials found (total: {len(seen)})")
    else:
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