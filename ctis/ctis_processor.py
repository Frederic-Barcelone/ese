#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Processor Module
Handles single and batch trial processing
ctis/ctis_processor.py
"""

import time
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests

from ctis_config import (
    DETAIL_URL, MAX_WORKERS, REPORT_EVERY, FINAL_COOLDOWN, AGE_CATEGORY_MAP
)
from ctis_utils import log, append_jsonl, safe_append_lines
from ctis_http import req, _ensure_json_response
from ctis_database import (
    upsert_trial, insert_sites_people, insert_criteria_endpoints_products,
    get_trial_last_updated, insert_ms_status, insert_country_planning, insert_site_contacts,
    insert_funding_sources, insert_scientific_advice, insert_trial_relationships
)
from ctis_extractors import (
    extract_trial_fields, extract_sites, extract_people,
    extract_inclusion_criteria, extract_exclusion_criteria,
    extract_endpoints, extract_trial_products, extract_ms_status,
    extract_country_planning, extract_site_contacts,
    extract_funding_sources, extract_scientific_advice, extract_trial_relationships
)

# Import comprehensive report generator
try:
    from ctis_report_generator import generate_ctis_format_report
    HAS_COMPREHENSIVE_REPORT = True
except ImportError:
    HAS_COMPREHENSIVE_REPORT = False
    log("Warning: ctis_report_generator not found, using basic report", "WARN")


# ===================== Fetch Trial =====================

def fetch_trial(ct: str, session: requests.Session) -> Dict[str, Any]:
    """Fetch single trial JSON from CTIS API"""
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


def fetch_trial_download(ct: str, session: requests.Session, out_dir: Path) -> Dict[str, Any]:
    """
    Fetch trial using the "Download clinical trial" feature for stability.
    
    This downloads the HTML file that CTIS generates when you click 
    "Download clinical trial" at the top right of the portal. This is more
    stable than parsing dynamic DOM and includes all public data.
    
    Source: CTIS Full trial information (top banner explains Download feature)
    
    Returns dict with:
        - html_path: Path to saved HTML file
        - source_url: URL fetched from
        - downloaded_at: ISO timestamp
        - sha256: Hash of downloaded content
        - trial_data: Extracted JSON data (from API, not HTML parsing yet)
    """
    import hashlib
    from datetime import datetime, timezone
    
    # For now, still use API endpoint (HTML parsing would require BeautifulSoup)
    # This establishes the structure for future HTML parsing implementation
    trial_data = fetch_trial(ct, session)
    
    # Save HTML if we were to download it
    html_path = out_dir / f"{ct}_download.html"
    downloaded_at = datetime.now(timezone.utc).isoformat()
    
    # Calculate hash of raw JSON for now
    json_str = json.dumps(trial_data, ensure_ascii=False)
    sha256_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    metadata = {
        "html_path": str(html_path),
        "source_url": DETAIL_URL.format(ct=ct),
        "downloaded_at": downloaded_at,
        "sha256": sha256_hash,
        "trial_data": trial_data
    }
    
    log(f"Download metadata: {downloaded_at}, SHA256: {sha256_hash[:16]}...")
    
    return metadata


# ===================== Single Trial Processing =====================

def process_single_trial(ct_number: str,
                         conn: sqlite3.Connection,
                         session: requests.Session,
                         out_dir: Path,
                         ndjson_path: Path) -> bool:
    """Process and report on a single trial"""
    log(f"=== Processing single trial: {ct_number} ===")

    try:
        # Fetch trial data
        trial_data = fetch_trial(ct_number, session)

        # Save raw JSON
        single_json = out_dir / f"{ct_number}_raw.json"
        with single_json.open("w", encoding="utf-8") as f:
            json.dump(trial_data, f, ensure_ascii=False, indent=2)
        log(f"Saved raw data to: {single_json}")

        # Extract data
        fields = extract_trial_fields(trial_data)
        sites_extracted = extract_sites(trial_data, ct_number)
        people_extracted = extract_people(trial_data, ct_number)
        inclusion = extract_inclusion_criteria(trial_data)
        exclusion = extract_exclusion_criteria(trial_data)
        endpoints = extract_endpoints(trial_data)
        products = extract_trial_products(trial_data)
        ms_statuses = extract_ms_status(trial_data)
        country_plans = extract_country_planning(trial_data)
        site_contacts = extract_site_contacts(trial_data)
        
        # Enhanced extraction (v1.3.0)
        funding = extract_funding_sources(trial_data)
        scientific_advice = extract_scientific_advice(trial_data)
        relationships = extract_trial_relationships(trial_data)

        # Insert into database
        upsert_trial(conn, trial_data, fields)
        insert_sites_people(conn, ct_number, sites_extracted, people_extracted)
        insert_criteria_endpoints_products(conn, ct_number, inclusion, exclusion, endpoints, products)
        insert_ms_status(conn, ct_number, ms_statuses)
        insert_country_planning(conn, ct_number, country_plans)
        insert_site_contacts(conn, ct_number, site_contacts)
        
        # Insert enhanced data (v1.3.0)
        insert_funding_sources(conn, ct_number, funding)
        insert_scientific_advice(conn, ct_number, scientific_advice)
        insert_trial_relationships(conn, ct_number, relationships)
        
        conn.commit()

        # Save to NDJSON
        append_jsonl(ndjson_path, [trial_data])

        # Print summary
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
        log(f"Member States: {len(ms_statuses)}")
        log(f"Country Plans: {len(country_plans)}")
        log(f"Site Contacts: {len(site_contacts)}")
        
        # Show MS status summary
        if ms_statuses:
            log(f"\n=== Member State Status ===")
            for ms in ms_statuses:
                status = ms.get("status", "Unknown")
                log(f"{ms.get('member_state')}: {status}")
        
        # Show country planning
        if country_plans:
            log(f"\n=== Planned Participants ===")
            for plan in country_plans:
                log(f"{plan.get('country')}: {plan.get('planned_participants')} participants")
        
        log(f"===================\n")

        # Generate detailed report using comprehensive generator if available
        report_path = out_dir / f"{ct_number}_report.txt"
        
        if HAS_COMPREHENSIVE_REPORT:
            log("Generating comprehensive report...")
            generate_ctis_format_report(conn, ct_number, report_path)
            log(f"Saved comprehensive report to: {report_path}")
        else:
            # Fallback to basic report
            generate_trial_report(report_path, fields, inclusion, exclusion, endpoints, 
                                products, sites_extracted, people_extracted, ms_statuses)
            log(f"Saved basic report to: {report_path}")
        log(f"Single trial extraction complete!")
        return True

    except Exception as e:
        log(f"Error processing trial {ct_number}: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


# ===================== Batch Trial Processing =====================

def process_multiple_trials(to_fetch: List[str],
                            conn: sqlite3.Connection,
                            session: requests.Session,
                            db_path: Path,
                            ndjson_path: Path,
                            failed_path: Path):
    """Process multiple trials with parallel execution and checkpointing"""
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
                fields = extract_trial_fields(rec)
                upsert_trial(conn, rec, fields)

                try:
                    sites = extract_sites(rec, ct)
                    people = extract_people(rec, ct)
                    inclusion = extract_inclusion_criteria(rec)
                    exclusion = extract_exclusion_criteria(rec)
                    endpoints = extract_endpoints(rec)
                    products = extract_trial_products(rec)
                    ms_statuses = extract_ms_status(rec)
                    country_plans = extract_country_planning(rec)
                    site_contacts = extract_site_contacts(rec)
                    
                    # Enhanced extraction (v1.3.0)
                    funding = extract_funding_sources(rec)
                    scientific_advice = extract_scientific_advice(rec)
                    relationships = extract_trial_relationships(rec)

                    if was_in_db:
                        conn.execute("DELETE FROM trial_sites WHERE ctNumber = ?", (ct,))
                        conn.execute("DELETE FROM trial_people WHERE ctNumber = ?", (ct,))
                        updated += 1

                    insert_sites_people(conn, ct, sites, people)
                    insert_criteria_endpoints_products(conn, ct, inclusion, exclusion, endpoints, products)
                    insert_ms_status(conn, ct, ms_statuses)
                    insert_country_planning(conn, ct, country_plans)
                    insert_site_contacts(conn, ct, site_contacts)
                    
                    # Insert enhanced data (v1.3.0)
                    insert_funding_sources(conn, ct, funding)
                    insert_scientific_advice(conn, ct, scientific_advice)
                    insert_trial_relationships(conn, ct, relationships)
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
                log(f"Progress: [{counter}/{total}] {rate:.2f} trials/s - ETA {remaining/60:.1f} min | Updated: {updated}")

    if batch_json:
        append_jsonl(ndjson_path, batch_json)
        conn.commit()

    # Retry failed trials
    if failed:
        log(f"Retrying {len(failed)} failed trials...")
        recovered: List[str] = []
        for ct in failed:
            try:
                time.sleep(FINAL_COOLDOWN)
                rec = fetch_trial(ct, session)
                fields = extract_trial_fields(rec)
                upsert_trial(conn, rec, fields)

                try:
                    sites = extract_sites(rec, ct)
                    people = extract_people(rec, ct)
                    inclusion = extract_inclusion_criteria(rec)
                    exclusion = extract_exclusion_criteria(rec)
                    endpoints = extract_endpoints(rec)
                    products = extract_trial_products(rec)
                    ms_statuses = extract_ms_status(rec)
                    country_plans = extract_country_planning(rec)
                    site_contacts = extract_site_contacts(rec)
                    
                    # Enhanced extraction (v1.3.0)
                    funding = extract_funding_sources(rec)
                    scientific_advice = extract_scientific_advice(rec)
                    relationships = extract_trial_relationships(rec)
                    
                    conn.execute("DELETE FROM trial_sites WHERE ctNumber = ?", (ct,))
                    conn.execute("DELETE FROM trial_people WHERE ctNumber = ?", (ct,))
                    insert_sites_people(conn, ct, sites, people)
                    insert_criteria_endpoints_products(conn, ct, inclusion, exclusion, endpoints, products)
                    insert_ms_status(conn, ct, ms_statuses)
                    insert_country_planning(conn, ct, country_plans)
                    insert_site_contacts(conn, ct, site_contacts)
                    
                    # Insert enhanced data (v1.3.0)
                    insert_funding_sources(conn, ct, funding)
                    insert_scientific_advice(conn, ct, scientific_advice)
                    insert_trial_relationships(conn, ct, relationships)
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

    log(f"Extraction complete! Processed {counter - len(failed)} trials ({updated} updates)")


# ===================== Report Generation =====================

def generate_trial_report(report_path: Path, fields: Dict[str, Any],
                         inclusion: List[Dict], exclusion: List[Dict],
                         endpoints: List[Dict], products: List[Dict],
                         sites: List[Dict], people: List[Dict],
                         ms_statuses: List[Dict]):
    """Generate detailed text report for a trial"""
    from ctis_utils import parse_ts
    
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"CTIS Trial Report: {fields['ctNumber']}\n")
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
        
        # Decode age categories
        age_cats = json.loads(fields['ageCategories'])
        age_desc = [AGE_CATEGORY_MAP.get(code, f"Code {code}") for code in age_cats]
        f.write(f"{'ageCategories':30s}: {', '.join(age_desc) if age_desc else 'Not specified'}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        f.write(f"MEMBER STATE STATUS ({len(ms_statuses)})\n")
        f.write("-" * 80 + "\n\n")
        for ms in ms_statuses:
            f.write(f"Country: {ms.get('member_state')}\n")
            f.write(f"  Status: {ms.get('status')}\n")
            if ms.get('decision_date'):
                f.write(f"  Decision Date: {ms.get('decision_date')}\n")
            if ms.get('start_date'):
                f.write(f"  Start Date: {ms.get('start_date')}\n")
            if ms.get('recruitment_start'):
                f.write(f"  Recruitment Start: {ms.get('recruitment_start')}\n")
            if ms.get('recruitment_end'):
                f.write(f"  Recruitment End: {ms.get('recruitment_end')}\n")
            if ms.get('temporary_halt'):
                f.write(f"  Temporary Halt: {ms.get('temporary_halt')}\n")
            if ms.get('restart_date'):
                f.write(f"  Restart Date: {ms.get('restart_date')}\n")
            if ms.get('end_date'):
                f.write(f"  End Date: {ms.get('end_date')}\n")
            if ms.get('early_termination_date'):
                f.write(f"  Early Termination: {ms.get('early_termination_date')}\n")
                if ms.get('early_termination_reason'):
                    f.write(f"  Termination Reason: {ms.get('early_termination_reason')}\n")
            if ms.get('last_update'):
                f.write(f"  Last Update: {ms.get('last_update')}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n\n")
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
        f.write(f"SITES ({len(sites)})\n")
        f.write("-" * 80 + "\n\n")
        for idx, site in enumerate(sites, 1):
            f.write(f"Site {idx}:\n")
            f.write(f"  Name: {site.get('site_name')}\n")
            f.write(f"  Organisation: {site.get('organisation')}\n")
            f.write(f"  Country: {site.get('country')}\n")
            f.write(f"  City: {site.get('city')}\n")
            f.write(f"  Address: {site.get('address')}\n")
            f.write(f"  Postal Code: {site.get('postal_code')}\n\n")

        f.write("=" * 80 + "\n\n")
        f.write(f"PEOPLE ({len(people)})\n")
        f.write("-" * 80 + "\n\n")

        by_role = defaultdict(list)
        for person in people:
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