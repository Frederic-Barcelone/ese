#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Database Module
Handles all database operations including schema, inserts, and queries
ctis/ctis_database.py
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from ctis_utils import log

# ===================== Database Schema =====================

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
    conditionMeddraCode TEXT,
    conditionMeddraLabel TEXT,
    conditionSynonyms TEXT,
    conditionAbbreviations TEXT,
    countries TEXT,
    decisionDate TEXT,
    publishDate TEXT,
    lastUpdated TEXT,
    
    -- FEASIBILITY FIELDS
    ageCategories TEXT,
    isPediatric INTEGER DEFAULT 0,
    isAdult INTEGER DEFAULT 0,
    gender TEXT,
    isRandomised INTEGER,
    blindingType TEXT,
    trialScope TEXT,
    mainObjective TEXT,
    primaryEndpointsCount INTEGER,
    secondaryEndpointsCount INTEGER,
    estimatedRecruitmentStartDate TEXT,
    estimatedEndDate TEXT,
    
    -- DISCLOSURE TIMING FIELDS
    trialCategory TEXT,
    expectedDosageDisclosureDate TEXT,
    dosageVisibleNow INTEGER,
    
    -- ENHANCED FIELDS (v1.3.0)
    -- Trial identifiers
    who_utn TEXT,
    nct_number TEXT,
    isrctn_number TEXT,
    additional_registry_ids TEXT,
    
    -- Regulatory
    pip_number TEXT,
    pip_decision_date TEXT,
    is_transition_trial INTEGER DEFAULT 0,
    eudract_number TEXT,
    
    -- Timeline
    global_end_date TEXT,
    
    -- Trial design details
    allocation_method TEXT,
    number_of_arms INTEGER,
    
    data_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_lastUpdated ON trials(lastUpdated);
CREATE INDEX IF NOT EXISTS idx_trials_medicalCondition ON trials(medicalCondition);
CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(trialPhase);
CREATE INDEX IF NOT EXISTS idx_trials_nct ON trials(nct_number);
CREATE INDEX IF NOT EXISTS idx_trials_pip ON trials(pip_number);
CREATE INDEX IF NOT EXISTS idx_trials_eudract ON trials(eudract_number);

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
    atcCode TEXT,
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

CREATE TABLE IF NOT EXISTS ms_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    member_state TEXT NOT NULL,
    status TEXT,
    decision_date TEXT,
    start_date TEXT,
    recruitment_start TEXT,
    recruitment_end TEXT,
    temporary_halt TEXT,
    restart_date TEXT,
    end_date TEXT,
    early_termination_date TEXT,
    early_termination_reason TEXT,
    last_update TEXT,
    captured_at TEXT NOT NULL,
    row_hash TEXT,
    UNIQUE(ctNumber, member_state, status, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_ms_status_ct ON ms_status(ctNumber);
CREATE INDEX IF NOT EXISTS idx_ms_status_country ON ms_status(member_state);
CREATE INDEX IF NOT EXISTS idx_ms_status_country_status ON ms_status(member_state, status);
CREATE INDEX IF NOT EXISTS idx_ms_status_recruiting ON ms_status(status, recruitment_start);

CREATE TABLE IF NOT EXISTS country_planning (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    country TEXT NOT NULL,
    planned_participants INTEGER,
    UNIQUE(ctNumber, country)
);
CREATE INDEX IF NOT EXISTS idx_country_planning_ct ON country_planning(ctNumber);

CREATE TABLE IF NOT EXISTS site_contact (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    country TEXT,
    org_name TEXT,
    site_name TEXT,
    address TEXT,
    city TEXT,
    postal_code TEXT,
    pi_name TEXT,
    pi_email TEXT,
    pi_phone TEXT,
    UNIQUE(ctNumber, country, org_name, site_name, pi_name, pi_email)
);
CREATE INDEX IF NOT EXISTS idx_site_contact_ct ON site_contact(ctNumber);

-- ENHANCED TABLES (v1.3.0)

CREATE TABLE IF NOT EXISTS trial_funding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    funding_source_type TEXT,
    funding_source_name TEXT,
    funding_source_country TEXT,
    is_primary_funder INTEGER DEFAULT 0,
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_funding_ct ON trial_funding(ctNumber);

CREATE TABLE IF NOT EXISTS trial_scientific_advice (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    advice_authority TEXT,
    advice_type TEXT,
    advice_date TEXT,
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_advice_ct ON trial_scientific_advice(ctNumber);

CREATE TABLE IF NOT EXISTS trial_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ctNumber TEXT NOT NULL,
    related_ctNumber TEXT NOT NULL,
    relationship_type TEXT,
    description TEXT,
    UNIQUE(ctNumber, related_ctNumber, relationship_type),
    FOREIGN KEY (ctNumber) REFERENCES trials(ctNumber)
);
CREATE INDEX IF NOT EXISTS idx_relationships_ct ON trial_relationships(ctNumber);
CREATE INDEX IF NOT EXISTS idx_relationships_related ON trial_relationships(related_ctNumber);
"""


# ===================== Database Initialization =====================

def init_db(db_path: Path, reset: bool = False) -> sqlite3.Connection:
    """Initialize database with schema"""
    if reset and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(DDL)
    log(f"Database initialized: {db_path}")
    return conn


# ===================== Safe Query Execution =====================

def safe_execute(cur, query, params=()):
    """Execute query with error handling"""
    try:
        cur.execute(query, params)
        return cur.fetchall()
    except sqlite3.OperationalError as e:
        log(f"Query failed: {e}", "WARN")
        return []


# ===================== Trial Operations =====================

def upsert_trial(conn: sqlite3.Connection, js: Dict[str, Any], fields: Dict[str, Any]):
    """Insert or update trial record"""
    ct = fields["ctNumber"]
    if not ct:
        return

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_json = json.dumps(js, ensure_ascii=False)

    conn.execute("""
        INSERT INTO trials (
            ctNumber, ctStatus, ctPublicStatusCode, title, shortTitle, sponsor,
            trialPhase, therapeuticAreas, medicalCondition, medicalConditionsList,
            isConditionRareDisease, conditionMeddraCode, conditionMeddraLabel, 
            conditionSynonyms, conditionAbbreviations,
            countries, decisionDate, publishDate,
            lastUpdated, ageCategories, isPediatric, isAdult, gender, isRandomised, blindingType,
            trialScope, mainObjective, primaryEndpointsCount, secondaryEndpointsCount,
            estimatedRecruitmentStartDate, estimatedEndDate,
            trialCategory, expectedDosageDisclosureDate, dosageVisibleNow,
            who_utn, nct_number, isrctn_number, additional_registry_ids,
            pip_number, pip_decision_date, is_transition_trial, eudract_number,
            global_end_date, allocation_method, number_of_arms,
            data_json, updated_at_utc
        )
        VALUES (
            :ctNumber, :ctStatus, :ctPublicStatusCode, :title, :shortTitle, :sponsor,
            :trialPhase, :therapeuticAreas, :medicalCondition, :medicalConditionsList,
            :isConditionRareDisease, :conditionMeddraCode, :conditionMeddraLabel,
            :conditionSynonyms, :conditionAbbreviations,
            :countries, :decisionDate, :publishDate,
            :lastUpdated, :ageCategories, :isPediatric, :isAdult, :gender, :isRandomised, :blindingType,
            :trialScope, :mainObjective, :primaryEndpointsCount, :secondaryEndpointsCount,
            :estimatedRecruitmentStartDate, :estimatedEndDate,
            :trialCategory, :expectedDosageDisclosureDate, :dosageVisibleNow,
            :who_utn, :nct_number, :isrctn_number, :additional_registry_ids,
            :pip_number, :pip_decision_date, :is_transition_trial, :eudract_number,
            :global_end_date, :allocation_method, :number_of_arms,
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
            conditionMeddraCode=excluded.conditionMeddraCode,
            conditionMeddraLabel=excluded.conditionMeddraLabel,
            conditionSynonyms=excluded.conditionSynonyms,
            conditionAbbreviations=excluded.conditionAbbreviations,
            countries=excluded.countries,
            decisionDate=excluded.decisionDate,
            publishDate=excluded.publishDate,
            lastUpdated=excluded.lastUpdated,
            ageCategories=excluded.ageCategories,
            isPediatric=excluded.isPediatric,
            isAdult=excluded.isAdult,
            gender=excluded.gender,
            isRandomised=excluded.isRandomised,
            blindingType=excluded.blindingType,
            trialScope=excluded.trialScope,
            mainObjective=excluded.mainObjective,
            primaryEndpointsCount=excluded.primaryEndpointsCount,
            secondaryEndpointsCount=excluded.secondaryEndpointsCount,
            estimatedRecruitmentStartDate=excluded.estimatedRecruitmentStartDate,
            estimatedEndDate=excluded.estimatedEndDate,
            trialCategory=excluded.trialCategory,
            expectedDosageDisclosureDate=excluded.expectedDosageDisclosureDate,
            dosageVisibleNow=excluded.dosageVisibleNow,
            who_utn=excluded.who_utn,
            nct_number=excluded.nct_number,
            isrctn_number=excluded.isrctn_number,
            additional_registry_ids=excluded.additional_registry_ids,
            pip_number=excluded.pip_number,
            pip_decision_date=excluded.pip_decision_date,
            is_transition_trial=excluded.is_transition_trial,
            eudract_number=excluded.eudract_number,
            global_end_date=excluded.global_end_date,
            allocation_method=excluded.allocation_method,
            number_of_arms=excluded.number_of_arms,
            data_json=excluded.data_json,
            updated_at_utc=excluded.updated_at_utc;
    """, {**fields, "data_json": data_json, "updated_at_utc": now})


# ===================== Sites and People Operations =====================

def insert_sites_people(conn: sqlite3.Connection, ct: str,
                        sites: List[Dict], people: List[Dict]):
    """Insert sites and people records"""
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


# ===================== Criteria, Endpoints, Products Operations =====================

def insert_criteria_endpoints_products(conn: sqlite3.Connection, ct: str,
                                      inclusion: List[Dict], exclusion: List[Dict],
                                      endpoints: List[Dict], products: List[Dict]):
    """Insert inclusion/exclusion criteria, endpoints, and products"""
    
    # Delete existing records for this trial (for updates)
    conn.execute("DELETE FROM inclusion_criteria WHERE ctNumber = ?", (ct,))
    conn.execute("DELETE FROM exclusion_criteria WHERE ctNumber = ?", (ct,))
    conn.execute("DELETE FROM endpoints WHERE ctNumber = ?", (ct,))
    conn.execute("DELETE FROM trial_products WHERE ctNumber = ?", (ct,))
    
    # Inclusion criteria
    for inc in inclusion:
        conn.execute("""
            INSERT INTO inclusion_criteria (ctNumber, criterionNumber, criterionText)
            VALUES (?, ?, ?)
        """, (ct, inc["criterionNumber"], inc["criterionText"]))
    
    # Exclusion criteria
    for exc in exclusion:
        conn.execute("""
            INSERT INTO exclusion_criteria (ctNumber, criterionNumber, criterionText)
            VALUES (?, ?, ?)
        """, (ct, exc["criterionNumber"], exc["criterionText"]))
    
    # Endpoints
    for ep in endpoints:
        conn.execute("""
            INSERT INTO endpoints (ctNumber, endpointType, endpointNumber, endpointText, timeFrame)
            VALUES (?, ?, ?, ?, ?)
        """, (ct, ep["endpointType"], ep["endpointNumber"], ep["endpointText"], ep["timeFrame"]))
    
    # Products
    for prod in products:
        raw_json = json.dumps(prod.get("raw", {}), ensure_ascii=False)
        conn.execute("""
            INSERT INTO trial_products (
                ctNumber, productRole, productName, activeSubstance, atcCode, pharmaceuticalForm,
                route, maxDailyDose, maxDailyDoseUnit, maxTreatmentPeriod, maxTreatmentPeriodUnit,
                isPaediatric, isOrphanDrug, authorizationStatus, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ct, prod["productRole"], prod["productName"], prod["activeSubstance"],
              prod.get("atcCode", ""), prod["pharmaceuticalForm"], prod["route"], prod["maxDailyDose"],
              prod["maxDailyDoseUnit"], prod["maxTreatmentPeriod"], prod["maxTreatmentPeriodUnit"],
              prod["isPaediatric"], prod["isOrphanDrug"], prod["authorizationStatus"], raw_json))


# ===================== Query Operations =====================

def load_processed_trials(db_path: Path) -> Set[str]:
    """Load set of already processed trial numbers"""
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
    """Get last updated timestamp for a trial"""
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
    """Check if trial needs updating based on timestamp"""
    from ctis_utils import ts_is_newer
    old_last_updated = get_trial_last_updated(ct, db_path)
    return ts_is_newer(new_last_updated, old_last_updated)


# ===================== MS Status Operations =====================

def insert_ms_status(conn: sqlite3.Connection, ct: str, ms_statuses: List[Dict]):
    """Insert member state status records with idempotent logic"""
    import hashlib
    
    # Get current timestamp
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    for ms in ms_statuses:
        # Create a hash of the key fields to detect changes
        hash_data = json.dumps({
            "ctNumber": ct,
            "member_state": ms.get("member_state"),
            "status": ms.get("status"),
            "decision_date": ms.get("decision_date"),
            "start_date": ms.get("start_date"),
            "recruitment_start": ms.get("recruitment_start"),
            "recruitment_end": ms.get("recruitment_end"),
            "temporary_halt": ms.get("temporary_halt"),
            "restart_date": ms.get("restart_date"),
            "end_date": ms.get("end_date"),
            "early_termination_date": ms.get("early_termination_date"),
            "early_termination_reason": ms.get("early_termination_reason"),
            "last_update": ms.get("last_update")
        }, sort_keys=True, ensure_ascii=False)
        
        row_hash = hashlib.md5(hash_data.encode('utf-8')).hexdigest()
        
        # Check if this exact record already exists
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM ms_status
            WHERE ctNumber = ? AND member_state = ? AND row_hash = ?
            ORDER BY captured_at DESC LIMIT 1
        """, (ct, ms.get("member_state"), row_hash))
        
        existing = cur.fetchone()
        
        # Only insert if this is a new/changed record
        if not existing:
            conn.execute("""
                INSERT INTO ms_status (
                    ctNumber, member_state, status, decision_date, start_date,
                    recruitment_start, recruitment_end, temporary_halt, restart_date,
                    end_date, early_termination_date, early_termination_reason, last_update,
                    captured_at, row_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ct, ms.get("member_state"), ms.get("status"), ms.get("decision_date"),
                  ms.get("start_date"), ms.get("recruitment_start"), ms.get("recruitment_end"),
                  ms.get("temporary_halt"), ms.get("restart_date"), ms.get("end_date"),
                  ms.get("early_termination_date"), ms.get("early_termination_reason"),
                  ms.get("last_update"), now, row_hash))


# ===================== Country Planning Operations =====================

def insert_country_planning(conn: sqlite3.Connection, ct: str, country_plans: List[Dict]):
    """Insert country planning records"""
    # Delete existing records for this trial
    conn.execute("DELETE FROM country_planning WHERE ctNumber = ?", (ct,))
    
    for plan in country_plans:
        conn.execute("""
            INSERT INTO country_planning (ctNumber, country, planned_participants)
            VALUES (?, ?, ?)
        """, (ct, plan.get("country"), plan.get("planned_participants")))


# ===================== Site Contact Operations =====================

def insert_site_contacts(conn: sqlite3.Connection, ct: str, site_contacts: List[Dict]):
    """Insert site contact records"""
    # Delete existing records for this trial
    conn.execute("DELETE FROM site_contact WHERE ctNumber = ?", (ct,))
    
    for contact in site_contacts:
        conn.execute("""
            INSERT INTO site_contact (
                ctNumber, country, org_name, site_name, address, city, postal_code,
                pi_name, pi_email, pi_phone
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ct, contact.get("country"), contact.get("org_name"), contact.get("site_name"),
              contact.get("address"), contact.get("city"), contact.get("postal_code"),
              contact.get("pi_name"), contact.get("pi_email"), contact.get("pi_phone")))


# ===================== Enhanced Tables Operations (v1.3.0) =====================

def insert_funding_sources(conn: sqlite3.Connection, ct: str, funding_sources: List[Dict]):
    """Insert funding source records"""
    # Delete existing records for this trial
    conn.execute("DELETE FROM trial_funding WHERE ctNumber = ?", (ct,))
    
    for funding in funding_sources:
        conn.execute("""
            INSERT INTO trial_funding (
                ctNumber, funding_source_type, funding_source_name, 
                funding_source_country, is_primary_funder
            )
            VALUES (?, ?, ?, ?, ?)
        """, (ct, funding.get("funding_source_type"), funding.get("funding_source_name"),
              funding.get("funding_source_country"), funding.get("is_primary_funder", 0)))


def insert_scientific_advice(conn: sqlite3.Connection, ct: str, advice_records: List[Dict]):
    """Insert scientific advice records"""
    # Delete existing records for this trial
    conn.execute("DELETE FROM trial_scientific_advice WHERE ctNumber = ?", (ct,))
    
    for advice in advice_records:
        conn.execute("""
            INSERT INTO trial_scientific_advice (
                ctNumber, advice_authority, advice_type, advice_date
            )
            VALUES (?, ?, ?, ?)
        """, (ct, advice.get("advice_authority"), advice.get("advice_type"),
              advice.get("advice_date")))


def insert_trial_relationships(conn: sqlite3.Connection, ct: str, relationships: List[Dict]):
    """Insert trial relationship records"""
    # Don't delete - relationships should accumulate
    
    for rel in relationships:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO trial_relationships (
                    ctNumber, related_ctNumber, relationship_type, description
                )
                VALUES (?, ?, ?, ?)
            """, (ct, rel.get("related_ctNumber"), rel.get("relationship_type"),
                  rel.get("description")))
        except sqlite3.IntegrityError:
            # Duplicate relationship, ignore
            pass