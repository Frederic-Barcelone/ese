#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTIS Extractors Module - ENHANCED VERSION
All data extraction functions for parsing CTIS trial JSON
ctis/ctis_extractors.py

ENHANCEMENTS APPLIED:
- Bug #1 (v1.4.1): Age category extraction - OPTION C implementation per EMA guidance
  * Primary = authoritative presence (coarse buckets: Adults, Elderly)
  * Secondary = optional refinements (e.g., 65-84, 85+)
  * CRITICAL FIX: Elderly (7 or 8) ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ automatically infers Adults (6) at DB extraction
  * QC check for pediatric conflicts with inclusion criteria
- Bug #2 (v1.2.0): Trial status conversion - properly converts string to integer code
- Bug #3 (v2.0.0): MedDRA extraction - now uses meddraConditionTerms array (FIXED!)
- NEW (v1.3.0): Third-party vendors extraction - extracts all CROs, labs, monitoring orgs
- NEW (v2.0.0): UTF-8 encoding fixed - all corrupted characters replaced
- Previous fixes: Member state extraction, country planning, trial category inference

VERSION: 2.0.0 (Enhanced MedDRA + UTF-8 Fixed)
DATE: 2024-11-17
SOURCE: https://hendrik.codes/post/scraping-the-clinical-trials-information-system
"""

import json
from typing import Dict, Any, List, Tuple
from ctis_config import AGE_CATEGORY_MAP, PHASE_MAP, PRODUCT_ROLE_MAP, BLINDING_MAP, MSC_PUBLIC_STATUS_CODE_MAP
from ctis_utils import get, coerce_int, log

# ===================== Medical Conditions (ENHANCED - Bug #3) =====================

def extract_medical_conditions(js: Dict[str, Any]) -> Tuple[str, str, int, str, str, str, str]:
    """
    Extract medical condition information from trial JSON.
    
    ENHANCED (v2.0.0): Now correctly extracts MedDRA from meddraConditionTerms array
    
    Returns: (primary_condition, conditions_json, is_rare_disease, meddra_code, meddra_label, 
              condition_synonyms, condition_abbreviations)
    """
    partI = (js.get("authorizedApplication", {}).get("authorizedPartI") or
             js.get("authorisedApplication", {}).get("authorisedPartI") or {})
    
    # Try new structure in trialInformation
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    med_cond_data = trial_info.get("medicalCondition", {}) or {}
    part_i_conditions = med_cond_data.get("partIMedicalConditions", []) or []
    
    meddra_code = ""
    meddra_label = ""
    
    if part_i_conditions:
        first_condition = part_i_conditions[0] if isinstance(part_i_conditions[0], dict) else {}
        primary_condition = first_condition.get("medicalCondition", "")
        is_rare_disease = 1 if first_condition.get("isConditionRareDisease") else 0
        
        # Extract MedDRA code and label - try multiple locations
        # Location 1: medicalConditionMeddraClassification (older structure)
        meddra_info = med_cond_data.get("medicalConditionMeddraClassification", {}) or {}
        if isinstance(meddra_info, dict) and meddra_info:
            meddra_code = meddra_info.get("code", "")
            meddra_label = meddra_info.get("term", "") or meddra_info.get("label", "")
        
        # Location 2: meddraConditionTerms (newer structure - array) ÃƒÂ¢Ã¢â‚¬Â Ã‚Â FIXED!
        if not meddra_code:
            meddra_terms = med_cond_data.get("meddraConditionTerms", []) or []
            if isinstance(meddra_terms, list) and meddra_terms:
                first_term = meddra_terms[0] if isinstance(meddra_terms[0], dict) else {}
                meddra_code = first_term.get("classificationCode", "")
                meddra_label = first_term.get("termName", "")
        
        condition_names = [c.get("medicalCondition") for c in part_i_conditions 
                          if isinstance(c, dict) and c.get("medicalCondition")]
        conditions_json = json.dumps(condition_names, ensure_ascii=False)
        
        # Return with empty synonyms and abbreviations (to be populated later)
        return (primary_condition, conditions_json, is_rare_disease, str(meddra_code), meddra_label, "", "")
    
    # Fallback to old structure
    medical_conditions = partI.get("medicalConditions", []) or []
    if not medical_conditions or not isinstance(medical_conditions, list):
        return ("", "[]", 0, "", "", "", "")
    
    primary_condition = ""
    is_rare_disease = 0
    
    if len(medical_conditions) > 0 and isinstance(medical_conditions[0], dict):
        first_condition = medical_conditions[0]
        primary_condition = first_condition.get("medicalCondition", "")
        is_rare_disease = 1 if first_condition.get("isConditionRareDisease") else 0
        
        # Try to get MedDRA from condition
        meddra_info = first_condition.get("meddraClassification", {}) or {}
        if isinstance(meddra_info, dict):
            meddra_code = meddra_info.get("code", "")
            meddra_label = meddra_info.get("term", "") or meddra_info.get("label", "")
    
    condition_names = []
    for cond in medical_conditions:
        if isinstance(cond, dict):
            name = cond.get("medicalCondition")
            if name:
                condition_names.append(name)
    
    conditions_json = json.dumps(condition_names, ensure_ascii=False)
    
    # Return with empty synonyms and abbreviations (to be populated later)
    return (primary_condition, conditions_json, is_rare_disease, str(meddra_code), meddra_label, "", "")


# ===================== Sites =====================

def extract_sites(js: Dict[str, Any], ct: str) -> List[Dict[str, Any]]:
    """Extract ALL sites from multiple locations"""
    sites: List[Dict[str, Any]] = []

    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    parts_ii = auth_app.get("authorizedPartsII") or auth_app.get("authorisedPartsII") or []
    
    if isinstance(parts_ii, list):
        for part_idx, part_ii in enumerate(parts_ii):
            if not isinstance(part_ii, dict):
                continue

            trial_sites = part_ii.get("trialSites", [])
            if isinstance(trial_sites, list):
                for site_idx, site in enumerate(trial_sites):
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


# ===================== People =====================

def extract_people(js: Dict[str, Any], ct: str) -> List[Dict[str, Any]]:
    """Extract ALL people from sponsors and sites"""
    from ctis_utils import extract_email_phone
    
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

    # Site investigators
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


# ===================== Inclusion/Exclusion Criteria =====================

def extract_inclusion_criteria(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract inclusion criteria"""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    eligibility = trial_info.get("eligibilityCriteria", {}) or {}
    
    inclusion = eligibility.get("principalInclusionCriteria", []) or []
    
    result = []
    for criterion in inclusion:
        if isinstance(criterion, dict):
            text = criterion.get("principalInclusionCriteria", "")
            number = criterion.get("number", 0)
            if text:
                result.append({
                    "criterionNumber": number,
                    "criterionText": text
                })
    
    return result


def extract_exclusion_criteria(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract exclusion criteria"""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    eligibility = trial_info.get("eligibilityCriteria", {}) or {}
    
    exclusion = eligibility.get("principalExclusionCriteria", []) or []
    
    result = []
    for criterion in exclusion:
        if isinstance(criterion, dict):
            text = criterion.get("principalExclusionCriteria", "")
            number = criterion.get("number", 0)
            if text:
                result.append({
                    "criterionNumber": number,
                    "criterionText": text
                })
    
    return result


# ===================== Endpoints =====================

def extract_endpoints(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract primary and secondary endpoints"""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    endpoint_data = trial_info.get("endPoint", {}) or {}
    
    result = []
    
    # Primary endpoints
    primary = endpoint_data.get("primaryEndPoints", []) or []
    for ep in primary:
        if isinstance(ep, dict):
            text = ep.get("endPoint", "")
            number = ep.get("number", 0)
            if text:
                result.append({
                    "endpointType": "primary",
                    "endpointNumber": number,
                    "endpointText": text,
                    "timeFrame": ""
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


# ===================== Trial Products =====================

def extract_trial_products(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detailed product information including ATC codes.
    Source: CTIS Full trial information, p.6 (ATC classification)
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    products = partI.get("products", []) or []
    
    result = []
    for prod in products:
        if not isinstance(prod, dict):
            continue
        
        prod_dict_info = prod.get("productDictionaryInfo", {}) or {}
        
        # Extract ATC code
        atc_code = ""
        atc_info = prod.get("atcClassification", {}) or prod_dict_info.get("atcClassification", {})
        if isinstance(atc_info, dict):
            atc_code = atc_info.get("code", "") or atc_info.get("atcCode", "")
        
        result.append({
            "productRole": PRODUCT_ROLE_MAP.get(prod.get("part1MpRoleTypeCode"), "unknown"),
            "productName": prod.get("productName", ""),
            "activeSubstance": prod_dict_info.get("activeSubstanceName", ""),
            "atcCode": str(atc_code),
            "pharmaceuticalForm": prod.get("pharmaceuticalFormDisplay", ""),
            "route": ", ".join(prod.get("routes", [])),
            "maxDailyDose": str(prod.get("maxDailyDoseAmount", "")),
            "maxDailyDoseUnit": prod.get("doseUom", ""),
            "maxTreatmentPeriod": coerce_int(prod.get("maxTreatmentPeriod")),
            "maxTreatmentPeriodUnit": "months",
            "isPaediatric": 1 if prod.get("isPaediatricFormulation") else 0,
            "isOrphanDrug": 1 if prod.get("orphanDrugEdit") else 0,
            "authorizationStatus": "authorized" if prod_dict_info.get("prodAuthStatus") == 2 else "not_authorized",
            "raw": prod
        })
    
    return result


# ===================== Third-Party Vendors (NEW - v1.3.0) =====================

def extract_third_party_vendors(js: Dict[str, Any], ct: str) -> List[Dict[str, Any]]:
    """
    Extract third-party vendors/service providers (CROs, labs, monitoring orgs, etc.)
    
    NEW in v1.3.0: Extracts vendors from sponsors.thirdParties
    - Correctly navigates organisationAddress.organisation.name
    - Extracts duties from sponsorDuties array
    - Includes contact info and address
    
    Returns: List of vendor dicts with name, duties, contact info
    """
    vendors = []
    
    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    part_i = auth_app.get("authorizedPartI") or auth_app.get("authorisedPartI") or {}
    
    # Extract from sponsors ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ thirdParties
    sponsors = part_i.get("sponsors", []) or []
    
    for sponsor_idx, sponsor in enumerate(sponsors):
        if not isinstance(sponsor, dict):
            continue
        
        third_parties = sponsor.get("thirdParties", []) or []
        
        for tp_idx, party in enumerate(third_parties):
            if not isinstance(party, dict):
                continue
            
            # Extract organization name from nested structure
            org_address = party.get("organisationAddress", {})
            if not isinstance(org_address, dict):
                continue
            
            org = org_address.get("organisation", {})
            if not isinstance(org, dict):
                continue
            
            org_name = org.get("name")
            if not org_name:
                continue
            
            # Extract duties from sponsorDuties array
            sponsor_duties = party.get("sponsorDuties", []) or []
            duties_list = []
            for duty in sponsor_duties:
                if isinstance(duty, dict) and "value" in duty:
                    duties_list.append(duty["value"])
            duties_str = ", ".join(duties_list) if duties_list else None
            
            # Extract address
            address_data = org_address.get("address", {})
            if isinstance(address_data, dict):
                address = address_data.get("addressLine1")
                city = address_data.get("city")
                postcode = address_data.get("postcode")
                country = address_data.get("countryName")
            else:
                address = city = postcode = country = None
            
            # Extract contact info
            email = party.get("email")
            phone = party.get("phoneNumber")
            
            # Get business key (organization ID)
            business_key = org.get("businessKey")
            
            vendor = {
                "ctNumber": ct,
                "vendor_id": business_key,
                "vendor_name": org_name,
                "address": address,
                "city": city,
                "postcode": postcode,
                "country": country,
                "phone": phone,
                "email": email,
                "duties": duties_str,
                "path": f"sponsors[{sponsor_idx}].thirdParties[{tp_idx}]",
                "raw": party
            }
            
            vendors.append(vendor)
    
    # Deduplicate by vendor name and country
    seen = set()
    unique_vendors = []
    for v in vendors:
        key = (v["vendor_name"], v["country"])
        if key not in seen:
            seen.add(key)
            unique_vendors.append(v)
    
    return unique_vendors


# ===================== Population (FIXED - v1.4.1 - Option C + UTF-8 Fixed) =====================


def extract_population(js: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract patient population details - age categories and gender
    
    FIXED (v3.0.0): CORRECT handling of CTIS dual age code systems
    
    CTIS uses TWO SEPARATE age code systems (confirmed by EMA documentation):
    
    1. COARSE CODES (ageRanges field) - 4 buckets:
       - Code 1: In utero
       - Code 2: 0-17 years (paediatric)
       - Code 3: 18-64 years (adults)
       - Code 4: 65+ years (elderly)
    
    2. GRANULAR CODES (ageRangeSecondaryIds ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ctAgeRangeCode) - 8 detailed ranges:
       - Code 1: Preterm newborn
       - Code 2: Newborns (0-27 days)
       - Code 3: Infants and toddlers (28 days-23 months)
       - Code 4: Children (2-11 years)
       - Code 5: Adolescents (12-17 years)
       - Code 6: Adults (18-64 years)
       - Code 7: 65-84 years
       - Code 8: 85+ years
    
    CRITICAL: The codes 3 and 4 have DIFFERENT meanings in each system!
    
    Strategy:
    - Extract COARSE codes from ageRanges
    - Extract GRANULAR codes from ageRangeSecondaryIds
    - Map COARSE ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ GRANULAR for consistent storage
    - Store only GRANULAR codes (1-8) in database
    
    Returns: (age_categories_json, gender)
        age_categories_json: JSON array of GRANULAR codes [1-8]
        gender: "male", "female", "both", or ""
    """
    from ctis_utils import get, log
    
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    population = trial_info.get("populationOfTrialSubjects", {}) or {}
    
    # Step 1: Extract COARSE age codes (ageRanges)
    coarse_ranges = population.get("ageRanges", []) or []
    coarse_codes = set()
    for age_range in coarse_ranges:
        if isinstance(age_range, dict):
            code = age_range.get("ageRangeCategoryCode") or age_range.get("ageRangeCategory")
            if code:
                try:
                    coarse_codes.add(int(code))
                except (ValueError, TypeError):
                    pass
    
    # Step 2: Extract GRANULAR age codes (ageRangeSecondaryIds)
    granular_ranges = population.get("ageRangeSecondaryIds", []) or []
    granular_codes = set()
    for age_range in granular_ranges:
        if isinstance(age_range, dict):
            code = age_range.get("ctAgeRangeCode") or age_range.get("ctAgeRange")
            if code:
                try:
                    granular_codes.add(int(code))
                except (ValueError, TypeError):
                    pass
    
    # Step 3: Map COARSE codes to GRANULAR equivalents
    # This is the CORRECT interpretation per EMA guidance
    mapped_granular = set()
    
    for coarse_code in coarse_codes:
        if coarse_code == 1:  # In utero
            mapped_granular.add(1)  # No granular equivalent, keep as is
        
        elif coarse_code == 2:  # 0-17 years (paediatric)
            # Map to all pediatric granular codes
            # If granular codes specify which, use those; otherwise use all
            pediatric_granular = granular_codes & {1, 2, 3, 4, 5}
            if pediatric_granular:
                mapped_granular.update(pediatric_granular)
            else:
                # No granular specified, include common pediatric ranges
                mapped_granular.update({3, 4, 5})  # Infants, Children, Adolescents
        
        elif coarse_code == 3:  # 18-64 years (adults)
            # Map to granular adults code
            mapped_granular.add(6)
        
        elif coarse_code == 4:  # 65+ years (elderly)
            # Check if granular provides refinement (65-84 vs 85+)
            elderly_granular = granular_codes & {7, 8}
            if elderly_granular:
                # Use the specific granular refinements
                mapped_granular.update(elderly_granular)
            else:
                # No granular refinement, include both by default
                mapped_granular.update({7, 8})
    
    # Step 4: Combine with any standalone granular codes
    # (in case granular codes exist without corresponding coarse codes)
    final_codes = mapped_granular | granular_codes
    
    # Step 5: CRITICAL - Auto-infer adults (6) if elderly (7 or 8) present
    # Rationale: People aged 65+ ARE adults by definition
    has_elderly = any(c in {7, 8} for c in final_codes)
    if has_elderly and 6 not in final_codes:
        final_codes.add(6)
        log(f"[AGE] Auto-inferred Adults (6) from Elderly codes {final_codes & {7, 8}}", "INFO")
    
    # Step 6: Validation check - warn if no age codes extracted
    if not final_codes:
        log(f"[AGE] WARNING: No age codes extracted. Coarse={coarse_codes}, Granular={granular_codes}", "WARN")
        
        # Ultimate fallback: try to extract from inclusion criteria
        try:
            trial_protocol = partI.get("trialProtocol", {})
            inclusion = trial_protocol.get("subjectInclusionCriteria", []) or []
            inclusion_text = " ".join([
                c.get("criterionText", "").lower() 
                for c in inclusion[:5] if isinstance(c, dict)
            ])
            
            import re
            # Look for age patterns
            if re.search(r'(?:ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥|>=|at least|minimum age)\s*18', inclusion_text):
                final_codes.add(6)  # Adults
                if re.search(r'(?:ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥|>=)\s*65', inclusion_text) or 'elderly' in inclusion_text:
                    final_codes.update({7, 8})  # Elderly
                log(f"[AGE] Extracted from inclusion text: {final_codes}", "INFO")
            elif 'pediatric' in inclusion_text or 'paediatric' in inclusion_text or 'children' in inclusion_text:
                final_codes.update({3, 4, 5})
                log(f"[AGE] Extracted pediatric from inclusion text", "INFO")
        except Exception as e:
            log(f"[AGE] Could not extract from inclusion text: {e}", "WARN")
    
    # Step 7: Log the final result for debugging
    if final_codes:
        age_names = {
            1: "Preterm", 2: "0-27d", 3: "28d-23mo", 4: "2-11y", 
            5: "12-17y", 6: "18-64y", 7: "65-84y", 8: "85+y"
        }
        age_labels = [age_names.get(c, f"Code{c}") for c in sorted(final_codes)]
        log(f"[AGE] Final codes {sorted(final_codes)}: {', '.join(age_labels)}", "INFO")
    
    # Convert to JSON string (sorted for consistency)
    age_categories_json = json.dumps(sorted(list(final_codes)))
    
    # Extract gender (unchanged from original)
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



# ===================== Trial Design =====================

def extract_trial_design(js: Dict[str, Any]) -> Tuple[int, str]:
    """Extract trial design details"""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    protocol_info = trial_details.get("protocolInformation", {}) or {}
    study_design = protocol_info.get("studyDesign", {}) or {}
    period_details = study_design.get("periodDetails", []) or []
    
    is_randomised = 0
    blinding_type = ""
    
    if period_details and len(period_details) > 0:
        first_period = period_details[0]
        
        # Blinding method
        blinding_code = first_period.get("blindingMethodCode")
        blinding_type = BLINDING_MAP.get(str(blinding_code), "")
        
        # Allocation method
        allocation_code = first_period.get("allocationMethod")
        if str(allocation_code) == "1":
            is_randomised = 1
    
    return (is_randomised, blinding_type)


# ===================== Objectives =====================

def extract_objectives(js: Dict[str, Any]) -> Tuple[str, str, int, int]:
    """Extract objectives"""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    trial_objective = trial_info.get("trialObjective", {}) or {}
    
    main_obj = trial_objective.get("mainObjective", "")
    
    # Trial scopes
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


# ===================== Trial Duration =====================

def extract_trial_duration(js: Dict[str, Any]) -> Tuple[str, str]:
    """Extract trial duration dates"""
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    duration = trial_info.get("trialDuration", {}) or {}
    
    start_date = duration.get("estimatedRecruitmentStartDate", "")
    end_date = duration.get("estimatedEndDate", "")
    
    return (start_date, end_date)


# ===================== Trial Category =====================

def extract_trial_category(js: Dict[str, Any], trial_phase: str) -> str:
    """
    Extract trial category with fallback to phase-based inference.
    
    Infers category from phase if not explicitly provided.
    
    Categories:
    - Category 1: Phase I trials
    - Category 2: Phase II, Phase III, Integrated I/II
    - Category 3: Phase IV, Low Intervention
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    category_data = trial_details.get("trialCategory", {}) or {}
    
    # Try explicit category first
    trial_category = category_data.get("trialCategoryCode", "") or category_data.get("category", "")
    
    # Fallback: infer from phase if category not provided
    if not trial_category and trial_phase:
        phase_lower = trial_phase.lower()
        
        # Category 1: Phase I (not integrated)
        if "phase i" in phase_lower and "phase ii" not in phase_lower and "phase iii" not in phase_lower:
            trial_category = "1"
            log(f"Trial category inferred from phase '{trial_phase}': Category 1", "INFO")
        
        # Category 2: Phase II, Phase III, or Integrated I/II, II/III
        elif any(p in phase_lower for p in ["phase ii", "phase iii", "phase i and phase ii", "integrated"]):
            trial_category = "2"
            log(f"Trial category inferred from phase '{trial_phase}': Category 2", "INFO")
        
        # Category 3: Phase IV or Low Intervention
        elif "phase iv" in phase_lower or "low intervention" in phase_lower:
            trial_category = "3"
            log(f"Trial category inferred from phase '{trial_phase}': Category 3", "INFO")
    
    return str(trial_category) if trial_category else ""


# ===================== Disclosure Timing =====================

def calculate_disclosure_timing(js: Dict[str, Any], trial_category: str, 
                                trial_phase: str, estimated_end_date: str) -> Tuple[str, str, int]:
    """
    Calculate expected disclosure date for dosage fields and whether they're visible now.
    
    Per Annex I, Table I: Category 1 trials (and integrated phase I/II) disclose
    max duration/dose fields 30 months after EU/EEA End of Trial.
    
    Source: Annex I, Table I (max duration/dose fields disclose 30 months after EU/EEA EoT)
    
    Returns: (trial_category, expected_disclosure_date, dosage_visible_now)
    """
    from ctis_utils import parse_ts
    from datetime import timedelta
    
    # Use provided category or extract it
    if not trial_category:
        trial_category = extract_trial_category(js, trial_phase)
    
    # Check if this is Category 1 or integrated phase I/II
    is_category_1 = (str(trial_category) == "1" or 
                    trial_phase in ["Phase I and Phase II (Integrated)", "Human Pharmacology (Phase I)"])
    
    if not is_category_1:
        # Category 2 and 3: dosage visible immediately upon decision
        return (str(trial_category), "", 1)
    
    # Category 1: dosage visible 30 months after EU/EEA End of Trial
    if not estimated_end_date:
        # End date unknown, can't calculate disclosure date
        return (str(trial_category), "", 0)
    
    # Parse end date and add 30 months
    end_dt = parse_ts(estimated_end_date)
    if not end_dt:
        return (str(trial_category), "", 0)
    
    # Add 30 months (approximate as 30 * 30.5 days = 915 days)
    disclosure_dt = end_dt + timedelta(days=915)
    disclosure_date = disclosure_dt.strftime("%Y-%m-%d")
    
    # Check if disclosure date has passed
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    dosage_visible = 1 if disclosure_dt <= now else 0
    
    return (str(trial_category), disclosure_date, dosage_visible)


# ===================== Member State Status =====================

def extract_ms_status(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract member state status information from trial JSON.
    
    Extracts per-country status block showing:
    - Current status (Authorised, recruiting; Ongoing, recruiting; Ended, etc.)
    - Decision date, start date, recruitment dates
    - Temporary halt/restart dates
    - End date and early termination info
    - Last update
    
    Source: CTIS Summary section, statuses & dates list (p.4-5 of documentation)
    """
    from ctis_config import STATUS_NORMALIZATION
    
    def normalize_status(status: str) -> str:
        """Normalize status string to consistent format"""
        if not status:
            return ""
        status_lower = status.strip().lower()
        return STATUS_NORMALIZATION.get(status_lower, status.strip())
    
    result = []
    
    # Member states are in authorizedApplication.memberStatesConcerned
    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    member_states = auth_app.get("memberStatesConcerned") or []
    
    if not isinstance(member_states, list):
        log("Member states data is not a list or is missing", "WARN")
        return result
    
    for ms in member_states:
        if not isinstance(ms, dict):
            continue
        
        # Extract member state name
        member_state = ms.get("mscName") or ms.get("countryName") or ""
        
        if not member_state:
            continue
        
        public_status_code = ms.get("mscPublicStatusCode")
        status_text = MSC_PUBLIC_STATUS_CODE_MAP.get(
            public_status_code, 
            f"Unknown status code {public_status_code}"
        )

        # Basic info from memberStatesConcerned
        ms_record = {
            "member_state": member_state,
            "status": "",  # Will be enriched from Parts II
            "decision_date": ms.get("firstDecisionDate") or ms.get("decisionDate") or "",
            "start_date": "",
            "recruitment_start": "",
            "recruitment_end": "",
            "temporary_halt": "",
            "restart_date": "",
            "end_date": "",
            "early_termination_date": "",
            "early_termination_reason": "",
            "last_update": ms.get("lastDecisionDate") or ms.get("lastUpdate") or ms.get("lastUpdated") or ""
        }
        
        result.append(ms_record)
    
    # Enrich with detailed status from Parts II ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ mscInfo
    parts_ii = auth_app.get("authorizedPartsII") or auth_app.get("authorisedPartsII") or []
    
    if isinstance(parts_ii, list):
        for part_ii in parts_ii:
            if not isinstance(part_ii, dict):
                continue
            
            msc_info = part_ii.get("mscInfo", {}) or {}
            if not isinstance(msc_info, dict):
                continue
            
            country = msc_info.get("mscName") or msc_info.get("countryName") or ""
            if not country:
                continue
            
            # Find matching record and enrich
            for record in result:
                if record["member_state"] == country:
                    # Add trial periods (start/end dates)
                    trial_periods = msc_info.get("trialPeriod", []) or []
                    if trial_periods and isinstance(trial_periods, list) and len(trial_periods) > 0:
                        first_period = trial_periods[0]
                        if isinstance(first_period, dict):
                            record["start_date"] = first_period.get("fromDate") or ""
                            record["end_date"] = first_period.get("toDate") or ""
                    
                    # Add recruitment periods
                    recruitment_periods = msc_info.get("trialRecruitmentPeriod", []) or []
                    if recruitment_periods and isinstance(recruitment_periods, list) and len(recruitment_periods) > 0:
                        first_rec = recruitment_periods[0]
                        if isinstance(first_rec, dict):
                            record["recruitment_start"] = first_rec.get("fromDate") or ""
                            record["recruitment_end"] = first_rec.get("toDate") or ""
                    
                    # Update decision date if more specific
                    if msc_info.get("firstDecisionDate") and not record["decision_date"]:
                        record["decision_date"] = msc_info.get("firstDecisionDate")
                    
                    # Check for early termination in status history
                    status_history = msc_info.get("clinicalTrialStatusHistory", []) or []
                    for hist in status_history:
                        if isinstance(hist, dict):
                            status = hist.get("trialStatus", "").lower()
                            if "terminat" in status or "ended" in status:
                                record["early_termination_date"] = hist.get("trialStatusDate") or ""
                    
                    break
    
    if not result:
        log("No member state status records extracted - check if trial is authorized", "WARN")
    
    return result


# ===================== Country Planning & Site Contacts =====================

def extract_country_planning(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract planned participant numbers per country.
    
    Uses recruitmentSubjectCount field.
    
    This extracts the "Planned number of participants" field shown when
    expanding each country in the Locations section.
    
    Source: CTIS Locations & contact points, p.1-2
    """
    result = []
    seen_countries = set()
    
    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    parts_ii = auth_app.get("authorizedPartsII") or auth_app.get("authorisedPartsII") or []
    
    if isinstance(parts_ii, list):
        for part_ii in parts_ii:
            if not isinstance(part_ii, dict):
                continue
            
            # Get country from mscInfo (more reliable than part II level)
            msc_info = part_ii.get("mscInfo", {}) or {}
            country = msc_info.get("mscName") or msc_info.get("countryName") or ""
            
            # Fallback to part II level
            if not country:
                country = part_ii.get("countryName") or part_ii.get("mscName") or ""
            
            # Use recruitmentSubjectCount (correct field)
            planned = part_ii.get("recruitmentSubjectCount")
            
            # Fallback to old field names if needed
            if planned is None:
                planned = (part_ii.get("plannedNumberOfSubjects") or 
                          part_ii.get("subjectsNumber") or
                          part_ii.get("numberOfSubjects"))
            
            if country and planned is not None and country not in seen_countries:
                result.append({
                    "country": country,
                    "planned_participants": coerce_int(planned)
                })
                seen_countries.add(country)
    
    # Also try from memberStatesConcerned if it has participant info
    member_states = auth_app.get("memberStatesConcerned") or []
    if isinstance(member_states, list):
        for ms in member_states:
            if not isinstance(ms, dict):
                continue
            
            country = ms.get("mscName") or ms.get("countryName") or ""
            planned = (ms.get("plannedNumberOfSubjects") or 
                      ms.get("numberOfSubjects") or
                      ms.get("subjectsNumber"))
            
            if country and planned is not None and country not in seen_countries:
                result.append({
                    "country": country,
                    "planned_participants": coerce_int(planned)
                })
                seen_countries.add(country)
    
    if not result:
        log("No country planning data extracted - may not be available yet", "WARN")
    
    return result


def extract_site_contacts(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract site contact information (PI details) from trial JSON.
    
    This extracts the main healthcare professional contact for each site,
    including PI name, email, and phone. This data appears when you expand
    a country in the "Locations and contact points" tab.
    
    Source: CTIS Locations & contact points, p.1-2
    """
    result = []
    
    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    parts_ii = auth_app.get("authorizedPartsII") or auth_app.get("authorisedPartsII") or []
    
    if not isinstance(parts_ii, list):
        return result
    
    for part_ii in parts_ii:
        if not isinstance(part_ii, dict):
            continue
        
        # Get country from mscInfo
        msc_info = part_ii.get("mscInfo", {}) or {}
        country = msc_info.get("mscName") or msc_info.get("countryName") or ""
        
        # Fallback
        if not country:
            country = part_ii.get("countryName") or part_ii.get("mscName") or ""
        
        trial_sites = part_ii.get("trialSites", [])
        if not isinstance(trial_sites, list):
            continue
        
        for site in trial_sites:
            if not isinstance(site, dict):
                continue
            
            # Extract organization and address info
            org_addr_info = site.get("organisationAddressInfo", {}) or {}
            org_obj = org_addr_info.get("organisation", {}) or {}
            org_name = org_obj.get("name") if isinstance(org_obj, dict) else None
            
            addr_obj = org_addr_info.get("address", {}) or {}
            if isinstance(addr_obj, dict):
                address = addr_obj.get("addressLine1") or addr_obj.get("street")
                city = addr_obj.get("city") or addr_obj.get("town")
                postal = addr_obj.get("postalCode") or addr_obj.get("postcode")
            else:
                address = city = postal = None
            
            # Extract site name
            site_name = org_name
            dept = site.get("departmentName")
            if dept:
                site_name = f"{org_name} - {dept}" if org_name else dept
            
            # Extract PI contact information
            person_info = site.get("personInfo", {}) or {}
            if isinstance(person_info, dict) and person_info:
                first_name = person_info.get("firstName")
                last_name = person_info.get("lastName")
                pi_name = f"{(first_name or '').strip()} {(last_name or '').strip()}".strip() or None
                pi_email = person_info.get("email") or org_addr_info.get("email")
                pi_phone = person_info.get("telephone") or person_info.get("phone") or org_addr_info.get("phone")
                
                # Only add if we have at least name or email
                if pi_name or pi_email:
                    result.append({
                        "country": country,
                        "org_name": org_name,
                        "site_name": site_name,
                        "address": address,
                        "city": city,
                        "postal_code": postal,
                        "pi_name": pi_name,
                        "pi_email": pi_email,
                        "pi_phone": pi_phone
                    })
    
    # Deduplicate
    def _norm(v):
        return v.strip() if isinstance(v, str) else (v if v is not None else "")
    
    seen = set()
    unique = []
    for contact in result:
        key = (
            _norm(contact.get("country")),
            _norm(contact.get("org_name")),
            _norm(contact.get("site_name")),
            _norm(contact.get("pi_name")),
            _norm(contact.get("pi_email"))
        )
        if key not in seen:
            seen.add(key)
            unique.append(contact)
    
    return unique


# ===================== Trial Fields (Main) (FIXED - Bug #2) =====================

def extract_trial_fields(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all main trial fields
    
    FIXED (v1.2.0): Trial status now properly converts string to integer
    ENHANCED (v2.0.0): Therapeutic areas with multiple fallback paths
    """
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

    # Therapeutic areas (ENHANCED - v2.0.0) - check multiple locations
    # Primary location
    ta_list = partI.get("therapeuticAreas") or []
    
    # Fallback: trialDetails
    if not ta_list or not isinstance(ta_list, list):
        trial_details_obj = partI.get("trialDetails", {})
        ta_list = trial_details_obj.get("therapeuticAreas") or []
    
    # Fallback: trialInformation
    if not ta_list or not isinstance(ta_list, list):
        trial_info = get(partI, "trialDetails", "trialInformation") or {}
        ta_list = trial_info.get("therapeuticAreas") or []
    
    # Extract names
    ta_names = []
    if isinstance(ta_list, list):
        for t in ta_list:
            if isinstance(t, dict):
                # Try "name" field first, then "therapeuticArea"
                name = t.get("name") or t.get("therapeuticArea") or ""
                if name:
                    ta_names.append(name)
    
    ta = "; ".join(ta_names) if ta_names else ""

    # Medical conditions (ENHANCED - includes MedDRA from meddraConditionTerms)
    medical_condition, medical_conditions_json, is_rare, meddra_code, meddra_label, condition_synonyms, condition_abbreviations = extract_medical_conditions(js)

    # Countries
    countries = ""
    # Use the fixed extraction path
    auth_app = js.get("authorizedApplication") or js.get("authorisedApplication") or {}
    root_msc = auth_app.get("memberStatesConcerned") or []
    
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
    phase = PHASE_MAP.get(str(phase_code), str(phase_code) if phase_code else "")

    # Status - FIXED: Now properly converts string to integer
    ct_status_value = root.get("ctStatus")
    
    # Define status mapping
    STATUS_MAP = {
        "Authorised": 2,
        "Not Authorised": 1,
        "Withdrawn": 3,
        "Suspended": 4,
        "Ended": 5,
    }
    
    # Convert string status to integer if needed
    if isinstance(ct_status_value, str):
        ct_status_num = STATUS_MAP.get(ct_status_value)
        if ct_status_num is None:
            log(f"Unknown status string: '{ct_status_value}', attempting integer conversion", "WARN")
            ct_status_num = coerce_int(ct_status_value)
    else:
        ct_status_num = coerce_int(ct_status_value)
    
    ct_public_status = root.get("ctPublicStatusCode") or ""

    # Timestamps
    from ctis_utils import ts_is_newer
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
    
    # Calculate pediatric and adult flags
    from ctis_utils import is_pediatric_trial, is_adult_trial
    age_codes = json.loads(age_categories) if age_categories else []
    is_pediatric = 1 if is_pediatric_trial(age_codes) else 0
    is_adult = 1 if is_adult_trial(age_codes) else 0
    
    # Extract category first, then calculate disclosure timing
    trial_category = extract_trial_category(js, phase)
    trial_category, expected_disclosure, dosage_visible = calculate_disclosure_timing(
        js, trial_category, phase, end_date
    )
    
    # Enhanced fields (v1.3.0)
    identifiers = extract_trial_identifiers(js)
    pip_info = extract_pip_information(js)
    transition_info = extract_transition_trial_info(js)
    global_end = extract_global_end_date(js)
    design_details = extract_trial_design_details(js)

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
        "conditionMeddraCode": meddra_code,
        "conditionMeddraLabel": meddra_label,
        "conditionSynonyms": condition_synonyms,
        "conditionAbbreviations": condition_abbreviations,
        "countries": countries,
        "decisionDate": decision,
        "publishDate": publish,
        "lastUpdated": freshest,
        
        # Feasibility fields
        "ageCategories": age_categories,
        "isPediatric": is_pediatric,
        "isAdult": is_adult,
        "gender": gender,
        "isRandomised": is_randomised,
        "blindingType": blinding_type,
        "trialScope": scope_json,
        "mainObjective": main_obj,
        "primaryEndpointsCount": primary_count,
        "secondaryEndpointsCount": secondary_count,
        "estimatedRecruitmentStartDate": start_date,
        "estimatedEndDate": end_date,
        
        # Disclosure timing
        "trialCategory": trial_category,
        "expectedDosageDisclosureDate": expected_disclosure,
        "dosageVisibleNow": dosage_visible,
        
        # Enhanced fields (v1.3.0)
        "who_utn": identifiers.get("who_utn"),
        "nct_number": identifiers.get("nct_number"),
        "isrctn_number": identifiers.get("isrctn_number"),
        "additional_registry_ids": identifiers.get("additional_registry_ids"),
        "pip_number": pip_info.get("pip_number"),
        "pip_decision_date": pip_info.get("pip_decision_date"),
        "is_transition_trial": transition_info.get("is_transition_trial", 0),
        "eudract_number": transition_info.get("eudract_number"),
        "global_end_date": global_end,
        "allocation_method": design_details.get("allocation_method"),
        "number_of_arms": design_details.get("number_of_arms"),
    }


# ===================== Enhanced Extraction Functions (v1.3.0) =====================

def extract_trial_identifiers(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract additional trial registry identifiers (WHO UTN, NCT, ISRCTN, etc.)
    
    FIXED VERSION - Now correctly extracts from secondaryIdentifyingNumbers
    
    JSON Path (from CTIS API):
    authorizedPartI ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ trialDetails ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ clinicalTrialIdentifiers ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ secondaryIdentifyingNumbers
    
    Returns dict with:
        - who_utn: WHO Universal Trial Number
        - nct_number: ClinicalTrials.gov identifier (NCT number)
        - isrctn_number: ISRCTN registry number
        - additional_registry_ids: JSON string with other registries
    """
    # Navigate to the identifiers section
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    identifiers = trial_details.get("clinicalTrialIdentifiers", {}) or {}
    
    # NEW: Extract from secondaryIdentifyingNumbers (correct path based on actual JSON)
    secondary_ids = identifiers.get("secondaryIdentifyingNumbers", {}) or {}
    
    # Extract WHO UTN from secondaryIdentifyingNumbers
    who_utn = None
    who_utn_obj = secondary_ids.get("whoUtn") or secondary_ids.get("whoUniversalTrialNumber")
    if isinstance(who_utn_obj, dict):
        who_utn = who_utn_obj.get("number")
    elif who_utn_obj:
        who_utn = who_utn_obj
    
    # Fallback to old paths for backwards compatibility
    if not who_utn:
        who_utn = (identifiers.get("whoUtn") or 
                   identifiers.get("whoUniversalTrialNumber") or 
                   identifiers.get("whoNumber"))
    
    # Extract NCT number from secondaryIdentifyingNumbers - THIS IS THE KEY FIX
    nct_number = None
    nct_obj = secondary_ids.get("nctNumber")
    if isinstance(nct_obj, dict):
        nct_number = nct_obj.get("number")
    elif nct_obj:
        nct_number = nct_obj
    
    # Fallback to old paths for backwards compatibility
    if not nct_number:
        nct_number = (identifiers.get("nctNumber") or 
                      identifiers.get("clinicalTrialsGovId") or
                      identifiers.get("clinicalTrialsGovIdentifier"))
    
    # Extract ISRCTN from secondaryIdentifyingNumbers
    isrctn_number = None
    isrctn_obj = secondary_ids.get("isrctnNumber") or secondary_ids.get("isrctn")
    if isinstance(isrctn_obj, dict):
        isrctn_number = isrctn_obj.get("number")
    elif isrctn_obj:
        isrctn_number = isrctn_obj
    
    # Fallback to old paths for backwards compatibility
    if not isrctn_number:
        isrctn_number = (identifiers.get("isrctnNumber") or 
                         identifiers.get("isrctn"))
    
    # Collect any additional registry identifiers
    additional_registries = []
    
    # Check in secondaryIdentifyingNumbers first
    additional_regs = secondary_ids.get("additionalRegistries", []) or []
    
    # Fallback to old location
    if not additional_regs:
        additional_regs = identifiers.get("additionalRegistries", []) or []
    
    for reg in additional_regs:
        if isinstance(reg, dict):
            # Try different possible field names
            reg_name = (reg.get("registryName") or 
                       reg.get("name") or 
                       reg.get("registry"))
            reg_id = (reg.get("registryId") or 
                     reg.get("identifier") or 
                     reg.get("number") or
                     reg.get("id"))
            
            if reg_name and reg_id:
                additional_registries.append({
                    "registry": reg_name,
                    "identifier": reg_id
                })
    
    additional_registry_ids = json.dumps(additional_registries) if additional_registries else None
    
    return {
        "who_utn": who_utn,
        "nct_number": nct_number,
        "isrctn_number": isrctn_number,
        "additional_registry_ids": additional_registry_ids
    }


def extract_pip_information(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Paediatric Investigation Plan (PIP) information
    
    Returns dict with:
        - pip_number: EMA PIP decision number
        - pip_decision_date: Date of PIP decision
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    
    # PIP information may be in different locations
    pip_info = (trial_details.get("paediatricInvestigationPlan", {}) or 
                trial_details.get("pipInformation", {}) or {})
    
    pip_number = (pip_info.get("pipNumber") or 
                  pip_info.get("pipDecisionNumber") or
                  pip_info.get("emaDecisionNumber"))
    
    pip_decision_date = (pip_info.get("pipDecisionDate") or 
                        pip_info.get("decisionDate"))
    
    # Log if paediatric trial but no PIP found
    age_categories = js.get("ageCategories", "[]")
    is_paediatric = any(cat in str(age_categories) for cat in ["1", "2", "3", "4", "5"])
    
    if is_paediatric and not pip_number:
        ct = js.get("ctNumber", "unknown")
        log(f"Paediatric trial {ct} but no PIP number found", "WARN")
    
    return {
        "pip_number": pip_number,
        "pip_decision_date": pip_decision_date
    }


def extract_transition_trial_info(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract transition trial information (trials moving from old Directive to new Regulation)
    
    Returns dict with:
        - is_transition_trial: Boolean (0/1)
        - eudract_number: EudraCT number from old system
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    transition_info = trial_details.get("transitionTrial", {}) or {}
    
    is_transition = bool(transition_info.get("isTransitionTrial") or 
                        js.get("isTransitionTrial"))
    
    eudract_number = (transition_info.get("eudractNumber") or 
                     trial_details.get("eudractNumber") or
                     js.get("eudractNumber"))
    
    return {
        "is_transition_trial": 1 if is_transition else 0,
        "eudract_number": eudract_number
    }


def extract_global_end_date(js: Dict[str, Any]) -> str:
    """
    Extract global end of trial date (worldwide, not just EU/EEA)
    
    Returns: ISO date string or empty string
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    duration = trial_info.get("trialDuration", {}) or {}
    
    global_end = (duration.get("estimatedGlobalEndDate") or 
                 duration.get("globalEndDate") or
                 duration.get("estimatedGlobalTrialEndDate"))
    
    return global_end or ""


def extract_trial_design_details(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract detailed trial design information beyond basic randomization/blinding
    
    Returns dict with:
        - allocation_method: How participants are allocated to arms
        - number_of_arms: Number of treatment arms
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    protocol_info = trial_details.get("protocolInformation", {}) or {}
    study_design = protocol_info.get("studyDesign", {}) or {}
    
    # Allocation method
    allocation_code = study_design.get("allocationMethod")
    allocation_map = {
        "1": "randomised",
        "2": "non-randomised",
        "3": "not applicable"
    }
    allocation_method = allocation_map.get(str(allocation_code)) if allocation_code else None
    
    # Number of arms - may need to count from study arms/groups
    arms = study_design.get("studyArms", []) or study_design.get("treatmentArms", []) or []
    number_of_arms = len(arms) if isinstance(arms, list) else None
    
    # If not in arms, try to get from period details
    if not number_of_arms:
        period_details = study_design.get("periodDetails", []) or []
        if period_details and isinstance(period_details, list):
            # Look for number of groups in first period
            first_period = period_details[0] if period_details else {}
            number_of_arms = first_period.get("numberOfGroups") or first_period.get("numberOfArms")
    
    return {
        "allocation_method": allocation_method,
        "number_of_arms": number_of_arms
    }


def extract_funding_sources(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract sources of monetary and material support
    
    Returns list of funding source dicts
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    trial_info = trial_details.get("trialInformation", {})
    
    funding = []
    support_sources = trial_info.get("sourceOfMonetarySupport", []) or []
    
    for idx, source in enumerate(support_sources):
        if not isinstance(source, dict):
            continue
        
        org_type = source.get("organisationType") or source.get("sponsorType")
        
        # Map organization types
        type_map = {
            "1": "commercial",
            "2": "non-commercial",
            "3": "mixed"
        }
        
        funding_type = type_map.get(str(org_type), str(org_type) if org_type else None)
        
        funding.append({
            "funding_source_type": funding_type,
            "funding_source_name": source.get("organisationName") or source.get("name"),
            "funding_source_country": source.get("country") or source.get("countryName"),
            "is_primary_funder": 1 if idx == 0 else 0  # First is typically primary
        })
    
    return funding


def extract_scientific_advice(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract scientific advice information
    
    Returns list of advice records
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    advice_list = trial_details.get("scientificAdvice", []) or []
    
    advice_records = []
    
    for advice in advice_list:
        if not isinstance(advice, dict):
            continue
        
        # Authority providing advice
        authority = (advice.get("competentAuthority") or 
                    advice.get("authority") or
                    advice.get("authorityName"))
        
        # Type of advice
        advice_type = (advice.get("adviceType") or 
                      advice.get("procedureType") or
                      "scientific advice")
        
        # Date
        advice_date = advice.get("adviceDate") or advice.get("date")
        
        if authority:
            advice_records.append({
                "advice_authority": authority,
                "advice_type": advice_type,
                "advice_date": advice_date
            })
    
    return advice_records


def extract_trial_relationships(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract related/associated clinical trials
    
    Returns list of relationships
    """
    partI = (get(js, "authorizedApplication", "authorizedPartI") or
             get(js, "authorisedApplication", "authorisedPartI") or {})
    
    trial_details = partI.get("trialDetails", {})
    associated = trial_details.get("associatedClinicalTrials", []) or []
    
    relationships = []
    
    for assoc in associated:
        if not isinstance(assoc, dict):
            continue
        
        related_ct = (assoc.get("ctNumber") or 
                     assoc.get("associatedCtNumber") or
                     assoc.get("relatedTrialId"))
        
        rel_type = (assoc.get("relationshipType") or 
                   assoc.get("relationType") or
                   "associated")
        
        description = assoc.get("description") or assoc.get("relationshipDescription")
        
        if related_ct:
            relationships.append({
                "related_ctNumber": related_ct,
                "relationship_type": rel_type,
                "description": description
            })
    
    return relationships