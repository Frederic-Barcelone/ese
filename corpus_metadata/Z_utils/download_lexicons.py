# corpus_metadata/Z_utils/download_lexicons.py
"""
Download and convert public lexicons for the extraction pipeline.

Lexicons:
1. Meta-Inventory (clinical abbreviations) - 104K+ abbreviations
2. MONDO (disease ontology) - Unified disease mappings
3. ChEMBL (drug database) - Open drug data

Usage:
    python -m corpus_metadata.Z_utils.download_lexicons
"""

import csv
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


OUTPUT_DIR = Path("/Users/frederictetard/Projects/ese/ouput_datasources")


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def generate_abbrev_regex(sf: str) -> str:
    """Generate regex pattern for abbreviation matching."""
    # Escape special characters
    escaped = re.escape(sf)
    # Allow optional separators between letters
    pattern_parts = []
    for char in sf:
        if char.isalnum():
            pattern_parts.append(re.escape(char))
        else:
            pattern_parts.append(re.escape(char))

    # Simple word boundary pattern
    pattern = r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])"
    return pattern


# =============================================================================
# 1. META-INVENTORY (Clinical Abbreviations)
# =============================================================================
def download_meta_inventory():
    """
    Download and convert Meta-Inventory clinical abbreviations.

    Source: https://github.com/lisavirginia/clinical-abbreviations
    Paper: https://www.nature.com/articles/s41597-021-00929-4

    Format: CSV with columns: sf, lf, source_inventory
    Output: JSON in pipeline format
    """
    print("\n" + "="*60)
    print("META-INVENTORY: Clinical Abbreviations")
    print("="*60)

    # Raw GitHub URL for the meta-inventory CSV (pipe-delimited)
    url = "https://raw.githubusercontent.com/lisavirginia/clinical-abbreviations/master/metainventory/Metainventory_Version1.0.0.csv"
    csv_path = OUTPUT_DIR / "meta_inventory_raw.csv"
    output_path = OUTPUT_DIR / "2025_meta_inventory_abbreviations.json"

    # Download CSV
    if not download_file(url, csv_path):
        print("Failed to download Meta-Inventory")
        return

    # Parse and convert (pipe-delimited CSV)
    abbrevs: Dict[str, Dict[str, Any]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)  # GroupID, RecordID, SF, SFUI, NormSF, LF, LFUI, NormLF, Source, Modified
        print(f"Columns: {header}")

        for row in reader:
            if len(row) < 9:
                continue
            sf = row[2].strip()   # SF column
            lf = row[5].strip()   # LF column
            source = row[8].strip() if len(row) > 8 else "meta-inventory"

            if not sf or not lf:
                continue

            # Skip very short or very long abbreviations
            if len(sf) < 2 or len(sf) > 15:
                continue

            # Skip if SF contains only lowercase (likely not an abbreviation)
            if sf.islower() and len(sf) > 3:
                continue

            # Use uppercase key for consistency
            key = sf.upper()

            if key not in abbrevs:
                abbrevs[key] = {
                    "canonical_expansion": lf,
                    "case_insensitive": True,
                    "regex": generate_abbrev_regex(sf),
                    "sources": [source],
                    "expansions": []
                }

            # Add expansion if different
            existing_exps = [e["expansion"].lower() for e in abbrevs[key]["expansions"]]
            if lf.lower() not in existing_exps:
                abbrevs[key]["expansions"].append({
                    "expansion": lf,
                    "confidence": 1.0,
                    "source": source
                })

            # Track sources
            if source not in abbrevs[key]["sources"]:
                abbrevs[key]["sources"].append(source)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(abbrevs, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(abbrevs)} abbreviations to: {output_path}")

    # Clean up raw file
    csv_path.unlink()

    return output_path


# =============================================================================
# 2. MONDO (Disease Ontology)
# =============================================================================
def download_mondo():
    """
    Download and convert MONDO disease ontology.

    Source: https://mondo.monarchinitiative.org/
    Format: OBO or JSON-LD
    Output: JSON in pipeline format
    """
    print("\n" + "="*60)
    print("MONDO: Disease Ontology")
    print("="*60)

    # MONDO provides JSON releases
    url = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.json"
    json_path = OUTPUT_DIR / "mondo_raw.json"
    output_path = OUTPUT_DIR / "2025_mondo_diseases.json"

    # Download JSON
    if not download_file(url, json_path):
        # Try alternative OBO format
        print("Trying OBO format...")
        url = "https://purl.obolibrary.org/obo/mondo.json"
        if not download_file(url, json_path):
            print("Failed to download MONDO")
            return

    # Parse MONDO JSON-LD format
    print("Parsing MONDO ontology...")
    with open(json_path, "r", encoding="utf-8") as f:
        mondo_data = json.load(f)

    diseases: List[Dict[str, Any]] = []

    # MONDO JSON-LD structure has "graphs" with "nodes"
    graphs = mondo_data.get("graphs", [])
    for graph in graphs:
        nodes = graph.get("nodes", [])
        for node in nodes:
            # Get node ID and label
            node_id = node.get("id", "")
            label = node.get("lbl", "")

            if not label or not node_id:
                continue

            # Filter to MONDO disease terms
            if "MONDO" not in node_id:
                continue

            # Extract MONDO ID
            mondo_id = node_id.split("/")[-1] if "/" in node_id else node_id

            # Get synonyms
            synonyms = []
            meta = node.get("meta", {})
            for syn in meta.get("synonyms", []):
                syn_val = syn.get("val", "")
                if syn_val and syn_val != label:
                    synonyms.append(syn_val)

            # Get cross-references (mappings to other ontologies)
            xrefs = []
            for xref in meta.get("xrefs", []):
                xref_val = xref.get("val", "")
                if xref_val:
                    # Parse xref format like "OMIM:123456"
                    if ":" in xref_val:
                        source, xid = xref_val.split(":", 1)
                        xrefs.append({"source": source, "id": xref_val})

            # Build disease entry
            sources = [{"source": "MONDO", "id": mondo_id}]
            sources.extend(xrefs)

            disease_entry = {
                "label": label,
                "sources": sources,
                "synonyms": synonyms[:10]  # Limit synonyms
            }
            diseases.append(disease_entry)

            # Also add synonyms as separate entries
            for syn in synonyms[:5]:
                if len(syn) >= 3:
                    diseases.append({
                        "label": syn,
                        "sources": sources,
                        "canonical": label
                    })

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diseases, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(diseases)} disease terms to: {output_path}")

    # Clean up raw file (large)
    json_path.unlink()

    return output_path


# =============================================================================
# 3. ChEMBL (Drug Database)
# =============================================================================
def download_chembl():
    """
    Download and convert ChEMBL drug data.

    Source: https://www.ebi.ac.uk/chembl/
    Note: Full ChEMBL is very large. We use the molecule dictionary.
    Output: JSON in pipeline format
    """
    print("\n" + "="*60)
    print("ChEMBL: Drug Database")
    print("="*60)

    # ChEMBL provides FTP downloads, but they're large
    # Use the ChEMBL web service API for approved drugs
    # Alternative: Use pre-built drug name file from ChEMBL

    # For a simpler approach, we'll use the ChEMBL SQLite or TSV
    # The molecule_dictionary contains drug names

    # Using ChEMBL's UniChem data which is smaller
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_34_chemreps.txt.gz"

    print("Note: ChEMBL full database is very large (>1GB).")
    print("For production use, consider:")
    print("  1. Download chembl_34_sqlite.tar.gz from ChEMBL FTP")
    print("  2. Extract molecule names from molecule_dictionary table")
    print("  3. Or use ChEMBL web API for specific drug lookups")

    # Create a placeholder with instructions
    output_path = OUTPUT_DIR / "2025_chembl_drugs.json"

    # For now, create a smaller curated list from ChEMBL's approved drugs
    # This would typically be populated by querying ChEMBL API or database

    placeholder = {
        "_info": "ChEMBL drug lexicon placeholder",
        "_source": "https://www.ebi.ac.uk/chembl/",
        "_instructions": [
            "To populate this lexicon:",
            "1. Download ChEMBL SQLite: ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/",
            "2. Query: SELECT pref_name, chembl_id, max_phase FROM molecule_dictionary WHERE max_phase >= 3",
            "3. Or use ChEMBL API: https://www.ebi.ac.uk/chembl/api/data/molecule?max_phase=4"
        ],
        "drugs": []
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, indent=2)

    print(f"Created placeholder at: {output_path}")
    print("Run the ChEMBL API script to populate with actual data.")

    return output_path


def download_chembl_api():
    """
    Download approved drugs from ChEMBL using their API.

    This is a more practical approach than downloading the full database.
    """
    import urllib.request
    import json

    print("\n" + "="*60)
    print("ChEMBL API: Approved Drugs")
    print("="*60)

    output_path = OUTPUT_DIR / "2025_chembl_drugs.json"

    # ChEMBL API for approved drugs (max_phase = 4)
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    params = "?max_phase=4&limit=1000"

    drugs: List[Dict[str, Any]] = []
    offset = 0
    total = None

    while True:
        url = f"{base_url}{params}&offset={offset}"
        print(f"Fetching: offset={offset}")

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        if total is None:
            total = data.get("page_meta", {}).get("total_count", 0)
            print(f"Total approved drugs: {total}")

        molecules = data.get("molecules", [])
        if not molecules:
            break

        for mol in molecules:
            chembl_id = mol.get("molecule_chembl_id", "")
            pref_name = mol.get("pref_name", "")

            if not pref_name:
                continue

            # Get synonyms from molecule_synonyms
            synonyms = []
            for syn in mol.get("molecule_synonyms", []) or []:
                syn_name = syn.get("molecule_synonym", "")
                if syn_name and syn_name != pref_name:
                    synonyms.append(syn_name)

            drug_entry = {
                "label": pref_name,
                "chembl_id": chembl_id,
                "max_phase": mol.get("max_phase"),
                "molecule_type": mol.get("molecule_type"),
                "synonyms": synonyms[:10]
            }
            drugs.append(drug_entry)

        offset += len(molecules)
        if offset >= total:
            break

        # Rate limiting
        import time
        time.sleep(0.5)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(drugs, f, indent=2, ensure_ascii=False)

    print(f"Downloaded {len(drugs)} approved drugs to: {output_path}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Download all lexicons."""
    print("="*60)
    print("LEXICON DOWNLOADER")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download each lexicon
    results = {}

    # 1. Meta-Inventory abbreviations
    results["meta_inventory"] = download_meta_inventory()

    # 2. MONDO diseases
    results["mondo"] = download_mondo()

    # 3. ChEMBL drugs (API version)
    try:
        results["chembl"] = download_chembl_api()
    except Exception as e:
        print(f"ChEMBL API failed: {e}")
        results["chembl"] = download_chembl()

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, path in results.items():
        status = "OK" if path and Path(path).exists() else "FAILED"
        print(f"  {name}: {status}")
        if path:
            print(f"    -> {path}")


if __name__ == "__main__":
    main()
