# corpus_metadata/Z_utils/Z09_download_gene_lexicon.py
"""
Download and build gene lexicon for rare disease extraction.

Sources:
1. Orphadata Product 6: Genes associated with rare diseases (~4,100 genes)
2. HGNC: Official gene symbols and aliases for normalization

Usage:
    python -m corpus_metadata.Z_utils.Z09_download_gene_lexicon
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from Z_utils.Z08_download_utils import download_file, get_default_output_dir

# URLs
ORPHADATA_GENES_URL = "https://www.orphadata.com/data/xml/en_product6.xml"
HGNC_TSV_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


def parse_orphadata_genes(xml_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse Orphadata Product 6 XML to extract genes associated with rare diseases.

    Returns dict keyed by HGNC symbol with gene metadata and associated diseases.
    """
    print("Parsing Orphadata genes XML...")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    genes: Dict[str, Dict[str, Any]] = {}

    # Navigate XML structure: JDBOR/DisorderList/Disorder/DisorderGeneAssociationList/DisorderGeneAssociation/Gene
    disorder_list = root.find(".//DisorderList")
    if disorder_list is None:
        print("  Warning: No DisorderList found in XML")
        return genes

    for disorder in disorder_list.findall("Disorder"):
        # Get disorder info
        orpha_code_elem = disorder.find("OrphaCode")
        disorder_name_elem = disorder.find("Name")

        if orpha_code_elem is None:
            continue

        orpha_code = orpha_code_elem.text
        disorder_name = disorder_name_elem.text if disorder_name_elem is not None else ""

        # Find gene associations
        gene_assoc_list = disorder.find("DisorderGeneAssociationList")
        if gene_assoc_list is None:
            continue

        for assoc in gene_assoc_list.findall("DisorderGeneAssociation"):
            gene_elem = assoc.find("Gene")
            if gene_elem is None:
                continue

            # Extract gene info
            symbol_elem = gene_elem.find("Symbol")
            name_elem = gene_elem.find("Name")

            if symbol_elem is None or not symbol_elem.text:
                continue

            symbol = symbol_elem.text.strip()
            gene_name = name_elem.text.strip() if name_elem is not None and name_elem.text else ""

            # Get external references
            ext_refs = {}
            ext_ref_list = gene_elem.find("ExternalReferenceList")
            if ext_ref_list is not None:
                for ext_ref in ext_ref_list.findall("ExternalReference"):
                    source_elem = ext_ref.find("Source")
                    ref_elem = ext_ref.find("Reference")
                    if source_elem is not None and ref_elem is not None:
                        source = source_elem.text
                        ref = ref_elem.text
                        if source and ref:
                            ext_refs[source.lower()] = ref

            # Get association type and status
            assoc_type_elem = assoc.find("DisorderGeneAssociationType/Name")
            assoc_status_elem = assoc.find("DisorderGeneAssociationStatus/Name")

            assoc_type = assoc_type_elem.text if assoc_type_elem is not None else ""
            assoc_status = assoc_status_elem.text if assoc_status_elem is not None else ""

            # Build disease entry
            disease_entry = {
                "orphacode": orpha_code,
                "name": disorder_name,
                "association_type": assoc_type,
                "association_status": assoc_status
            }

            # Add or update gene entry
            if symbol not in genes:
                genes[symbol] = {
                    "symbol": symbol,
                    "full_name": gene_name,
                    "hgnc_id": ext_refs.get("hgnc"),
                    "entrez_id": ext_refs.get("ncbi gene") or ext_refs.get("entrez gene"),
                    "ensembl_id": ext_refs.get("ensembl"),
                    "omim_id": ext_refs.get("omim"),
                    "uniprot_id": ext_refs.get("uniprot") or ext_refs.get("swissprot"),
                    "reactome_id": ext_refs.get("reactome"),
                    "aliases": [],
                    "associated_diseases": []
                }

            # Add disease association (avoid duplicates)
            existing_orphacodes = {d["orphacode"] for d in genes[symbol]["associated_diseases"]}
            if orpha_code not in existing_orphacodes:
                genes[symbol]["associated_diseases"].append(disease_entry)

    print(f"  Parsed {len(genes)} genes from Orphadata")
    return genes


def download_and_parse_hgnc(output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Download HGNC complete set and parse aliases/previous symbols.

    Args:
        output_dir: Output directory (default: get_default_output_dir())

    Returns dict keyed by symbol with aliases and metadata.
    """
    print("\nDownloading HGNC gene nomenclature...")

    if output_dir is None:
        output_dir = get_default_output_dir()

    tsv_path = output_dir / "hgnc_complete_set.txt"

    if not download_file(HGNC_TSV_URL, tsv_path):
        print("  Failed to download HGNC data")
        return {}

    print("Parsing HGNC TSV...")

    hgnc_data: Dict[str, Dict[str, Any]] = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        # Read header
        header = f.readline().strip().split("\t")

        # Find column indices
        col_indices = {col: idx for idx, col in enumerate(header)}

        symbol_idx = col_indices.get("symbol", -1)
        hgnc_id_idx = col_indices.get("hgnc_id", -1)
        name_idx = col_indices.get("name", -1)
        alias_symbol_idx = col_indices.get("alias_symbol", -1)
        prev_symbol_idx = col_indices.get("prev_symbol", -1)
        entrez_id_idx = col_indices.get("entrez_id", -1)
        ensembl_id_idx = col_indices.get("ensembl_gene_id", -1)
        uniprot_ids_idx = col_indices.get("uniprot_ids", -1)
        locus_type_idx = col_indices.get("locus_type", -1)

        for line in f:
            fields = line.strip().split("\t")
            if len(fields) <= symbol_idx:
                continue

            symbol = fields[symbol_idx].strip()
            if not symbol:
                continue

            # Parse aliases (pipe-separated)
            aliases: List[str] = []
            if alias_symbol_idx >= 0 and alias_symbol_idx < len(fields):
                alias_str = fields[alias_symbol_idx].strip()
                if alias_str:
                    aliases.extend([a.strip() for a in alias_str.split("|") if a.strip()])

            # Parse previous symbols
            if prev_symbol_idx >= 0 and prev_symbol_idx < len(fields):
                prev_str = fields[prev_symbol_idx].strip()
                if prev_str:
                    aliases.extend([p.strip() for p in prev_str.split("|") if p.strip()])

            # Remove duplicates while preserving order
            seen: Set[str] = set()
            unique_aliases = []
            for a in aliases:
                if a.upper() not in seen and a.upper() != symbol.upper():
                    seen.add(a.upper())
                    unique_aliases.append(a)

            hgnc_data[symbol] = {
                "hgnc_id": fields[hgnc_id_idx] if hgnc_id_idx >= 0 and hgnc_id_idx < len(fields) else None,
                "full_name": fields[name_idx] if name_idx >= 0 and name_idx < len(fields) else None,
                "aliases": unique_aliases,
                "entrez_id": fields[entrez_id_idx] if entrez_id_idx >= 0 and entrez_id_idx < len(fields) else None,
                "ensembl_id": fields[ensembl_id_idx] if ensembl_id_idx >= 0 and ensembl_id_idx < len(fields) else None,
                "uniprot_ids": fields[uniprot_ids_idx] if uniprot_ids_idx >= 0 and uniprot_ids_idx < len(fields) else None,
                "locus_type": fields[locus_type_idx] if locus_type_idx >= 0 and locus_type_idx < len(fields) else None,
            }

    print(f"  Parsed {len(hgnc_data)} genes from HGNC")

    # Clean up
    tsv_path.unlink()

    return hgnc_data


def merge_gene_data(
    orphadata_genes: Dict[str, Dict[str, Any]],
    hgnc_data: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge Orphadata genes with HGNC aliases.

    Only keeps genes from Orphadata (rare disease associated).
    Enriches with HGNC aliases and metadata.
    """
    print("\nMerging Orphadata with HGNC data...")

    merged: List[Dict[str, Any]] = []
    enriched_count = 0

    for symbol, gene_data in orphadata_genes.items():
        # Look up in HGNC
        hgnc_info = hgnc_data.get(symbol, {})

        # Merge aliases
        existing_aliases = set(gene_data.get("aliases", []))
        hgnc_aliases = hgnc_info.get("aliases", [])
        for alias in hgnc_aliases:
            if alias not in existing_aliases:
                existing_aliases.add(alias)

        # Build final entry
        entry = {
            "term": symbol,
            "term_normalized": symbol.lower(),
            "hgnc_id": gene_data.get("hgnc_id") or hgnc_info.get("hgnc_id"),
            "hgnc_symbol": symbol,
            "full_name": gene_data.get("full_name") or hgnc_info.get("full_name"),
            "aliases": list(existing_aliases),
            "entrez_id": gene_data.get("entrez_id") or hgnc_info.get("entrez_id"),
            "ensembl_id": gene_data.get("ensembl_id") or hgnc_info.get("ensembl_id"),
            "omim_id": gene_data.get("omim_id"),
            "uniprot_id": gene_data.get("uniprot_id"),
            "locus_type": hgnc_info.get("locus_type", "protein-coding"),
            "associated_diseases": gene_data.get("associated_diseases", []),
            "source": "orphadata_hgnc"
        }

        merged.append(entry)

        if hgnc_info:
            enriched_count += 1

        # Also add aliases as separate searchable terms pointing to main entry
        for alias in existing_aliases:
            alias_entry = {
                "term": alias,
                "term_normalized": alias.lower(),
                "hgnc_id": entry["hgnc_id"],
                "hgnc_symbol": symbol,  # Points to canonical symbol
                "full_name": entry["full_name"],
                "aliases": [],  # Don't duplicate
                "entrez_id": entry["entrez_id"],
                "ensembl_id": entry["ensembl_id"],
                "omim_id": entry["omim_id"],
                "uniprot_id": entry["uniprot_id"],
                "locus_type": entry["locus_type"],
                "associated_diseases": [],  # Only on main entry
                "source": "hgnc_alias",
                "is_alias_of": symbol
            }
            merged.append(alias_entry)

    print(f"  Total entries: {len(merged)} ({len(orphadata_genes)} genes + aliases)")
    print(f"  Enriched with HGNC: {enriched_count}")

    return merged


def build_gene_lexicon(output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Main function to build the gene lexicon.

    Args:
        output_dir: Output directory (default: get_default_output_dir())

    Returns:
        Path to output file, or None if failed
    """
    print("=" * 60)
    print("GENE LEXICON BUILDER (Rare Disease Focus)")
    print("=" * 60)

    if output_dir is None:
        output_dir = get_default_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and parse Orphadata genes
    xml_path = output_dir / "orphadata_genes_raw.xml"

    if not download_file(ORPHADATA_GENES_URL, xml_path):
        print("Failed to download Orphadata genes")
        return None

    orphadata_genes = parse_orphadata_genes(xml_path)

    # Clean up XML
    xml_path.unlink()

    if not orphadata_genes:
        print("No genes parsed from Orphadata")
        return None

    # 2. Download and parse HGNC for aliases
    hgnc_data = download_and_parse_hgnc(output_dir)

    # 3. Merge data
    merged_genes = merge_gene_data(orphadata_genes, hgnc_data)

    # 4. Write output
    output_path = output_dir / "2025_08_orphadata_genes.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_genes, f, indent=2, ensure_ascii=False)

    print(f"\nWrote gene lexicon to: {output_path}")

    # Summary statistics
    primary_genes = [g for g in merged_genes if g.get("source") == "orphadata_hgnc"]
    alias_entries = [g for g in merged_genes if g.get("source") == "hgnc_alias"]
    total_diseases = sum(len(g.get("associated_diseases", [])) for g in primary_genes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Primary genes: {len(primary_genes)}")
    print(f"  Alias entries: {len(alias_entries)}")
    print(f"  Total searchable terms: {len(merged_genes)}")
    print(f"  Disease associations: {total_diseases}")

    return output_path


if __name__ == "__main__":
    build_gene_lexicon()
