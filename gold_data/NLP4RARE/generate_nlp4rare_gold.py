#!/usr/bin/env python3
"""
Generate nlp4rare_gold.json from .ann files in NLP4RARE dataset.

NLP4RARE corpus format (BRAT standoff):
- T lines: Text annotations with entity type, span, and text
- R lines: Relations between entities

Extracts:
1. Abbreviations: `Is_acron` relations (short_form -> long_form)
2. Diseases: RAREDISEASE, DISEASE, SKINRAREDISEASE entity types

Reference: https://github.com/isegura/NLP4RARE-CM-UC3M
"""

import json
import re
from pathlib import Path


def looks_like_acronym(text: str) -> bool:
    """
    Check if text looks like an acronym (short, mostly uppercase).
    """
    text = text.strip()

    if len(text) < 2 or len(text) > 12:
        return False

    if ' ' in text and len(text) > 8:
        return False

    uppercase_count = sum(1 for c in text if c.isupper())
    letter_count = sum(1 for c in text if c.isalpha())

    if letter_count == 0:
        return False

    uppercase_ratio = uppercase_count / letter_count
    if uppercase_ratio < 0.4:
        return False

    if not text[0].isupper():
        return False

    return True


def is_valid_acronym(short_form: str) -> bool:
    """
    Check if short_form looks like a valid acronym/abbreviation.
    """
    sf = short_form.strip()

    if not looks_like_acronym(sf):
        return False

    anaphors = {
        "it", "this", "that", "these", "those", "they", "them",
        "the disorder", "the disease", "the condition", "the syndrome",
    }
    if sf.lower() in anaphors:
        return False

    non_acronym_words = {
        "aching", "pain", "fever", "syndrome", "disease", "disorder",
    }
    if sf.lower() in non_acronym_words:
        return False

    return True


def parse_ann_file(ann_path: Path) -> dict:
    """
    Parse a BRAT .ann file to extract:
    1. Is_acron relations (abbreviations)
    2. Disease entities (RAREDISEASE, DISEASE, SKINRAREDISEASE)
    """
    content = ann_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.strip().split("\n")

    # Store text annotations: id -> {type, text, start, end}
    text_annotations = {}
    # Store Is_acron relations
    acronym_relations = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse T (text-bound) annotations
        if line.startswith("T"):
            match = re.match(r"^(T\d+)\t(\S+)\s+(\d+)\s+(\d+)\t(.+)$", line)
            if match:
                tid, etype, start, end, text = match.groups()
                text_annotations[tid] = {
                    "type": etype,
                    "text": text.strip(),
                    "start": int(start),
                    "end": int(end)
                }

        # Parse R (relation) annotations - ONLY Is_acron
        elif line.startswith("R"):
            match = re.match(r"^R\d+\tIs_acron\s+Arg1:(T\d+)\s+Arg2:(T\d+)", line)
            if match:
                short_form_id, long_form_id = match.groups()
                acronym_relations.append((short_form_id, long_form_id))

    # Extract abbreviations from Is_acron relations
    abbreviations = []
    for short_id, long_id in acronym_relations:
        if short_id not in text_annotations or long_id not in text_annotations:
            continue

        short_ann = text_annotations[short_id]
        long_ann = text_annotations[long_id]

        short_form = short_ann["text"].strip()
        long_form = long_ann["text"].strip()
        category = long_ann["type"]

        # Skip self-references
        if short_form.lower() == long_form.lower():
            continue

        # Validate short_form looks like an acronym
        if not looks_like_acronym(short_form):
            if looks_like_acronym(long_form):
                short_form, long_form = long_form, short_form
            else:
                continue

        if len(short_form) >= len(long_form):
            continue

        abbreviations.append({
            "short_form": short_form,
            "long_form": long_form,
            "category": category
        })

    # Extract disease entities
    disease_types = {"RAREDISEASE", "DISEASE", "SKINRAREDISEASE"}
    diseases = []
    seen_diseases = set()

    for tid, ann in text_annotations.items():
        if ann["type"] in disease_types:
            text = ann["text"].strip()
            # Normalize for deduplication
            text_lower = text.lower()
            if text_lower not in seen_diseases and len(text) >= 3:
                seen_diseases.add(text_lower)
                diseases.append({
                    "text": text,
                    "type": ann["type"],
                    "start": ann["start"],
                    "end": ann["end"]
                })

    return {
        "abbreviations": abbreviations,
        "diseases": diseases
    }


def clean_abbreviations(annotations: list[dict]) -> list[dict]:
    """Clean and deduplicate abbreviation annotations."""
    cleaned = []
    seen = set()

    for ann in annotations:
        short_form = ann["short_form"].strip()
        long_form = ann["long_form"].strip()
        doc_id = ann.get("doc_id", "")

        if short_form.lower() == long_form.lower():
            continue

        if not looks_like_acronym(short_form):
            if looks_like_acronym(long_form) and len(long_form) < len(short_form):
                short_form, long_form = long_form, short_form
            else:
                continue

        if not is_valid_acronym(short_form):
            continue

        if len(short_form) >= len(long_form):
            continue

        key = (short_form.lower(), long_form.lower(), doc_id)
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({
            "short_form": short_form,
            "long_form": long_form,
            "category": ann["category"],
            "doc_id": doc_id
        })

    return cleaned


def process_dataset(base_path: Path, folders: list[str]) -> dict:
    """Process all .ann files across specified folders."""
    all_abbreviations = []
    all_diseases = []

    for folder_name in folders:
        folder_path = base_path / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue

        ann_files = list(folder_path.glob("*.ann"))
        print(f"\nProcessing {folder_name}/: {len(ann_files)} .ann files")

        folder_abbrev_count = 0
        folder_disease_count = 0

        for ann_file in sorted(ann_files):
            try:
                result = parse_ann_file(ann_file)
                doc_id = ann_file.stem + ".pdf"

                # Add abbreviations
                for ann in result["abbreviations"]:
                    ann["doc_id"] = doc_id
                    all_abbreviations.append(ann)
                folder_abbrev_count += len(result["abbreviations"])

                # Add diseases
                for disease in result["diseases"]:
                    disease["doc_id"] = doc_id
                    all_diseases.append(disease)
                folder_disease_count += len(result["diseases"])

            except Exception as e:
                print(f"  ! {ann_file.name}: {e}")

        print(f"  Abbreviations (Is_acron): {folder_abbrev_count}")
        print(f"  Disease entities: {folder_disease_count}")

    # Clean abbreviations
    print("\n" + "-" * 40)
    print(f"Raw abbreviations: {len(all_abbreviations)}")
    cleaned_abbreviations = clean_abbreviations(all_abbreviations)
    print(f"After cleaning: {len(cleaned_abbreviations)}")

    # Deduplicate diseases (same text in same doc)
    seen_diseases = set()
    unique_diseases = []
    for d in all_diseases:
        key = (d["text"].lower(), d["doc_id"])
        if key not in seen_diseases:
            seen_diseases.add(key)
            unique_diseases.append(d)

    print(f"Disease entities: {len(unique_diseases)} (unique per doc)")
    print("-" * 40)

    return {
        "corpus": "NLP4RARE-CM-UC3M",
        "abbreviations": {
            "total": len(cleaned_abbreviations),
            "annotations": cleaned_abbreviations
        },
        "diseases": {
            "total": len(unique_diseases),
            "annotations": unique_diseases
        },
        "genes": {
            "total": 0,
            "annotations": [],
            "note": "NLP4RARE does not contain gene annotations"
        }
    }


def main():
    base_path = Path(__file__).parent
    folders = ["dev", "test", "train"]
    output_file = base_path.parent / "nlp4rare_gold.json"

    print("=" * 60)
    print("NLP4RARE Gold Standard Generator")
    print("=" * 60)

    result = process_dataset(base_path, folders)

    # Write JSON output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Output: {output_file}")
    print(f"Abbreviations: {result['abbreviations']['total']}")
    print(f"Diseases: {result['diseases']['total']}")
    print(f"Genes: {result['genes']['total']} (not in corpus)")
    print("=" * 60)

    # Show samples
    if result["abbreviations"]["annotations"]:
        print("\nSample abbreviations:")
        for ann in result["abbreviations"]["annotations"][:5]:
            print(f"  {ann['short_form']} → {ann['long_form']}")

    if result["diseases"]["annotations"]:
        print("\nSample diseases:")
        for ann in result["diseases"]["annotations"][:5]:
            print(f"  [{ann['type']}] {ann['text']}")


if __name__ == "__main__":
    main()
