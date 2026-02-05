#!/usr/bin/env python3
"""
Generate cadec_gold.json from CADEC-for-NLP encoded CoNLL files.

CADEC (CSIRO Adverse Drug Event Corpus) format:
- Encoded CoNLL: column 0 = character-length, columns 1-5 = BIO tags
- Column order: ADR, Disease, Drug, Symptom, Finding
- Document headers are single-token lines (e.g. LIPITOR.108)
- Blank lines separate sentences

Decoding requires the original CADEC v2 text files from CSIRO.
If not available, the script falls back to decoded .conll files if present.

Reference: Stanovsky et al., EACL 2017
           Karimi et al., Journal of Biomedical Informatics, 2015

Usage:
    python generate_cadec_gold.py
"""

import json
import os
import re
import sys
from pathlib import Path


# Columns in CoNLL format (0-indexed after token/length column)
COL_ADR = 1
COL_DISEASE = 2
COL_DRUG = 3
COL_SYMPTOM = 4
COL_FINDING = 5


def read_sents_from_file(text_fn: str):
    """
    Read sentences from CADEC v2 text file.
    Reimplements decode logic from CADEC-for-NLP/src/decode_dataset.py.
    Each line becomes a sentence with whitespace stripped.
    """
    return iter([
        sent.strip().replace(" ", "")
        for sent in open(text_fn, encoding="utf8")
        if sent.strip()
    ])


def decode_encoded_conll(encoded_path: Path, text_base_dir: Path) -> list[dict]:
    """
    Decode an encoded CoNLL file using CADEC v2 text files.

    Returns list of document dicts, each with:
    - doc_id: str
    - sentences: list of list of (token, tags) tuples
    """
    documents = []
    cur_doc_id = None
    cur_sents = None
    cur_sent = None
    cur_sentences = []
    cur_sentence_tokens = []

    for line in open(encoded_path, encoding="utf8"):
        line = line.rstrip("\n")
        data = line.split()

        if len(data) == 1 and re.match(r"^[A-Za-z]", data[0]):
            # New document header
            if cur_doc_id and (cur_sentences or cur_sentence_tokens):
                if cur_sentence_tokens:
                    cur_sentences.append(cur_sentence_tokens)
                documents.append({
                    "doc_id": cur_doc_id,
                    "sentences": cur_sentences,
                })

            cur_doc_id = data[0]
            cur_sentences = []
            cur_sentence_tokens = []

            text_file = text_base_dir / f"{cur_doc_id}.txt"
            if not text_file.exists():
                print(f"  WARNING: Missing text file: {text_file}")
                cur_sents = iter([])
                cur_sent = None
            else:
                cur_sents = read_sents_from_file(str(text_file))
                try:
                    cur_sent = next(cur_sents)
                except StopIteration:
                    cur_sent = None

        elif len(data) == 0:
            # Sentence boundary
            if cur_sentence_tokens:
                cur_sentences.append(cur_sentence_tokens)
                cur_sentence_tokens = []
            try:
                cur_sent = next(cur_sents) if cur_sents else None
            except StopIteration:
                cur_sent = None

        elif len(data) >= 6:
            # Token line: length + 5 BIO tag columns
            cur_word_len = int(data[0])
            if cur_sent and len(cur_sent) >= cur_word_len:
                token = cur_sent[:cur_word_len]
                cur_sent = cur_sent[cur_word_len:]
            else:
                token = f"<UNK:{cur_word_len}>"

            tags = data[1:6]  # ADR, Disease, Drug, Symptom, Finding
            cur_sentence_tokens.append((token, tags))

    # Save last document
    if cur_doc_id and (cur_sentences or cur_sentence_tokens):
        if cur_sentence_tokens:
            cur_sentences.append(cur_sentence_tokens)
        documents.append({
            "doc_id": cur_doc_id,
            "sentences": cur_sentences,
        })

    return documents


def parse_decoded_conll(conll_path: Path) -> list[dict]:
    """
    Parse an already-decoded CoNLL file (with actual tokens).

    Returns same structure as decode_encoded_conll.
    """
    documents = []
    cur_doc_id = None
    cur_sentences = []
    cur_sentence_tokens = []

    for line in open(conll_path, encoding="utf8"):
        line = line.rstrip("\n")
        parts = line.split("\t")

        if len(parts) == 1 and parts[0].strip() and re.match(r"^[A-Za-z]", parts[0].strip()):
            # Document header
            if cur_doc_id and (cur_sentences or cur_sentence_tokens):
                if cur_sentence_tokens:
                    cur_sentences.append(cur_sentence_tokens)
                documents.append({
                    "doc_id": cur_doc_id,
                    "sentences": cur_sentences,
                })

            cur_doc_id = parts[0].strip()
            cur_sentences = []
            cur_sentence_tokens = []

        elif not line.strip():
            # Sentence boundary
            if cur_sentence_tokens:
                cur_sentences.append(cur_sentence_tokens)
                cur_sentence_tokens = []

        elif len(parts) >= 6:
            token = parts[0]
            tags = parts[1:6]
            cur_sentence_tokens.append((token, tags))

    if cur_doc_id and (cur_sentences or cur_sentence_tokens):
        if cur_sentence_tokens:
            cur_sentences.append(cur_sentence_tokens)
        documents.append({
            "doc_id": cur_doc_id,
            "sentences": cur_sentences,
        })

    return documents


def extract_drug_entities(doc: dict) -> list[dict]:
    """
    Extract drug entity spans from a document's BIO tags.

    Returns list of {"name": str, "tokens": [str, ...]} dicts.
    """
    entities = []
    current_entity_tokens = []

    for sentence in doc["sentences"]:
        for token, tags in sentence:
            drug_tag = tags[COL_DRUG - 1]  # -1 because tags is 0-indexed

            if drug_tag.startswith("B-"):
                # Save previous entity if exists
                if current_entity_tokens:
                    entities.append(current_entity_tokens)
                current_entity_tokens = [token]
            elif drug_tag.startswith("I-") and current_entity_tokens:
                current_entity_tokens.append(token)
            else:
                if current_entity_tokens:
                    entities.append(current_entity_tokens)
                    current_entity_tokens = []

    if current_entity_tokens:
        entities.append(current_entity_tokens)

    # Convert token lists to entity dicts, filtering noise
    result = []
    seen = set()
    for tokens in entities:
        name = " ".join(tokens)
        name_lower = name.lower()

        # Skip noisy annotations: single chars, pure numbers, punctuation
        name_stripped = name.strip()
        if len(name_stripped) < 2:
            continue
        if re.match(r"^[\d\s.,;:!?]+$", name_stripped):
            continue

        if name_lower not in seen:
            seen.add(name_lower)
            result.append({"name": name, "tokens": tokens})

    return result


def build_token_data(doc: dict) -> tuple[list[str], list[str]]:
    """
    Flatten document into token list and drug BIO tag list.

    Returns (tokens, drug_bio_tags).
    """
    tokens = []
    bio_tags = []

    for sentence in doc["sentences"]:
        for token, tags in sentence:
            tokens.append(token)
            drug_tag = tags[COL_DRUG - 1]
            # Normalize: B-<id> → B-Drug, I-<id> → I-Drug, O → O
            if drug_tag.startswith("B-"):
                bio_tags.append("B-Drug")
            elif drug_tag.startswith("I-"):
                bio_tags.append("I-Drug")
            else:
                bio_tags.append("O")

    return tokens, bio_tags


def main():
    base_path = Path(__file__).parent
    raw_path = base_path / "raw" / "CADEC-for-NLP"
    output_file = base_path / "cadec_gold.json"

    # CADEC v2 text files location (needed for decoding)
    text_base_dir = raw_path / "data" / "CADEC" / "v2" / "text"

    print("=" * 60)
    print("CADEC Drug Gold Standard Generator")
    print("=" * 60)

    all_documents = {}
    all_annotations = []
    split_counts = {}

    for split_name in ["train", "test"]:
        encoded_file = raw_path / "data" / f"{split_name}.encoded.conll"
        decoded_file = raw_path / "data" / f"{split_name}.conll"

        if decoded_file.exists():
            print(f"\nUsing decoded file: {decoded_file}")
            docs = parse_decoded_conll(decoded_file)
        elif encoded_file.exists() and text_base_dir.exists():
            print(f"\nDecoding: {encoded_file}")
            docs = decode_encoded_conll(encoded_file, text_base_dir)
        elif encoded_file.exists():
            print(f"\nERROR: Found encoded file but CADEC v2 text files are missing.")
            print(f"  Expected at: {text_base_dir}")
            print(f"  Download CADEC v2 from: https://doi.org/10.4225/08/570FB102BDAD2")
            print(f"  Unpack into: {raw_path / 'data' / 'CADEC'}")
            print(f"\n  Alternatively, run the decode script first:")
            print(f"    cd {raw_path}")
            print(f"    bash scripts/create_corpus.sh")
            sys.exit(1)
        else:
            print(f"\nWARNING: No data files found for {split_name} split")
            continue

        print(f"  Documents: {len(docs)}")

        doc_drug_count = 0
        for doc in docs:
            doc_id = doc["doc_id"]
            drug_entities = extract_drug_entities(doc)
            tokens, bio_tags = build_token_data(doc)

            all_documents[doc_id] = {
                "split": split_name,
                "text": " ".join(tokens),
                "tokens": tokens,
                "drug_bio_tags": bio_tags,
            }

            for entity in drug_entities:
                all_annotations.append({
                    "doc_id": doc_id,
                    "name": entity["name"],
                    "split": split_name,
                })
            doc_drug_count += len(drug_entities)

        split_counts[split_name] = {
            "documents": len(docs),
            "drug_annotations": doc_drug_count,
        }
        print(f"  Drug annotations: {doc_drug_count}")

    # Build output
    result = {
        "corpus": "CADEC",
        "source": "CADEC-for-NLP (Stanovsky et al., EACL 2017)",
        "splits": split_counts,
        "drugs": {
            "total": len(all_annotations),
            "annotations": all_annotations,
        },
        "documents": all_documents,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Output: {output_file}")
    print(f"Total documents: {len(all_documents)}")
    print(f"Total drug annotations: {len(all_annotations)}")
    for split_name, counts in split_counts.items():
        print(f"  {split_name}: {counts['documents']} docs, {counts['drug_annotations']} drugs")
    print("=" * 60)

    # Show samples
    if all_annotations:
        print("\nSample drug annotations:")
        for ann in all_annotations[:10]:
            print(f"  [{ann['split']}] {ann['doc_id']}: {ann['name']}")


if __name__ == "__main__":
    main()
