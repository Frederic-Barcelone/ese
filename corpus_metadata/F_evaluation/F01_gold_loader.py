# corpus_metadata/corpus_metadata/F_evaluation/F01_gold_loader.py

"""
Gold Standard Loader

Loads human-annotated abbreviation ground truth from JSON or CSV files.
Returns validated GoldStandard object and doc_id-indexed lookup dictionary.

Normalization applied:
    - doc_id: filename only (strips paths)
    - short_form: uppercased
    - long_form: whitespace-normalized
    - Duplicates removed by (doc_id, SF, LF) key

Modes:
    - strict=True: requires doc_id, short_form, long_form
    - strict=False: allows missing long_form

Used by F02_scorer.py for precision/recall/F1 calculation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class GoldAnnotation(BaseModel):
    """
    A single gold truth.
    Example: doc_id='protocol_101.pdf', short_form='TNF', long_form='Tumor Necrosis Factor'
    """

    doc_id: str
    short_form: str
    long_form: Optional[str] = None
    category: Optional[str] = None  # optional tag: Gene/Drug/etc.

    model_config = ConfigDict(extra="ignore")


class GoldStandard(BaseModel):
    annotations: List[GoldAnnotation] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class GoldLoader:
    """
    Loads gold annotations from JSON or CSV and returns:
      1) a validated GoldStandard object
      2) an index: doc_id -> List[GoldAnnotation]
    """

    def __init__(self, strict: bool = True):
        """
        strict=True:
          - requires doc_id and short_form
          - requires long_form (recommended for abbreviation definition evaluation)
        strict=False:
          - allows missing long_form (kept as None)
        """
        self.strict = strict

    # -------------------------
    # Public API
    # -------------------------

    def load_json(
        self, file_path: str
    ) -> Tuple[GoldStandard, Dict[str, List[GoldAnnotation]]]:
        raw = self._read_json(file_path)
        annos = self._parse_records(raw, source=str(file_path))
        gold = GoldStandard(annotations=annos)
        index = self._index_by_doc(gold.annotations)
        return gold, index

    def load_csv(
        self,
        file_path: str,
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> Tuple[GoldStandard, Dict[str, List[GoldAnnotation]]]:
        rows = self._read_csv(file_path, delimiter=delimiter, encoding=encoding)
        annos = self._parse_records(rows, source=str(file_path))
        gold = GoldStandard(annotations=annos)
        index = self._index_by_doc(gold.annotations)
        return gold, index

    # -------------------------
    # Readers
    # -------------------------

    def _read_json(self, file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_csv(
        self, file_path: str, *, delimiter: str, encoding: str
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                out.append(dict(row))
        return out

    # -------------------------
    # Parsing & normalization
    # -------------------------

    def _parse_records(self, raw: Any, *, source: str) -> List[GoldAnnotation]:
        """
        Accepts:
          - list[dict]
          - dict with key 'annotations' pointing to list[dict]
          - dict with key 'defined_annotations' pointing to list[dict] (v2 format)
        """
        records: List[Dict[str, Any]]

        if (
            isinstance(raw, dict)
            and "annotations" in raw
            and isinstance(raw["annotations"], list)
        ):
            records = raw["annotations"]
        elif (
            isinstance(raw, dict)
            and "defined_annotations" in raw
            and isinstance(raw["defined_annotations"], list)
        ):
            # v2 format: use defined_annotations (extractable pairs with definitions)
            records = raw["defined_annotations"]
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError(
                f"Gold file {source} must be a list or an object with 'annotations'/'defined_annotations' list"
            )

        annos: List[GoldAnnotation] = []
        seen: set[Tuple[str, str, Optional[str]]] = set()

        for item in records:
            if not isinstance(item, dict):
                continue

            doc_raw = (
                item.get("doc_id")
                or item.get("filename")
                or item.get("doc")
                or item.get("file")
            )
            sf_raw = (
                item.get("short_form")
                or item.get("short")
                or item.get("sf")
                or item.get("abbr")
            )
            lf_raw = (
                item.get("long_form")
                or item.get("long")
                or item.get("lf")
                or item.get("expansion")
            )
            cat_raw = item.get("category") or item.get("type") or item.get("label")

            doc_id = self._norm_doc_id(doc_raw)
            sf = self._norm_short_form(sf_raw)
            lf = self._norm_long_form(lf_raw)
            category = (
                str(cat_raw).strip()
                if cat_raw is not None and str(cat_raw).strip()
                else None
            )

            # Required fields
            if not doc_id or not sf:
                if self.strict:
                    raise ValueError(
                        f"Missing doc_id or short_form in {source}: {item}"
                    )
                else:
                    continue

            if self.strict and not lf:
                raise ValueError(f"Missing long_form in strict mode ({source}): {item}")

            key = (doc_id, sf, lf)
            if key in seen:
                continue
            seen.add(key)

            annos.append(
                GoldAnnotation(
                    doc_id=doc_id,
                    short_form=sf,
                    long_form=lf,
                    category=category,
                )
            )

        return annos

    def _norm_doc_id(self, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if not s:
            return ""
        # Keep only filename; avoids path mismatches in CI
        return Path(s).name

    def _norm_short_form(self, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if not s:
            return ""
        return s.upper()

    def _norm_long_form(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = " ".join(str(v).strip().split())
        return s if s else None

    # -------------------------
    # Indexing
    # -------------------------

    def _index_by_doc(
        self, annos: List[GoldAnnotation]
    ) -> Dict[str, List[GoldAnnotation]]:
        ds: Dict[str, List[GoldAnnotation]] = defaultdict(list)
        for a in annos:
            ds[a.doc_id].append(a)
        return ds


__all__ = [
    "GoldAnnotation",
    "GoldStandard",
    "GoldLoader",
]
