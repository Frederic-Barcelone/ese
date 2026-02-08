"""
Load data constants from YAML files in G_config/data/.

Provides typed loader functions for extracting word lists, mappings, and
pair lists from YAML data files. Includes type-safety assertions to catch
YAML boolean coercion (e.g., unquoted 'no' → False).

Key Components:
    - load_term_set: Load a Set[str] from a YAML key
    - load_term_list: Load a List[str] from a YAML key
    - load_mapping: Load a Dict[str, str] from a YAML key
    - load_pair_list: Load a Set[Tuple[str, str]] from a YAML key

Example:
    >>> from Z_utils.Z12_data_loader import load_term_set
    >>> terms = load_term_set("drug_fp_terms.yaml", "bacteria_organisms")
    >>> "influenza" in terms
    True

Dependencies:
    - pyyaml: YAML parsing
"""

from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple

_DATA_DIR = Path(__file__).resolve().parent.parent / "G_config" / "data"


def _check_strings(values: list, filename: str, key: str) -> None:
    """Raise TypeError if any value is not a string (catches YAML boolean coercion)."""
    for v in values:
        if not isinstance(v, str):
            raise TypeError(
                f"{filename}:{key} contains non-string value {v!r} — "
                f"quote it in YAML (no/on/off/yes are coerced to booleans)"
            )


@lru_cache(maxsize=None)
def _load_yaml(filename: str) -> dict:
    """Load and cache a YAML data file."""
    path = _DATA_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def load_term_set(filename: str, key: str) -> Set[str]:
    """Load a set of terms from a YAML file."""
    data = _load_yaml(filename)
    values = data[key]
    _check_strings(values, filename, key)
    return set(values)


def load_term_list(filename: str, key: str) -> List[str]:
    """Load a list of terms from a YAML file."""
    data = _load_yaml(filename)
    values = data[key]
    _check_strings(values, filename, key)
    return list(values)


def load_mapping(filename: str, key: str) -> Dict[str, str]:
    """Load a string-to-string mapping from a YAML file."""
    data = _load_yaml(filename)
    mapping = data[key]
    for k, v in mapping.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"{filename}:{key} contains non-string entry {k!r}: {v!r} — "
                f"quote all keys and values in YAML"
            )
    return dict(mapping)


def load_pair_list(filename: str, key: str) -> Set[Tuple[str, str]]:
    """Load a set of string pairs from a YAML file."""
    data = _load_yaml(filename)
    pairs = data[key]
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"{filename}:{key} contains non-pair {pair!r}")
        _check_strings(pair, filename, key)
    return {tuple(pair) for pair in pairs}
