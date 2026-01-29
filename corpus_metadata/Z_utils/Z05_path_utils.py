# corpus_metadata/Z_utils/Z05_path_utils.py
"""
Path resolution utilities for ESE pipeline.

Provides centralized base path resolution with environment variable support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def get_base_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Resolve the base path for the ESE pipeline.

    Resolution order:
    1. CORPUS_BASE_PATH environment variable (if set and non-empty)
    2. paths.base from config (if provided and non-null)
    3. Auto-detect: parent of corpus_metadata directory

    Args:
        config: Optional config dict with paths.base key

    Returns:
        Resolved base path as Path object
    """
    # 1. Check environment variable
    env_path = os.environ.get("CORPUS_BASE_PATH", "").strip()
    if env_path:
        return Path(env_path)

    # 2. Check config value
    if config:
        paths_cfg = config.get("paths", {})
        cfg_base = paths_cfg.get("base")
        if cfg_base:  # Not None or empty string
            return Path(cfg_base)

    # 3. Auto-detect from this file's location
    # This file is at: <base>/corpus_metadata/Z_utils/Z05_path_utils.py
    # So base is 3 levels up
    this_file = Path(__file__).resolve()
    return this_file.parent.parent.parent


__all__ = ["get_base_path"]
