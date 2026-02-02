# corpus_metadata/Z_utils/Z08_download_utils.py
"""
Shared utilities for downloading lexicons and data sources.

Provides:
- download_file(): Download a file from URL with error handling
- get_default_output_dir(): Get default output directory for downloaded files

Used by:
- Z09_download_gene_lexicon.py
- Z10_download_lexicons.py
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def get_default_output_dir() -> Path:
    """
    Get the default output directory for downloaded lexicons.

    Resolution order:
    1. LEXICON_OUTPUT_DIR environment variable (if set)
    2. Default: <project_root>/output_datasources

    Returns:
        Path to output directory
    """
    env_dir = os.environ.get("LEXICON_OUTPUT_DIR", "").strip()
    if env_dir:
        return Path(env_dir)

    # Default: project root / output_datasources
    # This file is at: <root>/corpus_metadata/Z_utils/Z08_download_utils.py
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "output_datasources"


def download_file(
    url: str,
    dest: Path,
    timeout: int = 60,
    verbose: bool = True,
) -> bool:
    """
    Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest: Destination file path
        timeout: Request timeout in seconds (default: 60)
        verbose: Print progress messages (default: True)

    Returns:
        True if download succeeded, False otherwise
    """
    if verbose:
        print(f"Downloading: {url}")

    try:
        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download with timeout
        urllib.request.urlretrieve(url, dest)

        if verbose:
            print(f"  Saved to: {dest}")
        return True

    except urllib.error.URLError as e:
        if verbose:
            print(f"  URL Error: {e}")
        return False

    except urllib.error.HTTPError as e:
        if verbose:
            print(f"  HTTP Error {e.code}: {e.reason}")
        return False

    except OSError as e:
        if verbose:
            print(f"  OS Error: {e}")
        return False

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return False


__all__ = [
    "get_default_output_dir",
    "download_file",
]
