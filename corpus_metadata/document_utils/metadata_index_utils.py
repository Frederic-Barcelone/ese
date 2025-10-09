#
# /Users/kxbb828/Library/CloudStorage/OneDrive-AZCollaboration/Desktop/WIP-ESE/17_Corpus/corpus_metadata/document_utils/metadata_index_utils.py
#
import re
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

def get_next_available_index(used_indices: Set[int]) -> int:
        """
        Find the next available index using gap-first strategy.
        Returns the actual integer index (not formatted string).
        
        Args:
            used_indices: Set of already used integer indices
            
        Returns:
            Next available integer index
            
        Examples:
            >>> get_next_available_index({1, 2, 4, 5})
            3  # Fills the gap at position 3
            
            >>> get_next_available_index({1, 2, 3})
            4  # No gaps, so returns next sequential
            
            >>> get_next_available_index(set())
            1  # Empty set, start at 1
        """
        if not used_indices:
            return 1
        
        max_index = max(used_indices)
        
        # First, look for gaps in the sequence from 1 to max_index
        for i in range(1, max_index + 1):
            if i not in used_indices:
                return i  # Found a gap, use it
        
        # No gaps found, use next sequential number
        return max_index + 1

def is_already_indexed(filename: str) -> bool:
    """Check if a filename already has an index prefix (00000_ or 00000-)."""
    return bool(re.match(r'^\d{5}[_-]', filename))

def extract_index_from_filename(filename: str) -> Optional[int]:
    """Extract the numeric index from a filename like '00096_document.pdf'."""
    m = re.match(r'^(\d{5})[_-]', filename)
    return int(m.group(1)) if m else None

def scan_used_indices(directories: List[Path]) -> Set[int]:
    """Scan multiple directories and return a set of all used indices."""
    used_indices: Set[int] = set()
    
    for directory in directories:
        if not directory.exists():
            continue
        
        try:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    idx = extract_index_from_filename(file_path.name)
                    if idx is not None:
                        used_indices.add(idx)
            logger.debug(f"Scanned {len(used_indices)} indices in {directory}")
        except PermissionError as e:
            logger.warning(f"Permission denied scanning {directory}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {directory}: {e}")
    
    return used_indices

def rename_with_index_and_intelligent_name(
    file_path: Path,
    index: str,
    intelligent_name: Optional[str] = None
) -> Path:
    """
    Rename a file by adding an index prefix and optionally using intelligent naming.
    
    Args:
        file_path: Original file path
        index: Index string (e.g., "00001")
        intelligent_name: Optional intelligent base name (without index/extension)
    
    Returns:
        New file path after renaming
    """
    parent, ext = file_path.parent, file_path.suffix
    base = f"{index}_{intelligent_name}" if intelligent_name else f"{index}_{file_path.stem}"
    new_path = parent / (base + ext)

    counter = 1
    while new_path.exists():
        new_path = parent / (f"{base}_{counter}{ext}")
        counter += 1

    shutil.move(str(file_path), str(new_path))
    logger.info(f"Renamed '{file_path.name}' → '{new_path.name}'")
    return new_path

def find_first_gap(
    used_indices: Set[int],
    start: int = 1,
    end: Optional[int] = None
) -> Optional[int]:
    """Find the first missing integer in a sequence of indices."""
    if not used_indices:
        return start
    end = end or max(used_indices)
    for i in range(start, end + 1):
        if i not in used_indices:
            return i
    return None

def get_next_available_index(used_indices: Set[int]) -> int:
    """
    Find the next available index using gap-first strategy.
    Returns the actual integer index (not formatted string).
    """
    if not used_indices:
        return 1
    
    max_index = max(used_indices)
    
    # First, look for gaps in the sequence from 1 to max_index
    for i in range(1, max_index + 1):
        if i not in used_indices:
            return i  # Found a gap, use it
    
    # No gaps found, use next sequential number
    return max_index + 1

def get_and_increment_last_index_file_based(base_output_dir: str = "./") -> str:
    """
    Determine the next available index based on existing files in target folders.
    
    Scans only:
      - documents_classified
      - documents_conditions
      - documents_issue
      - documents_unclassified
    """
    base_path = Path(base_output_dir)
    dest_dirs = [
        base_path / "documents_classified",
        base_path / "documents_conditions",
        base_path / "documents_issue",
        base_path / "documents_unclassified",
    ]

    logger.info("Scanning destination directories for existing indices")
    used: Set[int] = set()
    for d in dest_dirs:
        if not d.exists():
            logger.warning(f"Directory missing: {d}")
            continue
        count = 0
        try:
            for f in d.iterdir():
                if f.is_file():
                    idx = extract_index_from_filename(f.name)
                    if idx is not None:
                        used.add(idx)
                        count += 1
            logger.debug(f"{d.name}: {count} indexed files")
        except PermissionError as e:
            logger.warning(f"No permissions for {d}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {d}: {e}")

    if not used:
        next_idx = 1
        logger.info("No existing indices found; starting at 00001")
    else:
        gap = find_first_gap(used, start=1, end=max(used))
        if gap is not None:
            next_idx = gap
            logger.info(f"Filling gap at index {next_idx:05d}")
        else:
            next_idx = max(used) + 1
            logger.info(f"No gaps found; next index is {next_idx:05d}")

    if next_idx > 99999:
        raise ValueError(f"Index exceeded maximum of 99999: {next_idx}")

    return f"{next_idx:05d}"

def validate_indices(directories: List[Path]) -> Tuple[bool, List[str]]:
    """
    Validate that there are no duplicate indices across directories.
    Returns (is_valid, list_of_issues).
    """
    index_map: dict[int, Path] = {}
    issues: List[str] = []

    for d in directories:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file():
                idx = extract_index_from_filename(f.name)
                if idx is not None:
                    if idx in index_map:
                        issues.append(
                            f"Duplicate index {idx:05d}: {index_map[idx]} and {f}"
                        )
                    else:
                        index_map[idx] = f

    is_valid = not issues
    return is_valid, issues