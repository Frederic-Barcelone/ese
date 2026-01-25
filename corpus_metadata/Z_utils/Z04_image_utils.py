# corpus_metadata/Z_utils/Z04_image_utils.py
"""
Image utilities for Vision LLM processing.

Handles:
- Image size calculation from base64 strings
- Image compression to meet Vision API limits (5MB)
- Image format detection
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional, Tuple

# Vision API limits
MAX_VISION_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
DEFAULT_COMPRESSION_QUALITY = 85
DEFAULT_MAX_DIMENSION = 2048

# Optional PIL import
try:
    from PIL import Image
    PIL_AVAILABLE = True
    # Handle PIL version differences for resampling filter
    try:
        LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:
        LANCZOS = getattr(Image, 'LANCZOS', None)
except ImportError:
    Image = None
    PIL_AVAILABLE = False
    LANCZOS = None


def get_image_size_bytes(base64_str: str) -> int:
    """
    Calculate the decoded size of a base64-encoded image.

    Args:
        base64_str: Base64-encoded image string (without data URI prefix)

    Returns:
        Size in bytes of the decoded image
    """
    if not base64_str:
        return 0

    # Remove any data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    # Base64 encoding increases size by ~33%, so decoded size is ~75% of encoded
    # More accurate: each 4 base64 chars = 3 bytes
    # Account for padding
    padding = base64_str.count("=")
    return (len(base64_str) * 3 // 4) - padding


def detect_image_format(base64_str: str) -> str:
    """
    Detect image format from base64 header.

    Args:
        base64_str: Base64-encoded image string

    Returns:
        MIME type string (e.g., "image/png", "image/jpeg")
    """
    if not base64_str:
        return "image/png"

    # Remove any data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    # Check magic bytes in base64
    if base64_str.startswith("/9j/"):
        return "image/jpeg"
    elif base64_str.startswith("R0lGOD"):
        return "image/gif"
    elif base64_str.startswith("UklGR"):
        return "image/webp"
    elif base64_str.startswith("iVBORw"):
        return "image/png"

    # Default to PNG
    return "image/png"


def compress_image_for_vision(
    base64_str: str,
    max_size_bytes: int = MAX_VISION_IMAGE_SIZE_BYTES,
    quality: int = DEFAULT_COMPRESSION_QUALITY,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> Tuple[Optional[str], dict]:
    """
    Compress an image to meet Vision API size limits.

    Strategy:
    1. If already under limit, return as-is
    2. Resize to max_dimension while maintaining aspect ratio
    3. Convert to JPEG with specified quality
    4. Progressively reduce quality if still too large

    Args:
        base64_str: Base64-encoded image string
        max_size_bytes: Maximum allowed size (default 5MB)
        quality: Initial JPEG quality (default 85)
        max_dimension: Maximum width/height (default 2048)

    Returns:
        Tuple of (compressed_base64, info_dict)
        info_dict contains: original_size, final_size, was_compressed, compression_ratio, error
    """
    info: Dict[str, Any] = {
        "original_size": 0,
        "final_size": 0,
        "was_compressed": False,
        "compression_ratio": 1.0,
        "error": None,
        "original_format": None,
        "actions": [],
    }

    if not base64_str:
        info["error"] = "Empty input"
        return None, info

    # Remove any data URI prefix
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    # Calculate original size
    original_size = get_image_size_bytes(base64_str)
    info["original_size"] = original_size
    info["original_format"] = detect_image_format(base64_str)

    # If already under limit, return as-is
    if original_size <= max_size_bytes:
        info["final_size"] = original_size
        return base64_str, info

    # PIL required for compression
    if not PIL_AVAILABLE:
        info["error"] = "PIL not available for compression"
        return None, info

    try:
        # Decode image
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if needed (for JPEG output)
        if img.mode in ("RGBA", "P", "LA"):
            # Create white background for transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode in ("RGBA", "LA"):
                background.paste(img, mask=img.split()[-1])  # Use alpha as mask
                img = background
            else:
                img = img.convert("RGB")
            info["actions"].append("converted_to_rgb")
        elif img.mode != "RGB":
            img = img.convert("RGB")
            info["actions"].append("converted_to_rgb")

        # Resize if larger than max_dimension
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), LANCZOS)
            info["actions"].append(f"resized_{width}x{height}_to_{new_width}x{new_height}")

        # Compress as JPEG with progressive quality reduction
        current_quality = quality
        min_quality = 30

        while current_quality >= min_quality:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=current_quality, optimize=True)
            compressed_bytes = buffer.getvalue()

            if len(compressed_bytes) <= max_size_bytes:
                compressed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")
                info["final_size"] = len(compressed_bytes)
                info["was_compressed"] = True
                info["compression_ratio"] = original_size / len(compressed_bytes)
                info["actions"].append(f"jpeg_quality_{current_quality}")
                return compressed_base64, info

            # Reduce quality and try again
            current_quality -= 10

        # If still too large after quality reduction, try more aggressive resize
        width, height = img.size
        while width > 512 or height > 512:
            width = int(width * 0.7)
            height = int(height * 0.7)
            img_resized = img.resize((width, height), LANCZOS)

            buffer = io.BytesIO()
            img_resized.save(buffer, format="JPEG", quality=min_quality, optimize=True)
            compressed_bytes = buffer.getvalue()

            if len(compressed_bytes) <= max_size_bytes:
                compressed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")
                info["final_size"] = len(compressed_bytes)
                info["was_compressed"] = True
                info["compression_ratio"] = original_size / len(compressed_bytes)
                info["actions"].append(f"aggressive_resize_{width}x{height}")
                return compressed_base64, info

        # Could not compress enough
        info["error"] = f"Could not compress below {max_size_bytes} bytes"
        return None, info

    except Exception as e:
        info["error"] = f"Compression failed: {str(e)}"
        return None, info


def is_image_oversized(base64_str: str, max_size_bytes: int = MAX_VISION_IMAGE_SIZE_BYTES) -> bool:
    """
    Quick check if an image exceeds the size limit.

    Args:
        base64_str: Base64-encoded image string
        max_size_bytes: Maximum allowed size (default 5MB)

    Returns:
        True if image exceeds limit
    """
    return get_image_size_bytes(base64_str) > max_size_bytes


__all__ = [
    "MAX_VISION_IMAGE_SIZE_BYTES",
    "DEFAULT_COMPRESSION_QUALITY",
    "DEFAULT_MAX_DIMENSION",
    "get_image_size_bytes",
    "detect_image_format",
    "compress_image_for_vision",
    "is_image_oversized",
]
