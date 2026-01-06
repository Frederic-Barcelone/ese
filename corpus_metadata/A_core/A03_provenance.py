# corpus_metadata/corpus_metadata/A_core/A03_provenance.py

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def get_git_revision_hash(timeout_sec: float = 1.5) -> str:
    """
    Returns current git commit hash, or 'unknown-version' if unavailable.
    Timeout prevents hangs in odd environments.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_sec,
        )
        return out.strip()
    except Exception:
        return "unknown-version"


def hash_bytes(data: bytes) -> str:
    """SHA256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_string(content: str) -> str:
    """SHA256 hex digest of a UTF-8 string."""
    return hash_bytes(content.encode("utf-8"))


def hash_short(hex_digest: str, n: int = 12) -> str:
    """Helper for human-friendly short hashes (log IDs)."""
    return (hex_digest or "")[:n]


def _stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON string:
    - sort_keys ensures stable key order
    - separators removes whitespace
    - default=str prevents crashes on non-JSON-serializable values
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def compute_prompt_hash(
    system_prompt: str, user_template: str, parameters: Dict[str, Any]
) -> str:
    """
    Fingerprint of prompt + parameters.
    """
    fingerprint = {"sys": system_prompt, "usr": user_template, "params": parameters}
    return hash_string(_stable_json_dumps(fingerprint))


def compute_prompt_bundle_hash(
    system_prompt: str,
    user_template: str,
    schema: Optional[Dict[str, Any]],
    parameters: Dict[str, Any],
) -> str:
    """
    Stronger prompt hash including output schema.
    """
    fingerprint = {
        "sys": system_prompt,
        "usr": user_template,
        "schema": schema or {},
        "params": parameters,
    }
    return hash_string(_stable_json_dumps(fingerprint))


def compute_context_hash(context_text: str) -> str:
    """
    Hash exactly what the verifier saw as context (anti-hallucination debugging).
    """
    return hash_string(context_text or "")


def compute_doc_fingerprint(pdf_bytes: bytes) -> str:
    """
    SHA256 fingerprint of the *raw source document bytes*.
    Store this in ProvenanceMetadata.doc_fingerprint.
    """
    return hash_bytes(pdf_bytes or b"")


def generate_run_id(prefix: str = "RUN") -> str:
    """
    Unique ID for this pipeline execution batch.
    Includes timestamp + uuid for collision-free runs.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    u = uuid.uuid4().hex[:12]
    return f"{prefix}_{ts}_{u}"
