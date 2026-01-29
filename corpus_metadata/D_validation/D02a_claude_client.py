# corpus_metadata/D_validation/D02a_claude_client.py
"""
Claude (Anthropic) client implementation.

Provides:
- ClaudeClient: Anthropic Claude implementation of LLMClient protocol
- JSON extraction from Claude responses
- Vision API support with auto-compression

Extracted from D02_llm_engine.py to reduce file size.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from Z_utils.Z04_image_utils import (
    get_image_size_bytes,
    compress_image_for_vision,
    MAX_VISION_IMAGE_SIZE_BYTES,
)

# Optional anthropic import
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class ClaudeClient:
    """
    Anthropic Claude client implementing LLMClient protocol.

    Reads config from:
      1. Explicit parameters
      2. config.yaml (if config_path provided)
      3. Environment variables (ANTHROPIC_API_KEY)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        config_path: Optional[str] = None,
    ):
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        # Load config from YAML if provided
        cfg = self._load_config(config_path) if config_path else {}

        # Resolve API key: param > config > env
        self.api_key = (
            api_key or cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var, "
                "pass api_key param, or configure in config.yaml"
            )

        # Resolve model params: param > config > defaults
        self.default_model = model or cfg.get("model", "claude-sonnet-4-20250514")
        self.default_max_tokens = max_tokens or cfg.get("max_tokens", 1024)
        self.default_temperature = (
            temperature if temperature is not None else cfg.get("temperature", 0.0)
        )

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def messages(self):
        """
        Provide access to the messages API for direct usage.

        This allows code to use `client.messages.create(...)` syntax
        which is consistent with the raw Anthropic client interface.
        """
        return self.client.messages

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Claude config from YAML file."""
        path = Path(config_path)
        if not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Extract claude validation config
            # Expected structure: api.claude.validation.{model, max_tokens, temperature}
            claude_cfg = data.get("api", {}).get("claude", {})

            # Try validation config first, then fast config
            val_cfg = claude_cfg.get("validation", {})
            if not val_cfg:
                val_cfg = claude_cfg.get("fast", {})

            return {
                "api_key": claude_cfg.get("api_key"),
                "model": val_cfg.get("model"),
                "max_tokens": val_cfg.get("max_tokens"),
                "temperature": val_cfg.get("temperature"),
            }
        except (OSError, IOError) as e:
            print(f"Warning: Failed to read config file {config_path}: {e}")
            return {}
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse config YAML {config_path}: {e}")
            return {}
        except (KeyError, TypeError) as e:
            print(f"Warning: Invalid config structure in {config_path}: {e}")
            return {}

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        seed: Optional[int] = None,  # Claude doesn't support seed
        response_format: Optional[Dict[str, Any]] = None,  # Ignored, we parse JSON
    ) -> Dict[str, Any]:
        """
        Call Claude and return parsed JSON response.
        """
        use_model = model or self.default_model
        use_max_tokens = max_tokens or self.default_max_tokens
        use_temperature = (
            temperature if temperature is not None else self.default_temperature
        )

        # Call Claude API
        message = self.client.messages.create(
            model=use_model,
            max_tokens=use_max_tokens,
            temperature=use_temperature,
            top_p=top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text content
        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text

        # Parse JSON from response
        return self._extract_json(raw_text)

    def complete_json_any(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """
        Call Claude and return parsed JSON response - handles both objects and arrays.
        """
        use_model = model or self.default_model
        use_max_tokens = max_tokens or self.default_max_tokens
        use_temperature = (
            temperature if temperature is not None else self.default_temperature
        )

        # Call Claude API
        message = self.client.messages.create(
            model=use_model,
            max_tokens=use_max_tokens,
            temperature=use_temperature,
            top_p=top_p,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text content
        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text

        # Parse JSON from response (handles arrays and objects)
        return self._extract_json_any(raw_text)

    def complete_vision_json(
        self,
        *,
        image_base64: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        auto_compress: bool = True,
        max_image_size: int = MAX_VISION_IMAGE_SIZE_BYTES,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Claude Vision API with an image and return parsed JSON response.

        Automatically handles images exceeding the 5MB limit by compressing them.

        Args:
            image_base64: Base64-encoded image (PNG, JPEG, GIF, or WebP)
            prompt: Text prompt describing what to extract
            model: Model to use (default: claude-sonnet-4-20250514)
            max_tokens: Max response tokens
            temperature: Sampling temperature
            auto_compress: Whether to auto-compress oversized images (default True)
            max_image_size: Maximum image size in bytes (default 5MB)

        Returns:
            Parsed JSON dict or None if failed
        """
        use_model = model or self.default_model

        # Check image size and compress if needed
        image_size = get_image_size_bytes(image_base64)
        if image_size > max_image_size:
            if auto_compress:
                print(f"  [INFO] Image exceeds {max_image_size / 1024 / 1024:.1f}MB limit "
                      f"({image_size / 1024 / 1024:.1f}MB), compressing...")
                compressed, info = compress_image_for_vision(
                    image_base64,
                    max_size_bytes=max_image_size,
                )
                if compressed:
                    image_base64 = compressed
                    print(f"  [INFO] Compressed image: {info['original_size'] / 1024 / 1024:.1f}MB -> "
                          f"{info['final_size'] / 1024 / 1024:.1f}MB "
                          f"(ratio: {info['compression_ratio']:.1f}x)")
                else:
                    print(f"  [WARN] Could not compress image below limit: {info.get('error')}")
                    return None
            else:
                print(f"  [WARN] Image exceeds {max_image_size / 1024 / 1024:.1f}MB limit "
                      f"({image_size / 1024 / 1024:.1f}MB), skipping Vision analysis")
                return None

        # Detect media type from base64 header or default to PNG
        media_type = "image/png"
        if image_base64.startswith("/9j/"):
            media_type = "image/jpeg"
        elif image_base64.startswith("R0lGOD"):
            media_type = "image/gif"
        elif image_base64.startswith("UklGR"):
            media_type = "image/webp"

        # Build message with image block
        message = self.client.messages.create(
            model=use_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        # Extract text content
        raw_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw_text += block.text

        # Parse JSON from response
        return self._extract_json_any(raw_text)

    def _find_balanced(self, text: str, open_ch: str, close_ch: str) -> Optional[str]:
        """Find balanced brackets/braces and return the matched substring."""
        start = text.find(open_ch)
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _extract_json_any(self, text: str):
        """
        Extract JSON from Claude's response - handles both objects and arrays.
        Returns parsed JSON (dict or list) or None if parsing fails.
        """
        text = (text or "").strip()

        # Try markdown code block: ```json [...] ``` or ```json {...} ```
        code_block_match = re.search(
            r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```", text, re.DOTALL
        )
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try balanced array first, then object
        for open_ch, close_ch in [("[", "]"), ("{", "}")]:
            matched = self._find_balanced(text, open_ch, close_ch)
            if matched:
                try:
                    return json.loads(matched)
                except json.JSONDecodeError:
                    pass

        # Try entire response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Log failed parse for debugging
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"  [DEBUG] JSON parse failed. Raw response preview: {preview!r}")
            return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON object from Claude's response.
        Returns dict with "status" key or fallback AMBIGUOUS result.
        """
        parsed = self._extract_json_any(text)

        # Validate that parsed dict has required "status" key
        if isinstance(parsed, dict) and "status" in parsed:
            return parsed

        # Fallback: return AMBIGUOUS result
        return {
            "status": "AMBIGUOUS",
            "confidence": 0.0,
            "evidence": "",
            "reason": f"Failed to parse JSON from response: {(text or '')[:200]}",
            "corrected_long_form": None,
        }


__all__ = ["ClaudeClient"]
