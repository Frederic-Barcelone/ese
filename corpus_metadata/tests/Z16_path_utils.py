# corpus_metadata/tests/test_utils/test_path_utils.py
"""Tests for Z_utils/Z05_path_utils.py."""

import os
from pathlib import Path
from unittest import mock


from Z_utils.Z05_path_utils import get_base_path


class TestGetBasePath:
    """Tests for get_base_path function."""

    def test_auto_detect_without_config(self):
        """Test auto-detection when no config or env var is provided."""
        # Clear any env var that might be set
        with mock.patch.dict(os.environ, {}, clear=True):
            if "CORPUS_BASE_PATH" in os.environ:
                del os.environ["CORPUS_BASE_PATH"]

            path = get_base_path()

            # Should return a valid path
            assert isinstance(path, Path)
            # Should end with 'ese' (the project root)
            assert path.name == "ese"

    def test_env_var_takes_precedence(self):
        """Test that CORPUS_BASE_PATH env var takes precedence."""
        test_path = "/tmp/test_corpus"
        with mock.patch.dict(os.environ, {"CORPUS_BASE_PATH": test_path}):
            path = get_base_path()

            assert path == Path(test_path)

    def test_config_value_used_when_no_env_var(self):
        """Test config value is used when env var is not set."""
        config = {"paths": {"base": "/custom/path"}}

        with mock.patch.dict(os.environ, {}, clear=True):
            if "CORPUS_BASE_PATH" in os.environ:
                del os.environ["CORPUS_BASE_PATH"]

            path = get_base_path(config)

            assert path == Path("/custom/path")

    def test_null_config_triggers_auto_detect(self):
        """Test that null config value triggers auto-detection."""
        config = {"paths": {"base": None}}

        with mock.patch.dict(os.environ, {}, clear=True):
            if "CORPUS_BASE_PATH" in os.environ:
                del os.environ["CORPUS_BASE_PATH"]

            path = get_base_path(config)

            # Should auto-detect
            assert isinstance(path, Path)
            assert path.exists() or path.name == "ese"

    def test_empty_string_config_triggers_auto_detect(self):
        """Test that empty string config value triggers auto-detection."""
        config = {"paths": {"base": ""}}

        with mock.patch.dict(os.environ, {}, clear=True):
            if "CORPUS_BASE_PATH" in os.environ:
                del os.environ["CORPUS_BASE_PATH"]

            path = get_base_path(config)

            # Should auto-detect since empty string is falsy
            assert isinstance(path, Path)

    def test_env_var_overrides_config(self):
        """Test that env var takes precedence over config."""
        config = {"paths": {"base": "/config/path"}}
        env_path = "/env/path"

        with mock.patch.dict(os.environ, {"CORPUS_BASE_PATH": env_path}):
            path = get_base_path(config)

            assert path == Path(env_path)

    def test_whitespace_env_var_ignored(self):
        """Test that whitespace-only env var is ignored."""
        config = {"paths": {"base": "/config/path"}}

        with mock.patch.dict(os.environ, {"CORPUS_BASE_PATH": "   "}):
            path = get_base_path(config)

            # Should use config since env var is just whitespace
            assert path == Path("/config/path")
