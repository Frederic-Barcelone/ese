# corpus_metadata/tests/Z15_orchestrator_utils.py
"""Tests for orchestrator_utils.py."""

import time
from unittest import mock


from orchestrator_utils import StageTimer, setup_warning_suppression


class TestStageTimer:
    """Tests for StageTimer class."""

    def test_start_and_stop(self):
        """Test basic start/stop timing."""
        timer = StageTimer()

        timer.start("test_stage")
        time.sleep(0.01)  # Small delay
        elapsed = timer.stop("test_stage")

        assert elapsed > 0
        assert elapsed < 1.0  # Should be much less than 1 second

    def test_get_timing(self):
        """Test getting timing for a stage."""
        timer = StageTimer()

        timer.start("stage1")
        time.sleep(0.01)
        timer.stop("stage1")

        elapsed = timer.get("stage1")
        assert elapsed > 0

    def test_get_missing_stage(self):
        """Test getting timing for non-existent stage."""
        timer = StageTimer()

        elapsed = timer.get("nonexistent")
        assert elapsed == 0.0

    def test_stop_without_start(self):
        """Test stopping a stage that was never started."""
        timer = StageTimer()

        elapsed = timer.stop("never_started")
        assert elapsed == 0.0

    def test_total_time(self):
        """Test total time calculation."""
        timer = StageTimer()

        timer.start("stage1")
        time.sleep(0.01)
        timer.stop("stage1")

        timer.start("stage2")
        time.sleep(0.01)
        timer.stop("stage2")

        total = timer.total()
        stage1_time = timer.get("stage1")
        stage2_time = timer.get("stage2")

        assert total > 0
        assert abs(total - (stage1_time + stage2_time)) < 0.001

    def test_total_empty(self):
        """Test total time when no stages recorded."""
        timer = StageTimer()

        assert timer.total() == 0.0

    def test_multiple_stages(self):
        """Test timing multiple stages."""
        timer = StageTimer()

        for i in range(3):
            timer.start(f"stage{i}")
            time.sleep(0.005)
            timer.stop(f"stage{i}")

        assert len(timer.timings) == 3
        for i in range(3):
            assert timer.get(f"stage{i}") > 0

    def test_print_summary_with_timings(self):
        """Test print_summary outputs correct format."""
        timer = StageTimer()

        timer.start("parsing")
        timer.stop("parsing")
        timer.timings["parsing"] = 1.0  # Set explicit value for testing

        timer.start("validation")
        timer.stop("validation")
        timer.timings["validation"] = 2.0

        # Capture print output
        with mock.patch("builtins.print") as mock_print:
            timer.print_summary()

            # Should have been called multiple times
            assert mock_print.call_count >= 5

    def test_print_summary_empty(self):
        """Test print_summary with no timings does nothing."""
        timer = StageTimer()

        with mock.patch("builtins.print") as mock_print:
            timer.print_summary()

            # Should not print anything
            mock_print.assert_not_called()


class TestSetupWarningSuppressions:
    """Tests for setup_warning_suppression function."""

    def test_sets_environment_variables(self):
        """Test that warning suppression sets expected env vars."""
        import os

        # Clear before test
        for key in ["PYTHONWARNINGS", "TRANSFORMERS_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS"]:
            if key in os.environ:
                del os.environ[key]

        setup_warning_suppression()

        # Check env vars are set
        assert "PYTHONWARNINGS" in os.environ
        assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

    def test_can_call_multiple_times(self):
        """Test that setup can be called multiple times without error."""
        # Should not raise
        setup_warning_suppression()
        setup_warning_suppression()
