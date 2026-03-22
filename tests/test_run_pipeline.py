from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_pipeline import (
    ALL_TAGS,
    DEFAULT_STEPS,
    build_step_cmd,
    parse_args,
    run_combined,
    run_single,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["run_pipeline.py"])
        args = parse_args()
        assert args.tag == "tokyo"
        assert args.steps is None
        assert args.combined is False
        assert args.top_n == 20
        assert args.folds == 5
        assert args.dry_run is False

    def test_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", [
            "run_pipeline.py",
            "--tag", "osaka",
            "--steps", "fetch", "train",
            "--top-n", "10",
            "--folds", "3",
            "--dry-run",
        ])
        args = parse_args()
        assert args.tag == "osaka"
        assert args.steps == ["fetch", "train"]
        assert args.top_n == 10
        assert args.folds == 3
        assert args.dry_run is True

    def test_combined_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--combined"])
        args = parse_args()
        assert args.combined is True


# ---------------------------------------------------------------------------
# build_step_cmd
# ---------------------------------------------------------------------------
class TestBuildStepCmd:
    @pytest.fixture()
    def python(self) -> str:
        return sys.executable

    def test_fetch(self, python: str) -> None:
        cmd = build_step_cmd("fetch", "tokyo", top_n=20, folds=5)
        assert cmd == [python, "scripts/fetch_external.py", "--tag", "tokyo"]

    def test_integrate(self, python: str) -> None:
        cmd = build_step_cmd("integrate", "osaka", top_n=20, folds=5)
        assert cmd == [python, "scripts/integrate_estat.py", "--tag", "osaka", "--skip-fetch"]

    def test_train(self, python: str) -> None:
        cmd = build_step_cmd("train", "nagoya", top_n=15, folds=3)
        assert cmd == [
            python, "scripts/train_model.py", "--tag", "nagoya",
            "--top-n", "15", "--folds", "3", "--two-stage",
        ]

    def test_report(self, python: str) -> None:
        cmd = build_step_cmd("report", "fukuoka", top_n=10, folds=5)
        assert cmd == [python, "scripts/generate_report.py", "--tag", "fukuoka", "--top-n", "10"]

    def test_validate(self, python: str) -> None:
        cmd = build_step_cmd("validate", "tokyo", top_n=20, folds=5)
        assert cmd == [python, "scripts/validate_scores.py", "--tag", "tokyo", "--top-n", "20"]

    def test_unknown_step_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown step"):
            build_step_cmd("nonexistent", "tokyo", top_n=20, folds=5)


# ---------------------------------------------------------------------------
# run_single (dry_run=True)
# ---------------------------------------------------------------------------
class TestRunSingle:
    @patch("scripts.run_pipeline.subprocess.run")
    def test_dry_run_does_not_call_subprocess(self, mock_run: MagicMock) -> None:
        run_single("tokyo", ["fetch", "integrate"], top_n=20, folds=5, dry_run=True)
        mock_run.assert_not_called()

    @patch("scripts.run_pipeline.subprocess.run")
    def test_dry_run_all_default_steps(self, mock_run: MagicMock) -> None:
        run_single("osaka", DEFAULT_STEPS, top_n=20, folds=5, dry_run=True)
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# run_combined (dry_run=True)
# ---------------------------------------------------------------------------
class TestRunCombined:
    @patch("scripts.run_pipeline.subprocess.run")
    def test_dry_run_does_not_call_subprocess(self, mock_run: MagicMock) -> None:
        """dry_run=True should skip all subprocess calls regardless of file existence."""
        # Mock Path.exists so that raw csvs exist and processed csvs exist
        with patch("scripts.run_pipeline.Path.exists", return_value=True):
            run_combined(top_n=20, folds=5, dry_run=True)
        mock_run.assert_not_called()

    @patch("scripts.run_pipeline.subprocess.run")
    def test_skips_areas_without_hotpepper_data(self, mock_run: MagicMock) -> None:
        """Areas whose raw csv does not exist should be skipped entirely."""
        # All Path.exists calls return False -> no area has data
        with patch("scripts.run_pipeline.Path.exists", return_value=False):
            run_combined(top_n=20, folds=5, dry_run=True)
        mock_run.assert_not_called()
