"""tests/test_cli.py - Smoke tests for Quantora's headless CLI commands."""

import subprocess
import sys


def test_cli_hardware():
    """Verify `python cli.py hardware` executes with exit code 0."""
    res = subprocess.run(
        [sys.executable, "cli.py", "hardware"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "TELEMETRY" in res.stdout
    assert "cuda_available" in res.stdout


def test_cli_history():
    """Verify `python cli.py history` executes with exit code 0."""
    res = subprocess.run(
        [sys.executable, "cli.py", "history"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0


def test_cli_recommend():
    """Verify `python cli.py recommend` executes with exit code 0."""
    res = subprocess.run(
        [sys.executable, "cli.py", "recommend", "--budget", "2000", "--latency", "5.0"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0

