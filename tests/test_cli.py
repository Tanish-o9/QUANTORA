"""tests/test_cli.py - Fast in-process tests for Quantora's CLI commands."""

import sys
from cli import main


def test_cli_hardware(monkeypatch, capsys):
    """Verify `hardware` subcommand executes and prints telemetry."""
    monkeypatch.setattr(sys, "argv", ["cli.py", "hardware"])
    main()
    captured = capsys.readouterr()
    assert "TELEMETRY" in captured.out
    assert "cuda_available" in captured.out


def test_cli_history(monkeypatch, capsys):
    """Verify `history` subcommand executes cleanly."""
    monkeypatch.setattr(sys, "argv", ["cli.py", "history"])
    main()
    # Should not raise exception
    assert True


def test_cli_recommend(monkeypatch, capsys):
    """Verify `recommend` subcommand executes with budget constraints."""
    monkeypatch.setattr(sys, "argv", ["cli.py", "recommend", "--budget", "2000", "--latency", "5.0"])
    main()
    captured = capsys.readouterr()
    assert "OPTIMAL CONFIGURATION" in captured.out or "No configuration" in captured.out


def test_cli_export(monkeypatch, capsys):
    """Verify `export` subcommand exports CSV and JSON."""
    monkeypatch.setattr(sys, "argv", ["cli.py", "export"])
    main()
    captured = capsys.readouterr()
    assert "[Export]" in captured.out


def test_cli_report(monkeypatch, capsys, tmp_path):
    """Verify `report` subcommand generates standalone HTML report."""
    out_file = str(tmp_path / "cli_report.html")
    monkeypatch.setattr(sys, "argv", ["cli.py", "report", "--output", out_file])
    main()
    captured = capsys.readouterr()
    assert "[Report]" in captured.out
