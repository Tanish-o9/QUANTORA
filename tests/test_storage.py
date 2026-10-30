"""tests/test_storage.py - Unit tests for Quantora's SQLite persistence and recommender."""

import os
import sqlite3
import pytest
import pandas as pd
from storage import (
    init_db,
    save_experiment,
    load_experiments,
    export_results_to_csv_and_json,
    generate_html_report,
    recommend_best_configuration,
)


def test_init_db_and_save_experiment(tmp_path):
    """Verify experiment persistence to SQLite."""
    test_db = str(tmp_path / "test_experiments.db")
    
    # Save a mock benchmark experiment
    sample_exp = {
        "model_id": "test-org/test-model",
        "precision": "fp16",
        "device": "cuda",
        "prompt_id": "test-prompt",
        "prompt_text": "What is LLM quantization?",
        "max_new_tokens": 64,
        "num_trials": 2,
        "warmup_runs": 1,
        "avg_latency_s": 1.25,
        "median_latency_s": 1.24,
        "p95_latency_s": 1.26,
        "ttft_s": 0.05,
        "tpot_s": 0.015,
        "throughput_tokens_sec": 42.5,
        "total_throughput_tokens_sec": 42.5,
        "peak_gpu_allocated_mb": 950.0,
        "peak_gpu_reserved_mb": 1024.0,
        "peak_cpu_ram_mb": 1500.0,
        "load_time_s": 2.1,
        "total_params": 500000000,
        "generated_text": "Quantization reduces memory footprint.",
        "rouge1": 85.5,
        "rouge2": 60.0,
        "rougeL": 80.0,
    }

    exp_id = save_experiment(sample_exp, db_path=test_db)
    assert exp_id.startswith("exp_")

    # Load dataframe back
    df = load_experiments(db_path=test_db)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["model_id"] == "test-org/test-model"
    assert df.iloc[0]["precision"] == "fp16"


def test_export_utilities(tmp_path):
    """Verify CSV and JSON exporting."""
    test_db = str(tmp_path / "test_experiments.db")
    out_dir = str(tmp_path / "exports")

    sample_exp = {
        "model_id": "test-model",
        "precision": "fp32",
        "device": "cpu",
        "avg_latency_s": 3.0,
        "throughput_tokens_sec": 10.0,
    }
    save_experiment(sample_exp, db_path=test_db)

    csv_path, json_path = export_results_to_csv_and_json(output_dir=out_dir, db_path=test_db)
    assert os.path.exists(csv_path)
    assert os.path.exists(json_path)

    # Executive HTML report
    html_file = str(tmp_path / "report.html")
    html_out = generate_html_report(db_path=test_db, output_path=html_file)
    assert os.path.exists(html_out)
    with open(html_out, "r", encoding="utf-8") as f:
        html_str = f.read()
    assert "QUANTORA" in html_str
    assert "Executive Report" in html_str


def test_recommend_optimal_configuration(tmp_path):
    """Verify Pareto frontier recommender logic under budget constraints."""
    test_db = str(tmp_path / "test_experiments.db")

    exps = [
        {"model_id": "m1", "precision": "fp32", "device": "cuda", "peak_gpu_allocated_mb": 2000.0, "avg_latency_s": 4.0, "throughput_tokens_sec": 20.0, "rougeL": 70.0},
        {"model_id": "m1", "precision": "fp16", "device": "cuda", "peak_gpu_allocated_mb": 1000.0, "avg_latency_s": 2.0, "throughput_tokens_sec": 40.0, "rougeL": 70.0},
        {"model_id": "m1", "precision": "int4", "device": "cuda", "peak_gpu_allocated_mb": 500.0, "avg_latency_s": 2.5, "throughput_tokens_sec": 35.0, "rougeL": 65.0},
    ]
    for e in exps:
        save_experiment(e, db_path=test_db)

    # Constraint: max 1200MB memory, max 3.0s latency
    rec = recommend_best_configuration(
        max_memory_mb=1200.0,
        max_latency_s=3.0,
        db_path=test_db
    )
    assert rec is not None
    # Best under constraint should be fp16 (40 tok/s)
    assert rec["precision"] == "fp16"

