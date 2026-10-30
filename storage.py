"""storage.py - SQLite persistence, result queries, CSV/JSON export, and optimization recommendation."""

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DB_PATH = os.path.join("results", "experiments.db")


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite database schema for LLM Efficiency Lab experiments."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model_id TEXT NOT NULL,
            precision TEXT NOT NULL,
            device TEXT NOT NULL,
            prompt_id TEXT,
            prompt_text TEXT,
            max_new_tokens INTEGER,
            num_trials INTEGER,
            warmup_runs INTEGER,
            avg_latency_s REAL,
            median_latency_s REAL,
            p95_latency_s REAL,
            ttft_s REAL,
            tpot_s REAL,
            throughput_tokens_sec REAL,
            total_throughput_tokens_sec REAL,
            peak_gpu_allocated_mb REAL,
            peak_gpu_reserved_mb REAL,
            peak_cpu_ram_mb REAL,
            load_time_s REAL,
            total_params INTEGER,
            generated_text TEXT,
            rouge1 REAL,
            rouge2 REAL,
            rougeL REAL,
            avg_joules_per_token REAL,
            cost_per_million_tokens_usd REAL,
            hardware_summary TEXT,
            raw_trials_json TEXT
        )
        """
    )

    # Safely migrate existing tables if columns are missing
    cursor.execute("PRAGMA table_info(experiments)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "avg_joules_per_token" not in existing_cols:
        cursor.execute("ALTER TABLE experiments ADD COLUMN avg_joules_per_token REAL")
    if "cost_per_million_tokens_usd" not in existing_cols:
        cursor.execute("ALTER TABLE experiments ADD COLUMN cost_per_million_tokens_usd REAL")

    conn.commit()
    conn.close()


def save_experiment(record: Dict[str, Any], db_path: str = DB_PATH) -> str:
    """Insert a benchmark experiment record into SQLite."""
    init_db(db_path)
    exp_id = record.get("experiment_id") or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    ts = record.get("timestamp") or datetime.now().isoformat()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    raw_trials_str = json.dumps(record.get("raw_trials", []))
    hw_summary_str = json.dumps(record.get("hardware_info", {}))

    cursor.execute(
        """
        INSERT OR REPLACE INTO experiments (
            experiment_id, timestamp, model_id, precision, device,
            prompt_id, prompt_text, max_new_tokens, num_trials, warmup_runs,
            avg_latency_s, median_latency_s, p95_latency_s, ttft_s, tpot_s,
            throughput_tokens_sec, total_throughput_tokens_sec,
            peak_gpu_allocated_mb, peak_gpu_reserved_mb, peak_cpu_ram_mb,
            load_time_s, total_params, generated_text,
            rouge1, rouge2, rougeL, avg_joules_per_token, cost_per_million_tokens_usd, hardware_summary, raw_trials_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            exp_id,
            ts,
            record.get("model_id", "Unknown"),
            record.get("precision", "fp32"),
            record.get("device", "cuda"),
            record.get("prompt_id", "custom"),
            record.get("prompt_text", ""),
            record.get("max_new_tokens", 64),
            record.get("num_trials", 1),
            record.get("warmup_runs", 1),
            record.get("avg_latency_s"),
            record.get("median_latency_s"),
            record.get("p95_latency_s"),
            record.get("ttft_s"),
            record.get("tpot_s"),
            record.get("throughput_tokens_sec"),
            record.get("total_throughput_tokens_sec"),
            record.get("peak_gpu_allocated_mb"),
            record.get("peak_gpu_reserved_mb"),
            record.get("peak_cpu_ram_mb"),
            record.get("load_time_s"),
            record.get("total_params"),
            record.get("generated_text", ""),
            record.get("rouge1"),
            record.get("rouge2"),
            record.get("rougeL"),
            record.get("avg_joules_per_token"),
            record.get("cost_per_million_tokens_usd"),
            hw_summary_str,
            raw_trials_str,
        ),
    )
    conn.commit()
    conn.close()
    return exp_id


def load_experiments(
    model_id: Optional[str] = None,
    precision: Optional[str] = None,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Retrieve filtered experiments from SQLite database as a Pandas DataFrame."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM experiments"
    conditions = []
    params = []

    if model_id:
        conditions.append("model_id = ?")
        params.append(model_id)
    if precision:
        conditions.append("precision = ?")
        params.append(precision)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def export_results_to_csv_and_json(
    output_dir: str = "results",
    db_path: str = DB_PATH,
) -> Tuple[str, str]:
    """Export all stored experiments to CSV and JSON files for download and reporting."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_experiments(db_path=db_path)

    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    json_path = os.path.join(output_dir, "benchmark_results.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    return csv_path, json_path


def compute_comparison_metrics(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute relative performance improvements between baseline and optimized runs.

    Formulas:
    - Memory Reduction: ((Baseline - Optimized) / Baseline) * 100
    - Latency Improvement: ((Baseline - Optimized) / Baseline) * 100
    - Throughput Improvement: ((Optimized - Baseline) / Baseline) * 100
    - Quality Delta: Optimized - Baseline
    """
    base_mem = baseline.get("peak_gpu_allocated_mb") or baseline.get("peak_cpu_ram_mb") or 0.0
    opt_mem = optimized.get("peak_gpu_allocated_mb") or optimized.get("peak_cpu_ram_mb") or 0.0

    base_lat = baseline.get("avg_latency_s") or baseline.get("median_latency_s") or 0.0
    opt_lat = optimized.get("avg_latency_s") or optimized.get("median_latency_s") or 0.0

    base_tp = baseline.get("throughput_tokens_sec") or 0.0
    opt_tp = optimized.get("throughput_tokens_sec") or 0.0

    base_q = baseline.get("rougeL") or baseline.get("rouge1") or 0.0
    opt_q = optimized.get("rougeL") or optimized.get("rouge1") or 0.0

    mem_reduction_pct = ((base_mem - opt_mem) / base_mem * 100) if base_mem > 0 else 0.0
    lat_improvement_pct = ((base_lat - opt_lat) / base_lat * 100) if base_lat > 0 else 0.0
    tp_improvement_pct = ((opt_tp - base_tp) / base_tp * 100) if base_tp > 0 else 0.0
    quality_delta = opt_q - base_q

    return {
        "baseline_memory_mb": round(base_mem, 2),
        "optimized_memory_mb": round(opt_mem, 2),
        "memory_reduction_pct": round(mem_reduction_pct, 2),
        "baseline_latency_s": round(base_lat, 4),
        "optimized_latency_s": round(opt_lat, 4),
        "latency_improvement_pct": round(lat_improvement_pct, 2),
        "baseline_throughput_tok_s": round(base_tp, 2),
        "optimized_throughput_tok_s": round(opt_tp, 2),
        "throughput_improvement_pct": round(tp_improvement_pct, 2),
        "baseline_quality": round(base_q, 2),
        "optimized_quality": round(opt_q, 2),
        "quality_delta": round(quality_delta, 2),
    }


def recommend_best_configuration(
    max_memory_mb: Optional[float] = None,
    min_quality_score: Optional[float] = None,
    max_latency_s: Optional[float] = None,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Recommend the optimal measured configuration satisfying user constraints.

    Evaluates past benchmarked experiments in the SQLite store and returns the
    Pareto-optimal configuration with the highest throughput that meets constraints.
    Never recommends an unbenchmarked configuration.
    """
    df = load_experiments(db_path=db_path)
    if df.empty:
        return None

    # Filter by user constraints
    filtered = df.copy()

    if max_memory_mb is not None:
        filtered = filtered[
            (filtered["peak_gpu_allocated_mb"].fillna(999999) <= max_memory_mb)
            | (filtered["peak_cpu_ram_mb"].fillna(999999) <= max_memory_mb)
        ]

    if min_quality_score is not None:
        filtered = filtered[filtered["rougeL"].fillna(0) >= min_quality_score]

    if max_latency_s is not None:
        filtered = filtered[filtered["avg_latency_s"].fillna(999999) <= max_latency_s]

    if filtered.empty:
        return None

    # From matching candidates, choose the one with highest throughput
    best_row = filtered.sort_values(by="throughput_tokens_sec", ascending=False).iloc[0]
    return best_row.to_dict()


def generate_html_report(
    db_path: str = DB_PATH,
    output_path: str = "results/benchmark_report.html",
) -> str:
    """Generate a self-contained, professional executive HTML benchmark report.

    Includes executive summary cards, full performance metrics table,
    hardware telemetry, and print-ready CSS for PDF generation.
    """
    df = load_experiments(db_path=db_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_runs = len(df)
    unique_models = df["model_id"].nunique() if not df.empty and "model_id" in df else 0
    fastest_tok_s = df["throughput_tokens_sec"].max() if not df.empty and "throughput_tokens_sec" in df else 0.0
    lowest_mem_mb = df["peak_gpu_allocated_mb"].min() if not df.empty and "peak_gpu_allocated_mb" in df else 0.0

    # Best overall recommendation
    rec = recommend_best_configuration(db_path=db_path)
    rec_text = "N/A"
    if rec:
        rec_text = f"<strong>{rec.get('model_id')}</strong> ({str(rec.get('precision')).upper()}) - {rec.get('throughput_tokens_sec', 0.0):.1f} tok/s | {rec.get('avg_latency_s', 0.0):.2f}s latency"

    rows_html = ""
    if not df.empty:
        for _, row in df.iterrows():
            ts = str(row.get("timestamp", ""))[:19]
            mod = str(row.get("model_id", ""))
            prec = str(row.get("precision", "")).upper()
            dev = str(row.get("device", "")).upper()

            lat_val = row.get("avg_latency_s")
            lat = f"{float(lat_val):.3f}s" if pd.notna(lat_val) and lat_val is not None else "-"

            ttft_val = row.get("ttft_s")
            ttft = f"{float(ttft_val):.3f}s" if pd.notna(ttft_val) and ttft_val is not None else "-"

            tp_val = row.get("throughput_tokens_sec")
            tp = f"{float(tp_val):.1f} tok/s" if pd.notna(tp_val) and tp_val is not None else "-"

            gpu_mem = row.get("peak_gpu_allocated_mb")
            cpu_mem = row.get("peak_cpu_ram_mb")
            if pd.notna(gpu_mem) and gpu_mem is not None:
                mem = f"{float(gpu_mem):.1f} MB"
            elif pd.notna(cpu_mem) and cpu_mem is not None:
                mem = f"{float(cpu_mem):.1f} MB (CPU)"
            else:
                mem = "-"

            rL_val = row.get("rougeL")
            rL = f"{float(rL_val):.1f}%" if pd.notna(rL_val) and rL_val is not None else "-"

            rows_html += f"""
            <tr>
              <td>{ts}</td>
              <td><code>{mod}</code></td>
              <td><span class="badge">{prec}</span></td>
              <td>{dev}</td>
              <td>{lat}</td>
              <td>{ttft}</td>
              <td><strong>{tp}</strong></td>
              <td>{mem}</td>
              <td>{rL}</td>
            </tr>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantora — Executive Benchmark Report</title>
<style>
  :root {{
    --primary: #4f46e5;
    --primary-dark: #3730a3;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 2.5rem 1.5rem;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    background: var(--card);
    padding: 2.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
    border: 1px solid var(--border);
  }}
  .header {{
    border-bottom: 2px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .logo {{
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--primary);
  }}
  .subhead {{
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.25rem;
  }}
  .cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .stat-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
  }}
  .stat-card .label {{
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .stat-card .val {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.4rem;
  }}
  .recommendation-box {{
    background: #eef2ff;
    border-left: 4px solid var(--primary);
    padding: 1rem 1.25rem;
    border-radius: 4px;
    margin-bottom: 2rem;
    font-size: 0.95rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.75rem 0.9rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: var(--bg);
    color: var(--muted);
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
  }}
  .badge {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 4px;
    background: #e0e7ff;
    color: var(--primary-dark);
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85rem;
  }}
  .footer {{
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
  }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .container {{ box-shadow: none; border: none; padding: 0; width: 100%; max-width: 100%; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div class="logo">⚡ QUANTORA</div>
      <div class="subhead">LLM Performance & Optimization Engineering Platform — Executive Report</div>
    </div>
    <div style="text-align: right; color: var(--muted); font-size: 0.85rem;">
      Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
    </div>
  </div>

  <div class="cards-grid">
    <div class="stat-card">
      <div class="label">Total Benchmarks</div>
      <div class="val">{total_runs}</div>
    </div>
    <div class="stat-card">
      <div class="label">Models Evaluated</div>
      <div class="val">{unique_models}</div>
    </div>
    <div class="stat-card">
      <div class="label">Peak Throughput</div>
      <div class="val">{fastest_tok_s:.1f} <span style="font-size: 0.9rem; color: var(--muted);">tok/s</span></div>
    </div>
    <div class="stat-card">
      <div class="label">Min VRAM Footprint</div>
      <div class="val">{lowest_mem_mb:.1f} <span style="font-size: 0.9rem; color: var(--muted);">MB</span></div>
    </div>
  </div>

  <div class="recommendation-box">
    <strong>💡 Pareto-Optimal Recommendation:</strong> {rec_text}
  </div>

  <h3 style="margin-top: 2rem; margin-bottom: 0.5rem;">Benchmark Run History</h3>
  <div style="overflow-x: auto;">
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Model ID</th>
          <th>Precision</th>
          <th>Device</th>
          <th>Avg Latency</th>
          <th>TTFT</th>
          <th>Throughput</th>
          <th>Peak VRAM</th>
          <th>ROUGE-L</th>
        </tr>
      </thead>
      <tbody>
        {rows_html if rows_html else '<tr><td colspan="9" style="text-align: center; color: var(--muted);">No benchmark experiments recorded yet.</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="footer">
    <div>Developed & Maintained by <a href="https://github.com/Tanish-o9" style="color: var(--primary); text-decoration: none; font-weight: 600;">Tanish Kumar (@Tanish-o9)</a></div>
    <div>Quantora ML Systems Lab &bull; <a href="https://github.com/Tanish-o9/QUANTORA" style="color: var(--muted);">GitHub</a></div>
  </div>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


