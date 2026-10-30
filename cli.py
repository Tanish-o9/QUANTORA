"""cli.py - Command-Line Interface for LLM Efficiency Lab."""

import argparse
import json
import sys
import time

from optimizer import (
    get_hardware_capabilities,
    run_kv_cache_comparison,
    validate_configuration,
)
from storage import export_results_to_csv_and_json, load_experiments, recommend_best_configuration


def cmd_hardware(args):
    hw = get_hardware_capabilities()
    print("=" * 60)
    print("HARDWARE & LIBRARY TELEMETRY")
    print("=" * 60)
    for k, v in hw.items():
        print(f"  {k:25}: {v}")


def cmd_benchmark(args):
    from benchmark_engine import get_benchmark_engine
    from model_engine import get_model_engine

    hw = get_hardware_capabilities()
    device = "cuda" if hw["cuda_available"] and not args.cpu else "cpu"
    val, msg = validate_configuration(args.precision, device)
    if not val:
        print(f"[Error] Invalid configuration: {msg}")
        sys.exit(1)

    print(f"\n[1/2] Loading '{args.model}' ({args.precision.upper()}) on {device.upper()}...")
    engine = get_model_engine()
    t0 = time.perf_counter()
    info = engine.load_model(model_id=args.model, precision=args.precision, device=device)
    print(f"  Loaded in {info['load_time_seconds']}s | Params: {info['parameter_count_million']}M | Footprint: {info['memory_footprint_mb']} MB")

    print(f"\n[2/2] Running benchmark ({args.warmup} warmup, {args.trials} trials, max {args.max_tokens} tokens)...")
    bench = get_benchmark_engine()
    res = bench.run_benchmark(
        prompt=args.prompt,
        prompt_id=args.prompt_id,
        reference_answer=args.reference,
        max_new_tokens=args.max_tokens,
        warmup_runs=args.warmup,
        num_trials=args.trials,
    )

    print("=" * 60)
    print(f"BENCHMARK RESULTS (Experiment: {res['experiment_id']})")
    print("=" * 60)
    print(f"  Average Latency      : {res['avg_latency_s']} s")
    print(f"  Median Latency       : {res['median_latency_s']} s")
    print(f"  P95 Latency          : {res['p95_latency_s']} s")
    print(f"  TTFT                 : {res['ttft_s']} s" if res['ttft_s'] else "  TTFT                 : N/A")
    print(f"  TPOT                 : {res['tpot_s'] * 1000:.1f} ms" if res['tpot_s'] else "  TPOT                 : N/A")
    print(f"  Decode Throughput    : {res['throughput_tokens_sec']} tok/s")
    print(f"  Total Throughput     : {res['total_throughput_tokens_sec']} tok/s")
    if res['peak_gpu_allocated_mb']:
        print(f"  Peak GPU VRAM        : {res['peak_gpu_allocated_mb']} MB")
    print(f"  Peak CPU RAM         : {res['peak_cpu_ram_mb']} MB")
    if res.get('rougeL'):
        print(f"  ROUGE-L Score        : {res['rougeL']}%")
    print("\nGenerated Text Sample:")
    print(f"  {res['generated_text'][:160]}...")


def cmd_history(args):
    df = load_experiments()
    if df.empty:
        print("No past experiments recorded.")
        return
    cols = ["experiment_id", "model_id", "precision", "avg_latency_s", "throughput_tokens_sec", "peak_gpu_allocated_mb"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False))


def cmd_export(args):
    csv_p, json_p = export_results_to_csv_and_json()
    print(f"[Export] Saved CSV : {csv_p}")
    print(f"[Export] Saved JSON: {json_p}")


def cmd_recommend(args):
    rec = recommend_best_configuration(max_memory_mb=args.budget, max_latency_s=args.latency)
    if rec:
        print("=" * 60)
        print("OPTIMAL CONFIGURATION RECOMMENDED")
        print("=" * 60)
        print(f"  Model       : {rec['model_id']}")
        print(f"  Precision   : {rec['precision'].upper()}")
        print(f"  Latency     : {rec['avg_latency_s']} s")
        print(f"  Throughput  : {rec['throughput_tokens_sec']} tok/s")
        print(f"  Peak VRAM   : {rec['peak_gpu_allocated_mb']} MB")
    else:
        print("No configuration in history satisfies the provided constraints.")


def main():
    parser = argparse.ArgumentParser(description="LLM Efficiency Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Hardware
    subparsers.add_parser("hardware", help="Display hardware & library capabilities")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark experiment")
    p_bench.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model ID")
    p_bench.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "int8", "int4"])
    p_bench.add_argument("--prompt", default="Summarize the computational advantages of integer quantization.", help="Prompt text")
    p_bench.add_argument("--prompt-id", default="cli_prompt")
    p_bench.add_argument("--reference", default=None, help="Ground truth reference for ROUGE")
    p_bench.add_argument("--max-tokens", type=int, default=64)
    p_bench.add_argument("--warmup", type=int, default=1)
    p_bench.add_argument("--trials", type=int, default=3)
    p_bench.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p_bench.set_defaults(func=cmd_benchmark)

    # History
    p_hist = subparsers.add_parser("history", help="List past experiment records")
    p_hist.set_defaults(func=cmd_history)

    # Export
    p_exp = subparsers.add_parser("export", help="Export benchmark history to CSV and JSON")
    p_exp.set_defaults(func=cmd_export)

    # Recommend
    p_rec = subparsers.add_parser("recommend", help="Find Pareto-optimal configuration")
    p_rec.add_argument("--budget", type=float, default=2000.0, help="Max VRAM budget in MB")
    p_rec.add_argument("--latency", type=float, default=3.0, help="Max acceptable latency in seconds")
    p_rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    elif args.command == "hardware":
        cmd_hardware(args)


if __name__ == "__main__":
    main()

