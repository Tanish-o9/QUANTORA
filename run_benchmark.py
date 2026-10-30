import json
import time
import torch
from model_engine import get_model_engine
from benchmark_engine import get_benchmark_engine
from optimizer import (
    get_hardware_capabilities,
    validate_configuration,
    run_kv_cache_comparison,
    calculate_kv_cache_size_bytes,
    format_bytes_to_mb,
)
from evaluator import get_evaluator
from storage import save_experiment

def main():
    print("=== QUANTORA HARDWARE DETECTION ===")
    hw = get_hardware_capabilities()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTarget Model: {model_id} on {device.upper()}")

    # Test prompts
    prompt_obj = {
        "id": "sum-01",
        "category": "Summarization",
        "prompt": "Summarize the following passage in two concise sentences:\n\nArtificial intelligence has revolutionized numerous industries over the past decade, from healthcare diagnostics to financial fraud detection. Machine learning models, particularly deep neural networks and large language models, analyze massive datasets to discover intricate patterns that humans might miss. However, the high computational demands and massive memory requirements of these systems pose significant challenges for real-world deployment on edge devices and energy-constrained servers. Engineers are actively developing quantization, pruning, and low-rank adaptation techniques to make these models faster, lighter, and more environmentally sustainable.",
        "reference": "Artificial intelligence and deep neural networks have transformed industries by uncovering complex patterns in vast datasets. However, their high compute and memory requirements necessitate optimization techniques like quantization and pruning for efficient deployment.",
        "expected_max_tokens": 80
    }

    precisions = ["fp32", "fp16"]
    if hw.get("bitsandbytes_available") and device == "cuda":
        precisions.extend(["int8", "int4"])

    results_table = []
    engine = get_model_engine()
    bench = get_benchmark_engine()

    for prec in precisions:
        is_val, reason = validate_configuration(prec, device)
        if not is_val:
            print(f"Skipping {prec}: {reason}")
            continue

        print(f"\n==========================================")
        print(f"--> Benchmarking Configuration: {prec.upper()} on {device.upper()}")
        print(f"==========================================")

        # 1. Load Model
        t0 = time.perf_counter()
        info = engine.load_model(model_id=model_id, precision=prec, device=device)
        load_time = round(info.get("load_time_seconds", time.perf_counter() - t0), 3)
        footprint = info.get("memory_footprint_mb", 0.0)
        param_count = info.get("parameter_count_million", 0.0)

        print(f"Loaded {model_id} ({prec}): Load Time = {load_time}s | Weight Footprint = {footprint:.1f} MB | Params = {param_count:.1f}M")

        # 2. Benchmark Inference
        bench_res = bench.run_benchmark(
            prompt=prompt_obj["prompt"],
            prompt_id=prompt_obj["id"],
            reference_answer=prompt_obj["reference"],
            max_new_tokens=prompt_obj["expected_max_tokens"],
            temperature=0.0,
            do_sample=False,
            num_trials=3,
            warmup_runs=1,
            use_cache=True,
        )

        avg_latency = bench_res["avg_latency_s"]
        ttft = bench_res.get("ttft_s") or 0.0
        tpot_s = bench_res.get("tpot_s") or 0.0
        tpot_ms = tpot_s * 1000.0
        throughput = bench_res["throughput_tokens_sec"]
        peak_gpu = bench_res.get("peak_gpu_allocated_mb") or 0.0
        peak_cpu = bench_res.get("peak_cpu_ram_mb") or 0.0
        output_sample = bench_res.get("generated_text", "")
        rouge1 = bench_res.get("rouge1") or 0.0
        rouge2 = bench_res.get("rouge2") or 0.0
        rougeL = bench_res.get("rougeL") or 0.0

        print(f"Inference Results ({prec.upper()}):")
        print(f"  TTFT: {ttft:.4f} s")
        print(f"  TPOT: {tpot_ms:.2f} ms/token")
        print(f"  Avg Latency: {avg_latency:.4f} s")
        print(f"  Throughput: {throughput:.2f} tok/s")
        print(f"  Peak GPU VRAM: {peak_gpu:.1f} MB")
        print(f"Quality Scores (ROUGE):")
        print(f"  ROUGE-1: {rouge1:.2f}%")
        print(f"  ROUGE-2: {rouge2:.2f}%")
        print(f"  ROUGE-L: {rougeL:.2f}%")
        print(f"Sample Output:\n{output_sample.strip()[:200]}...")

        # Save record
        record = {
            "model_id": model_id,
            "precision": prec,
            "device": device,
            "load_time_s": load_time,
            "memory_footprint_mb": footprint,
            "avg_latency_s": avg_latency,
            "ttft_s": ttft,
            "tpot_ms": tpot_ms,
            "throughput_tokens_sec": throughput,
            "peak_gpu_memory_mb": peak_gpu,
            "peak_cpu_memory_mb": peak_cpu,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "output_sample": output_sample
        }
        results_table.append(record)

    # 4. KV Cache Analysis on FP16
    print(f"\n==========================================")
    print(f"--> Benchmarking KV Cache Impact (FP16)")
    print(f"==========================================")
    if "fp16" in [r["precision"] for r in results_table]:
        engine.load_model(model_id=model_id, precision="fp16", device=device)
        kv_res = run_kv_cache_comparison(
            model=engine.model,
            tokenizer=engine.tokenizer,
            prompt=prompt_obj["prompt"],
            max_new_tokens=64,
            device=device,
            num_trials=2
        )
        enabled_data = kv_res["kv_cache_enabled"]
        disabled_data = kv_res["kv_cache_disabled"]
        print("KV Cache Results:")
        print(f"  With Cache Latency: {enabled_data['median_latency_s']:.3f} s | Memory: {enabled_data['total_peak_memory_mb']:.1f} MB | Throughput: {enabled_data['throughput_tokens_sec']:.2f} tok/s")
        print(f"  Without Cache Latency: {disabled_data['median_latency_s']:.3f} s | Memory: {disabled_data['total_peak_memory_mb']:.1f} MB | Throughput: {disabled_data['throughput_tokens_sec']:.2f} tok/s")
        print(f"  Speedup Factor: {kv_res['speedup_factor']:.2f}x")
    else:
        kv_res = None

    # Write summary to json
    summary_data = {
        "hardware": hw,
        "results": results_table,
        "kv_cache_comparison": kv_res
    }
    with open("results/real_test_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n[OK] Benchmark testing completed successfully and saved to results/real_test_benchmark.json")

if __name__ == "__main__":
    main()
