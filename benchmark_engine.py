"""benchmark_engine.py - Reproducible LLM performance benchmarking with precision timing and memory profiling."""

import gc
import json
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import torch

from evaluator import get_evaluator
from model_engine import ModelEngine, get_model_engine
from optimizer import get_hardware_capabilities
from storage import save_experiment


class BenchmarkEngine:
    """Orchestrates controlled, reproducible LLM benchmarking trials.

    Guarantees:
    - Dedicated warm-up runs excluded from reported metrics
    - Hardware synchronization before and after timing barriers
    - Reset of peak memory counters at every trial boundary
    - Real token-by-token capture of TTFT and TPOT
    - Transparent trial-level granularity
    """

    def __init__(self, model_engine: Optional[ModelEngine] = None):
        self.model_engine = model_engine or get_model_engine()
        self.evaluator = get_evaluator()

    def run_benchmark(
        self,
        prompt: str,
        prompt_id: str = "custom",
        reference_answer: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        warmup_runs: int = 1,
        num_trials: int = 3,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a full benchmark session across warm-up and measured trials."""
        if not self.model_engine.is_loaded():
            raise RuntimeError("Model is not loaded. Call model_engine.load_model() before benchmarking.")

        is_cuda = self.model_engine.device == "cuda" and torch.cuda.is_available()
        hw_info = get_hardware_capabilities()
        model_info = self.model_engine.get_model_info()

        total_steps = warmup_runs + num_trials
        current_step = 0

        # -------------------------------------------------------------------
        # 1. Warm-Up Runs (Excluded from measured results)
        # -------------------------------------------------------------------
        for w in range(warmup_runs):
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, f"Warm-up run {w + 1}/{warmup_runs}...")

            # Run a small generation to warm up kernels, JIT, and memory allocators
            _ = self.model_engine.generate(
                prompt=prompt,
                max_new_tokens=min(16, max_new_tokens),
                temperature=temperature,
                do_sample=do_sample,
                use_cache=use_cache,
            )
            if is_cuda:
                torch.cuda.synchronize()

        # -------------------------------------------------------------------
        # 2. Measured Trials
        # -------------------------------------------------------------------
        trials_data: List[Dict[str, Any]] = []
        last_generated_text = ""

        for trial_idx in range(num_trials):
            current_step += 1
            if progress_callback:
                progress_callback(
                    current_step,
                    total_steps,
                    f"Measured trial {trial_idx + 1}/{num_trials}...",
                )

            # Pre-trial cleanup and memory baseline
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                gpu_mem_baseline = torch.cuda.memory_allocated()
            else:
                gpu_mem_baseline = 0

            proc = psutil.Process()
            cpu_mem_baseline = proc.memory_info().rss

            # Timing boundary
            if is_cuda:
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            gen_result = self.model_engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                use_cache=use_cache,
            )

            if is_cuda:
                torch.cuda.synchronize()
                peak_gpu_alloc = torch.cuda.max_memory_allocated()
                peak_gpu_res = torch.cuda.max_memory_reserved()
            else:
                peak_gpu_alloc = None
                peak_gpu_res = None

            t_end = time.perf_counter()
            wall_latency = t_end - t_start
            cpu_mem_peak = proc.memory_info().rss

            num_new_tokens = gen_result["generated_tokens"]
            prompt_tokens = gen_result["prompt_tokens"]
            total_tokens = gen_result["total_tokens"]

            decode_throughput = (num_new_tokens / wall_latency) if wall_latency > 0 else 0.0
            total_throughput = (total_tokens / wall_latency) if wall_latency > 0 else 0.0

            trial_record = {
                "trial_index": trial_idx + 1,
                "latency_seconds": round(wall_latency, 4),
                "ttft_seconds": gen_result.get("ttft_seconds"),
                "tpot_seconds": gen_result.get("tpot_seconds"),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": num_new_tokens,
                "total_tokens": total_tokens,
                "throughput_tokens_per_sec": round(decode_throughput, 2),
                "total_throughput_tokens_per_sec": round(total_throughput, 2),
                "peak_gpu_allocated_mb": round(peak_gpu_alloc / (1024 ** 2), 2) if peak_gpu_alloc is not None else None,
                "peak_gpu_reserved_mb": round(peak_gpu_res / (1024 ** 2), 2) if peak_gpu_res is not None else None,
                "peak_cpu_ram_mb": round(cpu_mem_peak / (1024 ** 2), 2),
            }
            trials_data.append(trial_record)
            last_generated_text = gen_result["generated_text"]

        # -------------------------------------------------------------------
        # 3. Aggregate Statistical Calculations
        # -------------------------------------------------------------------
        latencies = [t["latency_seconds"] for t in trials_data]
        throughputs = [t["throughput_tokens_per_sec"] for t in trials_data]
        total_throughputs = [t["total_throughput_tokens_per_sec"] for t in trials_data]

        ttfts = [t["ttft_seconds"] for t in trials_data if t["ttft_seconds"] is not None]
        tpots = [t["tpot_seconds"] for t in trials_data if t["tpot_seconds"] is not None]

        gpu_allocs = [t["peak_gpu_allocated_mb"] for t in trials_data if t["peak_gpu_allocated_mb"] is not None]
        gpu_res = [t["peak_gpu_reserved_mb"] for t in trials_data if t["peak_gpu_reserved_mb"] is not None]
        cpu_rams = [t["peak_cpu_ram_mb"] for t in trials_data]

        avg_latency = float(np.mean(latencies))
        median_latency = float(np.median(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        avg_throughput = float(np.mean(throughputs))
        avg_total_throughput = float(np.mean(total_throughputs))

        avg_ttft = float(np.mean(ttfts)) if ttfts else None
        avg_tpot = float(np.mean(tpots)) if tpots else None

        avg_gpu_alloc = float(np.mean(gpu_allocs)) if gpu_allocs else None
        avg_gpu_res = float(np.mean(gpu_res)) if gpu_res else None
        avg_cpu_ram = float(np.mean(cpu_rams))

        # -------------------------------------------------------------------
        # 4. Quality Evaluation (ROUGE if reference provided or text exists)
        # -------------------------------------------------------------------
        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        if reference_answer and reference_answer.strip():
            rouge_scores = self.evaluator.compute_rouge(last_generated_text, reference_answer)

        # -------------------------------------------------------------------
        # 5. Build Final Benchmark Record
        # -------------------------------------------------------------------
        benchmark_result = {
            "model_id": model_info["model_id"],
            "precision": model_info["precision"],
            "device": model_info["device"],
            "load_time_s": model_info["load_time_seconds"],
            "total_params": model_info["total_parameters"],
            "prompt_id": prompt_id,
            "prompt_text": prompt,
            "reference_answer": reference_answer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "warmup_runs": warmup_runs,
            "num_trials": num_trials,
            # Aggregate metrics
            "avg_latency_s": round(avg_latency, 4),
            "median_latency_s": round(median_latency, 4),
            "p95_latency_s": round(p95_latency, 4),
            "ttft_s": round(avg_ttft, 4) if avg_ttft is not None else None,
            "tpot_s": round(avg_tpot, 4) if avg_tpot is not None else None,
            "throughput_tokens_sec": round(avg_throughput, 2),
            "total_throughput_tokens_sec": round(avg_total_throughput, 2),
            "peak_gpu_allocated_mb": round(avg_gpu_alloc, 2) if avg_gpu_alloc is not None else None,
            "peak_gpu_reserved_mb": round(avg_gpu_res, 2) if avg_gpu_res is not None else None,
            "peak_cpu_ram_mb": round(avg_cpu_ram, 2),
            "generated_text": last_generated_text,
            "rouge1": rouge_scores.get("rouge1"),
            "rouge2": rouge_scores.get("rouge2"),
            "rougeL": rouge_scores.get("rougeL"),
            "hardware_info": hw_info,
            "raw_trials": trials_data,
        }

        # Auto-persist to SQLite
        try:
            exp_id = save_experiment(benchmark_result)
            benchmark_result["experiment_id"] = exp_id
        except Exception as e:
            benchmark_result["experiment_id"] = f"err_{int(time.time())}"

        return benchmark_result


def run_batched_throughput_comparison(
    model: Any,
    tokenizer: Any,
    prompt: str,
    batch_sizes: Optional[List[int]] = None,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Benchmark model throughput scaling across different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    results = []
    is_cuda = device == "cuda" and torch.cuda.is_available()

    for bs in batch_sizes:
        batch_prompts = [prompt] * bs
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        if is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if is_cuda:
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram = None

        t1 = time.perf_counter()
        lat = t1 - t0

        new_toks = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        total_tokens = new_toks * bs
        throughput = total_tokens / lat if lat > 0 else 0.0

        results.append({
            "batch_size": bs,
            "latency_s": round(lat, 4),
            "per_request_latency_s": round(lat / bs, 4),
            "total_tokens": int(total_tokens),
            "throughput_tok_s": round(throughput, 2),
            "peak_vram_mb": round(peak_vram, 2) if peak_vram else None,
        })

    return results


def run_context_length_scaling(
    model: Any,
    tokenizer: Any,
    base_text: Optional[str] = None,
    context_lengths: Optional[List[int]] = None,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Benchmark prefill latency (TTFT) and memory scaling across varying context lengths.

    Demonstrates quadratic attention compute growth during prefill and linear KV cache
    memory expansion across input token lengths (e.g., 128 to 2048+ tokens).
    """
    if context_lengths is None:
        context_lengths = [128, 256, 512, 1024]

    if not base_text:
        base_text = (
            "Large language models rely on multi-head self-attention mechanisms where each token "
            "attends to all prior tokens in the sequence. During the autoregressive generation phase, "
            "key and value activation vectors are cached to avoid quadratic recomputation. "
            "However, as the context length scales into thousands of tokens, the prefill phase consumes "
            "significant memory bandwidth and compute resources. Efficient architectures utilize "
            "FlashAttention, grouped-query attention, and speculative decoding to maintain interactivity."
        )

    is_cuda = device == "cuda" and torch.cuda.is_available()
    results = []

    # Encode base text tokens
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_tokens:
        base_tokens = [100] * 50

    for target_len in context_lengths:
        # Build prompt of exact target_len
        repeats = (target_len // len(base_tokens)) + 2
        extended_tokens = (base_tokens * repeats)[:target_len]
        prompt_text = tokenizer.decode(extended_tokens)
        inputs = tokenizer(prompt_text, return_tensors="pt")

        if is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        # Warm-up prefill run
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        if is_cuda:
            torch.cuda.synchronize()

        # Measured generation run
        streamer = BenchmarkTokenStreamer()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if is_cuda:
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            inc_mem = peak_vram - start_mem
        else:
            peak_vram = None
            inc_mem = None

        t1 = time.perf_counter()
        total_lat = t1 - t0
        ttft = streamer.get_ttft() or (total_lat / max(1, max_new_tokens))
        tpot = streamer.get_tpot() or (total_lat / max(1, max_new_tokens))
        new_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        throughput = new_tokens / total_lat if total_lat > 0 else 0.0

        results.append({
            "context_length_tokens": int(inputs["input_ids"].shape[-1]),
            "max_new_tokens": int(new_tokens),
            "ttft_s": round(ttft, 4),
            "tpot_ms": round(tpot * 1000.0, 2),
            "total_latency_s": round(total_lat, 4),
            "throughput_tok_s": round(throughput, 2),
            "peak_vram_mb": round(peak_vram, 2) if peak_vram else None,
            "incremental_vram_mb": round(max(0.0, inc_mem), 2) if inc_mem else None,
        })

    return results


def benchmark_speculative_decoding(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmark speculative decoding (Draft/Assistant Model + Target Model) against baseline autoregressive decoding.

    Speculative decoding uses a smaller draft model to propose K candidate tokens,
    which the larger target model verifies in parallel in a single forward pass.
    """
    is_cuda = device == "cuda" and torch.cuda.is_available()
    inputs = tokenizer(prompt, return_tensors="pt")
    if is_cuda:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        torch.cuda.synchronize()

    # 1. Baseline Run: Target Model alone
    t0_base = time.perf_counter()
    with torch.inference_mode():
        base_outputs = target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if is_cuda:
        torch.cuda.synchronize()
    t1_base = time.perf_counter()
    base_lat = t1_base - t0_base
    base_tokens = base_outputs.shape[-1] - inputs["input_ids"].shape[-1]
    base_tp = base_tokens / base_lat if base_lat > 0 else 0.0
    base_text = tokenizer.decode(base_outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # 2. Speculative Run: Target Model + Draft Model (assistant_model)
    t0_spec = time.perf_counter()
    with torch.inference_mode():
        spec_outputs = target_model.generate(
            **inputs,
            assistant_model=draft_model,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if is_cuda:
        torch.cuda.synchronize()
    t1_spec = time.perf_counter()
    spec_lat = t1_spec - t0_spec
    spec_tokens = spec_outputs.shape[-1] - inputs["input_ids"].shape[-1]
    spec_tp = spec_tokens / spec_lat if spec_lat > 0 else 0.0
    spec_text = tokenizer.decode(spec_outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    speedup = round(base_lat / spec_lat, 2) if spec_lat > 0 else 1.0

    # Token match agreement
    base_ids = base_outputs[0].tolist()
    spec_ids = spec_outputs[0].tolist()
    min_len = min(len(base_ids), len(spec_ids))
    matches = sum(1 for i in range(min_len) if base_ids[i] == spec_ids[i])
    match_rate = round(matches / max(1, min_len) * 100.0, 1)

    return {
        "baseline_latency_s": round(base_lat, 4),
        "speculative_latency_s": round(spec_lat, 4),
        "speedup_factor": speedup,
        "baseline_throughput_tok_s": round(base_tp, 2),
        "speculative_throughput_tok_s": round(spec_tp, 2),
        "output_match_rate_pct": match_rate,
        "baseline_text": base_text,
        "speculative_text": spec_text,
        "explanation": (
            f"Speculative decoding achieved a {speedup}x speedup ({spec_lat:.3f}s vs {base_lat:.3f}s) "
            f"with {match_rate}% token equivalence. The draft model generated candidate tokens that the "
            "target model validated concurrently in batched forward passes."
        ),
    }


# Singleton benchmark engine
_BENCHMARK_INSTANCE: Optional[BenchmarkEngine] = None


def get_benchmark_engine() -> BenchmarkEngine:
    """Return singleton BenchmarkEngine instance."""
    global _BENCHMARK_INSTANCE
    if _BENCHMARK_INSTANCE is None:
        _BENCHMARK_INSTANCE = BenchmarkEngine()
    return _BENCHMARK_INSTANCE

