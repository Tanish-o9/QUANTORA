# ⚡ QUANTORA
### LLM Performance & Optimization Engineering Platform

**Benchmark. Analyze. Optimize.**

**Quantora** is a production-grade ML systems engineering and benchmarking suite designed to measure the real-world resource footprints, runtime latencies, throughput profiles, and output qualities of open-source Large Language Models (LLMs) across precision levels (FP32, FP16, BF16, INT8, INT4), KV cache policies, and PEFT Low-Rank Adaptation (LoRA) configurations.

All metrics are gathered from **real model execution on local hardware**—never mocked or hardcoded.

---

## 1. Core Architecture

The repository adheres to a strict, human-written, decoupled modular design:

```
QUANTORA/
│
├── app.py                  # Professional Streamlit dashboard & analytics UI
├── cli.py                  # Terminal CLI for automated headless benchmarks & history
├── model_engine.py         # Hugging Face model lifecycle, device placement & streaming inference
├── benchmark_engine.py     # Controlled benchmark trials, CUDA-sync timing & NVML energy telemetry
├── optimizer.py            # Hardware validation, quantization, speculative decoding, SDPA & LoRA
├── evaluator.py            # ROUGE, token F1 & MMLU multi-choice reasoning evaluation
├── storage.py              # SQLite persistence, Pareto recommender & CSV/JSON export
├── requirements.txt        # Minimal, explicit project dependencies
├── LICENSE                 # MIT Open-Source License
├── README.md               # Architecture documentation, setup & methodology
│
├── data/
│   ├── sample_prompts.json # Fixed prompts across Summarization, Reasoning, Code & QA
│   └── reasoning_eval.json # Standardized ML Systems & LLM reasoning benchmark questions
│
└── results/
    ├── experiments.db      # SQLite database storing benchmark runs
    ├── benchmark_results.csv
    └── benchmark_results.json
```

### Module Responsibilities

| File | Primary Responsibility |
| :--- | :--- |
| `model_engine.py` | Loads Hugging Face models and tokenizers; manages GPU VRAM / CPU placement; token-by-token streamer for exact TTFT & TPOT measurement; memory cleanup. |
| `benchmark_engine.py` | Conducts warm-up runs and repeated trials; measures TTFT, TPOT, P95 latency, tokens/sec throughput, and peak memory; aggregates statistics. |
| `optimizer.py` | Validates host hardware support; configures BitsAndBytes INT8/INT4 and FP16/BF16; provides theoretical & empirical KV cache analysis; LoRA adaptation & SFT module. |
| `evaluator.py` | Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores; calculates relative quality differences ($\Delta = \text{Score}_{\text{opt}} - \text{Score}_{\text{base}}$) and token overlap. |
| `storage.py` | Persists experiment records to SQLite; exports results to CSV and JSON; computes percentage improvements; recommends Pareto-optimal configurations. |
| `app.py` | Interactive dashboard featuring Overview, Model Lab, Benchmark Lab, Optimization Lab, Performance Analytics, Experiment History, and Recommender. |

---

## 2. Benchmark Methodology

To ensure high reproducibility across runs, the benchmarking engine enforces:

1. **Separation of Loading and Inference:** Model loading time is recorded separately from inference latency.
2. **Deterministic Warm-Up:** Configurable warm-up runs (default: 1–2) are executed prior to data collection to prime CUDA kernels, PyTorch JIT compilation, and memory allocators. Warm-up iterations are strictly excluded from reported statistics.
3. **Hardware Synchronization:** `torch.cuda.synchronize()` is called immediately before starting and after finishing the wall-clock timer (`time.perf_counter()`) on GPU systems to prevent asynchronous CUDA dispatch skew.
4. **Peak Memory Reset:** `torch.cuda.reset_peak_memory_stats()` is invoked before each trial to measure exact memory allocation (`torch.cuda.max_memory_allocated()`) and reserved memory (`torch.cuda.max_memory_reserved()`). CPU memory is monitored via `psutil.Process().memory_info().rss`.
5. **Real TTFT & TPOT Measurement:** Employs a custom `BenchmarkTokenStreamer` that timestamps the arrival of the first generated token (Time To First Token / prefill latency) and computes average inter-token latency (Time Per Output Token / decode latency).
6. **Multi-Trial Statistics:** Reports mean latency, median latency, P95 latency, and generation throughput across repeated trials.

---

## 3. Supported Configurations & Hardware Compatibility

### Precisions Supported
- **FP32 (Float32):** Standard single-precision baseline; supported on both CUDA GPUs and CPUs.
- **FP16 (Float16):** Half-precision; requires CUDA hardware with Tensor Core support.
- **BF16 (BFloat16):** Brain floating-point format; supported on NVIDIA GPUs with Compute Capability $\ge 8.0$ (Ampere, Ada Lovelace, Hopper) or modern CPUs with AVX512_BF16/AMX.
- **INT8 (8-bit Quantization):** Enabled via Hugging Face `BitsAndBytesConfig(load_in_8bit=True)` on CUDA GPUs.
- **INT4 (4-bit NF4 Quantization):** Enabled via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`. Reduces memory footprint by up to 70–75%.

### KV Cache Analysis
- **Theoretical Calculator:** $2 \times \text{layers} \times \text{heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{batch\_size} \times \text{bytes\_per\_element}$
- **Empirical Evaluation:** Compares autoregressive decoding with `use_cache=True` vs `use_cache=False`.
- *Note:* The standard KV cache trades memory for compute efficiency (preventing $\mathcal{O}(N^2)$ recomputation); it does *not* reduce memory footprint.

### LoRA Adaptation (PEFT)
- Wraps target linear projection layers (`q_proj`, `v_proj`) with low-rank decomposition matrices ($A \times B$).
- Calculates trainable vs frozen parameter counts and percentages.
- Features an optional lightweight Supervised Fine-Tuning (SFT) test to verify training efficiency and memory consumption during backpropagation.

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- NVIDIA GPU with CUDA 12.x drivers installed (or CPU fallback mode)

### Installation
Clone or navigate to the project directory:
```bash
cd QUANTORA
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 5. Running the Application

### Interactive Streamlit Dashboard
Launch the web UI:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Headless CLI Benchmarking
You can run automated benchmarks, query hardware specs, and inspect history directly from your terminal:
```bash
# 1. Display hardware & library telemetry
python cli.py hardware

# 2. Run a full benchmark trial with live GPU energy telemetry
python cli.py benchmark --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16 --trials 3 --max-tokens 64

# 3. View benchmark run history
python cli.py history

# 4. Find Pareto-optimal configuration based on memory and latency constraints
python cli.py recommend --budget 1500 --latency 3.0

# 5. Export history to CSV and JSON
python cli.py export
```

---

## 6. Verification and Testing

You can run automated programmatic tests verifying all components (model loading, precision checks, benchmarking, ROUGE quality scoring, and database storage) via Python:

```bash
python -c "
from model_engine import get_model_engine
from benchmark_engine import get_benchmark_engine
from storage import load_experiments

engine = get_model_engine()
info = engine.load_model('Qwen/Qwen2.5-0.5B-Instruct', precision='fp16', device='cuda')
print('Loaded Model:', info['model_id'], 'Footprint:', info['memory_footprint_mb'], 'MB')

bench = get_benchmark_engine()
res = bench.run_benchmark('Summarize quantum computing in one sentence.', max_new_tokens=32, num_trials=2)
print('Avg Latency:', res['avg_latency_s'], 's | Throughput:', res['throughput_tokens_sec'], 'tok/s')

df = load_experiments()
print('Stored Experiments:', len(df))
"
```

---

## 7. Known Hardware Limitations
- **INT8 / INT4 Quantization:** Requires an NVIDIA GPU and compiled `bitsandbytes` CUDA binaries. On CPU-only environments, the lab gracefully disables 8-bit/4-bit quantization and directs users to FP32.
- **BFloat16:** Native hardware support requires Compute Capability $\ge 8.0$. Older architectures will report incompatibility and recommend FP16.

---

## 8. License
Distributed under the **MIT License**. See `LICENSE` for details.

