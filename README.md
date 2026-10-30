# ⚡ QUANTORA
### LLM Performance & Optimization Engineering Platform

[![CI/CD Pipeline](https://github.com/Tanish-o9/QUANTORA/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanish-o9/QUANTORA/actions/workflows/ci.yml)
[![Vercel Deployment](https://img.shields.io/badge/Deploy%20with-Vercel-black?logo=vercel)](https://vercel.com/new/clone?repository-url=https://github.com/Tanish-o9/QUANTORA/tree/main/web)
[![Docker Support](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://github.com/Tanish-o9/QUANTORA/blob/main/docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Tanish-o9/QUANTORA/tree/main/web)

**Benchmark. Analyze. Optimize.**

**Quantora** is a production-grade ML systems engineering and benchmarking suite designed to measure the real-world resource footprints, runtime latencies, throughput profiles, and output qualities of open-source Large Language Models (LLMs) across precision levels (FP32, FP16, BF16, INT8, INT4), KV cache policies, and PEFT Low-Rank Adaptation (LoRA) configurations.

All metrics are gathered from **real model execution on local hardware**—never mocked or hardcoded.

---

## 1. Core Architecture

The repository adheres to a strict, human-written, decoupled modular design:

```
QUANTORA/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated GitHub Actions CI/CD pipeline
├── web/                        # Modern React + Vite + Tailwind CSS Dashboard (Vercel Ready)
│   ├── src/components/         # LiveOperationsLab, BenchmarkMatrix, ParetoRecommender, KVCache
│   ├── Dockerfile              # Production multi-stage Nginx container
│   ├── nginx.conf              # Reverse proxy configuration
│   └── vercel.json             # 1-Click zero-config Vercel deployment spec
├── tests/
│   ├── test_optimizer.py       # Hardware detection, quantization & KV cache tests
│   ├── test_evaluator.py       # ROUGE, token F1 & Perplexity (PPL) tests
│   ├── test_storage.py         # SQLite persistence & Pareto recommender tests
│   └── test_cli.py             # Headless CLI command smoke tests
├── app.py                      # Professional Streamlit dashboard & analytics UI
├── server.py                   # Zero-dependency HTTP REST API backend (port 8000)
├── cli.py                      # Terminal CLI for automated benchmarks, history & perplexity
├── model_engine.py             # Model lifecycle, device placement, streaming & PPL
├── benchmark_engine.py         # Controlled benchmark trials, CUDA-sync timing & NVML telemetry
├── optimizer.py                # Hardware validation, quantization, KV cache & LoRA
├── evaluator.py                # ROUGE, token F1, reasoning & Perplexity (PPL) engine
├── storage.py                  # SQLite persistence, Pareto recommender & CSV/JSON export
├── docker-compose.yml          # Dual-service orchestrator (API + React Web)
├── Dockerfile                  # Python inference & benchmarking container
├── vercel.json                 # Vercel SPA routing configuration
├── requirements.txt            # Minimal, explicit project dependencies
├── LICENSE                     # MIT Open-Source License
├── README.md                   # Architecture documentation, setup & methodology
│
├── data/
│   ├── sample_prompts.json     # Fixed prompts across Summarization, Reasoning, Code & QA
│   └── reasoning_eval.json     # Standardized ML Systems & LLM reasoning questions
│
└── results/
    ├── experiments.db          # SQLite database storing benchmark runs
    ├── benchmark_report.html   # Standalone executive HTML report
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

# 6. Generate standalone executive HTML benchmark report
python cli.py report --output results/benchmark_report.html

# 7. Compute sliding-window Perplexity (PPL) for model fluency & quantization degradation
python cli.py perplexity --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16
```

### Docker & Microservices Deployment
Run the complete Quantora engineering platform (Python Inference API + React Frontend) inside isolated containers:

```bash
# Build and launch both Backend API & React Web UI
docker compose up --build

# Backend REST API will be live on: http://localhost:8000
# React Web Dashboard will be live on: http://localhost:3000
```

For NVIDIA GPU acceleration in Docker, ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed and uncomment the `deploy.resources.reservations.devices` block in `docker-compose.yml`.

### 🌐 Modern Web Dashboard (React 19 + Vite + Tailwind v4) & Vercel Deployment
A production-ready client-side interactive engineering dashboard is located in `web/` with live model interaction and zero-latency analytics:

```bash
# Local development
cd web
npm install
npm run dev
# Dashboard live at http://localhost:5173
```

#### 🚀 Deploy to Vercel in 1-Click
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Tanish-o9/QUANTORA/tree/main/web)

Both root `vercel.json` and `web/vercel.json` are pre-configured with SPA route rewrites:
1. **GitHub Import:** Import `https://github.com/Tanish-o9/QUANTORA` into [Vercel](https://vercel.com/new).
2. **Framework Preset:** Vite (Auto-detected).
3. **Dual Operation Mode:**
   - **Local Inference Mode:** Connects to `http://localhost:8000` when running the local Python API server.
   - **Edge Sandbox Simulator Mode:** When hosted purely on Vercel without a connected GPU, the **Live Operations Lab** automatically switches to edge simulation using exact transformer scaling laws.

---

## 6. Empirical Benchmark Case Study

The following benchmark results represent **real physical execution** on host hardware with no simulated or hardcoded values:

### Hardware Environment
- **GPU:** NVIDIA GeForce RTX 2050 (4.0 GB GDDR6 VRAM, Compute Capability 8.6)
- **CUDA Runtime:** 12.4 | **PyTorch:** 2.6.0+cu124
- **Host OS:** Windows 11 | **CPU RAM:** 12.0 GB
- **Evaluated Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (494.0M parameters)
- **Workload:** Multi-sentence AI systems summarization prompt (80 max new tokens, deterministic decoding)

### Empirical Results Across Numerical Precisions

| Precision Configuration | Weight Footprint | Peak GPU VRAM | TTFT (Prefill) | TPOT (Decode) | Avg Latency | Throughput | ROUGE-1 | ROUGE-L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** (Single Precision) | 1,884.6 MB | 1,903.4 MB | 2.3 ms | 12.6 ms | 2.44 s | 32.86 tok/s | 16.98% | 9.43% |
| **FP16** (Half Precision) | 942.3 MB | 956.2 MB | 1.7 ms | 12.9 ms | 2.49 s | 32.12 tok/s | 16.98% | 9.43% |
| **INT8** (BitsAndBytes) | 601.0 MB | 618.8 MB | 1.7 ms | 76.2 ms | 14.72 s | 5.55 tok/s | 26.67% | 15.24% |
| **INT4** (BitsAndBytes NF4) | 430.4 MB | 457.7 MB | 1.7 ms | 18.3 ms | 3.53 s | 22.67 tok/s | 23.53% | 15.69% |

### KV Cache Acceleration Profile (FP16, 64 tokens)

| Configuration | Median Latency | Generation Throughput | Peak GPU Memory | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **KV Cache Enabled** | **1.978 s** | **32.36 tok/s** | 956.1 MB | **1.19x** |
| **KV Cache Disabled** | 2.360 s | 27.11 tok/s | 959.6 MB | 1.00x (Baseline) |

### Key Systems Insights
1. **Memory Compression:** Transitioning from FP32 to FP16 halves model memory consumption (**50.0% reduction**, 1,884 MB $\rightarrow$ 942 MB) with zero latency degradation or quality drift.
2. **Extreme Edge Deployment (INT4 NF4):** 4-bit NormalFloat reduces the memory footprint by **76.0%** (1,903 MB $\rightarrow$ 457 MB peak VRAM), allowing modern LLMs to execute comfortably on entry-level GPUs with $\le$ 4GB VRAM while preserving practical interactive speeds (22.7 tok/s).
3. **KV Cache Speedup:** Caching autoregressive key-value tensors eliminates redundant quadratic attention passes, delivering a **1.19x speedup** on short completions and scaling further with longer sequence lengths.

---

## 7. Verification and Testing

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

## 8. Known Hardware Limitations
- **INT8 / INT4 Quantization:** Requires an NVIDIA GPU and compiled `bitsandbytes` CUDA binaries. On CPU-only environments, the lab gracefully disables 8-bit/4-bit quantization and directs users to FP32.
- **BFloat16:** Native hardware support requires Compute Capability $\ge 8.0$. Older architectures will report incompatibility and recommend FP16.

---

## 👤 Author & Maintainer

Developed and maintained by **[Tanish Kumar](https://github.com/Tanish-o9)**
- **GitHub:** [@Tanish-o9](https://github.com/Tanish-o9)
- **Repository:** [https://github.com/Tanish-o9/QUANTORA](https://github.com/Tanish-o9/QUANTORA)

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for details.

