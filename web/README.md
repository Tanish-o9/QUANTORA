# QUANTORA — Modern Web Dashboard

> **LLM Performance & Optimization Engineering Platform**  
> Engineered & Maintained by [Tanish Kumar (@Tanish-o9)](https://github.com/Tanish-o9)

A high-performance, dark-mode React + Vite + Tailwind CSS dashboard designed for deployment on [Vercel](https://vercel.com). Built as an alternative to Streamlit for instantaneous client-side interactivity, Pareto frontier solving, and interactive KV Cache & Speculative Decoding simulations.

---

## ⚡ Key Features

1. **Benchmark Matrix**: Real empirical GPU profiles for `Qwen2.5-0.5B-Instruct` across FP32, FP16, INT8, and INT4 (NF4) with real VRAM, TTFT, TPOT, throughput, and ROUGE-L scores.
2. **Pareto Frontier Recommender**: Interactive constraint sliders (VRAM Budget, Latency Threshold, Min Quality) that mathematically solve the optimal quantization strategy.
3. **KV Cache Engineering Simulator**: Dual mode showing both real 1.19x empirical hardware speedup and theoretical parameter sizing ($2 \times L \times H \times D \times S \times B \times P$).
4. **Speculative Decoding Simulator**: Interactive assistant model acceleration calculator showing theoretical speedup curves ($S = \frac{1}{(1-\alpha) + \alpha / (K+1)}$) without degradation.
5. **Context Scaling Visualizer**: Stress-testing prefill TTFT and KV memory growth from 128 to 4,096 tokens.
6. **Executive Export**: Client-side instant JSON export, CSV export, and print-ready PDF reports.

---

## 🚀 Quick Local Start

```bash
cd web
npm install
npm run dev
```

Visit `http://localhost:5173` in your browser.

---

## 🌐 Deploy to Vercel in 1 Click

### Option 1: Vercel CLI
```bash
npm i -g vercel
vercel
```

### Option 2: GitHub Integration (Recommended)
1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository: `https://github.com/Tanish-o9/QUANTORA`
3. If setting root directory: choose `web/` (or leave as root, root `vercel.json` automatically triggers `cd web && npm run build`)
4. Click **Deploy**!
