import React, { useState, useMemo } from 'react';
import { Database, Zap, Cpu, ArrowUpRight, CheckCircle2, Sliders, Info, ShieldCheck } from 'lucide-react';
import { KV_CACHE_DATA } from '../data/benchmarkData';

export default function KVCacheCalculator() {
  // Theoretical KV Cache parameters
  const [layers, setLayers] = useState(24);
  const [kvHeads, setKvHeads] = useState(16);
  const [headDim, setHeadDim] = useState(64);
  const [seqLength, setSeqLength] = useState(2048);
  const [batchSize, setBatchSize] = useState(1);
  const [precisionBytes, setPrecisionBytes] = useState(2); // 2 = FP16/BF16, 1 = FP8, 0.5 = INT4

  // Theoretical KV Cache memory calculation
  // Formula: 2 (Keys + Values) * layers * kvHeads * headDim * seqLength * batchSize * precisionBytes
  const { totalBytes, totalMB, totalGB, perTokenBytes } = useMemo(() => {
    const bytes = 2 * layers * kvHeads * headDim * seqLength * batchSize * precisionBytes;
    const mb = bytes / (1024 * 1024);
    const gb = mb / 1024;
    const perToken = 2 * layers * kvHeads * headDim * batchSize * precisionBytes;
    return {
      totalBytes: bytes,
      totalMB: mb,
      totalGB: gb,
      perTokenBytes: perToken,
    };
  }, [layers, kvHeads, headDim, seqLength, batchSize, precisionBytes]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-emerald-600/20 text-emerald-400 border border-emerald-500/30">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">
              KV Cache Engineering & Theoretical Sizing Calculator
            </h2>
            <p className="text-xs text-slate-400">
              Analyze the fundamental architectural trade-off of autoregressive generation: storing past Key/Value projections in memory to avoid quadratic recomputation.
            </p>
          </div>
        </div>
      </div>

      {/* Part 1: Real Hardware Empirical Benchmark */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            Empirical GPU Benchmark: Cache Enabled vs Disabled
          </h3>
          <span className="text-xs font-mono text-emerald-400 font-semibold px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-900">
            1.19x Real Speedup Measured
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Enabled Card */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border-2 border-emerald-500/40 relative overflow-hidden">
            <div className="flex items-center justify-between mb-3">
              <span className="px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 font-mono text-xs font-bold flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" />
                KV CACHE ENABLED (Standard)
              </span>
              <span className="text-xs font-mono text-emerald-400 font-bold">32.36 tok/s</span>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Median Latency</span>
                <span className="text-lg font-bold text-white mt-0.5 block">
                  {KV_CACHE_DATA.enabled.median_latency_s.toFixed(3)}s
                </span>
                <span className="text-[10px] text-emerald-400">19% faster response</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Incremental VRAM</span>
                <span className="text-lg font-bold text-indigo-300 mt-0.5 block">
                  +{KV_CACHE_DATA.enabled.incremental_memory_mb.toFixed(2)} MB
                </span>
                <span className="text-[10px] text-slate-400">Past keys cached</span>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-3 font-sans leading-relaxed">
              At step $t$, keys and values of previous $t-1$ tokens are fetched from VRAM instead of computing $O(t^2)$ attention matrices from scratch.
            </p>
          </div>

          {/* Disabled Card */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 relative overflow-hidden">
            <div className="flex items-center justify-between mb-3">
              <span className="px-2.5 py-1 rounded-md bg-slate-800 text-slate-400 font-mono text-xs font-bold">
                KV CACHE DISABLED (Eager Recompute)
              </span>
              <span className="text-xs font-mono text-slate-400">27.11 tok/s</span>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Median Latency</span>
                <span className="text-lg font-bold text-slate-300 mt-0.5 block">
                  {KV_CACHE_DATA.disabled.median_latency_s.toFixed(3)}s
                </span>
                <span className="text-[10px] text-rose-400">0.38s recompute penalty</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase block">Incremental VRAM</span>
                <span className="text-lg font-bold text-slate-300 mt-0.5 block">
                  +{KV_CACHE_DATA.disabled.incremental_memory_mb.toFixed(2)} MB
                </span>
                <span className="text-[10px] text-slate-400">Activations buffer</span>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-3 font-sans leading-relaxed">
              Without cache, every single output token triggers full forward-pass quadratic re-attention over all previous prompt and generated tokens.
            </p>
          </div>
        </div>
      </div>

      {/* Part 2: Interactive Theoretical Calculator */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
          <div>
            <h3 className="font-bold text-base text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-indigo-400" />
              Theoretical Production KV Cache Sizing Simulator
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Calculate exact VRAM capacity needed per concurrent user session for any transformer model architecture.
            </p>
          </div>
          <div className="text-xs font-mono px-3 py-1 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-900/60 self-start sm:self-auto">
            Formula: 2 &times; L &times; H &times; D &times; S &times; B &times; P
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls */}
          <div className="lg:col-span-2 space-y-5">
            {/* Layers & KV Heads */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Layers (L)</span>
                  <span className="font-mono font-bold text-indigo-400">{layers}</span>
                </div>
                <input
                  type="range"
                  min="12"
                  max="80"
                  step="2"
                  value={layers}
                  onChange={(e) => setLayers(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>12 (0.5B)</span>
                  <span>24 (1.5B)</span>
                  <span>32 (7B)</span>
                  <span>80 (70B)</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">KV Attention Heads (H)</span>
                  <span className="font-mono font-bold text-indigo-400">{kvHeads}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="64"
                  step="2"
                  value={kvHeads}
                  onChange={(e) => setKvHeads(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>2 (GQA 70B)</span>
                  <span>8 (GQA 8B)</span>
                  <span>16 (MHA)</span>
                  <span>32</span>
                </div>
              </div>
            </div>

            {/* Sequence Length & Batch Size */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Context Sequence Length (S)</span>
                  <span className="font-mono font-bold text-cyan-400">{seqLength.toLocaleString()} tokens</span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="32768"
                  step="512"
                  value={seqLength}
                  onChange={(e) => setSeqLength(Number(e.target.value))}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>512</span>
                  <span>4,096</span>
                  <span>16,384</span>
                  <span>32,768</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Concurrent Batch Size (B)</span>
                  <span className="font-mono font-bold text-cyan-400">{batchSize} sessions</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="64"
                  step="1"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>1 (Single)</span>
                  <span>8</span>
                  <span>32</span>
                  <span>64 (High Concurrency)</span>
                </div>
              </div>
            </div>

            {/* Precision Selector & Head Dim */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              <div className="space-y-2">
                <span className="text-xs text-slate-300 font-medium block">KV Cache Quantization / Precision</span>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => setPrecisionBytes(2)}
                    className={`py-2 px-3 rounded-lg text-xs font-mono font-semibold transition-all ${
                      precisionBytes === 2
                        ? 'bg-indigo-600 text-white shadow-md'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    FP16 (2 B)
                  </button>
                  <button
                    onClick={() => setPrecisionBytes(1)}
                    className={`py-2 px-3 rounded-lg text-xs font-mono font-semibold transition-all ${
                      precisionBytes === 1
                        ? 'bg-indigo-600 text-white shadow-md'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    FP8 (1 B)
                  </button>
                  <button
                    onClick={() => setPrecisionBytes(0.5)}
                    className={`py-2 px-3 rounded-lg text-xs font-mono font-semibold transition-all ${
                      precisionBytes === 0.5
                        ? 'bg-indigo-600 text-white shadow-md'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    INT4 (0.5 B)
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs text-slate-300 font-medium block">Head Dimension (D)</span>
                <div className="grid grid-cols-3 gap-2">
                  {[64, 128, 256].map((d) => (
                    <button
                      key={d}
                      onClick={() => setHeadDim(d)}
                      className={`py-2 px-3 rounded-lg text-xs font-mono font-semibold transition-all ${
                        headDim === d
                          ? 'bg-cyan-600 text-white shadow-md'
                          : 'bg-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      {d} dim
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Result Card */}
          <div className="p-5 rounded-2xl bg-gradient-to-b from-indigo-950/60 to-slate-950 border border-indigo-500/30 flex flex-col justify-between space-y-4">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-widest text-indigo-400 font-semibold block">
                Total KV Memory Footprint
              </span>
              <div className="text-3xl font-black text-white font-mono mt-1">
                {totalGB >= 1 ? `${totalGB.toFixed(2)} GB` : `${totalMB.toFixed(1)} MB`}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Allocated across <strong className="text-white font-mono">{batchSize}</strong> concurrent streams of{' '}
                <strong className="text-white font-mono">{seqLength.toLocaleString()}</strong> tokens.
              </div>
            </div>

            <div className="space-y-2.5 text-xs font-mono border-t border-slate-800 pt-3">
              <div className="flex justify-between text-slate-400">
                <span>Per-Token Footprint:</span>
                <span className="text-white font-semibold">{(perTokenBytes / 1024).toFixed(2)} KB/token</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Memory per Stream:</span>
                <span className="text-white font-semibold">{(totalMB / batchSize).toFixed(1)} MB</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Quantization Gain:</span>
                <span className="text-emerald-400 font-semibold">
                  {precisionBytes === 2 ? 'Baseline (FP16)' : precisionBytes === 1 ? '2.0x savings (FP8)' : '4.0x savings (INT4)'}
                </span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 text-[11px] text-slate-300 space-y-1">
              <div className="flex items-center gap-1 font-semibold text-indigo-300">
                <Info className="w-3.5 h-3.5" />
                VRAM Architecture Insight
              </div>
              <p className="text-slate-400">
                Models utilizing Grouped-Query Attention (GQA) reduce KV heads by 4x to 8x, slashing this KV cache footprint by up to 87.5% without degrading model reasoning quality.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

