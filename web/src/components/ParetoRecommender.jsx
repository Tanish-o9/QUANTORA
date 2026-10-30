import React, { useState, useMemo } from 'react';
import { Sliders, Award, Zap, HardDrive, Clock, Check, AlertCircle, ArrowRight } from 'lucide-react';
import { BENCHMARK_MODELS } from '../data/benchmarkData';

export default function ParetoRecommender() {
  const [memoryBudget, setMemoryBudget] = useState(1200);
  const [maxLatency, setMaxLatency] = useState(4.0);
  const [minQuality, setMinQuality] = useState(8.0);

  // Solve Pareto-optimal configuration
  const { winner, eligible, rejected } = useMemo(() => {
    const passed = [];
    const failed = [];

    BENCHMARK_MODELS.forEach((model) => {
      const memoryPass = model.peak_gpu_memory_mb <= memoryBudget;
      const latencyPass = model.avg_latency_s <= maxLatency;
      const qualityPass = model.rougeL >= minQuality;

      if (memoryPass && latencyPass && qualityPass) {
        passed.push(model);
      } else {
        const reasons = [];
        if (!memoryPass) reasons.push(`VRAM exceeds ${memoryBudget} MB (${model.peak_gpu_memory_mb.toFixed(0)} MB)`);
        if (!latencyPass) reasons.push(`Latency exceeds ${maxLatency}s (${model.avg_latency_s.toFixed(2)}s)`);
        if (!qualityPass) reasons.push(`ROUGE-L below ${minQuality}% (${model.rougeL.toFixed(1)}%)`);
        failed.push({ ...model, reasons });
      }
    });

    // Best candidate is the one with highest throughput
    passed.sort((a, b) => b.throughput_tokens_sec - a.throughput_tokens_sec);

    return {
      winner: passed[0] || null,
      eligible: passed,
      rejected: failed,
    };
  }, [memoryBudget, maxLatency, minQuality]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-indigo-950/40 via-slate-900 to-slate-900 border border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">
              Pareto Frontier Optimization Engine
            </h2>
            <p className="text-xs text-slate-400">
              Set your production constraints below. Quantora mathematically solves the Pareto-optimal configuration maximizing throughput without sacrificing memory or response quality.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sliders Control Panel */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-400" />
            Deployment Constraints
          </h3>

          {/* VRAM Budget Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300 flex items-center gap-1.5">
                <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
                VRAM Budget
              </span>
              <span className="font-mono font-bold text-indigo-400 px-2 py-0.5 rounded bg-indigo-950/80 border border-indigo-900">
                {memoryBudget} MB
              </span>
            </div>
            <input
              type="range"
              min="400"
              max="2200"
              step="50"
              value={memoryBudget}
              onChange={(e) => setMemoryBudget(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>400 MB (Edge)</span>
              <span>1200 MB</span>
              <span>2200 MB (Cloud)</span>
            </div>
          </div>

          {/* Max Latency Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-cyan-400" />
                Max Acceptable Latency
              </span>
              <span className="font-mono font-bold text-cyan-400 px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-900">
                {maxLatency.toFixed(1)} s
              </span>
            </div>
            <input
              type="range"
              min="2.0"
              max="16.0"
              step="0.5"
              value={maxLatency}
              onChange={(e) => setMaxLatency(Number(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>2.0 s (Real-time)</span>
              <span>8.0 s</span>
              <span>16.0 s (Batch)</span>
            </div>
          </div>

          {/* Minimum Quality Slider */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-purple-400" />
                Min Quality (ROUGE-L)
              </span>
              <span className="font-mono font-bold text-purple-400 px-2 py-0.5 rounded bg-purple-950/80 border border-purple-900">
                {minQuality.toFixed(0)} %
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="20"
              step="1"
              value={minQuality}
              onChange={(e) => setMinQuality(Number(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0% (Any)</span>
              <span>10%</span>
              <span>20% (Strict)</span>
            </div>
          </div>

          {/* Quick Presets */}
          <div className="pt-3 border-t border-slate-800 space-y-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Constraint Presets
            </span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                onClick={() => {
                  setMemoryBudget(600);
                  setMaxLatency(5.0);
                  setMinQuality(8.0);
                }}
                className="px-3 py-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-all text-left"
              >
                <div className="font-semibold text-white">Edge / 4GB VRAM</div>
                <div className="text-[10px] text-slate-400">&le; 600MB &bull; 5.0s</div>
              </button>
              <button
                onClick={() => {
                  setMemoryBudget(2000);
                  setMaxLatency(3.0);
                  setMinQuality(8.0);
                }}
                className="px-3 py-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-all text-left"
              >
                <div className="font-semibold text-white">High Throughput</div>
                <div className="text-[10px] text-slate-400">&le; 2000MB &bull; 3.0s</div>
              </button>
            </div>
          </div>
        </div>

        {/* Winner Showcase Card */}
        <div className="lg:col-span-2 space-y-6">
          {winner ? (
            <div className="p-6 rounded-2xl bg-gradient-to-br from-indigo-950/70 via-slate-900 to-slate-900 border-2 border-indigo-500/50 shadow-2xl relative overflow-hidden">
              <div className="absolute top-4 right-4 flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-xs font-mono font-bold">
                <Award className="w-3.5 h-3.5" />
                PARETO OPTIMAL
              </div>

              <div className="space-y-4">
                <div>
                  <span className="text-xs uppercase font-mono tracking-widest text-indigo-400 font-semibold">
                    Recommended Model Configuration
                  </span>
                  <h3 className="text-2xl font-black text-white mt-1">
                    {winner.model_id} &bull;{' '}
                    <span className="uppercase text-indigo-400">{winner.precision}</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Matches all user constraints with maximum decoded throughput of{' '}
                    <strong className="text-emerald-400 font-mono">
                      {winner.throughput_tokens_sec.toFixed(1)} tokens/sec
                    </strong>
                    .
                  </p>
                </div>

                {/* Metric Cards Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Peak VRAM</span>
                    <span className="text-base font-bold text-white font-mono mt-0.5 block">
                      {winner.peak_gpu_memory_mb.toFixed(1)} MB
                    </span>
                    <span className="text-[10px] text-emerald-400 font-medium">
                      {(100 - (winner.peak_gpu_memory_mb / 1903.35) * 100).toFixed(0)}% vs FP32
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Throughput</span>
                    <span className="text-base font-bold text-emerald-400 font-mono mt-0.5 block">
                      {winner.throughput_tokens_sec.toFixed(1)} tok/s
                    </span>
                    <span className="text-[10px] text-slate-400">TPOT {winner.tpot_ms.toFixed(1)}ms</span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Avg Latency</span>
                    <span className="text-base font-bold text-white font-mono mt-0.5 block">
                      {winner.avg_latency_s.toFixed(2)} s
                    </span>
                    <span className="text-[10px] text-slate-400">TTFT {winner.ttft_s.toFixed(3)}s</span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 uppercase font-mono block">Quality Score</span>
                    <span className="text-base font-bold text-purple-400 font-mono mt-0.5 block">
                      {winner.rougeL.toFixed(1)}%
                    </span>
                    <span className="text-[10px] text-slate-400">ROUGE-L Score</span>
                  </div>
                </div>

                {/* Candidate Analysis */}
                <div className="pt-2">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block mb-2">
                    Why this configuration won:
                  </span>
                  <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed font-sans">
                    {winner.precision === 'fp16' && (
                      <p>
                        <strong>FP16 Half-Precision</strong> offers the ideal sweet spot: exactly 50% memory reduction compared to FP32 while sustaining the maximum hardware generation rate (32.12 tok/s) with 100% bit-for-bit quality parity.
                      </p>
                    )}
                    {winner.precision === 'int4' && (
                      <p>
                        <strong>INT4 / NF4 NormalFloat</strong> cuts VRAM down to 457.7 MB (a massive 76.0% memory reduction) enabling execution on strict edge devices while maintaining a high throughput of 22.67 tok/s.
                      </p>
                    )}
                    {winner.precision === 'fp32' && (
                      <p>
                        <strong>FP32 Single Precision</strong> delivers the highest raw throughput (32.86 tok/s) within your generous memory budget.
                      </p>
                    )}
                    {winner.precision === 'int8' && (
                      <p>
                        <strong>INT8</strong> achieves a 67.5% memory compression to fit inside your low VRAM target.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-8 rounded-2xl bg-slate-900/40 border border-rose-900/40 text-center space-y-3">
              <AlertCircle className="w-8 h-8 text-rose-400 mx-auto" />
              <h3 className="font-bold text-white text-base">No Matching Configuration Found</h3>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                No tested model precision meets your strict combination of VRAM ({memoryBudget} MB), Latency ({maxLatency}s), and Quality ({minQuality}%). Try loosening the constraints.
              </p>
            </div>
          )}

          {/* Candidates Evaluation Log */}
          <div className="rounded-2xl bg-slate-900/60 border border-slate-800 p-5 space-y-3">
            <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Candidates Comparison Status ({eligible.length} Satisfied, {rejected.length} Filtered)
            </h4>
            <div className="space-y-2">
              {BENCHMARK_MODELS.map((item) => {
                const isWinner = winner?.precision === item.precision;
                const rej = rejected.find((r) => r.precision === item.precision);
                return (
                  <div
                    key={item.precision}
                    className={`p-3 rounded-xl border text-xs flex items-center justify-between font-mono ${
                      isWinner
                        ? 'bg-indigo-950/40 border-indigo-500/50 text-white'
                        : rej
                        ? 'bg-slate-950/40 border-slate-800/80 text-slate-500'
                        : 'bg-slate-900/60 border-slate-800 text-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {isWinner ? (
                        <Check className="w-4 h-4 text-emerald-400" />
                      ) : rej ? (
                        <AlertCircle className="w-4 h-4 text-rose-500/70" />
                      ) : (
                        <span className="w-4 h-4 text-center text-slate-500">&bull;</span>
                      )}
                      <span className="font-bold uppercase text-slate-200">{item.precision}</span>
                      <span className="text-[11px] text-slate-400">
                        {item.peak_gpu_memory_mb.toFixed(0)} MB | {item.throughput_tokens_sec.toFixed(1)} tok/s | {item.avg_latency_s.toFixed(2)}s
                      </span>
                    </div>

                    <div>
                      {isWinner ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-bold">
                          WINNER
                        </span>
                      ) : rej ? (
                        <span className="text-[10px] text-rose-400">
                          {rej.reasons[0]}
                        </span>
                      ) : (
                        <span className="text-[10px] text-slate-400">Eligible</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
