import React, { useState } from 'react';
import { HardDrive, Gauge, Clock, Sparkles, ChevronRight, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import { BENCHMARK_MODELS } from '../data/benchmarkData';

export default function BenchmarkMatrix() {
  const [selectedModel, setSelectedModel] = useState(BENCHMARK_MODELS[3]); // INT4 default preview

  // Compute maximums for relative bars
  const maxVRAM = Math.max(...BENCHMARK_MODELS.map(m => m.peak_gpu_memory_mb));
  const maxThroughput = Math.max(...BENCHMARK_MODELS.map(m => m.throughput_tokens_sec));
  const maxLatency = Math.max(...BENCHMARK_MODELS.map(m => m.avg_latency_s));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Headline Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-900/40 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full blur-xl group-hover:bg-indigo-500/10 transition-all"></div>
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-semibold tracking-wider uppercase text-slate-400">Peak VRAM Savings</span>
            <div className="p-2 rounded-xl bg-indigo-950/60 text-indigo-400 border border-indigo-900/40">
              <HardDrive className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-black text-white tracking-tight">
            76.0% <span className="text-xs font-normal text-emerald-400">(-1,445 MB)</span>
          </div>
          <div className="text-xs text-slate-400 mt-2 flex items-center gap-1.5">
            <span className="text-indigo-400 font-mono font-medium">INT4/NF4</span> vs FP32 Baseline
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-900/40 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-xl group-hover:bg-emerald-500/10 transition-all"></div>
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-semibold tracking-wider uppercase text-slate-400">Max Throughput</span>
            <div className="p-2 rounded-xl bg-emerald-950/60 text-emerald-400 border border-emerald-900/40">
              <Gauge className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-black text-white tracking-tight">
            32.86 <span className="text-xs font-normal text-slate-400">tokens/s</span>
          </div>
          <div className="text-xs text-slate-400 mt-2 flex items-center gap-1.5">
            <span className="text-emerald-400 font-medium">FP32 / FP16</span> near-identical speed
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-900/40 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-xl group-hover:bg-cyan-500/10 transition-all"></div>
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-semibold tracking-wider uppercase text-slate-400">Time Per Output Token</span>
            <div className="p-2 rounded-xl bg-cyan-950/60 text-cyan-400 border border-cyan-900/40">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-black text-white tracking-tight">
            12.6 <span className="text-xs font-normal text-slate-400">ms/tok</span>
          </div>
          <div className="text-xs text-slate-400 mt-2 flex items-center gap-1.5">
            <span className="text-cyan-400 font-medium">TTFT: 1.7 ms</span> hardware sync
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-900/40 border border-slate-800 relative overflow-hidden group hover:border-slate-700 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full blur-xl group-hover:bg-purple-500/10 transition-all"></div>
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-semibold tracking-wider uppercase text-slate-400">Quality Parity</span>
            <div className="p-2 rounded-xl bg-purple-950/60 text-purple-400 border border-purple-900/40">
              <Sparkles className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-black text-white tracking-tight">
            100% <span className="text-xs font-normal text-emerald-400">ROUGE-L Parity</span>
          </div>
          <div className="text-xs text-slate-400 mt-2 flex items-center gap-1.5">
            FP16 retains bit-for-bit exactness
          </div>
        </div>
      </div>

      {/* Visual Analytics Graphs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Peak VRAM Comparison */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-indigo-400" />
              Peak VRAM (MB)
            </h3>
            <span className="text-xs text-slate-400 font-mono">Lower is better</span>
          </div>
          <div className="space-y-4">
            {BENCHMARK_MODELS.map((item) => {
              const widthPct = (item.peak_gpu_memory_mb / maxVRAM) * 100;
              const isInt4 = item.precision === 'int4';
              return (
                <div key={item.precision} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="font-semibold text-slate-300 uppercase">{item.precision}</span>
                    <span className={isInt4 ? 'text-emerald-400 font-bold' : 'text-slate-300'}>
                      {item.peak_gpu_memory_mb.toFixed(1)} MB {isInt4 && '(🏆 -76%)'}
                    </span>
                  </div>
                  <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden p-0.5">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        isInt4
                          ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                          : item.precision === 'int8'
                          ? 'bg-gradient-to-r from-cyan-500 to-blue-400'
                          : item.precision === 'fp16'
                          ? 'bg-gradient-to-r from-indigo-500 to-purple-400'
                          : 'bg-gradient-to-r from-slate-600 to-slate-400'
                      }`}
                      style={{ width: `${widthPct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Throughput Tokens/sec */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Gauge className="w-4 h-4 text-emerald-400" />
              Throughput (Tokens / sec)
            </h3>
            <span className="text-xs text-slate-400 font-mono">Higher is better</span>
          </div>
          <div className="space-y-4">
            {BENCHMARK_MODELS.map((item) => {
              const widthPct = (item.throughput_tokens_sec / maxThroughput) * 100;
              return (
                <div key={item.precision} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="font-semibold text-slate-300 uppercase">{item.precision}</span>
                    <span className="text-emerald-400 font-bold">
                      {item.throughput_tokens_sec.toFixed(1)} tok/s
                    </span>
                  </div>
                  <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden p-0.5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-700"
                      style={{ width: `${widthPct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Latency Comparison */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-purple-400" />
              Average Latency (Seconds)
            </h3>
            <span className="text-xs text-slate-400 font-mono">Lower is better</span>
          </div>
          <div className="space-y-4">
            {BENCHMARK_MODELS.map((item) => {
              const widthPct = (item.avg_latency_s / maxLatency) * 100;
              return (
                <div key={item.precision} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="font-semibold text-slate-300 uppercase">{item.precision}</span>
                    <span className="text-purple-300 font-bold">
                      {item.avg_latency_s.toFixed(2)} s
                    </span>
                  </div>
                  <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden p-0.5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-purple-600 to-indigo-400 transition-all duration-700"
                      style={{ width: `${widthPct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Comprehensive Results Table */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden shadow-xl">
        <div className="px-6 py-4 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              Controlled Benchmark Execution Records
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Empirical measurements on NVIDIA GeForce RTX 2050 (4GB VRAM) running Qwen/Qwen2.5-0.5B-Instruct
            </p>
          </div>
          <div className="text-xs font-mono px-3 py-1 rounded-md bg-slate-800/80 border border-slate-700 text-slate-300">
            64 Max New Tokens &bull; 3 Controlled Trials
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-950/70 border-b border-slate-800 text-slate-400 uppercase font-mono tracking-wider">
                <th className="py-3 px-4">Configuration</th>
                <th className="py-3 px-4">Device</th>
                <th className="py-3 px-4">Load Time</th>
                <th className="py-3 px-4">Peak VRAM</th>
                <th className="py-3 px-4">Compression</th>
                <th className="py-3 px-4">TPOT (Decode)</th>
                <th className="py-3 px-4">Throughput</th>
                <th className="py-3 px-4">ROUGE-L</th>
                <th className="py-3 px-4 text-right">Inspect Output</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {BENCHMARK_MODELS.map((row) => {
                const isSelected = selectedModel.precision === row.precision;
                return (
                  <tr
                    key={row.precision}
                    onClick={() => setSelectedModel(row)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-indigo-950/30' : 'hover:bg-slate-800/30'
                    }`}
                  >
                    <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                      <span
                        className={`inline-block w-2 h-2 rounded-full ${
                          row.precision === 'int4'
                            ? 'bg-emerald-400 shadow-sm shadow-emerald-400'
                            : row.precision === 'fp16'
                            ? 'bg-indigo-400'
                            : row.precision === 'int8'
                            ? 'bg-cyan-400'
                            : 'bg-slate-400'
                        }`}
                      ></span>
                      <span className="uppercase text-sm">{row.precision}</span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-300 uppercase">{row.device}</td>
                    <td className="py-3.5 px-4 text-slate-300">{row.load_time_s.toFixed(2)}s</td>
                    <td className="py-3.5 px-4 font-semibold text-slate-100">
                      {row.peak_gpu_memory_mb.toFixed(1)} MB
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-indigo-300 border border-slate-700">
                        {row.compression_ratio}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-300">{row.tpot_ms.toFixed(1)} ms</td>
                    <td className="py-3.5 px-4 font-bold text-emerald-400">
                      {row.throughput_tokens_sec.toFixed(1)} tok/s
                    </td>
                    <td className="py-3.5 px-4 text-purple-300">{row.rougeL.toFixed(1)}%</td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedModel(row);
                        }}
                        className={`px-2.5 py-1 rounded text-[11px] font-sans font-medium transition-all ${
                          isSelected
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-800 text-slate-300 hover:text-white'
                        }`}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Output Sample Preview Card */}
      {selectedModel && (
        <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 to-indigo-950/20 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span className="font-bold text-white text-sm">
                Generated Text Inspection &bull; {selectedModel.model_id} ({selectedModel.precision.toUpperCase()})
              </span>
            </div>
            <span className="text-xs font-mono text-slate-400">
              Avg Latency: {selectedModel.avg_latency_s.toFixed(3)}s | ROUGE-1: {selectedModel.rouge1}% | ROUGE-L: {selectedModel.rougeL}%
            </span>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 font-mono text-xs text-slate-300 leading-relaxed">
            "{selectedModel.output_sample}"
          </div>
        </div>
      )}
    </div>
  );
}
