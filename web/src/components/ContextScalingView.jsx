import React from 'react';
import { TrendingUp, Clock, HardDrive, AlertTriangle, Layers } from 'lucide-react';
import { CONTEXT_SCALING_SERIES } from '../data/benchmarkData';

export default function ContextScalingView() {
  const maxTTFT = Math.max(...CONTEXT_SCALING_SERIES.map((s) => s.ttft_ms));
  const maxKV = Math.max(...CONTEXT_SCALING_SERIES.map((s) => s.kv_cache_mb));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-cyan-950/40 via-slate-900 to-slate-900 border border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-cyan-600/20 text-cyan-400 border border-cyan-500/30">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">
              Context Length Scaling Stress-Test
            </h2>
            <p className="text-xs text-slate-400">
              Empirical and simulated stress-testing of prefill latency (TTFT) and KV Cache memory footprint as prompt context length scales from 128 to 4,096 tokens.
            </p>
          </div>
        </div>
      </div>

      {/* Visual Progression Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* TTFT Prefill Latency Scaling */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-cyan-400" />
              Time-To-First-Token (TTFT Prefill) Scaling
            </h3>
            <span className="text-xs font-mono text-cyan-400">ms latency</span>
          </div>

          <div className="space-y-3.5 pt-2">
            {CONTEXT_SCALING_SERIES.map((item) => {
              const widthPct = (item.ttft_ms / maxTTFT) * 100;
              return (
                <div key={item.context_tokens} className="space-y-1.5 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-300 font-medium">{item.context_tokens} tokens</span>
                    <span className="text-cyan-300 font-bold">{item.ttft_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden p-0.5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-700"
                      style={{ width: `${widthPct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-slate-400 font-sans pt-2">
            Prefill latency scales with input prompt length. At 4,096 tokens, prompt ingestion time increases to ~310ms before the first generated token appears.
          </p>
        </div>

        {/* KV Cache Allocation Scaling */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-indigo-400" />
              KV Cache VRAM Consumption Scaling
            </h3>
            <span className="text-xs font-mono text-indigo-400">MB memory</span>
          </div>

          <div className="space-y-3.5 pt-2">
            {CONTEXT_SCALING_SERIES.map((item) => {
              const widthPct = (item.kv_cache_mb / maxKV) * 100;
              return (
                <div key={item.context_tokens} className="space-y-1.5 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-300 font-medium">{item.context_tokens} tokens</span>
                    <span className="text-indigo-300 font-bold">{item.kv_cache_mb.toFixed(0)} MB</span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden p-0.5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-700"
                      style={{ width: `${widthPct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-slate-400 font-sans pt-2">
            KV Cache grows linearly with sequence length ($O(N)$). At 4k context, the KV cache alone occupies 384 MB per single stream.
          </p>
        </div>
      </div>

      {/* Stress-Test Data Table */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-800">
          <h3 className="font-bold text-sm text-white">Context Length Performance Matrix</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-950/70 border-b border-slate-800 text-slate-400 uppercase font-mono tracking-wider">
                <th className="py-3 px-4">Context Length</th>
                <th className="py-3 px-4">Prefill TTFT</th>
                <th className="py-3 px-4">Decode Speed</th>
                <th className="py-3 px-4">KV Cache MB</th>
                <th className="py-3 px-4">Degradation vs 128 tok</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {CONTEXT_SCALING_SERIES.map((row) => {
                const ttftRatio = (row.ttft_ms / CONTEXT_SCALING_SERIES[0].ttft_ms).toFixed(1);
                return (
                  <tr key={row.context_tokens} className="hover:bg-slate-800/30">
                    <td className="py-3 px-4 font-bold text-white">{row.context_tokens.toLocaleString()} tokens</td>
                    <td className="py-3 px-4 text-cyan-400 font-semibold">{row.ttft_ms.toFixed(1)} ms</td>
                    <td className="py-3 px-4 text-emerald-400 font-semibold">{row.decode_tok_s.toFixed(1)} tok/s</td>
                    <td className="py-3 px-4 text-indigo-300">{row.kv_cache_mb.toFixed(0)} MB</td>
                    <td className="py-3 px-4 text-slate-400">
                      {row.context_tokens === 128 ? '1.0x (Baseline)' : `${ttftRatio}x slower TTFT`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
