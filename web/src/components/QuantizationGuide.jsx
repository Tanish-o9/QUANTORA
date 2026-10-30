import React from 'react';
import { BookOpen, Cpu, ShieldCheck, Zap, HardDrive, CheckCircle2 } from 'lucide-react';
import { QUANTIZATION_TAXONOMY } from '../data/benchmarkData';

export default function QuantizationGuide() {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-blue-950/40 via-slate-900 to-slate-900 border border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-blue-600/20 text-blue-400 border border-blue-500/30">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">
              LLM Quantization & Hardware Engineering Guide
            </h2>
            <p className="text-xs text-slate-400">
              Technical trade-offs, bit widths, memory compression factors, and target deployment environments supported by the Quantora optimization engine.
            </p>
          </div>
        </div>
      </div>

      {/* Taxonomy Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {QUANTIZATION_TAXONOMY.map((item, index) => (
          <div
            key={item.name}
            className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between space-y-4"
          >
            <div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-900">
                  {item.bytes * 8} BITS / WEIGHT
                </span>
                <span className="text-xs font-mono font-bold text-emerald-400">
                  {item.compression} Compression
                </span>
              </div>
              <h3 className="font-bold text-white text-base mt-2.5">{item.name}</h3>
              <p className="text-xs text-slate-400 mt-1">{item.best_for}</p>
            </div>

            <div className="space-y-2 text-xs font-mono border-t border-slate-800/80 pt-3">
              <div className="flex justify-between text-slate-400">
                <span>Speed:</span>
                <span className="text-slate-200">{item.speed}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Accuracy:</span>
                <span className="text-emerald-300 font-semibold">{item.accuracy}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Target:</span>
                <span className="text-indigo-300 text-[11px] truncate max-w-[170px]">{item.hardware}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Engineering Takeaways Callout */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 to-indigo-950/30 border border-slate-800 space-y-3">
        <h3 className="font-bold text-sm text-white flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          Quantora Engineering Insights from Empirical GPU Profiling
        </h3>
        <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside font-sans leading-relaxed">
          <li>
            <strong>FP16 is the standard production baseline:</strong> Halves VRAM with zero loss in generation quality and maximum Tensor Core acceleration on modern GPUs.
          </li>
          <li>
            <strong>INT4 / NF4 enables low-VRAM edge execution:</strong> Reduces memory by 76.0% (down to 457 MB for 0.5B parameters), making LLM inference possible on commodity consumer laptops and edge chips without OOM errors.
          </li>
          <li>
            <strong>Beware naive INT8 dequantization latency:</strong> BitsAndBytes INT8 can incur memory-to-register dequantization overhead on small batch sizes, making INT4 (fused dequant) faster in real-world decoding.
          </li>
        </ul>
      </div>
    </div>
  );
}
