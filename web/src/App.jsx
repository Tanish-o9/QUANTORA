import React, { useState } from 'react';
import Navbar from './components/Navbar';
import BenchmarkMatrix from './components/BenchmarkMatrix';
import ParetoRecommender from './components/ParetoRecommender';
import KVCacheCalculator from './components/KVCacheCalculator';
import SpeculativeDecodingSimulator from './components/SpeculativeDecodingSimulator';
import ContextScalingView from './components/ContextScalingView';
import QuantizationGuide from './components/QuantizationGuide';
import { BENCHMARK_MODELS, HARDWARE_SPECS, KV_CACHE_DATA, CONTEXT_SCALING_SERIES } from './data/benchmarkData';

export default function App() {
  const [activeTab, setActiveTab] = useState('matrix');

  // Client-side instant JSON export
  const handleExportJSON = () => {
    const payload = {
      platform: "QUANTORA LLM Efficiency Lab",
      exported_at: new Date().toISOString(),
      maintainer: "Tanish Kumar (@Tanish-o9)",
      hardware: HARDWARE_SPECS,
      benchmarks: BENCHMARK_MODELS,
      kv_cache_analysis: KV_CACHE_DATA,
      context_scaling: CONTEXT_SCALING_SERIES,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantora_benchmark_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Client-side instant CSV export
  const handleExportCSV = () => {
    const headers = [
      "model_id",
      "precision",
      "device",
      "load_time_s",
      "peak_gpu_memory_mb",
      "throughput_tokens_sec",
      "avg_latency_s",
      "tpot_ms",
      "ttft_s",
      "rougeL",
      "compression_ratio",
    ];
    const rows = BENCHMARK_MODELS.map((m) => [
      m.model_id,
      m.precision,
      m.device,
      m.load_time_s,
      m.peak_gpu_memory_mb,
      m.throughput_tokens_sec,
      m.avg_latency_s,
      m.tpot_ms,
      m.ttft_s,
      m.rougeL,
      m.compression_ratio,
    ]);
    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantora_benchmarks_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#070b14] text-slate-100 font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onExportJSON={handleExportJSON}
        onExportCSV={handleExportCSV}
        onPrintReport={handlePrint}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'matrix' && <BenchmarkMatrix />}
        {activeTab === 'recommender' && <ParetoRecommender />}
        {activeTab === 'kvcache' && <KVCacheCalculator />}
        {activeTab === 'speculative' && <SpeculativeDecodingSimulator />}
        {activeTab === 'scaling' && <ContextScalingView />}
        {activeTab === 'taxonomy' && <QuantizationGuide />}
      </main>

      <footer className="border-t border-slate-800/80 bg-[#070b14]/90 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="font-bold text-white tracking-wide">QUANTORA</span>
            <span>&bull;</span>
            <span>LLM Performance & Optimization Engineering Platform</span>
          </div>

          <div className="flex items-center gap-3">
            <span>
              Engineered & Maintained by{' '}
              <a
                href="https://github.com/Tanish-o9"
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-400 hover:text-indigo-300 font-semibold underline underline-offset-2"
              >
                Tanish Kumar (@Tanish-o9)
              </a>
            </span>
            <span>&bull;</span>
            <a
              href="https://github.com/Tanish-o9/QUANTORA"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 hover:text-white"
            >
              GitHub Repository
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
