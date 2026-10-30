import React from 'react';
import { Cpu, Zap, Activity, Download, FileText, Code2 } from 'lucide-react';
import { HARDWARE_SPECS } from '../data/benchmarkData';

export default function Navbar({ activeTab, setActiveTab, onExportJSON, onExportCSV, onPrintReport }) {
  const tabs = [
    { id: 'matrix', label: 'Benchmark Matrix' },
    { id: 'recommender', label: 'Pareto Recommender' },
    { id: 'kvcache', label: 'KV Cache Simulator' },
    { id: 'speculative', label: 'Speculative Decoding' },
    { id: 'scaling', label: 'Context Scaling' },
    { id: 'taxonomy', label: 'Quantization Guide' },
  ];

  return (
    <header className="border-b border-slate-800/80 bg-[#070b14]/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-300 bg-clip-text text-transparent">
                  QUANTORA
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-950/80 text-indigo-400 border border-indigo-800/60 uppercase tracking-wider">
                  v2.0 PRO
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                LLM Efficiency & Quantization Engineering Lab
              </p>
            </div>
          </div>

          {/* Hardware Telemetry Badge */}
          <div className="hidden lg:flex items-center gap-3 px-3.5 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs font-mono text-slate-300">
            <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
            <span>{HARDWARE_SPECS.device_name}</span>
            <span className="text-slate-600">|</span>
            <span className="text-indigo-400">{HARDWARE_SPECS.total_vram_gb} GB VRAM</span>
            <span className="text-slate-600">|</span>
            <span className="text-emerald-400">CUDA {HARDWARE_SPECS.cuda_version}</span>
          </div>

          {/* Action Buttons & Author */}
          <div className="flex items-center gap-2">
            <button
              onClick={onPrintReport}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800/90 hover:bg-slate-700 text-slate-200 border border-slate-700/80 transition-all cursor-pointer shadow-sm"
              title="Print / Save PDF Report"
            >
              <FileText className="w-3.5 h-3.5 text-indigo-400" />
              <span className="hidden sm:inline">Print / PDF</span>
            </button>
            <button
              onClick={onExportJSON}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-all cursor-pointer shadow-md shadow-indigo-600/30"
              title="Download Benchmark JSON"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Export JSON</span>
            </button>
            <a
              href="https://github.com/Tanish-o9/QUANTORA"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-all border border-slate-700/60"
              title="View on GitHub"
            >
              <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
            </a>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 sm:space-x-4 overflow-x-auto py-2.5 no-scrollbar text-xs font-medium border-t border-slate-800/40">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 rounded-lg whitespace-nowrap transition-all duration-150 cursor-pointer ${
                  isActive
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 font-semibold shadow-sm shadow-indigo-900/50'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
