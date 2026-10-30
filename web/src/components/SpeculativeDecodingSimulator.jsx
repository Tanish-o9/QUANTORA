import React, { useState, useMemo } from 'react';
import { FastForward, Zap, ShieldAlert, CheckCircle2, Sliders, ArrowRight, Activity } from 'lucide-react';

export default function SpeculativeDecodingSimulator() {
  const [draftK, setDraftK] = useState(4);
  const [acceptanceRate, setAcceptanceRate] = useState(0.75); // alpha
  const [targetLatencyMs, setTargetLatencyMs] = useState(45.0); // e.g. 7B / 14B model
  const [draftLatencyMs, setDraftLatencyMs] = useState(8.0); // e.g. 0.5B draft model

  // Theoretical speedup calculation
  // Expected accepted tokens per verification round = sum_{i=0}^K alpha^i
  // Round latency = K * T_draft + T_target
  // Effective cost per token = (K * T_draft + T_target) / (Expected tokens)
  // Speedup = T_target / (Effective cost per token)
  const { expectedAcceptedTokens, roundLatencyMs, effectiveTokenLatencyMs, speedupFactor, targetTokS, speculativeTokS } = useMemo(() => {
    let expectedTokens = 0;
    for (let i = 0; i <= draftK; i++) {
      expectedTokens += Math.pow(acceptanceRate, i);
    }
    // Expected accepted tokens is at least 1 (the target model correction bonus)
    const effectiveAccepted = Math.max(1, expectedTokens);

    const roundLat = (draftK * draftLatencyMs) + targetLatencyMs;
    const effLat = roundLat / effectiveAccepted;
    const speedup = targetLatencyMs / effLat;

    const baseTokS = 1000 / targetLatencyMs;
    const specTokS = 1000 / effLat;

    return {
      expectedAcceptedTokens: effectiveAccepted,
      roundLatencyMs: roundLat,
      effectiveTokenLatencyMs: effLat,
      speedupFactor: speedup,
      targetTokS: baseTokS,
      speculativeTokS: specTokS,
    };
  }, [draftK, acceptanceRate, targetLatencyMs, draftLatencyMs]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-purple-950/40 via-slate-900 to-slate-900 border border-slate-800">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-purple-600/20 text-purple-400 border border-purple-500/30">
            <FastForward className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">
              Speculative Decoding Acceleration Simulator
            </h2>
            <p className="text-xs text-slate-400">
              Simulate assistant model speculative decoding: pairing a lightweight draft model (e.g., Qwen-0.5B) with a high-capacity target model (e.g., Qwen-7B) to accelerate inference without loss in mathematical generation quality.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-5">
          <h3 className="font-bold text-sm text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            Decoding Parameters
          </h3>

          {/* Draft Lookahead Tokens (K) */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300">Draft Lookahead Tokens (K)</span>
              <span className="font-mono font-bold text-purple-400 px-2 py-0.5 rounded bg-purple-950/80 border border-purple-900">
                {draftK} tokens
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={draftK}
              onChange={(e) => setDraftK(Number(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>1</span>
              <span>3</span>
              <span>5</span>
              <span>8 (Aggressive)</span>
            </div>
          </div>

          {/* Acceptance Rate (Alpha) */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300">Target Acceptance Rate (&alpha;)</span>
              <span className="font-mono font-bold text-emerald-400 px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-900">
                {(acceptanceRate * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.95"
              step="0.05"
              value={acceptanceRate}
              onChange={(e) => setAcceptanceRate(Number(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>30% (Loose)</span>
              <span>60%</span>
              <span>80%</span>
              <span>95% (Aligned)</span>
            </div>
          </div>

          {/* Target Model Latency */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300">Target Latency (T_target)</span>
              <span className="font-mono font-bold text-slate-200 px-2 py-0.5 rounded bg-slate-800">
                {targetLatencyMs.toFixed(0)} ms/tok
              </span>
            </div>
            <input
              type="range"
              min="20"
              max="150"
              step="5"
              value={targetLatencyMs}
              onChange={(e) => setTargetLatencyMs(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>20 ms (Small)</span>
              <span>75 ms</span>
              <span>150 ms (70B)</span>
            </div>
          </div>

          {/* Draft Model Latency */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-slate-300">Draft Latency (T_draft)</span>
              <span className="font-mono font-bold text-slate-200 px-2 py-0.5 rounded bg-slate-800">
                {draftLatencyMs.toFixed(0)} ms/tok
              </span>
            </div>
            <input
              type="range"
              min="2"
              max="30"
              step="1"
              value={draftLatencyMs}
              onChange={(e) => setDraftLatencyMs(Number(e.target.value))}
              className="w-full accent-slate-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>2 ms (INT4)</span>
              <span>15 ms</span>
              <span>30 ms</span>
            </div>
          </div>
        </div>

        {/* Results & Speedup Visualizer */}
        <div className="lg:col-span-2 space-y-6">
          <div className="p-6 rounded-2xl bg-gradient-to-br from-purple-950/50 via-slate-900 to-slate-900 border-2 border-purple-500/40 space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-widest text-purple-400 font-semibold block">
                  Simulated Speedup Factor
                </span>
                <div className="text-4xl font-black text-white font-mono mt-1 flex items-baseline gap-2">
                  <span>{speedupFactor.toFixed(2)}x</span>
                  <span className={`text-xs font-normal ${speedupFactor >= 1.0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {speedupFactor >= 1.0 ? `(+${((speedupFactor - 1) * 100).toFixed(0)}% faster)` : '(Negative speedup)'}
                  </span>
                </div>
              </div>

              <div className="text-right">
                <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400 font-semibold block">
                  Expected Accepted Tokens / Round
                </span>
                <div className="text-2xl font-black text-emerald-400 font-mono mt-1">
                  {expectedAcceptedTokens.toFixed(2)} / {draftK + 1}
                </div>
              </div>
            </div>

            {/* Throughput comparison bar */}
            <div className="grid grid-cols-2 gap-3 pt-2">
              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono">
                <span className="text-slate-400 text-[10px] uppercase block">Target Baseline</span>
                <span className="text-lg font-bold text-slate-200 mt-0.5 block">{targetTokS.toFixed(1)} tok/s</span>
                <span className="text-[10px] text-slate-500">{targetLatencyMs.toFixed(0)} ms / token</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-purple-800/60 text-xs font-mono">
                <span className="text-purple-400 text-[10px] uppercase block">Speculative Accelerated</span>
                <span className="text-lg font-bold text-purple-300 mt-0.5 block">{speculativeTokS.toFixed(1)} tok/s</span>
                <span className="text-[10px] text-emerald-400">{effectiveTokenLatencyMs.toFixed(1)} ms eff. latency</span>
              </div>
            </div>

            {/* Token Acceptance Probability Progression */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
                Probability of Token Acceptance across Lookahead K={draftK}
              </span>
              <div className="grid grid-cols-8 gap-1.5 font-mono text-center">
                {Array.from({ length: 8 }).map((_, idx) => {
                  const isWithinK = idx < draftK;
                  const prob = isWithinK ? Math.pow(acceptanceRate, idx + 1) : 0;
                  return (
                    <div
                      key={idx}
                      className={`p-2 rounded-lg border text-xs transition-all ${
                        isWithinK
                          ? 'bg-purple-950/40 border-purple-800/80 text-white'
                          : 'bg-slate-950/30 border-slate-800/40 text-slate-600 opacity-40'
                      }`}
                    >
                      <div className="text-[10px] text-slate-400 font-bold">K={idx + 1}</div>
                      <div className="text-xs font-bold text-emerald-400 mt-0.5">
                        {isWithinK ? `${(prob * 100).toFixed(0)}%` : '-'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Verification Guarantee */}
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-300 leading-relaxed font-sans flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <p>
                <strong>Zero Quality Degradation:</strong> The target model computes parallel logits across all $K$ tokens. Any token where the draft diverges from the target's greedy choice or temperature sampling is rejected, and the target immediately injects the exact correct token without re-running draft generation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

