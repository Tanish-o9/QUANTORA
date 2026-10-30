"""app.py - LLM Efficiency Lab Streamlit Dashboard.

Memory, Latency, Throughput & Quality Benchmarking Platform.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from benchmark_engine import get_benchmark_engine, run_batched_throughput_comparison
from evaluator import get_evaluator
from model_engine import get_model_engine
from optimizer import (
    apply_lora_to_model,
    calculate_kv_cache_size_bytes,
    format_bytes_to_mb,
    get_hardware_capabilities,
    run_kv_cache_comparison,
    run_lightweight_sft_experiment,
    validate_configuration,
)
from storage import (
    compute_comparison_metrics,
    export_results_to_csv_and_json,
    load_experiments,
    recommend_best_configuration,
    save_experiment,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Efficiency Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .hardware-badge {
        font-family: monospace;
        background-color: #2b303c;
        color: #58a6ff;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_sample_prompts() -> List[Dict[str, Any]]:
    path = os.path.join("data", "sample_prompts.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {
            "id": "default-01",
            "category": "General",
            "prompt": "Explain the architectural differences between INT4 and FP16 quantization.",
            "reference": "INT4 represents model weights using 4-bit integers with a scale factor, reducing memory by ~75% compared to FP16, at the cost of slight quantization noise.",
        }
    ]


@st.cache_data
def load_reasoning_dataset() -> List[Dict[str, Any]]:
    path = os.path.join("data", "reasoning_eval.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


model_engine = get_model_engine()
benchmark_engine = get_benchmark_engine()
evaluator = get_evaluator()
hw_info = get_hardware_capabilities()

# ---------------------------------------------------------------------------
# Sidebar: System Telemetry & Model Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ LLM Efficiency Lab")
    st.caption("Hardware, Latency & Quality Benchmarking")
    st.divider()

    st.markdown("#### 🖥️ Hardware Telemetry")
    if hw_info["cuda_available"]:
        st.success(f"GPU: {hw_info['device_name']}")
        st.write(f"**VRAM:** {hw_info['total_vram_gb']} GB")
        st.write(f"**Compute:** {hw_info['compute_capability']}")
        st.write(f"**BF16 Support:** {'Yes' if hw_info['bf16_supported'] else 'No'}")
    else:
        st.info("Device: CPU Only (CUDA unavailable)")

    st.write(f"**System RAM:** {hw_info['total_ram_gb']} GB")
    st.write(f"**PyTorch:** `{hw_info['torch_version']}`")
    st.write(f"**BitsAndBytes:** {'✅ Ready' if hw_info['bitsandbytes_available'] else '❌ Not Found'}")
    st.write(f"**PEFT (LoRA):** {'✅ Ready' if hw_info['peft_available'] else '❌ Not Found'}")

    st.divider()
    st.markdown("#### 🧠 Loaded Model")
    if model_engine.is_loaded():
        specs = model_engine.get_model_info()
        st.markdown(f"**ID:** `{specs['model_id']}`")
        st.markdown(f"**Precision:** `{specs['precision'].upper()}`")
        st.markdown(f"**Device:** `{specs['device']}`")
        st.markdown(f"**Params:** `{specs['parameter_count_million']}M`")
        st.markdown(f"**Footprint:** `{specs['memory_footprint_mb']} MB`")
        if st.button("Unload Model", use_container_width=True):
            model_engine.unload_model()
            st.rerun()
    else:
        st.warning("No model currently loaded in memory.")

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_model, tab_benchmark, tab_opt, tab_analytics, tab_history, tab_reports = st.tabs(
    [
        "📊 Overview",
        "🔬 Model Lab",
        "⏱️ Benchmark Lab",
        "⚡ Optimization Lab",
        "📈 Performance Analytics",
        "📜 Experiment History",
        "📑 Reports & Recommender",
    ]
)

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab_overview:
    st.markdown("<div class='main-title'>LLM Efficiency Lab</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Empirical benchmarking of memory, latency, throughput, and output quality across LLM precisions.</div>",
        unsafe_allow_html=True,
    )

    df_exp = load_experiments()
    total_runs = len(df_exp)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Experiments", total_runs)

    if not df_exp.empty:
        max_tp = df_exp["throughput_tokens_sec"].max()
        best_tp_row = df_exp[df_exp["throughput_tokens_sec"] == max_tp].iloc[0]
        c2.metric("Highest Throughput", f"{max_tp:.1f} tok/s", f"{best_tp_row['model_id']} ({best_tp_row['precision']})")

        valid_mems = df_exp["peak_gpu_allocated_mb"].dropna()
        if not valid_mems.empty:
            min_mem = valid_mems.min()
            best_mem_row = df_exp[df_exp["peak_gpu_allocated_mb"] == min_mem].iloc[0]
            c3.metric("Lowest Peak VRAM", f"{min_mem:.1f} MB", f"{best_mem_row['precision']}")
        else:
            min_cpu = df_exp["peak_cpu_ram_mb"].min()
            c3.metric("Lowest RAM", f"{min_cpu:.1f} MB")

        best_lat = df_exp["avg_latency_s"].min()
        c4.metric("Lowest Latency", f"{best_lat:.3f} s")
    else:
        c2.metric("Highest Throughput", "N/A")
        c3.metric("Lowest Memory", "N/A")
        c4.metric("Lowest Latency", "N/A")

    st.markdown("### 📋 Recent Benchmarks")
    if not df_exp.empty:
        cols_to_show = [
            "experiment_id",
            "timestamp",
            "model_id",
            "precision",
            "device",
            "avg_latency_s",
            "throughput_tokens_sec",
            "peak_gpu_allocated_mb",
            "peak_cpu_ram_mb",
            "rougeL",
        ]
        available_cols = [c for c in cols_to_show if c in df_exp.columns]
        st.dataframe(df_exp[available_cols].head(5), use_container_width=True)
    else:
        st.info("No benchmark experiments logged yet. Head to **Model Lab** to load a model and **Benchmark Lab** to run your first trial.")

# ===========================================================================
# TAB 2: MODEL LAB
# ===========================================================================
with tab_model:
    st.subheader("Model Configuration & Inspection")

    col_cfg1, col_cfg2 = st.columns([2, 1])

    with col_cfg1:
        preset_models = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "gpt2",
            "Custom Model ID",
        ]
        selected_preset = st.selectbox("Select Model Preset:", preset_models)

        if selected_preset == "Custom Model ID":
            model_id_input = st.text_input("Hugging Face Model ID:", "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            model_id_input = selected_preset

        precision_options = ["fp16", "fp32", "bf16", "int8", "int4"] if hw_info["cuda_available"] else ["fp32", "bf16"]
        selected_precision = st.selectbox("Select Precision / Quantization:", precision_options, index=0)

        device_options = ["cuda", "cpu"] if hw_info["cuda_available"] else ["cpu"]
        selected_device = st.selectbox("Target Device:", device_options, index=0)

        # Hardware verification check
        is_valid, validation_msg = validate_configuration(selected_precision, selected_device)
        if not is_valid:
            st.error(f"⚠️ Incompatible configuration: {validation_msg}")
        else:
            st.success(f"✓ Configuration compatible: {validation_msg}")

        load_btn = st.button("🚀 Load Model into Memory", disabled=not is_valid, type="primary")

    with col_cfg2:
        st.markdown("#### Architecture Specs")
        if model_engine.is_loaded():
            m_info = model_engine.get_model_info()
            st.json(m_info)
        else:
            st.info("Load a model to view active parameters and memory footprint.")

    if load_btn:
        with st.spinner(f"Loading '{model_id_input}' with {selected_precision.upper()} on {selected_device.upper()}..."):
            try:
                res = model_engine.load_model(
                    model_id=model_id_input,
                    precision=selected_precision,
                    device=selected_device,
                    force_reload=True,
                )
                st.success(f"Model successfully loaded in {res['load_time_seconds']}s!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    # Interactive Playground
    st.divider()
    st.subheader("Interactive Generation Playground")
    if model_engine.is_loaded():
        play_prompt = st.text_area("Prompt:", "What are the key trade-offs between FP16 and INT4 quantization?", height=80)
        c_p1, c_p2, c_p3 = st.columns(3)
        max_toks = c_p1.slider("Max New Tokens:", 16, 256, 64)
        temp = c_p2.slider("Temperature:", 0.0, 1.0, 0.0)
        do_sample = c_p3.checkbox("Enable Sampling", value=False)

        if st.button("Generate Text"):
            with st.spinner("Generating..."):
                t_out = model_engine.generate(
                    prompt=play_prompt,
                    max_new_tokens=max_toks,
                    temperature=temp,
                    do_sample=do_sample,
                )
                st.markdown("##### Model Output:")
                st.write(t_out["generated_text"])

                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Latency", f"{t_out['latency_seconds']} s")
                g2.metric("TTFT", f"{t_out['ttft_seconds']} s" if t_out['ttft_seconds'] is not None else "N/A")
                g3.metric("TPOT", f"{t_out['tpot_seconds'] * 1000:.1f} ms" if t_out['tpot_seconds'] is not None else "N/A")
                g4.metric("Throughput", f"{t_out['throughput_tokens_per_sec']} tok/s")
    else:
        st.info("Load a model above to unlock the generation playground.")

# ===========================================================================
# TAB 3: BENCHMARK LAB
# ===========================================================================
with tab_benchmark:
    st.subheader("Controlled Empirical Benchmarking")
    st.caption("Executes synchronized inference trials, warm-up iterations, and logs hardware metrics.")

    if not model_engine.is_loaded():
        st.warning("⚠️ No model loaded. Please load a model in the 'Model Lab' tab before benchmarking.")
    else:
        active_specs = model_engine.get_model_info()
        st.info(f"Active Model: **{active_specs['model_id']}** | Precision: **{active_specs['precision'].upper()}** | Device: **{active_specs['device']}**")

        prompts_db = load_sample_prompts()
        prompt_options = [f"[{p['category']}] {p['id']}: {p['prompt'][:60]}..." for p in prompts_db]
        prompt_options.append("Custom Prompt")

        selected_prompt_idx = st.selectbox("Select Benchmark Prompt:", range(len(prompt_options)), format_func=lambda i: prompt_options[i])

        if selected_prompt_idx < len(prompts_db):
            selected_prompt_data = prompts_db[selected_prompt_idx]
            bench_prompt = st.text_area("Prompt Content:", selected_prompt_data["prompt"], height=100)
            bench_reference = st.text_area("Ground Truth Reference (for ROUGE quality scoring):", selected_prompt_data.get("reference", ""), height=70)
            bench_prompt_id = selected_prompt_data["id"]
        else:
            bench_prompt = st.text_area("Custom Prompt:", "Explain why matrix multiplication dominates transformer inference latency.", height=100)
            bench_reference = st.text_area("Optional Reference Answer:", "", height=70)
            bench_prompt_id = "custom"

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        warmup_runs = col_b1.number_input("Warm-up Runs:", min_value=0, max_value=5, value=1, help="Excluded from measured metrics.")
        num_trials = col_b2.number_input("Measured Trials:", min_value=1, max_value=10, value=3, help="Number of repeated inference trials.")
        max_output_tokens = col_b3.number_input("Max Output Tokens:", min_value=16, max_value=512, value=64)
        use_kv_cache = col_b4.checkbox("Use KV Cache", value=True)

        if st.button("⚡ Run Benchmark Experiment", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, msg):
                progress_bar.progress(int((current / total) * 100))
                status_text.text(msg)

            try:
                result = benchmark_engine.run_benchmark(
                    prompt=bench_prompt,
                    prompt_id=bench_prompt_id,
                    reference_answer=bench_reference if bench_reference.strip() else None,
                    max_new_tokens=max_output_tokens,
                    warmup_runs=warmup_runs,
                    num_trials=num_trials,
                    use_cache=use_kv_cache,
                    progress_callback=update_progress,
                )

                status_text.text("Benchmark complete!")
                st.success(f"Benchmark finished and saved! (Experiment ID: `{result['experiment_id']}`)")

                # Display KPI Cards
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Avg Latency", f"{result['avg_latency_s']} s", f"P95: {result['p95_latency_s']}s")
                k2.metric("Decode Throughput", f"{result['throughput_tokens_sec']} tok/s")
                k3.metric("TTFT", f"{result['ttft_s']} s" if result['ttft_s'] is not None else "N/A")
                k4.metric("TPOT", f"{result['tpot_s'] * 1000:.1f} ms" if result['tpot_s'] is not None else "N/A")
                if result["peak_gpu_allocated_mb"] is not None:
                    k5.metric("Peak VRAM", f"{result['peak_gpu_allocated_mb']} MB")
                else:
                    k5.metric("Peak RAM", f"{result['peak_cpu_ram_mb']} MB")

                # Quality evaluation summary
                if result.get("rougeL"):
                    st.markdown("#### Quality Metrics (ROUGE)")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("ROUGE-1 F1", f"{result['rouge1']:.1f}%")
                    q2.metric("ROUGE-2 F1", f"{result['rouge2']:.1f}%")
                    q3.metric("ROUGE-L F1", f"{result['rougeL']:.1f}%")

                # Raw trial data table
                st.markdown("#### Trial-by-Trial Granular Data")
                df_trials = pd.DataFrame(result["raw_trials"])
                st.dataframe(df_trials, use_container_width=True)

                st.markdown("#### Generated Output")
                st.write(result["generated_text"])

            except Exception as e:
                st.error(f"Benchmark execution failed: {e}")

# ===========================================================================
# TAB 4: OPTIMIZATION LAB
# ===========================================================================
with tab_opt:
    st.subheader("Optimization Deep-Dive & Comparative Experiments")

    opt_choice = st.radio(
        "Select Optimization Domain:",
        [
            "Quantization Comparison (FP32 vs FP16 vs INT8 vs INT4)",
            "KV Cache Scaling Experiment",
            "PEFT LoRA Module",
            "Batched Concurrency Saturation Curve",
            "MMLU Reasoning Accuracy across Precisions",
        ],
        horizontal=True,
    )

    # -----------------------------------------------------------------------
    # Domain 1: Quantization Sweep
    # -----------------------------------------------------------------------
    if opt_choice.startswith("Quantization"):
        st.markdown("### Quantization Precision Sweep")
        st.markdown(
            "Automatically evaluates the loaded model under available precisions on fixed prompts "
            "to calculate empirical **memory reduction**, **throughput gains**, and **quality deltas**."
        )

        available_precisions = []
        if hw_info["cuda_available"]:
            available_precisions = ["fp16", "fp32"]
            if hw_info["bf16_supported"]:
                available_precisions.append("bf16")
            if hw_info["bitsandbytes_available"]:
                available_precisions.extend(["int8", "int4"])
        else:
            available_precisions = ["fp32"]

        selected_precisions_to_test = st.multiselect(
            "Select Precisions to Benchmark:",
            available_precisions,
            default=available_precisions[:min(3, len(available_precisions))],
        )

        test_prompt = st.text_area(
            "Fixed Evaluation Prompt:",
            "Summarize the computational advantages and hardware trade-offs of integer quantization in modern GPUs.",
            height=80,
            key="quant_prompt",
        )
        test_ref = st.text_area(
            "Reference Answer:",
            "Integer quantization reduces memory bandwidth bottlenecks and VRAM footprint by using INT8 or INT4, enabling higher throughput with minimal accuracy loss.",
            height=70,
            key="quant_ref",
        )

        if st.button("Run Quantization Sweep", type="primary"):
            if not selected_precisions_to_test:
                st.error("Select at least one precision.")
            else:
                sweep_results = []
                sweep_bar = st.progress(0)
                active_model_id = model_engine.model_id or "Qwen/Qwen2.5-0.5B-Instruct"

                for idx, prec in enumerate(selected_precisions_to_test):
                    sweep_bar.progress(int((idx / len(selected_precisions_to_test)) * 100))
                    st.write(f"Testing precision **{prec.upper()}**...")

                    try:
                        model_engine.load_model(
                            model_id=active_model_id,
                            precision=prec,
                            device="cuda" if hw_info["cuda_available"] else "cpu",
                        )
                        b_res = benchmark_engine.run_benchmark(
                            prompt=test_prompt,
                            prompt_id=f"quant_{prec}",
                            reference_answer=test_ref,
                            max_new_tokens=48,
                            warmup_runs=1,
                            num_trials=2,
                        )
                        sweep_results.append(b_res)
                    except Exception as e:
                        st.error(f"Error testing {prec}: {e}")

                sweep_bar.progress(100)
                st.success("Quantization sweep completed!")

                # Render Comparison Table
                if len(sweep_results) >= 2:
                    st.markdown("#### Baseline vs Optimized Comparison")
                    baseline_run = sweep_results[0]
                    comparison_records = []

                    for opt_run in sweep_results[1:]:
                        comp = compute_comparison_metrics(baseline_run, opt_run)
                        comparison_records.append({
                            "Baseline": f"{baseline_run['precision'].upper()}",
                            "Optimized": f"{opt_run['precision'].upper()}",
                            "Baseline Memory (MB)": comp["baseline_memory_mb"],
                            "Optimized Memory (MB)": comp["optimized_memory_mb"],
                            "Memory Reduction (%)": f"{comp['memory_reduction_pct']}%",
                            "Baseline Latency (s)": comp["baseline_latency_s"],
                            "Optimized Latency (s)": comp["optimized_latency_s"],
                            "Latency Improvement (%)": f"{comp['latency_improvement_pct']}%",
                            "Throughput Gain (%)": f"{comp['throughput_improvement_pct']}%",
                            "Quality Delta (ROUGE-L)": comp["quality_delta"],
                        })
                    st.dataframe(pd.DataFrame(comparison_records), use_container_width=True)

    # -----------------------------------------------------------------------
    # Domain 2: KV Cache Scaling
    # -----------------------------------------------------------------------
    elif opt_choice.startswith("KV Cache"):
        st.markdown("### KV Cache Scaling & Mechanism Analysis")
        st.markdown(
            "> [!NOTE]\n"
            "> **Technical Principle:** The standard Key-Value (KV) cache **does not reduce memory**.\n"
            r"> It consumes additional runtime VRAM/RAM to store past attention keys and values, avoiding the $\mathcal{O}(N^2)$ "
            "quadratic recomputation of prior tokens during autoregressive decoding."
        )

        col_kv1, col_kv2 = st.columns(2)
        with col_kv1:
            st.markdown("#### Theoretical KV Memory Calculator")
            c_layers = st.number_input("Number of Layers:", value=24, step=2)
            c_heads = st.number_input("Attention Key/Value Heads:", value=14, step=2)
            c_dim = st.number_input("Head Dimension:", value=64, step=16)
            c_seq = st.slider("Context Sequence Length:", 128, 8192, 1024, step=128)
            c_batch = st.slider("Batch Size:", 1, 16, 1)

            kv_bytes = calculate_kv_cache_size_bytes(c_layers, c_heads, c_dim, c_seq, c_batch, bytes_per_elem=2)
            kv_mb = format_bytes_to_mb(kv_bytes)
            st.metric("Theoretical KV Cache Footprint", f"{kv_mb} MB", f"{kv_mb / 1024:.2f} GB")

        with col_kv2:
            st.markdown("#### Live KV Cache On/Off Benchmark")
            if model_engine.is_loaded():
                kv_prompt = st.text_area("Test Prompt:", "Explain how autoregressive language models generate tokens step by step.", height=80)
                kv_tokens = st.slider("Tokens to Generate:", 16, 64, 32)

                if st.button("Run KV Cache On vs Off Comparison"):
                    with st.spinner("Measuring KV cache impact..."):
                        kv_comparison = run_kv_cache_comparison(
                            model=model_engine.model,
                            tokenizer=model_engine.tokenizer,
                            prompt=kv_prompt,
                            max_new_tokens=kv_tokens,
                            device=model_engine.device,
                        )
                        st.info(kv_comparison["explanation"])
                        st.json({
                            "KV Cache Enabled": kv_comparison["kv_cache_enabled"],
                            "KV Cache Disabled": kv_comparison["kv_cache_disabled"],
                        })
            else:
                st.info("Load a model in the Model Lab to run live KV cache comparison.")

    # -----------------------------------------------------------------------
    # Domain 3: PEFT LoRA Module
    # -----------------------------------------------------------------------
    elif opt_choice.startswith("PEFT LoRA"):
        st.markdown("### PEFT LoRA (Low-Rank Adaptation) Module")
        if not hw_info["peft_available"]:
            st.error("PEFT package is not installed.")
        elif not model_engine.is_loaded():
            st.warning("Load a base model in Model Lab first.")
        else:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                lora_r = st.slider("LoRA Rank (r):", 2, 64, 8)
                lora_alpha = st.slider("LoRA Alpha:", 4, 128, 16)
                lora_dropout = st.slider("Dropout:", 0.0, 0.2, 0.05)

                if st.button("Apply LoRA Adapter"):
                    try:
                        lora_m, stats = apply_lora_to_model(
                            model_engine.model,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        )
                        st.success("LoRA adapter successfully attached to model!")
                        st.json(stats)
                    except Exception as e:
                        st.error(f"Failed to attach LoRA: {e}")

            with col_l2:
                st.markdown("#### Optional Lightweight Fine-Tuning Step")
                st.caption("Runs 3-5 gradient descent steps on sample prompts to verify training loop efficiency.")
                sft_steps = st.slider("Training Steps:", 1, 5, 3)

                if st.button("Run Quick SFT Trial"):
                    sample_data = load_sample_prompts()
                    with st.spinner("Executing SFT steps..."):
                        try:
                            sft_res = run_lightweight_sft_experiment(
                                model=model_engine.model,
                                tokenizer=model_engine.tokenizer,
                                training_samples=sample_data,
                                num_steps=sft_steps,
                                device=model_engine.device,
                            )
                            st.success(f"Trained {sft_steps} steps in {sft_res['training_duration_seconds']}s!")
                            st.metric("Initial Loss", sft_res["initial_loss"])
                            st.metric("Final Loss", sft_res["final_loss"], f"{sft_res['final_loss'] - sft_res['initial_loss']:.4f}")
                        except Exception as e:
                            st.error(f"SFT failed: {e}")

    # -----------------------------------------------------------------------
    # Domain 4: Batched Concurrency Saturation Curve
    # -----------------------------------------------------------------------
    elif opt_choice.startswith("Batched Concurrency"):
        st.markdown("### Batched Throughput Saturation Curve")
        st.markdown(
            "Analyzes how GPU tensor cores saturate as batch concurrency increases from 1 to 8, "
            "highlighting the trade-off between total system throughput (tokens/sec) and individual user latency."
        )

        if not model_engine.is_loaded():
            st.warning("Load a model in the Model Lab tab first.")
        else:
            b_prompt = st.text_area("Test Prompt:", "Explain why batching increases hardware arithmetic intensity.", height=70)
            b_max_toks = st.slider("Tokens per Sequence:", 16, 64, 24)

            if st.button("Benchmark Batch Sizes (1, 2, 4, 8)", type="primary"):
                with st.spinner("Executing batched inference sweep..."):
                    try:
                        b_records = run_batched_throughput_comparison(
                            model=model_engine.model,
                            tokenizer=model_engine.tokenizer,
                            prompt=b_prompt,
                            batch_sizes=[1, 2, 4, 8],
                            max_new_tokens=b_max_toks,
                            device=model_engine.device,
                        )
                        df_bs = pd.DataFrame(b_records)
                        st.dataframe(df_bs, use_container_width=True)

                        fig_bs = px.line(
                            df_bs,
                            x="batch_size",
                            y="throughput_tok_s",
                            markers=True,
                            title="Throughput Saturation Curve (Total Tokens Generated / Second)",
                            labels={"batch_size": "Concurrent Batch Size", "throughput_tok_s": "Throughput (tok/s)"},
                        )
                        st.plotly_chart(fig_bs, use_container_width=True)

                    except Exception as e:
                        st.error(f"Batched sweep failed: {e}")

    # -----------------------------------------------------------------------
    # Domain 5: MMLU Reasoning Accuracy across Precisions
    # -----------------------------------------------------------------------
    elif opt_choice.startswith("MMLU Reasoning"):
        st.markdown("### MMLU Reasoning Accuracy across Precisions")
        st.markdown(
            "Runs a standardized 5-question multi-choice ML Systems reasoning benchmark "
            "to verify whether lower precision (e.g. INT4) causes reasoning degradation."
        )

        reasoning_data = load_reasoning_dataset()
        st.write(f"Loaded **{len(reasoning_data)} standardized reasoning questions** from `data/reasoning_eval.json`.")

        if not model_engine.is_loaded():
            st.warning("Load a model in Model Lab first.")
        else:
            if st.button("Evaluate Current Model on Reasoning Benchmark", type="primary"):
                with st.spinner("Evaluating multi-choice accuracy..."):
                    def predict_wrapper(prompt_text):
                        gen_out = model_engine.generate(
                            prompt=prompt_text,
                            max_new_tokens=16,
                            do_sample=False,
                        )
                        return gen_out["generated_text"]

                    eval_res = evaluator.evaluate_multiple_choice(
                        questions=reasoning_data,
                        predict_fn=predict_wrapper,
                    )

                    c_acc1, c_acc2 = st.columns(2)
                    c_acc1.metric("Reasoning Accuracy", f"{eval_res['accuracy_pct']}%", f"{eval_res['correct']} / {eval_res['total_questions']} Correct")
                    c_acc2.metric("Active Precision", model_engine.precision.upper())

                    df_det = pd.DataFrame(eval_res["details"])
                    st.dataframe(df_det, use_container_width=True)

# ===========================================================================
# TAB 5: PERFORMANCE ANALYTICS
# ===========================================================================
with tab_analytics:
    st.subheader("Performance Analytics & Trade-Off Visualizations")
    df_chart = load_experiments()

    if df_chart.empty:
        st.info("No benchmark data available yet to render analytics. Execute benchmarks in the Benchmark Lab first.")
    else:
        chart_type = st.selectbox(
            "Select Analytics Chart:",
            [
                "Latency by Precision (Bar & P95)",
                "Generation Throughput (Tokens/s)",
                "Peak Memory Usage (VRAM / RAM)",
                "Throughput vs Memory Pareto Frontier",
                "Quality (ROUGE-L) vs Latency Trade-off",
            ],
        )

        if chart_type == "Latency by Precision (Bar & P95)":
            fig = px.bar(
                df_chart,
                x="precision",
                y="avg_latency_s",
                color="model_id",
                barmode="group",
                title="Average Inference Latency across Precisions",
                labels={"avg_latency_s": "Average Latency (Seconds)", "precision": "Precision"},
                hover_data=["median_latency_s", "p95_latency_s", "throughput_tokens_sec"],
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Generation Throughput (Tokens/s)":
            fig = px.bar(
                df_chart,
                x="precision",
                y="throughput_tokens_sec",
                color="model_id",
                barmode="group",
                title="Generation Throughput (Tokens Generated / Second)",
                labels={"throughput_tokens_sec": "Throughput (tok/s)", "precision": "Precision"},
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Peak Memory Usage (VRAM / RAM)":
            y_col = "peak_gpu_allocated_mb" if df_chart["peak_gpu_allocated_mb"].notna().any() else "peak_cpu_ram_mb"
            fig = px.bar(
                df_chart,
                x="precision",
                y=y_col,
                color="model_id",
                title="Peak Memory Footprint (MB)",
                labels={y_col: "Peak Memory (MB)", "precision": "Precision"},
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Throughput vs Memory Pareto Frontier":
            x_col = "peak_gpu_allocated_mb" if df_chart["peak_gpu_allocated_mb"].notna().any() else "peak_cpu_ram_mb"
            fig = px.scatter(
                df_chart,
                x=x_col,
                y="throughput_tokens_sec",
                color="precision",
                size="total_params",
                hover_name="model_id",
                title="Throughput vs Memory Footprint (Top-Left = Ideal Efficiency)",
                labels={x_col: "Peak Memory (MB)", "throughput_tokens_sec": "Throughput (tok/s)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Quality (ROUGE-L) vs Latency Trade-off":
            valid_rouge = df_chart[df_chart["rougeL"].notna()]
            if valid_rouge.empty:
                st.warning("No ROUGE quality scores recorded yet. Run benchmarks with ground-truth references.")
            else:
                fig = px.scatter(
                    valid_rouge,
                    x="avg_latency_s",
                    y="rougeL",
                    color="precision",
                    hover_name="model_id",
                    title="Output Quality (ROUGE-L) vs Latency (Top-Left = Ideal)",
                    labels={"avg_latency_s": "Latency (s)", "rougeL": "ROUGE-L F1 (%)"},
                )
                st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 6: EXPERIMENT HISTORY
# ===========================================================================
with tab_history:
    st.subheader("Historical Benchmark Runs")
    df_hist = load_experiments()

    if df_hist.empty:
        st.info("No experiment history logged yet.")
    else:
        # Filter controls
        f1, f2 = st.columns(2)
        unique_models = ["All"] + list(df_hist["model_id"].unique())
        sel_m = f1.selectbox("Filter by Model:", unique_models)

        unique_precisions = ["All"] + list(df_hist["precision"].unique())
        sel_p = f2.selectbox("Filter by Precision:", unique_precisions)

        filtered_df = df_hist.copy()
        if sel_m != "All":
            filtered_df = filtered_df[filtered_df["model_id"] == sel_m]
        if sel_p != "All":
            filtered_df = filtered_df[filtered_df["precision"] == sel_p]

        st.dataframe(filtered_df, use_container_width=True)

        # Side-by-side run comparison
        st.divider()
        st.markdown("#### Side-by-Side Run Comparison")
        if len(df_hist) >= 2:
            c_comp1, c_comp2 = st.columns(2)
            exp_choices = df_hist["experiment_id"].tolist()
            id_a = c_comp1.selectbox("Baseline Run (A):", exp_choices, index=0)
            id_b = c_comp2.selectbox("Optimized Run (B):", exp_choices, index=min(1, len(exp_choices) - 1))

            if id_a and id_b and id_a != id_b:
                run_a = df_hist[df_hist["experiment_id"] == id_a].iloc[0].to_dict()
                run_b = df_hist[df_hist["experiment_id"] == id_b].iloc[0].to_dict()

                comp_metrics = compute_comparison_metrics(run_a, run_b)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Memory Reduction", f"{comp_metrics['memory_reduction_pct']}%", f"{comp_metrics['optimized_memory_mb']} MB vs {comp_metrics['baseline_memory_mb']} MB")
                m2.metric("Latency Gain", f"{comp_metrics['latency_improvement_pct']}%", f"{comp_metrics['optimized_latency_s']}s vs {comp_metrics['baseline_latency_s']}s")
                m3.metric("Throughput Gain", f"{comp_metrics['throughput_improvement_pct']}%", f"{comp_metrics['optimized_throughput_tok_s']} tok/s")
                m4.metric("Quality Delta (ROUGE-L)", f"{comp_metrics['quality_delta']:.2f}")

# ===========================================================================
# TAB 7: REPORTS & RECOMMENDER
# ===========================================================================
with tab_reports:
    st.subheader("Engineering Reports & Optimization Recommender")

    col_rec, col_rep = st.columns([1, 1])

    with col_rec:
        st.markdown("#### 🎯 Constraint-Based Optimization Recommender")
        st.caption("Specify deployment constraints to identify the optimal verified configuration.")

        req_vram = st.number_input("Max Memory Budget (MB):", min_value=100.0, max_value=64000.0, value=2000.0)
        req_lat = st.number_input("Max Acceptable Latency (s):", min_value=0.01, max_value=60.0, value=2.0)
        req_qual = st.number_input("Min Acceptable ROUGE-L (%):", min_value=0.0, max_value=100.0, value=0.0)

        if st.button("Find Recommended Configuration"):
            rec = recommend_best_configuration(
                max_memory_mb=req_vram,
                max_latency_s=req_lat,
                min_quality_score=req_qual if req_qual > 0 else None,
            )
            if rec:
                st.success(f"Recommended Configuration: **{rec['model_id']}** ({rec['precision'].upper()})")
                st.json({
                    "Model": rec["model_id"],
                    "Precision": rec["precision"],
                    "Avg Latency (s)": rec["avg_latency_s"],
                    "Throughput (tok/s)": rec["throughput_tokens_sec"],
                    "Peak GPU Memory (MB)": rec["peak_gpu_allocated_mb"],
                    "Peak CPU Memory (MB)": rec["peak_cpu_ram_mb"],
                    "ROUGE-L Score": rec["rougeL"],
                })
            else:
                st.warning("No benchmarked experiment satisfies all these constraints. Try relaxing criteria or benchmarking additional precisions.")

    with col_rep:
        st.markdown("#### 📥 Export Benchmark Reports")
        csv_file, json_file = export_results_to_csv_and_json()

        with open(csv_file, "r", encoding="utf-8") as f:
            csv_data = f.read()
        st.download_button(
            label="Download CSV Results",
            data=csv_data,
            file_name="benchmark_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with open(json_file, "r", encoding="utf-8") as f:
            json_data = f.read()
        st.download_button(
            label="Download JSON Results",
            data=json_data,
            file_name="benchmark_results.json",
            mime="application/json",
            use_container_width=True,
        )

        # Markdown Engineering Report
        st.markdown("#### 📄 Generate Engineering Report")
        if st.button("Generate Markdown Report", use_container_width=True):
            df_curr = load_experiments()
            report_md = f"""# LLM Efficiency Lab - Benchmark Engineering Report
**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Host Platform:** {hw_info['os']}
**GPU Device:** {hw_info['device_name']} ({hw_info['total_vram_gb']} GB VRAM)
**PyTorch Version:** {hw_info['torch_version']} (CUDA: {hw_info['cuda_version']})

## Summary Statistics
- Total Experiments Executed: {len(df_curr)}
- Evaluated Precisions: {', '.join(df_curr['precision'].unique()) if not df_curr.empty else 'N/A'}
- Highest Measured Throughput: {df_curr['throughput_tokens_sec'].max() if not df_curr.empty else 'N/A'} tok/s

## Experiment Log
{df_curr[['experiment_id', 'model_id', 'precision', 'avg_latency_s', 'throughput_tokens_sec', 'peak_gpu_allocated_mb']].to_markdown() if not df_curr.empty else 'No runs logged.'}
"""
            st.download_button(
                label="Download Engineering Report (.md)",
                data=report_md,
                file_name="LLM_Efficiency_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )
