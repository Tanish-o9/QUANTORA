"""optimizer.py - Hardware compatibility, quantization configurations, KV cache analysis, and PEFT LoRA.

Core optimization module for Quantora:
- Hardware capability detection and precision validation
- Model loading arguments for FP32/FP16/BF16 and BitsAndBytes INT8/INT4
- Theoretical and empirical KV cache benchmarking
- PEFT LoRA adapter application and lightweight fine-tuning
"""

import gc
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from transformers import BitsAndBytesConfig

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except Exception:
    HAS_BITSANDBYTES = False

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


# ---------------------------------------------------------------------------
# Hardware Detection
# ---------------------------------------------------------------------------

def get_hardware_capabilities() -> Dict[str, Any]:
    """Detect and return host hardware capabilities and installed library support."""
    cuda_available = torch.cuda.is_available()
    device_name = "CPU Only"
    total_vram_gb = 0.0
    compute_capability = (0, 0)
    bf16_supported = False

    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            compute_capability = torch.cuda.get_device_capability(0)
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            cuda_available = False

    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024 ** 3)
    available_ram_gb = vm.available / (1024 ** 3)

    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_name": device_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}" if cuda_available else "N/A",
        "bf16_supported": bf16_supported,
        "bitsandbytes_available": HAS_BITSANDBYTES,
        "peft_available": HAS_PEFT,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
    }


# ---------------------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------------------

def validate_configuration(precision: str, device: str) -> Tuple[bool, str]:
    """Validate whether requested precision and device are supported on the current host.

    Supported precisions: 'fp32', 'fp16', 'bf16', 'int8', 'int4'
    Supported devices: 'cuda', 'cpu'
    """
    precision = precision.lower()
    device = device.lower()
    hw = get_hardware_capabilities()

    if device not in ("cuda", "cpu"):
        return False, f"Unsupported device '{device}'. Choose 'cuda' or 'cpu'."

    valid_precisions = ("fp32", "fp16", "bf16", "int8", "int4")
    if precision not in valid_precisions:
        return False, f"Unknown precision option '{precision}'. Supported: {', '.join(valid_precisions)}."

    if device == "cuda" and not hw["cuda_available"]:
        return False, "CUDA is not available on this system. Switch device to 'cpu'."

    if device == "cpu":
        if precision in ("fp16", "int8", "int4"):
            return False, f"Precision '{precision}' requires a CUDA GPU. CPU supports 'fp32' and 'bf16'."
        if precision == "bf16" and not hasattr(torch, "bfloat16"):
            return False, "BFloat16 is not supported by this PyTorch CPU build."
        return True, "Valid CPU configuration."

    # CUDA device validations
    if precision == "fp32":
        return True, "Valid FP32 configuration on CUDA."

    if precision == "fp16":
        return True, "Valid FP16 configuration on CUDA."

    if precision == "bf16":
        if not hw["bf16_supported"]:
            return False, (
                f"GPU {hw['device_name']} (Compute Capability {hw['compute_capability']}) "
                "does not natively support BFloat16. Requires Ampere (CC 8.0) or newer."
            )
        return True, "Valid BF16 configuration on CUDA."

    if precision == "int8":
        if not hw["bitsandbytes_available"]:
            return False, "BitsAndBytes is not installed or failed to initialize. Cannot load INT8."
        return True, "Valid BitsAndBytes INT8 configuration."

    if precision == "int4":
        if not hw["bitsandbytes_available"]:
            return False, "BitsAndBytes is not installed or failed to initialize. Cannot load INT4."
        return True, "Valid BitsAndBytes INT4 (NF4) configuration."

    return False, f"Unknown precision option '{precision}'. Supported: fp32, fp16, bf16, int8, int4."


# ---------------------------------------------------------------------------
# Quantization Configuration
# ---------------------------------------------------------------------------

def get_model_loading_kwargs(precision: str, device: str) -> Dict[str, Any]:
    """Generate model-loading kwargs for Hugging Face AutoModelForCausalLM.from_pretrained.

    Raises ValueError if configuration is invalid; never silently changes precision.
    """
    valid, err_msg = validate_configuration(precision, device)
    if not valid:
        raise ValueError(f"Incompatible configuration: {err_msg}")

    precision = precision.lower()
    device = device.lower()
    kwargs: Dict[str, Any] = {}

    if precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = "cuda:0" if device == "cuda" else "cpu"

    elif precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "cuda:0"

    elif precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "cuda:0" if device == "cuda" else "cpu"

    elif precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        kwargs["device_map"] = "auto"

    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"

    return kwargs


def quantize_model_cpu(model: Any) -> Any:
    """Apply PyTorch dynamic INT8 quantization to linear layers for CPU execution.

    Allows CPU-bound systems to run integer-quantized inference without CUDA or BitsAndBytes.
    """
    try:
        return torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    except Exception as e:
        raise RuntimeError(f"PyTorch dynamic CPU quantization failed: {e}") from e


# ---------------------------------------------------------------------------
# KV Cache Analysis
# ---------------------------------------------------------------------------

def calculate_kv_cache_size_bytes(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_length: int,
    batch_size: int = 1,
    bytes_per_elem: int = 2,
) -> int:
    """Calculate theoretical key-value cache memory footprint in bytes.

    Formula: 2 (Key + Value) * num_layers * num_heads * head_dim * seq_length * batch_size * bytes_per_elem
    """
    return 2 * num_layers * num_heads * head_dim * seq_length * batch_size * bytes_per_elem


def format_bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes (MB) rounded to two decimal places."""
    return round(size_bytes / (1024 ** 2), 2)


def run_kv_cache_comparison(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    device: str = "cuda",
    num_trials: int = 3,
) -> Dict[str, Any]:
    """Benchmark autoregressive inference with KV cache enabled vs disabled.

    Standard KV caching stores past key/value attention vectors in VRAM/RAM to prevent
    quadratic recomputation of prior tokens, trading additional memory for lower latency.
    """
    is_cuda = device == "cuda" and torch.cuda.is_available()
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = "cuda" if is_cuda else "cpu"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    results: Dict[str, Any] = {}

    for cache_flag in [True, False]:
        # Dedicated warm-up run for both configurations to prime CUDA kernels & allocators
        with torch.inference_mode():
            _ = model.generate(
                **inputs,
                max_new_tokens=4,
                use_cache=cache_flag,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if is_cuda:
            torch.cuda.synchronize()

        trial_latencies: List[float] = []
        peak_allocations: List[float] = []
        total_tokens_generated = 0
        last_generated_text = ""

        for _ in range(num_trials):
            gc.collect()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
            else:
                start_mem = psutil.Process().memory_info().rss

            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=cache_flag,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if is_cuda:
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated()
            else:
                peak_mem = psutil.Process().memory_info().rss
            t1 = time.perf_counter()

            trial_latencies.append(t1 - t0)
            peak_allocations.append(peak_mem - start_mem)
            total_tokens_generated = outputs.shape[-1] - inputs["input_ids"].shape[-1]
            last_generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

        med_lat = float(np.median(trial_latencies))
        mean_lat = float(np.mean(trial_latencies))
        med_incremental_mb = float(np.median(peak_allocations)) / (1024 ** 2)
        total_peak_mb = (peak_mem / (1024 ** 2)) if is_cuda else (peak_mem / (1024 ** 2))
        throughput = (total_tokens_generated / med_lat) if med_lat > 0 else 0.0

        key = "kv_cache_enabled" if cache_flag else "kv_cache_disabled"
        results[key] = {
            "use_cache": cache_flag,
            "median_latency_s": round(med_lat, 4),
            "mean_latency_s": round(mean_lat, 4),
            "tokens_generated": int(total_tokens_generated),
            "throughput_tokens_sec": round(throughput, 2),
            "incremental_memory_mb": round(max(0.0, med_incremental_mb), 2),
            "total_peak_memory_mb": round(total_peak_mb, 2),
            "text": last_generated_text,
        }

    enabled_lat = results["kv_cache_enabled"]["median_latency_s"]
    disabled_lat = results["kv_cache_disabled"]["median_latency_s"]
    speedup = round(disabled_lat / enabled_lat, 2) if enabled_lat > 0 else 1.0

    results["speedup_factor"] = speedup
    results["explanation"] = (
        f"Enabling KV cache achieved a {speedup}x speedup ({enabled_lat}s vs {disabled_lat}s median latency). "
        "Standard KV cache stores past key/value states in memory to avoid quadratic recomputation, "
        "trading runtime RAM/VRAM for significantly higher decoding throughput."
    )
    return results


# ---------------------------------------------------------------------------
# PEFT LoRA Support
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: Any,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Attach a PEFT Low-Rank Adaptation (LoRA) adapter to a causal language model."""
    if not HAS_PEFT:
        raise RuntimeError("PEFT is not installed. Install with 'pip install peft'.")

    # Discover and validate target linear modules against actual model architecture
    named_module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    if target_modules is None:
        candidate_names = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj", "query_key_value", "dense"]
        target_modules = [c for c in candidate_names if c in named_module_names]
        if not target_modules:
            # Fallback to standard attention projection if named discovery is ambiguous
            target_modules = ["q_proj", "v_proj"]
    else:
        # Validate user-specified target modules
        matched = [t for t in target_modules if any(t in full_name for full_name in named_module_names)]
        if not matched:
            raise ValueError(
                f"None of target_modules {target_modules} found in model architecture. "
                f"Available linear modules include: {sorted(list(named_module_names))[:10]}"
            )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    lora_model = get_peft_model(model, lora_config)
    trainable_params, all_params = lora_model.get_nb_trainable_parameters()
    trainable_pct = (trainable_params / all_params) * 100 if all_params > 0 else 0.0

    stats = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "trainable_percentage": round(trainable_pct, 4),
    }
    return lora_model, stats


def run_lightweight_sft_experiment(
    model: Any,
    tokenizer: Any,
    training_samples: List[Dict[str, str]],
    num_steps: int = 5,
    learning_rate: float = 1e-4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute a lightweight Supervised Fine-Tuning trial to measure LoRA training performance."""
    if not training_samples:
        raise ValueError("training_samples cannot be empty.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters. Apply a LoRA adapter before fine-tuning.")

    is_cuda = device == "cuda" and torch.cuda.is_available()
    model.train()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated()
    else:
        mem_start = psutil.Process().memory_info().rss

    loss_history: List[float] = []
    t0 = time.perf_counter()

    for step in range(num_steps):
        sample = training_samples[step % len(training_samples)]
        prompt_text = sample.get("prompt", "")
        ref_text = sample.get("reference", sample.get("response", ""))
        full_text = f"Prompt: {prompt_text}\nResponse: {ref_text}"

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=128)
        if is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loss_history.append(round(loss.item(), 4))

    if is_cuda:
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()
    else:
        mem_peak = psutil.Process().memory_info().rss

    t1 = time.perf_counter()
    model.eval()

    return {
        "num_steps": num_steps,
        "training_duration_seconds": round(t1 - t0, 3),
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "peak_training_memory_mb": round((mem_peak - mem_start) / (1024 ** 2), 2),
    }
