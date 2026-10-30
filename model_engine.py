"""model_engine.py - Hugging Face model loading, device placement, precision, and inference."""

import gc
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from transformers.generation.streamers import BaseStreamer
except ImportError:
    try:
        from transformers import BaseStreamer
    except ImportError:
        class BaseStreamer:
            def put(self, value): pass
            def end(self): pass

from optimizer import get_hardware_capabilities, get_model_loading_kwargs, validate_configuration


class BenchmarkTokenStreamer(BaseStreamer):
    """Streamer that captures the exact arrival time of each generated token.

    Allows mathematically precise, hardware-synchronized measurement of:
    - Time To First Token (TTFT) / prefill latency
    - Inter-token latency / Time Per Output Token (TPOT)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, is_cuda: bool = False):
        self.tokenizer = tokenizer
        self.is_cuda = is_cuda
        self.token_timestamps: List[float] = []
        self.generated_token_ids: List[int] = []
        self.start_time: float = 0.0

    def start_timing(self) -> None:
        """Record inference launch timestamp with CUDA sync if applicable."""
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        self.token_timestamps = []
        self.generated_token_ids = []

    def put(self, value: Any) -> None:
        """Callback executed by generate() when new tokens are produced."""
        now = time.perf_counter()
        if isinstance(value, torch.Tensor):
            if value.dim() > 1:
                token_ids = value[0].tolist()
            else:
                token_ids = value.tolist()
            for tid in token_ids:
                self.generated_token_ids.append(tid)
                self.token_timestamps.append(now)

    def end(self) -> None:
        """Called when generation terminates."""
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_ttft(self) -> Optional[float]:
        """Time To First Token in seconds (first token timestamp - start timestamp)."""
        if self.token_timestamps and self.start_time > 0:
            return max(0.0, self.token_timestamps[0] - self.start_time)
        return None

    def get_tpot(self) -> Optional[float]:
        """Average Time Per Output Token in seconds across all subsequent tokens."""
        if len(self.token_timestamps) > 1:
            decode_duration = self.token_timestamps[-1] - self.token_timestamps[0]
            num_decode_tokens = len(self.token_timestamps) - 1
            return decode_duration / num_decode_tokens if num_decode_tokens > 0 else 0.0
        return None


class ModelEngine:
    """Manages Hugging Face model lifecycle, device placement, precision, and inference."""

    def __init__(self):
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model_id: Optional[str] = None
        self.precision: Optional[str] = None
        self.device: Optional[str] = None
        self.load_time_seconds: float = 0.0
        self.loaded_specs: Dict[str, Any] = {}

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded in memory."""
        return self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Extract and return architecture and memory metrics of the loaded model."""
        if not self.is_loaded():
            return {"status": "No model loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate memory footprint
        footprint_bytes = 0
        if hasattr(self.model, "get_memory_footprint"):
            footprint_bytes = self.model.get_memory_footprint()
        else:
            footprint_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        dtype_str = str(next(self.model.parameters()).dtype) if list(self.model.parameters()) else "Unknown"
        device_str = str(next(self.model.parameters()).device) if list(self.model.parameters()) else self.device

        return {
            "model_id": self.model_id,
            "precision": self.precision,
            "device": device_str,
            "dtype": dtype_str,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_count_million": round(total_params / 1e6, 2),
            "memory_footprint_mb": round(footprint_bytes / (1024 ** 2), 2),
            "memory_footprint_gb": round(footprint_bytes / (1024 ** 3), 3),
            "load_time_seconds": round(self.load_time_seconds, 3),
        }

    def load_model(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        precision: str = "fp16",
        device: str = "cuda",
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """Load a Hugging Face model and tokenizer with the requested precision and device.

        Never silently switches precision or device. Raises informative errors on failure.
        """
        # Validate configuration first
        valid, err_msg = validate_configuration(precision, device)
        if not valid:
            raise ValueError(f"Configuration rejected: {err_msg}")

        # If already loaded with identical config and not force_reload, return cached specs
        if (
            not force_reload
            and self.is_loaded()
            and self.model_id == model_id
            and self.precision == precision
            and self.device == device
        ):
            return self.get_model_info()

        # Clean up existing model before loading new one
        self.unload_model()

        kwargs = get_model_loading_kwargs(precision, device)

        t0 = time.perf_counter()

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side="left",
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for '{model_id}': {e}") from e

        # Load model weights
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                **kwargs,
            )
            model.eval()
        except Exception as e:
            self.unload_model()
            raise RuntimeError(
                f"Failed to load model '{model_id}' with precision '{precision}' on '{device}': {e}"
            ) from e

        t1 = time.perf_counter()
        self.load_time_seconds = t1 - t0

        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.precision = precision
        self.device = device

        self.loaded_specs = self.get_model_info()
        return self.loaded_specs

    def unload_model(self) -> None:
        """Purge model and tokenizer from memory and clear PyTorch CUDA caches."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.model_id = None
        self.precision = None
        self.device = None
        self.loaded_specs = {}

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Generate text from a prompt, capturing output text and timing metrics."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")

        is_cuda = self.device == "cuda" and torch.cuda.is_available()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        target_device = "cuda" if is_cuda else "cpu"
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        prompt_token_count = inputs["input_ids"].shape[-1]

        streamer = BenchmarkTokenStreamer(self.tokenizer, is_cuda=is_cuda)
        streamer.start_timing()

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": use_cache,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_latency = t1 - t0
        total_tokens = output_ids.shape[-1]
        new_tokens_count = total_tokens - prompt_token_count

        generated_text = self.tokenizer.decode(
            output_ids[0][prompt_token_count:],
            skip_special_tokens=True,
        )

        ttft = streamer.get_ttft()
        tpot = streamer.get_tpot()
        throughput = (new_tokens_count / total_latency) if total_latency > 0 else 0.0

        return {
            "generated_text": generated_text,
            "prompt_tokens": int(prompt_token_count),
            "generated_tokens": int(new_tokens_count),
            "total_tokens": int(total_tokens),
            "latency_seconds": round(total_latency, 4),
            "ttft_seconds": round(ttft, 4) if ttft is not None else None,
            "tpot_seconds": round(tpot, 4) if tpot is not None else None,
            "throughput_tokens_per_sec": round(throughput, 2),
        }


# Global singleton instance for Streamlit session sharing
_ENGINE_INSTANCE: Optional[ModelEngine] = None


def get_model_engine() -> ModelEngine:
    """Return singleton ModelEngine instance to preserve loaded model across reruns."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ModelEngine()
    return _ENGINE_INSTANCE
