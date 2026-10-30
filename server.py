"""server.py - Lightweight REST API server for Quantora Web Dashboard.

Uses Python standard library (http.server) with zero external server dependencies.
Provides real-time endpoints for model loading, benchmark execution, KV cache
analysis, Pareto recommendations, and history queries.
"""

import json
import os
import sys
import time
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict

# Ensure utf-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PORT = int(os.environ.get("QUANTORA_PORT", 8000))
_benchmark_engine = None


def get_benchmark_engine():
    global _benchmark_engine
    if _benchmark_engine is None:
        from benchmark_engine import BenchmarkEngine
        _benchmark_engine = BenchmarkEngine()
    return _benchmark_engine


class QuantoraAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing CORS-enabled REST endpoints for Quantora."""

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status: int, data: Dict[str, Any]):
        response_bytes = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        return {}

    def do_GET(self):
        url = self.path.split("?")[0]

        if url == "/api/health":
            from optimizer import get_hardware_capabilities
            hw = get_hardware_capabilities()
            self._send_json(HTTPStatus.OK, {
                "status": "online",
                "service": "Quantora ML Systems API",
                "cuda_available": hw.get("cuda_available", False),
                "device_name": hw.get("device_name", "CPU"),
            })
            return

        if url == "/api/hardware":
            from optimizer import get_hardware_capabilities
            hw = get_hardware_capabilities()
            self._send_json(HTTPStatus.OK, hw)
            return

        if url == "/api/history":
            from storage import load_experiments
            df = load_experiments()
            records = df.to_dict(orient="records") if not df.empty else []
            self._send_json(HTTPStatus.OK, {"count": len(records), "records": records})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint not found: {self.path}"})

    def do_POST(self):
        url = self.path.split("?")[0]

        try:
            body = self._read_json_body()

            if url == "/api/validate":
                from optimizer import validate_configuration
                precision = body.get("precision", "fp16")
                device = body.get("device", "cuda")
                valid, reason = validate_configuration(precision, device)
                self._send_json(HTTPStatus.OK, {"valid": valid, "reason": reason})
                return

            if url == "/api/benchmark":
                from optimizer import validate_configuration
                from model_engine import get_model_engine

                model_id = body.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
                precision = body.get("precision", "fp16")
                prompt = body.get("prompt", "Summarize the computational advantages of integer quantization.")
                prompt_id = body.get("prompt_id", "web_benchmark")
                reference = body.get("reference_answer")
                max_tokens = int(body.get("max_tokens", 64))
                warmup = int(body.get("warmup", 1))
                trials = int(body.get("trials", 2))
                device = body.get("device", "cuda")

                valid, reason = validate_configuration(precision, device)
                if not valid:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": reason})
                    return

                engine = get_model_engine()
                engine.load_model(model_id=model_id, precision=precision, device=device)

                bench_engine = get_benchmark_engine()
                results = bench_engine.run_benchmark(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    reference_answer=reference,
                    max_new_tokens=max_tokens,
                    warmup_runs=warmup,
                    num_trials=trials,
                )

                self._send_json(HTTPStatus.OK, {
                    "success": True,
                    "model_id": model_id,
                    "precision": precision,
                    "results": results,
                })
                return

            if url == "/api/generate":
                from model_engine import get_model_engine

                model_id = body.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
                precision = body.get("precision", "fp16")
                prompt = body.get("prompt", "Hello")
                max_tokens = int(body.get("max_tokens", 64))
                temperature = float(body.get("temperature", 0.7))
                device = body.get("device", "cuda")

                engine = get_model_engine()
                engine.load_model(model_id=model_id, precision=precision, device=device)

                t0 = time.perf_counter()
                text = engine.generate_text(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                )
                duration = time.perf_counter() - t0

                self._send_json(HTTPStatus.OK, {
                    "generated_text": text,
                    "duration_s": round(duration, 3),
                    "model_id": model_id,
                    "precision": precision,
                })
                return

            if url == "/api/recommend":
                from storage import recommend_best_configuration

                max_mem = float(body.get("max_memory_mb", 2000.0))
                max_lat = float(body.get("max_latency_s", 3.0))
                min_q = float(body.get("min_quality_score", 0.0)) if body.get("min_quality_score") else None

                rec = recommend_best_configuration(
                    max_memory_mb=max_mem,
                    max_latency_s=max_lat,
                    min_quality_score=min_q,
                )
                self._send_json(HTTPStatus.OK, {"recommendation": rec})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint not found: {self.path}"})

        except Exception as e:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": str(e),
                "type": type(e).__name__,
            })

    def log_message(self, format, *args):
        sys.stderr.write(f"[Quantora API] {self.address_string()} - {format % args}\n")


def run_server(port: int = PORT):
    server = HTTPServer(("0.0.0.0", port), QuantoraAPIHandler)
    print(f"\n============================================================")
    print(f"[QUANTORA] LIVE API SERVER ONLINE")
    print(f"============================================================")
    print(f"  Local Endpoint  : http://localhost:{port}")
    print(f"  Health Check    : http://localhost:{port}/api/health")
    print(f"  Hardware Probe  : http://localhost:{port}/api/hardware")
    print(f"  Benchmark Run   : POST http://localhost:{port}/api/benchmark")
    print(f"============================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    run_server()
