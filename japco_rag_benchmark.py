#!/usr/bin/env python3
"""
japco_rag_benchmark.py

Local GPU SLA Profiler for a mixed YOLO + RAG + local LLM workload.

What it measures:
  1) Background YOLO-like FP16 GPU workload at ~1 FPS.
  2) Local Ollama-compatible LLM inference with strict JSON mode and thinking disabled by default.
  3) TTFT, tokens/sec, total latency, JSON parse/validation failures over repeated runs.
  4) VRAM and thermal snapshots via nvidia-smi and torch.cuda fallback.
  5) Vector-search latency over 31,000 synthetic chunks using FAISS if installed, otherwise NumPy.

Typical setup:
  pip install torch pydantic numpy
  pip install ultralytics        # optional, for real YOLO .pt simulation
  pip install faiss-cpu          # optional, for FAISS vector search

  # Make sure Ollama is running and the models are pulled, e.g.:
  # ollama pull qwen2.5:7b
  # ollama pull gemma2:9b
  # or use any locally executable model, e.g. qwen3.5:35b

Example:
  python japco_rag_benchmark.py \
    --models qwen2.5:7b gemma2:9b \
    --runs 10 \
    --yolo-model yolov8n.pt \
    --output sla_report.jsonl

Notes:
  - For exact GGUF quant variants, pass the exact model names/tags available in your local Ollama.
  - The script does not fake measurements. If Ollama/GPU/YOLO are unavailable, it reports the error.
  - For thinking-capable Ollama models, this script sends think=false so the final JSON response is not empty.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception as e:  # pragma: no cover
    raise RuntimeError("pydantic is required: pip install pydantic") from e


# -----------------------------
# Strict JSON schema definition
# -----------------------------

class DecisionJSON(BaseModel):
    """Rigid model output schema for the Deadly JSON Audit."""

    decision: Literal["allow", "review", "block"]
    risk_score: float = Field(ge=0.0, le=1.0)
    latency_budget_ms: int = Field(ge=1, le=60000)
    evidence: List[str] = Field(min_length=2, max_length=5)
    recommended_action: str = Field(min_length=3, max_length=200)


def pydantic_json_schema(model_cls: Any) -> Dict[str, Any]:
    """Support both Pydantic v1 and v2."""
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    return model_cls.schema()


def pydantic_validate_json(model_cls: Any, text: str) -> Any:
    """Support both Pydantic v1 and v2 JSON validation."""
    if hasattr(model_cls, "model_validate_json"):
        return model_cls.model_validate_json(text)
    return model_cls.parse_raw(text)


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class GpuSnapshot:
    timestamp: str
    label: str
    gpu_index: int
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]
    temperature_c: Optional[float]
    utilization_gpu_pct: Optional[float]
    power_w: Optional[float]
    torch_allocated_gb: Optional[float]
    torch_reserved_gb: Optional[float]


@dataclass
class VectorSearchResult:
    backend: str
    chunks: int
    dim: int
    k: int
    latency_ms: float


@dataclass
class LLMRunResult:
    model: str
    run_index: int
    ok: bool
    json_valid: bool
    validation_error: Optional[str]
    ttft_ms: Optional[float]
    total_ms_wall: Optional[float]
    total_duration_ms_ollama: Optional[float]
    load_duration_ms_ollama: Optional[float]
    prompt_eval_count: Optional[int]
    prompt_eval_duration_ms: Optional[float]
    eval_count: Optional[int]
    eval_duration_ms: Optional[float]
    tokens_per_sec: Optional[float]
    vector_search_ms: Optional[float]
    response_chars: int
    error: Optional[str]
    gpu_before: GpuSnapshot
    gpu_after: GpuSnapshot


# -----------------------------
# Utility functions
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ns_to_ms(x: Optional[int | float]) -> Optional[float]:
    if x is None:
        return None
    return float(x) / 1_000_000.0


def safe_float(x: str) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def get_gpu_snapshot(label: str, gpu_index: int = 0) -> GpuSnapshot:
    """Collect VRAM/temp/utilization via nvidia-smi and torch.cuda fallback."""
    vram_used_gb = vram_total_gb = temperature_c = utilization_gpu_pct = power_w = None

    if shutil.which("nvidia-smi"):
        query = (
            "memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw"
        )
        cmd = [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3)
            fields = [x.strip() for x in out.strip().split(",")]
            if len(fields) >= 5:
                mem_used_mb = safe_float(fields[0])
                mem_total_mb = safe_float(fields[1])
                temperature_c = safe_float(fields[2])
                utilization_gpu_pct = safe_float(fields[3])
                power_w = safe_float(fields[4])
                if mem_used_mb is not None:
                    vram_used_gb = mem_used_mb / 1024.0
                if mem_total_mb is not None:
                    vram_total_gb = mem_total_mb / 1024.0
        except Exception:
            pass

    torch_allocated_gb = torch_reserved_gb = None
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        try:
            torch_allocated_gb = torch.cuda.memory_allocated(gpu_index) / (1024**3)
            torch_reserved_gb = torch.cuda.memory_reserved(gpu_index) / (1024**3)
            # nvidia-smi may be unavailable; use torch for total memory fallback.
            if vram_total_gb is None:
                props = torch.cuda.get_device_properties(gpu_index)
                vram_total_gb = props.total_memory / (1024**3)
        except Exception:
            pass

    return GpuSnapshot(
        timestamp=now_iso(),
        label=label,
        gpu_index=gpu_index,
        vram_used_gb=vram_used_gb,
        vram_total_gb=vram_total_gb,
        temperature_c=temperature_c,
        utilization_gpu_pct=utilization_gpu_pct,
        power_w=power_w,
        torch_allocated_gb=torch_allocated_gb,
        torch_reserved_gb=torch_reserved_gb,
    )


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def post_json_stream(url: str, payload: Dict[str, Any], timeout: float = 600.0):
    """Yield decoded JSON lines from a streaming Ollama endpoint."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_bad_json_stream_line": line}


# -----------------------------
# Vector search benchmark
# -----------------------------

def build_vector_index(chunks: int, dim: int, seed: int = 123) -> Tuple[str, Any, np.ndarray]:
    """Create a synthetic vector DB for 31k chunks. FAISS if available; NumPy fallback."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((chunks, dim)).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors = vectors / norms

    try:
        import faiss  # type: ignore
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return "faiss.IndexFlatIP", index, vectors
    except Exception:
        return "numpy.dot+argpartition", None, vectors


def benchmark_vector_search(backend: str, index: Any, vectors: np.ndarray, dim: int, k: int) -> VectorSearchResult:
    rng = np.random.default_rng(random.randint(0, 1_000_000))
    q = rng.standard_normal((1, dim)).astype("float32")
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

    t0 = time.perf_counter_ns()
    if backend.startswith("faiss"):
        _dist, _idx = index.search(q, k)
    else:
        scores = vectors @ q[0]
        _idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        _idx = _idx[np.argsort(-scores[_idx])]
    t1 = time.perf_counter_ns()

    return VectorSearchResult(
        backend=backend,
        chunks=int(vectors.shape[0]),
        dim=dim,
        k=k,
        latency_ms=(t1 - t0) / 1_000_000.0,
    )


# -----------------------------
# YOLO / GPU background worker
# -----------------------------

class YoloBackgroundWorker:
    """Runs a light FP16 vision workload at roughly 1 FPS in the background."""

    def __init__(self, yolo_model: Optional[str], gpu_index: int, fps: float, imgsz: int, enabled: bool = True):
        self.yolo_model = yolo_model
        self.gpu_index = gpu_index
        self.fps = fps
        self.imgsz = imgsz
        self.enabled = enabled
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.errors: "queue.Queue[str]" = queue.Queue()
        self.iterations = 0
        self.backend = "disabled"

    def start(self) -> None:
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, name="yolo-background-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        period = 1.0 / max(self.fps, 1e-6)
        try:
            if self.yolo_model:
                self._run_ultralytics(period)
            else:
                self._run_dummy_torch(period)
        except Exception:
            self.errors.put(traceback.format_exc(limit=10))

    def _run_ultralytics(self, period: float) -> None:
        from ultralytics import YOLO  # type: ignore

        self.backend = f"ultralytics:{self.yolo_model}"
        model = YOLO(self.yolo_model)
        if torch is not None and torch.cuda.is_available():
            try:
                # Ultralytics also supports half=True in predict; this explicitly follows the requested model.half() style.
                model.model.to(f"cuda:{self.gpu_index}").half().eval()
            except Exception:
                pass

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            try:
                _ = model.predict(
                    source=dummy,
                    imgsz=self.imgsz,
                    device=self.gpu_index if torch is not None and torch.cuda.is_available() else "cpu",
                    half=True,
                    verbose=False,
                    stream=False,
                )
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize(self.gpu_index)
                self.iterations += 1
            except Exception:
                self.errors.put(traceback.format_exc(limit=5))
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def _run_dummy_torch(self, period: float) -> None:
        if torch is None or not torch.cuda.is_available():
            self.backend = "dummy-cpu-sleep-no-cuda"
            while not self.stop_event.is_set():
                time.sleep(period)
            return

        self.backend = "dummy-torch-fp16-conv"
        device = torch.device(f"cuda:{self.gpu_index}")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        ).to(device).half().eval()
        x = torch.randn(1, 3, self.imgsz, self.imgsz, device=device).half()

        with torch.no_grad():
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize(self.gpu_index)
                self.iterations += 1
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, period - elapsed))


# -----------------------------
# Ollama LLM benchmark
# -----------------------------

def make_prompt(vector_latency_ms: float) -> str:
    schema = pydantic_json_schema(DecisionJSON)
    return f"""
You are a local RAG safety auditor. Return ONLY one valid JSON object matching the schema below.
No markdown. No comments. No extra keys. Do not output reasoning or explanations.

Schema:
{json.dumps(schema, ensure_ascii=False)}

Context:
- A local RAG pipeline retrieved 5 chunks from a 31,000-chunk vector database.
- Current vector retrieval latency is {vector_latency_ms:.3f} ms.
- YOLO stream simulation is running at approximately 1 FPS on the same GPU.
- The task is to decide whether the pipeline is within a safe SLA.

Required decision logic:
- If retrieval latency is below 50 ms, prefer "allow" unless there are other risks.
- If latency is 50 to 200 ms, prefer "review".
- If latency is above 200 ms, prefer "block".
""".strip()


def run_ollama_once(
    model: str,
    run_index: int,
    ollama_url: str,
    keep_alive: str,
    num_predict: int,
    temperature: float,
    vector_latency_ms: float,
    gpu_index: int,
) -> LLMRunResult:
    gpu_before = get_gpu_snapshot(f"before_llm_{model}_run_{run_index}", gpu_index)
    prompt = make_prompt(vector_latency_ms)
    payload = {
        "model": model,
        "prompt": prompt,
        # Stable Ollama JSON mode. Pydantic still validates the rigid schema below.
        "format": "json",
        # Required for thinking-capable models such as qwen3.5:35b; otherwise
        # Ollama may emit reasoning tokens while leaving the final response empty.
        "think": False,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    response_parts: List[str] = []
    final_obj: Dict[str, Any] = {}
    ttft_ms: Optional[float] = None
    t0 = time.perf_counter_ns()
    error: Optional[str] = None
    ok = False

    try:
        for obj in post_json_stream(f"{ollama_url.rstrip('/')}/api/generate", payload):
            if "_bad_json_stream_line" in obj:
                continue
            chunk = obj.get("response") or ""
            if chunk and ttft_ms is None:
                ttft_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if chunk:
                response_parts.append(chunk)
            if obj.get("done") is True:
                final_obj = obj
                ok = True
                break
    except urllib.error.URLError as e:
        error = f"Ollama connection error: {e}"
    except Exception as e:
        error = f"Ollama runtime error: {type(e).__name__}: {e}"

    total_ms_wall = (time.perf_counter_ns() - t0) / 1_000_000.0
    response_text = "".join(response_parts).strip()

    json_valid = False
    validation_error = None
    if response_text:
        try:
            _ = pydantic_validate_json(DecisionJSON, response_text)
            json_valid = True
        except Exception as e:
            validation_error = str(e)[:1000]
    elif error is None:
        validation_error = "empty response; if eval_count > 0, check whether the model produced thinking-only output or disable thinking"

    eval_count = final_obj.get("eval_count")
    eval_duration_ns = final_obj.get("eval_duration")
    tokens_per_sec = None
    if isinstance(eval_count, int) and isinstance(eval_duration_ns, int) and eval_duration_ns > 0:
        tokens_per_sec = eval_count / (eval_duration_ns / 1_000_000_000.0)

    gpu_after = get_gpu_snapshot(f"after_llm_{model}_run_{run_index}", gpu_index)

    return LLMRunResult(
        model=model,
        run_index=run_index,
        ok=ok,
        json_valid=json_valid,
        validation_error=validation_error,
        ttft_ms=ttft_ms,
        total_ms_wall=total_ms_wall,
        total_duration_ms_ollama=ns_to_ms(final_obj.get("total_duration")),
        load_duration_ms_ollama=ns_to_ms(final_obj.get("load_duration")),
        prompt_eval_count=final_obj.get("prompt_eval_count"),
        prompt_eval_duration_ms=ns_to_ms(final_obj.get("prompt_eval_duration")),
        eval_count=eval_count,
        eval_duration_ms=ns_to_ms(eval_duration_ns),
        tokens_per_sec=tokens_per_sec,
        vector_search_ms=vector_latency_ms,
        response_chars=len(response_text),
        error=error,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
    )


# -----------------------------
# Reporting
# -----------------------------

def summarize_model(results: List[LLMRunResult]) -> Dict[str, Any]:
    def vals(name: str) -> List[float]:
        out = []
        for r in results:
            v = getattr(r, name)
            if isinstance(v, (int, float)) and math.isfinite(v):
                out.append(float(v))
        return out

    def mean_std(xs: List[float]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"mean": None, "std": None, "min": None, "max": None}
        arr = np.array(xs, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    n = len(results)
    ok = sum(1 for r in results if r.ok)
    valid = sum(1 for r in results if r.json_valid)

    return {
        "runs": n,
        "successful_api_runs": ok,
        "api_failure_rate": None if n == 0 else (n - ok) / n,
        "json_valid_runs": valid,
        "json_failure_rate": None if n == 0 else (n - valid) / n,
        "ttft_ms": mean_std(vals("ttft_ms")),
        "tokens_per_sec": mean_std(vals("tokens_per_sec")),
        "total_ms_wall": mean_std(vals("total_ms_wall")),
        "ollama_total_duration_ms": mean_std(vals("total_duration_ms_ollama")),
        "vector_search_ms": mean_std(vals("vector_search_ms")),
        "vram_after_gb": mean_std([
            r.gpu_after.vram_used_gb for r in results if isinstance(r.gpu_after.vram_used_gb, (int, float))
        ]),
        "temperature_after_c": mean_std([
            r.gpu_after.temperature_c for r in results if isinstance(r.gpu_after.temperature_c, (int, float))
        ]),
    }


def print_human_summary(all_results: Dict[str, List[LLMRunResult]], yolo: YoloBackgroundWorker, vector_backend: str) -> None:
    print("\n" + "=" * 80)
    print("LOCAL GPU SLA PROFILER - SUMMARY")
    print("=" * 80)
    print(f"Host: {platform.node()} | OS: {platform.platform()} | Python: {platform.python_version()}")
    print(f"Torch: {getattr(torch, '__version__', 'not-installed') if torch is not None else 'not-installed'}")
    if torch is not None and torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"YOLO worker backend: {yolo.backend}; iterations: {yolo.iterations}; errors: {yolo.errors.qsize()}")
    print(f"Vector search backend: {vector_backend}")
    print("-" * 80)

    for model, results in all_results.items():
        s = summarize_model(results)
        print(f"\nModel: {model}")
        print(f"  API success: {s['successful_api_runs']}/{s['runs']} | JSON valid: {s['json_valid_runs']}/{s['runs']}")
        print(f"  JSON failure rate: {s['json_failure_rate']:.2%}" if s['json_failure_rate'] is not None else "  JSON failure rate: n/a")
        for key, label, unit in [
            ("ttft_ms", "TTFT", "ms"),
            ("tokens_per_sec", "Decode speed", "tok/s"),
            ("total_ms_wall", "Total wall time", "ms"),
            ("vector_search_ms", "Vector search", "ms"),
            ("vram_after_gb", "VRAM after", "GB"),
            ("temperature_after_c", "GPU temp after", "C"),
        ]:
            stat = s[key]
            if stat["mean"] is None:
                print(f"  {label}: n/a")
            else:
                print(
                    f"  {label}: {stat['mean']:.2f} ± {stat['std']:.2f} {unit} "
                    f"(min={stat['min']:.2f}, max={stat['max']:.2f})"
                )
    print("=" * 80 + "\n")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local GPU SLA Profiler for YOLO+RAG+LLM workloads")
    p.add_argument("--models", nargs="+", default=["qwen2.5:7b", "gemma2:9b"], help="Ollama model names/tags")
    p.add_argument("--runs", type=int, default=10, help="Runs per model")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--keep-alive", default="10m", help="Ollama keep_alive value")
    p.add_argument("--num-predict", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu-index", type=int, default=0)

    p.add_argument("--yolo-model", default=None, help="Path/name of YOLO .pt model, e.g. yolov8n.pt or yolo11n.pt")
    p.add_argument("--disable-yolo", action="store_true", help="Disable background vision workload")
    p.add_argument("--yolo-fps", type=float, default=1.0)
    p.add_argument("--yolo-imgsz", type=int, default=640)

    p.add_argument("--chunks", type=int, default=31_000)
    p.add_argument("--vector-dim", type=int, default=384)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="sla_report.jsonl", help="JSONL output path")
    p.add_argument("--summary-output", default="sla_summary.json", help="JSON summary output path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Clear previous outputs only if they exist, to avoid mixing runs.
    for path in [args.output, args.summary_output]:
        if os.path.exists(path):
            os.remove(path)

    metadata = {
        "type": "metadata",
        "timestamp": now_iso(),
        "args": vars(args),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(args.gpu_index) if torch is not None and torch.cuda.is_available() else None,
        "initial_gpu": asdict(get_gpu_snapshot("initial", args.gpu_index)),
    }
    write_jsonl(args.output, metadata)

    print("Building synthetic vector index...")
    vector_backend, vector_index, vectors = build_vector_index(args.chunks, args.vector_dim, args.seed)
    print(f"Vector backend: {vector_backend}; chunks={args.chunks}; dim={args.vector_dim}")

    yolo_worker = YoloBackgroundWorker(
        yolo_model=args.yolo_model,
        gpu_index=args.gpu_index,
        fps=args.yolo_fps,
        imgsz=args.yolo_imgsz,
        enabled=not args.disable_yolo,
    )

    all_results: Dict[str, List[LLMRunResult]] = {m: [] for m in args.models}

    try:
        print("Starting background YOLO/vision simulation...")
        yolo_worker.start()
        time.sleep(2.0)  # warm-up background load

        for model in args.models:
            print(f"\nBenchmarking model: {model}")
            # Optional warm-up request to reduce cold-start bias in repeated runs.
            for run_idx in range(1, args.runs + 1):
                vs = benchmark_vector_search(vector_backend, vector_index, vectors, args.vector_dim, args.top_k)
                result = run_ollama_once(
                    model=model,
                    run_index=run_idx,
                    ollama_url=args.ollama_url,
                    keep_alive=args.keep_alive,
                    num_predict=args.num_predict,
                    temperature=args.temperature,
                    vector_latency_ms=vs.latency_ms,
                    gpu_index=args.gpu_index,
                )
                all_results[model].append(result)
                write_jsonl(args.output, {"type": "run", **asdict(result)})

                status = "OK" if result.ok else "ERR"
                jstatus = "JSON_OK" if result.json_valid else "JSON_FAIL"
                print(
                    f"  run {run_idx:02d}/{args.runs} [{status}/{jstatus}] "
                    f"TTFT={result.ttft_ms if result.ttft_ms is not None else 'n/a'} ms, "
                    f"TPS={result.tokens_per_sec if result.tokens_per_sec is not None else 'n/a'}, "
                    f"Vec={vs.latency_ms:.3f} ms"
                )

    finally:
        print("Stopping background YOLO/vision simulation...")
        yolo_worker.stop()

    summary = {
        "type": "summary",
        "timestamp": now_iso(),
        "yolo_backend": yolo_worker.backend,
        "yolo_iterations": yolo_worker.iterations,
        "yolo_errors": list(yolo_worker.errors.queue),
        "vector_backend": vector_backend,
        "models": {model: summarize_model(results) for model, results in all_results.items()},
        "final_gpu": asdict(get_gpu_snapshot("final", args.gpu_index)),
    }

    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_jsonl(args.output, summary)

    print_human_summary(all_results, yolo_worker, vector_backend)
    print(f"Detailed JSONL report: {args.output}")
    print(f"Summary JSON report: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
