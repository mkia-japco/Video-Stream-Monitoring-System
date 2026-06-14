#!/usr/bin/env python3
"""
Local GPU SLA Profiler v2

Production-ready updates included:
1) Thread-safe VramSampler with threading.Lock
2) Batched NumPy fallback for large vector datasets
3) Explicit structured logging with optional log file and strict GPU monitor mode
4) Ollama /api/chat support for chat/instruct models, while keeping /api/generate support

Benchmarks:
- Device-wide VRAM via NVML + optional PyTorch CUDA stats
- Vector search latency over synthetic chunks using FAISS or NumPy fallback
- Local LLM inference through LM Studio, Ollama Chat, Ollama Generate, or OpenAI-compatible APIs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

logger = logging.getLogger("local_gpu_sla_profiler")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def now_perf() -> float:
    return time.perf_counter()


def bytes_to_gib(x: Optional[int]) -> Optional[float]:
    if x is None:
        return None
    return x / (1024 ** 3)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_values[int(k)]

    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def estimate_tokens_rough(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


@dataclass
class VramSnapshot:
    label: str
    timestamp_s: float
    nvml_used_gib: Optional[float] = None
    nvml_total_gib: Optional[float] = None
    torch_allocated_gib: Optional[float] = None
    torch_reserved_gib: Optional[float] = None
    torch_peak_allocated_gib: Optional[float] = None
    torch_peak_reserved_gib: Optional[float] = None


@dataclass
class VectorBenchmarkResult:
    backend: str
    chunks: int
    dim: int
    top_k: int
    runs: int
    warmup_runs: int
    numpy_batch_size: int
    build_time_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass
class LLMBenchmarkResult:
    backend: str
    model: str
    url: str
    prompt_chars: int
    max_tokens: int
    ttft_ms: Optional[float]
    total_time_ms: float
    generation_time_ms: Optional[float]
    output_tokens: Optional[int]
    tokens_per_second: Optional[float]
    metric_source: str
    response_preview: str


@dataclass
class SLAResult:
    retrieval_latency_target_ms: float
    llm_tokens_per_second_target: float
    vram_target_gib: float
    retrieval_pass: Optional[bool]
    llm_speed_pass: Optional[bool]
    vram_pass: Optional[bool]
    overall_pass: Optional[bool]


class GPUMonitor:
    def __init__(self, device_index: int = 0, strict: bool = False) -> None:
        self.device_index = device_index
        self.strict = strict
        self.nvml_ok = False
        self.nvml_handle = None
        self.torch_ok = bool(torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available())

        if torch is None:
            logger.warning("PyTorch is not available. torch.cuda memory stats will be disabled.")
        elif not self.torch_ok:
            logger.warning("PyTorch is available, but CUDA is not visible to torch.cuda.")

        if pynvml is None:
            msg = "pynvml/nvidia-ml-py is not available. Device-wide NVML monitoring is disabled."
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return

        try:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.nvml_ok = True
            logger.info("NVML initialized successfully for device index %s", device_index)
        except Exception:
            logger.exception("Failed to initialize NVML for GPU monitoring")
            self.nvml_ok = False
            self.nvml_handle = None
            if strict:
                raise

    def close(self) -> None:
        if pynvml is not None and self.nvml_ok:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shutdown completed")
            except Exception:
                logger.exception("Failed to shutdown NVML cleanly")
                if self.strict:
                    raise

    def reset_torch_peak(self) -> None:
        if not self.torch_ok:
            return
        try:
            torch.cuda.set_device(self.device_index)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device_index)
            torch.cuda.synchronize(self.device_index)
            logger.info("PyTorch CUDA peak memory stats reset")
        except Exception:
            logger.exception("Failed to reset PyTorch CUDA memory stats")
            if self.strict:
                raise

    def snapshot(self, label: str) -> VramSnapshot:
        nvml_used = None
        nvml_total = None

        if self.nvml_ok and self.nvml_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                nvml_used = bytes_to_gib(int(mem.used))
                nvml_total = bytes_to_gib(int(mem.total))
            except Exception:
                logger.exception("Failed to read NVML memory snapshot: %s", label)
                if self.strict:
                    raise

        allocated = reserved = peak_alloc = peak_res = None

        if self.torch_ok:
            try:
                allocated = bytes_to_gib(int(torch.cuda.memory_allocated(self.device_index)))
                reserved = bytes_to_gib(int(torch.cuda.memory_reserved(self.device_index)))
                peak_alloc = bytes_to_gib(int(torch.cuda.max_memory_allocated(self.device_index)))
                peak_res = bytes_to_gib(int(torch.cuda.max_memory_reserved(self.device_index)))
            except Exception:
                logger.exception("Failed to read PyTorch CUDA memory snapshot: %s", label)
                if self.strict:
                    raise

        return VramSnapshot(
            label=label,
            timestamp_s=time.time(),
            nvml_used_gib=nvml_used,
            nvml_total_gib=nvml_total,
            torch_allocated_gib=allocated,
            torch_reserved_gib=reserved,
            torch_peak_allocated_gib=peak_alloc,
            torch_peak_reserved_gib=peak_res,
        )


class VramSampler:
    def __init__(self, monitor: GPUMonitor, interval_s: float = 0.05) -> None:
        self.monitor = monitor
        self.interval_s = interval_s
        self.samples: List[VramSnapshot] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, label_prefix: str = "sample") -> None:
        def run() -> None:
            i = 0
            while not self._stop.is_set():
                try:
                    sample = self.monitor.snapshot(f"{label_prefix}_{i}")
                    with self._lock:
                        self.samples.append(sample)
                    i += 1
                except Exception:
                    logger.exception("VRAM sampler failed while collecting sample")
                    if self.monitor.strict:
                        break
                time.sleep(self.interval_s)

        self._stop.clear()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        logger.info("VRAM sampler started with interval %.3fs", self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("VRAM sampler thread did not stop within timeout")
            else:
                logger.info("VRAM sampler stopped cleanly")

    def sample_count(self) -> int:
        with self._lock:
            return len(self.samples)

    def get_samples_copy(self) -> List[VramSnapshot]:
        with self._lock:
            return list(self.samples)

    def peak_nvml_used_gib(self) -> Optional[float]:
        samples_copy = self.get_samples_copy()
        vals = [s.nvml_used_gib for s in samples_copy if s.nvml_used_gib is not None]
        return max(vals) if vals else None


def numpy_batched_search(vectors: np.ndarray, q: np.ndarray, top_k: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    best_scores = np.empty((0,), dtype=np.float32)
    best_ids = np.empty((0,), dtype=np.int64)
    n = vectors.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = vectors[start:end]
        block_scores = block @ q

        local_k = min(top_k, block_scores.shape[0])
        local_idx = np.argpartition(block_scores, -local_k)[-local_k:]
        local_scores = block_scores[local_idx]
        local_ids = local_idx.astype(np.int64) + start

        merged_scores = np.concatenate([best_scores, local_scores])
        merged_ids = np.concatenate([best_ids, local_ids])

        keep_k = min(top_k, merged_scores.shape[0])
        keep_idx = np.argpartition(merged_scores, -keep_k)[-keep_k:]

        best_scores = merged_scores[keep_idx]
        best_ids = merged_ids[keep_idx]

    order = np.argsort(best_scores)[::-1]
    return best_scores[order], best_ids[order]


def benchmark_vector_search(
    chunks: int,
    dim: int,
    top_k: int,
    runs: int,
    warmup_runs: int,
    seed: int,
    numpy_batch_size: int,
    force_numpy: bool,
) -> VectorBenchmarkResult:
    logger.info("Starting vector benchmark: chunks=%s dim=%s top_k=%s runs=%s warmup=%s", chunks, dim, top_k, runs, warmup_runs)

    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((chunks, dim), dtype=np.float32)
    queries = rng.standard_normal((runs + warmup_runs, dim), dtype=np.float32)

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    build_start = now_perf()

    if faiss is not None and not force_numpy:
        backend = "faiss.IndexFlatIP"
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        build_time_ms = (now_perf() - build_start) * 1000.0

        def search_one(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            q2 = q.reshape(1, -1).astype(np.float32, copy=False)
            scores, ids = index.search(q2, top_k)
            return scores, ids

        logger.info("Using FAISS IndexFlatIP backend")
    else:
        backend = "numpy.batched_bruteforce_fallback"
        build_time_ms = (now_perf() - build_start) * 1000.0

        def search_one(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return numpy_batched_search(vectors=vectors, q=q, top_k=top_k, batch_size=numpy_batch_size)

        if faiss is None:
            logger.warning("FAISS is not available. Using batched NumPy fallback.")
        if force_numpy:
            logger.warning("FAISS disabled by --force-numpy. Using batched NumPy fallback.")

    for i in range(warmup_runs):
        search_one(queries[i])

    latencies: List[float] = []
    for i in range(warmup_runs, warmup_runs + runs):
        start = now_perf()
        search_one(queries[i])
        latencies.append((now_perf() - start) * 1000.0)

    mean_ms = float(statistics.mean(latencies))
    p50_ms = float(percentile(latencies, 50) or 0.0)
    p95_ms = float(percentile(latencies, 95) or 0.0)
    min_ms = float(min(latencies))
    max_ms = float(max(latencies))
    std_ms = float(statistics.pstdev(latencies)) if len(latencies) > 1 else 0.0

    logger.info("Vector benchmark completed: mean=%.2fms p95=%.2fms", mean_ms, p95_ms)

    return VectorBenchmarkResult(
        backend=backend,
        chunks=chunks,
        dim=dim,
        top_k=top_k,
        runs=runs,
        warmup_runs=warmup_runs,
        numpy_batch_size=numpy_batch_size,
        build_time_ms=build_time_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
    )


def benchmark_lmstudio(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: int) -> LLMBenchmarkResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise local benchmark assistant. Answer briefly and deterministically."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    logger.info("Starting LM Studio benchmark: model=%s url=%s", model, url)
    start = now_perf()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except Exception:
        logger.exception("LM Studio HTTP request failed")
        raise
    total_time_ms = (now_perf() - start) * 1000.0

    if resp.status_code >= 400:
        logger.error("LM Studio returned HTTP %s: %s", resp.status_code, resp.text[:1000])
        raise RuntimeError(f"LM Studio request failed: HTTP {resp.status_code} - {resp.text[:1000]}")

    data = resp.json()
    stats = data.get("stats") or {}
    usage = data.get("usage") or {}

    content = ""
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except Exception:
        logger.debug("Could not read LM Studio content from choices[0].message.content", exc_info=True)
        try:
            content = data.get("message", {}).get("content", "") or ""
        except Exception:
            logger.exception("Could not parse LM Studio response content")
            content = ""

    ttft_s = safe_float(stats.get("time_to_first_token_seconds")) or safe_float(stats.get("time_to_first_token")) or safe_float(stats.get("ttft"))
    generation_time_s = safe_float(stats.get("generation_time_seconds")) or safe_float(stats.get("generation_time")) or safe_float(stats.get("time_generation"))
    tps = safe_float(stats.get("tokens_per_second")) or safe_float(stats.get("tokens_per_sec")) or safe_float(stats.get("tok_per_sec"))
    output_tokens = stats.get("total_output_tokens") or stats.get("output_tokens") or usage.get("completion_tokens")

    try:
        output_tokens = int(output_tokens) if output_tokens is not None else None
    except Exception:
        logger.exception("Failed to parse LM Studio output token count")
        output_tokens = None

    metric_source = "lmstudio_native_stats"
    if ttft_s is None and tps is None and output_tokens is None:
        metric_source = "wall_clock_with_rough_token_estimate"
        output_tokens = estimate_tokens_rough(content)
        if total_time_ms > 0:
            tps = output_tokens / (total_time_ms / 1000.0)

    logger.info("LM Studio benchmark completed: ttft=%s tps=%s", ttft_s, tps)

    return LLMBenchmarkResult(
        backend="lmstudio",
        model=model,
        url=url,
        prompt_chars=len(prompt),
        max_tokens=max_tokens,
        ttft_ms=ttft_s * 1000.0 if ttft_s is not None else None,
        total_time_ms=total_time_ms,
        generation_time_ms=generation_time_s * 1000.0 if generation_time_s is not None else None,
        output_tokens=output_tokens,
        tokens_per_second=tps,
        metric_source=metric_source,
        response_preview=content[:500].replace("\n", "\\n"),
    )


def benchmark_openai_compatible_stream(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: int) -> LLMBenchmarkResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise local benchmark assistant. Answer briefly and deterministically."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    logger.info("Starting OpenAI-compatible streaming benchmark: model=%s url=%s", model, url)
    start = now_perf()
    first_token_time: Optional[float] = None
    chunks: List[str] = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as resp:
            if resp.status_code >= 400:
                logger.error("OpenAI-compatible backend returned HTTP %s: %s", resp.status_code, resp.text[:1000])
                raise RuntimeError(f"OpenAI-compatible request failed: HTTP {resp.status_code} - {resp.text[:1000]}")

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except Exception:
                    logger.debug("Failed to parse OpenAI-compatible stream line: %s", line[:300], exc_info=True)
                    continue
                delta = ""
                try:
                    delta = event["choices"][0]["delta"].get("content") or ""
                except Exception:
                    logger.debug("OpenAI-compatible stream event did not include content delta", exc_info=True)
                if delta:
                    if first_token_time is None:
                        first_token_time = now_perf()
                    chunks.append(delta)
    except Exception:
        logger.exception("OpenAI-compatible streaming benchmark failed")
        raise

    end = now_perf()
    total_time_ms = (end - start) * 1000.0
    ttft_ms = (first_token_time - start) * 1000.0 if first_token_time is not None else None
    content = "".join(chunks)
    output_tokens = estimate_tokens_rough(content)
    tps = output_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else None

    return LLMBenchmarkResult(
        backend="openai-compatible-stream",
        model=model,
        url=url,
        prompt_chars=len(prompt),
        max_tokens=max_tokens,
        ttft_ms=ttft_ms,
        total_time_ms=total_time_ms,
        generation_time_ms=None,
        output_tokens=output_tokens,
        tokens_per_second=tps,
        metric_source="stream_timing_with_rough_token_estimate",
        response_preview=content[:500].replace("\n", "\\n"),
    )


def parse_ollama_final_metrics(final_event: Dict[str, Any], output_text: str, total_time_ms: float) -> Tuple[Optional[int], Optional[float], Optional[float], str]:
    eval_count = final_event.get("eval_count")
    eval_duration_ns = final_event.get("eval_duration")
    try:
        output_tokens = int(eval_count) if eval_count is not None else estimate_tokens_rough(output_text)
    except Exception:
        logger.exception("Failed to parse Ollama eval_count")
        output_tokens = estimate_tokens_rough(output_text)

    generation_time_ms = None
    tokens_per_second = None
    metric_source = "wall_clock_with_rough_token_estimate"

    if eval_count is not None and eval_duration_ns:
        try:
            generation_time_ms = float(eval_duration_ns) / 1e6
            tokens_per_second = float(eval_count) / (float(eval_duration_ns) / 1e9)
            metric_source = "ollama_native_eval_metrics"
        except Exception:
            logger.exception("Failed to parse Ollama native eval metrics")
    elif total_time_ms > 0:
        tokens_per_second = output_tokens / (total_time_ms / 1000.0)

    return output_tokens, generation_time_ms, tokens_per_second, metric_source


def benchmark_ollama_generate(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: int) -> LLMBenchmarkResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    logger.info("Starting Ollama generate benchmark: model=%s url=%s", model, url)
    start = now_perf()
    first_token_time: Optional[float] = None
    chunks: List[str] = []
    final_event: Dict[str, Any] = {}

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as resp:
            if resp.status_code >= 400:
                logger.error("Ollama generate returned HTTP %s: %s", resp.status_code, resp.text[:1000])
                raise RuntimeError(f"Ollama generate request failed: HTTP {resp.status_code} - {resp.text[:1000]}")
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except Exception:
                    logger.debug("Failed to parse Ollama generate stream line: %s", raw_line[:300], exc_info=True)
                    continue
                token = event.get("response") or ""
                if token:
                    if first_token_time is None:
                        first_token_time = now_perf()
                    chunks.append(token)
                if event.get("done") is True:
                    final_event = event
                    break
    except Exception:
        logger.exception("Ollama generate benchmark failed")
        raise

    end = now_perf()
    total_time_ms = (end - start) * 1000.0
    ttft_ms = (first_token_time - start) * 1000.0 if first_token_time is not None else None
    output_text = "".join(chunks)
    output_tokens, generation_time_ms, tps, metric_source = parse_ollama_final_metrics(final_event, output_text, total_time_ms)
    if metric_source == "ollama_native_eval_metrics":
        metric_source = "ollama_generate_native_eval_metrics"

    return LLMBenchmarkResult(
        backend="ollama-generate",
        model=model,
        url=url,
        prompt_chars=len(prompt),
        max_tokens=max_tokens,
        ttft_ms=ttft_ms,
        total_time_ms=total_time_ms,
        generation_time_ms=generation_time_ms,
        output_tokens=output_tokens,
        tokens_per_second=tps,
        metric_source=metric_source,
        response_preview=output_text[:500].replace("\n", "\\n"),
    )


def benchmark_ollama_chat(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: int) -> LLMBenchmarkResult:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise local benchmark assistant. Answer briefly and deterministically."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    logger.info("Starting Ollama chat benchmark: model=%s url=%s", model, url)
    start = now_perf()
    first_token_time: Optional[float] = None
    chunks: List[str] = []
    final_event: Dict[str, Any] = {}

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as resp:
            if resp.status_code >= 400:
                logger.error("Ollama chat returned HTTP %s: %s", resp.status_code, resp.text[:1000])
                raise RuntimeError(f"Ollama chat request failed: HTTP {resp.status_code} - {resp.text[:1000]}")
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except Exception:
                    logger.debug("Failed to parse Ollama chat stream line: %s", raw_line[:300], exc_info=True)
                    continue
                content = ""
                try:
                    content = event.get("message", {}).get("content") or ""
                except Exception:
                    logger.debug("Ollama chat stream event did not contain message.content", exc_info=True)
                if content:
                    if first_token_time is None:
                        first_token_time = now_perf()
                    chunks.append(content)
                if event.get("done") is True:
                    final_event = event
                    break
    except Exception:
        logger.exception("Ollama chat benchmark failed")
        raise

    end = now_perf()
    total_time_ms = (end - start) * 1000.0
    ttft_ms = (first_token_time - start) * 1000.0 if first_token_time is not None else None
    output_text = "".join(chunks)
    output_tokens, generation_time_ms, tps, metric_source = parse_ollama_final_metrics(final_event, output_text, total_time_ms)
    if metric_source == "ollama_native_eval_metrics":
        metric_source = "ollama_chat_native_eval_metrics"

    return LLMBenchmarkResult(
        backend="ollama-chat",
        model=model,
        url=url,
        prompt_chars=len(prompt),
        max_tokens=max_tokens,
        ttft_ms=ttft_ms,
        total_time_ms=total_time_ms,
        generation_time_ms=generation_time_ms,
        output_tokens=output_tokens,
        tokens_per_second=tps,
        metric_source=metric_source,
        response_preview=output_text[:500].replace("\n", "\\n"),
    )


def benchmark_llm(args: argparse.Namespace) -> LLMBenchmarkResult:
    if args.backend == "lmstudio":
        return benchmark_lmstudio(args.llm_url, args.model, args.prompt, args.max_tokens, args.temperature, args.timeout_s)
    if args.backend == "openai-compatible":
        return benchmark_openai_compatible_stream(args.llm_url, args.model, args.prompt, args.max_tokens, args.temperature, args.timeout_s)
    if args.backend in ("ollama", "ollama-generate"):
        return benchmark_ollama_generate(args.llm_url, args.model, args.prompt, args.max_tokens, args.temperature, args.timeout_s)
    if args.backend == "ollama-chat":
        return benchmark_ollama_chat(args.llm_url, args.model, args.prompt, args.max_tokens, args.temperature, args.timeout_s)
    raise ValueError(f"Unsupported backend: {args.backend}")


def get_environment_info(monitor: GPUMonitor) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_available": torch is not None,
        "torch_cuda_available": bool(torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "faiss_available": faiss is not None,
        "pynvml_available": pynvml is not None,
        "nvml_ok": monitor.nvml_ok,
    }

    if torch is not None:
        try:
            info["torch_version"] = torch.__version__
        except Exception:
            logger.exception("Failed to read torch version")
        try:
            if torch.cuda.is_available():
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_device_total_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            logger.exception("Failed to read torch CUDA device information")

    if monitor.nvml_ok and monitor.nvml_handle is not None:
        try:
            name = pynvml.nvmlDeviceGetName(monitor.nvml_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            info["nvml_device_name"] = name
            mem = pynvml.nvmlDeviceGetMemoryInfo(monitor.nvml_handle)
            info["nvml_total_memory_gib"] = mem.total / (1024 ** 3)
        except Exception:
            logger.exception("Failed to read NVML device information")
            if monitor.strict:
                raise

    return info


def make_sla(vector: VectorBenchmarkResult, llm: LLMBenchmarkResult, peak_vram_gib: Optional[float], retrieval_target_ms: float, llm_tps_target: float, vram_target_gib: float) -> SLAResult:
    retrieval_pass = vector.p95_ms <= retrieval_target_ms if vector.p95_ms is not None else None
    llm_speed_pass = llm.tokens_per_second >= llm_tps_target if llm.tokens_per_second is not None else None
    vram_pass = peak_vram_gib <= vram_target_gib if peak_vram_gib is not None else None
    known = [x for x in [retrieval_pass, llm_speed_pass, vram_pass] if x is not None]
    overall = all(known) if known else None
    return SLAResult(retrieval_target_ms, llm_tps_target, vram_target_gib, retrieval_pass, llm_speed_pass, vram_pass, overall)


def verdict(flag: Optional[bool]) -> str:
    if flag is True:
        return "PASS"
    if flag is False:
        return "FAIL"
    return "UNKNOWN"


def fmt_gib(x: Optional[float]) -> str:
    return f"{x:.3f} GiB" if x is not None else "N/A"


def fmt_ms(x: Optional[float]) -> str:
    return f"{x:.2f} ms" if x is not None else "N/A"


def fmt_num(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "N/A"


def print_report(env: Dict[str, Any], snapshots: List[VramSnapshot], peak_vram_gib: Optional[float], vector: VectorBenchmarkResult, llm: LLMBenchmarkResult, sla: SLAResult, notes: List[str]) -> None:
    line = "=" * 72
    print("\n" + line)
    print("LOCAL GPU SLA PROFILER REPORT")
    print(line)

    print("\n[Environment]")
    print(f"Platform: {env.get('platform', 'N/A')}")
    print(f"Python: {str(env.get('python', 'N/A')).split()[0]}")
    print(f"GPU (NVML): {env.get('nvml_device_name', 'N/A')}")
    print(f"GPU (Torch): {env.get('cuda_device_name', 'N/A')}")
    print(f"Total VRAM (NVML): {fmt_gib(env.get('nvml_total_memory_gib'))}")
    print(f"Total VRAM (Torch): {fmt_gib(env.get('cuda_device_total_memory_gib'))}")
    print(f"Torch CUDA Available: {env.get('torch_cuda_available')}")
    print(f"FAISS Available: {env.get('faiss_available')}")
    print(f"NVML Available/OK: {env.get('pynvml_available')} / {env.get('nvml_ok')}")

    print("\n[VRAM Snapshots]")
    for s in snapshots:
        print(f"{s.label:<28} | NVML used: {fmt_gib(s.nvml_used_gib):>12} | Torch allocated: {fmt_gib(s.torch_allocated_gib):>12} | Torch reserved: {fmt_gib(s.torch_reserved_gib):>12}")
    print(f"Peak Device VRAM During LLM: {fmt_gib(peak_vram_gib)}")

    print("\n[Vector Search]")
    print(f"Backend: {vector.backend}")
    print(f"Chunks: {vector.chunks:,}")
    print(f"Dimension: {vector.dim}")
    print(f"Top-K: {vector.top_k}")
    print(f"Runs/Warmup: {vector.runs}/{vector.warmup_runs}")
    print(f"NumPy Batch Size: {vector.numpy_batch_size}")
    print(f"Index Build Time: {vector.build_time_ms:.2f} ms")
    print(f"Mean Latency: {vector.mean_ms:.2f} ms")
    print(f"P50 Latency: {vector.p50_ms:.2f} ms")
    print(f"P95 Latency: {vector.p95_ms:.2f} ms")
    print(f"Min/Max Latency: {vector.min_ms:.2f} / {vector.max_ms:.2f} ms")
    print(f"Std Latency: {vector.std_ms:.2f} ms")

    print("\n[LLM Inference]")
    print(f"Backend: {llm.backend}")
    print(f"Model: {llm.model}")
    print(f"URL: {llm.url}")
    print(f"Prompt Chars: {llm.prompt_chars}")
    print(f"Max Tokens: {llm.max_tokens}")
    print(f"TTFT: {fmt_ms(llm.ttft_ms)}")
    print(f"Total Time: {fmt_ms(llm.total_time_ms)}")
    print(f"Generation Time: {fmt_ms(llm.generation_time_ms)}")
    print(f"Output Tokens: {llm.output_tokens if llm.output_tokens is not None else 'N/A'}")
    print(f"Tokens/sec: {fmt_num(llm.tokens_per_second)}")
    print(f"Metric Source: {llm.metric_source}")
    print(f"Response Preview: {llm.response_preview}")

    print("\n[SLA Verdict]")
    print(f"Retrieval P95 <= {sla.retrieval_latency_target_ms:.1f} ms: {verdict(sla.retrieval_pass)}")
    print(f"LLM Speed >= {sla.llm_tokens_per_second_target:.1f} tokens/sec: {verdict(sla.llm_speed_pass)}")
    print(f"Peak VRAM <= {sla.vram_target_gib:.1f} GiB: {verdict(sla.vram_pass)}")
    print(f"Overall: {verdict(sla.overall_pass)}")

    print("\n[Notes]")
    for note in notes:
        print(f"- {note}")
    print(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local GPU SLA Profiler for VRAM, vector search latency, and local LLM inference.")
    parser.add_argument("--backend", choices=["lmstudio", "openai-compatible", "ollama", "ollama-generate", "ollama-chat"], default="lmstudio", help="LLM backend. Use ollama-chat for /api/chat and ollama-generate for /api/generate.")
    parser.add_argument("--model", default="qwen3-coder-30b-a3b-instruct")
    parser.add_argument("--llm-url", default="http://localhost:1234/api/v0/chat/completions")
    parser.add_argument("--prompt", default=("You are benchmarking a local RAG system. Briefly explain how to reduce latency when YOLO vision inference, vector retrieval, and local LLM generation run on one RTX 3090."))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--chunks", type=int, default=31000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numpy-batch-size", type=int, default=8192, help="Batch size for NumPy brute-force fallback to avoid large temporary memory spikes.")
    parser.add_argument("--force-numpy", action="store_true", help="Force batched NumPy fallback even when FAISS is available.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--vram-sample-interval-s", type=float, default=0.05)
    parser.add_argument("--strict-gpu-monitor", action="store_true", help="Fail fast if NVML/CUDA monitoring fails instead of continuing with partial metrics.")
    parser.add_argument("--retrieval-sla-ms", type=float, default=150.0)
    parser.add_argument("--llm-sla-tps", type=float, default=15.0)
    parser.add_argument("--vram-sla-gib", type=float, default=24.0)
    parser.add_argument("--json-out", default="sla_report.json")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    logger.info("Local GPU SLA Profiler started")
    logger.info("Arguments: %s", vars(args))

    monitor = GPUMonitor(device_index=args.device_index, strict=args.strict_gpu_monitor)
    monitor.reset_torch_peak()

    snapshots: List[VramSnapshot] = []
    snapshots.append(monitor.snapshot("before_benchmark"))
    env = get_environment_info(monitor)

    notes: List[str] = [
        "Device-wide VRAM is measured via NVML when available.",
        "PyTorch CUDA memory stats are shown when torch.cuda is available.",
        "VramSampler uses threading.Lock around shared samples for thread-safe append/read.",
        "Vector search uses FAISS IndexFlatIP when available; otherwise batched NumPy fallback is used.",
        "NumPy fallback processes vectors in batches to reduce temporary memory spikes.",
        "LM Studio native stats are preferred for TTFT and tokens/sec when backend=lmstudio.",
        "Ollama supports both /api/generate via backend=ollama-generate and /api/chat via backend=ollama-chat.",
        "Use --strict-gpu-monitor for fail-fast behavior on NVML/CUDA monitoring errors.",
    ]

    try:
        vector = benchmark_vector_search(args.chunks, args.dim, args.top_k, args.query_runs, args.warmup_runs, args.seed, args.numpy_batch_size, args.force_numpy)
        snapshots.append(monitor.snapshot("after_vector_search"))

        sampler = VramSampler(monitor, interval_s=args.vram_sample_interval_s)
        sampler.start("during_llm")
        try:
            llm = benchmark_llm(args)
        finally:
            sampler.stop()

        logger.info("VRAM sampler collected %s samples", sampler.sample_count())
        snapshots.append(monitor.snapshot("after_llm_inference"))

        peak_vram = sampler.peak_nvml_used_gib()
        snapshot_peaks = [s.nvml_used_gib for s in snapshots if s.nvml_used_gib is not None]
        if peak_vram is None and snapshot_peaks:
            peak_vram = max(snapshot_peaks)
        elif peak_vram is not None and snapshot_peaks:
            peak_vram = max([peak_vram] + snapshot_peaks)

        sla = make_sla(vector, llm, peak_vram, args.retrieval_sla_ms, args.llm_sla_tps, args.vram_sla_gib)
        report = {
            "environment": env,
            "vram_snapshots": [asdict(s) for s in snapshots],
            "peak_vram_gib": peak_vram,
            "vector_search": asdict(vector),
            "llm_inference": asdict(llm),
            "sla": asdict(sla),
            "notes": notes,
        }

        print_report(env, snapshots, peak_vram, vector, llm, sla, notes)

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"JSON report written to: {args.json_out}")
            logger.info("JSON report written to %s", args.json_out)

        logger.info("Local GPU SLA Profiler finished successfully")
        return 0

    except Exception as e:
        logger.exception("Profiler failed")
        print("\nERROR:", str(e), file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("- Check that the selected LLM server is running and the model is loaded.", file=sys.stderr)
        print("- LM Studio models: curl http://localhost:1234/api/v0/models", file=sys.stderr)
        print("- Ollama chat: use --backend ollama-chat --llm-url http://localhost:11434/api/chat", file=sys.stderr)
        print("- Ollama generate: use --backend ollama-generate --llm-url http://localhost:11434/api/generate", file=sys.stderr)
        print("- Verify CUDA visibility using: nvidia-smi", file=sys.stderr)
        print("- If FAISS install fails, the script can still run with --force-numpy batched fallback.", file=sys.stderr)
        print("- For detailed logs, run with --log-level DEBUG --log-file profiler_debug.log", file=sys.stderr)
        return 1

    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
