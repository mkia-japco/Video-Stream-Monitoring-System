# Local GPU SLA Profiler

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX%203090-green.svg)]()
[![Backend](https://img.shields.io/badge/LLM-LM%20Studio%20%7C%20Ollama%20%7C%20OpenAI--Compatible-purple.svg)]()
[![Status](https://img.shields.io/badge/SLA-PASS-brightgreen.svg)]()

A standalone Python benchmark tool for profiling **local GPU VRAM usage**, **vector search latency**, and **local LLM inference speed** on single-GPU systems such as **RTX 3090**.

This project is designed for MVP-stage local/offline AI systems where **Computer Vision**, **RAG retrieval**, and **local LLM inference** may compete for limited GPU memory and compute resources.

---

## Table of Contents

- [Overview](#overview)
- [Why This Project Exists](#why-this-project-exists)
- [Project Status (V2)](#project-status-v2)
- [Installation](#installation)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create a virtual environment](#2-create-a-virtual-environment)
  - [3. Install dependencies](#3-install-dependencies)
- [LM Studio Setup](#lm-studio-setup)
  - [Steps](#steps)
- [What This Profiler Measures](#what-this-profiler-measures)
  - [1. GPU VRAM Usage](#1-gpu-vram-usage)
  - [2. Vector Search Latency](#2-vector-search-latency)
  - [3. Local LLM Inference Speed](#3-local-llm-inference-speed)
- [Current Version Documentation (V2)](#current-version-documentation-v2)
  - [What This Profiler Measures (V2)](#what-this-profiler-measures-v2)
  - [v2 Improvements](#v2-improvements)
  - [Repository Structure (V2)](#repository-structure-v2)
  - [Latest Benchmark Summary (V2)](#latest-benchmark-summary-v2)
  - [Usage Script (V2)](#usage-script-v2)
  - [Ollama Usage  (V2)](#ollama-usage-v2)
  - [CLI Options (V2)](#cli-options-v2)
  - [Output Files (V2)](#output-files-v2)
  - [Why NVML and PyTorch Stats Are Both Shown](#why-nvml-and-pytorch-stats-are-both-shown)
  - [Why Batched NumPy Fallback Exists](#why-batched-numpy-fallback-exists)
  - [Result Documentation](#result-documentation)
  - [Interpreting the Latest Results](#interpreting-the-latest-results)
- [Historical Documentation (V1)](#historical-documentation-v1)
  - [Tested Hardware](#tested-hardware)
  - [Benchmark Result Summary (v1)](#benchmark-result-summary-v1)
  - [Project Structure (V1)](#project-structure-v1)
  - [How to Run (V1)](#how-to-run-v1)
  - [Example SLA Report](#example-sla-report)
  - [Output Files (V1)](#output-files-v1)
  - [CLI Arguments (V1)](#cli-arguments-v1)
- [Metrics and Technical Notes](#metrics-and-technical-notes)
  - [How Metrics Are Calculated](#how-metrics-are-calculated)
  - [Important Technical Notes](#important-technical-notes)
- [Final Notes](#final-notes)

---

## Overview

`local_gpu_sla_profiler.py` is a lightweight benchmark script that helps answer three practical questions:

1. **How much GPU memory is being used?**
2. **How fast is the local vector retrieval layer?**
3. **How fast is the local LLM inference backend?**

The profiler prints a clean terminal report and also exports a structured JSON report for future comparison.

It is especially useful for local AI systems running on a single GPU, where the same GPU may need to support:

- Vision inference
- Embedding generation
- Vector retrieval
- Reranking
- Local LLM generation

---

## Why This Project Exists

In local AI systems, latency problems can come from different layers:

- GPU memory pressure
- Vector database search
- LLM inference
- Model loading
- CPU bottlenecks
- Concurrent execution between vision and language workloads

This profiler provides a first-stage SLA audit for local deployment environments. It does not try to replace full production observability tools, but it gives a fast and practical view of core performance metrics.

---

## Project Status (V2)

Current recommended version:

```text
local_gpu_sla_profiler.py
```

The `main` branch contains the latest v2 implementation.

Historical branches:

| Branch                        | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| `main`                        | Latest stable implementation, currently v2    |
| `feature/company-feedback-v1` | Initial profiler version                      |
| `feature/company-feedback-v2` | v2 implementation based on technical feedback |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mkia-japco/Video-Stream-Monitoring-System.git
cd local-gpu-sla-profiler
```

### 2. Create a virtual environment

If PyTorch with CUDA is already installed and working on your system, using `--system-site-packages` can help preserve the existing CUDA-enabled PyTorch installation:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
requests
torch
faiss-cpu
nvidia-ml-py
```

---

## LM Studio Setup

This benchmark can run against LM Studio as a local LLM server.

### Steps

1. Open LM Studio.
2. Load your local model.
3. Open the **Developer** tab.
4. Start the local server.
5. Confirm that the server is running:

```bash
curl http://localhost:1234/api/v0/models
```

Default LM Studio endpoint used by this project: (V2)

```bash
http://localhost:1234/api/v0/chat/completions
```

If the model is loaded correctly, this command should return a list of available local models.

---

## What This Profiler Measures

### 1. GPU VRAM Usage

The profiler captures GPU memory at multiple points:

- Before benchmark execution
- After vector search benchmark
- During LLM inference
- After LLM inference
- Peak device VRAM during inference

It uses:

- **NVML / nvidia-ml-py** for device-wide GPU memory monitoring
- **torch.cuda** for PyTorch CUDA memory stats when available

This distinction is important because local inference backends such as **LM Studio**, **Ollama**, or **llama.cpp server** often run in a separate process. In that case, `torch.cuda` inside the profiler process may not see the memory used by the external LLM runtime, while NVML can still capture device-wide GPU memory usage.

---

### 2. Vector Search Latency

The profiler creates a synthetic vector database with a configurable number of chunks.

Default:

```text
chunks = 31,000
dimension = 768
top_k = 100
```

It benchmarks semantic retrieval using:

- **FAISS IndexFlatIP** when FAISS is installed
- NumPy brute-force fallback if FAISS is unavailable

Reported metrics:

- Mean latency
- P50 latency
- P95 latency
- Min latency
- Max latency
- Standard deviation

---

### 3. Local LLM Inference Speed

The profiler sends a test prompt to a local LLM backend and measures:

- **TTFT** — Time To First Token
- **Total response time**
- **Generation time**
- **Output token count**
- **Tokens per second**

Supported backends:

- `lmstudio`
- `ollama`
- `openai-compatible`

For LM Studio, the profiler uses the local REST API endpoint:

```text
http://localhost:1234/api/v0/chat/completions
```

When native backend stats are available, the profiler reports them directly and marks:

```text
Metric Source: lmstudio_native_stats
```

---

## Current Version Documentation (V2)

### What This Profiler Measures (V2)

The script benchmarks three key areas:

1. **VRAM Monitoring**

   * Measures device-wide GPU memory through NVML.
   * Shows PyTorch CUDA memory stats when available.
   * Samples VRAM during LLM inference to capture peak usage.

2. **Vector Search Latency**

   * Builds a synthetic local vector database.
   * Benchmarks retrieval latency over configurable chunk counts.
   * Uses FAISS `IndexFlatIP` when available.
   * Falls back to batched NumPy brute-force search when FAISS is unavailable or disabled.

3. **Local LLM Inference Speed**

   * Sends a test prompt to a local LLM runtime.
   * Supports LM Studio, Ollama, and OpenAI-compatible local endpoints.
   * Measures:

     * Time To First Token, TTFT
     * Total response time
     * Generation time
     * Output tokens
     * Tokens per second

---

### v2 Improvements

Version 2 includes the following production-readiness improvements:

* Thread-safe `VramSampler` using `threading.Lock`
* Batched NumPy fallback to reduce temporary memory spikes
* Structured logging with `--log-level` and `--log-file`
* Optional fail-fast GPU monitoring with `--strict-gpu-monitor`
* Ollama `/api/chat` support via `--backend ollama-chat`
* Ollama `/api/generate` support via `--backend ollama-generate`
* Separate benchmark result directories under `examples/v2`
* Updated benchmark report under `docs/BENCHMARK_RESULTS.md`

---

### Repository Structure (V2)

```text
local-gpu-sla-profiler/
├── CHANGELOG.md
├── docs/
│   └── BENCHMARK_RESULTS.md
├── examples/
│   ├── v1/
│   │   ├── gpu_after_profiler.csv
│   │   ├── gpu_before_model_load.csv
│   │   ├── SLA_EXECUTION_REPORT_256.md
│   │   ├── sla_log_lmstudio_256.txt
│   │   └── sla_report_lmstudio_256.json
│   └── v2/
│       ├── faiss-31k/
│       │   ├── profiler_v2.log
│       │   ├── sla_log_lmstudio_v2.txt
│       │   └── sla_report_lmstudio_v2.json
│       ├── faiss-31k-strict-debug/
│       │   ├── profiler_v2_debug.log
│       │   ├── sla_log_lmstudio_v2_debug.txt
│       │   └── sla_report_lmstudio_v2_debug.json
│       ├── numpy-batched-31k/
│       │   ├── profiler_numpy_batched.log
│       │   ├── sla_log_numpy_batched.txt
│       │   └── sla_report_numpy_batched.json
│       └── numpy-batched-100k/
│           ├── profiler_numpy_batched_100k.log
│           ├── sla_log_numpy_batched_100k.txt
│           └── sla_report_numpy_batched_100k.json
├── .gitignore
├── LICENSE
├── local_gpu_sla_profiler.py
├── README.md
└── requirements.txt
```

---

### Latest Benchmark Summary (V2)

All tested configurations passed the defined SLA thresholds.

| Scenario               | Retrieval P95 | LLM TTFT |         LLM Speed |  Peak VRAM | Verdict |
| ---------------------- | ------------: | -------: | ----------------: | ---------: | ------- |
| FAISS 31k              |       4.05 ms |    34 ms |  99.20 tokens/sec | 18.424 GiB | PASS    |
| FAISS 31k Strict Debug |       4.04 ms |    45 ms |  99.61 tokens/sec | 18.416 GiB | PASS    |
| NumPy Batched 31k      |       3.06 ms |    58 ms | 100.71 tokens/sec | 18.422 GiB | PASS    |
| NumPy Batched 100k     |      27.54 ms |    66 ms |  98.37 tokens/sec | 18.422 GiB | PASS    |

SLA targets:

| Metric                |           Target |
| --------------------- | ---------------: |
| Retrieval P95 Latency |        <= 150 ms |
| LLM Inference Speed   | >= 15 tokens/sec |
| Peak Device VRAM      |        <= 24 GiB |

Full result details are available in:

```text
docs/BENCHMARK_RESULTS.md
```

---

### Usage Script (V2)

#### 1. Standard v2 FAISS 31k benchmark  (V2)

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out examples/v2/faiss-31k/sla_report_lmstudio_v2.json \
  --log-level INFO \
  --log-file examples/v2/faiss-31k/profiler_v2.log \
  2>&1 | tee examples/v2/faiss-31k/sla_log_lmstudio_v2.txt
```

---

#### 2. Strict debug run  (V2)

This run verifies the strict GPU-monitoring and debug logging path.

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out examples/v2/faiss-31k-strict-debug/sla_report_lmstudio_v2_debug.json \
  --log-level DEBUG \
  --log-file examples/v2/faiss-31k-strict-debug/profiler_v2_debug.log \
  --strict-gpu-monitor \
  2>&1 | tee examples/v2/faiss-31k-strict-debug/sla_log_lmstudio_v2_debug.txt
```

---

#### 3. Force NumPy batched fallback on 31k chunks  (V2)

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 20 \
  --warmup-runs 3 \
  --max-tokens 128 \
  --force-numpy \
  --numpy-batch-size 8192 \
  --json-out examples/v2/numpy-batched-31k/sla_report_numpy_batched.json \
  --log-level INFO \
  --log-file examples/v2/numpy-batched-31k/profiler_numpy_batched.log \
  2>&1 | tee examples/v2/numpy-batched-31k/sla_log_numpy_batched.txt
```

---

#### 4. Force NumPy batched fallback on 100k chunks  (V2)

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 100000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 20 \
  --warmup-runs 3 \
  --max-tokens 128 \
  --force-numpy \
  --numpy-batch-size 8192 \
  --json-out examples/v2/numpy-batched-100k/sla_report_numpy_batched_100k.json \
  --log-level INFO \
  --log-file examples/v2/numpy-batched-100k/profiler_numpy_batched_100k.log \
  2>&1 | tee examples/v2/numpy-batched-100k/sla_log_numpy_batched_100k.txt
```

---

### Ollama Usage  (V2)

The script supports both Ollama generate-style and chat-style APIs.

#### Ollama Chat  (V2)

Recommended for chat/instruct models:

```bash
python local_gpu_sla_profiler.py \
  --backend ollama-chat \
  --model qwen2.5:7b \
  --llm-url http://localhost:11434/api/chat \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out examples/v2/ollama-chat/sla_report_ollama_chat.json \
  --log-level INFO \
  --log-file examples/v2/ollama-chat/profiler_ollama_chat.log \
  2>&1 | tee examples/v2/ollama-chat/sla_log_ollama_chat.txt
```

#### Ollama Generate  (V2)

```bash
python local_gpu_sla_profiler.py \
  --backend ollama-generate \
  --model qwen2.5:7b \
  --llm-url http://localhost:11434/api/generate \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out examples/v2/ollama-generate/sla_report_ollama_generate.json \
  --log-level INFO \
  --log-file examples/v2/ollama-generate/profiler_ollama_generate.log \
  2>&1 | tee examples/v2/ollama-generate/sla_log_ollama_generate.txt
```

---

### CLI Options (V2)

| Option                     | Description                                                                     |
| -------------------------- | ------------------------------------------------------------------------------- |
| `--backend`                | Backend type: `lmstudio`, `openai-compatible`, `ollama-chat`, `ollama-generate` |
| `--model`                  | Local model name                                                                |
| `--llm-url`                | Local LLM API endpoint                                                          |
| `--prompt`                 | Test prompt                                                                     |
| `--max-tokens`             | Maximum output tokens                                                           |
| `--temperature`            | Sampling temperature                                                            |
| `--timeout-s`              | HTTP timeout                                                                    |
| `--chunks`                 | Number of synthetic vector chunks                                               |
| `--dim`                    | Vector dimension                                                                |
| `--top-k`                  | Number of retrieval candidates                                                  |
| `--query-runs`             | Number of measured retrieval runs                                               |
| `--warmup-runs`            | Number of warmup runs                                                           |
| `--numpy-batch-size`       | Batch size for NumPy fallback                                                   |
| `--force-numpy`            | Force NumPy fallback even when FAISS is available                               |
| `--device-index`           | CUDA/NVML GPU index                                                             |
| `--vram-sample-interval-s` | VRAM sampling interval                                                          |
| `--strict-gpu-monitor`     | Fail fast on NVML/CUDA monitoring errors                                        |
| `--retrieval-sla-ms`       | Retrieval latency SLA threshold                                                 |
| `--llm-sla-tps`            | LLM tokens/sec SLA threshold                                                    |
| `--vram-sla-gib`           | VRAM SLA threshold                                                              |
| `--json-out`               | JSON report output path                                                         |
| `--log-level`              | Logging level                                                                   |
| `--log-file`               | Structured log output path                                                      |

---

### Output Files (V2)

Each run can generate:

| File Type | Description                        |
| --------- | ---------------------------------- |
| `.txt`    | Human-readable terminal output     |
| `.json`   | Structured machine-readable report |
| `.log`    | Structured runtime/debug logs      |

Example output directory:

```text
examples/v2/faiss-31k/
├── profiler_v2.log
├── sla_log_lmstudio_v2.txt
└── sla_report_lmstudio_v2.json
```

---

### Why NVML and PyTorch Stats Are Both Shown

The profiler reports both:

* **NVML memory**
* **PyTorch CUDA memory**

NVML is used as the primary source for device-wide GPU memory usage. This is important when inference is performed by an external process such as LM Studio.

PyTorch CUDA memory can remain at `0.000 GiB` when the profiler process itself is not allocating CUDA tensors. This is expected when the LLM is served by LM Studio in a separate process.

---

### Why Batched NumPy Fallback Exists

FAISS is preferred when available.

However, the script includes a batched NumPy fallback for environments where FAISS is unavailable or intentionally disabled. The fallback processes vectors in batches and only keeps the global Top-K candidates, reducing temporary memory spikes compared to computing all scores in one full-dataset operation.

This was validated on:

```text
31,000 synthetic chunks
100,000 synthetic chunks
```

Both fallback scenarios passed the configured SLA thresholds.

---

### Result Documentation

Detailed results:

```text
docs/BENCHMARK_RESULTS.md
```

Versioned outputs:

```text
examples/v1/
examples/v2/
```

---

### Interpreting the Latest Results

The latest v2 results show:

* Retrieval latency is well below the 150 ms P95 threshold.
* LLM generation speed is consistently around 98-101 tokens/sec.
* Peak device-wide VRAM remains around 18.4 GiB.
* The batched NumPy fallback remains stable even at 100,000 synthetic chunks.
* All tested scenarios pass the SLA targets.

The 100k NumPy fallback test naturally shows higher latency variance than 31k, but it still remains comfortably below the retrieval SLA threshold.

---

## Historical Documentation (V1)

### Tested Hardware

The current benchmark was executed on:

| Component | Value |
|---|---|
| CPU | Intel Core i7-11700K |
| RAM | 64GB DDR4-3200 |
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 24GB |
| OS | Debian 12 |
| Python | 3.11.2 |
| LLM Runtime | LM Studio Local Server |
| LLM Model | qwen3-coder-30b-a3b-instruct |

---

### Benchmark Result Summary (v1)

Final benchmark command used `max_tokens=256`.

| Metric | Result | SLA Target | Verdict |
|---|---:|---:|---|
| Vector Search P95 Latency | 6.10 ms | <= 150 ms | PASS |
| Vector Search Mean Latency | 4.82 ms | — | — |
| Vector Search P50 Latency | 4.72 ms | — | — |
| LLM TTFT | 329 ms | — | — |
| LLM Total Time | 2481.76 ms | — | — |
| LLM Generation Time | 2102 ms | — | — |
| LLM Output Tokens | 185 | — | — |
| LLM Speed | 86.99 tokens/sec | >= 15 tokens/sec | PASS |
| Peak Device VRAM During LLM | 18.492 GiB | <= 24 GiB | PASS |
| Overall SLA | — | — | PASS |

---

### Project Structure (V1)

```text
local-gpu-sla-profiler/
├── local_gpu_sla_profiler.py
├── requirements.txt
└── examples/
    ├── gpu_after_profiler.csv
    ├── gpu_before_model_load.csv
    ├── sla_log_lmstudio_256.txt
    ├── sla_report_lmstudio_256.json
    └── SLA_EXECUTION_REPORT_256.md
```

#### File Descriptions (V1)

| File | Description |
|---|---|
| `local_gpu_sla_profiler.py` | Main benchmark script |
| `requirements.txt` | Python dependencies |
| `gpu_before_model_load.csv` | Shows before profiler GPU status |
| `gpu_after_profiler.csv` | Shows after profiler GPU status |
| `sla_log_lmstudio_256.txt` | Terminal output from the benchmark run |
| `sla_report_lmstudio_256.json` | Structured JSON benchmark report |
| `SLA_EXECUTION_REPORT_256.md` | Clean final execution report |

---

### How to Run (V1)

#### LM Studio Backend (V1)

```bash
python local_gpu_sla_profiler.py \
  --backend lmstudio \
  --model qwen3-coder-30b-a3b-instruct \
  --llm-url http://localhost:1234/api/v0/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out sla_report_lmstudio_256.json \
  2>&1 | tee sla_log_lmstudio_256.txt
```

#### Ollama Backend (V1)

```bash
python local_gpu_sla_profiler.py \
  --backend ollama \
  --model llama3.1:8b \
  --llm-url http://localhost:11434/api/generate \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out sla_report_ollama.json \
  2>&1 | tee sla_log_ollama.txt
```

#### OpenAI-Compatible Backend (V1)

Use this mode for llama.cpp server, vLLM, or other OpenAI-compatible local servers.

```bash
python local_gpu_sla_profiler.py \
  --backend openai-compatible \
  --model local-model \
  --llm-url http://localhost:8000/v1/chat/completions \
  --chunks 31000 \
  --dim 768 \
  --top-k 100 \
  --query-runs 50 \
  --warmup-runs 5 \
  --max-tokens 256 \
  --temperature 0.2 \
  --json-out sla_report_openai_compatible.json \
  2>&1 | tee sla_log_openai_compatible.txt
```

---

### Example SLA Report

Example output:

```text
========================================================================
LOCAL GPU SLA PROFILER REPORT
========================================================================

[Environment]
Platform: Linux-6.1.0-49-amd64-x86_64-with-glibc2.36
Python: 3.11.2
GPU (NVML): NVIDIA GeForce RTX 3090
GPU (Torch): NVIDIA GeForce RTX 3090
Total VRAM (NVML): 24.000 GiB
Total VRAM (Torch): 23.688 GiB
Torch CUDA Available: True
FAISS Available: True
NVML Available/OK: True / True

[VRAM Snapshots]
before_benchmark             | NVML used:   18.388 GiB | Torch allocated:    0.000 GiB | Torch reserved:    0.000 GiB
after_vector_search          | NVML used:   18.388 GiB | Torch allocated:    0.000 GiB | Torch reserved:    0.000 GiB
after_llm_inference          | NVML used:   18.490 GiB | Torch allocated:    0.000 GiB | Torch reserved:    0.000 GiB
Peak Device VRAM During LLM: 18.492 GiB

[Vector Search]
Backend: faiss.IndexFlatIP
Chunks: 31,000
Dimension: 768
Top-K: 100
Runs/Warmup: 50/5
Mean Latency: 4.82 ms
P50 Latency: 4.72 ms
P95 Latency: 6.10 ms

[LLM Inference]
Backend: lmstudio
Model: qwen3-coder-30b-a3b-instruct
TTFT: 329.00 ms
Total Time: 2481.76 ms
Output Tokens: 185
Tokens/sec: 86.99
Metric Source: lmstudio_native_stats

[SLA Verdict]
Retrieval P95 <= 150.0 ms: PASS
LLM Speed >= 15.0 tokens/sec: PASS
Peak VRAM <= 24.0 GiB: PASS
Overall: PASS
```

---

### Output Files (V1)

After running the profiler, two main output files are generated:

| File | Purpose |
|---|---|
| `sla_log_lmstudio_256.txt` | Human-readable terminal log |
| `sla_report_lmstudio_256.json` | Machine-readable structured report |

The JSON output can be used for:

- comparing multiple benchmark runs
- tracking optimization improvements
- building dashboards
- automated SLA checks

---

### CLI Arguments (V1)

| Argument | Default | Description |
|---|---:|---|
| `--backend` | `lmstudio` | LLM backend: `lmstudio`, `ollama`, or `openai-compatible` |
| `--model` | `qwen3-coder-30b-a3b-instruct` | Local model name |
| `--llm-url` | `http://localhost:1234/api/v0/chat/completions` | LLM API endpoint |
| `--chunks` | `31000` | Number of synthetic vector chunks |
| `--dim` | `768` | Vector dimension |
| `--top-k` | `100` | Number of retrieved candidates |
| `--query-runs` | `50` | Number of measured vector search runs |
| `--warmup-runs` | `5` | Warmup runs before measurement |
| `--max-tokens` | `128` | Maximum LLM output tokens |
| `--temperature` | `0.2` | LLM sampling temperature |
| `--json-out` | `sla_report.json` | JSON output path |
| `--retrieval-sla-ms` | `150.0` | Retrieval latency SLA threshold |
| `--llm-sla-tps` | `15.0` | LLM tokens/sec SLA threshold |
| `--vram-sla-gib` | `24.0` | VRAM usage SLA threshold |

---

## Metrics and Technical Notes

### How Metrics Are Calculated

#### Vector Search Latency

The script runs multiple vector search queries and reports:

- Mean latency
- P50 latency
- P95 latency
- Min/max latency
- Standard deviation

P95 is especially useful for SLA evaluation because it shows the latency under which 95% of requests completed.

---

#### TTFT

**TTFT** means **Time To First Token**.

It measures how long the system takes to start producing the first token after receiving the request.

Lower TTFT usually improves perceived responsiveness.

---

#### Tokens per Second

`tokens/sec` measures how many output tokens the model generates per second.

Higher values generally indicate faster generation speed.

---

#### Peak VRAM

Peak VRAM is measured through repeated NVML sampling while the LLM request is running.

This helps detect whether inference stays inside the GPU memory budget.

---

### Important Technical Notes

#### 1. Why NVML is used

`torch.cuda` only reports memory allocated by PyTorch tensors inside the current Python process.

If the model is served by LM Studio, Ollama, or llama.cpp in another process, PyTorch may show:

```text
Torch allocated: 0.000 GiB
```

This does not mean the model is not using GPU memory.

NVML is used to capture device-wide GPU memory usage.

---

#### 2. Why `before_benchmark` may already show high VRAM usage

If the model is already loaded in LM Studio before the profiler starts, `before_benchmark` includes the loaded model memory.

Example:

```text
before_benchmark | NVML used: 18.388 GiB
```

This means:

- The model was already loaded.
- The profiler measured memory before benchmark execution.
- It was not a pre-model-load baseline.

For a strict pre-load baseline, run:

```bash
nvidia-smi --query-gpu=timestamp,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv > gpu_before_model_load.csv
```

before loading the model in LM Studio.

---

#### 3. Synthetic vectors vs real embeddings

This benchmark uses synthetic vectors because the technical task targets retrieval latency over a fixed number of chunks.

Synthetic vectors are useful for measuring retrieval speed, but they do not measure semantic quality.

For production validation, benchmark with real project embeddings and real queries.

---

#### 4. This is a first-stage profiler

This project measures the core SLA metrics requested for the benchmark task.

It does not yet simulate full concurrent execution of:

- YOLO inference
- embedding generation
- vector retrieval
- reranking
- LLM generation

That can be added as a future extension.

---

## Final Notes

This profiler is intended for practical local AI performance auditing.

It is most useful when you need a quick, reproducible answer to:

```text
Can this local single-GPU system meet the required SLA for RAG retrieval, LLM generation, and GPU memory usage?
```

Current benchmark result on RTX 3090:

```text
Retrieval SLA: PASS
LLM Speed SLA: PASS
VRAM SLA: PASS
Overall: PASS
```

---
---