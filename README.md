# Local GPU SLA Profiler

A standalone Python benchmark utility designed by JAPCO AI Department for profiling local GPU VRAM usage, vector search latency, and local LLM inference speed on single-GPU environments.

## What This Profiler Measures
- Background YOLO-like FP16 GPU workload at ~1 FPS.
- Local LLM inference (TTFT, tokens/sec, total latency) using strict JSON schemas with thinking-capable models.
- Local vector retrieval latency over 31,000 synthetic chunks (using FAISS IndexFlatIP).
- Dynamic device-wide GPU memory (VRAM) and thermal snapshots.

## Project Structure
- japco_rag_benchmark.py: Main benchmark script.
- requirements.txt: Package dependencies.
- reports/: Benchmark execution and model audit reports.
- reports/rtx_a6000_qwen3.5_35b/ollama_model_audit.json: Comprehensive audit log of 60 local Ollama models and their API capabilities.

## Official SLA Benchmarks (RTX A6000 / Qwen 3.5)
The official SLA audit was executed on an NVIDIA RTX A6000 (48GB VRAM) with the latest Qwen 3.5 (35B-Instruct) GGUF Q4_K_M model.

Hardware Environment:
- GPU: NVIDIA RTX A6000 (48GB VRAM)
- OS: Windows 10 (10.0.19045-SP0)
- Python: 3.8.20
- Torch: 1.12.1

Benchmark SLA Results:
- LLM Output: 10/10 Successful runs, 10/10 JSON Valid (0.00% Failure Rate)
- LLM TTFT (Time to First Token): 2854.74 ± 27.20 ms (min=2806.84, max=2891.48 ms)
- LLM Inference Speed: 90.95 ± 1.21 tokens/sec
- Total Wall Time: 4042.02 ± 42.57 ms
- Vector Search Latency (31k chunks): 5.90 ± 0.62 ms (min=5.08, max=7.21 ms)
- Peak GPU VRAM: 34.00 GB
- GPU Temperature: 37.30 ± 2.79 C
- YOLO background worker: 40 iterations / 0 errors

## How to Run the Benchmark
Ensure Ollama is running and your models are pulled, then run:

python japco_rag_benchmark.py --models qwen2.5:7b gemma2:9b --runs 10 --yolo-model yolov8n.pt --output reports/sla_report_official.jsonl --summary-output reports/sla_summary_official.json
