# Local GPU SLA Profiler — Execution Report

## Test Machine

| Item | Value |
|---|---|
| OS | Debian 12 / Linux 6.1.0-49-amd64 |
| Python | 3.11.2 |
| GPU | NVIDIA GeForce RTX 3090 |
| GPU VRAM | 24.000 GiB by NVML / 23.688 GiB by PyTorch |
| CUDA visible to PyTorch | True |
| FAISS | Available |
| NVML | Available / OK |
| LLM Runtime | LM Studio Local Server |
| LLM Model | qwen3-coder-30b-a3b-instruct |

## Benchmark Command

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

## Results Summary

| Metric | Result | SLA Target | Verdict |
|---|---:|---:|---|
| Vector Search P95 Latency | 6.10 ms | <= 150 ms | PASS |
| Vector Search Mean Latency | 4.82 ms | — | — |
| Vector Search P50 Latency | 4.72 ms | — | — |
| Vector Search Min / Max Latency | 3.95 / 6.68 ms | — | — |
| LLM TTFT | 329.00 ms | — | — |
| LLM Total Time | 2481.76 ms | — | — |
| LLM Generation Time | 2102.00 ms | — | — |
| LLM Output Tokens | 185 | — | — |
| LLM Speed | 86.99 tokens/sec | >= 15 tokens/sec | PASS |
| Peak Device VRAM During LLM | 18.492 GiB | <= 24 GiB | PASS |
| Overall SLA | — | — | PASS |

## Technical Interpretation

The benchmark passed all requested SLA checks on a local RTX 3090 system. The vector retrieval layer is significantly below the 150 ms SLA target, with FAISS P95 latency at 6.10 ms over 31,000 synthetic chunks and Top-K=100. The LLM inference layer also passed the speed target with 86.99 tokens/sec and a 329 ms TTFT. Peak device-wide VRAM during the benchmark was 18.492 GiB, leaving approximately 5.5 GiB of headroom on a 24 GiB RTX 3090.

## VRAM Notes

The reported `before_benchmark` VRAM value already includes the model loaded by LM Studio:

```text
before_benchmark             | NVML used:   18.388 GiB
after_vector_search          | NVML used:   18.388 GiB
after_llm_inference          | NVML used:   18.490 GiB
Peak Device VRAM During LLM: 18.492 GiB
```

This is expected because the LLM is served by LM Studio as a separate process before the Python profiler starts. Therefore:

- NVML is the correct source for device-wide GPU memory usage.
- PyTorch CUDA memory remains 0 GiB because the Python script itself does not allocate model tensors.
- For a strict “before model load” baseline, capture a separate `nvidia-smi` snapshot before loading the model in LM Studio.

## Raw Output Excerpt

```text
[Vector Search]
Backend: faiss.IndexFlatIP
Chunks: 31,000
Dimension: 768
Top-K: 100
Runs/Warmup: 50/5
Index Build Time: 18.34 ms
Mean Latency: 4.82 ms
P50 Latency: 4.72 ms
P95 Latency: 6.10 ms
Min/Max Latency: 3.95 / 6.68 ms
Std Latency: 0.71 ms

[LLM Inference]
Backend: lmstudio
Model: qwen3-coder-30b-a3b-instruct
TTFT: 329.00 ms
Total Time: 2481.76 ms
Generation Time: 2102.00 ms
Output Tokens: 185
Tokens/sec: 86.99
Metric Source: lmstudio_native_stats

[SLA Verdict]
Retrieval P95 <= 150.0 ms: PASS
LLM Speed >= 15.0 tokens/sec: PASS
Peak VRAM <= 24.0 GiB: PASS
Overall: PASS
```

