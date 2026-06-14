# Changelog

All notable changes to this project will be documented in this file.

## v2.0.0 - Company Feedback Update

### Added

- Added thread-safe `VramSampler` using `threading.Lock`.
- Added batched NumPy fallback for large vector search workloads.
- Added structured logging with `--log-level` and `--log-file`.
- Added `--strict-gpu-monitor` for fail-fast NVML/CUDA monitoring.
- Added Ollama `/api/chat` support via `--backend ollama-chat`.
- Added explicit support for Ollama `/api/generate` via `--backend ollama-generate`.
- Added `--force-numpy` to test the NumPy fallback path even when FAISS is available.
- Added `--numpy-batch-size` to control memory usage in NumPy fallback mode.

### Changed

- Improved `VramSampler` to protect shared `samples` list during append/read operations.
- Replaced direct NumPy fallback search over the full vector matrix with batch-based search.
- Improved exception handling by logging detailed tracebacks instead of silently ignoring errors.
- Improved backend naming to distinguish `ollama-chat` and `ollama-generate`.
- Expanded SLA report notes to document thread safety, batching, logging, and backend behavior.

### Kept

- LM Studio backend support.
- OpenAI-compatible streaming backend support.
- FAISS `IndexFlatIP` vector search benchmark.
- NVML-based device-wide VRAM monitoring.
- PyTorch CUDA memory stats when available.
- Terminal SLA report output.
- JSON report export.

## v1.0.0 - Initial SLA Profiler

### Added

- Initial Local GPU SLA Profiler implementation.
- Device-wide VRAM monitoring with NVML.
- Optional PyTorch CUDA memory reporting.
- FAISS vector search benchmark over 31,000 synthetic chunks.
- LM Studio local inference benchmark.
- Ollama `/api/generate` benchmark support.
- OpenAI-compatible streaming benchmark support.
- TTFT, total response time, tokens/sec, and output token metrics.
- SLA PASS/FAIL report.
- JSON report export.