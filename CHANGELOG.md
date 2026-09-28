# Changelog

## 0.2.0

- **Sources:** build cartridges from folders, Git repositories and websites
  (`kvpack build https://github.com/org/repo`). Cartridges remember their sources, and
  `kvpack sync` rebuilds only when they changed.
- **kvpack Studio** (`studio/`, formerly "kvpack Cloud"): a self-hosted web app. Connect
  sources, keep cartridges in sync on a schedule with zero-downtime updates, and use
  them through an OpenAI-compatible API with API keys. It runs on one GPU machine or on
  your own Modal account.
- **`kvpack eval`:** score a cartridge against RAG (BM25), the full documents, no
  context, and cartridge + RAG on your own questions, with memory and time-to-first-token
  numbers. Includes a 28-question benchmark for the Kestrel Station example.
- `kvpack serve`: `stream_options.include_usage`, content given as a list of parts.
- Self-study datasets always hold out at least one conversation for quality metrics.

## 0.1.0

First release.

- `kvpack build`: self-study data generation and cartridge training, locally or with an
  OpenAI-compatible server (vLLM, SGLang) for generation.
- `kvpack serve`: OpenAI-compatible server with streaming, API keys, a bounded request
  queue, cancellation on client disconnect, and a watched folder of cartridges loaded on
  demand.
- `kvpack chat`, `compare`, `info`, `push` and `pull` (Hugging Face Hub).
- Memory-efficient training: per-layer gradient checkpointing and a chunked loss.
- Supports Llama- and Qwen3-style models on CUDA, Apple Silicon and CPU.
