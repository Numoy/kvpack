"""Run the Kestrel benchmark on a cloud GPU with Modal: build a cartridge, then evaluate it.

    uv run --with modal modal run scripts/modal_benchmark.py --model Qwen/Qwen3-4B --tokens 512 --samples 2048

Writes the `kvpack eval` reports to runs/gpu-<model>-<tokens>/ so you can chart them:

    uv run python scripts/benchmark_chart.py runs/gpu-*/eval.json runs/gpu-*/eval-rag-*.json -o benchmark.svg
"""

import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "safetensors==0.8.0",
        "huggingface-hub==1.33.0",
        "fastapi==0.141.1",
        "httpx==0.28.1",
        "pydantic==2.13.5",
        "rich==15.0.0",
        "typer==0.27.2",
        "uvicorn==0.54.0",
    )
    .env({"HF_HOME": "/hf-cache"})
    .add_local_dir(ROOT / "examples", "/examples")
    .add_local_python_source("kvpack")
)
app = modal.App("kvpack-benchmark", image=image)
hf_cache = modal.Volume.from_name("kvpack-hf-cache", create_if_missing=True)


@app.function(gpu="H100", timeout=6 * 60 * 60, volumes={"/hf-cache": hf_cache})
def run(model: str, tokens: int, samples: int, epochs: int, synth_batch_size: int) -> dict:
    import time

    import kvpack
    from kvpack.benchmark import load_questions, run_benchmark

    corpus = kvpack.load_corpus(["/examples/kestrel-station/manual.md"])
    questions = load_questions("/examples/kestrel-station/questions.jsonl")
    lm, _, chat_format = kvpack.load_model(model)

    started = time.time()
    result = kvpack.build(
        lm,
        chat_format,
        corpus,
        name="kestrel",
        num_tokens=tokens,
        num_samples=samples,
        epochs=epochs,
        synth_batch_size=synth_batch_size,
        train_batch_size=8,
    )
    build_minutes = (time.time() - started) / 60
    cartridge = result.cartridge

    standard = run_benchmark(lm, chat_format, corpus, cartridge, questions)
    same_budget = run_benchmark(
        lm, chat_format, corpus, cartridge, questions, setups=("rag",), rag_chunks=1, rag_chunk_tokens=tokens
    )
    for report in (standard, same_budget):
        report["build"] = {"samples": samples, "epochs": epochs, "minutes": round(build_minutes, 1), "gpu": "H100"}
    return {"eval": standard, "eval_rag_same_budget": same_budget, "history": vars(result.history.eval_after)}


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-4B", tokens: int = 512, samples: int = 2048, epochs: int = 2, synth_batch_size: int = 64
):
    out = ROOT / "runs" / f"gpu-{model.split('/')[-1]}-{tokens}"
    out.mkdir(parents=True, exist_ok=True)
    reports = run.remote(model, tokens, samples, epochs, synth_batch_size)
    (out / "eval.json").write_text(json.dumps(reports["eval"], indent=2))
    (out / "eval-rag-same-budget.json").write_text(json.dumps(reports["eval_rag_same_budget"], indent=2))
    for r in reports["eval"]["results"] + reports["eval_rag_same_budget"]["results"]:
        print(f"{r['label']:<16} {r['correct']:>2}/{r['total']}  {r['context_tokens']:>6} tokens  {r['ttft_ms']} ms")
    print(f"Build: {reports['eval']['build']['minutes']} min on an H100. Reports in {out}")
