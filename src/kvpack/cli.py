"""The `kvpack` command-line tool."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

app = typer.Typer(
    help="Pack your documents into a trained KV cache (a 'cartridge') that any chat can load.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Shared options -------------------------------------------------------------------

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Hugging Face model id or local path.")]
DeviceOpt = Annotated[str, typer.Option(help="cuda, mps, cpu or auto.")]
DtypeOpt = Annotated[str, typer.Option(help="bfloat16, float16, float32 or auto.")]
GenUrlOpt = Annotated[
    str | None,
    typer.Option(help="OpenAI-compatible base URL (e.g. a vLLM server at http://localhost:8000/v1) to generate "
                 "self-study data much faster. It must serve the same model."),
]  # fmt: skip
GenModelOpt = Annotated[str | None, typer.Option(help="Model name to request from --generator-url.")]


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[info]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _log_line(message: str) -> None:
    """Progress bars don't render in log files, so print plain lines there instead."""
    if not console.is_terminal:
        console.print(f"[{time.strftime('%H:%M:%S')}] {message}", markup=False, highlight=False)


def _load(model: str, device: str, dtype: str):
    from .models import load_model

    with console.status(f"Loading [bold]{model}[/]..."):
        return load_model(model, device=device, dtype=dtype)


def _make_generator(model, chat_format, url: str | None, generator_model: str | None, model_name: str):
    from .synthesize import LocalGenerator, OpenAIGenerator

    if url:
        return OpenAIGenerator(chat_format, url, generator_model or model_name)
    return LocalGenerator(model, chat_format)


SourcesArg = Annotated[
    list[str],
    typer.Argument(
        help="Where the knowledge comes from: folders, files, Git repositories (https://github.com/org/repo, "
        "git+https://...) or web pages (https://docs.example.com/guide/ fetches the pages under it).",
        metavar="SOURCES...",
    ),
]


def _fetch(sources):
    """Fetch every source into one snapshot. `sources` are URIs (CLI input) or Source objects."""
    from .sources import SourceError, fetch_all, source_from_uri

    try:
        resolved = [source_from_uri(s) if isinstance(s, str) else s for s in sources]
        with console.status("Fetching " + ", ".join(r.describe() for r in resolved) + "..."):
            snapshot = fetch_all(resolved)
    except SourceError as e:
        raise typer.BadParameter(str(e)) from None
    n = len(snapshot.documents)
    console.print(f"Fetched [bold]{n:,}[/] document{'' if n == 1 else 's'}")
    return snapshot


def _default_name(source: str) -> str:
    from urllib.parse import urlsplit

    if Path(source).exists():
        path = Path(source).resolve()
        return path.stem if path.is_file() else path.name
    parts = urlsplit(source.removeprefix("git+"))
    segments = [s for s in parts.path.split("/") if s]
    return (segments[-1].removesuffix(".git").removesuffix(".html") if segments else parts.netloc) or "cartridge"


def _save_atomic(cartridge, out: Path) -> None:
    """Write next to the destination, then rename, so a watching server never reads a partial file."""
    partial = out.with_name(out.name + ".partial")
    cartridge.save(partial)
    partial.replace(out)


def _build_and_save(snapshot, out: Path, *, name: str, model: str, tokens: int, samples: int, epochs: int,
                    lr: float, batch_size: int, gradient_checkpointing: bool, synth_batch_size: int,
                    data: Path | None, save_data: Path | None, generator_url: str | None,
                    generator_model: str | None, seed: int, device: str, dtype: str) -> None:  # fmt: skip
    from .pipeline import build as build_cartridge
    from .synthesize import SelfStudyDataset

    corpus = snapshot.corpus()
    lm, tokenizer, chat_format = _load(model, device, dtype)
    corpus_tokens = len(tokenizer.encode(corpus, add_special_tokens=False))
    console.print(f"Corpus: [bold]{corpus_tokens:,}[/] tokens -> cartridge of [bold]{tokens:,}[/] tokens")
    if tokens >= corpus_tokens:
        console.print("[yellow]The cartridge is not smaller than the corpus. Consider fewer --tokens.[/]")

    dataset = SelfStudyDataset.load(data) if data else None
    with _progress() as progress:
        synth_task = progress.add_task("Self-study", total=samples, info="", visible=dataset is None)
        train_task = progress.add_task("Training", total=None, info="", visible=False)

        def on_synth(n: int):
            progress.advance(synth_task, n)
            _log_line(f"self-study {int(progress.tasks[synth_task].completed)}/{samples}")

        def on_step(step: int, total: int, loss: float):
            progress.update(train_task, total=total, completed=step, visible=True, info=f"loss {loss:.3f}")
            _log_line(f"training step {step}/{total} loss {loss:.3f}")

        result = build_cartridge(
            lm, chat_format, corpus,
            name=name, num_tokens=tokens, num_samples=samples, dataset=dataset,
            generator=_make_generator(lm, chat_format, generator_url, generator_model, model),
            synth_batch_size=synth_batch_size, train_batch_size=batch_size, lr=lr, epochs=epochs, seed=seed,
            gradient_checkpointing=gradient_checkpointing,
            sources=snapshot.sources, fingerprint=snapshot.fingerprint,
            on_synth_progress=on_synth, on_train_step=on_step,
        )  # fmt: skip

    if save_data and data is None:
        result.dataset.save(save_data)
        console.print(f"Saved self-study data to [bold]{save_data}[/]")
    _save_atomic(result.cartridge, out)
    _print_summary(result.cartridge, result.history)
    console.print(f"\n[green]Saved cartridge to [bold]{out}[/][/]  Try it:  [bold]kvpack chat {out}[/]")


# Commands -------------------------------------------------------------------------


@app.command()
def build(
    sources: SourcesArg,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output .safetensors file.")] = None,
    model: ModelOpt = "Qwen/Qwen3-4B",
    tokens: Annotated[int, typer.Option(help="Cartridge size in tokens. Smaller = less memory.")] = 2048,
    samples: Annotated[int, typer.Option(help="Self-study conversations to generate.")] = 1024,
    epochs: Annotated[int, typer.Option(help="Passes over the self-study data.")] = 1,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 2e-2,
    batch_size: Annotated[int, typer.Option(help="Training batch size (conversations per step).")] = 8,
    gradient_checkpointing: Annotated[
        bool, typer.Option(help="Recompute activations in the backward pass: much less memory, somewhat slower.")
    ] = True,
    synth_batch_size: Annotated[int, typer.Option(help="Conversations generated in parallel.")] = 16,
    data: Annotated[Path | None, typer.Option(help="Reuse a dataset from `kvpack synthesize` / --save-data.")] = None,
    save_data: Annotated[Path | None, typer.Option(help="Also save the generated self-study data here.")] = None,
    name: Annotated[str | None, typer.Option(help="Cartridge name (defaults to the source's name).")] = None,
    generator_url: GenUrlOpt = None,
    generator_model: GenModelOpt = None,
    seed: int = 0,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Build a cartridge from your sources: generate self-study data, then train.

    The cartridge remembers its sources, so `kvpack sync` can rebuild it when they change.
    """
    name = name or _default_name(sources[0])
    out = out or Path(f"{name}.safetensors")
    _build_and_save(
        _fetch(sources), out, name=name, model=model, tokens=tokens, samples=samples, epochs=epochs, lr=lr,
        batch_size=batch_size, gradient_checkpointing=gradient_checkpointing, synth_batch_size=synth_batch_size,
        data=data, save_data=save_data, generator_url=generator_url, generator_model=generator_model, seed=seed,
        device=device, dtype=dtype,
    )  # fmt: skip


@app.command()
def sync(
    cartridge: Annotated[Path, typer.Argument(help="Cartridge .safetensors file.")],
    force: Annotated[bool, typer.Option(help="Rebuild even if nothing changed.")] = False,
    batch_size: Annotated[int, typer.Option(help="Training batch size (conversations per step).")] = 8,
    synth_batch_size: Annotated[int, typer.Option(help="Conversations generated in parallel.")] = 16,
    generator_url: GenUrlOpt = None,
    generator_model: GenModelOpt = None,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Rebuild a cartridge if its sources changed since it was built.

    Uses the sources and settings recorded in the cartridge. Safe to run from cron: when
    nothing changed it exits after fetching, and a rebuild replaces the file in one step.
    """
    from .cartridge import peek
    from .sources import SourceError, source_from_config

    meta = peek(cartridge).metadata
    if not meta.get("sources"):
        raise typer.BadParameter(
            "This cartridge doesn't record its sources (it was built from Python or by an older kvpack). "
            "Rebuild it once with `kvpack build`."
        )
    try:
        snapshot = _fetch([source_from_config(c) for c in meta["sources"]])
    except SourceError as e:
        raise typer.BadParameter(str(e)) from None
    if not force and snapshot.fingerprint == meta.get("sources_fingerprint"):
        console.print(f"[green]{cartridge} is up to date.[/]")
        return
    console.print("Sources changed. Rebuilding with the original settings...")
    _build_and_save(
        snapshot, cartridge, name=meta.get("name") or cartridge.stem, model=meta["model"],
        tokens=meta.get("num_tokens") or peek(cartridge).num_tokens, samples=meta.get("num_samples", 1024),
        epochs=meta.get("epochs", 1), lr=meta.get("lr", 2e-2), batch_size=batch_size, gradient_checkpointing=True,
        synth_batch_size=synth_batch_size, data=None, save_data=None, generator_url=generator_url,
        generator_model=generator_model, seed=0, device=device, dtype=dtype,
    )  # fmt: skip


@app.command()
def synthesize(
    sources: SourcesArg,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")],
    model: ModelOpt = "Qwen/Qwen3-4B",
    samples: Annotated[int, typer.Option(help="Self-study conversations to generate.")] = 1024,
    synth_batch_size: Annotated[int, typer.Option(help="Conversations generated in parallel.")] = 16,
    generator_url: GenUrlOpt = None,
    generator_model: GenModelOpt = None,
    seed: int = 0,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Only generate self-study data (train later with `kvpack build --data`)."""
    from .synthesize import synthesize as run_synthesis

    corpus = _fetch(sources).corpus()
    lm, _, chat_format = _load(model, device, dtype)
    with _progress() as progress:
        task = progress.add_task("Self-study", total=samples, info="")

        def on_synth(n: int):
            progress.advance(task, n)
            _log_line(f"self-study {int(progress.tasks[task].completed)}/{samples}")

        dataset = run_synthesis(
            lm, chat_format, corpus, samples,
            generator=_make_generator(lm, chat_format, generator_url, generator_model, model),
            batch_size=synth_batch_size, seed=seed, on_progress=on_synth,
        )  # fmt: skip
    dataset.save(out)
    console.print(f"[green]Saved {len(dataset)} conversations to [bold]{out}[/][/]")


@app.command()
def chat(
    cartridge: Annotated[Path, typer.Argument(help="Cartridge .safetensors file.")],
    message: Annotated[str | None, typer.Option("--message", "-q", help="Ask one question and exit.")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override the cartridge's model.")] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Chat with a cartridge in the terminal."""
    from .cartridge import Cartridge
    from .generate import generate

    cart = Cartridge.load(cartridge)
    lm, _, chat_format = _load(model or cart.metadata["model"], device, dtype)
    cart.to(lm.device)
    console.print(f"[dim]{cart}[/]")
    history: list[dict[str, str]] = []

    while True:
        text = message if message is not None else console.input("\n[bold cyan]you>[/] ")
        if text.strip().lower() in {"exit", "quit", ":q"}:
            break
        history.append({"role": "user", "content": text})
        console.print("[bold magenta]kvpack>[/] ", end="")
        reply = _stream_reply(generate, lm, chat_format, history, cart, max_tokens, temperature)
        history.append({"role": "assistant", "content": reply})
        if message is not None:
            break


def _stream_reply(generate, lm, chat_format, history, cart, max_tokens: int, temperature: float) -> str:
    """Generate in a background thread and print tokens as they arrive."""
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(chat_format.tokenizer, skip_prompt=True, skip_special_tokens=True)
    result: list[str] = []
    kwargs = dict(cartridge=cart, max_new_tokens=max_tokens, temperature=temperature, streamer=streamer)
    worker = threading.Thread(target=lambda: result.append(generate(lm, chat_format, history, **kwargs)))
    worker.start()
    for piece in streamer:
        console.print(piece, end="", markup=False, highlight=False)
    worker.join()
    console.print()
    return result[0]


@app.command()
def compare(
    cartridge: Annotated[Path, typer.Argument(help="Cartridge .safetensors file.")],
    docs: Annotated[
        list[Path], typer.Option("--docs", "-d", help="The original documents (for the full-context answer).")
    ],
    question: Annotated[list[str], typer.Option("--question", "-q", help="A question to ask. Repeatable.")],
    max_tokens: int = 256,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Answer questions three ways: with the cartridge, with the full documents, and with nothing."""
    from .cartridge import Cartridge
    from .generate import generate

    cart = Cartridge.load(cartridge)
    corpus = _fetch(docs).corpus()
    lm, tokenizer, chat_format = _load(cart.metadata["model"], device, dtype)
    cart.to(lm.device)
    corpus_tokens = len(tokenizer.encode(corpus, add_special_tokens=False))

    for q in question:
        messages = [{"role": "user", "content": q}]
        table = Table(title=q, show_lines=True, expand=True, title_justify="left")
        table.add_column("Setup", style="bold", no_wrap=True)
        table.add_column("Answer")
        for label, kwargs in [
            (f"Cartridge\n{cart.num_tokens:,} tokens", {"cartridge": cart}),
            (f"Full documents\n{corpus_tokens:,} tokens", {"context": corpus}),
            ("No context", {}),
        ]:
            answer = generate(lm, chat_format, messages, max_new_tokens=max_tokens, temperature=0, **kwargs)
            table.add_row(label, answer)
        console.print(table)


@app.command("eval")
def eval_(
    cartridge: Annotated[Path, typer.Argument(help="Cartridge .safetensors file.")],
    docs: Annotated[list[str], typer.Option("--docs", "-d", help="The cartridge's sources (folders, repos, URLs).")],
    questions: Annotated[
        Path, typer.Option("--questions", "-q", help="JSONL: {question, must_include: [facts]} per line.")
    ],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Save the full report as JSON.")] = None,
    rag_chunks: Annotated[int, typer.Option(help="Chunks the RAG baseline retrieves.")] = 4,
    rag_chunk_tokens: Annotated[int, typer.Option(help="Tokens per RAG chunk.")] = 256,
    setups: Annotated[
        str, typer.Option(help="Comma-separated: cartridge, rag, full, none.")
    ] = "cartridge,rag,full,none",
    max_tokens: int = 200,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Score the cartridge against RAG, the full documents and no context on your questions."""
    import json

    from .benchmark import ALL_SETUPS, LABELS, load_questions, run_benchmark
    from .cartridge import Cartridge

    cart = Cartridge.load(cartridge)
    corpus = _fetch(docs).corpus()
    qs = load_questions(questions)
    chosen = tuple(x.strip() for x in setups.split(",") if x.strip())
    if unknown := set(chosen) - set(ALL_SETUPS):
        raise typer.BadParameter(f"Unknown setups {sorted(unknown)}. Choose from {', '.join(ALL_SETUPS)}.")
    lm, _, chat_format = _load(cart.metadata["model"], device, dtype)
    cart.to(lm.device)

    with _progress() as progress:
        task = progress.add_task("Answering", total=len(qs) * len(chosen), info="")

        def on_answer(setup, q, answer, ok):
            progress.advance(task)
            progress.update(task, info=f"{LABELS[setup]}: {'correct' if ok else 'wrong'}")
            _log_line(f"{LABELS[setup]:<15} {'✓' if ok else '✗'} {q.question[:70]}")

        report = run_benchmark(
            lm, chat_format, corpus, cart, qs, setups=chosen,
            rag_chunks=rag_chunks, rag_chunk_tokens=rag_chunk_tokens, max_new_tokens=max_tokens, on_answer=on_answer,
        )  # fmt: skip

    table = Table(title=f"{len(qs)} questions · {report['model']}", title_justify="left")
    for column, justify in [("Setup", "left"), ("Correct", "right"), ("Context", "right"),
                            ("KV cache", "right"), ("Time to first token", "right")]:  # fmt: skip
        table.add_column(column, justify=justify)
    for r in report["results"]:
        if r["skipped"]:
            table.add_row(r["label"], "skipped", "", "", r["skipped"])
            continue
        table.add_row(
            r["label"],
            f"{r['correct']}/{r['total']} ({r['accuracy']:.0%})",
            f"{r['context_tokens']:,} tokens",
            f"{r['kv_cache_mib']:,.1f} MiB",
            f"{r['ttft_ms']:,} ms" if r["ttft_ms"] is not None else "",
        )
    console.print(table)
    if out:
        out.write_text(json.dumps(report, indent=2))
        console.print(f"Saved the report to [bold]{out}[/]")


@app.command()
def serve(
    sources: Annotated[
        list[Path], typer.Argument(help="Cartridge files and/or folders of cartridges (all for the same model).")
    ],
    host: str = "127.0.0.1",
    port: int = 8000,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Base model. Defaults to the cartridges' model.")
    ] = None,
    api_key: Annotated[
        list[str] | None,
        typer.Option(help="Require this API key (repeatable). Also read from KVPACK_API_KEYS, comma-separated."),
    ] = None,
    max_queue: Annotated[int, typer.Option(help="Requests that may wait before the server answers 429.")] = 32,
    max_loaded: Annotated[int, typer.Option(help="Cartridges kept in GPU memory at once.")] = 8,
    max_output_tokens: Annotated[int, typer.Option(help="Largest max_tokens a request may ask for.")] = 4096,
    device: DeviceOpt = "auto",
    dtype: DtypeOpt = "auto",
):
    """Serve cartridges behind an OpenAI-compatible API.

    Folders are watched: cartridges copied into them later become available without a restart.
    """
    import logging
    import os

    import uvicorn

    from .cartridge import peek
    from .server import create_app
    from .store import CartridgeStore

    files = [p for p in sources if p.is_file()]
    directories = [p for p in sources if p.is_dir()]
    missing = [p for p in sources if not p.exists()]
    if missing:
        raise typer.BadParameter(f"Not found: {', '.join(map(str, missing))}")
    if len(directories) > 1:
        raise typer.BadParameter("Pass at most one folder.")

    found = [peek(f) for f in files] + [peek(f) for d in directories for f in sorted(d.glob("*.safetensors"))]
    models = {model} if model else {info.model for info in found if info.model}
    if len(models) != 1:
        hint = "Pass --model." if not models else f"They were built for different models: {sorted(models)}."
        raise typer.BadParameter(f"Can't tell which base model to serve. {hint}")

    keys = list(api_key or []) + [k.strip() for k in os.environ.get("KVPACK_API_KEYS", "").split(",") if k.strip()]
    lm, _, chat_format = _load(models.pop(), device, dtype)
    store = CartridgeStore(
        lm.config.name_or_path,
        device=lm.device,
        files=files,
        directory=directories[0] if directories else None,
        max_loaded=max_loaded,
    )
    app_ = create_app(lm, chat_format, store, api_keys=keys, max_queue=max_queue, max_output_tokens=max_output_tokens)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    names = [info.name for info in store.list()]
    console.print(f"Serving {', '.join(f'[bold]{n}[/]' for n in names) or 'no cartridges yet'} "
                  f"on [bold]{lm.config.name_or_path}[/] at [bold]http://{host}:{port}/v1[/]")  # fmt: skip
    if not keys and host not in ("127.0.0.1", "localhost"):
        console.print("[yellow]Listening on a public interface without --api-key. Anyone can use it.[/]")
    uvicorn.run(app_, host=host, port=port, log_level="warning")


@app.command()
def push(
    cartridge: Annotated[Path, typer.Argument(help="Cartridge .safetensors file.")],
    repo_id: Annotated[str, typer.Argument(help="Hugging Face repo, e.g. your-name/handbook-cartridge.")],
    private: Annotated[bool, typer.Option(help="Create the repo as private.")] = False,
):
    """Share a cartridge on the Hugging Face Hub (uses your `hf auth login` token)."""
    from .hub import push as push_to_hub

    with console.status(f"Uploading to [bold]{repo_id}[/]..."):
        url = push_to_hub(cartridge, repo_id, private=private)
    console.print(f"[green]Pushed to {url}[/]")


@app.command()
def pull(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face repo, e.g. someone/handbook-cartridge.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Where to save it.")] = None,
    revision: Annotated[str | None, typer.Option(help="Branch, tag or commit.")] = None,
):
    """Download a cartridge from the Hugging Face Hub."""
    import shutil

    from .hub import pull as pull_from_hub

    with console.status(f"Downloading [bold]{repo_id}[/]..."):
        cached = pull_from_hub(repo_id, revision=revision)
    out = out or Path(f"{repo_id.split('/')[-1]}.safetensors")
    shutil.copyfile(cached, out)
    console.print(f"[green]Saved to {out}[/]  Try it:  [bold]kvpack chat {out}[/]")


@app.command()
def info(cartridge: Annotated[Path, typer.Argument(help="Cartridge .safetensors file.")]):
    """Show what's inside a cartridge."""
    from .cartridge import Cartridge

    cart = Cartridge.load(cartridge)
    meta = cart.metadata
    table = Table(show_header=False, box=None)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Name", str(meta.get("name", cartridge.stem)))
    table.add_row("Model", str(meta.get("model")))
    table.add_row("Cartridge tokens", f"{cart.num_tokens:,}")
    if meta.get("corpus_tokens"):
        table.add_row(
            "Corpus tokens", f"{meta['corpus_tokens']:,} ({meta['corpus_tokens'] / cart.num_tokens:.1f}x compression)"
        )
    table.add_row("Size (bf16)", f"{cart.size_bytes() / 2**20:.1f} MiB")
    table.add_row("Layers", str(cart.num_layers))
    if meta.get("num_samples"):
        table.add_row("Trained on", f"{meta['num_samples']:,} self-study conversations, {meta.get('epochs')} epoch(s)")
    if meta.get("saved_at"):
        table.add_row("Saved", time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["saved_at"])))
    console.print(table)
    if meta.get("eval_after"):
        _print_eval(meta.get("eval_no_context"), meta.get("eval_before"), meta["eval_after"])


# Output helpers -------------------------------------------------------------------


def _print_summary(cart, history) -> None:
    if history.eval_after is None:
        return
    _print_eval(vars(history.eval_no_context), vars(history.eval_before), vars(history.eval_after))


def _print_eval(no_context: dict | None, before: dict | None, after: dict) -> None:
    table = Table(title="Held-out self-study questions", title_justify="left")
    table.add_column("Setup")
    table.add_column("Token agreement with in-context answers", justify="right")
    table.add_column("Distillation loss", justify="right")
    for label, m in [("No context", no_context), ("Cartridge, untrained", before), ("Cartridge, trained", after)]:
        if m:
            table.add_row(label, f"{m['agreement']:.1%}", f"{m['loss']:.3f}")
    console.print(table)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
