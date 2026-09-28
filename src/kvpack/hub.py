"""Sharing cartridges on the Hugging Face Hub.

A cartridge repo holds `cartridge.safetensors` plus a generated model card that
links the base model, so the Hub shows which model it works with.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .cartridge import Cartridge, peek

FILENAME = "cartridge.safetensors"


def push(path: str | Path, repo_id: str, *, private: bool = False, token: str | None = None) -> str:
    """Upload a cartridge and its model card. Returns the repo URL."""
    path = Path(path)
    api = HfApi(token=token)
    url = api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path), path_in_repo=FILENAME, repo_id=repo_id, commit_message="Upload cartridge"
    )
    api.upload_file(
        path_or_fileobj=model_card(path, repo_id).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Update model card",
    )
    return str(url)


def pull(repo_id: str, *, revision: str | None = None, token: str | None = None) -> Path:
    """Download a cartridge (cached locally) and return its path."""
    return Path(hf_hub_download(repo_id, FILENAME, revision=revision, token=token))


def load_from_hub(repo_id: str, *, device: str = "cpu", revision: str | None = None) -> Cartridge:
    return Cartridge.load(pull(repo_id, revision=revision), device=device)


def model_card(path: str | Path, repo_id: str) -> str:
    info = peek(path)
    meta = info.metadata
    name = repo_id.split("/")[-1]
    lines = [
        "---",
        "library_name: kvpack",
        *([f"base_model: {info.model}"] if info.model else []),
        "tags:",
        "- kvpack",
        "- cartridge",
        "- kv-cache",
        "---",
        "",
        f"# {name}",
        "",
        f"A [kvpack](https://github.com/Numoy/kvpack) cartridge: a trained KV cache that gives "
        f"`{info.model}` the knowledge of a document collection without putting it in the prompt.",
        "",
        "| | |",
        "| --- | --- |",
        f"| Base model | `{info.model}` |",
        f"| Cartridge size | {info.num_tokens:,} tokens |",
    ]
    if corpus_tokens := meta.get("corpus_tokens"):
        ratio = corpus_tokens / info.num_tokens
        lines.append(f"| Corpus size | {corpus_tokens:,} tokens ({ratio:.1f}x compression) |")
    if meta.get("num_samples"):
        lines.append(
            f"| Trained on | {meta['num_samples']:,} self-study conversations, {meta.get('epochs', 1)} epoch(s) |"
        )
    after, none = meta.get("eval_after"), meta.get("eval_no_context")
    if after and none:
        lines.append(
            f"| Held-out token agreement | {after['agreement']:.1%} (vs {none['agreement']:.1%} with no context) |"
        )
    lines += [
        "",
        "## Use it",
        "",
        "```bash",
        "pip install git+https://github.com/Numoy/kvpack",
        f"kvpack pull {repo_id} -o {name}.safetensors",
        f"kvpack chat {name}.safetensors",
        "```",
        "",
        "```python",
        "from kvpack import generate, load_model",
        "from kvpack.hub import load_from_hub",
        "",
        f'model, tokenizer, chat_format = load_model("{info.model}")',
        f'cartridge = load_from_hub("{repo_id}", device=model.device)',
        'print(generate(model, chat_format, [{"role": "user", "content": "..."}], cartridge=cartridge))',
        "```",
        "",
    ]
    return "\n".join(lines)
