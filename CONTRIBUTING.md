# Contributing to kvpack

Thanks for helping. Bug reports, benchmark results on your own documents and pull requests
are all welcome.

## Set up

```bash
git clone https://github.com/Numoy/kvpack && cd kvpack
uv sync --group dev
uv run pytest            # the core library, runs on CPU in seconds
cd studio && uv sync --group dev && uv run pytest   # kvpack Studio
```

The tests use a tiny randomly initialized model, so they need no GPU. They do download the
Qwen3 tokenizer once.

## Before opening a pull request

- `uv run ruff check . && uv run ruff format .`
- Add or update tests. If you touch how cartridges are built or applied, make sure
  `test_untrained_cartridge_equals_document_in_context` still passes: it proves the cache,
  positions and chat templates line up.
- Keep the code readable. kvpack is meant to be a place where people can learn how
  cartridges work, so clarity beats cleverness.

## Sharing results

The most useful contribution right now is evidence. If you build a cartridge for your own
documents, run `kvpack eval` and open an issue with the table (model, corpus size,
cartridge size, number of self-study conversations). Good and bad results both help.
