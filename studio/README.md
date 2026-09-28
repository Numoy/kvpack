# kvpack Studio

A self-hosted web app for [kvpack](../README.md). Connect a Git repository, a docs site, a
server folder or uploaded files. Studio packs them into a cartridge, keeps it in sync,
and serves it to any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://your-studio:8080/v1", api_key="kvp_...")
reply = client.chat.completions.create(
    model="handbook",  # a cartridge name or id
    messages=[{"role": "user", "content": "What's the escalation path for a Level Red alert?"}],
)
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/assets/studio-connect-dark.png">
  <img src="../docs/assets/studio-connect-light.png" alt="kvpack Studio: connect a source and keep it in sync" width="900">
</picture>

- **Connections, not uploads:** a cartridge is defined by its sources. Studio fetches
  them, fingerprints the content, and on each sync rebuilds only if something changed.
- **No downtime:** the current version keeps answering while a new one builds, and stays
  in place if an update fails.
- **Automatic sync:** hourly, every 6 hours, daily or weekly, or on demand with
  "Sync now" (`POST /v1/cartridges/{id}/sync`).
- **A playground and code snippets** for every cartridge, plus API keys for your team
  and per-account usage.
- **Download any cartridge** to serve it yourself with `kvpack serve`, or import one you
  built with the CLI.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/assets/studio-playground-dark.png">
  <img src="../docs/assets/studio-playground-light.png" alt="A cartridge's quality check, playground and code snippet" width="900">
</picture>

## Run it

On a machine with a GPU (or a Mac with Apple Silicon for small models):

```bash
cd studio
uv sync
uv run kvpack-studio serve --host 0.0.0.0 --port 8080
```

Open the page and create your account. The first account needs no invite. After that,
sign-up needs an invite code (`KVPACK_STUDIO_INVITE_CODES`), or an admin adds people
with `uv run kvpack-studio create-account someone@example.com`.

Builds run one at a time in a separate process, so a build never blocks or crashes the
API, and a scheduler starts automatic syncs as they fall due. Interrupted builds restart
when Studio starts.

## Sources and security

Studio is multi-user, so every source is treated as untrusted input:

| Source | Available | Safety rule |
| --- | --- | --- |
| Git repository | always | `https://` only. Private repos use a token from an environment variable the admin allowlists (`KVPACK_STUDIO_GIT_TOKEN_ENVS`). Users pick the variable's name and never see the token. |
| Website | always | Only public addresses. Private, loopback and link-local addresses (like cloud metadata endpoints) are refused, including on every page and redirect during a crawl. Set `KVPACK_STUDIO_ALLOW_PRIVATE_URLS=1` to crawl an intranet on a trusted setup. |
| Folder on the server | off by default | Only inside the folders listed in `KVPACK_STUDIO_FOLDER_ROOTS`. |
| Upload | always | File types, sizes and counts are limited, and file names are reduced to a safe base name. |

API keys are 256-bit random tokens, stored only as SHA-256 hashes and shown once. Every
cartridge route checks ownership and answers 404 for other accounts' cartridges, and
errors from the model server are rewritten so they never reveal other cartridges.

## Deploy on Modal

For GPUs that scale to zero, deploy Studio to your own [Modal](https://modal.com) account:

```mermaid
flowchart LR
    U[Browser / OpenAI client] -->|HTTPS + API key| API
    subgraph Modal
        API[api<br/>CPU web function] -->|spawn| B[build_cartridge<br/>GPU job]
        API -->|private RPC| I[Inference<br/>GPU, kvpack serve]
        C[sync_due_cartridges<br/>every 15 min] -->|spawn| B
        B -->|writes cartridge| V[(Volume)]
        I -->|loads cartridges| V
    end
    API --> DB[(Postgres)]
    B -->|progress| DB
```

You need a Modal account (`uv tool install modal`, then `modal token new`) and a Postgres
database, for example from [Neon](https://neon.tech) or [Supabase](https://supabase.com).
SQLite won't work there, because several containers write at once.

```bash
modal secret create kvpack-studio \
  KVPACK_STUDIO_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST/DBNAME' \
  KVPACK_STUDIO_INVITE_CODES='choose-an-invite-code' \
  HF_TOKEN='hf_...'   # optional, for gated models

cd studio
uv sync --extra modal --extra postgres
export KVPACK_STUDIO_BASE_MODEL=Qwen/Qwen3-4B      # optional; also KVPACK_STUDIO_BUILD_GPU, _INFERENCE_GPU
uv run modal deploy -m kvpack_studio.modal_app
```

Modal prints the URL of the `kvpack-studio` endpoint. Builds run on
`KVPACK_STUDIO_BUILD_GPU` (default H100). Inference containers (default L40S) have no
public URL, bill while warm, and shut down after 10 idle minutes.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `KVPACK_STUDIO_DATABASE_URL` | `sqlite:///kvpack-studio.db` | SQLAlchemy URL. Use Postgres on Modal. |
| `KVPACK_STUDIO_STORAGE_DIR` | `kvpack-studio-data` | Where uploads and cartridges live. |
| `KVPACK_STUDIO_BASE_MODELS` | `Qwen/Qwen3-4B` | Comma-separated models users can build for. |
| `KVPACK_STUDIO_INVITE_CODES` | *(empty)* | Codes that let more people sign up after the first account. |
| `KVPACK_STUDIO_FOLDER_ROOTS` | *(empty: off)* | Server folders users may connect. |
| `KVPACK_STUDIO_ALLOW_PRIVATE_URLS` | off | Allow websites and Git remotes on private networks. |
| `KVPACK_STUDIO_GIT_TOKEN_ENVS` | *(empty)* | Environment variables holding Git tokens that sources may use. |
| `KVPACK_STUDIO_MAX_UPLOAD_MB` | `50` | Per upload. |
| `KVPACK_STUDIO_MAX_CORPUS_TOKENS` | `250000` | Per build. |
| `KVPACK_STUDIO_MAX_SAMPLES` | `8192` | Self-study conversations per build. |
| `KVPACK_STUDIO_MAX_CARTRIDGES_PER_ACCOUNT` | `25` | |

## API

Everything except `/health`, `/v1/config` and `/v1/signup` needs
`Authorization: Bearer <api key>`. Errors use OpenAI's format, and interactive docs are at
`/docs`.

| Endpoint | |
| --- | --- |
| `POST /v1/cartridges` | Connect sources and build: `{"name", "sources": [{"type": "git", "url": ...}], "sync_every_hours", "tokens", "samples"}` |
| `POST /v1/cartridges/upload` | Build from uploaded files (multipart `files`). |
| `POST /v1/cartridges/import` | Host a cartridge built with the kvpack CLI (multipart `file`). |
| `POST /v1/cartridges/{id}/sync` | Fetch the sources and rebuild if they changed (`?force=true` always rebuilds). |
| `PATCH /v1/cartridges/{id}` | Rename, or change `sync_every_hours` (0 turns it off). |
| `GET /v1/cartridges`, `GET /v1/cartridges/{id}` | Status, version, sources, progress and quality check. |
| `GET /v1/cartridges/{id}/download` | The `.safetensors` file. |
| `DELETE /v1/cartridges/{id}` | Delete it and its documents. |
| `GET /v1/models`, `POST /v1/chat/completions` | OpenAI-compatible, including streaming. |
| `GET /v1/usage?days=30` | Builds, GPU time, requests and tokens. |
| `GET/POST /v1/keys`, `DELETE /v1/keys/{id}` | Manage API keys. |

## Development

```bash
uv sync --group dev --extra modal
uv run pytest
```

The tests cover the whole journey with a tiny model on CPU: connecting a website and a
folder, syncs that skip unchanged sources, the previous version serving through a failed
update, the scheduler, isolation between accounts, the source security rules, and the
Modal RPC bridge. The Modal deployment is checked to load, but it's only been run in
tests through the same bridge, not on Modal itself.
