"""kvpack Studio on Modal.

Three pieces, all in one Modal app:

* `api`: the public web API and dashboard (CPU, scales with traffic).
* `build_cartridge`: one GPU job per cartridge build.
* `Inference`: GPU containers running `kvpack serve` for the base model. They have
  no public URL; the API reaches them through Modal RPC (see `bridge.py`).
* `sync_due_cartridges`: every 15 minutes, starts the automatic syncs that are due.

Documents and cartridges live on a Modal Volume, account data in Postgres.

Deploy (see studio/README.md for the full walkthrough):

    modal secret create kvpack-studio \\
        KVPACK_STUDIO_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DB \\
        KVPACK_STUDIO_INVITE_CODES=your-invite-code
    modal deploy -m kvpack_studio.modal_app
"""

import os

import modal

APP_NAME = "kvpack-studio"

# Read at deploy time and baked into the image, so containers see the same values.
BASE_MODEL = os.environ.get("KVPACK_STUDIO_BASE_MODEL", "Qwen/Qwen3-4B")
INFERENCE_GPU = os.environ.get("KVPACK_STUDIO_INFERENCE_GPU", "L40S")
BUILD_GPU = os.environ.get("KVPACK_STUDIO_BUILD_GPU", "H100")
SYNTH_BATCH_SIZE = int(os.environ.get("KVPACK_STUDIO_SYNTH_BATCH_SIZE", "64"))

DATA_DIR = "/data"
HF_DIR = "/hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # for Git sources
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
        "sqlalchemy==2.1.1",
        "python-multipart==0.0.32",
        "psycopg[binary]==3.3.6",
    )
    .env(
        {
            "HF_HOME": HF_DIR,
            "KVPACK_STUDIO_BASE_MODEL": BASE_MODEL,
            "KVPACK_STUDIO_BASE_MODELS": BASE_MODEL,
            "KVPACK_STUDIO_INFERENCE_GPU": INFERENCE_GPU,
            "KVPACK_STUDIO_BUILD_GPU": BUILD_GPU,
            "KVPACK_STUDIO_SYNTH_BATCH_SIZE": str(SYNTH_BATCH_SIZE),
            "KVPACK_STUDIO_STORAGE_DIR": DATA_DIR,
        }
    )
    # Ship our code plus the dashboard's HTML/CSS/JS (the default would skip non-Python files).
    .add_local_python_source("kvpack", "kvpack_studio", ignore=["**/__pycache__", "**/*.pyc"])
)

app = modal.App(APP_NAME, image=image)
data = modal.Volume.from_name("kvpack-studio-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("kvpack-hf-cache", create_if_missing=True)
secret = modal.Secret.from_name("kvpack-studio", required_keys=["KVPACK_STUDIO_DATABASE_URL"])


@app.function(
    gpu=BUILD_GPU,
    volumes={DATA_DIR: data, HF_DIR: hf_cache},
    secrets=[secret],
    timeout=6 * 60 * 60,
    # If the container is lost mid-build (e.g. preempted), start the build again once.
    retries=modal.Retries(max_retries=1, initial_delay=30),
)
def build_cartridge(spec: dict) -> None:
    import logging

    import kvpack

    from kvpack_studio.builds import BuildSpec, run_build
    from kvpack_studio.config import Settings
    from kvpack_studio.db import Database

    logging.basicConfig(level=logging.INFO)
    data.reload()  # see the documents the API just uploaded
    db = Database(Settings.from_env().database_url)
    run_build(BuildSpec(**spec), db, kvpack.load_model, None, {"synth_batch_size": SYNTH_BATCH_SIZE})
    data.commit()  # make the cartridge visible to the inference containers


@app.cls(
    gpu=INFERENCE_GPU,
    volumes={DATA_DIR: data, HF_DIR: hf_cache},
    secrets=[secret],
    scaledown_window=10 * 60,  # stay warm for 10 minutes after the last request
    timeout=15 * 60,
)
@modal.concurrent(max_inputs=64)  # requests queue inside kvpack's own scheduler
class Inference:
    @modal.enter()
    def start(self) -> None:
        import kvpack
        from kvpack.server import create_app

        from kvpack_studio.storage import Storage
        from kvpack_studio.volumes import VolumeCartridgeStore

        model, _, chat_format = kvpack.load_model(BASE_MODEL)
        directory = Storage(DATA_DIR).cartridge_dir(BASE_MODEL)
        directory.mkdir(parents=True, exist_ok=True)
        store = VolumeCartridgeStore(model.config.name_or_path, device=model.device, directory=directory, volume=data)
        self.app = create_app(model, chat_format, store, max_queue=64)

    @modal.method()
    async def handle(self, method: str, path: str, headers: list, body: bytes):
        from kvpack_studio.bridge import call_asgi

        async for item in call_asgi(self.app, method, path, headers, body):
            yield item


@app.function(volumes={DATA_DIR: data}, secrets=[secret], schedule=modal.Period(minutes=15))
def sync_due_cartridges() -> None:
    """Every 15 minutes: start the automatic syncs that are due."""
    from kvpack_studio.api import queue_due_syncs
    from kvpack_studio.config import Settings
    from kvpack_studio.db import Database
    from kvpack_studio.volumes import VolumeStorage

    settings = Settings.from_env()
    for spec in queue_due_syncs(Database(settings.database_url), VolumeStorage(DATA_DIR, data), settings):
        build_cartridge.spawn(spec.to_dict())


class ModalBuildRunner:
    async def submit(self, spec) -> None:
        await build_cartridge.spawn.aio(spec.to_dict())


@app.function(volumes={DATA_DIR: data}, secrets=[secret], memory=4096, scaledown_window=5 * 60, timeout=15 * 60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="kvpack-studio")
def api():
    import logging

    from kvpack_studio.api import create_app
    from kvpack_studio.config import Settings
    from kvpack_studio.db import Database
    from kvpack_studio.inference import BridgeUpstreams
    from kvpack_studio.volumes import VolumeStorage

    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    db = Database(settings.database_url)
    db.create_tables()

    def inference_channel(*args):
        return Inference().handle.remote_gen.aio(*args)

    return create_app(
        settings,
        db,
        VolumeStorage(DATA_DIR, data),
        ModalBuildRunner(),
        BridgeUpstreams({BASE_MODEL: inference_channel}),
    )
