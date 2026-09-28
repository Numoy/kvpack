"""Wiring the API together with local (in-process) builds and inference."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import kvpack

from .api import create_app, queue_due_syncs, requeue_unfinished
from .builds import LocalBuildRunner
from .config import Settings
from .db import Database
from .inference import LocalUpstreams
from .storage import Storage


def cached_model_loader(load: Callable = kvpack.load_model) -> Callable:
    """Load each base model once and share it between builds and inference."""
    cache: dict[str, tuple] = {}
    lock = threading.Lock()

    def load_model(name: str):
        with lock:
            if name not in cache:
                cache[name] = load(name)
            return cache[name]

    return load_model


def create_local_app(
    settings: Settings | None = None,
    *,
    load_model: Callable | None = None,
    make_generator: Callable | None = None,
    build_options: dict | None = None,
    build_isolation: str = "process",
    sync_check_seconds: float | None = 60,
):
    """kvpack Studio with everything in this process: good for development and tests.

    Builds run one at a time in a separate process; chat goes to in-process kvpack
    servers. On a GPU machine this is also a perfectly usable single-node deployment.
    """
    settings = settings or Settings.from_env()
    db = Database(settings.database_url)
    db.create_tables()
    storage = Storage(settings.storage_dir)
    load = load_model or cached_model_loader()
    runner = LocalBuildRunner(db, load, make_generator, isolation=build_isolation, **(build_options or {}))
    app = create_app(settings, db, storage, runner, LocalUpstreams(storage, load))
    requeue_unfinished(db, storage, settings, runner)
    app.state.db, app.state.runner, app.state.settings = db, runner, settings
    if sync_check_seconds:
        app.state.stop_sync = start_sync_scheduler(db, storage, settings, runner, sync_check_seconds)
    return app


def start_sync_scheduler(db, storage, settings, runner, every_seconds: float) -> threading.Event:
    """Queue automatic syncs as they fall due. Set the returned event to stop."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_seconds):
            try:
                for spec in queue_due_syncs(db, storage, settings):
                    runner.submit(spec)
            except Exception:
                logging.getLogger("kvpack_studio").exception("Sync scheduler failed; retrying next round")

    threading.Thread(target=loop, name="kvpack-sync", daemon=True).start()
    return stop
