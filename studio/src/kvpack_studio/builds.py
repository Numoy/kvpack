"""Running cartridge builds (and syncs) and reporting their progress.

`run_build` is the whole job: fetch the cartridge's sources, skip the rest if a sync
finds nothing changed, run kvpack's self-study and training, write the new version,
and keep the database row up to date. The same function runs in a local process
during development and on a Modal GPU in production.

While a new version builds, the previous one keeps serving; if the build fails, the
previous version stays in place.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kvpack import build
from kvpack.sources import Snapshot, SourceError

from .db import Database, now
from .sources import UPLOAD, make_source

log = logging.getLogger("kvpack_studio")

# Share of the progress bar for each stage. Self-study dominates the running time.
SYNTH_SHARE = 0.7


@dataclass
class BuildSpec:
    cartridge_id: str
    account_id: str
    base_model: str
    num_tokens: int
    num_samples: int
    documents_dir: str
    output_path: str
    max_corpus_tokens: int
    sources: list = field(default_factory=list)
    # A sync passes the fingerprint of the current version and force=False: if the
    # sources haven't changed, nothing is rebuilt.
    previous_fingerprint: str | None = None
    force: bool = True
    allow_private_urls: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class BuildError(Exception):
    """A problem with the user's input, reported to them as-is."""


def run_build(
    spec: BuildSpec,
    db: Database,
    load_model: Callable,
    make_generator: Callable | None = None,
    build_options: dict | None = None,
) -> None:
    """Build one cartridge. Never raises: failures are recorded on the cartridge."""
    started = time.monotonic()
    db.update_cartridge(
        spec.cartridge_id, status="building", stage="fetching", progress=0.0, error=None, started_at=now()
    )
    try:
        snapshot = fetch(spec)
        if not spec.force and snapshot.fingerprint == spec.previous_fingerprint:
            db.update_cartridge(spec.cartridge_id, status="ready", stage=None, progress=1.0, last_synced_at=now())
            log.info("%s is up to date", spec.cartridge_id)
            return
        try:
            corpus = snapshot.corpus()
        except ValueError:
            raise BuildError("The sources contain no readable text.") from None
        model, tokenizer, chat_format = load_model(spec.base_model)
        corpus_tokens = len(tokenizer.encode(corpus, add_special_tokens=False))
        if corpus_tokens > spec.max_corpus_tokens:
            raise BuildError(f"The documents are {corpus_tokens:,} tokens; the limit is {spec.max_corpus_tokens:,}.")
        db.update_cartridge(spec.cartridge_id, corpus_tokens=corpus_tokens, stage="synthesizing")

        progress = _ProgressReporter(db, spec.cartridge_id)
        synthesized = 0

        def on_synth(n: int) -> None:
            nonlocal synthesized
            synthesized += n
            progress.report("synthesizing", SYNTH_SHARE * synthesized / spec.num_samples)

        def on_step(step: int, total: int, loss: float) -> None:
            progress.report("training", SYNTH_SHARE + (1 - SYNTH_SHARE) * step / total)

        result = build(
            model,
            chat_format,
            corpus,
            name=spec.cartridge_id,
            num_tokens=min(spec.num_tokens, corpus_tokens),
            num_samples=spec.num_samples,
            generator=make_generator(model, chat_format) if make_generator else None,
            on_synth_progress=on_synth,
            on_train_step=on_step,
            **{"synth_batch_size": _synth_batch_size(spec.num_samples), **(build_options or {})},
        )

        # Write next to the destination, then rename: servers watching the folder
        # never see a half-written file.
        output = Path(spec.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".partial")
        result.cartridge.save(partial)
        partial.replace(output)
        if db.get_cartridge(spec.cartridge_id) is None:  # deleted while it was building
            output.unlink(missing_ok=True)
            return

        history = result.history
        metrics = {
            key: vars(value)
            for key, value in [
                ("no_context", history.eval_no_context),
                ("before_training", history.eval_before),
                ("after_training", history.eval_after),
            ]
            if value is not None
        }
        metrics["held_out_conversations"] = result.cartridge.metadata.get("eval_conversations", 0)
        seconds = time.monotonic() - started
        record = db.get_cartridge(spec.cartridge_id)
        db.update_cartridge(
            spec.cartridge_id,
            status="ready", stage=None, progress=1.0, metrics=metrics, gpu_seconds=seconds, finished_at=now(),
            num_tokens=result.cartridge.num_tokens, sources_fingerprint=snapshot.fingerprint,
            last_synced_at=now(), version=(record.version if record else 0) + 1,
        )  # fmt: skip
        db.record_usage(spec.account_id, "build", cartridge_id=spec.cartridge_id, gpu_seconds=seconds)
        log.info("Built %s in %.0fs", spec.cartridge_id, seconds)
    except Exception as e:
        seconds = time.monotonic() - started
        message = str(e) if isinstance(e, BuildError) else "The build failed unexpectedly. We've logged the error."
        log.error("Build %s failed:\n%s", spec.cartridge_id, traceback.format_exc())
        db.update_cartridge(
            spec.cartridge_id, status="failed", stage=None, error=message, gpu_seconds=seconds, finished_at=now()
        )
        db.record_usage(spec.account_id, "build", cartridge_id=spec.cartridge_id, gpu_seconds=seconds)


def fetch(spec: BuildSpec) -> Snapshot:
    """Fetch every source of a build into one snapshot."""
    configs = spec.sources or [{"type": UPLOAD}]
    snapshots = []
    for config in configs:
        source = make_source(config, spec.documents_dir, spec.allow_private_urls)
        try:
            snapshots.append(source.fetch())
        except SourceError as e:
            raise BuildError(str(e)) from None
    return Snapshot.merge(snapshots)


def _synth_batch_size(num_samples: int) -> int:
    """Big enough to keep the GPU busy, small enough for at least ~8 progress updates."""
    return max(1, min(16, num_samples // 8))


class _ProgressReporter:
    """Writes progress to the database at most every couple of seconds."""

    def __init__(self, db: Database, cartridge_id: str, interval: float = 2.0):
        self.db, self.cartridge_id, self.interval = db, cartridge_id, interval
        self._last = 0.0

    def report(self, stage: str, progress: float) -> None:
        t = time.monotonic()
        if t - self._last >= self.interval or progress >= 1.0:
            self._last = t
            self.db.update_cartridge(self.cartridge_id, stage=stage, progress=min(progress, 0.999))


class LocalBuildRunner:
    """Runs builds one at a time on this machine.

    By default each build runs in a separate process: it gets its own GPU context
    (Apple's Metal backend crashes when two threads use the GPU at once), and a
    crashing build can't take the API server down with it. `isolation="thread"`
    runs builds in a background thread instead, which tests use with a tiny model.
    """

    def __init__(
        self,
        db: Database,
        load_model: Callable,
        make_generator: Callable | None = None,
        isolation: str = "process",
        **build_options,
    ):
        self.db, self.load_model, self.make_generator = db, load_model, make_generator
        self.build_options = build_options
        self.isolation = isolation
        if isolation == "process":
            self._executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        elif isolation == "thread":
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpack-build")
        else:
            raise ValueError(f"isolation must be 'process' or 'thread', not {isolation!r}")

    def submit(self, spec: BuildSpec):
        if self.isolation == "process":
            return self._executor.submit(
                _build_in_subprocess, spec, self.db.engine.url.render_as_string(hide_password=False), self.build_options
            )
        return self._executor.submit(run_build, spec, self.db, self.load_model, self.make_generator, self.build_options)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def _build_in_subprocess(spec: BuildSpec, database_url: str, build_options: dict) -> None:
    import kvpack

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_build(spec, Database(database_url), kvpack.load_model, None, build_options)
