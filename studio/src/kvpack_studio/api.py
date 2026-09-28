"""The kvpack Studio HTTP API.

Management endpoints (accounts, keys, cartridges, usage) live here. Chat requests
are authenticated, checked for ownership, metered, and forwarded to the kvpack
server for the cartridge's base model (see `inference.py`). The API is
OpenAI-compatible: point any OpenAI client at `/v1` and use a cartridge id or
name as the model.
"""

import inspect
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from kvpack import Cartridge
from kvpack.cartridge import peek
from kvpack.corpus import TEXT_SUFFIXES
from kvpack.server import APIError
from pydantic import BaseModel

from .builds import BuildSpec
from .config import Settings
from .db import Account, CartridgeRecord, Database, now
from .inference import Upstreams
from .sources import UPLOAD, InvalidSource
from .sources import validate as validate_source
from .storage import Storage, safe_filename

log = logging.getLogger("kvpack_studio")

UPLOAD_SUFFIXES = TEXT_SUFFIXES | {".pdf"}
WEB_DIR = Path(__file__).parent / "web"


class BuildRunner(Protocol):
    def submit(self, spec: BuildSpec) -> Any: ...


class SignupRequest(BaseModel):
    email: str
    invite_code: str = ""


class KeyRequest(BaseModel):
    name: str = "default"


class CreateRequest(BaseModel):
    name: str
    sources: list[dict[str, Any]]
    base_model: str | None = None
    tokens: int | None = None
    samples: int | None = None
    sync_every_hours: int | None = None


class UpdateRequest(BaseModel):
    name: str | None = None
    sync_every_hours: int | None = None  # 0 turns automatic sync off


SYNCABLE = {"git", "web", "folder"}


def create_app(settings: Settings, db: Database, storage: Storage, runner: BuildRunner, upstreams: Upstreams):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await upstreams.aclose()

    app = FastAPI(title="kvpack Studio", docs_url="/docs", redoc_url=None, lifespan=lifespan)

    @app.exception_handler(APIError)
    async def api_error(_: Request, exc: APIError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return APIError(400, f"Invalid request: {where} {first.get('msg', '')}".strip()).response()

    def account(request: Request) -> Account:
        header = request.headers.get("authorization", "")
        raw = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        found = db.authenticate(raw) if raw else None
        if found is None:
            raise APIError(401, "Missing or invalid API key.", "authentication_error", "invalid_api_key")
        return found

    Auth = Annotated[Account, Depends(account)]

    def owned(acct: Account, cartridge_id: str) -> CartridgeRecord:
        record = db.get_cartridge(cartridge_id)
        if record is None or record.account_id != acct.id:
            raise APIError(404, f"No cartridge {cartridge_id!r}.", "not_found_error", "cartridge_not_found")
        return record

    # ------------------------------------------------------------------ service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/config")
    def public_config() -> dict:
        """What the web app needs to know before sign-in."""
        return {
            "base_models": settings.base_models,
            "signup_open": bool(settings.invite_codes) or db.count_accounts() == 0,
            "first_account": db.count_accounts() == 0,
            "sources": {
                "folder": bool(settings.folder_roots),
                "folder_roots": [str(r) for r in settings.folder_roots],
                "git_token_envs": settings.git_token_envs,
            },
            "limits": {
                "max_upload_mb": settings.max_upload_mb,
                "max_files": settings.max_files,
                "max_corpus_tokens": settings.max_corpus_tokens,
                "max_cartridge_tokens": settings.max_cartridge_tokens,
                "max_samples": settings.max_samples,
                "max_cartridges": settings.max_cartridges_per_account,
            },
            "defaults": {"tokens": settings.default_cartridge_tokens, "samples": settings.default_samples},
            "accepted_file_types": sorted(UPLOAD_SUFFIXES),
        }

    # ------------------------------------------------------------------ accounts and keys

    @app.post("/v1/signup", status_code=201)
    def signup(req: SignupRequest) -> dict:
        first_account = db.count_accounts() == 0  # whoever sets up the server
        if not first_account and req.invite_code not in settings.invite_codes:
            raise APIError(403, "That invite code isn't valid.", "permission_error", "invalid_invite_code")
        email = req.email.strip()
        if "@" not in email or len(email) > 320:
            raise APIError(400, "Enter a valid email address.")
        try:
            acct, raw_key = db.create_account(email)
        except ValueError:
            raise APIError(409, "An account with this email already exists.", code="account_exists") from None
        return {"account": {"id": acct.id, "email": acct.email}, "api_key": raw_key}

    @app.get("/v1/me")
    def me(acct: Auth) -> dict:
        return {
            "id": acct.id,
            "email": acct.email,
            "created_at": acct.created_at.isoformat() + "Z",
            "cartridges": db.count_cartridges(acct.id),
        }

    @app.get("/v1/keys")
    def list_keys(acct: Auth) -> dict:
        keys = [
            {"id": k.id, "name": k.name, "prefix": k.prefix, "created_at": k.created_at.isoformat() + "Z",
             "last_used_at": k.last_used_at.isoformat() + "Z" if k.last_used_at else None}
            for k in db.list_keys(acct.id)
        ]  # fmt: skip
        return {"object": "list", "data": keys}

    @app.post("/v1/keys", status_code=201)
    def create_key(acct: Auth, req: KeyRequest) -> dict:
        key, raw = db.create_key(acct.id, req.name.strip()[:100] or "default")
        return {"id": key.id, "name": key.name, "prefix": key.prefix, "api_key": raw}

    @app.delete("/v1/keys/{key_id}")
    def revoke_key(acct: Auth, key_id: str) -> dict:
        if not db.revoke_key(acct.id, key_id):
            raise APIError(404, f"No key {key_id!r}.", "not_found_error", "key_not_found")
        return {"id": key_id, "deleted": True}

    # ------------------------------------------------------------------ cartridges

    def check_new_cartridge(acct: Account, base_model: str | None, tokens: int | None, samples: int | None):
        base_model = base_model or settings.base_models[0]
        tokens = tokens or settings.default_cartridge_tokens
        samples = samples or settings.default_samples
        if base_model not in settings.base_models:
            raise APIError(400, f"base_model must be one of {settings.base_models}.")
        if not 16 <= tokens <= settings.max_cartridge_tokens:
            raise APIError(400, f"tokens must be between 16 and {settings.max_cartridge_tokens}.")
        if not 8 <= samples <= settings.max_samples:
            raise APIError(400, f"samples must be between 8 and {settings.max_samples}.")
        if db.count_cartridges(acct.id) >= settings.max_cartridges_per_account:
            raise APIError(
                403,
                f"You've reached the limit of {settings.max_cartridges_per_account} cartridges. Delete one first.",
                "permission_error",
                "cartridge_limit_reached",
            )
        return base_model, tokens, samples

    def check_sync_interval(hours: int | None) -> int | None:
        if hours in (None, 0):
            return None
        if not 1 <= hours <= 24 * 30:
            raise APIError(400, "sync_every_hours must be between 1 and 720, or 0 to turn it off.")
        return hours

    async def submit(spec: BuildSpec) -> None:
        submitted = runner.submit(spec)
        if inspect.isawaitable(submitted):
            await submitted

    @app.post("/v1/cartridges", status_code=202)
    async def create_cartridge(acct: Auth, req: CreateRequest) -> dict:
        """Connect sources (Git repositories, websites, server folders) and build a cartridge from them."""
        base_model, tokens, samples = check_new_cartridge(acct, req.base_model, req.tokens, req.samples)
        if not 1 <= len(req.sources) <= 10:
            raise APIError(400, "Connect between 1 and 10 sources.")
        try:
            sources = [validate_source(config, settings) for config in req.sources]
        except InvalidSource as e:
            raise APIError(400, str(e), code="invalid_source") from None
        record = CartridgeRecord(
            account_id=acct.id,
            name=req.name.strip()[:100] or "cartridge",
            base_model=base_model,
            num_tokens=tokens,
            num_samples=samples,
            sources=sources,
            sync_every_hours=check_sync_interval(req.sync_every_hours),
        )
        db.add_cartridge(record)
        await submit(build_spec(record, storage, settings))
        log.info("Queued %s for %s (%d sources)", record.id, acct.id, len(sources))
        return record.public()

    @app.post("/v1/cartridges/upload", status_code=202)
    async def upload_cartridge(
        acct: Auth,
        files: Annotated[list[UploadFile], File(description="The documents")],
        name: Annotated[str | None, Form()] = None,
        base_model: Annotated[str | None, Form()] = None,
        tokens: Annotated[int | None, Form()] = None,
        samples: Annotated[int | None, Form()] = None,
    ) -> dict:
        """Build a cartridge from uploaded files."""
        base_model, tokens, samples = check_new_cartridge(acct, base_model, tokens, samples)
        if not files:
            raise APIError(400, "Upload at least one document.")
        if len(files) > settings.max_files:
            raise APIError(400, f"Upload at most {settings.max_files} files.")
        for f in files:
            if Path(f.filename or "").suffix.lower() not in UPLOAD_SUFFIXES:
                raise APIError(400, f"{f.filename!r} isn't a supported file type.", code="unsupported_file_type")

        record = CartridgeRecord(
            account_id=acct.id,
            name=(name or Path(files[0].filename or "documents").stem).strip()[:100] or "documents",
            base_model=base_model,
            num_tokens=tokens,
            num_samples=samples,
            file_count=len(files),
        )
        limit, total, names = settings.max_upload_mb * 2**20, 0, []
        for f in files:
            data = await f.read()
            total += len(data)
            if total > limit:
                storage.delete(record.id, base_model)
                raise APIError(413, f"The upload is larger than {settings.max_upload_mb} MB.", code="upload_too_large")
            names.append(storage.save_upload(record.id, safe_filename(f.filename or "document.txt"), data).name)
        record.upload_bytes = total
        record.sources = [{"type": UPLOAD, "files": names}]
        storage.persist()  # the build may run on another machine and must see the uploads
        db.add_cartridge(record)
        await submit(build_spec(record, storage, settings))
        log.info("Queued %s for %s (%d files, %d bytes)", record.id, acct.id, len(files), total)
        return record.public()

    @app.post("/v1/cartridges/{cartridge_id}/sync", status_code=202)
    async def sync_cartridge(acct: Auth, cartridge_id: str, force: bool = False) -> dict:
        """Fetch the sources again and rebuild if anything changed (or always, with force=true)."""
        record = owned(acct, cartridge_id)
        if not any(src.get("type") in SYNCABLE | {UPLOAD} for src in record.sources or []):
            raise APIError(400, "This cartridge has no sources to sync.", code="nothing_to_sync")
        if record.status in ("queued", "building"):
            raise APIError(409, "This cartridge is already being built.", code="already_building")
        db.update_cartridge(record.id, status="queued", stage=None, progress=0.0)
        await submit(build_spec(record, storage, settings, force=force or record.version == 0))
        return db.get_cartridge(record.id).public()

    @app.patch("/v1/cartridges/{cartridge_id}")
    def update_cartridge(acct: Auth, cartridge_id: str, req: UpdateRequest) -> dict:
        record = owned(acct, cartridge_id)
        fields: dict[str, Any] = {}
        if req.name is not None:
            fields["name"] = req.name.strip()[:100] or record.name
        if "sync_every_hours" in req.model_fields_set:
            if req.sync_every_hours and not any(src.get("type") in SYNCABLE for src in record.sources or []):
                raise APIError(400, "Only cartridges with connected sources can sync automatically.")
            fields["sync_every_hours"] = check_sync_interval(req.sync_every_hours)
        db.update_cartridge(record.id, **fields)
        return db.get_cartridge(record.id).public()

    @app.post("/v1/cartridges/import", status_code=201)
    async def import_cartridge(
        acct: Auth,
        file: Annotated[UploadFile, File(description="A cartridge built with `kvpack build`")],
        name: Annotated[str | None, Form()] = None,
    ) -> dict:
        """Host a cartridge you built yourself with the open-source kvpack CLI."""
        if db.count_cartridges(acct.id) >= settings.max_cartridges_per_account:
            raise APIError(403, "You've reached your cartridge limit.", "permission_error", "cartridge_limit_reached")
        data = await file.read(settings.max_upload_mb * 2**20 + 1)
        if len(data) > settings.max_upload_mb * 2**20:
            raise APIError(413, f"The file is larger than {settings.max_upload_mb} MB.", code="upload_too_large")
        record = CartridgeRecord(account_id=acct.id, name="import", base_model="", num_tokens=0, num_samples=0)
        staging = storage.root / "imports" / f"{record.id}.safetensors"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_bytes(data)
        try:
            info = peek(staging)
            Cartridge.load(staging)  # make sure the tensors are intact, not just the header
        except Exception:
            staging.unlink(missing_ok=True)
            raise APIError(400, "That file isn't a kvpack cartridge.", code="invalid_cartridge") from None
        if info.model not in settings.base_models:
            staging.unlink(missing_ok=True)
            raise APIError(400, f"The cartridge was built for {info.model!r}; supported: {settings.base_models}.")

        meta = info.metadata
        metrics = {
            label: meta[key]
            for label, key in [("no_context", "eval_no_context"), ("before_training", "eval_before"),
                               ("after_training", "eval_after")]
            if meta.get(key)
        }  # fmt: skip
        if metrics and meta.get("eval_conversations"):
            metrics["held_out_conversations"] = meta["eval_conversations"]
        destination = storage.cartridge_path(info.model, record.id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        record.name = (name or meta.get("name") or Path(file.filename or "cartridge").stem).strip()[:100]
        record.base_model, record.status, record.progress, record.version = info.model, "ready", 1.0, 1
        record.sources = [{"type": "import", "file": safe_filename(file.filename or "cartridge.safetensors")}]
        record.num_tokens, record.num_samples = info.num_tokens, meta.get("num_samples") or 0
        record.corpus_tokens, record.metrics, record.finished_at = meta.get("corpus_tokens"), metrics or None, now()
        record.upload_bytes = len(data)
        storage.persist()
        db.add_cartridge(record)
        return record.public()

    @app.get("/v1/cartridges")
    def list_cartridges(acct: Auth) -> dict:
        return {"object": "list", "data": [c.public() for c in db.list_cartridges(acct.id)]}

    @app.get("/v1/cartridges/{cartridge_id}")
    def get_cartridge(acct: Auth, cartridge_id: str) -> dict:
        return owned(acct, cartridge_id).public()

    @app.delete("/v1/cartridges/{cartridge_id}")
    def delete_cartridge(acct: Auth, cartridge_id: str) -> dict:
        record = owned(acct, cartridge_id)
        db.delete_cartridge(record.id)
        storage.delete(record.id, record.base_model)
        storage.persist()
        return {"id": record.id, "deleted": True}

    @app.get("/v1/cartridges/{cartridge_id}/download")
    def download_cartridge(acct: Auth, cartridge_id: str) -> FileResponse:
        record = owned(acct, cartridge_id)
        storage.sync()  # the cartridge was written by a build on another machine
        path = storage.cartridge_path(record.base_model, record.id)
        if record.version == 0 or not path.exists():
            raise APIError(409, "This cartridge isn't ready yet.", code="cartridge_not_ready")
        return FileResponse(path, filename=f"{safe_filename(record.name)}.safetensors")

    # ------------------------------------------------------------------ OpenAI-compatible

    @app.get("/v1/models")
    def list_models(acct: Auth) -> dict:
        ready = [c for c in db.list_cartridges(acct.id) if c.version > 0]
        data = [
            {"id": c.id, "object": "model", "created": int(c.created_at.timestamp()), "owned_by": acct.id,
             "name": c.name, "base_model": c.base_model, "cartridge_tokens": c.num_tokens}
            for c in ready
        ]  # fmt: skip
        data += [{"id": m, "object": "model", "created": 0, "owned_by": "kvpack"} for m in settings.base_models]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(acct: Auth, request: Request):
        try:
            body = await request.json()
        except ValueError:
            raise APIError(400, "The request body must be JSON.") from None
        requested = body.get("model") if isinstance(body, dict) else None
        if not isinstance(requested, str):
            raise APIError(400, "Set `model` to a cartridge id or name.")

        if requested in settings.base_models:
            base_model, upstream_model, cartridge_id = requested, requested, None
        else:
            record = db.find_cartridge(acct.id, requested)
            if record is None:
                raise APIError(404, f"No cartridge or model {requested!r}.", "not_found_error", "model_not_found")
            if record.version == 0:
                detail = f"{record.progress:.0%} done" if record.status == "building" else record.status
                raise APIError(409, f"Cartridge {record.name!r} isn't ready ({detail}).", code="cartridge_not_ready")
            base_model, upstream_model, cartridge_id = record.base_model, record.id, record.id

        client = upstreams.client(base_model)
        upstream_body = {**body, "model": upstream_model}
        stream = bool(body.get("stream"))

        def meter(usage: dict | None) -> None:
            if usage:
                db.record_usage(
                    acct.id, "chat", cartridge_id=cartridge_id,
                    prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                )  # fmt: skip

        if not stream:
            r = await client.post("/v1/chat/completions", json=upstream_body)
            if r.status_code != 200:
                return _upstream_error(r.status_code, r.content, requested)
            data = r.json()
            data["model"] = requested
            meter(data.get("usage"))
            return JSONResponse(data)

        client_wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        upstream_body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        r = await client.send(client.build_request("POST", "/v1/chat/completions", json=upstream_body), stream=True)
        if r.status_code != 200:
            content = await r.aread()
            await r.aclose()
            return _upstream_error(r.status_code, content, requested)

        async def relay():
            usage = None
            try:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line.removeprefix("data: "))
                        chunk["model"] = requested
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                            if not client_wants_usage:
                                continue
                        line = "data: " + json.dumps(chunk)
                    yield line + "\n\n"
            finally:
                await r.aclose()
                meter(usage)

        return StreamingResponse(relay(), media_type="text/event-stream")

    @app.get("/v1/usage")
    def usage(acct: Auth, days: int = 30) -> dict:
        days = max(1, min(days, 365))
        return {"days": days, **db.usage_summary(acct.id, since=now() - timedelta(days=days))}

    # ------------------------------------------------------------------ web app

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app


def build_spec(record: CartridgeRecord, storage: Storage, settings: Settings, force: bool = True) -> BuildSpec:
    return BuildSpec(
        cartridge_id=record.id,
        account_id=record.account_id,
        base_model=record.base_model,
        num_tokens=record.num_tokens,
        num_samples=record.num_samples,
        documents_dir=str(storage.documents_dir(record.id)),
        output_path=str(storage.cartridge_path(record.base_model, record.id)),
        max_corpus_tokens=settings.max_corpus_tokens,
        sources=list(record.sources or []),
        previous_fingerprint=record.sources_fingerprint,
        force=force,
        allow_private_urls=settings.allow_private_urls,
    )


def queue_due_syncs(db: Database, storage: Storage, settings: Settings) -> list[BuildSpec]:
    """Mark cartridges whose automatic sync is due as queued, and return their build specs."""
    specs = []
    for record in db.due_for_sync():
        if not any(src.get("type") in SYNCABLE for src in record.sources or []):
            continue
        db.update_cartridge(record.id, status="queued", stage=None, progress=0.0)
        specs.append(build_spec(record, storage, settings, force=record.version == 0))
    return specs


def requeue_unfinished(db: Database, storage: Storage, settings: Settings, runner: BuildRunner) -> int:
    """Restart builds that were interrupted (e.g. by a restart or crash). Returns how many."""
    unfinished = db.unfinished_cartridges()
    for record in unfinished:
        db.update_cartridge(record.id, status="queued", stage=None, progress=0.0)
        runner.submit(build_spec(record, storage, settings, force=record.version == 0))  # local runners are sync
    return len(unfinished)


def _upstream_error(status: int, content: bytes, requested: str) -> JSONResponse:
    """Pass the upstream error through, except details that could reveal other accounts' cartridges."""
    if status == 404:
        return APIError(404, f"No cartridge or model {requested!r}.", "not_found_error", "model_not_found").response()
    try:
        body = json.loads(content)
    except ValueError:
        body = {"error": {"message": "The model server returned an error.", "type": "api_error"}}
    return JSONResponse(body, status_code=status if status < 500 else 502)
