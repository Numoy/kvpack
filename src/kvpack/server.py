"""An OpenAI-compatible HTTP server for one model and any number of cartridges.

Every cartridge shows up as its own "model" in `/v1/models`, so any OpenAI client
can talk to it:

    client.chat.completions.create(model="kestrel", messages=[...])

The server runs one generation at a time on a single model instance. Requests
beyond that wait in a bounded queue; when the queue is full the server answers
429 so clients can back off, instead of piling up work it can't finish.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedModel, TextIteratorStreamer

from .cartridge import Cartridge
from .chat_format import ChatFormat
from .generate import complete
from .store import CartridgeNotFound, CartridgeStore

log = logging.getLogger("kvpack")


# --------------------------------------------------------------------------- API types


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]

    def text(self) -> str:
        """OpenAI clients may send content as a list of parts; we use the text parts."""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # accept (and ignore) OpenAI fields we don't use

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    n: int = 1
    stream: bool = False
    stream_options: dict[str, Any] | None = None


class APIError(Exception):
    """An error returned to the client in OpenAI's format."""

    def __init__(self, status: int, message: str, type: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code = status, message, type, code

    def response(self) -> JSONResponse:
        body = {"error": {"message": self.message, "type": self.type, "param": None, "code": self.code}}
        return JSONResponse(body, status_code=self.status)


# --------------------------------------------------------------------------- scheduling


class _Gate:
    """One generation at a time, with at most `max_queue` requests waiting."""

    def __init__(self, max_queue: int):
        self.max_queue = max_queue
        self.gpu = threading.Lock()
        self._count_lock = threading.Lock()
        self.in_flight = 0

    def enter(self) -> None:
        with self._count_lock:
            if self.in_flight > self.max_queue:
                raise APIError(429, "The server is busy. Retry shortly.", "rate_limit_error", "server_busy")
            self.in_flight += 1

    def leave(self) -> None:
        with self._count_lock:
            self.in_flight -= 1


# --------------------------------------------------------------------------- app


def create_app(
    model: PreTrainedModel,
    chat_format: ChatFormat,
    cartridges: CartridgeStore | dict[str, Cartridge],
    *,
    api_keys: list[str] | tuple[str, ...] = (),
    max_queue: int = 32,
    max_output_tokens: int = 4096,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        cartridges: a `CartridgeStore`, or a plain {name: Cartridge} dict.
        api_keys: if given, every /v1 request must send `Authorization: Bearer <key>`.
        max_queue: requests allowed to wait behind the running one before 429s.
        max_output_tokens: upper bound on `max_tokens` per request.
    """
    base_name = model.config.name_or_path
    store = (
        cartridges
        if isinstance(cartridges, CartridgeStore)
        else CartridgeStore.from_cartridges(base_name, cartridges, device=model.device)
    )
    gate = _Gate(max_queue)
    context_window = getattr(model.config, "max_position_embeddings", None)
    keys = set(api_keys)

    app = FastAPI(title="kvpack", description="OpenAI-compatible server for KV cartridges")

    @app.exception_handler(APIError)
    async def api_error(_: Request, exc: APIError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return APIError(400, f"Invalid request: {where} {first.get('msg', '')}".strip()).response()

    def authenticate(request: Request) -> None:
        if not keys:
            return
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or header.removeprefix("Bearer ").strip() not in keys:
            raise APIError(401, "Missing or invalid API key.", "authentication_error", "invalid_api_key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": base_name, "in_flight": gate.in_flight}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def list_models() -> dict:
        created = int(time.time())
        entries = [
            {"id": info.name, "object": "model", "created": created, "owned_by": "kvpack",
             "cartridge_tokens": info.num_tokens, "base_model": base_name}
            for info in store.list()
        ]  # fmt: skip
        entries.append({"id": base_name, "object": "model", "created": created, "owned_by": "kvpack"})
        return {"object": "list", "data": entries}

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    def chat_completions(req: ChatRequest):
        started = time.perf_counter()
        cartridge = _resolve(store, req.model, base_name)
        messages = [{"role": m.role, "content": m.text()} for m in req.messages]
        if not messages:
            raise APIError(400, "messages must not be empty.")
        if req.n != 1:
            raise APIError(400, "Only n=1 is supported.")
        max_new = req.max_completion_tokens or req.max_tokens or min(1024, max_output_tokens)
        if max_new > max_output_tokens:
            raise APIError(400, f"max_tokens is {max_new}, but this server allows at most {max_output_tokens}.")
        prompt_tokens = len(chat_format.conversation_ids(messages))
        used = prompt_tokens + (cartridge.num_tokens if cartridge else 0) + max_new
        if context_window and used > context_window:
            raise APIError(
                400,
                f"This request needs {used} tokens of context (cartridge + messages + max_tokens), "
                f"but the model supports {context_window}.",
                code="context_length_exceeded",
            )

        options = dict(cartridge=cartridge, max_new_tokens=max_new, temperature=req.temperature, top_p=req.top_p)
        meta = dict(id=f"chatcmpl-{uuid.uuid4().hex}", created=int(time.time()), model=req.model)
        gate.enter()
        if req.stream:
            include_usage = bool((req.stream_options or {}).get("include_usage"))
            stream = _stream(model, chat_format, messages, options, gate, meta, prompt_tokens, started, include_usage)
            return StreamingResponse(stream, media_type="text/event-stream")

        try:
            with gate.gpu:
                completion = complete(model, chat_format, messages, **options)
        finally:
            gate.leave()
        completion_tokens = len(completion.token_ids)
        _log_request(req.model, prompt_tokens, completion_tokens, started)
        return {
            **meta,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion.text},
                    "finish_reason": completion.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def _resolve(store: CartridgeStore, name: str, base_name: str) -> Cartridge | None:
    if name == base_name:
        return None  # the plain model, without a cartridge
    try:
        return store.get(name)
    except CartridgeNotFound:
        available = [info.name for info in store.list()] + [base_name]
        raise APIError(404, f"Unknown model {name!r}. Available: {available}", code="model_not_found") from None


def _stream(
    model, chat_format, messages, options, gate, meta, prompt_tokens, started, include_usage=False
) -> Iterator[str]:
    """Server-sent events in OpenAI's chunk format.

    With `include_usage` (OpenAI's `stream_options.include_usage`), a final chunk with
    empty `choices` reports token usage.

    If the client disconnects, the generator is closed; we then cancel generation and
    wait for it to stop before releasing the model, so a dropped connection never
    leaves two generations running at once.
    """

    def chunk(delta: dict, finish_reason: str | None = None) -> str:
        body = {**meta, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}  # fmt: skip
        return f"data: {json.dumps(body)}\n\n"

    cancel = threading.Event()
    result: list = []
    try:
        with gate.gpu:
            streamer = TextIteratorStreamer(chat_format.tokenizer, skip_prompt=True, skip_special_tokens=True)
            worker = threading.Thread(
                target=lambda: result.append(
                    complete(model, chat_format, messages, **options, streamer=streamer, cancel=cancel)
                )
            )
            worker.start()
            finished = False
            try:
                yield chunk({"role": "assistant"})
                for text in streamer:
                    if text:
                        yield chunk({"content": text})
                finished = True
            finally:
                if not finished:  # the client went away mid-stream
                    cancel.set()
                worker.join()
        finish = result[0].finish_reason if result else "stop"
        completion_tokens = len(result[0].token_ids) if result else 0
        _log_request(meta["model"], prompt_tokens, completion_tokens, started)
        yield chunk({}, finish_reason=finish)
        if include_usage:
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens + completion_tokens}  # fmt: skip
            yield f"data: {json.dumps({**meta, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        gate.leave()


def _log_request(model: str, prompt_tokens: int, completion_tokens: int, started: float) -> None:
    log.info(
        "chat model=%s prompt_tokens=%d completion_tokens=%d latency_ms=%d",
        model,
        prompt_tokens,
        completion_tokens,
        (time.perf_counter() - started) * 1000,
    )
