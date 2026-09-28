"""Database tables and the operations the API needs on them.

SQLite for development, Postgres in production (set KVPACK_STUDIO_DATABASE_URL).
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def hash_key(raw: str) -> str:
    # API keys are 256 random bits, so a fast hash is enough; they're never stored raw.
    return hashlib.sha256(raw.encode()).hexdigest()


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("acct"))
    email: Mapped[str] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=now)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("key"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), default="default")
    prefix: Mapped[str] = mapped_column(String(16))  # shown in the UI so people can tell keys apart
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=now)
    last_used_at: Mapped[datetime | None] = mapped_column(default=None)
    revoked_at: Mapped[datetime | None] = mapped_column(default=None)


class CartridgeRecord(Base):
    __tablename__ = "cartridges"

    def __init__(self, **fields: Any):
        # Assign the id up front (column defaults only apply on insert): uploads are
        # stored under it before the row is written.
        fields.setdefault("id", new_id("ctg"))
        super().__init__(**fields)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("ctg"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    base_model: Mapped[str] = mapped_column(String(200))
    # queued -> building -> ready | failed
    status: Mapped[str] = mapped_column(String(16), default="queued")
    stage: Mapped[str | None] = mapped_column(String(32), default=None)  # while building: synthesizing / training
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    error: Mapped[str | None] = mapped_column(Text, default=None)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    upload_bytes: Mapped[int] = mapped_column(Integer, default=0)
    num_tokens: Mapped[int] = mapped_column(Integer)
    num_samples: Mapped[int] = mapped_column(Integer)
    corpus_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    # Where the knowledge comes from: kvpack source configs, or [{"type": "upload", ...}].
    sources: Mapped[list[Any]] = mapped_column(JSON, default=list)
    sources_fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    sync_every_hours: Mapped[int | None] = mapped_column(Integer, default=None)  # None: only on request
    last_synced_at: Mapped[datetime | None] = mapped_column(default=None)
    # How many times a cartridge file was built. Above 0 it can be served, even while an
    # update is building or after an update failed.
    version: Mapped[int] = mapped_column(Integer, default=0)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(default=now)
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    finished_at: Mapped[datetime | None] = mapped_column(default=None)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "cartridge",
            "name": self.name,
            "base_model": self.base_model,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "error": self.error,
            "file_count": self.file_count,
            "num_tokens": self.num_tokens,
            "num_samples": self.num_samples,
            "corpus_tokens": self.corpus_tokens,
            "metrics": self.metrics,
            "sources": self.sources or [],
            "sync_every_hours": self.sync_every_hours,
            "last_synced_at": _iso(self.last_synced_at),
            "version": self.version,
            "servable": self.version > 0,
            "created_at": _iso(self.created_at),
            "finished_at": _iso(self.finished_at),
        }


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    cartridge_id: Mapped[str | None] = mapped_column(String(32), default=None)
    kind: Mapped[str] = mapped_column(String(16))  # "build" or "chat"
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(default=now, index=True)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() + "Z" if value else None


class Database:
    def __init__(self, url: str):
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, **kwargs)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions() as session, session.begin():
            yield session

    # ------------------------------------------------------------------ accounts and keys

    def create_account(self, email: str) -> tuple[Account, str]:
        """Create an account and its first API key. The raw key is returned once, never stored."""
        with self.session() as s:
            if s.scalar(select(Account).where(Account.email == email.lower())):
                raise ValueError(f"An account for {email} already exists.")
            account = Account(email=email.lower())
            s.add(account)
            s.flush()
            raw = self._add_key(s, account.id, "default")
            return account, raw

    def count_accounts(self) -> int:
        with self.session() as s:
            return s.scalar(select(func.count()).select_from(Account))

    def account_by_email(self, email: str) -> Account | None:
        with self.session() as s:
            return s.scalar(select(Account).where(Account.email == email.lower()))

    def create_key(self, account_id: str, name: str = "default") -> tuple[ApiKey, str]:
        with self.session() as s:
            raw = self._add_key(s, account_id, name)
            return s.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(raw))), raw

    @staticmethod
    def _add_key(s: Session, account_id: str, name: str) -> str:
        raw = "kvp_" + secrets.token_urlsafe(32)
        s.add(ApiKey(account_id=account_id, name=name, prefix=raw[:10], key_hash=hash_key(raw)))
        return raw

    def authenticate(self, raw: str) -> Account | None:
        with self.session() as s:
            key = s.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(raw), ApiKey.revoked_at.is_(None)))
            if key is None:
                return None
            key.last_used_at = now()
            return s.get(Account, key.account_id)

    def list_keys(self, account_id: str) -> list[ApiKey]:
        with self.session() as s:
            query = select(ApiKey).where(ApiKey.account_id == account_id, ApiKey.revoked_at.is_(None))
            return list(s.scalars(query.order_by(ApiKey.created_at)))

    def revoke_key(self, account_id: str, key_id: str) -> bool:
        with self.session() as s:
            key = s.get(ApiKey, key_id)
            if key is None or key.account_id != account_id or key.revoked_at:
                return False
            key.revoked_at = now()
            return True

    # ------------------------------------------------------------------ cartridges

    def add_cartridge(self, record: CartridgeRecord) -> CartridgeRecord:
        with self.session() as s:
            s.add(record)
        return record

    def get_cartridge(self, cartridge_id: str) -> CartridgeRecord | None:
        with self.session() as s:
            return s.get(CartridgeRecord, cartridge_id)

    def find_cartridge(self, account_id: str, id_or_name: str) -> CartridgeRecord | None:
        """Look up one of the account's cartridges by id, or else by name (newest ready one)."""
        with self.session() as s:
            record = s.get(CartridgeRecord, id_or_name)
            if record and record.account_id == account_id:
                return record
            query = (
                select(CartridgeRecord)
                .where(CartridgeRecord.account_id == account_id, CartridgeRecord.name == id_or_name)
                .order_by((CartridgeRecord.status == "ready").desc(), CartridgeRecord.created_at.desc())
            )
            return s.scalars(query).first()

    def list_cartridges(self, account_id: str) -> list[CartridgeRecord]:
        with self.session() as s:
            query = select(CartridgeRecord).where(CartridgeRecord.account_id == account_id)
            return list(s.scalars(query.order_by(CartridgeRecord.created_at.desc())))

    def due_for_sync(self) -> list[CartridgeRecord]:
        """Cartridges with automatic sync whose last sync is older than their interval."""
        current = now()
        with self.session() as s:
            query = select(CartridgeRecord).where(
                CartridgeRecord.sync_every_hours.is_not(None),
                CartridgeRecord.status.not_in(("queued", "building")),
            )
            return [
                r
                for r in s.scalars(query)
                if r.last_synced_at is None or current - r.last_synced_at >= timedelta(hours=r.sync_every_hours)
            ]

    def unfinished_cartridges(self) -> list[CartridgeRecord]:
        with self.session() as s:
            query = select(CartridgeRecord).where(CartridgeRecord.status.in_(("queued", "building")))
            return list(s.scalars(query.order_by(CartridgeRecord.created_at)))

    def count_cartridges(self, account_id: str) -> int:
        with self.session() as s:
            query = select(func.count()).select_from(CartridgeRecord)
            return s.scalar(query.where(CartridgeRecord.account_id == account_id))

    def update_cartridge(self, cartridge_id: str, **fields: Any) -> None:
        with self.session() as s:
            record = s.get(CartridgeRecord, cartridge_id)
            if record is None:
                return  # deleted while building
            for key, value in fields.items():
                setattr(record, key, value)

    def delete_cartridge(self, cartridge_id: str) -> None:
        with self.session() as s:
            record = s.get(CartridgeRecord, cartridge_id)
            if record:
                s.delete(record)

    # ------------------------------------------------------------------ usage

    def record_usage(self, account_id: str, kind: str, **fields: Any) -> None:
        with self.session() as s:
            s.add(UsageEvent(account_id=account_id, kind=kind, **fields))

    def usage_summary(self, account_id: str, since: datetime | None = None) -> dict[str, Any]:
        with self.session() as s:
            query = select(
                UsageEvent.kind,
                func.count(),
                func.coalesce(func.sum(UsageEvent.prompt_tokens), 0),
                func.coalesce(func.sum(UsageEvent.completion_tokens), 0),
                func.coalesce(func.sum(UsageEvent.gpu_seconds), 0.0),
            ).where(UsageEvent.account_id == account_id)
            if since:
                query = query.where(UsageEvent.created_at >= since)
            rows = s.execute(query.group_by(UsageEvent.kind)).all()
        summary = {
            "builds": 0, "build_gpu_seconds": 0.0, "chat_requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
        }  # fmt: skip
        for kind, count, prompt, completion, gpu in rows:
            if kind == "build":
                summary["builds"], summary["build_gpu_seconds"] = count, round(float(gpu), 1)
            elif kind == "chat":
                summary["chat_requests"], summary["prompt_tokens"], summary["completion_tokens"] = (
                    count, int(prompt), int(completion),
                )  # fmt: skip
        return summary
