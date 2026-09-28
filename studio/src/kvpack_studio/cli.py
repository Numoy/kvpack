"""`kvpack-studio`: run the service locally and manage accounts."""

from __future__ import annotations

from typing import Annotated

import typer

from .config import Settings
from .db import Database

app = typer.Typer(help="kvpack Studio administration.", no_args_is_help=True, add_completion=False)


def _db() -> Database:
    db = Database(Settings.from_env().database_url)
    db.create_tables()
    return db


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080):
    """Run kvpack Studio locally, with builds and inference on this machine."""
    import logging

    import uvicorn

    from .app import create_local_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    typer.echo(f"kvpack Studio on http://{host}:{port}  (models: {', '.join(settings.base_models)})")
    uvicorn.run(create_local_app(settings), host=host, port=port, log_level="warning")


@app.command("create-account")
def create_account(email: str):
    """Create an account and print its first API key (shown only once)."""
    try:
        account, key = _db().create_account(email)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    typer.echo(f"Created {account.id} for {account.email}\nAPI key: {key}")


@app.command("create-key")
def create_key(email: str, name: Annotated[str, typer.Option(help="A label for the key.")] = "default"):
    """Create another API key for an existing account."""
    db = _db()
    account = db.account_by_email(email)
    if account is None:
        raise typer.BadParameter(f"No account for {email}")
    _, key = db.create_key(account.id, name)
    typer.echo(f"API key: {key}")


def main() -> None:  # pragma: no cover
    app()
