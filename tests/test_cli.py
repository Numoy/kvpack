from conftest import DOCUMENT
from typer.testing import CliRunner

from kvpack import Cartridge
from kvpack.cli import app


def test_info_shows_cartridge_details(model, chat_format, tmp_path):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=20)
    cartridge.metadata.update(name="kestrel", corpus_tokens=200)
    path = cartridge.save(tmp_path / "kestrel.safetensors")

    result = CliRunner().invoke(app, ["info", str(path)])
    assert result.exit_code == 0, result.output
    assert "kestrel" in result.output
    assert "tiny-qwen3" in result.output
    assert "10.0x compression" in result.output


def test_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("build", "synthesize", "chat", "compare", "serve", "info"):
        assert command in result.output


def test_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import kvpack

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert kvpack.__version__ == pyproject["project"]["version"]
