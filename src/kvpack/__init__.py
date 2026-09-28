"""kvpack: pack your documents into a trained KV cache."""

from .benchmark import run_benchmark
from .cartridge import Cartridge
from .chat_format import ChatFormat, UnsupportedModelError
from .corpus import load_corpus
from .generate import Completion, complete, generate
from .models import load_model
from .pipeline import BuildResult, build
from .sources import FolderSource, GitSource, Snapshot, WebSource, fetch_all, source_from_uri
from .synthesize import LocalGenerator, OpenAIGenerator, SelfStudyDataset, synthesize
from .train import Metrics, evaluate, train

__version__ = "0.2.0"

__all__ = [
    "BuildResult",
    "Cartridge",
    "ChatFormat",
    "Completion",
    "FolderSource",
    "GitSource",
    "LocalGenerator",
    "Metrics",
    "OpenAIGenerator",
    "SelfStudyDataset",
    "Snapshot",
    "UnsupportedModelError",
    "WebSource",
    "build",
    "complete",
    "evaluate",
    "fetch_all",
    "generate",
    "load_corpus",
    "load_model",
    "run_benchmark",
    "source_from_uri",
    "synthesize",
    "train",
]
