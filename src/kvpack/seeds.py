"""Seed prompts that steer what kind of question gets asked about each chunk.

Diversity matters: a cartridge trained only on factual questions gets worse at
summarizing, and vice versa. These prompt families are adapted from the
Cartridges paper's self-study recipe (Eyuboglu et al., 2025; Apache-2.0).
"""

from __future__ import annotations

import random

SEED_TYPES = ("question", "summarization", "structuring", "use_case", "creative")


def _question(rng: random.Random) -> str:
    return rng.choice(
        [
            "Generate a question that tests knowledge of the information in the corpus above. "
            "Include specific details (names, ids, titles, dates, numbers) so it is clear what you are asking about. "
            "Output only the question.",
            "Write one question about the corpus above that could be answered in a closed-book setting. "
            "Mention enough specifics (names, titles, dates) that the question is unambiguous. "
            "Output only the question.",
            "You are quizzing someone on the section of the corpus above. Ask one precise question about it. "
            "Output only the question.",
        ]
    )


def _summarization(rng: random.Random) -> str:
    return rng.choice(
        [
            "Write a single chat message asking an assistant to summarize a specific part of the corpus. "
            "Be explicit about which section you mean, using names, titles or dates. Output only the message.",
            "Write a single chat message asking for a summary of one section of the corpus above, "
            "naming the section and document clearly. Output only the message.",
        ]
    )


def _structuring(rng: random.Random) -> str:
    fmt = rng.choice(["JSON", "YAML", "TOML", "a Markdown table", "a bulleted list", "plain text"])
    return (
        f"Write a single chat message asking an assistant to extract information from a specific part of the "
        f"corpus above and format it as {fmt}. Ask it to include precise details like dates, names and numbers. "
        "Make it clear which section you mean. Output only the message."
    )


def _use_case(rng: random.Random) -> str:
    return (
        "Think about a practical, real-world task someone could accomplish using the information in the corpus "
        "above - applying it, not just recalling it. Write the single question or request that person would ask. "
        "Output only the question."
    )


def _creative(rng: random.Random) -> str:
    return (
        "Start a thoughtful, creative conversation inspired by the corpus above. Write the opening question for "
        "your conversation partner. Output only the question."
    )


_GENERATORS = {
    "question": _question,
    "summarization": _summarization,
    "structuring": _structuring,
    "use_case": _use_case,
    "creative": _creative,
}


def sample_seed_prompt(rng: random.Random, seed_types: tuple[str, ...] = SEED_TYPES) -> tuple[str, str]:
    """Returns (seed_type, prompt)."""
    seed_type = rng.choice(seed_types)
    return seed_type, _GENERATORS[seed_type](rng)
