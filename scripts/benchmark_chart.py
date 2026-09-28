"""Draw the README benchmark chart from `kvpack eval --out` reports.

    uv run python scripts/benchmark_chart.py runs/eval-*.json -o docs/assets/benchmark.svg

Three small charts share one row per setup: correct answers, KV cache memory, and
time to first token. They're separate charts on purpose (different units, so no
shared axis). The cartridge rows use the accent color; everything else is neutral.
Light and dark colors switch with the viewer's theme.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

ROW_H, BAR_H, LABEL_W, PANEL_W, GAP = 34, 18, 250, 230, 36
TOP, BOTTOM = 72, 44


def rows_from(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        rag = report["rag"]
        for r in report["results"]:
            if r["skipped"]:
                continue
            if r["setup"] == "cartridge":
                label, accent = f"Cartridge ({report['cartridge_tokens']:,} tokens)", True
            elif r["setup"] == "cartridge+rag":
                chunks = "chunk" if rag["chunks"] == 1 else "chunks"
                label, accent = f"Cartridge + {rag['chunks']} RAG {chunks} ({r['context_tokens']:,} tokens)", True
            elif r["setup"] == "rag":
                label, accent = f"RAG, {rag['chunks']} × {rag['chunk_tokens']}-token chunks", False
            elif r["setup"] == "full":
                label, accent = f"Full documents ({r['context_tokens']:,} tokens)", False
            else:
                label, accent = "No context", False
            rows.append({**r, "row_label": label, "accent": accent})
    order = {"full": 0, "rag": 1, "cartridge+rag": 2, "cartridge": 3, "none": 4}
    rows.sort(key=lambda r: (order[r["setup"]], -r["context_tokens"]))
    return rows


def panel(x: float, title: str, rows: list[dict], value, fmt, max_value: float) -> list[str]:
    out = [f'<text x="{x}" y="{TOP - 22}" class="title">{escape(title)}</text>']
    out.append(f'<line x1="{x}" y1="{TOP - 8}" x2="{x}" y2="{TOP + len(rows) * ROW_H - 8}" class="axis"/>')
    for i, r in enumerate(rows):
        v = value(r)
        y = TOP + i * ROW_H
        width = 0 if max_value == 0 else max(v / max_value * (PANEL_W - 60), 0)
        if width > 0:
            cls = "bar accent" if r["accent"] else "bar"
            # 4px rounded data end, square at the baseline
            w, h = width, BAR_H
            r_ = min(4, w)
            path = f"M{x},{y} h{w - r_} q{r_},0 {r_},{r_} v{h - 2 * r_} q0,{r_} -{r_},{r_} h-{w - r_} z"
            out.append(f'<path class="{cls}" d="{path}"/>')
        out.append(f'<text x="{x + width + 6}" y="{y + BAR_H / 2 + 4}" class="value">{escape(fmt(r))}</text>')
    return out


def render(rows: list[dict], subtitle: str) -> str:
    n = len(rows)
    width = LABEL_W + 3 * PANEL_W + 2 * GAP + 20
    height = TOP + n * ROW_H + BOTTOM
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Benchmark: correct answers, memory and time to first token per setup">',
        "<style>",
        ":root{--bg:#ffffff;--ink:#141a22;--muted:#566170;--line:#d3d9e0;--bar:#7b8594;--accent:#2b45d4}",
        "@media (prefers-color-scheme:dark){:root{--bg:#0d1015;--ink:#e6eaf0;--muted:#97a1ae;--line:#28303a;"
        "--bar:#6d7787;--accent:#6f86f2}}",
        "text{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;fill:var(--ink)}",
        ".title{font-size:14px;font-weight:600}.label{font-size:13px}.label.accent{font-weight:600}",
        ".value{font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums}",
        ".sub{font-size:12px;fill:var(--muted)}.axis{stroke:var(--line);stroke-width:1}",
        ".bar{fill:var(--bar)}.bar.accent{fill:var(--accent)}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="var(--bg)"/>',
    ]
    for i, r in enumerate(rows):
        y = TOP + i * ROW_H + BAR_H / 2 + 4
        cls = "label accent" if r["accent"] else "label"
        parts.append(f'<text x="10" y="{y}" class="{cls}">{escape(r["row_label"])}</text>')

    x0 = LABEL_W
    parts += panel(
        x0, "Correct answers", rows, lambda r: r["accuracy"],
        lambda r: f"{r['correct']}/{r['total']} ({r['accuracy']:.0%})", 1.0,
    )  # fmt: skip
    x1 = x0 + PANEL_W + GAP
    parts += panel(
        x1, "KV cache per conversation", rows, lambda r: r["kv_cache_mib"],
        lambda r: f"{r['kv_cache_mib']:,.0f} MiB", max(r["kv_cache_mib"] for r in rows),
    )  # fmt: skip
    x2 = x1 + PANEL_W + GAP
    parts += panel(
        x2, "Time to first token", rows, lambda r: r["ttft_ms"] or 0,
        lambda r: f"{r['ttft_ms']:,} ms", max(r["ttft_ms"] or 0 for r in rows),
    )  # fmt: skip
    parts.append(f'<text x="10" y="{height - 16}" class="sub">{escape(subtitle)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    first = reports[0]
    subtitle = (
        f"{first['questions']} questions about a {first['corpus_tokens']:,}-token document · {first['model']} · "
        "graded by required facts · lower is better for memory and time"
    )
    args.out.write_text(render(rows_from(reports), subtitle))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
