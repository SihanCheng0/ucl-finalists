"""HTML/CSS chart builders for the report.

Dataviz specs: bars <= 24px with a 4px rounded data-end and a square baseline end, dots >= 8px with a
2px surface ring, hairline axes, text in ink tokens (never the series colour), a legend for two or more
series and a table twin for every chart. Each row is focusable and carries its tooltip in data-tip.
"""
from __future__ import annotations

from html import escape

GROUP_COLORS = {"pedigree": "var(--s1)", "results": "var(--s2)", "style": "var(--s3)"}
STAGES = ["KO", "QF", "SF", "Final", "Won"]


def _attr(text: str) -> str:
    return escape(text, quote=True)


def legend(items: list[tuple[str, str, str]]) -> str:
    """items: (label, css colour, shape), shape 'bar', 'dot' or 'tick' to mirror the marks."""
    keys = "".join(
        f'<span class="key"><span class="swatch {shape}" style="--c:{color}"></span>{escape(label)}</span>'
        for label, color, shape in items
    )
    return f'<div class="legend">{keys}</div>'


def bar_chart(rows: list[dict], *, baseline: float = 0.0, fmt: str = "{:.2f}", caption: str = "") -> str:
    """Horizontal bars from `baseline`; if any value is below it the bars diverge left and right.

    Rows need label, value, color and tip; tag (chip after the label) and note (under the value) are optional.
    """
    extent = max((abs(r["value"] - baseline) for r in rows), default=0.0) or 1.0
    diverging = any(r["value"] < baseline for r in rows)
    html = []
    for r in rows:
        delta = r["value"] - baseline
        bar = f'<span class="bar" style="--w:{abs(delta) / extent * 100:.1f}%;--c:{r["color"]}"></span>'
        if diverging:
            neg, pos = (bar, "") if delta < 0 else ("", bar)
            track = (f'<span class="track split"><span class="half neg">{neg}</span>'
                     f'<span class="half pos">{pos}</span></span>')
        else:
            track = f'<span class="track">{bar}</span>'
        tag = f'<span class="chip">{escape(r["tag"])}</span>' if r.get("tag") else ""
        note = f'<span class="note">{escape(r["note"])}</span>' if r.get("note") else ""
        html.append(
            f'<div class="bar-row" tabindex="0" data-tip="{_attr(r["tip"])}">'
            f'<span class="bar-label">{escape(r["label"])}{tag}</span>{track}'
            f'<span class="bar-value">{escape(fmt.format(r["value"]))}{note}</span></div>'
        )
    cap = f'<p class="chart-caption">{escape(caption)}</p>' if caption else ""
    kind = "bars diverging" if diverging else "bars"
    return f'<div class="{kind}">{"".join(html)}</div>{cap}'


def dot_chart(rows: list[dict], *, extent: float) -> str:
    """One row per item on a 0..extent scale: a base-rate tick and a dot for the value.

    Rows need label, sub, p, base, value_text and tip.
    """
    html = []
    for r in rows:
        x_dot = min(r["p"] / extent, 1.0) * 100
        x_tick = min(r["base"] / extent, 1.0) * 100
        html.append(
            f'<div class="dot-row" tabindex="0" data-tip="{_attr(r["tip"])}">'
            f'<span class="bar-label">{escape(r["label"])}<small>{escape(r["sub"])}</small></span>'
            f'<span class="track dots"><span class="tick" style="--x:{x_tick:.1f}%"></span>'
            f'<span class="dot" style="--x:{x_dot:.1f}%"></span></span>'
            f'<span class="bar-value">{escape(r["value_text"])}</span></div>'
        )
    ticks = "".join(f'<span style="--x:{i * 50}%">{extent * i / 2:.0%}</span>' for i in range(3))
    axis = f'<div class="axis-row" aria-hidden="true"><span></span><span class="axis">{ticks}</span><span></span></div>'
    return f'<div class="dots">{"".join(html)}{axis}</div>'


def stage_ladder(actual: int, expected: float) -> str:
    """Five rungs from 'knockouts' to 'won'; a caret marks the model's expected stage (0-4)."""
    rungs = "".join(
        f'<span class="rung{" reached" if i <= actual else ""}{" here" if i == actual else ""}">{name}</span>'
        for i, name in enumerate(STAGES)
    )
    x = (min(max(expected, 0.0), 4.0) + 0.5) / 5 * 100
    label = f"Reached: {STAGES[actual]}. Model's expected stage: {expected:.2f} of 4."
    return (f'<div class="ladder" role="img" aria-label="{_attr(label)}">{rungs}'
            f'<span class="expect" style="--x:{x:.1f}%"></span></div>')


def table(headers: list[str], rows: list[list[str]], *, numeric_from: int = 1, label: str = "Table") -> str:
    def cell(tag: str, i: int, value) -> str:
        cls = ' class="num"' if i >= numeric_from else ""
        return f"<{tag}{cls}>{escape(str(value))}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>" for row in rows)
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="{_attr(label)}">'
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def table_twin(summary: str, headers: list[str], rows: list[list[str]], *, numeric_from: int = 1) -> str:
    """The table view every chart needs: tooltips enhance, they never gate."""
    return (f'<details class="twin"><summary>{escape(summary)}</summary>'
            f"{table(headers, rows, numeric_from=numeric_from, label=summary)}</details>")
