"""Pure checks on LLM output: answer validity, heading normalisation and numeric grounding (spec §7)."""
from __future__ import annotations

import math
import numbers
import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
# hyphen-minus, U+2010-U+2015 (hyphen to horizontal bar), U+2212 (minus) or slash: models vary the dash
_SEASON = re.compile(r"\b(?:19|20)\d{2}\s*[-\u2010-\u2015\u2212/]\s*\d{2,4}\b")
_NUMBER = re.compile(r"(?<![\w.])[-−+]?\d+(?:,\d{3})*(?:\.\d+)?")
_PERCENT = re.compile(r"\s*(?:%|percent\b)", re.IGNORECASE)


def strip_think(text: str | None) -> str:
    body = _THINK_BLOCK.sub("", text or "")
    if "</think>" in body and "<think>" not in body:
        body = body.rsplit("</think>", 1)[1]  # reasoning leaked without its opening tag
    return body.strip()


def _heading_key(line: str) -> str:
    """A line reduced to bare text: no #s, numbering, bold/italic markers or trailing colon."""
    core = line.strip().lstrip("#").strip().strip("*_").strip()
    core = re.sub(r"^\d+[.)]\s*", "", core).strip("*_").strip()
    return core.rstrip(":.").strip().strip("*_").strip().casefold()


def _is_heading(line: str, heading: str) -> bool:
    """True when the line is only the heading (#s, numbering, bold and a trailing colon allowed)."""
    return len(line) <= 200 and _heading_key(line) == heading.casefold()


def validate_output(text: str | None, finish_reason: str | None, required_headings: list[str]) -> str | None:
    """Why an answer is unusable, or None if it is fine."""
    if finish_reason == "length":
        return "the answer was cut off at the token limit"
    body = strip_think(text)
    if "<think>" in body:
        return "the answer contains an unclosed <think> block"
    if not body:
        return "the answer was empty"
    lines = body.splitlines()
    missing = [h for h in required_headings if not any(_is_heading(line, h) for line in lines)]
    if missing:
        return "missing heading(s): " + "; ".join(missing)
    return None


def normalize_headings(text: str, headings: list[str]) -> str:
    """Rewrite each recognised heading line as '## Heading' so later parsing is uniform."""
    out = []
    for line in text.splitlines():
        match = next((h for h in headings if _is_heading(line, h)), None)
        out.append(f"## {match}" if match else line)
    return "\n".join(out)


def _value(token: str) -> float:
    return float(token.replace(",", "").replace("−", "-").lstrip("+"))


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def fact_numbers(facts) -> list[float]:
    """Every number in a facts structure, including numbers inside strings and keys."""
    found: list[float] = []

    def walk(obj) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, numbers.Real):
            if math.isfinite(float(obj)):
                found.append(float(obj))
        elif isinstance(obj, str):
            found.extend(_value(t) for t in _NUMBER.findall(_SEASON.sub(" ", obj)))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(facts)
    return found


def _ignored(token: str) -> bool:
    if "." in token:
        return False
    value = abs(_value(token))
    return value <= 10 or 2011 <= value <= 2027


def check_grounding(text: str, facts) -> list[str]:
    """Numbers in `text` that no fact supports (spec §7 tolerance); a fact is scaled x100 only for a percentage."""
    values = [abs(v) for v in fact_numbers(facts)]
    unsupported: list[str] = []
    text = _SEASON.sub(" ", text)
    for match in _NUMBER.finditer(text):
        token = match.group()
        if _ignored(token):
            continue
        scales = (1, 100) if _PERCENT.match(text, match.end()) else (1,)
        target, tolerance = abs(_value(token)), 0.5 * 10 ** -_decimals(token) + 1e-9
        if not any(abs(target - value * scale) <= tolerance for value in values for scale in scales):
            clean = token.lstrip("+-−")
            if clean not in unsupported:
                unsupported.append(clean)
    return unsupported
