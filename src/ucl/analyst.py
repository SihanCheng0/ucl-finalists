"""Prompts, the narrative retry policy and persistence for the local-LLM analyst (spec §7)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .facts import build_facts
from .grounding import check_grounding, normalize_headings, strip_think, validate_output
from .llm import LLMError, LMStudio
from .model import ModelResults

TEAM_HEADINGS = ["How they got there", "Would the model have picked them?", "Weak spots"]
SYNTH_HEADINGS = ["Why the best teams win", "Winners vs runners-up"]
RETRY_EXCERPT_CHARS = 1500

SYSTEM_PROMPT = (
    "You are a football performance analyst writing for a curious, data-literate reader. "
    "Use ONLY the facts provided. Quote every number exactly as it appears in the facts, with the same rounding. "
    "Do not introduce players, managers, matches, events or numbers that are not in the facts. "
    "Percentiles compare a team with other Champions League teams: 90 means higher than 90% of them. "
    "For stats marked '(lower is better)', a LOW percentile is good. "
    "SHAP contributions are measured in knockout stages: +0.30 means the stat pushed the model's expected "
    "stage up by 0.30 of a round. Write plain markdown without tables."
)
VALIDITY_FEEDBACK = (
    "Your previous answer could not be used: {reason}. Write the complete answer again, following every "
    "instruction and using exactly the required headings."
)
GROUNDING_FEEDBACK = (
    "These numbers in your answer do not appear in the facts: {numbers}. Rewrite the complete answer with the "
    "same headings, quoting only numbers that appear in the facts (same rounding), or describe those points "
    "without numbers."
)
TEAM_INSTRUCTIONS = (
    "Write a scouting report on this Champions League finalist: 180-250 words of markdown with exactly these "
    "three headings, in this order:\n## How they got there\n## Would the model have picked them?\n## Weak spots\n\n"
    "Under 'Would the model have picked them?', compare the model's probability with the base rate and the rank. "
    "The model never saw this season but was trained on the other seasons, including later ones, so describe "
    "whether it would have picked them, not a forecast made at the time.\n\nFacts (JSON):\n"
)
SYNTH_INSTRUCTIONS = (
    "Write 350-450 words of markdown with exactly these two headings, in this order:\n"
    "## Why the best teams win\n## Winners vs runners-up\n\n"
    "Explain which stats drive deep knockout runs and whether pedigree, results or style matters most, using the "
    "feature-set comparison. Only call a stat 'robust' if its robustness label is robust. Never call a "
    "winners-vs-runners-up difference significant unless its Holm-adjusted p is below the significance "
    "threshold in the facts.\n\nFacts (JSON):\n"
)


@dataclass
class Narrative:
    key: str
    status: str  # "ok" | "unavailable"
    text: str | None = None
    unsupported: list[str] = field(default_factory=list)
    calls: int = 0
    reason: str | None = None


@dataclass
class Analysis:
    status: str  # "ok" | "unavailable" | "skipped"
    model: str
    narratives: dict[str, Narrative] = field(default_factory=dict)
    facts: dict[str, dict] = field(default_factory=dict)


def write_narrative(llm, key: str, messages: list[dict], headings: list[str], facts: dict) -> Narrative:
    """At most 3 calls: initial, one validity retry, one grounding retry (spec §7)."""
    calls = 0
    try:
        answer = llm.chat(messages)
        calls += 1
    except LLMError as exc:
        return Narrative(key, "unavailable", calls=calls, reason=f"LLM call failed: {exc}")
    reason = validate_output(answer.content, answer.finish_reason, headings)
    if reason:
        messages = messages + [
            {"role": "assistant", "content": (answer.content or "")[:RETRY_EXCERPT_CHARS]},
            {"role": "user", "content": VALIDITY_FEEDBACK.format(reason=reason)},
        ]
        try:
            answer = llm.chat(messages)
            calls += 1
        except LLMError as exc:
            return Narrative(key, "unavailable", calls=calls, reason=f"LLM call failed: {exc}")
        reason = validate_output(answer.content, answer.finish_reason, headings)
        if reason:
            return Narrative(key, "unavailable", calls=calls, reason=reason)
    text = normalize_headings(strip_think(answer.content), headings)
    unsupported = check_grounding(text, facts)
    if unsupported:
        retry = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": GROUNDING_FEEDBACK.format(numbers=", ".join(unsupported))},
        ]
        try:
            second = llm.chat(retry)
            calls += 1
            if validate_output(second.content, second.finish_reason, headings) is None:
                second_text = normalize_headings(strip_think(second.content), headings)
                second_unsupported = check_grounding(second_text, facts)
                if len(second_unsupported) < len(unsupported):
                    text, unsupported = second_text, second_unsupported
        except LLMError:
            pass  # keep the earlier valid text with its flags
    return Narrative(key, "ok", text=text, unsupported=unsupported, calls=calls)


def _messages(instructions: str, sheet: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instructions + json.dumps(sheet, indent=1, ensure_ascii=False)},
    ]


def run(
    team_seasons: pd.DataFrame,
    finals: pd.DataFrame,
    results: ModelResults,
    llm_model: str = config.LLM_MODEL,
    enabled: bool = True,
    llm=None,
    log: Callable[[str], None] = print,
) -> Analysis:
    if not enabled:
        return Analysis("skipped", llm_model)
    llm = llm or LMStudio(model=llm_model)
    if not llm.ensure_ready():
        return Analysis("unavailable", llm_model)
    sheets = build_facts(team_seasons, finals, results)
    narratives: dict[str, Narrative] = {}
    for key, sheet in sheets.items():
        synth = key == "synthesis"
        messages = _messages(SYNTH_INSTRUCTIONS if synth else TEAM_INSTRUCTIONS, sheet)
        narrative = write_narrative(llm, key, messages, SYNTH_HEADINGS if synth else TEAM_HEADINGS, sheet)
        narratives[key] = narrative
        log(f"  {key}: {narrative.status}, {narrative.calls} call(s), {len(narrative.unsupported)} unsupported")
    return Analysis("ok", llm_model, narratives, sheets)


def save(analysis: Analysis, path: Path = config.OUT_DIR / "analysis.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": analysis.status,
        "model": analysis.model,
        "narratives": {k: asdict(n) for k, n in analysis.narratives.items()},
        "facts": analysis.facts,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load(path: Path = config.OUT_DIR / "analysis.json") -> Analysis:
    path = Path(path)
    if not path.exists():
        return Analysis("skipped", config.LLM_MODEL)
    data = json.loads(path.read_text())
    narratives = {k: Narrative(**n) for k, n in data["narratives"].items()}
    return Analysis(data["status"], data["model"], narratives, data["facts"])
