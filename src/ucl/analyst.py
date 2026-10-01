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
    "Do not introduce players, managers, matches, events, analyses or numbers that are not in the facts. "
    "'Teams beaten (%)' figures say how many teams the club did better than on a stat, so higher is always "
    "better, including for stats marked '(lower is better)', where doing better means having the lower value. "
    "Write them as 'better than X% of teams', never as 'top X%'. "
    "SHAP contributions are measured in knockout stages: a positive contribution means the stat pushed the "
    "model's expected stage up by that fraction of a round, a negative one pushed it down. A SHAP contribution "
    "is the model's view, not a measure of how good the team was at the stat: a team can rank high on a stat "
    "the model marked down, so never call a stat a strength or a weakness because of its SHAP sign. "
    "Everything here is a statistical association, not a cause: never say a stat or a team's profile drove, "
    "powered, propelled, carried, helped, hindered or prevented a result, and do not explain why a stat got its "
    "contribution. Use 'significant' or 'significantly' only for a Holm-adjusted p-value comparison, never for "
    "other differences. Write plain markdown without tables."
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
    "Write a scouting report on this Champions League finalist: 180-250 words of markdown in total, with one "
    "paragraph under each of exactly these three headings, in this order:\n"
    "## How they got there\n## Would the model have picked them?\n## Weak spots\n\n"
    "For each stat, give its value and the share of that season's teams it beat.\n"
    "Under 'How they got there' (2-3 sentences), state the result without mentioning the model or the number of "
    "knockout teams, then describe the team's profile using the stats under 'Strongest stats' and no others.\n"
    "Under 'Would the model have picked them?' (3-4 sentences), give the verdict from the facts, then compare the "
    "model's probability with the base rate, name the stage given as nearest to the model's expectation, and name "
    "the first stat under 'Stats that pushed the prediction up most' and the first under 'Stats that pushed the "
    "prediction down most', with their SHAP contributions. The model never saw this season but was trained on the "
    "other seasons, including later ones, so describe whether it would have picked them, not a forecast made at "
    "the time.\n"
    "Under 'Weak spots', write one sentence for each stat under 'Weakest stats', however few, and nothing else; "
    "do not mention SHAP.\n\nFacts (JSON):\n"
)
SYNTH_INSTRUCTIONS = (
    "Write 350-450 words of markdown in total, two paragraphs of about 100 words under each of exactly these two "
    "headings, in this order:\n## Why the best teams win\n## Winners vs runners-up\n\n"
    "Explain which stats are associated with deep knockout runs and whether pedigree, results or style matters "
    "most, using the feature-set comparison. Say how well the model does by comparing its Brier score with the "
    "base-rate Brier score and stating what share of the finalists were in its top 4; the model's edge over the "
    "base-rate guess was not tested, so do not call it significant or superior. 'Importance' is the average size "
    "of a stat's push on the expected stage, in either direction. The model learns from the knockout team-seasons, "
    "using each team's group/league-phase stats. Report findings only: do not explain how SHAP or the model works. "
    "Only call a stat 'robust' if its robustness label is robust. "
    "Winners vs runners-up compares those same group/league-phase stats, not what happened in the final; discuss "
    "only the first four stats in that comparison (it is sorted from smallest to largest Holm-adjusted p) and say "
    "whether any difference was significant. "
    "Never call a winners-vs-runners-up difference significant unless its Holm-adjusted p is below the "
    "significance threshold in the facts.\n\nFacts (JSON):\n"
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
    reason: str | None = None  # why the LLM was unavailable


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
        return Analysis("unavailable", llm_model, reason=getattr(llm, "reason", None))
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
        "reason": analysis.reason,
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
    return Analysis(data["status"], data["model"], narratives, data["facts"], data.get("reason"))
