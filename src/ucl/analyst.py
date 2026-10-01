"""Prompts, the narrative retry policy and persistence for the local-LLM analyst (spec §7)."""
from __future__ import annotations

import json
import re
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
# The team facts hold no significance test, so these words in a team narrative are always unsupported.
BANNED_TEAM_WORDS = ("significant", "significantly", "significance")
# The most words the instructions ask for; a longer answer is a style issue, so it gets the one style retry.
TEAM_MAX_WORDS, SYNTH_MAX_WORDS = 250, 450
HEADING_LINE = re.compile(r"\s{0,3}#{1,6}(?:\s|$)")  # a markdown heading: 1-6 #s, then a space or the line's end
OVER_LENGTH = re.compile(r"over (\d+) words \((\d+)\)")  # how style_issues lists an answer that runs long
# What a team narrative may not call its season's first phase: the other format's name, or a hedge between the two.
# The local model took 'group/league phase' from the fact sheets and either guessed or hedged.
FORMAT_NAME = {True: "league phase", False: "group stage"}  # by whether the season used the league format
HEDGE = r"group\s*/\s*league|league\s*/\s*group|group\s+or\s+league|league\s+or\s+group"
WRONG_NAME = {True: r"group[\s-]+(?:stage|phase)|groups", False: r"league[\s-]+phase"}
# how format_issues lists a wrong name, so that _feedback can tell it from a banned word
WRONG_FORMAT = re.compile(r"says '.+' but this season had a (?:league phase|group stage)")
# A final placed inside the first phase. The local model read the result and the format facts together and wrote
# "reached the final of the league phase" or "a 0-1 decider during the group stage".
MISPLACED_FINAL = re.compile(
    r"\b(?:final|decider)\b[^.;:!?]{0,40}?\b(?:of|in|during)\s+the\s+(?:\d{4}-\d{2}\s+)?"
    r"(?:group[\s-]+stage|league[\s-]+phase)\b",
    re.IGNORECASE,
)
FINAL_ISSUE = re.compile(r"puts the final in the first phase \('(.+)'\)")  # how final_issues lists a hit

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
STYLE_FEEDBACK = (
    "Don't use these words, which imply a statistical test the facts don't contain: {words}. Rewrite the "
    "complete answer with the same headings, without them."
)
FORMAT_FEEDBACK = (
    "Your answer names the first phase wrongly: {issues}. Rewrite the complete answer with the same headings, "
    "calling the first phase only by the name given for 'First-phase format that season' and never hedging "
    "between the two names."
)
FINAL_FEEDBACK = (
    "Your answer places the final inside the first phase: {issues}. The final is the Champions League final, "
    "played after the knockout rounds; the first phase (league phase or group stage) comes before them. Rewrite "
    "the complete answer with the same headings, saying the team reached, won or lost the Champions League final, "
    "and describing the first phase separately."
)
LENGTH_FEEDBACK = (
    "Your answer is {count} words, not counting the headings, over the {limit}-word limit. Rewrite the complete "
    "answer with the same headings, shorter and within {limit} words."
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
    "do not mention SHAP. For each weak spot give the stat's value and say it beat N% of that season's teams, "
    "using exactly that phrase.\n"
    "Call the first phase exactly as 'First-phase format that season' names it (league phase or group stage).\n\n"
    "Facts (JSON):\n"
)
SYNTH_INSTRUCTIONS = (
    "Write 350-450 words of markdown in total, with exactly two paragraphs of about 80 words under each of "
    "exactly these two headings, in this order:\n"
    "## Why the best teams win\n"
    "## Winners vs runners-up\n\n"
    "Under 'Why the best teams win', the first paragraph says how well the model does: its Brier score against "
    "the base-rate Brier score, and what share of the finalists were in its top 4 against the share a random "
    "ranking would give; give the Spearman, AUC, Brier skill and top-4 share each with its 95% interval. For the "
    "Brier skill and the top-4 share the facts say whether the interval includes its no-skill value: where it "
    "does, say the model's edge on that measure is inconclusive; where it does not, say which side of that value "
    "the whole interval lies on. Describe the model's edge only that way, never as significant or superior. The "
    "model learns from the knockout team-seasons, using each team's group/league-phase stats: give the knockout "
    "count, not the group/league-phase count, as what it learns from. The second paragraph says which stats are "
    "associated with deep knockout runs and whether pedigree, results or style matters most, using the "
    "feature-set comparison in a sentence without quoting its intervals: they overlap, so the differences between "
    "pedigree, results and style are tentative (do not use 'significant' or 'significantly' for them: nothing "
    "about the feature sets was tested). 'Importance' is the average size of a stat's push on the expected stage, "
    "in either direction. Only call a stat 'robust' if its robustness label is robust. A 'conditional' stat helps "
    "only alongside the other stats (on its own it points the other way or barely at all) and a 'model-dependent' "
    "stat is unstable: describe each that way.\n"
    "Under 'Winners vs runners-up', which compares those same group/league-phase stats, not what happened in the "
    "final, discuss only the first four stats in that comparison (it is sorted from smallest to largest "
    "Holm-adjusted p) and no other stat from it: the first paragraph gives each one's mean difference and "
    "Holm-adjusted p, the second says in how many finals the winner was higher and in how many lower, and whether "
    "any difference was significant. Never call a difference significant unless its Holm-adjusted p is below the "
    "significance threshold in the facts.\n"
    "Report findings only, with no summary or implication, and do not explain how SHAP or the model works.\n\n"
    "Facts (JSON):\n"
)


@dataclass
class Narrative:
    key: str
    status: str  # "ok" | "unavailable"
    text: str | None = None
    unsupported: list[str] = field(default_factory=list)
    calls: int = 0
    reason: str | None = None
    # banned words, wrong first-phase names, finals placed in the first phase and 'over N words (count)' left in the
    # text; defaulted so old files load
    style: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    status: str  # "ok" | "unavailable" | "skipped"
    model: str
    narratives: dict[str, Narrative] = field(default_factory=dict)
    facts: dict[str, dict] = field(default_factory=dict)
    reason: str | None = None  # why the LLM was unavailable


def word_count(text: str) -> int:
    """Whitespace-separated words, not counting markdown heading lines."""
    return sum(len(line.split()) for line in text.splitlines() if not HEADING_LINE.match(line))


def format_issues(text: str, league_format: bool) -> list[str]:
    """The names in `text` that don't fit its season's first phase (whole words, ignoring case, each once in order
    of first appearance): the other format's name and any 'group or league' hedge, each worded like
    "says 'group stage' but this season had a league phase"."""
    pattern = re.compile(rf"\b(?:{HEDGE}|{WRONG_NAME[league_format]})\b", re.IGNORECASE)
    said = dict.fromkeys(" ".join(match.group().lower().split()) for match in pattern.finditer(text))
    return [f"says '{name}' but this season had a {FORMAT_NAME[league_format]}" for name in said]


def final_issues(text: str) -> list[str]:
    """The phrases in `text` that put the final inside the first phase (ignoring case, each once in order of first
    appearance), each worded like "puts the final in the first phase ('final of the league phase')"."""
    said = dict.fromkeys(" ".join(match.group().lower().split()) for match in MISPLACED_FINAL.finditer(text))
    return [f"puts the final in the first phase ('{phrase}')" for phrase in said]


def style_issues(text: str, banned: tuple[str, ...], max_words: int | None = None,
                 league_format: bool | None = None) -> list[str]:
    """The banned words in `text` (whole words only, ignoring case, each once in order of first appearance), then,
    when `league_format` is given, its wrong first-phase names and any final placed inside the first phase, then
    'over N words (count)' when it has more than `max_words` words."""
    issues: list[str] = []
    if banned:
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(word) for word in banned) + r")\b", re.IGNORECASE)
        issues = list(dict.fromkeys(match.group().lower() for match in pattern.finditer(text)))
    if league_format is not None:
        issues += format_issues(text, league_format) + final_issues(text)
    if max_words is not None and (count := word_count(text)) > max_words:
        issues.append(f"over {max_words} words ({count})")
    return issues


def _feedback(unsupported: list[str], style: list[str]) -> str:
    """What the one retry tells the model is wrong with its answer: numbers, wording, first-phase names, a final
    placed inside the first phase, length or a mix."""
    names = [issue for issue in style if WRONG_FORMAT.fullmatch(issue)]
    finals = [match for match in map(FINAL_ISSUE.fullmatch, style) if match]
    words = [issue for issue in style
             if issue not in names and not FINAL_ISSUE.fullmatch(issue) and not OVER_LENGTH.fullmatch(issue)]
    over = next((match for match in map(OVER_LENGTH.fullmatch, style) if match), None)
    parts = []
    if unsupported:
        parts.append(GROUNDING_FEEDBACK.format(numbers=", ".join(unsupported)))
    if words:
        parts.append(STYLE_FEEDBACK.format(words=", ".join(words)))
    if names:
        parts.append(FORMAT_FEEDBACK.format(issues="; ".join(names)))
    if finals:
        parts.append(FINAL_FEEDBACK.format(issues="; ".join(f"'{match[1]}'" for match in finals)))
    if over:
        parts.append(LENGTH_FEEDBACK.format(limit=over[1], count=over[2]))
    return " ".join(parts)


def write_narrative(llm, key: str, messages: list[dict], headings: list[str], facts: dict,
                    banned: tuple[str, ...] = (), max_words: int | None = None,
                    league_format: bool | None = None) -> Narrative:
    """At most 3 calls: initial, one validity retry, one retry for unsupported numbers, banned words, a wrong
    first-phase name or a final placed inside it (both checked when `league_format` is given) or an answer over
    `max_words` (spec §7)."""
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
    unsupported, style = check_grounding(text, facts), style_issues(text, banned, max_words, league_format)
    if unsupported or style:
        retry = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": _feedback(unsupported, style)},
        ]
        try:
            second = llm.chat(retry)
            calls += 1
            if validate_output(second.content, second.finish_reason, headings) is None:
                second_text = normalize_headings(strip_think(second.content), headings)
                second_unsupported = check_grounding(second_text, facts)
                second_style = style_issues(second_text, banned, max_words, league_format)
                if len(second_unsupported) + len(second_style) < len(unsupported) + len(style):
                    text, unsupported, style = second_text, second_unsupported, second_style
        except LLMError:
            pass  # keep the earlier valid text with its flags
    return Narrative(key, "ok", text=text, unsupported=unsupported, calls=calls, style=style)


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
        # team keys are '{season}-{team_id}'; the synthesis spans both formats, so it is not held to either name
        league_format = None if synth else config.is_league_format(int(key.split("-", 1)[0]))
        narrative = write_narrative(llm, key, messages, SYNTH_HEADINGS if synth else TEAM_HEADINGS, sheet,
                                    banned=() if synth else BANNED_TEAM_WORDS,
                                    max_words=SYNTH_MAX_WORDS if synth else TEAM_MAX_WORDS,
                                    league_format=league_format)
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
