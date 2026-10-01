"""Render the report: an artifact-ready page fragment plus a standalone wrapper (spec §8)."""
from __future__ import annotations

import math
import re
from datetime import date
from html import escape

import pandas as pd

from . import charts, config
from .analyst import Analysis, Narrative
from .assets import CSS, JS
from .facts import DIRECTION, SIGNIFICANCE, beats, result_text, stat_label
from .features import display_value
from .model import ModelResults, feature_list

TITLE = "What Makes a Champions League Finalist?"
FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900"
    "&family=Public+Sans:ital,wght@0,400..700;1,400..700&display=swap"
)
GROUP_NAMES = {"pedigree": "Pedigree", "results": "Results", "style": "Style"}
SET_NAMES = {
    "pedigree": "Pedigree only",
    "results": "Results only",
    "style": "Style only",
    "all": "Everything",
    "all minus pedigree": "Everything but pedigree",
    "all minus results": "Everything but results",
    "all minus style": "Everything but style",
}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _span(bounds, fmt: str = "{:.2f}") -> str:
    """A 95% interval as 'a to b' (no dash, so a negative bound reads clearly)."""
    lo, hi = bounds
    return f"{fmt.format(lo)} to {fmt.format(hi)}"


def _flag_pattern(token: str) -> str:
    return rf"(?<![\w.]){re.escape(token)}(?!\.?\d)"


def _inline(text: str, flagged: list[str]) -> str:
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", out)
    for token in flagged:
        out = re.sub(_flag_pattern(token),
                     lambda m: f'<mark class="unverified" title="Not found in the data">{m.group(0)}</mark>', out)
    return out


def md_to_html(text: str, flagged=()) -> str:
    """The markdown subset the analyst writes: headings, paragraphs, bullet lists, bold, italic."""
    flagged = list(flagged)
    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(f"<p>{_inline(' '.join(paragraph), flagged)}</p>")
            paragraph.clear()
        if items:
            blocks.append("<ul>" + "".join(f"<li>{_inline(i, flagged)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in (text or "").splitlines():
        line = raw.strip()
        heading = re.match(r"^#{1,6}\s*(.+?)\s*#*$", line)
        bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if not line:
            flush()
        elif heading:
            flush()
            blocks.append(f"<h4>{_inline(heading.group(1), flagged)}</h4>")
        elif bullet:
            if paragraph:
                flush()
            items.append(bullet.group(1))
        else:
            if items:
                flush()
            paragraph.append(line)
    flush()
    return "\n".join(blocks)


def split_sections(text: str) -> dict[str, str]:
    """Markdown body under each heading, keyed by the lower-cased heading text."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in (text or "").splitlines():
        heading = re.match(r"^\s*#{1,6}\s*(.+?)\s*[:.]?\s*#*$", line)
        if heading:
            current = heading.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def badge(narrative: Narrative | None) -> str:
    if narrative is None or narrative.status != "ok":
        return '<span class="badge"><span class="icon" aria-hidden="true">–</span>AI write-up unavailable</span>'
    n = len(narrative.unsupported)
    if n == 0:
        return ('<span class="badge good"><span class="icon" aria-hidden="true">✓</span>'
                "All figures found in the data</span>")
    noun = "figure" if n == 1 else "figures"
    return (f'<span class="badge warn"><span class="icon" aria-hidden="true">!</span>'
            f"{n} {noun} not found in the data</span>")


def _story(narrative: Narrative | None) -> str:
    if narrative is None:
        return ""
    if narrative.status != "ok" or not narrative.text:
        return badge(None)
    return badge(narrative) + md_to_html(narrative.text, narrative.unsupported)


def _half(synthesis: Narrative | None, text: str | None) -> Narrative | None:
    """One half of the synthesis, carrying only the flags that occur in that half."""
    if synthesis is None or synthesis.status != "ok":
        return synthesis
    if not text:
        return Narrative(synthesis.key, "unavailable", reason="section missing")
    flagged = [t for t in synthesis.unsupported if re.search(_flag_pattern(t), text)]
    return Narrative(synthesis.key, "ok", text, flagged, synthesis.calls)


def _summary_block(narrative: Narrative | None) -> str:
    story = _story(narrative)
    return f'<div class="ai-summary"><p class="eyebrow">AI summary</p>{story}</div>' if story else ""


def _masthead(ai_ran: bool) -> str:
    first = config.season_label(config.SEASONS[0])
    last = config.season_label(config.SEASONS[-1])
    ai = " A local AI model then writes up what it found." if ai_ran else ""
    return (
        '<header class="masthead">'
        f'<p class="eyebrow">UEFA Champions League · {first} to {last}</p>'
        f"<h1>{escape(TITLE)}</h1>"
        '<p class="dek">The ten finalists of 2022–2026, measured against every knockout team since '
        f"{first}. A model learns from group and league-phase play and is tested on seasons it never saw.{ai}</p>"
        "</header>"
    )


def _ai_notice(analysis: Analysis) -> str:
    if analysis.status == "skipped":
        return ('<p class="notice">The AI write-ups were skipped for this run. Run <code>uv run ucl analyze</code> '
                "and then <code>uv run ucl report</code> to add them.</p>")
    if analysis.status == "unavailable":
        why = f" ({escape(analysis.reason)})" if getattr(analysis, "reason", None) else ""
        return (f'<p class="notice">LM Studio wasn\'t reachable{why}, so the AI write-ups are missing. Open LM Studio, '
                "then run <code>uv run ucl analyze</code> and <code>uv run ucl report</code>.</p>")
    return ""


def _finals(finals: pd.DataFrame) -> str:
    items = []
    recent = finals.loc[finals["season"].isin(config.TARGET_SEASONS)].sort_values("season", ascending=False)
    for f in recent.itertuples():
        winner = config.DISPLAY_NAMES.get(f.winner_id, f.winner)
        runner_up = config.DISPLAY_NAMES.get(f.runner_up_id, f.runner_up)
        pens = (f'<span class="pens">{int(f.winner_pens)}–{int(f.runner_up_pens)} on penalties</span>'
                if pd.notna(f.winner_pens) else "")
        items.append(
            f'<li><span class="season">{config.season_label(int(f.season))}</span>'
            f'<span class="tie"><strong>{escape(winner)}</strong>'
            f'<span class="score">{int(f.winner_goals)}–{int(f.runner_up_goals)}</span>'
            f"{escape(runner_up)}{pens}</span>"
            f'<span class="city">{escape(str(f.city))}</span></li>'
        )
    return (
        '<section aria-labelledby="finals-h"><h2 id="finals-h">The 10 finalists</h2>'
        "<p>Five finals, winner first on each line. Every side is one team in one season, so Real Madrid "
        "2021-22 and 2023-24 count as two finalists.</p>"
        f'<ol class="final-list">{"".join(items)}</ol></section>'
    )


def _picks(targets: pd.DataFrame, m: dict) -> str:
    extent = max(0.5, math.ceil(float(targets["p_final"].max()) * 10) / 10)
    rows, table_rows = [], []
    for t in targets.itertuples():
        season = config.season_label(int(t.season))
        result = "Winner" if int(t.ko_stage) == 4 else "Runner-up"
        rank = f"#{int(t.rank_in_season)} of {int(t.ko_size)}"
        rows.append({
            "label": t.team_display, "sub": f"{season} · {result} · expected {t.exp_stage:.2f}",
            "p": float(t.p_final), "base": float(t.base_rate), "value_text": f"{t.p_final:.0%} · {rank}",
            "tip": (f"{t.p_final:.1%} chance of reaching the final\n{t.team_display} {season}: {rank} knockout "
                    f"teams. Base rate {t.base_rate:.1%}. Expected stage {t.exp_stage:.2f} of 4."),
        })
        table_rows.append([t.team_display, season, result, f"{t.p_final:.1%}", rank,
                           f"{t.base_rate:.1%}", f"{t.exp_stage:.2f}"])
    picked = int((targets["rank_in_season"] <= 2).sum())
    legend = charts.legend([("Model's probability of reaching the final", "var(--s1)", "dot"),
                            ("Base rate for a random knockout team", "var(--muted)", "tick")])
    return (
        '<section aria-labelledby="picks-h"><h2 id="picks-h">Would the model have picked them?</h2>'
        f"<p>Each finalist was scored by a model trained on the other {m['n_seasons'] - 1} seasons, using only "
        "how the team played in the group or league phase and its pre-season club coefficient. Training "
        "includes later seasons, so this checks whether the pattern holds across eras. Two teams reach each "
        f"final, so a pick means ranking first or second: {picked} of the 10 finalists did.</p>"
        f'<div class="chart">{legend}{charts.dot_chart(rows, extent=extent)}</div>'
        + charts.table_twin("Show as a table", ["Finalist", "Season", "Result", "P(final)", "Rank", "Base rate",
                                                "Expected stage"], table_rows, numeric_from=3)
        + "</section>"
    )


def _ablation(ablation: pd.DataFrame) -> str:
    def chart(key: str, baseline: float, title: str, caption: str) -> str:
        rows = [{
            "label": SET_NAMES.get(r.feature_set, r.feature_set), "value": float(getattr(r, key)),
            "color": "var(--s1)",
            "tip": (f"{getattr(r, key):.2f} (95% interval {_span((getattr(r, key + '_lo'), getattr(r, key + '_hi')))})"
                    f"\n{SET_NAMES.get(r.feature_set, r.feature_set)}, {int(r.n_features)} stats"),
        } for r in ablation.itertuples()]
        return (f'<div class="chart"><h4>{escape(title)}</h4>'
                f"{charts.bar_chart(rows, baseline=baseline, fmt='{:.2f}', caption=caption)}</div>")

    table_rows = [[SET_NAMES.get(r.feature_set, r.feature_set), int(r.n_features),
                   f"{r.spearman:.2f}", _span((r.spearman_lo, r.spearman_hi)),
                   f"{r.auc:.2f}", _span((r.auc_lo, r.auc_hi))] for r in ablation.itertuples()]
    return (
        "<h3>Pedigree, results or style?</h3>"
        "<p>The same models retrained on subsets of the stats. Dropping a group shows what it adds that the "
        "others don't already carry. The 95% intervals overlap heavily, so treat the differences between "
        "groups as tentative.</p>"
        '<div class="pair">'
        + chart("spearman", 0.0, "Ranking teams within a season",
                "Spearman correlation between predicted and actual stage. Bars start at 0, a random ranking.")
        + chart("auc", 0.5, "Spotting the finalists", "AUC for reaching the final. Bars start at 0.5, a coin flip.")
        + "</div>"
        + charts.table_twin("Show as a table, with 95% intervals",
                            ["Stats used", "Count", "Spearman", "95% interval", "AUC", "95% interval"], table_rows)
    )


def _drivers(results: ModelResults, summary: Narrative | None, m: dict) -> str:
    rows, table_rows = [], []
    for r in results.drivers.itertuples():
        label, direction = stat_label(r.feature), DIRECTION[int(r.direction)]
        tag = r.label if isinstance(r.label, str) else ""
        rows.append({
            "label": label, "value": float(r.importance), "color": charts.GROUP_COLORS[r.group],
            "tag": f"{tag} · {direction}" if tag else "",
            "tip": (f"{r.importance:.3f} knockout stages on average\n{label} · {GROUP_NAMES[r.group]} · {direction}; "
                    f"on its own ρ = {r.marginal_rho:+.2f}"),
        })
        table_rows.append([int(r.rank), label, GROUP_NAMES[r.group], f"{r.importance:.3f}", direction,
                           f"{r.marginal_rho:+.2f}", tag, int(r.sign_agree_folds)])
    needed = math.ceil(config.ROBUST_SHARE * m["n_seasons"])
    legend = charts.legend([(GROUP_NAMES[g], color, "bar") for g, color in charts.GROUP_COLORS.items()])
    return (
        '<section aria-labelledby="drivers-h"><h2 id="drivers-h">What drives deep runs</h2>'
        "<p>Bar length is how far each stat moved the model's expected stage on average, in knockout rounds "
        "(mean absolute SHAP value, measured only on seasons the model didn't train on). The top "
        f"{config.TOP_DRIVERS} carry a label. <strong>Robust</strong>: a simpler logistic model agrees on the "
        f"direction in at least {needed} of {m['n_seasons']} seasons, and the stat points the same way on its own. "
        "<strong>Conditional</strong>: the direction holds only with the other stats held fixed; on its own the "
        "stat points the other way or barely at all. <strong>Model-dependent</strong>: the simpler model "
        "disagrees too often.</p>"
        f'<div class="chart">{legend}{charts.bar_chart(rows, fmt="{:.3f}")}</div>'
        + charts.table_twin("Show as a table", ["Rank", "Stat", "Group", "Importance", "Direction",
                                                "On its own (ρ)", "Label", "Seasons agreeing"],
                            table_rows, numeric_from=3)
        + _ablation(results.ablation)
        + _summary_block(summary)
        + "</section>"
    )


def _card(row: pd.Series, pred: pd.Series, shap_row: pd.Series, final: pd.Series,
          narrative: Narrative | None, ai_ran: bool, features: list[str]) -> str:
    season = config.season_label(int(row["season"]))
    contributions = {f: float(shap_row[f"shap_{f}"]) for f in features}
    top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:7]
    top.sort(key=lambda kv: kv[1], reverse=True)
    bars = [{
        "label": stat_label(f), "value": c, "color": "var(--s1)" if c >= 0 else "var(--neg)",
        "tip": (f"{c:+.2f} knockout stages\n{stat_label(f)}: {display_value(row, f)}, beat "
                f"{beats(row, f, 'season')}% of that season's teams"),
    } for f, c in top]
    legend = charts.legend([("Pushed the prediction up", "var(--s1)", "bar"),
                            ("Pushed it down", "var(--neg)", "bar")])
    stats = [[stat_label(f), display_value(row, f), beats(row, f, "season"), beats(row, f, "all"),
              f"{contributions[f]:+.2f}"] for f in features]
    expected = float(pred["exp_stage"])
    note = (f"▲ marks the model's expected stage, {expected:.2f} of 4. It gave a {float(pred['p_final']):.0%} "
            f"chance of reaching the final, #{int(pred['rank_in_season'])} of {int(pred['ko_size'])} knockout teams.")
    if ai_ran:
        story = _story(narrative) or badge(None)
    else:
        story = '<p class="ai-missing">No AI write-up for this run.</p>'
    return (
        f'<article class="card" id="team-{int(row["season"])}-{escape(str(row["team_id"]))}">'
        f'<div class="card-head"><h3>{escape(str(row["team_display"]))}<span>{season}</span></h3>'
        f"<p>{escape(result_text(row['team_id'], final))}</p></div>"
        f'{charts.stage_ladder(int(pred["ko_stage"]), expected)}<p class="ladder-note">{escape(note)}</p>'
        f'<div class="narrative">{story}</div>'
        f'<div class="chart"><h4>What moved the prediction</h4>{legend}{charts.bar_chart(bars, fmt="{:+.2f}")}</div>'
        + charts.table_twin(f"All {len(features)} stats",
                            ["Stat", "Value", "Teams beaten that season (%)", "Teams beaten since 2011-12 (%)",
                             "Moved the prediction"], stats)
        + "</article>"
    )


def _cards(team_seasons, finals, results: ModelResults, analysis: Analysis, targets: pd.DataFrame) -> str:
    features = feature_list(results)
    rows = team_seasons.set_index(["season", "team_id"], drop=False)
    preds = results.predictions.set_index(["season", "team_id"])
    shap_rows = results.shap.set_index(["season", "team_id"])
    finals_by_season = finals.set_index("season")
    ai_ran = analysis.status == "ok"
    cards = []
    for t in targets.itertuples():
        key = (t.season, t.team_id)
        cards.append(_card(rows.loc[key], preds.loc[key], shap_rows.loc[key], finals_by_season.loc[t.season],
                           analysis.narratives.get(f"{int(t.season)}-{t.team_id}"), ai_ran, features))
    ai = ", with the AI scouting report" if ai_ran else ""
    return ('<section aria-labelledby="teams-h"><h2 id="teams-h">The finalists, one by one</h2>'
            "<p>Where each side finished, what the model expected, and the stats that moved its prediction "
            f'most{ai}. "Teams beaten" is the share of teams a side did better than, so higher is always better, '
            "including on the stats where a lower number is better.</p>"
            f'<div class="cards">{"".join(cards)}</div></section>')


def _winners(results: ModelResults, summary: Narrative | None, n_finals: int) -> str:
    fc = results.finals_compare.sort_values("mean_diff", ascending=False, kind="stable")
    rows, table_rows = [], []
    for r in fc.itertuples():
        label = stat_label(r.feature)
        rows.append({
            "label": label, "value": float(r.mean_diff), "color": "var(--s1)",
            "note": f"{int(r.higher)}–{int(r.lower)}–{int(r.tied)} · p {r.p_holm:.2f}",
            "tip": (f"{r.mean_diff:+.2f} standard deviations on average\n{label}: the winner was higher in "
                    f"{int(r.higher)} of {n_finals} finals, lower in {int(r.lower)}. Holm-adjusted p {r.p_holm:.2f}."),
        })
        table_rows.append([label, f"{r.mean_diff:+.2f}", int(r.higher), int(r.lower), int(r.tied),
                           f"{r.p:.3f}", f"{r.p_holm:.3f}",
                           *[f"{getattr(r, f'diff_{s}'):+.2f}" for s in config.TARGET_SEASONS]])
    significant = int((fc["p_holm"] < SIGNIFICANCE).sum())
    verdict = (f"No difference survives a Holm correction at p < {SIGNIFICANCE}." if significant == 0
               else f"{significant} of {len(fc)} differences survive a Holm correction at p < {SIGNIFICANCE}.")
    return (
        '<section aria-labelledby="winners-h"><h2 id="winners-h">Winners vs runners-up</h2>'
        f"<p>For each of the {n_finals} finals since 2011-12: the winner's group or league-phase stats minus the "
        f"runner-up's, in within-season standard deviations. Under each value: the finals in which the winner was "
        f"higher–lower–tied, and the Holm-adjusted p across all {len(fc)} stats. {verdict} With so few finals, "
        "read these as hints at most.</p>"
        f'<div class="chart">{charts.bar_chart(rows, fmt="{:+.2f}")}</div>'
        + charts.table_twin("Show as a table, including the five recent finals",
                            ["Stat", "Mean difference", "Winner higher", "Lower", "Tied", "p", "Holm p",
                             *[config.season_label(s) for s in config.TARGET_SEASONS]], table_rows)
        + _summary_block(summary)
        + "</section>"
    )


def _method(team_seasons: pd.DataFrame, finals: pd.DataFrame, results: ModelResults, analysis: Analysis,
            generated: date) -> str:
    m = results.metrics
    ci = m.get("ci", {})
    first = config.season_label(config.SEASONS[0])
    last = config.season_label(config.SEASONS[-1])
    counts = finals["winner_id"].value_counts()
    top_id, top_wins = counts.idxmax(), int(counts.max())
    top_name = config.DISPLAY_NAMES.get(top_id, finals.loc[finals["winner_id"] == top_id, "winner"].iloc[0])
    skill_lo, skill_hi = ci.get("brier_skill", (math.nan, math.nan))
    skill_verdict = ("includes zero, so that edge is inconclusive" if skill_lo <= 0 <= skill_hi
                     else "lies entirely above zero")
    method = [
        f"Data: UEFA's public match, team-statistics and club-coefficient feeds for {m['n_seasons']} seasons "
        f"({first} to {last}). That covers {len(team_seasons)} group or league-phase team-seasons, of which "
        f"{m['n_knockout']} reached the knockouts.",
        f"Inputs: {len(feature_list(results))} per-game stats from group and league-phase matches only, "
        "standardised within each season, plus the club's five-year UEFA coefficient from before the season. "
        "Save rate was dropped because the 2011-12 feed lacks saves for too many matches.",
        f"Validation: each season is predicted by models trained on the other {m['n_seasons'] - 1}. The ranking "
        f"correlation averages {m['spearman_mean']:.2f} (95% interval {_span(ci.get('spearman_mean', (0, 0)))}), "
        f"and the AUC for reaching the final is {m['auc']:.2f} ({_span(ci.get('auc', (0, 0)))}). "
        f"{m['finalists_in_top4']:.0%} of finalists were in their season's predicted top four "
        f"({_span(ci.get('finalists_in_top4', (0, 0)), '{:.0%}')}), against "
        f"{m['finalists_in_top4_chance']:.0%} for a random ranking. Against always guessing the base rate, the "
        f"probabilities have a Brier skill of {m['brier_skill']:.2f}; its interval ({_span((skill_lo, skill_hi))}) "
        f"{skill_verdict}. Intervals come from resampling seasons.",
        "Probabilities of reaching the final are scaled within each season to add up to two, because "
        "exactly two teams get there.",
    ]
    if analysis.status == "ok":
        method.append(
            f"AI write-ups: {analysis.model} running locally in LM Studio, given only the numbers behind this "
            "page. Every figure it writes is checked against those numbers. The check confirms that the figure "
            "appears in the data, not that it is attached to the right stat, and the wording itself is not checked."
        )
    caveats = [
        "Correlation is not causation, and group or league-phase stats depend on the opponents drawn.",
        "Knockout football is high-variance. The 2026 final was decided on penalties.",
        f"{top_name} won {top_wins} of the {len(finals)} finals, so one club's profile weighs heavily on what "
        "winning looks like.",
        "In 2024-25 and 2025-26 a top-eight league finish skips the play-off, which builds in an advantage "
        "for results.",
        "UEFA publishes no expected-goals data for these seasons, and the shot counts leave out blocked shots.",
        "UEFA's feed reported possession in seconds and distance in metres for some 2014-16 matches. These were "
        "converted using the feed's own figures, and partial tracking and placeholder zeros were ignored.",
        "Some stat definitions changed between seasons, so comparisons with all teams since 2011-12 mix them. "
        "The model itself only compares teams within a season.",
        f"{m['n_seasons']} seasons contain only {m['n_finalists']} finalists, so the probabilities are rough.",
        "Models are trained on seasons after the one being scored as well as before it.",
        "UEFA's APIs are undocumented. Their values are used as published.",
        "The AI text is limited to the numbers and checked against them, but its interpretations are not "
        "causal evidence.",
    ]
    items = "".join(f"<li>{escape(x)}</li>" for x in method)
    risks = "".join(f"<li>{escape(x)}</li>" for x in caveats)
    source = f" and a local model ({escape(analysis.model)})" if analysis.status == "ok" else ""
    return (
        '<section class="method" aria-labelledby="method-h"><h2 id="method-h">Method and caveats</h2>'
        f"<ul>{items}</ul><h3>Caveats</h3><ul>{risks}</ul></section>"
        f"<footer>Generated {generated.isoformat()} from UEFA data{source}.</footer>"
    )


def render(team_seasons: pd.DataFrame, finals: pd.DataFrame, results: ModelResults, analysis: Analysis,
           generated: date | None = None) -> str:
    """The artifact-ready page fragment: <title> first, then styles, content and the tooltip script."""
    m = results.metrics
    preds = results.predictions
    targets = preds.loc[preds["is_target"].astype(bool)].sort_values(["season", "ko_stage"], ascending=[False, False])
    synthesis = analysis.narratives.get("synthesis")
    parts = split_sections(synthesis.text) if synthesis and synthesis.status == "ok" else {}
    body = "".join([
        _masthead(analysis.status == "ok"),
        _ai_notice(analysis),
        _finals(finals),
        _picks(targets, m),
        _drivers(results, _half(synthesis, parts.get("why the best teams win")), m),
        _cards(team_seasons, finals, results, analysis, targets),
        _winners(results, _half(synthesis, parts.get("winners vs runners-up")), len(finals)),
        _method(team_seasons, finals, results, analysis, generated or date.today()),
    ])
    return (
        f"<title>{escape(TITLE)}</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link rel="stylesheet" href="{escape(FONTS_URL)}">\n'
        f"<style>{CSS}</style>\n"
        f'<main class="page">{body}</main>\n'
        '<div id="tip" class="tip" role="tooltip" hidden></div>\n'
        f"<script>{JS}</script>\n"
    )


def standalone(fragment: str) -> str:
    """Wrap the fragment as a full document for opening out/report.html directly."""
    head, _, body = fragment.partition('<main class="page">')
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'{head}</head>\n<body>\n<main class="page">{body}</body>\n</html>\n'
    )
