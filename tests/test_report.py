import re
from datetime import date

from ucl import assets, report
from ucl.analyst import Analysis, Narrative

TEAM_TEXT = ("## How they got there\nThey had **47%** possession.\n"
             "## Would the model have picked them?\nYes.\n## Weak spots\n- Few\n- None")
SYNTH_TEXT = "## Why the best teams win\nResults matter.\n## Winners vs runners-up\nNothing significant at 47."


def analysis_for(results, unsupported=(), synth_unsupported=(), failed_key=None):
    preds = results.predictions
    keys = [f"{int(r.season)}-{r.team_id}" for r in preds[preds["is_target"]].itertuples()]
    narratives = {k: Narrative(k, "ok", TEAM_TEXT, list(unsupported), 1) for k in keys}
    if failed_key is not None:
        narratives[keys[failed_key]] = Narrative(keys[failed_key], "unavailable", reason="timeout")
    narratives["synthesis"] = Narrative("synthesis", "ok", SYNTH_TEXT, list(synth_unsupported), 1)
    return Analysis("ok", "test/model", narratives, {})


def render(built, analysis, **kwargs):
    ds, res = built
    return report.render(ds.team_seasons, ds.finals, res, analysis, generated=date(2026, 10, 1), **kwargs)


def test_page_has_every_section(built):
    page = render(built, analysis_for(built[1]))
    for heading in ["The 10 finalists", "Would the model have picked them?", "What drives deep runs",
                    "Pedigree, results or style?", "Winners vs runners-up", "Method and caveats"]:
        assert heading in page
    assert page.count('<article class="card"') == 10
    assert "Results matter." in page and "Nothing significant at" in page
    assert "Generated 2026-10-01" in page


def test_grounded_narratives_get_the_good_badge(built):
    page = render(built, analysis_for(built[1]))
    assert page.count("All figures found in the data") == 12  # 10 cards + 2 synthesis halves


def test_unsupported_numbers_are_highlighted_and_badged(built):
    page = render(built, analysis_for(built[1], unsupported=["47"]))
    assert '<mark class="unverified" title="Not found in the data">47</mark>' in page
    assert "1 figure not found in the data" in page


def test_synthesis_flags_are_counted_per_half(built):
    page = render(built, analysis_for(built[1], synth_unsupported=["47"]))
    drivers, winners = page.split('id="winners-h"')
    assert "1 figure not found in the data" not in drivers.split('id="drivers-h"')[1]
    assert "1 figure not found in the data" in winners


def test_a_failed_narrative_shows_the_unavailable_badge(built):
    page = render(built, analysis_for(built[1], failed_key=0))
    assert page.count("AI write-up unavailable") == 1


def test_skipped_and_unavailable_runs_show_notices_without_ai_claims(built):
    for status, notice in [("skipped", "AI write-ups were skipped"), ("unavailable", "LM Studio wasn’t reachable")]:
        page = render(built, Analysis(status, "test/model"))
        assert notice in page
        assert "All figures found in the data" not in page
        assert "running locally in LM Studio" not in page
        assert page.count("No AI write-up for this run.") == 10


def test_unavailable_notice_gives_the_reason(built):
    page = render(built, Analysis("unavailable", "test/model", reason="lms not found"))
    assert "(lms not found)" in page


def test_method_reports_intervals_and_chance(built):
    page = render(built, analysis_for(built[1]))
    assert "95% interval" in page and "for a random ranking" in page and "Brier skill" in page


def test_penalty_finals_are_shown(built):
    ds, res = built
    finals = ds.finals.copy()
    finals.loc[finals["season"] == 2026, ["winner_pens", "runner_up_pens"]] = [4, 3]
    page = report.render(ds.team_seasons, finals, res, analysis_for(res), generated=date(2026, 10, 1))
    assert "4–3 on penalties" in page


def test_markdown_subset_is_escaped():
    html = report.md_to_html("## Head\nPara with **bold** and <script>x</script>\n\n- a\n- b")
    assert "<h4>Head</h4>" in html and "<strong>bold</strong>" in html
    assert "<ul><li>a</li><li>b</li></ul>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_split_sections_keys_by_heading():
    parts = report.split_sections(SYNTH_TEXT)
    assert parts == {"why the best teams win": "Results matter.",
                     "winners vs runners-up": "Nothing significant at 47."}


def test_fragment_is_artifact_ready_and_standalone_is_a_document(built):
    page = render(built, analysis_for(built[1]))
    assert page.startswith("<title>")
    lowered = page.lower()
    assert "<html" not in lowered and "<body" not in lowered and "<!doctype" not in lowered
    doc = report.standalone(page)
    assert doc.startswith("<!doctype html>")
    assert doc.index("<title>") < doc.index("</head>") < doc.index('<main class="page">')


def test_css_covers_both_themes_and_scopes_the_marks():
    css = assets.CSS
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root[data-theme="dark"]' in css and ':root:not([data-theme="light"])' in css
    # mark rules must not leak onto legend swatches ("swatch dot", "swatch tick", "swatch bar")
    assert not re.search(r"(?m)^\.(dot|tick|bar)\s*\{", css)
    assert ".twin { min-width: 0;" in css


def test_ordinal_suffixes():
    assert [report.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 95, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "95th", "100th"]
