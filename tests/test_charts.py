from ucl import charts


def row(label, value, **extra):
    return {"label": label, "value": value, "color": "var(--s1)", "tip": f"{value}\n{label}", **extra}


def test_labels_and_tips_are_escaped():
    html = charts.bar_chart([row('<b>"x"</b>', 1.0, tip='a "quoted" <tip>')])
    assert "<b>" not in html and "&lt;b&gt;" in html
    assert 'data-tip="a &quot;quoted&quot; &lt;tip&gt;"' in html


def test_bar_widths_scale_to_the_largest_value():
    html = charts.bar_chart([row("a", 1.0), row("b", 0.5)])
    assert "--w:100.0%" in html and "--w:50.0%" in html
    assert "diverging" not in html


def test_values_below_the_baseline_diverge():
    html = charts.bar_chart([row("a", 0.6), row("b", 0.4)], baseline=0.5)
    assert 'class="bars diverging"' in html
    assert '<span class="half neg"><span class="bar"' in html


def test_dot_chart_positions_dot_base_rate_and_axis():
    rows = [{"label": "A", "sub": "s", "p": 0.25, "base": 0.125, "value_text": "25%", "tip": "t"}]
    html = charts.dot_chart(rows, extent=0.5)
    assert "--x:50.0%" in html and "--x:25.0%" in html
    assert ">0%<" in html and ">25%<" in html and ">50%<" in html


def test_stage_ladder_marks_reached_rungs_and_expected_stage():
    html = charts.stage_ladder(3, 1.5)
    assert html.count("reached") == 4 and html.count(" here") == 1
    assert "--x:40.0%" in html


def test_legend_mirrors_the_mark_shape():
    html = charts.legend([("Model", "var(--s1)", "dot"), ("Base rate", "var(--muted)", "tick")])
    assert 'class="swatch dot"' in html and 'class="swatch tick"' in html


def test_table_twin_marks_numeric_columns():
    html = charts.table_twin("Show", ["Name", "Value"], [["A", "1.0"]])
    assert '<td class="num">1.0</td>' in html and "<td>A</td>" in html
    assert html.startswith('<details class="twin">')
