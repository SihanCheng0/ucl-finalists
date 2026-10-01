from ucl.grounding import normalize_headings, strip_think, validate_output

HEADINGS = ["How they got there", "Weak spots"]
GOOD = "## How they got there\nText.\n\n## Weak spots\nMore text."


def test_good_answer_is_valid():
    assert validate_output(GOOD, "stop", HEADINGS) is None


def test_length_cutoff_is_invalid():
    assert "cut off" in validate_output(GOOD, "length", HEADINGS)


def test_empty_answer_is_invalid():
    assert "empty" in validate_output("   ", "stop", HEADINGS)


def test_unclosed_think_is_invalid():
    assert "unclosed" in validate_output("<think>still thinking about " + GOOD, "stop", HEADINGS)


def test_closed_and_stray_think_tags_are_stripped():
    assert strip_think("<think>plan</think>\n" + GOOD) == GOOD
    assert strip_think("leaked reasoning</think>\n" + GOOD) == GOOD
    assert validate_output("<think>plan</think>\n" + GOOD, "stop", HEADINGS) is None


def test_missing_heading_is_invalid():
    reason = validate_output("## How they got there\nText only.", "stop", HEADINGS)
    assert "Weak spots" in reason


def test_heading_variants_are_accepted_and_normalised():
    text = "## how they got there:\nx\n**2. Weak spots**\ny"
    assert validate_output(text, "stop", HEADINGS) is None
    assert normalize_headings(text, HEADINGS) == "## How they got there\nx\n## Weak spots\ny"


def test_crlf_answers_are_valid():
    assert validate_output("## How they got there\r\nx\r\n## Weak spots\r\ny", "stop", HEADINGS) is None
