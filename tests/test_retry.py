from fakes import FakeLLM

from ucl.analyst import BANNED_TEAM_WORDS, TEAM_HEADINGS, style_issues, word_count, write_narrative
from ucl.llm import ChatResult, LLMError

FACTS = {"Shots per game": 17.4}
GOOD = "## How they got there\nThey averaged 17.4 shots.\n## Would the model have picked them?\nYes.\n## Weak spots\nFew."
UNGROUNDED_1 = GOOD.replace("Few.", "A 47% risk.")
UNGROUNDED_1B = GOOD.replace("Few.", "A 48% risk.")
UNGROUNDED_2 = GOOD.replace("Few.", "A 47% risk and 88 problems.")
LOOSE = GOOD.replace("They averaged", "They significantly outshot rivals, averaging")
LOOSE_B = LOOSE.replace("rivals", "opponents")
LOOSE_AND_UNGROUNDED = LOOSE.replace("Few.", "A 47% risk.")
INVALID = ChatResult("", "length")
MESSAGES = [{"role": "user", "content": "write"}]
GOOD_WORDS = 6  # GOOD's words outside its headings, which are not counted


def ok(text):
    return ChatResult(text, "stop")


def with_words(total):
    """GOOD with filler added to make `total` counted words."""
    return GOOD + " pad" * (total - GOOD_WORDS)


LONG = with_words(262)
LONG_B = with_words(258)


def test_valid_grounded_answer_takes_one_call():
    n = write_narrative(FakeLLM([ok(GOOD)]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.unsupported) == ("ok", 1, [])


def test_invalid_then_valid_takes_two_distinct_calls():
    llm = FakeLLM([INVALID, ok(GOOD)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls) == ("ok", 2)
    assert llm.requests[0] != llm.requests[1]
    assert "cut off" in llm.requests[1][-1]["content"]


def test_validity_retry_carries_a_truncated_excerpt():
    llm = FakeLLM([ok("x" * 5000), ok(GOOD)])
    write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert len(llm.requests[1][-2]["content"]) == 1500


def test_rerun_is_served_entirely_from_cache():
    llm = FakeLLM([INVALID, ok(GOOD)])
    first = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    second = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert len(llm.requests) == 2
    assert first.text == second.text


def test_two_invalid_answers_make_the_narrative_unavailable():
    n = write_narrative(FakeLLM([INVALID, INVALID]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.text) == ("unavailable", 2, None)


def test_grounding_retry_fixes_numbers():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), ok(GOOD)]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.unsupported) == ("ok", 2, [])


def test_grounding_retry_with_as_many_problems_keeps_the_earlier_text():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), ok(UNGROUNDED_1B)]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.text, n.unsupported) == (UNGROUNDED_1, ["47"])


def test_never_more_than_three_calls_and_keep_best():
    llm = FakeLLM([INVALID, ok(UNGROUNDED_1), ok(UNGROUNDED_2)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert n.calls == 3 and len(llm.requests) == 3
    assert n.text == UNGROUNDED_1 and n.unsupported == ["47"]


def test_invalid_grounding_retry_keeps_the_earlier_text():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), INVALID]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.calls, n.text, n.unsupported) == ("ok", 2, UNGROUNDED_1, ["47"])


def test_timeout_on_grounding_retry_keeps_the_earlier_text():
    n = write_narrative(FakeLLM([ok(UNGROUNDED_1), LLMError("timeout")]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.status, n.text, n.unsupported) == ("ok", UNGROUNDED_1, ["47"])


def test_first_call_failure_is_unavailable():
    n = write_narrative(FakeLLM([LLMError("refused")]), "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert n.status == "unavailable" and "refused" in n.reason


def test_a_banned_word_triggers_exactly_one_retry_that_names_it():
    llm = FakeLLM([ok(LOOSE), ok(GOOD)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, banned=BANNED_TEAM_WORDS)
    assert n.calls == 2 and len(llm.requests) == 2
    last = llm.requests[1][-1]
    assert last["role"] == "user" and "significantly" in last["content"]


def test_a_clean_second_answer_replaces_one_with_a_banned_word():
    n = write_narrative(FakeLLM([ok(LOOSE), ok(GOOD)]), "k", MESSAGES, TEAM_HEADINGS, FACTS,
                        banned=BANNED_TEAM_WORDS)
    assert (n.status, n.text, n.style, n.unsupported) == ("ok", GOOD, [], [])


def test_a_second_answer_that_still_has_the_banned_word_keeps_the_first():
    n = write_narrative(FakeLLM([ok(LOOSE), ok(LOOSE_B)]), "k", MESSAGES, TEAM_HEADINGS, FACTS,
                        banned=BANNED_TEAM_WORDS)
    assert (n.text, n.style, n.calls) == (LOOSE, ["significantly"], 2)


def test_without_banned_words_the_word_is_not_retried():
    llm = FakeLLM([ok(LOOSE)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)  # the synthesis path passes no banned words
    assert (n.calls, n.text, n.style) == (1, LOOSE, []) and len(llm.requests) == 1


def test_one_retry_names_the_unsupported_numbers_and_the_banned_words_together():
    llm = FakeLLM([ok(LOOSE_AND_UNGROUNDED), ok(GOOD)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, banned=BANNED_TEAM_WORDS)
    assert (n.calls, n.text, n.unsupported, n.style) == (2, GOOD, [], [])
    feedback = llm.requests[1][-1]["content"]
    assert "47" in feedback and "significantly" in feedback


def test_the_second_answer_must_have_strictly_fewer_problems_of_both_kinds():
    # one banned word in the first answer, one unsupported number in the second: no better, so the first stays
    n = write_narrative(FakeLLM([ok(LOOSE), ok(UNGROUNDED_1)]), "k", MESSAGES, TEAM_HEADINGS, FACTS,
                        banned=BANNED_TEAM_WORDS)
    assert (n.text, n.unsupported, n.style) == (LOOSE, [], ["significantly"])
    # a number and a banned word, then only the banned word left: fewer in total, so the second is kept
    n = write_narrative(FakeLLM([ok(LOOSE_AND_UNGROUNDED), ok(LOOSE)]), "k", MESSAGES, TEAM_HEADINGS, FACTS,
                        banned=BANNED_TEAM_WORDS)
    assert (n.text, n.unsupported, n.style) == (LOOSE, [], ["significantly"])


def test_the_banned_word_retry_shares_the_three_call_cap():
    llm = FakeLLM([INVALID, ok(LOOSE), ok(LOOSE_B)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, banned=BANNED_TEAM_WORDS)
    assert n.calls == 3 and len(llm.requests) == 3
    assert (n.text, n.style) == (LOOSE, ["significantly"])


def test_style_issues_are_whole_words_found_ignoring_case_once_each():
    text = "Significantly ahead, and significantly so. Not insignificant; the significance is unproven."
    assert style_issues(text, BANNED_TEAM_WORDS) == ["significantly", "significance"]
    assert style_issues("A clear, statistically sound lead.", BANNED_TEAM_WORDS) == []
    assert style_issues(text, ()) == []


def test_word_count_skips_markdown_heading_lines_and_splits_on_any_whitespace():
    text = "## Heading with five words\nOne two three.\n\n  four   five\t six\n### Another heading\n#1 seed"
    assert word_count(text) == 8  # '#1 seed' is not a heading: a heading needs a space after its #s
    assert word_count("") == 0 and word_count(GOOD) == GOOD_WORDS


def test_style_issues_list_an_answer_over_the_word_limit_after_any_banned_words():
    text = "Significantly so.\n" + GOOD
    assert word_count(text) == GOOD_WORDS + 2
    assert style_issues(text, (), 8) == [] and style_issues(text, BANNED_TEAM_WORDS, 8) == ["significantly"]
    assert style_issues(text, (), 7) == ["over 7 words (8)"]
    assert style_issues(text, BANNED_TEAM_WORDS, 7) == ["significantly", "over 7 words (8)"]
    assert style_issues(text, BANNED_TEAM_WORDS) == ["significantly"]  # no limit given, so no length check


def test_an_over_length_answer_triggers_exactly_one_retry_that_names_the_limit():
    llm = FakeLLM([ok(LONG), ok(GOOD)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, max_words=250)
    assert n.calls == 2 and len(llm.requests) == 2
    last = llm.requests[1][-1]
    assert last["role"] == "user" and "250" in last["content"] and "262" in last["content"]  # limit and word count


def test_an_in_limit_second_answer_replaces_an_over_length_one():
    n = write_narrative(FakeLLM([ok(LONG), ok(with_words(250))]), "k", MESSAGES, TEAM_HEADINGS, FACTS, max_words=250)
    assert (n.status, n.text, n.style, n.unsupported, n.calls) == ("ok", with_words(250), [], [], 2)


def test_a_second_answer_that_is_still_over_length_keeps_the_first_and_lists_it():
    n = write_narrative(FakeLLM([ok(LONG), ok(LONG_B)]), "k", MESSAGES, TEAM_HEADINGS, FACTS, max_words=250)
    assert (n.text, n.style, n.calls) == (LONG, ["over 250 words (262)"], 2)  # shorter, but no fewer issues


def test_the_word_limit_is_inclusive_and_only_applies_when_given():
    llm = FakeLLM([ok(with_words(250))])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, max_words=250)
    assert (n.calls, n.style) == (1, []) and len(llm.requests) == 1
    llm = FakeLLM([ok(LONG)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS)
    assert (n.calls, n.text, n.style) == (1, LONG, []) and len(llm.requests) == 1


def test_the_length_retry_shares_the_three_call_cap():
    llm = FakeLLM([INVALID, ok(LONG), ok(LONG_B)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, max_words=250)
    assert n.calls == 3 and len(llm.requests) == 3
    assert (n.text, n.style) == (LONG, ["over 250 words (262)"])


def test_one_retry_covers_unsupported_numbers_banned_words_and_length_together():
    messy = LONG.replace("They averaged", "They significantly outshot rivals, averaging").replace("Few.", "A 47% risk.")
    llm = FakeLLM([ok(messy), ok(GOOD)])
    n = write_narrative(llm, "k", MESSAGES, TEAM_HEADINGS, FACTS, banned=BANNED_TEAM_WORDS, max_words=250)
    assert (n.calls, n.text, n.unsupported, n.style) == (2, GOOD, [], [])
    feedback = llm.requests[1][-1]["content"]
    assert "47" in feedback and "significantly" in feedback and "250" in feedback
    assert "over 250 words" not in feedback  # the length is not passed off as a banned word
