from fakes import FakeLLM

from ucl.analyst import TEAM_HEADINGS, write_narrative
from ucl.llm import ChatResult, LLMError

FACTS = {"Shots per game": 17.4}
GOOD = "## How they got there\nThey averaged 17.4 shots.\n## Would the model have picked them?\nYes.\n## Weak spots\nFew."
UNGROUNDED_1 = GOOD.replace("Few.", "A 47% risk.")
UNGROUNDED_1B = GOOD.replace("Few.", "A 48% risk.")
UNGROUNDED_2 = GOOD.replace("Few.", "A 47% risk and 88 problems.")
INVALID = ChatResult("", "length")
MESSAGES = [{"role": "user", "content": "write"}]


def ok(text):
    return ChatResult(text, "stop")


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
