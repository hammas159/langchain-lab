"""Tests for project 05.

The corpus tests pin the instrument. Tie pairs only measure bias if the two answers really are
of equal quality and really do differ in length — if a "tie" pair had a better answer in it,
the preference the judge shows would be judgement rather than bias, and the headline number
would mean the opposite of what it claims.

The `BothOrders` tests pin the distinction the whole project rests on: choosing the same
*answer* in both orders versus choosing the same *slot*.
"""

from __future__ import annotations

import pytest

from projects.p05_judge_bias.answers import (
    BY_ID,
    CORPUS,
    length_ratio,
    quality_pairs,
    tie_pairs,
)
from projects.p05_judge_bias.benchmark import tally
from projects.p05_judge_bias.judge import (
    PROMPTS,
    RUBRIC,
    TIE_ALLOWED,
    BothOrders,
    Verdict,
    parse,
)
from shared.llm import server_is_up

# --- the corpus ----------------------------------------------------------------------------


def test_question_ids_are_unique():
    assert len({q.id for q in CORPUS}) == len(CORPUS)


def test_every_question_has_all_three_variants():
    for q in CORPUS:
        assert q.concise.variant == "concise"
        assert q.verbose.variant == "verbose"
        assert q.wrong.variant == "wrong"


def test_every_wrong_answer_documents_its_error():
    """A `wrong` variant with no stated error cannot be checked by a reader."""
    for q in CORPUS:
        assert len(q.error) > 30, q.id


def test_verbose_is_substantially_longer_than_concise():
    """If the length difference were small, a length effect would be unmeasurable."""
    for q in CORPUS:
        assert length_ratio(q) >= 2.0, f"{q.id} ratio {length_ratio(q):.1f}"


def test_the_wrong_answer_is_not_the_longest():
    """Otherwise 'picks the wrong answer' and 'picks the longest' are the same behaviour."""
    for q in CORPUS:
        assert q.wrong.words < q.verbose.words, q.id


def test_concise_and_wrong_are_similar_in_length():
    """Quality pairs must not be decidable on length alone."""
    for q in CORPUS:
        ratio = max(q.concise.words, q.wrong.words) / min(q.concise.words, q.wrong.words)
        assert ratio < 1.75, f"{q.id} ratio {ratio:.2f}"


def test_tie_pairs_use_the_two_correct_variants():
    for _q, a, b in tie_pairs():
        assert {a.variant, b.variant} == {"concise", "verbose"}


def test_quality_pairs_contain_exactly_one_wrong_answer():
    for _q, a, b in quality_pairs():
        assert [a.variant, b.variant].count("wrong") == 1


def test_there_is_one_pair_of_each_type_per_question():
    assert len(tie_pairs()) == len(quality_pairs()) == len(CORPUS)


# --- parsing -------------------------------------------------------------------------------


def test_parses_a_bare_letter():
    assert parse("A") == "A"
    assert parse("B") == "B"


def test_parses_a_tie():
    assert parse("TIE") == "TIE"
    assert parse("tie") == "TIE"


def test_takes_the_first_verdict_in_a_sentence():
    """ "A is better than B" is a vote for A, not for B."""
    assert parse("A is better than B") == "A"


def test_returns_empty_when_there_is_no_verdict():
    assert parse("I cannot decide between these.") == ""


def test_does_not_match_a_letter_inside_a_word():
    assert parse("Both are adequate") == ""


# --- the judge prompts ------------------------------------------------------------------------


def test_all_three_prompts_are_registered():
    assert set(PROMPTS) == {"plain", "tie_allowed", "rubric"}


def test_the_plain_prompt_offers_no_tie():
    """A forced choice on equal answers must record a preference that does not exist."""
    assert "TIE" not in PROMPTS["plain"]


def test_the_other_prompts_offer_a_tie():
    assert "TIE" in TIE_ALLOWED
    assert "TIE" in RUBRIC


def test_only_the_rubric_rules_out_length():
    assert "Length is not a criterion" in RUBRIC
    assert "Length is not a criterion" not in PROMPTS["plain"]


# --- consistency, and the difference between an answer and a slot -------------------------------


def _both(
    forward_raw: str, forward_chose: str, reversed_raw: str, reversed_chose: str
) -> BothOrders:
    q = BY_ID["q1"]
    return BothOrders(
        question=q,
        a=q.concise,
        b=q.verbose,
        forward=Verdict(raw=forward_raw, chose=forward_chose, first="concise", text=""),
        reversed=Verdict(raw=reversed_raw, chose=reversed_chose, first="verbose", text=""),
    )


def test_same_answer_in_both_orders_is_consistent():
    r = _both("A", "concise", "B", "concise")
    assert r.consistent
    assert not r.flipped


def test_same_slot_in_both_orders_is_a_flip_not_a_verdict():
    """Choosing 'A' twice means two different answers were chosen. That is position bias."""
    r = _both("A", "concise", "A", "verbose")
    assert not r.consistent
    assert r.flipped
    assert r.chose_first_both_times


def test_choosing_the_second_slot_twice_is_also_position_bias():
    r = _both("B", "verbose", "B", "concise")
    assert r.flipped
    assert r.chose_second_both_times
    assert not r.chose_first_both_times


def test_an_unparsed_verdict_is_neither_consistent_nor_flipped():
    r = _both("", "", "A", "verbose")
    assert not r.consistent
    assert not r.flipped


def test_a_tie_in_both_orders_is_consistent():
    r = _both("TIE", "tie", "TIE", "tie")
    assert r.consistent
    assert not r.flipped


# --- tallying -----------------------------------------------------------------------------------


def test_position_bias_is_counted_and_excluded_from_usable_verdicts():
    cell = tally("plain", "tie", [_both("A", "concise", "A", "verbose")])
    assert cell.position_a == 1
    assert cell.flipped == 1
    assert cell.consistent == 0
    assert cell.consistent_total == 0


def test_a_consistent_preference_for_the_longer_answer_is_recorded():
    cell = tally("plain", "tie", [_both("B", "verbose", "A", "verbose")])
    assert cell.consistent == 1
    assert cell.consistent_verbose == 1
    assert cell.flipped == 0


def test_ties_are_excluded_from_the_length_denominator():
    """A tie expresses no length preference and must not dilute the rate either way."""
    cell = tally("tie_allowed", "tie", [_both("TIE", "tie", "TIE", "tie")])
    assert cell.consistent == 1
    assert cell.consistent_total == 0
    assert cell.tie_both == 1


def test_unparsed_verdicts_are_counted():
    cell = tally("plain", "tie", [_both("", "", "", "")])
    assert cell.unparsed == 1
    assert cell.n == 1


def test_rates_are_zero_rather_than_dividing_by_zero_on_an_empty_cell():
    cell = tally("plain", "tie", [])
    assert cell.flip_rate == 0.0
    assert cell.position_rate == 0.0
    assert cell.decisive == 0.0


# --- live ----------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_a_judge_returns_a_parseable_verdict():
    from projects.p05_judge_bias.judge import compare
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip("model not pulled")

    q = BY_ID["q1"]
    assert compare(q, q.concise, q.wrong, prompt="plain", model=tag).raw in {"A", "B"}
