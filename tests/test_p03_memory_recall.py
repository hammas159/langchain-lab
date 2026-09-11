"""Tests for project 03.

The corpus tests matter most here. Every headline number is "did this value survive", checked
by string match, so a fact whose answer does not literally appear in the conversation would be
scored as lost by every strategy including `full` — an instrument that reads as a finding.
`test_every_fact_is_present_in_the_full_transcript` is the guard against that.
"""

from __future__ import annotations

import pytest

from projects.p03_memory_recall.conversation import (
    FACTS,
    KINDS,
    TURNS,
    Fact,
    Turn,
    facts_within,
    position,
    transcript,
    turns_upto,
)
from projects.p03_memory_recall.memories import Config, full, window
from projects.p03_memory_recall.probe import Probe, check_survives
from shared.llm import server_is_up

# --- the conversation --------------------------------------------------------------------


def test_fact_ids_are_unique():
    assert len({f.id for f in FACTS}) == len(FACTS)


def test_every_fact_is_present_in_the_full_transcript():
    """If `full` cannot recall a fact, the fact is broken, not the strategy."""
    text = transcript(TURNS)
    missing = [f.id for f in FACTS if not f.matches(text)]
    assert missing == [], f"not findable in the conversation: {missing}"


def test_every_fact_kind_is_declared():
    assert {f.kind for f in FACTS} <= set(KINDS)


def test_all_five_kinds_are_represented():
    assert {f.kind for f in FACTS} == set(KINDS)


def test_facts_are_spread_across_both_halves():
    """A conversation with every fact at the end would flatter every windowing strategy."""
    half = len(TURNS) / 2
    assert any(f.turn < half for f in FACTS)
    assert any(f.turn >= half for f in FACTS)


def test_facts_are_stated_in_user_turns():
    for f in FACTS:
        assert TURNS[f.turn].role == "user", f.id


def test_turns_alternate_roles():
    for i, t in enumerate(TURNS):
        assert t.role == ("user" if i % 2 == 0 else "assistant"), i


def test_facts_within_excludes_the_unspoken():
    early = facts_within(6)
    assert all(f.turn < 6 for f in early)
    assert "sla" not in {f.id for f in early}


def test_turns_upto_slices_from_the_start():
    assert turns_upto(4) == TURNS[:4]
    assert turns_upto() == TURNS


def test_position_is_zero_at_the_start_and_one_at_the_end():
    first = Fact("a", 0, "value", "?", "x")
    last = Fact("b", len(TURNS) - 1, "value", "?", "x")
    assert position(first, len(TURNS)) == 0.0
    assert position(last, len(TURNS)) == 1.0


# --- matching ------------------------------------------------------------------------------


def test_matching_is_case_insensitive():
    f = Fact("x", 0, "name", "?", "Metabase")
    assert f.matches("we are keeping metabase")


def test_accepted_alternatives_match():
    f = Fact("x", 0, "value", "?", "85,000", ("85000",))
    assert f.matches("the ceiling is 85000 dollars")


def test_a_near_miss_does_not_match():
    """`eu-west-2` must not satisfy a fact whose answer is `eu-west-1`."""
    f = Fact("x", 0, "constraint", "?", "eu-west-1", ("eu west 1",))
    assert not f.matches("data stays in eu-west-2")


def test_check_survives_reads_the_context_not_the_conversation():
    f = Fact("x", 0, "value", "?", "99.5")
    assert check_survives(f, "the SLA is 99.5 percent")
    assert not check_survives(f, "the SLA was agreed with the business")


# --- strategies -----------------------------------------------------------------------------


def test_full_keeps_everything_verbatim():
    m = full(TURNS, Config())
    assert m.verbatim_turns == len(TURNS)
    assert m.summarised_turns == 0
    assert m.maintenance_calls == 0


def test_full_recalls_every_fact_by_construction():
    m = full(TURNS, Config())
    assert all(check_survives(f, m.context) for f in FACTS)


def test_window_keeps_only_the_last_k_turns():
    m = window(TURNS, Config(window=4))
    assert m.verbatim_turns == 4
    assert TURNS[-1].text in m.context
    assert TURNS[0].text not in m.context


def test_window_larger_than_the_conversation_keeps_all_of_it():
    m = window(TURNS, Config(window=999))
    assert m.verbatim_turns == len(TURNS)


def test_window_costs_nothing_to_maintain():
    """The reason it is the baseline worth beating: it is free."""
    assert window(TURNS, Config(window=6)).maintenance_calls == 0


def test_window_drops_the_earliest_facts():
    m = window(TURNS, Config(window=6))
    assert not check_survives(next(f for f in FACTS if f.id == "budget"), m.context)
    assert check_survives(next(f for f in FACTS if f.id == "sla"), m.context)


def test_transcript_labels_every_turn_with_its_role():
    text = transcript((Turn("user", "hello"), Turn("assistant", "hi")))
    assert text == "user: hello\nassistant: hi"


# --- probe verdicts ---------------------------------------------------------------------------


def _probe(**kw) -> Probe:
    base = {
        "fact": FACTS[0],
        "survives": True,
        "answered": True,
        "declined": False,
        "reply": "x",
    }
    return Probe(**{**base, **kw})


def test_buried_means_present_but_not_produced():
    assert _probe(survives=True, answered=False).buried


def test_a_recalled_fact_is_not_buried():
    assert not _probe(survives=True, answered=True).buried


def test_a_dropped_fact_cannot_be_buried():
    assert not _probe(survives=False, answered=False).buried


def test_confabulation_is_a_lost_value_answered_anyway():
    assert _probe(survives=False, answered=False, declined=False).confabulated


def test_declining_a_lost_value_is_not_confabulation():
    assert not _probe(survives=False, answered=False, declined=True).confabulated


def test_a_surviving_value_is_never_confabulated():
    assert not _probe(survives=True, answered=False, declined=False).confabulated


# --- live ---------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_summary_costs_calls_and_window_does_not():
    from projects.p03_memory_recall.memories import summary_guarded
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    if default_chat_model() not in installed_tags():
        pytest.skip("model not pulled")

    cfg = Config(window=6, every=8)
    assert summary_guarded(TURNS[:8], cfg).maintenance_calls >= 1
    assert window(TURNS[:8], cfg).maintenance_calls == 0
