"""Tests for project 01.

Split deliberately: everything that can be tested without a model is, and the handful that
genuinely needs one is marked `live` and skipped when ollama is not running. A test suite
that only passes when a GPU is warm is a test suite nobody runs.

The corpus tests are the important ones. Every headline number in the project README is a
rate whose denominator comes from the corpus, so the corpus is pinned: if someone adds an
abstract without ground truth, or shifts the balance of omissions, these fail loudly rather
than silently changing what "27% fabrication" means.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from projects.p01_structured_output.corpus import BY_ID, CORPUS, omission_counts
from projects.p01_structured_output.schemas import LADDER, NULLABLE_FIELDS, L1Flat, L5Union
from projects.p01_structured_output.scoring import (
    Aggregate,
    score,
    scoreable_fields,
    strings_match,
    values_match,
)
from projects.p01_structured_output.strategies import (
    extract_json_blob,
    repair_json,
)
from shared.llm import server_is_up
from shared.models import BY_TAG, FLEET, with_role

# --- the corpus ---------------------------------------------------------------------------


def test_corpus_ids_are_unique():
    assert len({a.id for a in CORPUS}) == len(CORPUS)


def test_every_abstract_has_the_core_fields():
    for a in CORPUS:
        for field in ("population", "intervention", "comparator", "direction", "design"):
            assert field in a.expected, f"{a.id} is missing ground truth for {field}"


def test_omission_balance_is_pinned():
    """The corpus must keep enough genuine absences for a fabrication rate to mean anything."""
    assert omission_counts() == {
        "n_randomised": 4,
        "interval": 6,
        "p_value": 5,
        "registration": 5,
    }


def test_direction_values_are_valid():
    allowed = {"favours_intervention", "favours_comparator", "no_difference"}
    for a in CORPUS:
        assert a.expected["direction"] in allowed, a.id


def test_both_trial_designs_are_represented():
    designs = [a.expected["design"] for a in CORPUS]
    assert designs.count("non_inferiority") == 2
    assert designs.count("superiority") == 10


def test_non_inferiority_abstracts_carry_margin_and_met():
    for a in CORPUS:
        if a.expected["design"] == "non_inferiority":
            assert isinstance(a.expected["margin"], float), a.id
            assert isinstance(a.expected["met"], bool), a.id


def test_registration_ground_truth_matches_the_schema_pattern():
    """If ground truth cannot satisfy the schema, the schema is testing the wrong thing."""
    for a in CORPUS:
        reg = a.expected.get("registration")
        if reg is not None:
            assert reg.startswith("NCT") and len(reg) == 11 and reg[3:].isdigit(), a.id


def test_ground_truth_intervals_are_ordered():
    for a in CORPUS:
        interval = a.expected.get("interval")
        if interval is not None:
            assert interval["low"] < interval["high"], a.id


# --- the schema ladder ---------------------------------------------------------------------


def test_ladder_is_five_levels_of_base_models():
    assert len(LADDER) == 5
    assert all(issubclass(s, BaseModel) for s in LADDER.values())


def test_every_level_produces_a_json_schema():
    for name, schema in LADDER.items():
        spec = schema.model_json_schema()
        assert json.dumps(spec), name


def test_nullable_field_map_covers_every_level():
    assert set(NULLABLE_FIELDS) == set(LADDER)


def test_declared_nullable_fields_exist_on_their_schema():
    for level, fields in NULLABLE_FIELDS.items():
        model_fields = LADDER[level].model_fields
        for f in fields:
            assert f in model_fields, f"{level} declares {f}, which it does not have"


def test_l1_rejects_a_missing_field():
    with pytest.raises(ValidationError):
        L1Flat.model_validate({"population": "adults", "intervention": "x"})


def test_l5_union_discriminates_on_design():
    value = L5Union.model_validate(
        {
            "population": "adults",
            "intervention": "x",
            "comparator": "y",
            "direction": "no_difference",
            "analysis": {
                "design": "non_inferiority",
                "primary_outcome": "cure",
                "margin": 10.0,
                "met": True,
            },
        }
    )
    assert value.analysis.design == "non_inferiority"
    assert value.analysis.margin == 10.0


# --- JSON repair ---------------------------------------------------------------------------


def test_extracts_from_a_fenced_block():
    assert extract_json_blob('talk\n```json\n{"a": 1}\n```\nmore') == '{"a": 1}'


def test_extracts_nested_objects_without_truncating():
    """The naive regex version stops at the first closing brace. This is that regression."""
    text = 'prefix {"a": {"b": {"c": 1}}, "d": 2} suffix'
    assert extract_json_blob(text) == '{"a": {"b": {"c": 1}}, "d": 2}'


def test_braces_inside_strings_do_not_confuse_the_scanner():
    text = '{"note": "a } brace", "n": 1}'
    assert extract_json_blob(text) == text


def test_escaped_quote_inside_string_is_handled():
    text = '{"note": "he said \\"hi\\" }", "n": 1}'
    assert json.loads(extract_json_blob(text))["n"] == 1


def test_repair_fixes_python_literals_and_trailing_commas():
    fixed, changed = repair_json('{"a": None, "b": True, "c": [1, 2,],}')
    assert changed
    assert json.loads(fixed) == {"a": None, "b": True, "c": [1, 2]}


def test_repair_leaves_the_word_none_inside_a_string_alone():
    fixed, _ = repair_json('{"a": "None of the above"}')
    assert json.loads(fixed)["a"] == "None of the above"


def test_repair_reports_when_it_changed_nothing():
    _, changed = repair_json('{"a": 1}')
    assert changed is False


def test_repair_returns_unusable_text_unchanged_rather_than_guessing():
    fixed, _ = repair_json("I could not find that information.")
    assert "could not find" in fixed


# --- scoring --------------------------------------------------------------------------------


def test_strings_match_ignores_case_and_punctuation():
    assert strings_match("Adults with insomnia.", "adults with insomnia")


def test_strings_match_allows_containment_in_both_directions():
    assert strings_match("adults", "adults with insomnia")
    assert strings_match("adults with insomnia", "adults")


def test_null_and_value_never_match():
    assert not values_match(0, None)
    assert not values_match(None, 0)
    assert not values_match("", None)
    assert values_match(None, None)


def test_zero_is_not_confused_with_absent():
    """`0` is a reported value and `None` is an absence. Conflating them is the bug."""
    assert not values_match(0.0, None)
    assert values_match(0.0, 0)


def test_booleans_are_not_compared_as_numbers():
    assert not values_match(True, 1.5)
    assert values_match(True, True)


def test_nested_interval_compares_field_by_field():
    assert values_match({"low": 0.9, "high": 2.7, "level": 0.95}, {"low": 0.9, "high": 2.7})
    assert not values_match({"low": 0.9, "high": 9.9}, {"low": 0.9, "high": 2.7})


def test_failed_extraction_is_charged_for_every_field_it_owed():
    a = BY_ID["t09"]
    s = score(None, a, LADDER["L5_union"])
    assert s.valid is False
    assert s.n_fields == len(scoreable_fields(LADDER["L5_union"], a))
    assert s.n_correct == 0
    assert s.accuracy == 0.0


def test_failed_extraction_fabricates_nothing():
    """A parse failure produced no document, so it cannot have invented a value."""
    s = score(None, BY_ID["t02"], LADDER["L4_constrained"])
    assert s.n_fabricated == 0
    assert s.n_omitted == 0


def test_scoreable_fields_lifts_the_union_members():
    fields = set(scoreable_fields(LADDER["L5_union"], BY_ID["t09"]))
    assert {"design", "margin", "met"} <= fields
    assert "analysis" not in fields


def test_perfect_extraction_scores_one():
    a = BY_ID["t01"]
    value = L1Flat.model_validate(
        {
            "population": a.expected["population"],
            "intervention": a.expected["intervention"],
            "comparator": a.expected["comparator"],
        }
    )
    s = score(value, a, L1Flat)
    assert s.accuracy == 1.0
    assert s.n_fabricated == 0


def test_aggregate_excludes_failures_from_the_fabrication_denominator():
    """Otherwise a strategy that fails often looks like one that invents rarely."""
    agg = Aggregate()
    agg.add(score(None, BY_ID["t02"], LADDER["L4_constrained"]))
    assert agg.n_nullable == 0
    assert agg.fabrication_rate == 0.0
    assert agg.validity == 0.0
    assert agg.accuracy == 0.0


def test_aggregate_rates_come_from_counts_not_averaged_rates():
    agg = Aggregate()
    agg.n_fields, agg.n_correct = 10, 7
    assert agg.accuracy == pytest.approx(0.7)


# --- the fleet registry ----------------------------------------------------------------------


def test_fleet_tags_are_unique():
    assert len({m.tag for m in FLEET}) == len(FLEET)


def test_with_role_returns_smallest_first():
    tools = with_role("tools")
    assert [m.params_b for m in tools] == sorted(m.params_b for m in tools)


def test_exactly_one_embedding_model_is_registered():
    assert len(with_role("embedding")) == 1


def test_embedding_model_is_not_advertised_for_tools():
    for m in FLEET:
        if m.embedding:
            assert not m.tools, f"{m.tag} cannot both embed and call tools here"


def test_every_registry_entry_has_a_note():
    assert all(m.notes for m in FLEET), "a model with no note is a model nobody can choose"


def test_by_tag_covers_the_whole_fleet():
    assert set(BY_TAG) == {m.tag for m in FLEET}


def test_two_fresh_ledgers_are_not_equal():
    """A plain @dataclass would make these equal, and LangChain deduplicates handlers.

    That silently dropped the second of `callbacks=[inner, outer]`, so an outer ledger
    reported zero calls for work that had happened. `eq=False` is what stops it.
    """
    from shared.llm import Ledger

    assert Ledger() != Ledger()


def test_a_ledger_stays_usable_as_a_dict_key():
    """Dedup and lookup both need identity, so it must remain hashable."""
    from shared.llm import Ledger

    a, b = Ledger(), Ledger()
    assert len({a, b}) == 2


# --- live ------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_constrained_decoding_always_validates():
    """The one guarantee `constrained` actually makes. If this fails, ollama changed."""
    from projects.p01_structured_output.strategies import constrained
    from shared.llm import installed_tags
    from shared.models import default_chat_model

    tag = default_chat_model()
    if tag not in installed_tags():
        pytest.skip(f"{tag} is not pulled")

    result = constrained(L1Flat, BY_ID["t01"].text, model=tag)
    assert result.ok, result.last_error
