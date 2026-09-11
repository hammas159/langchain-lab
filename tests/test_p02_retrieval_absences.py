"""Tests for project 02.

The important ones are on `classify`. It is the function that decides whether a wrong answer
was the model's fault or the retriever's, and every headline number in the project is a count
of its verdicts. It is also pure, so there is no excuse for not testing it exhaustively.

The paper tests pin the instrument: a field is only scoreable as "withheld" if the paper really
does state it, so the mapping from ground truth to passages has to stay honest.
"""

from __future__ import annotations

import pytest

from projects.p01_structured_output.corpus import BY_ID as ABSTRACT_BY_ID
from projects.p01_structured_output.corpus import CORPUS
from projects.p02_retrieval_absences.papers import (
    BY_ID,
    PAPERS,
    Paper,
    Passage,
    build_papers,
)
from projects.p02_retrieval_absences.pipeline import Result, classify
from projects.p02_retrieval_absences.retrieval import (
    PER_FIELD_QUERIES,
    Retrieved,
    context_text,
    cosine,
    full_context,
)
from shared.llm import server_is_up

# --- papers -----------------------------------------------------------------------------


def test_one_paper_per_abstract():
    assert len(PAPERS) == len(CORPUS)
    assert {p.id for p in PAPERS} == {a.id for a in CORPUS}


def test_papers_are_long_enough_to_need_selecting():
    for p in PAPERS:
        assert p.words > 400, p.id
        assert len(p.passages) >= 15, p.id


def test_ground_truth_is_not_restated():
    """Papers must inherit project 01's ground truth, never keep their own copy."""
    for p in PAPERS:
        assert p.expected is ABSTRACT_BY_ID[p.id].expected


def test_a_field_absent_from_ground_truth_has_no_passage():
    """The heart of the instrument: a truly-silent field must have no evidence anywhere."""
    for p in PAPERS:
        for name in ("n_randomised", "interval", "p_value", "registration"):
            if p.expected.get(name) is None:
                assert p.passages_for(name) == (), f"{p.id}/{name} invented evidence"


def test_a_field_present_in_ground_truth_has_exactly_one_passage():
    for p in PAPERS:
        for name in ("n_randomised", "interval", "p_value", "registration"):
            if p.expected.get(name) is not None:
                assert len(p.passages_for(name)) == 1, f"{p.id}/{name}"


def test_t02_states_nothing_it_should_not():
    """t02 omits four fields in project 01; its paper must omit the same four."""
    p = BY_ID["t02"]
    assert "n_randomised" not in p.evidenced_fields
    assert "p_value" not in p.evidenced_fields
    assert "registration" not in p.evidenced_fields
    assert "interval" not in p.evidenced_fields


def test_filler_passages_carry_no_fields():
    for p in PAPERS:
        empty = [x for x in p.passages if not x.fields]
        assert len(empty) >= 10, f"{p.id} has too little distractor material"


def test_registration_is_the_last_passage_when_present():
    """It sits at the end as it does in a real paper; that placement is part of the effect."""
    for p in PAPERS:
        idx = p.passages_for("registration")
        if idx:
            assert idx[0] == len(p.passages) - 1, p.id


def test_non_inferiority_papers_state_margin_and_met():
    for p in PAPERS:
        if p.expected["design"] == "non_inferiority":
            assert "margin" in p.evidenced_fields, p.id
            assert "met" in p.evidenced_fields, p.id


def test_build_papers_is_deterministic():
    assert [p.words for p in build_papers()] == [p.words for p in PAPERS]


# --- retrieval ---------------------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_a_zero_vector_without_dividing_by_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_full_context_misses_nothing():
    p = BY_ID["t01"]
    sel = full_context(p)
    assert sel.missed == frozenset()
    assert sel.recall == 1.0
    assert sel.covered == p.evidenced_fields


def test_recall_is_one_when_the_paper_states_nothing_retrievable():
    """A paper with no evidenced fields must not produce a divide-by-zero recall."""
    empty = Paper(id="x", title="x", passages=(Passage("Background", "text"),), expected={})
    assert full_context(empty).recall == 1.0


def test_context_is_rendered_in_document_order_not_relevance_order():
    p = BY_ID["t01"]
    sel = Retrieved(paper_id="t01", query="q", k=3, indices=(9, 2, 5), scores=(1.0, 0.9, 0.8))
    text = context_text(p, sel)
    assert text.index(p.passages[2].text) < text.index(p.passages[5].text)
    assert text.index(p.passages[5].text) < text.index(p.passages[9].text)


def test_every_scoreable_field_has_a_per_field_query():
    """A field with no query is silently never retrievable, which would fake the effect."""
    needed = set()
    for p in PAPERS:
        needed |= p.evidenced_fields
    assert needed <= set(PER_FIELD_QUERIES), needed - set(PER_FIELD_QUERIES)


# --- classify ------------------------------------------------------------------------------


def test_silent_paper_and_a_value_is_fabrication():
    assert classify("p_value", 0.04, None, in_paper=False, retrieved=False) == "fabricated"


def test_silent_paper_and_null_is_true_null():
    assert classify("p_value", None, None, in_paper=False, retrieved=False) == "true_null"


def test_withheld_evidence_and_a_wrong_value_is_a_phantom():
    assert classify("effect", 1.85, 1.8, in_paper=True, retrieved=False) == "phantom"


def test_withheld_evidence_and_a_right_value_is_a_lucky_phantom():
    """Right for the wrong reason. Pooling it with grounded answers hides the phantom rate."""
    assert classify("effect", 1.8, 1.8, in_paper=True, retrieved=False) == "lucky_phantom"


def test_withheld_evidence_and_null_is_honest():
    assert classify("effect", None, 1.8, in_paper=True, retrieved=False) == "honest_null"


def test_retrieved_evidence_and_a_right_value_is_grounded():
    assert classify("effect", 1.8, 1.8, in_paper=True, retrieved=True) == "grounded_correct"


def test_retrieved_evidence_and_a_wrong_value_is_the_models_fault():
    assert classify("effect", 9.9, 1.8, in_paper=True, retrieved=True) == "grounded_wrong"


def test_retrieved_evidence_and_null_is_missed():
    assert classify("effect", None, 1.8, in_paper=True, retrieved=True) == "missed"


def test_a_true_absence_is_never_called_a_phantom():
    """`in_paper=False` must dominate: retrieval cannot withhold what does not exist."""
    for retrieved in (True, False):
        verdict = classify("p_value", 0.04, None, in_paper=False, retrieved=retrieved)
        assert verdict == "fabricated"


def test_phantom_rate_is_zero_when_nothing_was_withheld():
    result = Result(
        paper_id="t01",
        strategy="full_context",
        retrieved=full_context(BY_ID["t01"]),
        valid=True,
    )
    assert result.n_withheld == 0
    assert result.phantom_rate == 0.0


# --- live -------------------------------------------------------------------------------------

live = pytest.mark.skipif(not server_is_up(), reason="ollama is not running")


@live
@pytest.mark.live
def test_per_field_retrieval_beats_a_single_schema_query():
    """The project's central retrieval claim, asserted rather than only reported."""
    from projects.p01_structured_output.schemas import LADDER
    from projects.p01_structured_output.scoring import scoreable_fields
    from projects.p02_retrieval_absences.retrieval import (
        EXTRACTION_QUERY,
        retrieve,
        retrieve_per_field,
    )
    from shared.llm import installed_tags
    from shared.models import embedding_model

    if embedding_model() not in installed_tags():
        pytest.skip("embedding model is not pulled")

    schema = LADDER["L4_constrained"]
    naive = []
    tuned = []
    for paper in PAPERS[:4]:
        fields = tuple(scoreable_fields(schema, ABSTRACT_BY_ID[paper.id]))
        naive.append(retrieve(paper, EXTRACTION_QUERY, k=8).recall)
        tuned.append(retrieve_per_field(paper, fields, per_field_k=1).recall)

    assert sum(tuned) / len(tuned) > sum(naive) / len(naive)
