"""Retrieve, extract, and classify each field by what the model *could have known*.

Project 01 scored a field as fabricated when ground truth was `null` and the model returned a
value. With a retriever in the way that definition is no longer sufficient, because there are
now two distinct ways for a field to be absent from the model's context:

    true absence        the paper does not report it       -> project 01's fabrication
    manufactured absence the paper reports it, retrieval missed it

The model sees exactly the same thing in both cases: nothing. So the interesting quantity is
what it does when the evidence was withheld by the pipeline rather than by the source. That is
a **phantom**: a value produced for a field whose evidence the model was never shown.

A phantom is worse than an ordinary fabrication in one specific way. An ordinary fabrication can
be blamed on the document being silent. A phantom happens on a document that *does* state the
answer, so a reviewer checking the paper will find the field there and assume the extraction
was grounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from projects.p01_structured_output.corpus import BY_ID as ABSTRACT_BY_ID
from projects.p01_structured_output.scoring import scoreable_fields, values_match
from projects.p01_structured_output.strategies import constrained
from projects.p02_retrieval_absences.papers import Paper
from projects.p02_retrieval_absences.retrieval import (
    EXTRACTION_QUERY,
    Retrieved,
    context_text,
    full_context,
    retrieve,
    retrieve_per_field,
)
from shared.llm import Ledger

Verdict = Literal[
    "grounded_correct",  # evidence retrieved, value right
    "grounded_wrong",  # evidence retrieved, value wrong
    "phantom",  # evidence NOT retrieved but in the paper, model produced a value anyway
    "lucky_phantom",  # ... and the invented value happened to be right
    "honest_null",  # evidence not retrieved, model said null. Correct given its context.
    "fabricated",  # paper genuinely silent, model produced a value (project 01's sense)
    "true_null",  # paper genuinely silent, model said null
    "missed",  # evidence retrieved and the model still said null
]


@dataclass
class FieldOutcome:
    name: str
    got: object
    want: object
    evidence_in_paper: bool
    evidence_retrieved: bool
    verdict: Verdict


@dataclass
class Result:
    paper_id: str
    strategy: str
    retrieved: Retrieved
    valid: bool
    outcomes: list[FieldOutcome] = field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0
    context_words: int = 0
    raw: str = ""

    def count(self, *verdicts: Verdict) -> int:
        return sum(1 for o in self.outcomes if o.verdict in verdicts)

    @property
    def n_withheld(self) -> int:
        """Fields the paper states but retrieval did not supply — the phantom denominator."""
        return sum(1 for o in self.outcomes if o.evidence_in_paper and not o.evidence_retrieved)

    @property
    def phantom_rate(self) -> float:
        return self.count("phantom", "lucky_phantom") / self.n_withheld if self.n_withheld else 0.0


def classify(
    name: str,
    got: object,
    want: object,
    *,
    in_paper: bool,
    retrieved: bool,
) -> Verdict:
    """Decide what a single field's outcome actually was.

    The ordering matters. Whether the evidence reached the model is checked *before* whether
    the answer was right, because a correct value produced without evidence is still a guess
    and pooling it with grounded correct answers is how a phantom rate gets hidden.
    """
    produced = got is not None

    if not in_paper:
        # Project 01's world: the source is silent.
        return "fabricated" if produced else "true_null"

    if not retrieved:
        if not produced:
            return "honest_null"
        return "lucky_phantom" if values_match(got, want) else "phantom"

    if not produced:
        return "missed"
    return "grounded_correct" if values_match(got, want) else "grounded_wrong"


def run(
    paper: Paper,
    schema: type[BaseModel],
    *,
    strategy: str,
    model: str,
    k: int = 8,
    per_field_k: int = 1,
) -> Result:
    """One paper through one retrieval strategy, extracted and classified.

    Extraction is always `constrained` from project 01. That is the strategy project 01 found
    to be both the most reliable and the most accurate, so using it here means any failure
    observed belongs to retrieval rather than to the decoder — the variable under test is the
    context, and everything else is held still.
    """
    fields = tuple(scoreable_fields(schema, ABSTRACT_BY_ID[paper.id]))

    if strategy == "full_context":
        selection = full_context(paper)
    elif strategy == "per_field":
        selection = retrieve_per_field(paper, fields, per_field_k=per_field_k)
    else:
        selection = retrieve(paper, EXTRACTION_QUERY, k=k)

    context = context_text(paper, selection)
    book = Ledger()
    extraction = constrained(schema, context, model=model, callbacks=[book])

    result = Result(
        paper_id=paper.id,
        strategy=strategy,
        retrieved=selection,
        valid=extraction.ok,
        calls=book.n_calls,
        seconds=round(book.seconds, 2),
        context_words=len(context.split()),
        raw=extraction.attempts[-1].raw if extraction.attempts else "",
    )
    if not extraction.ok:
        return result

    data = extraction.value.model_dump(mode="python")
    analysis = data.pop("analysis", None)
    if isinstance(analysis, dict):
        data.update(analysis)

    evidenced = paper.evidenced_fields
    for name in fields:
        got = data.get(name)
        if hasattr(got, "value"):
            got = got.value
        if isinstance(got, BaseModel):
            got = got.model_dump()
        want = paper.expected.get(name)
        in_paper = name in evidenced
        was_retrieved = name in selection.covered
        result.outcomes.append(
            FieldOutcome(
                name=name,
                got=got,
                want=want,
                evidence_in_paper=in_paper,
                evidence_retrieved=was_retrieved,
                verdict=classify(name, got, want, in_paper=in_paper, retrieved=was_retrieved),
            )
        )
    return result
