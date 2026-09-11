"""Scoring, with fabrication counted separately from every other kind of wrong.

Most structured-output evaluations report one number: did it validate. That number is why
constrained decoding looks like a solved problem. Here a result is broken into four:

  validity      the object parsed and satisfied the schema
  exactness     the extracted value equals ground truth, field by field
  fabrication   ground truth is null and the model returned a value
  omission      ground truth is a value and the model returned null

Fabrication and omission are both "wrong", and treating them as the same error is the
mistake. An omission costs a reader a fact they must look up. A fabrication puts a number
they will not check in front of them. In an extraction pipeline feeding a database, the
second is the one that does damage, and it is the one that a validity-only metric hides.

String fields are compared leniently (normalised, containment either way) because
"adults with chronic tension headache" and "Adults with chronic tension headache."
are the same answer and scoring them as different measures punctuation. Numeric and null
comparisons are exact — that is where the interesting failures live.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from projects.p01_structured_output.corpus import Abstract

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def strings_match(got: str, want: str) -> bool:
    """Lenient string comparison: equal after normalisation, or one contains the other.

    Containment is allowed in both directions because a model that answers "adults" where the
    truth is "adults with insomnia" is under-specific but not inventing, and this project is
    about invention. The looseness is stated rather than hidden; a stricter comparison would
    lower every strategy's score by roughly the same amount and change no conclusion.
    """
    g, w = normalise(got), normalise(want)
    if not g or not w:
        return g == w
    return g == w or g in w or w in g


def values_match(got: object, want: object) -> bool:
    if want is None or got is None:
        return got is None and want is None
    # `bool` is a subclass of `int`, so this has to come before the numeric branch, and a bool
    # may only be compared with a bool: `bool(1.5) == bool(True)` is how a non-inferiority
    # margin of 1.5 silently scores as a correct `met: true`.
    if isinstance(want, bool) != isinstance(got, bool):
        return False
    if isinstance(want, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(want, dict) and isinstance(got, dict):
        return all(values_match(got.get(k), v) for k, v in want.items())
    if isinstance(want, str) and isinstance(got, str):
        return strings_match(got, want)
    return got == want


@dataclass
class FieldResult:
    name: str
    got: object
    want: object
    correct: bool
    fabricated: bool
    omitted: bool


@dataclass
class Score:
    valid: bool
    fields: list[FieldResult] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        return sum(1 for f in self.fields if f.correct)

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_fields if self.n_fields else 0.0

    @property
    def n_fabricated(self) -> int:
        return sum(1 for f in self.fields if f.fabricated)

    @property
    def n_nullable_truth(self) -> int:
        return sum(1 for f in self.fields if f.want is None)

    @property
    def n_omitted(self) -> int:
        return sum(1 for f in self.fields if f.omitted)


def _flatten(value: BaseModel) -> dict[str, object]:
    """Model to a flat dict, lifting the L5 union's discriminator to the top level.

    The union member is flattened rather than compared as a nested object so that `design`,
    `margin` and `met` are scored as ordinary fields and one ground-truth dictionary serves
    every level of the ladder.
    """
    data = value.model_dump(mode="python")
    analysis = data.pop("analysis", None)
    if isinstance(analysis, dict):
        data.update(analysis)
    outcomes = data.get("outcomes")
    if isinstance(outcomes, list):
        data["n_outcomes"] = len(outcomes)
    if isinstance(data.get("direction"), str):
        pass
    elif data.get("direction") is not None:
        data["direction"] = getattr(data["direction"], "value", data["direction"])
    return data


def scoreable_fields(schema: type[BaseModel], abstract: Abstract) -> list[str]:
    """The ground-truth fields this schema could in principle produce.

    Needed so that a failed extraction has the same denominator as a successful one. Without
    it, accuracy is computed only over the attempts that survived, and a strategy that
    validates a quarter of the time is credited with the accuracy of its lucky quarter.
    """
    names: set[str] = set(schema.model_fields)

    # The L5 union contributes its members' fields once flattened.
    analysis = schema.model_fields.get("analysis")
    if analysis is not None:
        for member in getattr(analysis.annotation, "__args__", ()):  # the union members
            names.update(getattr(member, "model_fields", {}))
        names.discard("analysis")

    if "outcomes" in names:
        names.add("n_outcomes")

    return [n for n in abstract.expected if n in names]


def score(
    value: BaseModel | None,
    abstract: Abstract,
    schema: type[BaseModel] | None = None,
) -> Score:
    """Score one extraction against ground truth.

    Only fields the schema actually has are scored, so the same ground truth can be used at
    every level without penalising L1 for not extracting a p-value it was never asked for.

    When `value` is None the extraction failed. Passing `schema` then charges that failure for
    every field it should have produced, rather than quietly dropping the row. Omitting
    `schema` keeps the old behaviour and is what the per-request UI wants, where a failure is
    shown as a failure rather than folded into a rate.
    """
    if value is None:
        if schema is None:
            return Score(valid=False)
        return Score(
            valid=False,
            fields=[
                FieldResult(
                    name=name,
                    got=None,
                    want=abstract.expected[name],
                    correct=False,
                    # A failed parse invented nothing and omitted nothing; it produced no
                    # document at all. Counting it as either would corrupt both rates.
                    fabricated=False,
                    omitted=False,
                )
                for name in scoreable_fields(schema, abstract)
            ],
        )

    got = _flatten(value)
    results: list[FieldResult] = []
    for name, want in abstract.expected.items():
        if name not in got:
            continue
        actual = got[name]
        if hasattr(actual, "value"):
            actual = actual.value
        if isinstance(actual, BaseModel):
            actual = actual.model_dump()
        correct = values_match(actual, want)
        results.append(
            FieldResult(
                name=name,
                got=actual,
                want=want,
                correct=correct,
                fabricated=want is None and actual is not None,
                omitted=want is not None and actual is None,
            )
        )
    return Score(valid=True, fields=results)


@dataclass
class Aggregate:
    """Totals across a run. Rates are computed from counts, never averaged from rates."""

    n: int = 0
    n_valid: int = 0
    n_fields: int = 0
    n_correct: int = 0
    n_nullable: int = 0
    n_fabricated: int = 0
    n_present: int = 0
    n_omitted: int = 0
    calls: int = 0
    seconds: float = 0.0

    def add(self, s: Score) -> None:
        self.n += 1
        self.n_valid += int(s.valid)

        # Accuracy is charged over every field the schema should have produced, including the
        # fields of extractions that never parsed. That is the whole point of the denominator:
        # a strategy does not get to be accurate on the attempts it failed to complete.
        self.n_fields += s.n_fields
        self.n_correct += s.n_correct

        # Fabrication and omission describe what a *returned document* did, so they are only
        # counted over documents that exist. A failed parse fabricated nothing; folding it in
        # would make an unreliable strategy look honest.
        if s.valid:
            self.n_nullable += s.n_nullable_truth
            self.n_fabricated += s.n_fabricated
            self.n_present += s.n_fields - s.n_nullable_truth
            self.n_omitted += s.n_omitted

    @property
    def validity(self) -> float:
        return self.n_valid / self.n if self.n else 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_fields if self.n_fields else 0.0

    @property
    def fabrication_rate(self) -> float:
        """Of the fields whose true value is null, how many were filled in anyway."""
        return self.n_fabricated / self.n_nullable if self.n_nullable else 0.0

    @property
    def omission_rate(self) -> float:
        return self.n_omitted / self.n_present if self.n_present else 0.0
