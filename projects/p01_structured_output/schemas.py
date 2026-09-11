"""A difficulty ladder for structured extraction, and the field that matters most.

The task is PICO extraction from clinical-trial abstracts: population, intervention,
comparator, outcome. It was chosen over the usual invoice demo for one reason — a real
abstract frequently *does not report* a field, and the correct answer is then `null`.

That is the whole experiment. Every schema below has at least one optional field, and the
corpus deliberately contains abstracts where that field is absent. A model that fills it in
anyway has not made a formatting error; it has made something up, and the JSON it produced is
perfectly valid. Schema validity and correctness are independent, and most structured-output
demos only measure the first.

The five levels raise structural difficulty while holding the task constant, so a drop in
accuracy between L1 and L5 is attributable to the schema rather than to the question.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Direction(StrEnum):
    favours_intervention = "favours_intervention"
    favours_comparator = "favours_comparator"
    no_difference = "no_difference"


# --- L1: flat, three required strings -------------------------------------------------


class L1Flat(BaseModel):
    """The shape every structured-output tutorial stops at."""

    population: str = Field(description="Who was studied, as stated in the abstract.")
    intervention: str = Field(description="The treatment given to the active arm.")
    comparator: str = Field(description="What the intervention was compared against.")


# --- L2: an enum and an optional field ------------------------------------------------


class L2Enum(BaseModel):
    population: str
    intervention: str
    comparator: str
    direction: Direction = Field(description="Which arm the primary outcome favoured.")
    n_randomised: int | None = Field(
        default=None,
        description="Total participants randomised. Null if the abstract does not say.",
    )


# --- L3: a list of nested objects -----------------------------------------------------


class Outcome(BaseModel):
    name: str
    effect: float | None = Field(
        default=None, description="The point estimate. Null if not reported numerically."
    )
    unit: str | None = None


class L3Nested(BaseModel):
    population: str
    intervention: str
    comparator: str
    direction: Direction
    n_randomised: int | None = None
    outcomes: list[Outcome] = Field(
        default_factory=list, description="Every outcome the abstract reports a result for."
    )


# --- L4: constrained numerics and a formatted string ----------------------------------

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Interval(BaseModel):
    """A confidence interval, which must be ordered."""

    low: float
    high: float
    level: Probability = Field(default=0.95, description="Coverage, as a proportion.")


class L4Constrained(BaseModel):
    population: str
    intervention: str
    comparator: str
    direction: Direction
    n_randomised: int | None = Field(default=None, ge=1)
    primary_outcome: str
    effect: float | None = None
    interval: Interval | None = Field(
        default=None, description="Confidence interval for the effect. Null if not reported."
    )
    p_value: Probability | None = Field(
        default=None, description="Null if the abstract reports no p-value."
    )
    registration: str | None = Field(
        default=None,
        pattern=r"^NCT\d{8}$",
        description="Trial registration ID, exactly as NCT followed by eight digits.",
    )


# --- L5: a discriminated union --------------------------------------------------------


class Superiority(BaseModel):
    design: Literal["superiority"]
    primary_outcome: str
    effect: float | None = None


class NonInferiority(BaseModel):
    design: Literal["non_inferiority"]
    primary_outcome: str
    margin: float = Field(description="The pre-specified non-inferiority margin.")
    met: bool


class L5Union(BaseModel):
    population: str
    intervention: str
    comparator: str
    direction: Direction
    n_randomised: int | None = None
    analysis: Superiority | NonInferiority = Field(
        discriminator="design",
        description="Superiority trials report an effect; non-inferiority trials report a margin.",
    )


LADDER: dict[str, type[BaseModel]] = {
    "L1_flat": L1Flat,
    "L2_enum": L2Enum,
    "L3_nested": L3Nested,
    "L4_constrained": L4Constrained,
    "L5_union": L5Union,
}

#: Fields whose correct value is `None` for at least one abstract in the corpus. These are
#: the ones worth scoring separately — they are where a valid document is most often a wrong
#: one.
NULLABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "L1_flat": (),
    "L2_enum": ("n_randomised",),
    "L3_nested": ("n_randomised",),
    "L4_constrained": ("n_randomised", "interval", "p_value", "registration"),
    "L5_union": ("n_randomised",),
}
