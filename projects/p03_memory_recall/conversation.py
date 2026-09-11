"""A long conversation with facts planted at known turns.

Every memory strategy is a lossy compression of a conversation, and the usual way of judging
one is to read a summary and decide it "looks right". A summary that looks right is exactly the
failure mode worth measuring: summarisation is good at preserving what a conversation was
*about* and bad at preserving the values inside it.

So the conversation below is built with **facts planted at known turn indices**, each with a
question whose answer appears exactly once in the whole dialogue. After the conversation has
been through a memory strategy, each fact is probed. Whether the answer survived is then a
measured quantity rather than an impression.

Facts are tagged by `kind` because the central hypothesis is that loss is not uniform:

    value        a number that matters  (budget, retry count, version)
    name         a person or system name
    date         a deadline
    constraint   a hard rule  ("must never", "only if")
    preference   a soft steer ("prefer", "ideally")

A summary that keeps every `preference` and loses every `value` scores the same as one that does
the reverse, unless the kinds are counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: str
    turn: int
    kind: str
    question: str
    answer: str
    #: Accepted alternative spellings of the answer. Kept explicit rather than fuzzy-matched,
    #: so that "eu-west-1" and "EU West 1" both pass and "eu-west-2" does not.
    accept: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(v.lower() in lowered for v in (self.answer, *self.accept))


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


#: A planning conversation for a data-platform migration. Twenty-four turns, twelve facts, and
#: enough ordinary discussion between them that a summariser has plenty it could keep instead.
TURNS: tuple[Turn, ...] = (
    Turn(
        "user",
        "We're planning the migration of our analytics warehouse off the legacy box. Can you help me think it through?",
    ),
    Turn(
        "assistant",
        "Yes. Let's start with what you're moving, what constraints you're under, and who is involved.",
    ),
    Turn(
        "user",
        "The warehouse is about 40 TB. The hard budget ceiling for the whole migration is 85,000 dollars and we cannot go over it.",
    ),
    Turn(
        "assistant",
        "Noted. A ceiling that firm usually means we should price the destination before designing anything else.",
    ),
    Turn(
        "user",
        "Right. We're going to Postgres 17 with the pgvector extension, because half our new workload is embedding search.",
    ),
    Turn(
        "assistant",
        "Sensible. Postgres 17 handles both the relational side and the vector index, which saves you running a second store.",
    ),
    Turn(
        "user",
        "The migration lead is Ayesha Rahman. She runs the platform team and signs off on anything touching production.",
    ),
    Turn(
        "assistant",
        "Understood, so any cutover plan needs her approval before it goes anywhere near production.",
    ),
    Turn(
        "user",
        "One absolute rule: customer data must never leave the eu-west-1 region, not even for a temporary staging copy. That's a legal requirement, not a preference.",
    ),
    Turn(
        "assistant",
        "That rules out several managed migration services that stage through a US region. I'll keep it in mind for every option.",
    ),
    Turn(
        "user",
        "Good. The cutover deadline is the 14th of March 2027, tied to the legacy contract expiring.",
    ),
    Turn(
        "assistant",
        "That gives a reasonable runway. I'd suggest working backwards from it with a freeze period before the date.",
    ),
    Turn(
        "user",
        "We'd ideally prefer a blue-green cutover rather than a big-bang switch, but it isn't mandatory if cost gets tight.",
    ),
    Turn(
        "assistant",
        "Blue-green costs you a period of double running. That is usually where a fixed budget gets tested.",
    ),
    Turn(
        "user",
        "Understood. What about the ingestion side? We have roughly 200 pipelines feeding the warehouse daily.",
    ),
    Turn(
        "assistant",
        "At that count you will want to migrate them in waves grouped by upstream system rather than individually.",
    ),
    Turn(
        "user",
        "That makes sense. On reliability: the retry policy for failed loads should be 5 attempts with exponential backoff.",
    ),
    Turn(
        "assistant",
        "Five is a reasonable ceiling. Beyond that you're usually masking a real upstream problem rather than recovering from a blip.",
    ),
    Turn(
        "user",
        "We also need to keep the old system readable for a while after cutover. Audit wants 90 days of parallel read access.",
    ),
    Turn(
        "assistant",
        "Ninety days of read-only parallel access is cheap compared with the cost of failing an audit.",
    ),
    Turn(
        "user",
        "The reporting layer is Metabase and we are not changing it. The team has just been trained on it.",
    ),
    Turn(
        "assistant",
        "Then the migration has to preserve whatever Metabase depends on, which usually means the view layer and any saved questions.",
    ),
    Turn(
        "user",
        "Last thing for now: our SLA with the business is 99.5 percent availability on the warehouse during business hours.",
    ),
    Turn(
        "assistant",
        "That target is achievable with a blue-green approach, and it is the figure the cutover window should be designed against.",
    ),
)

FACTS: tuple[Fact, ...] = (
    Fact(
        "budget",
        2,
        "value",
        "What is the hard budget ceiling for the migration?",
        "85,000",
        ("85000", "85 000", "$85,000"),
    ),
    Fact(
        "target_db",
        4,
        "name",
        "Which database and extension are they migrating to?",
        "Postgres 17",
        ("PostgreSQL 17", "postgres17"),
    ),
    Fact("lead", 6, "name", "Who is the migration lead?", "Ayesha Rahman", ("Ayesha",)),
    Fact(
        "region",
        8,
        "constraint",
        "Which region must customer data never leave?",
        "eu-west-1",
        ("eu west 1", "euwest1"),
    ),
    Fact(
        "deadline",
        10,
        "date",
        "What is the cutover deadline?",
        "14 March 2027",
        ("14th of March 2027", "March 14, 2027", "2027-03-14", "14/03/2027"),
    ),
    Fact(
        "cutover_style",
        12,
        "preference",
        "Which cutover style do they prefer?",
        "blue-green",
        ("blue green", "bluegreen"),
    ),
    Fact(
        "pipelines",
        14,
        "value",
        "Roughly how many pipelines feed the warehouse daily?",
        "200",
        ("two hundred",),
    ),
    Fact(
        "retries", 16, "value", "How many retry attempts should failed loads make?", "5", ("five",)
    ),
    Fact(
        "parallel_days",
        18,
        "value",
        "How many days of parallel read access does audit require?",
        "90",
        ("ninety",),
    ),
    Fact("reporting", 20, "name", "Which reporting tool are they keeping?", "Metabase", ()),
    Fact(
        "sla",
        22,
        "value",
        "What availability SLA is agreed with the business?",
        "99.5",
        ("99.5%", "99.5 percent"),
    ),
    Fact(
        "size", 2, "value", "Roughly how large is the warehouse?", "40 TB", ("40TB", "40 terabytes")
    ),
)

BY_ID = {f.id: f for f in FACTS}
KINDS = ("value", "name", "date", "constraint", "preference")


def turns_upto(n: int | None = None) -> tuple[Turn, ...]:
    return TURNS if n is None else TURNS[:n]


def facts_within(n_turns: int) -> tuple[Fact, ...]:
    """Facts that have actually been stated by turn `n_turns`.

    Probing for a fact the conversation has not reached yet would score a strategy for failing
    to recall something nobody said.
    """
    return tuple(f for f in FACTS if f.turn < n_turns)


def transcript(turns: tuple[Turn, ...]) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in turns)


def position(fact: Fact, n_turns: int) -> float:
    """How far back in the conversation a fact sits, as a fraction. 0.0 = oldest.

    Reported alongside recall because the interesting result is not an average, it is the shape
    of the curve: which end of the conversation a strategy drops.
    """
    return fact.turn / max(n_turns - 1, 1)
