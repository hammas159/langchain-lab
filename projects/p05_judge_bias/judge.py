"""The judge, and the two orders every comparison is run in.

A pairwise judge is asked which of two answers is better. The essential detail is that it is
asked **twice** — once with the candidates in each order — because a verdict that flips when
the order flips carries no information about the answers.

Three judge prompts are provided, and the difference between them is the experiment:

    plain       "which is better". What most evaluation harnesses send.
    tie_allowed the same, with an explicit tie option. Removes the pressure to pick.
    rubric      names the criteria and states that length is not one of them.

`tie_allowed` matters more than it looks. A judge given only A and B **must** return one of
them, so a forced choice on two equally good answers is guaranteed to record a preference that
does not exist. Any measured bias in the `plain` condition is therefore partly an artefact of
the question it was asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from projects.p05_judge_bias.answers import Answer, Question
from shared.llm import Ledger, chat

PLAIN = (
    "You are judging which of two answers to a question is better.\n"
    "Reply with exactly one character: A or B."
)

TIE_ALLOWED = (
    "You are judging which of two answers to a question is better.\n"
    "Reply with exactly one of: A, B, or TIE.\n"
    "Answer TIE when the two answers are of equal quality. Two answers that state the same "
    "correct information are a tie even if one is longer."
)

RUBRIC = (
    "You are judging which of two answers to a question is better.\n"
    "Judge on factual correctness first, then on whether the answer addresses the question "
    "asked. Length is not a criterion: a short answer that is correct and complete is not "
    "worse than a long one that adds no further information. Do not reward elaboration.\n"
    "Reply with exactly one of: A, B, or TIE."
)

PROMPTS = {"plain": PLAIN, "tie_allowed": TIE_ALLOWED, "rubric": RUBRIC}


@dataclass
class Verdict:
    """One judgement, together with the position each candidate occupied."""

    #: "A", "B" or "TIE" as the judge said it.
    raw: str
    #: The variant the judge chose, resolved through the ordering: "concise", "verbose",
    #: "wrong", or "tie". This is what the analysis uses — the letter is meaningless once the
    #: order is randomised.
    chose: str
    #: Which variant was presented first.
    first: str
    text: str


_LETTER = re.compile(r"\b(A|B|TIE)\b", re.I)


def parse(reply: str) -> str:
    """Pull A / B / TIE out of whatever the model said.

    A judge told to answer with one character frequently answers with a sentence. Failing to
    parse that as a verdict and silently dropping it would bias the result towards whichever
    condition produces tidier output, so the match is deliberately permissive — but it takes
    the **first** token found, so "A is better than B" resolves to A rather than to whichever
    letter appears last.
    """
    match = _LETTER.search(reply.strip())
    if not match:
        return ""
    token = match.group(1).upper()
    return "TIE" if token == "TIE" else token


def compare(
    question: Question,
    left: Answer,
    right: Answer,
    *,
    prompt: str,
    model: str,
    ledger: Ledger | None = None,
) -> Verdict:
    """Judge `left` as A and `right` as B, exactly once, in that order."""
    llm = chat(model, callbacks=[ledger] if ledger else None)
    reply = llm.invoke(
        [
            SystemMessage(content=PROMPTS[prompt]),
            HumanMessage(
                content=(
                    f"Question: {question.text}\n\n"
                    f"Answer A:\n{left.text}\n\n"
                    f"Answer B:\n{right.text}\n\n"
                    "Which is better?"
                )
            ),
        ]
    )
    text = str(reply.content).strip()
    raw = parse(text)
    chose = {"A": left.variant, "B": right.variant, "TIE": "tie", "": ""}[raw]
    return Verdict(raw=raw, chose=chose, first=left.variant, text=text)


@dataclass
class BothOrders:
    """The same pair judged in both orders. The unit of analysis."""

    question: Question
    a: Answer
    b: Answer
    forward: Verdict
    reversed: Verdict

    @property
    def consistent(self) -> bool:
        """Did the judge pick the same *answer* regardless of where it sat?"""
        return bool(self.forward.chose) and self.forward.chose == self.reversed.chose

    @property
    def flipped(self) -> bool:
        """The judge changed its mind when the order changed — a pure position effect."""
        return (
            bool(self.forward.chose)
            and bool(self.reversed.chose)
            and self.forward.chose != self.reversed.chose
        )

    @property
    def chose_first_both_times(self) -> bool:
        """Picked whatever was in position A, twice. The signature of position bias."""
        return self.forward.raw == "A" and self.reversed.raw == "A"

    @property
    def chose_second_both_times(self) -> bool:
        return self.forward.raw == "B" and self.reversed.raw == "B"


def compare_both_orders(
    question: Question,
    a: Answer,
    b: Answer,
    *,
    prompt: str,
    model: str,
    ledger: Ledger | None = None,
) -> BothOrders:
    return BothOrders(
        question=question,
        a=a,
        b=b,
        forward=compare(question, a, b, prompt=prompt, model=model, ledger=ledger),
        reversed=compare(question, b, a, prompt=prompt, model=model, ledger=ledger),
    )
