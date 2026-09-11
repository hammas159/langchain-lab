"""Asking whether a fact survived, and separating two failures that look identical.

A memory strategy can lose a fact in two quite different ways:

    dropped     the value is no longer anywhere in the context
    buried      the value is still in the context and the model does not produce it

Almost every write-up on conversational memory measures only the second, by asking the model a
question and marking the answer. That conflates the compression with the reader. A summary that
faithfully preserved "85,000 dollars" and a model that could not find it are different bugs with
different fixes, and only one of them is the memory strategy's fault.

So each fact is checked twice:

  `survives`  — deterministic string check against the context. Free, exact, no model.
  `answered`  — the model is asked the question with only that context and must reply
                "not stated" if it cannot find it.

The gap between them is the interesting column.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from projects.p03_memory_recall.conversation import Fact
from projects.p03_memory_recall.memories import Memory
from shared.llm import Ledger, chat

PROBE_SYSTEM = (
    "Answer the question using only the context provided.\n"
    "If the context does not state the answer, reply exactly: not stated\n"
    "Do not guess and do not infer a plausible value. Answer in as few words as possible."
)

NOT_STATED = "not stated"


@dataclass
class Probe:
    fact: Fact
    survives: bool
    answered: bool
    declined: bool
    reply: str

    @property
    def buried(self) -> bool:
        """The value is in the context and the model still did not produce it."""
        return self.survives and not self.answered

    @property
    def confabulated(self) -> bool:
        """The value is gone and the model supplied something other than a refusal.

        The same failure project 02 measures as a phantom, arriving here by a different route:
        there the pipeline withheld the evidence, here the memory strategy discarded it.
        """
        return not self.survives and not self.declined


def check_survives(fact: Fact, context: str) -> bool:
    return fact.matches(context)


def ask(
    fact: Fact,
    memory: Memory,
    *,
    model: str | None = None,
    ledger: Ledger | None = None,
) -> Probe:
    llm = chat(model, callbacks=[ledger] if ledger else None)
    reply = llm.invoke(
        [
            SystemMessage(content=PROBE_SYSTEM),
            HumanMessage(content=f"Context:\n{memory.context}\n\nQuestion: {fact.question}"),
        ]
    )
    text = str(reply.content).strip()
    declined = NOT_STATED in text.lower()
    return Probe(
        fact=fact,
        survives=check_survives(fact, memory.context),
        answered=(not declined) and fact.matches(text),
        declined=declined,
        reply=text,
    )
