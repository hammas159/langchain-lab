"""Four ways to carry a conversation forward, and what each one throws away.

Every strategy here answers the same question — "what do I put in the context window this
turn?" — and every one except `full` is a lossy compression. The project measures what each
loses, not how good the summary reads.

    full                    everything. The control, and usually not affordable.
    window                  the last `k` turns verbatim. Loses the beginning, keeps detail.
    summary_naive           older turns folded into an LLM summary, standard prompt.
    summary_guarded         the same, with one paragraph added to that prompt.
    summary_window_naive    a summary of the old plus the last `k` verbatim. The standard advice.
    summary_window_guarded  the same, guarded.

The naive/guarded pair exists because the first version of this project used only the guarded
prompt — which had been written to say "preserve every number, name and date" — and therefore
measured a summariser that had already been told the answer. Everything scored 100%. The
prompt is the variable; the architecture is held still across the pair.

Summarising costs model calls to *maintain*, which is a real cost and is counted. A strategy
that recalls no better than a free sliding window while charging five calls per conversation
is a finding, and it is only visible if the upkeep is reported next to the recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from projects.p03_memory_recall.conversation import Turn, transcript
from shared.llm import Ledger, chat

#: The prompt people actually write. It is not a straw man — it is the instruction in most
#: tutorials and in the default memory chains, and it asks for exactly what a summary usually
#: means: shorter, faithful to the discussion.
NAIVE_SUMMARISE = (
    "Summarise the conversation so far so it can be continued later. "
    "Be concise. Return the summary only."
)

#: The same job with one paragraph added. That paragraph is the variable this project isolates.
GUARDED_SUMMARISE = (
    "You maintain a running summary of a working conversation so it can continue after the "
    "earlier turns are discarded.\n"
    "Preserve every specific value: numbers, names, dates, regions, versions, limits and hard "
    "rules. A summary that keeps the topic but loses the figures is useless to the person who "
    "has to act on it.\n"
    "Be concise. Return the summary only."
)


@dataclass
class Memory:
    """What a strategy produced for one turn of a conversation."""

    strategy: str
    context: str
    #: Turns reproduced verbatim, for reporting how much detail survived untouched.
    verbatim_turns: int
    summarised_turns: int
    maintenance_calls: int = 0
    maintenance_seconds: float = 0.0
    summary: str = ""

    @property
    def words(self) -> int:
        return len(self.context.split())


@dataclass
class Config:
    window: int = 6
    #: Re-summarise every `every` turns rather than on every single turn, which is what a real
    #: deployment does — summarising on every turn is the dominant cost and nobody pays it.
    every: int = 4
    model: str | None = None
    ledger: Ledger | None = field(default=None)


def full(turns: tuple[Turn, ...], cfg: Config) -> Memory:
    return Memory(
        strategy="full",
        context=transcript(turns),
        verbatim_turns=len(turns),
        summarised_turns=0,
    )


def window(turns: tuple[Turn, ...], cfg: Config) -> Memory:
    kept = turns[-cfg.window :]
    return Memory(
        strategy="window",
        context=transcript(kept),
        verbatim_turns=len(kept),
        summarised_turns=0,
    )


def _summarise(text: str, cfg: Config, system: str, previous: str = "") -> tuple[str, int, float]:
    """Fold `text` into a running summary. Returns the summary, calls made, seconds spent."""
    book = Ledger()
    llm = chat(cfg.model, callbacks=[book] + ([cfg.ledger] if cfg.ledger else []))
    instruction = (
        f"Existing summary:\n{previous}\n\nNew turns to fold in:\n{text}\n\nUpdated summary:"
        if previous
        else f"Conversation so far:\n{text}\n\nSummary:"
    )
    reply = llm.invoke([SystemMessage(content=system), HumanMessage(content=instruction)])
    return str(reply.content).strip(), book.n_calls, book.seconds


def _fold(turns: tuple[Turn, ...], cfg: Config, system: str) -> tuple[str, int, float]:
    """Summarise incrementally, in chunks of `cfg.every`.

    Incremental rather than one-shot because that is what a deployed chain does: it cannot
    re-read the whole history each turn, which is the entire reason it is summarising. It also
    means loss compounds — a value dropped at fold 2 cannot be recovered at fold 5.
    """
    running = ""
    calls = 0
    seconds = 0.0
    for start in range(0, len(turns), cfg.every):
        chunk = turns[start : start + cfg.every]
        running, c, s = _summarise(transcript(chunk), cfg, system, previous=running)
        calls += c
        seconds += s
    return running, calls, seconds


def summary_naive(turns: tuple[Turn, ...], cfg: Config) -> Memory:
    """ "Summarise the conversation so far." The default instruction, and the control prompt."""
    running, calls, seconds = _fold(turns, cfg, NAIVE_SUMMARISE)
    return Memory(
        strategy="summary_naive",
        context=running,
        verbatim_turns=0,
        summarised_turns=len(turns),
        maintenance_calls=calls,
        maintenance_seconds=seconds,
        summary=running,
    )


def summary_guarded(turns: tuple[Turn, ...], cfg: Config) -> Memory:
    """The same architecture with one paragraph added to the prompt."""
    running, calls, seconds = _fold(turns, cfg, GUARDED_SUMMARISE)
    return Memory(
        strategy="summary_guarded",
        context=running,
        verbatim_turns=0,
        summarised_turns=len(turns),
        maintenance_calls=calls,
        maintenance_seconds=seconds,
        summary=running,
    )


def _summary_plus_window(turns: tuple[Turn, ...], cfg: Config, system: str, label: str) -> Memory:
    if len(turns) <= cfg.window:
        return full(turns, cfg)

    old, recent = turns[: -cfg.window], turns[-cfg.window :]
    running, calls, seconds = _fold(old, cfg, system)
    context = f"Summary of earlier conversation:\n{running}\n\nRecent turns:\n{transcript(recent)}"
    return Memory(
        strategy=label,
        context=context,
        verbatim_turns=len(recent),
        summarised_turns=len(old),
        maintenance_calls=calls,
        maintenance_seconds=seconds,
        summary=running,
    )


def summary_window_naive(turns: tuple[Turn, ...], cfg: Config) -> Memory:
    """The standard recommendation, with the standard prompt."""
    return _summary_plus_window(turns, cfg, NAIVE_SUMMARISE, "summary_window_naive")


def summary_window_guarded(turns: tuple[Turn, ...], cfg: Config) -> Memory:
    return _summary_plus_window(turns, cfg, GUARDED_SUMMARISE, "summary_window_guarded")


STRATEGIES = {
    "full": full,
    "window": window,
    "summary_naive": summary_naive,
    "summary_guarded": summary_guarded,
    "summary_window_naive": summary_window_naive,
    "summary_window_guarded": summary_window_guarded,
}
