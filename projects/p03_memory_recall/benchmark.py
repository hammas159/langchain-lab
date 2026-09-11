"""Run every memory strategy over the conversation and write RESULTS.md.

    uv run python -m projects.p03_memory_recall.benchmark

Reports survival and recall broken down by **fact kind** and by **position in the
conversation**, because the pooled average is the one number that hides the result: strategies
do not lose facts uniformly, and which facts they lose is the whole point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from projects.p03_memory_recall.conversation import (
    FACTS,
    KINDS,
    TURNS,
    Fact,
    position,
)
from projects.p03_memory_recall.memories import STRATEGIES, Config
from projects.p03_memory_recall.probe import Probe, ask
from shared.llm import Ledger, installed_tags, server_is_up
from shared.models import default_chat_model

HERE = Path(__file__).parent
RESULTS = HERE / "RESULTS.md"
RAW = HERE / "results.json"


@dataclass
class Row:
    strategy: str
    context_words: int
    verbatim_turns: int
    summarised_turns: int
    maintenance_calls: int
    probe_calls: int
    seconds: float
    n: int
    survived: int
    answered: int
    buried: int
    confabulated: int
    declined: int
    by_kind_survived: dict = field(default_factory=dict)
    by_kind_n: dict = field(default_factory=dict)
    by_half_survived: dict = field(default_factory=dict)
    by_half_n: dict = field(default_factory=dict)
    summary: str = ""
    #: Every probe, verbatim. Kept because the aggregate cannot show *how* a strategy failed,
    #: and in this project the how is the finding: a value sitting in the context while the
    #: model answers "not stated" looks identical to a lost value in any pooled number.
    replies: list = field(default_factory=list)

    @property
    def survival(self) -> float:
        return self.survived / self.n if self.n else 0.0

    @property
    def recall(self) -> float:
        return self.answered / self.n if self.n else 0.0


def summarise(strategy: str, memory, probes: list[Probe], probe_book: Ledger) -> Row:
    by_kind_s: dict[str, int] = defaultdict(int)
    by_kind_n: dict[str, int] = defaultdict(int)
    by_half_s: dict[str, int] = defaultdict(int)
    by_half_n: dict[str, int] = defaultdict(int)

    for p in probes:
        by_kind_n[p.fact.kind] += 1
        by_kind_s[p.fact.kind] += int(p.survives)
        half = "first" if position(p.fact, len(TURNS)) < 0.5 else "second"
        by_half_n[half] += 1
        by_half_s[half] += int(p.survives)

    return Row(
        strategy=strategy,
        context_words=memory.words,
        verbatim_turns=memory.verbatim_turns,
        summarised_turns=memory.summarised_turns,
        maintenance_calls=memory.maintenance_calls,
        probe_calls=probe_book.n_calls,
        seconds=round(memory.maintenance_seconds + probe_book.seconds, 1),
        n=len(probes),
        survived=sum(p.survives for p in probes),
        answered=sum(p.answered for p in probes),
        buried=sum(p.buried for p in probes),
        confabulated=sum(p.confabulated for p in probes),
        declined=sum(p.declined for p in probes),
        by_kind_survived=dict(by_kind_s),
        by_kind_n=dict(by_kind_n),
        by_half_survived=dict(by_half_s),
        by_half_n=dict(by_half_n),
        summary=memory.summary,
        replies=[
            {
                "fact": p.fact.id,
                "kind": p.fact.kind,
                "want": p.fact.answer,
                "reply": p.reply,
                "survives": p.survives,
                "answered": p.answered,
                "buried": p.buried,
            }
            for p in probes
        ],
    )


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


@dataclass
class Repeated:
    """A strategy's results across repeats.

    Repeats exist because the first two runs of this benchmark disagreed sharply: the naive
    summariser produced a 191-word summary preserving every value on one run and a 92-word
    summary preserving half on the next. Reporting either single run would have been a claim
    about one sample of a noisy process. The spread is the result worth reporting.
    """

    strategy: str
    rows: list[Row] = field(default_factory=list)

    def _series(self, attr: str) -> list[float]:
        return [getattr(r, attr) for r in self.rows]

    def mean(self, attr: str) -> float:
        series = self._series(attr)
        return sum(series) / len(series) if series else 0.0

    def lo(self, attr: str) -> float:
        return min(self._series(attr), default=0.0)

    def hi(self, attr: str) -> float:
        return max(self._series(attr), default=0.0)

    def band(self, attr: str, as_pct: bool = True) -> str:
        """mean, and the range across repeats when it is not a single point."""
        scale = 100 if as_pct else 1
        suffix = "%" if as_pct else ""
        m, lo, hi = self.mean(attr) * scale, self.lo(attr) * scale, self.hi(attr) * scale
        if abs(hi - lo) < 0.5:
            return f"{m:.0f}{suffix}"
        return f"{m:.0f}{suffix} ({lo:.0f}–{hi:.0f})"


def render(groups: list[Repeated], elapsed: float, facts: tuple[Fact, ...], repeats: int) -> str:
    out = ["# Results\n"]
    out.append("Generated by `projects/p03_memory_recall/benchmark.py`. Do not edit.\n")
    out.append(
        f"{len(groups)} strategies over a {len(TURNS)}-turn conversation carrying "
        f"{len(facts)} planted facts, {repeats} repeats each. {elapsed / 60:.1f} minutes.\n"
    )
    out.append(
        "\nFigures are the mean across repeats, with the range in brackets where runs "
        "disagreed. Temperature is 0 throughout; the variation is in what the summariser "
        "writes, not in how it is sampled.\n"
    )

    out.append("\n## Survival and recall\n")
    out.append(
        "`survived` is a deterministic string check: is the value still anywhere in the "
        "context. `recalled` is the model answering the question from that context. The gap "
        "between them is the value being present and unfindable.\n"
    )
    out.append("| strategy | context | survived | recalled | buried | upkeep calls |")
    out.append("|---|---|---|---|---|---|")
    for g in groups:
        out.append(
            f"| `{g.strategy}` | {g.band('context_words', as_pct=False)} w "
            f"| {g.band('survival')} | {g.band('recall')} "
            f"| {g.band('buried', as_pct=False)} | {g.rows[0].maintenance_calls} |"
        )

    out.append("\n## Survival by fact kind\n")
    out.append("Pooled over repeats. This is the table the headline average hides.\n")
    out.append("| strategy | " + " | ".join(KINDS) + " |")
    out.append("|---" * (len(KINDS) + 1) + "|")
    for g in groups:
        cells = []
        for k in KINDS:
            s = sum(r.by_kind_survived.get(k, 0) for r in g.rows)
            n = sum(r.by_kind_n.get(k, 0) for r in g.rows)
            cells.append(_pct(s, n))
        out.append(f"| `{g.strategy}` | " + " | ".join(cells) + " |")

    out.append("\n## Survival by position in the conversation\n")
    out.append("| strategy | first half | second half |")
    out.append("|---|---|---|")
    for g in groups:
        fs = sum(r.by_half_survived.get("first", 0) for r in g.rows)
        fn = sum(r.by_half_n.get("first", 0) for r in g.rows)
        ss = sum(r.by_half_survived.get("second", 0) for r in g.rows)
        sn = sum(r.by_half_n.get("second", 0) for r in g.rows)
        out.append(f"| `{g.strategy}` | {_pct(fs, fn)} | {_pct(ss, sn)} |")

    out.append("\n## Buried: the value was in the context and the model did not produce it\n")
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for g in groups:
        for r in g.rows:
            for rep in r.replies:
                key = (g.strategy, rep["fact"])
                if rep["buried"] and key not in seen:
                    seen.add(key)
                    lines.append(
                        f"| `{g.strategy}` | {rep['fact']} | {rep['want']} | "
                        f"{rep['reply'].replace('|', '/')[:60]} |"
                    )
    if lines:
        out.append("| strategy | fact | in the context | the model said |")
        out.append("|---|---|---|---|")
        out.extend(lines)
    else:
        out.append("None in this run.\n")

    out.append("\n## A summary from each summarising strategy\n")
    out.append("First repeat, verbatim.\n")
    for g in groups:
        if g.rows and g.rows[0].summary:
            out.append(f"\n**`{g.strategy}`** ({g.rows[0].context_words} words)\n")
            out.append("```")
            out.append(g.rows[0].summary)
            out.append("```")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="runs per strategy; the naive summariser is not stable across them",
    )
    args = parser.parse_args()

    if not server_is_up():
        print("ollama is not reachable.", file=sys.stderr)
        return 2
    model = args.model or default_chat_model()
    if model not in installed_tags():
        print(f"{model} is not pulled.", file=sys.stderr)
        return 2

    cfg = Config(window=args.window, every=args.every, model=model)
    started = time.perf_counter()
    groups: list[Repeated] = []

    for label, build in STRATEGIES.items():
        group = Repeated(strategy=label)
        # Deterministic strategies are run once. Repeating `full` five times measures the
        # probe's noise, not the strategy's, and pads the run for nothing.
        n = 1 if label in ("full", "window") else args.repeats
        for i in range(n):
            print(f"  {label} [{i + 1}/{n}]", flush=True)
            memory = build(TURNS, cfg)
            book = Ledger()
            probes = [ask(f, memory, model=model, ledger=book) for f in FACTS]
            group.rows.append(summarise(label, memory, probes, book))
        groups.append(group)

    elapsed = time.perf_counter() - started
    RAW.write_text(
        json.dumps({g.strategy: [asdict(r) for r in g.rows] for g in groups}, indent=2),
        encoding="utf-8",
    )
    RESULTS.write_text(render(groups, elapsed, FACTS, args.repeats), encoding="utf-8")
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
