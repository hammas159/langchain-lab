"""Retrieval over a paper's passages, and an honest full-context control.

A note on scale, because it decides how this project should be read. These papers are ~650
words. They **fit** in the model's context window with room to spare, so nothing here is forced
by a context limit.

That is deliberate. Retrieval is routinely put in front of documents that would fit, because
the pipeline was built for a corpus and applies the same path to every document in it. This
project measures what that choice costs when it was not necessary — which makes the
`full_context` baseline the control, not an afterthought: it is the same model, the same schema
and the same prompt, differing only in whether a retriever stood in the way.

Embeddings come from `nomic-embed-text` via ollama. Similarity is cosine, computed in plain
Python: 20 passages per paper does not justify a vector database, and a dependency that hides
the ranking makes the failure harder to see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache

from projects.p02_retrieval_absences.papers import Paper
from shared.llm import embeddings


@dataclass
class Retrieved:
    """What retrieval returned, and — the point of the project — what it left behind."""

    paper_id: str
    query: str
    k: int
    indices: tuple[int, ...]
    scores: tuple[float, ...]
    #: Fields whose evidence is in the retrieved passages.
    covered: frozenset[str] = field(default_factory=frozenset)
    #: Fields the paper states but retrieval did not fetch. These are the manufactured
    #: absences: the model will be asked for them with no evidence in front of it.
    missed: frozenset[str] = field(default_factory=frozenset)

    @property
    def recall(self) -> float:
        total = len(self.covered) + len(self.missed)
        return len(self.covered) / total if total else 1.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@cache
def _embed_paper(paper_id: str, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """Embed a paper's passages once and keep them.

    Cached on the text tuple rather than the id alone so that editing a paper invalidates the
    cache — an embedding cache keyed only on an id is a quiet source of stale results.
    """
    vectors = embeddings().embed_documents(list(texts))
    return tuple(tuple(v) for v in vectors)


def retrieve(paper: Paper, query: str, k: int = 4) -> Retrieved:
    """Top-`k` passages by cosine similarity, with coverage recorded against ground truth."""
    texts = tuple(p.text for p in paper.passages)
    vectors = _embed_paper(paper.id, texts)
    q = embeddings().embed_query(query)

    ranked = sorted(
        ((cosine(list(v), q), i) for i, v in enumerate(vectors)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top = ranked[:k]
    indices = tuple(i for _, i in top)

    covered: set[str] = set()
    for i in indices:
        covered.update(paper.passages[i].fields)

    return Retrieved(
        paper_id=paper.id,
        query=query,
        k=k,
        indices=indices,
        scores=tuple(round(s, 4) for s, _ in top),
        covered=frozenset(covered),
        missed=frozenset(paper.evidenced_fields - covered),
    )


def full_context(paper: Paper) -> Retrieved:
    """The control: every passage, in order. Retrieval recall is 1.0 by construction."""
    indices = tuple(range(len(paper.passages)))
    return Retrieved(
        paper_id=paper.id,
        query="",
        k=len(indices),
        indices=indices,
        scores=tuple(1.0 for _ in indices),
        covered=frozenset(paper.evidenced_fields),
        missed=frozenset(),
    )


def context_text(paper: Paper, selection: Retrieved) -> str:
    """Render the selected passages as the context the model actually sees.

    Passages are emitted in document order rather than relevance order. Relevance order is
    what a naive pipeline does and it scrambles the narrative, which is a second, separate
    problem; keeping document order here means the only variable under test is *which*
    passages were selected.
    """
    ordered = sorted(selection.indices)
    return "\n\n".join(f"[{paper.passages[i].section}] {paper.passages[i].text}" for i in ordered)


#: The query a real extraction pipeline writes: the schema's field names, joined. This is the
#: naive strategy, and it is naive in a specific, measurable way — see `PER_FIELD_QUERIES`.
EXTRACTION_QUERY = (
    "trial population, intervention, comparator, number randomised, primary outcome, "
    "effect size, confidence interval, p-value, trial registration number"
)

#: One query per field, phrased as *the sentence the answer would appear in* rather than as the
#: field's name.
#:
#: This is the fix, and the reason it is needed is the finding. A query built from schema
#: vocabulary ("population, intervention, comparator") and a document written in world
#: vocabulary ("adults with chronic tension headache were randomised to exercise therapy") do
#: not land near each other in embedding space. On `t01` the single passage carrying all three
#: of those fields ranks **18th of 20**, below every piece of generic filler, because generic
#: methodological prose is what abstract field names actually resemble.
PER_FIELD_QUERIES: dict[str, str] = {
    "population": "Eligible participants were adults with the condition under study.",
    "intervention": "Participants allocated to the intervention arm received the treatment.",
    "comparator": "The comparator arm received the control treatment or usual care.",
    "n_randomised": "A total of participants were randomised between the two arms.",
    "primary_outcome": "The primary outcome was measured at the end of follow-up.",
    "effect": "The primary analysis gave an estimated effect of this size.",
    "interval": "The 95% confidence interval for the estimate ran from one value to another.",
    "p_value": "The comparison between arms returned a p-value of.",
    "registration": "This trial was prospectively registered. Registration number NCT.",
    "direction": "Taken together the primary analysis favoured one of the arms.",
    "design": "The trial was designed as a superiority or non-inferiority comparison.",
    "margin": "The pre-specified non-inferiority margin was set at this value.",
    "met": "Assessed against the margin, non-inferiority was or was not established.",
}


def retrieve_per_field(paper: Paper, fields: tuple[str, ...], per_field_k: int = 1) -> Retrieved:
    """Run one query per field and union the results.

    Costs one embedding call per field instead of one per document, and that trade is the
    point: the naive strategy is cheap and loses evidence, this one is dearer and keeps it.
    Reporting both is what makes the comparison honest.
    """
    texts = tuple(p.text for p in paper.passages)
    vectors = _embed_paper(paper.id, texts)

    chosen: dict[int, float] = {}
    for name in fields:
        query = PER_FIELD_QUERIES.get(name)
        if query is None:
            continue
        q = embeddings().embed_query(query)
        ranked = sorted(
            ((cosine(list(v), q), i) for i, v in enumerate(vectors)),
            key=lambda pair: pair[0],
            reverse=True,
        )
        for score, i in ranked[:per_field_k]:
            chosen[i] = max(chosen.get(i, 0.0), score)

    indices = tuple(sorted(chosen, key=lambda i: chosen[i], reverse=True))
    covered: set[str] = set()
    for i in indices:
        covered.update(paper.passages[i].fields)

    return Retrieved(
        paper_id=paper.id,
        query=f"per-field x{len(fields)}",
        k=len(indices),
        indices=indices,
        scores=tuple(round(chosen[i], 4) for i in indices),
        covered=frozenset(covered),
        missed=frozenset(paper.evidenced_fields - covered),
    )
