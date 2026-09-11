"""Full papers assembled from labelled passages.

Project 01 handed the model a whole abstract, so every fact it needed was in front of it and a
`null` could only ever mean "the source does not state this". This project makes the document
too long to fit, puts a retriever in front of it, and that single change splits `null` into two
situations the model **cannot tell apart**:

  - the paper genuinely does not report the field, or
  - the paper reports it and retrieval did not fetch that passage.

To measure the difference you have to know where every fact lives. So papers are not written as
prose and then hoped about — they are composed from `Passage` objects, each of which declares
the ground-truth fields it contains. A passage with `fields=()` contains no answer to anything,
and most of a real paper is exactly that.

Ground truth itself is imported from project 01 rather than restated, so the two projects can
never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from projects.p01_structured_output.corpus import CORPUS, Abstract


@dataclass(frozen=True)
class Passage:
    """One retrievable unit, and the fields whose evidence it carries.

    `fields` is the instrument. It is not used to build the paper or shown to any model; it
    exists so that after retrieval runs we can ask "was the evidence for `p_value` in the
    context the model saw?" and get an answer that is true by construction rather than by
    string-matching the output.
    """

    section: str
    text: str
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    passages: tuple[Passage, ...]
    expected: dict[str, object]

    @property
    def words(self) -> int:
        return sum(len(p.text.split()) for p in self.passages)

    def passages_for(self, field: str) -> tuple[int, ...]:
        """Indices of the passages carrying evidence for `field`."""
        return tuple(i for i, p in enumerate(self.passages) if field in p.fields)

    @property
    def evidenced_fields(self) -> set[str]:
        """Fields this paper actually states somewhere."""
        return {f for p in self.passages for f in p.fields}


# --- filler ---------------------------------------------------------------------------------
#
# Filler is not padding for its own sake. A retriever only fails in an interesting way when
# there is plausible-but-irrelevant material to rank above the answer, so the filler is written
# to be topically close: it discusses the same condition, the same outcome and the same
# statistics without ever stating a value the schema asks for.

FILLER = (
    (
        "Background",
        "The condition described here imposes a substantial burden on health services, and "
        "management varies considerably between centres. Previous trials have been small, "
        "heterogeneous in their choice of endpoint, and frequently underpowered for the "
        "comparisons clinicians most want to make. Systematic reviews have repeatedly called "
        "for adequately powered randomised evidence using a consistent primary outcome.",
    ),
    (
        "Background",
        "Guideline recommendations in this area rest largely on consensus rather than on "
        "randomised data. Where randomised evidence exists it is often derived from selected "
        "populations treated in specialist centres, which limits the confidence with which it "
        "can be applied to routine practice.",
    ),
    (
        "Methods",
        "The trial was approved by the relevant research ethics committee and all participants "
        "gave written informed consent before any study procedure was undertaken. The protocol "
        "and statistical analysis plan were finalised before the first participant was "
        "enrolled, and both were made available to reviewers on request.",
    ),
    (
        "Methods",
        "Randomisation was performed centrally using a computer-generated sequence with "
        "permuted blocks of varying size, stratified by recruiting centre. Allocation was "
        "concealed from the recruiting clinician until after consent and baseline assessment "
        "were complete.",
    ),
    (
        "Methods",
        "Outcome assessors were masked to allocation throughout. Where masking of participants "
        "was not feasible given the nature of the intervention, the primary outcome was "
        "collected by an assessor with no other involvement in the trial.",
    ),
    (
        "Statistical analysis",
        "Analyses followed the intention-to-treat principle, with all randomised participants "
        "analysed in the group to which they were allocated. Missing primary outcome data were "
        "handled by multiple imputation under a missing-at-random assumption, and a complete "
        "case analysis was pre-specified as a sensitivity analysis.",
    ),
    (
        "Statistical analysis",
        "Prespecified subgroup analyses were conducted by age band, sex and baseline severity, "
        "and were interpreted as exploratory. Tests for interaction were used in preference to "
        "comparisons of within-subgroup significance, which are known to mislead.",
    ),
    (
        "Discussion",
        "The findings should be interpreted in the light of several limitations. Recruitment "
        "was confined to a small number of centres, and participants who agreed to take part "
        "may differ systematically from those who declined. Longer follow-up would be required "
        "to establish whether any difference observed is durable.",
    ),
    (
        "Discussion",
        "These results are broadly consistent with the direction of effect reported in earlier "
        "observational work, although such studies are vulnerable to confounding by indication "
        "and their effect sizes should not be compared directly with those reported here.",
    ),
    (
        "Discussion",
        "Implementation would require changes to existing care pathways, and the resource "
        "implications of doing so have not been evaluated in this trial. A formal economic "
        "evaluation is reported separately.",
    ),
    (
        "Conclusion",
        "Further research should concentrate on identifying which participants stand to benefit "
        "most, since an average effect estimated across a heterogeneous population may conceal "
        "clinically important variation.",
    ),
)


def _filler_passages() -> tuple[Passage, ...]:
    return tuple(Passage(section=s, text=t) for s, t in FILLER)


# --- the fact-bearing passages ----------------------------------------------------------------


def _fact_passages(a: Abstract) -> list[Passage]:
    """Build the passages that actually carry this trial's reportable facts.

    Each is labelled with the fields it states. Crucially, a field whose ground truth is `None`
    gets **no passage at all** — the paper really does not report it, exactly as in project 01.
    That keeps "truly absent" a real category rather than a simulated one.
    """
    e = a.expected
    out: list[Passage] = []

    out.append(
        Passage(
            section="Methods",
            text=(
                f"Eligible participants were {e['population']}. Those allocated to the "
                f"intervention arm received {e['intervention']}. The comparator arm received "
                f"{e['comparator']}. Eligibility was assessed by a clinician independent of the "
                "randomisation process."
            ),
            fields=("population", "intervention", "comparator"),
        )
    )

    if e.get("n_randomised") is not None:
        out.append(
            Passage(
                section="Participants",
                text=(
                    f"A total of {e['n_randomised']} participants were randomised between the "
                    "two arms. Recruitment proceeded over twenty-two months and stopped once "
                    "the pre-specified target had been reached. Baseline characteristics were "
                    "well balanced across the groups."
                ),
                fields=("n_randomised",),
            )
        )

    if e.get("primary_outcome") is not None:
        out.append(
            Passage(
                section="Outcomes",
                text=(
                    f"The primary outcome was {e['primary_outcome']}. Secondary outcomes were "
                    "specified in the protocol and are reported in full in the supplementary "
                    "material, together with the adverse events recorded in each arm."
                ),
                fields=("primary_outcome",),
            )
        )

    # The headline result. Effect, interval and p-value are split across separate passages on
    # purpose: they are typically reported in one sentence, and splitting them means a retriever
    # can fetch the effect while missing its uncertainty — which is the more damaging half.
    if e.get("effect") is not None:
        out.append(
            Passage(
                section="Results",
                text=(
                    f"The primary analysis gave an estimated effect of {e['effect']} in favour "
                    "of the arm indicated by the pre-specified direction of comparison. The "
                    "estimate was stable across the pre-specified sensitivity analyses."
                ),
                fields=("effect",),
            )
        )

    if e.get("interval") is not None:
        iv = e["interval"]
        out.append(
            Passage(
                section="Results",
                text=(
                    f"The {int(iv['level'] * 100)}% confidence interval for the primary "
                    f"estimate ran from {iv['low']} to {iv['high']}. Interval estimates are "
                    "reported in preference to significance statements throughout."
                ),
                fields=("interval",),
            )
        )

    if e.get("p_value") is not None:
        out.append(
            Passage(
                section="Results",
                text=(
                    "The comparison of the primary outcome between arms returned "
                    f"p={e['p_value']}. "
                    "No adjustment for multiplicity was applied to the primary comparison, "
                    "which was single and pre-specified."
                ),
                fields=("p_value",),
            )
        )

    if e.get("margin") is not None:
        out.append(
            Passage(
                section="Statistical analysis",
                text=(
                    f"This was a non-inferiority trial with a pre-specified margin of "
                    f"{e['margin']}. The margin was justified on clinical grounds during "
                    "protocol development and agreed with the trial steering committee before "
                    "recruitment opened."
                ),
                fields=("margin", "design"),
            )
        )
        out.append(
            Passage(
                section="Results",
                text=(
                    "Assessed against the pre-specified margin, non-inferiority was "
                    + ("established." if e.get("met") else "not established.")
                    + " The conclusion was unchanged in the per-protocol analysis."
                ),
                fields=("met",),
            )
        )
    else:
        out.append(
            Passage(
                section="Statistical analysis",
                text=(
                    "The trial was designed as a superiority comparison. The sample size was "
                    "calculated to detect the smallest difference considered clinically "
                    "worthwhile, with 90% power at a two-sided 5% significance level."
                ),
                fields=("design",),
            )
        )

    # Direction is stated in the conclusion, which is where a reader looks and where a retriever
    # ranked on a factual query often does not.
    direction_words = {
        "favours_intervention": "favoured the intervention arm",
        "favours_comparator": "favoured the comparator arm",
        "no_difference": "showed no meaningful difference between the arms",
    }
    out.append(
        Passage(
            section="Conclusion",
            text=(
                "Taken together, the primary analysis "
                f"{direction_words[str(e['direction'])]}. Clinicians should weigh this "
                "alongside the practical considerations discussed above."
            ),
            fields=("direction",),
        )
    )

    # Registration sits last, as it does in a real paper. It is the single most reliably missed
    # field, because nothing about a query for clinical facts ranks a registration line highly.
    if e.get("registration") is not None:
        out.append(
            Passage(
                section="Registration",
                text=(
                    f"This trial was prospectively registered. Registration number: "
                    f"{e['registration']}. The protocol is available from the corresponding "
                    "author on reasonable request."
                ),
                fields=("registration",),
            )
        )

    return out


def _interleave(facts: list[Passage], filler: tuple[Passage, ...]) -> tuple[Passage, ...]:
    """Order passages the way a paper is ordered, not the way the facts were generated.

    Grouping by section matters: it puts the filler next to the facts it resembles, so a
    retriever has to discriminate within a section rather than between obviously different
    parts of the document.
    """
    order = [
        "Background",
        "Methods",
        "Participants",
        "Outcomes",
        "Statistical analysis",
        "Results",
        "Discussion",
        "Conclusion",
        "Registration",
    ]
    combined = list(filler) + facts
    return tuple(
        sorted(combined, key=lambda p: (order.index(p.section) if p.section in order else 99,))
    )


def build_papers() -> tuple[Paper, ...]:
    papers = []
    for a in CORPUS:
        passages = _interleave(_fact_passages(a), _filler_passages())
        papers.append(
            Paper(
                id=a.id,
                title=f"Randomised trial of {a.expected['intervention']}",
                passages=passages,
                expected=a.expected,
            )
        )
    return tuple(papers)


PAPERS: tuple[Paper, ...] = build_papers()
BY_ID: dict[str, Paper] = {p.id: p for p in PAPERS}
