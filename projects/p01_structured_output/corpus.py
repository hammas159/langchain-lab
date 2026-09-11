"""Twelve synthetic trial abstracts, with hand-written ground truth.

**These are not real trials.** Every abstract here was written for this repo, and the
registration IDs are invented. Real abstracts were not used for two reasons: their ground
truth would itself be a judgement call, and publishing extracted claims about real trials
attached to a toy model is a good way to put a wrong medical number on the internet.

What makes the corpus useful is not realism, it is the pattern of *omissions*. The corpus is
built so that:

  - four abstracts omit the participant count,
  - six omit any confidence interval,
  - five omit a p-value,
  - five have no registration ID,

and in every one of those cases the correct extracted value is `None`. `expected` records
that. A model that returns a number there is scored as wrong, not as "close".

Two abstracts (`t09`, `t10`) are non-inferiority designs and the rest are superiority; that
split is what the L5 discriminated union is measured on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Abstract:
    id: str
    text: str
    #: Ground truth, keyed by field name. Fields absent from a given schema level are ignored
    #: when scoring that level, so one dictionary serves all five.
    expected: dict[str, object]


CORPUS: tuple[Abstract, ...] = (
    Abstract(
        id="t01",
        text=(
            "Background: Chronic tension headache is commonly managed with amitriptyline. "
            "Methods: We randomised 412 adults with chronic tension headache to either "
            "structured exercise therapy or amitriptyline 25 mg nightly for 16 weeks. The "
            "primary outcome was headache days per month. Results: Exercise therapy reduced "
            "headache days by 4.1 compared with 2.3 for amitriptyline (difference 1.8 days, "
            "95% CI 0.9 to 2.7, p=0.001). Conclusion: Exercise therapy was superior. "
            "Registered as NCT10000001."
        ),
        expected={
            "population": "adults with chronic tension headache",
            "intervention": "structured exercise therapy",
            "comparator": "amitriptyline 25 mg nightly",
            "direction": "favours_intervention",
            "n_randomised": 412,
            "primary_outcome": "headache days per month",
            "effect": 1.8,
            "interval": {"low": 0.9, "high": 2.7, "level": 0.95},
            "p_value": 0.001,
            "registration": "NCT10000001",
            "design": "superiority",
        },
    ),
    Abstract(
        id="t02",
        text=(
            "Methods: Adults hospitalised with community-acquired pneumonia were randomised "
            "to a five-day or ten-day course of oral antibiotics. The primary outcome was "
            "clinical cure at day 30. Results: Clinical cure occurred in 88.1% of the "
            "five-day group and 87.4% of the ten-day group. The difference was not "
            "statistically significant. Conclusion: A shorter course appears comparable."
        ),
        # No participant count, no interval, no p-value, no registration. All four null.
        expected={
            "population": "adults hospitalised with community-acquired pneumonia",
            "intervention": "five-day course of oral antibiotics",
            "comparator": "ten-day course of oral antibiotics",
            "direction": "no_difference",
            "n_randomised": None,
            "primary_outcome": "clinical cure at day 30",
            "effect": None,
            "interval": None,
            "p_value": None,
            "registration": None,
            "design": "superiority",
        },
    ),
    Abstract(
        id="t03",
        text=(
            "We enrolled 1,204 children aged 6 to 11 years with mild persistent asthma and "
            "randomised them to a digital inhaler-adherence application or to usual care. "
            "Over 12 months, exacerbations per child per year were 0.71 with the application "
            "and 0.94 with usual care (rate ratio 0.76, 95% CI 0.62 to 0.93, p=0.008). "
            "Secondary outcomes included school days missed and rescue-inhaler use. "
            "Trial registration: NCT10000003."
        ),
        expected={
            "population": "children aged 6 to 11 years with mild persistent asthma",
            "intervention": "digital inhaler-adherence application",
            "comparator": "usual care",
            "direction": "favours_intervention",
            "n_randomised": 1204,
            "primary_outcome": "exacerbations per child per year",
            "effect": 0.76,
            "interval": {"low": 0.62, "high": 0.93, "level": 0.95},
            "p_value": 0.008,
            "registration": "NCT10000003",
            "design": "superiority",
        },
    ),
    Abstract(
        id="t04",
        text=(
            "Objective: To compare telephone-delivered cognitive behavioural therapy with "
            "in-person cognitive behavioural therapy for adults with insomnia. Methods: 268 "
            "adults were randomised. Primary outcome: Insomnia Severity Index at 6 months. "
            "Results: Mean ISI was 9.2 in the telephone group and 8.8 in the in-person group "
            "(difference 0.4, 95% CI -0.9 to 1.7). Conclusion: The two delivery modes "
            "produced similar outcomes."
        ),
        # An interval is reported but no p-value, and no registration.
        expected={
            "population": "adults with insomnia",
            "intervention": "telephone-delivered cognitive behavioural therapy",
            "comparator": "in-person cognitive behavioural therapy",
            "direction": "no_difference",
            "n_randomised": 268,
            "primary_outcome": "Insomnia Severity Index at 6 months",
            "effect": 0.4,
            "interval": {"low": -0.9, "high": 1.7, "level": 0.95},
            "p_value": None,
            "registration": None,
            "design": "superiority",
        },
    ),
    Abstract(
        id="t05",
        text=(
            "In this trial, patients with type 2 diabetes and a body mass index above 30 were "
            "randomised to a structured low-carbohydrate diet or to standard dietary advice. "
            "HbA1c fell by 0.9 percentage points in the low-carbohydrate arm and by 1.2 "
            "points with standard advice (p=0.03), favouring standard advice. Registration: "
            "NCT10000005."
        ),
        expected={
            "population": "patients with type 2 diabetes and body mass index above 30",
            "intervention": "structured low-carbohydrate diet",
            "comparator": "standard dietary advice",
            "direction": "favours_comparator",
            "n_randomised": None,
            "primary_outcome": "HbA1c",
            "effect": 0.3,
            "interval": None,
            "p_value": 0.03,
            "registration": "NCT10000005",
            "design": "superiority",
        },
    ),
    Abstract(
        id="t06",
        text=(
            "Methods: 96 adults with treatment-resistant hypertension were randomised to "
            "renal denervation or a sham procedure. The primary outcome was 24-hour "
            "ambulatory systolic blood pressure at 6 months. Results: Systolic pressure fell "
            "by 6.8 mmHg after denervation and by 3.1 mmHg after sham (difference 3.7 mmHg, "
            "95% CI 0.4 to 7.0, p=0.03). Secondary outcomes were office systolic pressure and "
            "medication burden. Registered NCT10000006."
        ),
        expected={
            "population": "adults with treatment-resistant hypertension",
            "intervention": "renal denervation",
            "comparator": "sham procedure",
            "direction": "favours_intervention",
            "n_randomised": 96,
            "primary_outcome": "24-hour ambulatory systolic blood pressure at 6 months",
            "effect": 3.7,
            "interval": {"low": 0.4, "high": 7.0, "level": 0.95},
            "p_value": 0.03,
            "registration": "NCT10000006",
            "design": "superiority",
        },
    ),
    Abstract(
        id="t07",
        text=(
            "Pregnant women at high risk of pre-eclampsia were randomised to low-dose aspirin "
            "or placebo from 12 weeks' gestation. Pre-eclampsia occurred in 4.1% versus 6.9% "
            "of pregnancies. The trial was stopped early for benefit."
        ),
        # Everything numeric beyond the two percentages is absent.
        expected={
            "population": "pregnant women at high risk of pre-eclampsia",
            "intervention": "low-dose aspirin from 12 weeks' gestation",
            "comparator": "placebo",
            "direction": "favours_intervention",
            "n_randomised": None,
            "primary_outcome": "pre-eclampsia",
            "effect": None,
            "interval": None,
            "p_value": None,
            "registration": None,
            "design": "superiority",
        },
    ),
    Abstract(
        id="t08",
        text=(
            "We randomised 2,310 older adults living alone to a weekly volunteer telephone "
            "call or to no contact, and measured UCLA Loneliness Scale score at 12 weeks, "
            "depressive symptoms, and self-rated health. Loneliness scores were 38.2 and 41.0 "
            "respectively (difference 2.8, 95% CI 1.6 to 4.0, p<0.001). NCT10000008."
        ),
        expected={
            "population": "older adults living alone",
            "intervention": "weekly volunteer telephone call",
            "comparator": "no contact",
            "direction": "favours_intervention",
            "n_randomised": 2310,
            "primary_outcome": "UCLA Loneliness Scale score at 12 weeks",
            "effect": 2.8,
            "interval": {"low": 1.6, "high": 4.0, "level": 0.95},
            "p_value": 0.001,
            "registration": "NCT10000008",
            "design": "superiority",
        },
    ),
    Abstract(
        id="t09",
        text=(
            "This non-inferiority trial randomised 640 adults with uncomplicated appendicitis "
            "to antibiotics alone or to appendicectomy, with a pre-specified non-inferiority "
            "margin of 10 percentage points for treatment success at one year. Success was "
            "achieved in 72.1% of the antibiotic group and 97.8% of the surgical group. The "
            "lower bound of the confidence interval crossed the margin. Conclusion: "
            "Non-inferiority was not established. NCT10000009."
        ),
        expected={
            "population": "adults with uncomplicated appendicitis",
            "intervention": "antibiotics alone",
            "comparator": "appendicectomy",
            "direction": "favours_comparator",
            "n_randomised": 640,
            "primary_outcome": "treatment success at one year",
            "margin": 10.0,
            "met": False,
            "registration": "NCT10000009",
            "design": "non_inferiority",
        },
    ),
    Abstract(
        id="t10",
        text=(
            "Adults requiring long-term anticoagulation were randomised to a once-daily oral "
            "agent or to warfarin in a non-inferiority design with a margin of 1.38 for the "
            "hazard ratio of stroke or systemic embolism. The observed hazard ratio was 0.89 "
            "and the upper confidence bound lay below the margin, establishing "
            "non-inferiority."
        ),
        expected={
            "population": "adults requiring long-term anticoagulation",
            "intervention": "once-daily oral anticoagulant",
            "comparator": "warfarin",
            "direction": "no_difference",
            "n_randomised": None,
            "primary_outcome": "stroke or systemic embolism",
            "margin": 1.38,
            "met": True,
            "registration": None,
            "design": "non_inferiority",
        },
    ),
    Abstract(
        id="t11",
        text=(
            "Methods: 148 patients undergoing elective knee arthroplasty were randomised to a "
            "prehabilitation programme or to standard preoperative care. Outcomes were length "
            "of stay, Oxford Knee Score at 6 weeks, and 90-day readmission. Results: Length "
            "of stay was 3.1 days versus 3.4 days (p=0.21); Oxford Knee Score was 34.2 versus "
            "33.8 (p=0.44); readmission was 5.4% versus 6.8% (p=0.71). Conclusion: "
            "Prehabilitation did not change any measured outcome."
        ),
        # Three outcomes: this is the abstract the L3 list is really testing.
        expected={
            "population": "patients undergoing elective knee arthroplasty",
            "intervention": "prehabilitation programme",
            "comparator": "standard preoperative care",
            "direction": "no_difference",
            "n_randomised": 148,
            "primary_outcome": "length of stay",
            "effect": 0.3,
            "interval": None,
            "p_value": 0.21,
            "registration": None,
            "n_outcomes": 3,
            "design": "superiority",
        },
    ),
    Abstract(
        id="t12",
        text=(
            "A cluster-randomised trial across 24 primary schools assigned classrooms to a "
            "daily mindfulness curriculum or to the standard personal-development curriculum. "
            "Among 3,150 pupils, the Strengths and Difficulties Questionnaire total score at "
            "one year was 11.4 with mindfulness and 11.2 with standard teaching (difference "
            "-0.2, 95% CI -0.8 to 0.4, p=0.52). Registration NCT10000012."
        ),
        expected={
            "population": "primary school pupils",
            "intervention": "daily mindfulness curriculum",
            "comparator": "standard personal-development curriculum",
            "direction": "no_difference",
            "n_randomised": 3150,
            "primary_outcome": "Strengths and Difficulties Questionnaire total score at one year",
            "effect": -0.2,
            "interval": {"low": -0.8, "high": 0.4, "level": 0.95},
            "p_value": 0.52,
            "registration": "NCT10000012",
            "design": "superiority",
        },
    ),
)

BY_ID = {a.id: a for a in CORPUS}


def omission_counts() -> dict[str, int]:
    """How many abstracts genuinely omit each nullable field.

    Asserted in the tests. If someone edits the corpus and the balance of omissions shifts,
    every "fabrication rate" in the README silently changes meaning, so it is pinned.
    """
    fields = ("n_randomised", "interval", "p_value", "registration")
    return {f: sum(1 for a in CORPUS if a.expected.get(f, None) is None) for f in fields}
