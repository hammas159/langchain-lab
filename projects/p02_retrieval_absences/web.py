"""The UI for project 02.

Pick a paper and a retrieval strategy; it retrieves, extracts and classifies live, then shows
two things side by side that are usually never seen together:

  - every field, coloured by **what the model could have known** — grounded, phantom, honest
    null, or a true fabrication;
  - every passage in the paper, marked as retrieved or not, with the ones holding withheld
    evidence called out.

The second panel is the point. A phantom is invisible in the extracted JSON — it is a
well-typed value in the right field. It only becomes obvious when you can see the passage that
would have answered it sitting unretrieved a few rows down.

Run it:

    uv run uvicorn projects.p02_retrieval_absences.web:app --reload --port 8102
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p01_structured_output.schemas import LADDER
from projects.p02_retrieval_absences.benchmark import STRATEGIES
from projects.p02_retrieval_absences.papers import BY_ID, PAPERS
from projects.p02_retrieval_absences.pipeline import run
from shared.llm import installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Retrieval manufactures absences")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))

#: How each verdict should read in the UI. Keeping the wording here rather than in the template
#: means the vocabulary stays identical to the one the README and RESULTS.md use.
VERDICTS: dict[str, dict[str, str]] = {
    "grounded_correct": {"label": "grounded", "tag": "ok", "note": "evidence retrieved, correct"},
    "grounded_wrong": {"label": "wrong", "tag": "no", "note": "evidence retrieved, still wrong"},
    "phantom": {
        "label": "phantom",
        "tag": "no",
        "note": "the paper says it, retrieval did not fetch it, the model answered anyway",
    },
    "lucky_phantom": {
        "label": "lucky phantom",
        "tag": "mid",
        "note": "a guess with no evidence that happened to be right",
    },
    "honest_null": {
        "label": "honest null",
        "tag": "ok",
        "note": "evidence withheld and the model correctly declined",
    },
    "fabricated": {
        "label": "fabricated",
        "tag": "no",
        "note": "the paper is genuinely silent and the model supplied a value",
    },
    "true_null": {"label": "true null", "tag": "ok", "note": "genuinely absent, correctly null"},
    "missed": {"label": "missed", "tag": "mid", "note": "evidence was there and was not used"},
}


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "papers": PAPERS,
        "strategies": list(STRATEGIES),
        "models": _models(),
        "verdicts": VERDICTS,
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p02_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    paper_id: str = Form(...),
    strategy: str = Form("__all__"),
    model: str = Form(...),
):
    paper = BY_ID[paper_id]
    schema = LADDER["L4_constrained"]
    chosen = list(STRATEGIES) if strategy == "__all__" else [strategy]

    results = []
    for label in chosen:
        try:
            result = run(paper, schema, model=model, **STRATEGIES[label])
        except Exception as exc:
            results.append({"label": label, "crashed": f"{type(exc).__name__}: {exc}"})
            continue

        retrieved = set(result.retrieved.indices)
        # Which passages held evidence the model never saw. This is what makes a phantom
        # legible: the answer was in the document, a few rows below the cut.
        withheld_fields = set(result.retrieved.missed)
        passages = [
            {
                "i": i,
                "section": p.section,
                "text": p.text,
                "fields": p.fields,
                "retrieved": i in retrieved,
                "withheld_evidence": bool(set(p.fields) & withheld_fields),
            }
            for i, p in enumerate(paper.passages)
        ]

        results.append(
            {
                "label": label,
                "crashed": None,
                "valid": result.valid,
                "recall": result.retrieved.recall,
                "context_words": result.context_words,
                "n_passages": len(result.retrieved.indices),
                "n_total_passages": len(paper.passages),
                "n_withheld": result.n_withheld,
                "phantom_rate": result.phantom_rate,
                "n_phantoms": result.count("phantom", "lucky_phantom"),
                "seconds": result.seconds,
                "outcomes": result.outcomes,
                "passages": passages,
            }
        )

    return templates.TemplateResponse(
        request,
        "p02_index.html",
        _context(
            request,
            results=results,
            paper=paper,
            chosen_strategy=strategy,
            chosen_model=model,
        ),
    )
