"""The UI for project 01.

One page. Choose an abstract, a schema level and a strategy; it runs the extraction live
against ollama and shows the object it got back next to the ground truth, with every
disagreement marked and fabrications marked differently from everything else.

Run it:

    uv run uvicorn projects.p01_structured_output.web:app --reload

The most informative thing to run is `t09` at `L5_union`: three strategies fail to produce a
valid document at all, and `constrained` returns one that is valid and partly wrong. That is
the whole finding on one screen.

Note that a failed extraction is scored against the fields it *owed*, so it reads 0% rather
than disappearing from the average — which is the bug that made the first version of this
project reach the opposite conclusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p01_structured_output.corpus import BY_ID, CORPUS
from projects.p01_structured_output.schemas import LADDER
from projects.p01_structured_output.scoring import score
from projects.p01_structured_output.strategies import STRATEGIES
from shared.llm import Ledger, installed_tags, server_is_up
from shared.models import default_chat_model, with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Structured output under pressure")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    """Tool-capable models actually present on this machine, best-known first."""
    present = installed_tags()
    tags = [m.tag for m in with_role("tools") if m.tag in present]
    return tags or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "corpus": CORPUS,
        "levels": list(LADDER),
        "strategies": list(STRATEGIES),
        "models": _models(),
        "default_model": default_chat_model(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p01_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run(
    request: Request,
    abstract_id: str = Form(...),
    level: str = Form(...),
    model: str = Form(...),
    strategy: str = Form("__all__"),
):
    abstract = BY_ID[abstract_id]
    schema = LADDER[level]
    chosen = list(STRATEGIES) if strategy == "__all__" else [strategy]

    results = []
    for name in chosen:
        book = Ledger()
        error: str | None = None
        try:
            extraction = STRATEGIES[name](schema, abstract.text, model=model, callbacks=[book])
        except Exception as exc:
            results.append(
                {
                    "strategy": name,
                    "crashed": f"{type(exc).__name__}: {exc}",
                    "fields": [],
                    "valid": False,
                }
            )
            continue

        # The schema is passed so a failed extraction is charged for the fields it owed,
        # exactly as the benchmark does. Without it the card reads "0/0 fields", which is
        # both meaningless and a different metric from the one in RESULTS.md.
        s = score(extraction.value, abstract, schema)
        results.append(
            {
                "strategy": name,
                "valid": extraction.ok,
                "error": extraction.last_error,
                "attempts": extraction.n_attempts,
                "calls": book.n_calls,
                "seconds": round(book.seconds, 2),
                "tokens": book.tokens,
                "accuracy": s.accuracy,
                "n_correct": s.n_correct,
                "n_fields": s.n_fields,
                "n_fabricated": s.n_fabricated,
                "n_nullable": s.n_nullable_truth,
                "n_omitted": s.n_omitted,
                "fields": s.fields,
                "raw": extraction.attempts[-1].raw if extraction.attempts else "",
                "crashed": error,
                "json": json.dumps(extraction.value.model_dump(mode="json"), indent=2, default=str)
                if extraction.value
                else "",
            }
        )

    return templates.TemplateResponse(
        request,
        "p01_index.html",
        _context(
            request,
            results=results,
            abstract=abstract,
            chosen_level=level,
            chosen_model=model,
            chosen_strategy=strategy,
        ),
    )
