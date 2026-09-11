"""The UI for project 05.

Every comparison is shown **in both orders at once**, side by side, because that is the only
presentation in which a position flip is visible as a flip rather than as a verdict.

    uv run uvicorn projects.p05_judge_bias.web:app --reload --port 8105
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p05_judge_bias.answers import BY_ID, CORPUS, length_ratio
from projects.p05_judge_bias.judge import PROMPTS, compare_both_orders
from shared.llm import Ledger, installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="The judge prefers the longer answer")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "corpus": CORPUS,
        "prompts": list(PROMPTS),
        "models": _models(),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p05_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    question_id: str = Form("__all__"),
    pair_type: str = Form("tie"),
    prompt: str = Form(...),
    model: str = Form(...),
):
    questions = list(CORPUS) if question_id == "__all__" else [BY_ID[question_id]]
    book = Ledger()

    results = []
    for q in questions:
        a = q.concise
        b = q.verbose if pair_type == "tie" else q.wrong
        try:
            r = compare_both_orders(q, a, b, prompt=prompt, model=model, ledger=book)
        except Exception as exc:
            results.append({"q": q, "crashed": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(
            {
                "q": q,
                "crashed": None,
                "a": a,
                "b": b,
                "r": r,
                "ratio": round(length_ratio(q), 1),
            }
        )

    # Totals are the point on a full run: a single pair cannot show a 30-for-30 effect.
    decisive = [x for x in results if not x["crashed"] and x["r"].consistent]
    non_tie = [x for x in decisive if x["r"].forward.chose != "tie"]
    summary = {
        "n": len([x for x in results if not x["crashed"]]),
        "flipped": len([x for x in results if not x["crashed"] and x["r"].flipped]),
        "consistent": len(decisive),
        "ties": len([x for x in decisive if x["r"].forward.chose == "tie"]),
        "longer_won": len([x for x in non_tie if x["r"].forward.chose == "verbose"]),
        "non_tie": len(non_tie),
        "calls": book.n_calls,
        "seconds": round(book.seconds, 1),
    }

    return templates.TemplateResponse(
        request,
        "p05_index.html",
        _context(
            request,
            results=results,
            summary=summary,
            chosen_question=question_id,
            chosen_pair=pair_type,
            chosen_prompt=prompt,
            chosen_model=model,
        ),
    )
