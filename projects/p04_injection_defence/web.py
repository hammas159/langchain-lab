"""The UI for project 04.

Pick a defence and a payload and watch the attack run. The page shows the document exactly as
the model received it — after the defence has had its turn — beside the answer, so you can see
whether the payload was removed, ignored, or obeyed.

The pairing control is the useful one: it runs a loud payload and its quiet twin against the
same defence. Same objective, same detector, different register.

    uv run uvicorn projects.p04_injection_defence.web:app --reload --port 8104
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p04_injection_defence.chain import DEFENCES, DOCUMENT, QUESTION, answer
from projects.p04_injection_defence.payloads import ALL, BY_ID, paired
from projects.p04_injection_defence.scoring import judge
from shared.llm import Ledger, installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Injections that arrive inside the document")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "defences": DEFENCES,
        "payloads": ALL,
        "pairs": paired(),
        "models": _models(),
        "question": QUESTION,
        "document": DOCUMENT,
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p04_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    payload_id: str = Form(...),
    defence: str = Form(...),
    model: str = Form(...),
):
    # "__pairs__" runs every loud payload beside its quiet twin, which is the comparison the
    # project exists to make.
    if payload_id == "__pairs__":
        chosen = [p for pair in paired() for p in pair]
    else:
        chosen = [BY_ID[payload_id]]

    book = Ledger()
    results = []
    for payload in chosen:
        try:
            result = answer(defence, payload, model=model, ledger=book)
        except Exception as exc:
            results.append({"payload": payload, "crashed": f"{type(exc).__name__}: {exc}"})
            continue
        v = judge(payload, result)
        results.append(
            {
                "payload": payload,
                "crashed": None,
                "verdict": v,
                "answer": result.text,
                "document_shown": result.document_shown,
                "stripped_lines": result.stripped_lines,
                "seconds": result.seconds,
            }
        )

    return templates.TemplateResponse(
        request,
        "p04_index.html",
        _context(
            request,
            results=results,
            chosen_payload=payload_id,
            chosen_defence=defence,
            chosen_model=model,
        ),
    )
