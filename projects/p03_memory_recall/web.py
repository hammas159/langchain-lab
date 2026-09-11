"""The UI for project 03.

Pick a memory strategy and watch a 24-turn conversation get compressed, then see each planted
fact checked twice: is the value still in the context at all, and can the model produce it when
asked.

The two checks are shown side by side because they disagree, and the disagreement is the
project. A fact can be **present and unfindable** — sitting in the summary while the model
answers "not stated" — and no single-number memory benchmark can show you that.

    uv run uvicorn projects.p03_memory_recall.web:app --reload --port 8103
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from projects.p03_memory_recall.conversation import FACTS, TURNS, transcript
from projects.p03_memory_recall.memories import STRATEGIES, Config
from projects.p03_memory_recall.probe import ask
from shared.llm import Ledger, installed_tags, server_is_up
from shared.models import with_role

WEB = Path(__file__).parents[2] / "shared" / "web"

app = FastAPI(title="Memory that forgets the wrong thing")
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB / "templates"))


def _models() -> list[str]:
    present = installed_tags()
    return [m.tag for m in with_role("tools") if m.tag in present] or sorted(present)


def _context(request: Request, **extra: Any) -> dict:
    return {
        "request": request,
        "strategies": list(STRATEGIES),
        "models": _models(),
        "turns": TURNS,
        "n_turns": len(TURNS),
        "n_facts": len(FACTS),
        "full_words": len(transcript(TURNS).split()),
        "server_up": server_is_up(),
        **extra,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "p03_index.html", _context(request))


@app.post("/run", response_class=HTMLResponse)
def run_one(
    request: Request,
    strategy: str = Form(...),
    model: str = Form(...),
    window: int = Form(6),
):
    chosen = list(STRATEGIES) if strategy == "__all__" else [strategy]
    cfg = Config(window=window, model=model)

    results = []
    for label in chosen:
        try:
            memory = STRATEGIES[label](TURNS, cfg)
        except Exception as exc:
            results.append({"label": label, "crashed": f"{type(exc).__name__}: {exc}"})
            continue

        book = Ledger()
        probes = [ask(f, memory, model=model, ledger=book) for f in FACTS]
        results.append(
            {
                "label": label,
                "crashed": None,
                "context": memory.context,
                "words": memory.words,
                "verbatim_turns": memory.verbatim_turns,
                "summarised_turns": memory.summarised_turns,
                "maintenance_calls": memory.maintenance_calls,
                "seconds": round(memory.maintenance_seconds + book.seconds, 1),
                "survived": sum(p.survives for p in probes),
                "answered": sum(p.answered for p in probes),
                "buried": sum(p.buried for p in probes),
                "confabulated": sum(p.confabulated for p in probes),
                "n": len(probes),
                "probes": probes,
            }
        )

    return templates.TemplateResponse(
        request,
        "p03_index.html",
        _context(
            request,
            results=results,
            chosen_strategy=strategy,
            chosen_model=model,
            chosen_window=window,
        ),
    )
