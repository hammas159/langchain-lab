"""Four ways to get a schema-shaped object out of a model, in increasing order of force.

The ordering matters, because the results do not improve monotonically along it. Each
strategy fixes the failure mode of the one before it, and the last one fixes the failure
mode so thoroughly that it introduces a worse one.

  prompt_only      Ask for JSON. Parse what comes back.
  repaired         Ask for JSON. Repair the common damage (fences, trailing commas, single
                   quotes, prose either side) before parsing.
  error_feedback   Ask for JSON; on a validation error, hand the model its own errors and let
                   it try again, up to `max_attempts`.
  constrained      Pass the JSON Schema to ollama as `format`, so decoding is grammar-
                   constrained and the output *cannot* be invalid.

`constrained` reaches 100% schema validity by construction. The reason this project exists is
that its accuracy on nullable fields goes *down*: a grammar that permits `null` does not make
`null` likely, and a model that would otherwise have produced malformed output admitting it
found nothing will instead produce clean, well-typed, invented values.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from shared.llm import chat

SYSTEM = (
    "You extract structured data from clinical trial abstracts.\n"
    "Return a single JSON object and nothing else.\n"
    "If the abstract does not report a value, use null. Do not guess, infer, or compute a "
    "value that is not stated. A null is a correct answer; an invented number is not."
)


@dataclass
class Attempt:
    """One round trip, kept even when it failed. The failures are the data."""

    raw: str
    parsed: dict[str, Any] | None = None
    error: str | None = None
    repaired: bool = False


@dataclass
class Extraction:
    strategy: str
    model: str
    level: str
    abstract_id: str
    attempts: list[Attempt] = field(default_factory=list)
    value: BaseModel | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def last_error(self) -> str | None:
        return self.attempts[-1].error if self.attempts else None


# --- repair -----------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_blob(text: str) -> str | None:
    """Pull the most plausible JSON object out of arbitrary model output.

    Tries, in order: a fenced block, then the longest balanced `{...}` span. The balanced scan
    is not a regex because nested objects are not a regular language, and the naive
    `\\{.*\\}` version silently truncates at the first closing brace inside a nested field.
    """
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


_TRAILING_COMMA = re.compile(r",\s*([}\]])")

_LITERALS = {"None": "null", "True": "true", "False": "false"}
_WORD = re.compile(r"\b(None|True|False)\b")


def _replace_python_literals(text: str) -> str:
    """Turn Python literals into JSON ones, but only outside string values.

    A word-boundary regex over the whole document is the obvious implementation and it is
    wrong: an extracted field legitimately containing the text "None of the above" comes back
    as "null of the above". The corpus is medical, where mangling a string is not cosmetic, so
    the scan tracks whether it is inside a string and leaves those spans alone.
    """
    out: list[str] = []
    i = 0
    in_string = False
    escaped = False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        match = _WORD.match(text, i)
        if match:
            out.append(_LITERALS[match.group(1)])
            i = match.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def repair_json(text: str) -> tuple[str, bool]:
    """Apply the small set of fixes that account for most parse failures.

    Returns the repaired text and whether anything was changed, because "how often did repair
    have to do something" is a reportable number and a repair that silently always runs is
    indistinguishable from a parser that is too lenient.
    """
    original = text
    blob = extract_json_blob(text)
    if blob is not None:
        text = blob

    text = _TRAILING_COMMA.sub(r"\1", text)
    text = text.replace("“", '"').replace("”", '"')
    text = _replace_python_literals(text)
    return text, text != original


# --- prompting --------------------------------------------------------------------------


def build_prompt(schema: type[BaseModel], abstract: str) -> list:
    spec = json.dumps(schema.model_json_schema(), indent=2)
    return [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"JSON Schema the object must satisfy:\n{spec}\n\n"
                f"Abstract:\n{abstract}\n\n"
                "JSON object:"
            )
        ),
    ]


def _validate(
    schema: type[BaseModel], text: str
) -> tuple[BaseModel | None, dict | None, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, None, f"json: {exc.msg} at position {exc.pos}"
    if not isinstance(data, dict):
        return None, None, f"json: top level is {type(data).__name__}, expected object"
    try:
        return schema.model_validate(data), data, None
    except ValidationError as exc:
        parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6]]
        return None, data, "schema: " + "; ".join(parts)


# --- the four strategies ------------------------------------------------------------------


def prompt_only(
    schema: type[BaseModel],
    abstract: str,
    *,
    model: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Extraction:
    llm = chat(model, callbacks=callbacks)
    raw = llm.invoke(build_prompt(schema, abstract)).content
    value, _, error = _validate(schema, raw)
    return Extraction(
        strategy="prompt_only",
        model=model,
        level=schema.__name__,
        abstract_id="",
        attempts=[Attempt(raw=raw, error=error)],
        value=value,
    )


def repaired(
    schema: type[BaseModel],
    abstract: str,
    *,
    model: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Extraction:
    llm = chat(model, callbacks=callbacks)
    raw = llm.invoke(build_prompt(schema, abstract)).content
    fixed, changed = repair_json(raw)
    value, _, error = _validate(schema, fixed)
    return Extraction(
        strategy="repaired",
        model=model,
        level=schema.__name__,
        abstract_id="",
        attempts=[Attempt(raw=raw, error=error, repaired=changed)],
        value=value,
    )


def error_feedback(
    schema: type[BaseModel],
    abstract: str,
    *,
    model: str,
    max_attempts: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Extraction:
    """Repair, then hand the model its own validation errors and ask again."""
    llm = chat(model, callbacks=callbacks)
    messages = build_prompt(schema, abstract)
    result = Extraction(
        strategy="error_feedback", model=model, level=schema.__name__, abstract_id=""
    )

    for _ in range(max_attempts):
        raw = llm.invoke(messages).content
        fixed, changed = repair_json(raw)
        value, _, error = _validate(schema, fixed)
        result.attempts.append(Attempt(raw=raw, error=error, repaired=changed))
        if value is not None:
            result.value = value
            return result
        messages = messages + [
            HumanMessage(
                content=(
                    f"That output was rejected.\n{error}\n\n"
                    "Return the corrected JSON object only. Remember that null is the correct "
                    "value for anything the abstract does not state."
                )
            )
        ]
    return result


def constrained(
    schema: type[BaseModel],
    abstract: str,
    *,
    model: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> Extraction:
    """Grammar-constrained decoding: ollama is given the JSON Schema as `format`.

    Invalid output is not rejected here, it is unrepresentable. Which is exactly why the
    accuracy columns, not the validity column, are the ones worth reading.
    """
    llm = chat(model, callbacks=callbacks, format=schema.model_json_schema())
    raw = llm.invoke(build_prompt(schema, abstract)).content
    value, _, error = _validate(schema, raw)
    return Extraction(
        strategy="constrained",
        model=model,
        level=schema.__name__,
        abstract_id="",
        attempts=[Attempt(raw=raw, error=error)],
        value=value,
    )


STRATEGIES = {
    "prompt_only": prompt_only,
    "repaired": repaired,
    "error_feedback": error_feedback,
    "constrained": constrained,
}
