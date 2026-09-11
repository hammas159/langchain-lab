# langchain-lab

[![ci](https://github.com/hammas159/langchain-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/hammas159/langchain-lab/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![models](https://img.shields.io/badge/models-local%20via%20ollama-success)
![api%20keys](https://img.shields.io/badge/API%20keys-none%20required-success)
![license](https://img.shields.io/badge/license-MIT-green)

**LangChain projects built around the failure each one is usually demoed past.**

Every project here runs entirely on local models through ollama. There is no API key in this
repo, no hosted call, and no cost — which is the point: a result nobody can reproduce without
a billing account is a result nobody checks.

---

## The through-line

Each project takes a technique that is normally shown working and measures the case where it
does not. The first one turned out to be about something sharper than that:

> **A technique looked worse than the alternative because the metric quietly dropped its
> failures. Fixing the denominator inverted the conclusion.**

## Projects

| | Project | The finding | Tests |
|---|---|---|---|
| 01 | [Structured output under pressure](projects/p01_structured_output/) | Grammar-constrained decoding appears **19 points less accurate** than plain prompting, and is really **3x more accurate**. The gap is entirely survivorship bias in the standard metric. | 41 |

Four more are planned and not yet written. They are listed at the bottom, deliberately
without numbers, because nothing in this repo claims a result it has not produced.

### 01 · Structured output under pressure — 41 tests

Four strategies for getting schema-shaped JSON out of a small model, scored on clinical-trial
abstracts where some fields are **genuinely absent** and the correct answer is `null`.

Built to show that constrained decoding invents content. It does not: fabrication sits at
**27% for every strategy**, plain prompting included, because on this corpus fabrication is a
property of the model rather than of the decoding method. What the project caught instead was
the evaluation method inventing a result.

- **Stack:** `langchain-core`, `langchain-ollama`, Pydantic v2, FastAPI + Jinja2
- **In:** a trial abstract and a schema, at one of five difficulty levels
- **Out:** the extracted object, scored field by field, with **fabrication counted separately
  from every other kind of wrong**

## Why fabrication is scored separately

Most structured-output evaluations report one number: did it validate. That number is why
constrained decoding looks like a solved problem. Here a result breaks into four:

| | |
|---|---|
| **validity** | the object parsed and satisfied the schema |
| **exactness** | the value equals ground truth, field by field |
| **fabrication** | ground truth is `null` and the model returned a value |
| **omission** | ground truth is a value and the model returned `null` |

An omission costs a reader a fact they must look up. A fabrication puts a number in front of
them that they will not check. Feeding an extraction pipeline into a database, the second is
the one that does damage, and a validity-only metric hides it completely.

## The model fleet

`shared/models.py` is the single source of truth. Projects ask for a **capability** ("something
that can call tools") rather than a tag, so adding a model is a `pull`, not an edit.

| model | params | context | tools | role |
|---|---|---|---|---|
| `granite3.3:2b` | 2.5B | 128k | yes | the floor — it is here to fail |
| `qwen2.5:3b-instruct` | 3.1B | 32k | yes | default |
| `qwen2.5-coder:3b` | 3.1B | 32k | yes | structured text |
| `llama3.2:3b` | 3.2B | 128k | yes | a second family, so results are not just about Qwen |
| `qwen2.5:7b-instruct` | 7.6B | 32k | yes | the quality ceiling |
| `nomic-embed-text` | 0.14B | 8k | — | embeddings |

**Only `qwen2.5:3b-instruct` was installed when the current numbers were produced.** The
benchmark intersects this registry with what is actually pulled and names the models it used,
so `RESULTS.md` never implies a fleet that was not there.

## Quick start

```bash
git clone https://github.com/hammas159/langchain-lab
cd langchain-lab

make install
ollama pull qwen2.5:3b-instruct

make test          # 41 tests, no GPU and no ollama needed
make bench         # regenerates RESULTS.md from real calls
make web           # http://127.0.0.1:8101
```

## Layout

```
shared/
  models.py          the fleet registry: capabilities, not hard-coded tags
  llm.py             chat/embeddings + Ledger, the callback that counts calls
  web/               design system and templates, shared by every project's UI
projects/
  p01_structured_output/
    schemas.py       five levels of schema difficulty
    corpus.py        12 synthetic abstracts + hand-written ground truth
    strategies.py    the four extraction strategies
    scoring.py       where fabrication is separated from omission
    benchmark.py     writes RESULTS.md; no number is typed by hand
    web.py           the live UI
scripts/shoot.mjs    drives the real app in a real browser for the screenshots
docs/BUILD_LOG.md    what went wrong while building this
```

## Requirements

Python 3.11+, `uv`, and ollama running locally. A GPU is not required to run the tests — only
to run the benchmarks and the UI. These numbers were produced on a Quadro RTX 5000 (16 GB).

## Tests

```bash
make test        # 41, deselects the `live` mark
make test-live   # adds the tests that need a running ollama
```

The suite is split deliberately. Scoring rules, JSON repair and corpus invariants are pure
functions, and those are where a regression hides; a test suite that only passes when a GPU is
warm is a test suite nobody runs. CI runs the pure set on every push.

## Screenshots

Every image in this repo is a real capture of the running app, taken by
`scripts/shoot.mjs` driving Chromium. Nothing is a mockup. Each is shot in both colour schemes,
because the design system defines both and a dark-mode bug is invisible if you only screenshot
in light.

## What this repo does NOT do

- **It does not test hosted models.** Every number is a local model on one machine.
- **It is not a LangChain tutorial.** It assumes you know what a chain is and goes at the parts
  that break.
- **It does not benchmark LangChain against alternatives.** LangChain is the tool here, not the
  subject.
- **It does not yet contain five projects.** One is finished. The others are named below and
  nothing is claimed for them.

## Problems hit while building this

Full account in [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md). A sample:

- **The scorer was survivorship-biased** and produced a confident, plausible, *inverted*
  result. It did not crash. Failed extractions contributed nothing to the denominator, so the
  strategy that failed most often looked the most accurate.
- **`bool` is a subclass of `int`**, so `bool(1.5) == bool(True)` scored a non-inferiority
  margin of 1.5 as a correct `met: true`.
- **A regex rewriting `None` to `null`** across a whole document turned the extracted string
  `"None of the above"` into `"null of the above"`.
- **`npx playwright` working does not mean `import playwright` works** — the CLI, the library
  and the browser binary are three separate installs.
- **Model downloads ran at 349 KB/s**, which is why the fleet registry is capability-based and
  the benchmark runs on whatever has arrived.

## Planned

Not started. Listed so the intent is on record, with no results attached:

- **02 · Semantic cache with a false-hit gate** — a paraphrase cache that returns the *wrong*
  cached answer, and what it costs to detect that.
- **03 · Memory that forgets the wrong thing** — summarisation buffers measured on fact recall
  against turn count.
- **04 · Indirect prompt injection** — where the injection arrives inside a retrieved document
  rather than the user's message.
- **05 · LLM-as-judge, biased** — position and length bias measured, then corrected.

## License

MIT
