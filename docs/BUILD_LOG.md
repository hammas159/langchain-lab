# Build log

How this repo was actually built, and what went wrong while building it. Written as the work
happened rather than reconstructed afterwards, so the mistakes are the ones that were really
made and not the tidy ones that make a good story.

The entries that matter most are the two under "Bugs the tests found", because both of them
were producing *plausible numbers*. A crash announces itself. A scoring bug hands you a
result table that looks exactly like a finding.

---

## The machine this was built on

| | |
|---|---|
| GPU | Quadro RTX 5000, 16 GB |
| Models | ollama, served locally on `127.0.0.1:11434` |
| Python | 3.11+, managed with `uv` |
| Browser automation | Playwright + Chromium, for the screenshots |

No hosted API was called at any point. That was a hard constraint, not a preference: a repo
whose results cannot be reproduced without a billing account is a repo whose results nobody
checks.

---

## Problems hit while building this

### 1. "ollama is not installed" — it was, and that was my error

The first capability check ran `ollama list` in a Bash shell, got `command not found`, and
concluded no local model runtime was available. The whole plan was about to be built around
that, using injected fake backends so nothing needed a GPU.

It was wrong. Ollama was installed at
`AppData\Local\Programs\Ollama` and the server was already running — it simply was not on the
shell's `PATH`. Probing the port directly is the check that actually answers the question:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

**Lesson, and it generalises past ollama:** `command -v` answers "is this on my PATH", not "is
this service available". For anything that runs as a server, ask the server.

### 2. Downloads at 349 KB/s reshaped the project

The plan was a fleet of five or six small models, so that every cross-model claim rested on
more than one model family. Measured throughput to the ollama registry was **349 KB/s**,
which puts the ~10 GB fleet at roughly eight hours.

This is why `shared/models.py` is a registry with *roles* rather than hard-coded tags, and why
`benchmark.py` intersects the registry with `installed_tags()` before running. The grid runs
on whatever is actually present and the results table states which models those were. Adding a
model later is a pull, not an edit.

**Lesson:** when an external resource is slow enough to change what you can build, make the
code indifferent to how much of it has arrived.

### 3. `uv sync` failed because `README.md` did not exist yet

```
OSError: Readme file does not exist: README.md
```

`pyproject.toml` declared `readme = "README.md"`, and hatchling validates that when building
the project itself. The install had resolved the entire dependency tree — a 376 KB lockfile —
and then failed on a missing text file. Exit code was `0`, so the failure was easy to miss,
and the venv existed but contained nothing.

**Lesson:** an exit code of 0 from a wrapper does not mean the wrapped build succeeded. Check
for the artefact, not the status.

### 4. Starlette changed the `TemplateResponse` signature

The first page render returned 500 with a genuinely confusing message:

```
TypeError: cannot use 'tuple' as a dict key (unhashable type: 'dict')
```

raised from inside Jinja's template *cache*, not from rendering. The cause: modern Starlette
takes `TemplateResponse(request, name, context)`, and the older `TemplateResponse(name,
context)` form was being used. Starlette took the template name as the request and the context
dict as the name, then tried to use that dict as a cache key.

**Lesson:** a traceback that terminates in a library's caching layer is usually an argument
that arrived in the wrong position, not a bug in the cache.

### 5. Jinja cannot slice the result of a dotted dict lookup

`{{ a.expected.intervention[:38] }}` raised the same unhashable-type error. `a.expected` is a
plain dict, so `.intervention` resolves through `__getitem__`, and the following slice is then
applied as another subscript in a way Jinja does not handle. `{{ a.expected['intervention'] |
truncate(40, true) }}` works, and reads better.

### 6. `npx playwright` working does not mean `import playwright` works

`npx --yes playwright --version` printed `1.63.0`, and `npx playwright install chromium`
downloaded the browser successfully. The screenshot script then died with
`ERR_MODULE_NOT_FOUND`. `npx` had fetched the CLI into its own cache; nothing was installed in
the project. The fix was an explicit dev-only `package.json`.

The browser binary and the library are separate installs, and having one tells you nothing
about the other.

---

## Bugs the tests found

Both of these shipped wrong numbers before they were caught, and neither crashed.

### 7. The scorer was survivorship-biased, and it flattered the wrong strategy

The first full grid produced this row for the hardest schema level:

| level | strategy | valid | accuracy |
|---|---|---|---|
| L5_union | `prompt_only` | 25% | **96%** |
| L5_union | `constrained` | 100% | 77% |

Read naively: plain prompting is far more accurate, and grammar constraints cost you 19 points
of accuracy to buy validity. That reading is wrong, and it is wrong in the direction that
makes the more interesting story.

`score()` returned an empty result for an extraction that failed to parse, and an empty result
contributed **no fields to the denominator**. So `prompt_only`'s 96% was its accuracy *on the
three attempts out of twelve that happened to validate* — necessarily the easy ones — while
`constrained`'s 77% was its accuracy across all twelve.

The fix is `scoreable_fields()`: a failed extraction is charged for every field it owed,
scoring 0 for each. Re-running the identical grid:

| level | strategy | valid | accuracy (biased) | accuracy (corrected) |
|---|---|---|---|---|
| L5_union | `prompt_only` | 25% | 96% | **23%** |
| L5_union | `constrained` | 100% | 77% | **77%** |

The conclusion does not shift, it **inverts**. Constrained decoding is not paying accuracy for
validity; it is more than three times more accurate at this level, and the naive metric hid
that behind a number computed over the survivors. Two tests pin the behaviour, including
`test_failed_extraction_is_charged_for_every_field_it_owed`.

One subtlety came out of fixing it. Fabrication and omission *must not* use the same
denominator: a parse failure produced no document, so it invented nothing. Folding failures
into the fabrication denominator makes an unreliable strategy look like an honest one.
Accuracy is charged over everything; fabrication is charged only over documents that exist.

**Lesson:** whenever you compare methods with different success rates, the first question is
what happened to the failures. If they vanished from the average, the comparison is measuring
which method fails on the hard cases.

### 8. `bool` is a subclass of `int`, so `True` matched `1.5`

`values_match` compared booleans with `bool(got) == bool(want)`. Since `bool(1.5)` is `True`, a
non-inferiority margin of `1.5` scored as a correct `met: true`. Caught by
`test_booleans_are_not_compared_as_numbers`.

The ordering matters: the bool check has to come *before* the numeric branch, because
`isinstance(True, int)` is `True` in Python.

### 9. JSON repair corrupted a legitimate string

Repairing Python literals with `re.sub(r"\bNone\b", "null", text)` across the whole document
turned the extracted value `"None of the above"` into `"null of the above"`. On a medical
corpus, silently rewriting the inside of a string is not a cosmetic bug.

`_replace_python_literals` now tracks whether the scan is inside a JSON string and leaves those
spans alone. Same class of error as using a regex to find a JSON object: the naive version
works on every example you think to try.

---

---

## Project 02

### 10. Retrieval recall of 9% looked like a bug, and was the finding

The first coverage numbers were so bad they read as broken code: at k=2 the retriever fetched
evidence for 1 field in 11. The instinct was to go looking for a mismatch between the embedding
of the query and the embedding of the passages.

Printing the **full ranking** instead of the metric settled it in one look. Scores were sensible,
well separated and correctly ordered — and the passage carrying `population`, `intervention` and
`comparator` sat 18th of 20, below every piece of generic filler.

The query was the problem. It was built from the schema's field names, the way a real extraction
pipeline builds one, and abstract field names resemble abstract methodological prose far more
than they resemble a sentence about chronic tension headache.

**Lesson:** when a retrieval metric looks broken, read the ranking before reading the code. A
ranking that is *correct and useless* looks identical to a bug from the metric alone.

### 11. `nomic-embed-text` is reported as `nomic-embed-text:latest`

The registry stores the bare tag; ollama's `/api/tags` returns it suffixed. A set-membership
check therefore said a model that was pulled and working was missing, and the benchmark would
have skipped it **without saying anything** — a silently narrower results table, which is the
worst failure mode available to a benchmark. `installed_tags()` now returns both spellings.

### 12. A metric whose denominator shrinks as the problem is fixed

The phantom rate — of the fields retrieval withheld, how many the model filled in anyway — rises
from 69% to 100% as retrieval gets better, while the phantom *count* falls from 61 to 11.

Both numbers are correct. The rate is conditioned on "fields retrieval withheld", and good
retrieval withholds fewer and harder fields, so the pipeline that fixed almost everything scores
worst on the headline. Ranking strategies by phantom rate selects the worst one.

This is the same species of error as project 01's, from the opposite direction: there, failures
left the denominator; here, the denominator is the thing being improved. Both were the obvious
way to measure the quantity in question.

### 13. `lucky_phantom` had to exist

A value produced for a field whose evidence was never retrieved is a guess, and 8 of them in the
`single_query_k4` row happened to be **right**. Scoring those as `grounded_correct` because they
matched ground truth would have hidden 8 phantoms and overstated grounding. A guess that comes
out right is still a guess, so it gets its own verdict and is counted as correct *and* as a
phantom.

---

## Things that turned out not to be true

Kept deliberately. A build log that only records confirmed hypotheses is a marketing document.

- **"Constrained decoding will cause fabrication."** This was the hypothesis the entire project
  was designed around, and the data does not support it. Fabrication sits at **27% for every
  strategy**, plain prompting included, identical to three significant figures. On this corpus
  fabrication is a property of the *model*, not of the decoding method — the schema never
  enters into it.

  Worse for the hypothesis, constrained decoding came out **ahead** overall: 100% validity and
  the best pooled accuracy at 84%, against 74% for plain prompting. The corpus was built to
  catch grammar-constrained decoding inventing values, and instead it caught the naive
  evaluation method inventing a result.

  What survives is much smaller and worth stating precisely: at **L4 only**, forcing validity
  does cost accuracy (82% against 88%), and the omission rate across the grid rises from 1% to
  4%. Those are real, and they are a footnote rather than a headline.

- **JSON repair was expected to matter.** `repaired` and `prompt_only` produced *identical*
  numbers across the whole grid. The repair layer never changed a single outcome for this
  model: when it emitted JSON at all, the JSON was already clean. The layer is kept because it
  is nearly free and the result is itself worth reporting, but it solves a problem this model
  does not have.
