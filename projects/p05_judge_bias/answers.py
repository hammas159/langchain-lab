"""Questions and answer variants, built so that bias can be separated from judgement.

An LLM judge is usually evaluated by agreement with humans on pairs where one answer is better.
That measures judgement and bias mixed together, and it cannot tell you which you are looking
at: a judge that always picks the longer answer will agree with humans often, because longer
answers are frequently better.

So the corpus is built around **tie pairs**. Each question has two answers that are *equally
correct* — the same facts, one stated plainly and one padded with true but redundant
elaboration. There is no quality difference to detect, so any systematic preference is bias and
nothing else. The correct verdict on a tie pair is "tie".

Alongside those are **quality pairs**, where one answer contains a real error. Those measure
whether the judge can judge at all, which is the thing the tie pairs cannot tell you.

Every fact here is ordinary technical material stated accurately; the `wrong` variants each
contain exactly one specific, checkable error, described in `error`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    variant: str  # "concise" | "verbose" | "wrong"
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    concise: Answer
    verbose: Answer
    wrong: Answer
    #: What is actually wrong in the `wrong` variant. Stated so a reader can check the corpus
    #: rather than trust it.
    error: str


CORPUS: tuple[Question, ...] = (
    Question(
        id="q1",
        text="What does a database index cost you?",
        concise=Answer(
            "concise",
            "An index speeds up reads that match it and slows down writes, because every "
            "insert, update and delete has to maintain the index as well as the table. It also "
            "takes disk space.",
        ),
        verbose=Answer(
            "verbose",
            "An index is a separate data structure that the database maintains alongside your "
            "table. The benefit is on the read side: queries whose predicates match the index "
            "can find rows without scanning everything. The cost falls on the write side, "
            "because every insert, update and delete must now update the index as well as the "
            "table itself, so write throughput drops as you add more indexes. There is also a "
            "storage cost, since the index occupies disk space of its own in addition to the "
            "table. This is why indexing every column is not a strategy.",
        ),
        wrong=Answer(
            "wrong",
            "An index speeds up reads that match it and also speeds up writes, because the "
            "database can locate the row to update more quickly. The only real cost is the "
            "disk space it occupies.",
        ),
        error="Claims indexes speed up writes. Maintaining the index makes writes slower.",
    ),
    Question(
        id="q2",
        text="Why use a connection pool?",
        concise=Answer(
            "concise",
            "Opening a database connection is expensive, so a pool keeps a set of them open and "
            "hands them out, which avoids paying that cost per request and caps how many "
            "connections the database has to handle at once.",
        ),
        verbose=Answer(
            "verbose",
            "Establishing a new database connection involves a network round trip, "
            "authentication, and session setup on the server, all of which take measurable time "
            "and server resources. A connection pool keeps a number of connections open and "
            "lends them to application code for the duration of a request, returning them "
            "afterwards. Two benefits follow. The first is latency: you no longer pay the setup "
            "cost on every request. The second is protection of the database, because the pool "
            "size puts a ceiling on how many concurrent connections the server is ever asked to "
            "support, which matters because databases degrade badly past a certain count.",
        ),
        wrong=Answer(
            "wrong",
            "A connection pool exists mainly to let an unlimited number of clients talk to the "
            "database at once, since each request gets its own dedicated connection from the "
            "pool that stays open for the client's whole session.",
        ),
        error="Inverts the purpose: a pool caps concurrency and connections are returned "
        "after each request, not held for a session.",
    ),
    Question(
        id="q3",
        text="What is the difference between a process and a thread?",
        concise=Answer(
            "concise",
            "Processes have separate memory spaces; threads within one process share memory. "
            "That makes threads cheaper to create and to switch between, and it makes shared "
            "state a hazard that needs synchronisation.",
        ),
        verbose=Answer(
            "verbose",
            "A process is an instance of a running program with its own virtual address space, "
            "file descriptors and other operating-system resources. A thread is a unit of "
            "execution inside a process, and all threads in the same process share that address "
            "space. The practical consequences follow from the sharing. Creating a thread is "
            "cheaper than creating a process and switching between threads is usually cheaper "
            "than switching between processes. On the other hand, because threads share memory, "
            "two threads touching the same data need explicit synchronisation, which is a class "
            "of bug that separate processes do not have.",
        ),
        wrong=Answer(
            "wrong",
            "Processes and threads both have their own separate memory spaces; the difference "
            "is that threads are scheduled by the language runtime while processes are "
            "scheduled by the operating system.",
        ),
        error="Says threads have separate memory. Threads in a process share an address space.",
    ),
    Question(
        id="q4",
        text="What does an HTTP 504 mean?",
        concise=Answer(
            "concise",
            "Gateway Timeout: a proxy or gateway did not get a response in time from the "
            "upstream server it was relying on. The failure is upstream of the gateway, not in "
            "the client's request.",
        ),
        verbose=Answer(
            "verbose",
            "A 504 is Gateway Timeout. It is returned by a server acting as a gateway or proxy "
            "when it forwarded a request to an upstream server and did not receive a response "
            "within the time it was prepared to wait. The important part for debugging is where "
            "the problem is: the client's request reached your infrastructure successfully and "
            "the gateway is working well enough to answer, so the thing to investigate is the "
            "upstream service or the network path to it, not the client. It is distinct from a "
            "502, which means the upstream gave an invalid response rather than none at all.",
        ),
        wrong=Answer(
            "wrong",
            "A 504 is Gateway Timeout, returned when the client took too long to send its "
            "request. It usually indicates a slow or unreliable client connection, so the fix "
            "is normally on the client side.",
        ),
        error="Blames the client. 504 is an upstream timeout seen by a gateway; the client "
        "timeout status is 408.",
    ),
    Question(
        id="q5",
        text="Why is `SELECT *` discouraged in application code?",
        concise=Answer(
            "concise",
            "It fetches columns you do not need, and it breaks silently when the schema "
            "changes, because the shape of the result depends on the table rather than on "
            "anything written in the query.",
        ),
        verbose=Answer(
            "verbose",
            "There are two separate objections. The first is efficiency: you transfer and "
            "deserialise every column, including large ones you had no use for, and you give up "
            "the chance of the query being served entirely from an index that covers only the "
            "columns you actually wanted. The second, and the more serious one in a long-lived "
            "codebase, is coupling: the result shape is defined by the current state of the "
            "table rather than by the query text, so adding or reordering a column changes what "
            "your application receives without any change to your code. Naming the columns makes "
            "the contract explicit and the breakage loud.",
        ),
        wrong=Answer(
            "wrong",
            "It is discouraged because `SELECT *` always causes a full table scan, so the "
            "database cannot use any index to satisfy the query. Naming the columns explicitly "
            "is what allows the planner to use an index.",
        ),
        error="Claims SELECT * forces a full scan. Column selection does not determine whether "
        "an index is used for the predicate; it affects covering-index eligibility only.",
    ),
    Question(
        id="q6",
        text="What is idempotency and why does it matter for an API?",
        concise=Answer(
            "concise",
            "An idempotent operation has the same effect whether it is applied once or several "
            "times. It matters because clients retry, and without it a retry after a timeout "
            "can perform the action twice.",
        ),
        verbose=Answer(
            "verbose",
            "An operation is idempotent when applying it repeatedly leaves the system in the "
            "same state as applying it once. This matters in practice because networks fail in "
            "a specific and awkward way: a client can send a request, have it succeed on the "
            "server, and never receive the response. The client cannot distinguish that case "
            "from the request never arriving, so it retries. If the operation is not idempotent "
            "the retry charges the card a second time or creates a second order. The usual "
            "remedy is an idempotency key supplied by the client, which the server records so "
            "that a repeat of the same key returns the original result instead of acting again.",
        ),
        wrong=Answer(
            "wrong",
            "Idempotency means an operation always returns the same response body to the same "
            "client. It matters because it lets responses be cached, which reduces load on the "
            "server when clients repeat a request.",
        ),
        error="Defines idempotency as response equality and frames it as a caching concern. It "
        "is about the effect on server state.",
    ),
    Question(
        id="q7",
        text="What problem does a message queue solve between two services?",
        concise=Answer(
            "concise",
            "It decouples them in time: the producer can keep working when the consumer is "
            "down or slow, because messages wait in the queue instead of being lost or blocking "
            "the caller.",
        ),
        verbose=Answer(
            "verbose",
            "The central benefit is temporal decoupling. With a direct synchronous call the "
            "producer's availability is bounded by the consumer's: if the consumer is down, the "
            "call fails, and if the consumer is slow, the producer waits. Putting a queue "
            "between them means the producer hands the message over and continues, and the "
            "consumer processes it whenever it is able to. This also absorbs bursts, since a "
            "spike in production becomes a growing backlog rather than a wave of failures, and "
            "it gives you a natural place to retry work that failed. The cost is that the "
            "interaction becomes asynchronous, so the producer no longer learns the outcome.",
        ),
        wrong=Answer(
            "wrong",
            "A message queue's main purpose is to make communication between the two services "
            "faster, because sending a message through a queue has lower latency than a direct "
            "HTTP call and guarantees the work completes sooner.",
        ),
        error="Claims a queue lowers end-to-end latency. It adds a hop; it trades latency for "
        "decoupling and burst absorption.",
    ),
    Question(
        id="q8",
        text="Why do floating-point comparisons need a tolerance?",
        concise=Answer(
            "concise",
            "Most decimal fractions cannot be represented exactly in binary floating point, so "
            "arithmetic introduces small representation errors and two values that should be "
            "equal often differ in their last bits.",
        ),
        verbose=Answer(
            "verbose",
            "Binary floating point represents numbers as a sign, a significand and an exponent "
            "in base two. Many decimal fractions, 0.1 among them, have no exact finite binary "
            "representation, so what gets stored is the nearest representable value rather than "
            "the number you wrote. Arithmetic on those approximations accumulates small errors, "
            "with the consequence that expressions which are mathematically equal can produce "
            "results differing in their lowest bits. Comparing with `==` then reports inequality "
            "for values that are equal in every sense you care about, which is why comparisons "
            "are normally written against a tolerance instead.",
        ),
        wrong=Answer(
            "wrong",
            "Floating-point numbers are stored exactly, but comparison operators on them are "
            "implemented approximately for speed, so `==` can return false for identical "
            "values. Using a tolerance works around that implementation detail.",
        ),
        error="Says the values are exact and the comparison is approximate. It is the other "
        "way round.",
    ),
)

BY_ID = {q.id: q for q in CORPUS}


def tie_pairs() -> tuple[tuple[Question, Answer, Answer], ...]:
    """Equal-quality pairs. The correct verdict is a tie; any preference is bias."""
    return tuple((q, q.concise, q.verbose) for q in CORPUS)


def quality_pairs() -> tuple[tuple[Question, Answer, Answer], ...]:
    """One answer is genuinely wrong. The correct verdict is the concise one."""
    return tuple((q, q.concise, q.wrong) for q in CORPUS)


def length_ratio(q: Question) -> float:
    return q.verbose.words / q.concise.words
