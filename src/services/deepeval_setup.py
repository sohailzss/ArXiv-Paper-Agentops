"""DeepEval environment defaults. Call configure() before deepeval is first imported (it reads these
once at import time). Used by both the online evaluator and the offline golden evaluation.

- Telemetry off: DeepEval otherwise reports usage to its own servers.
- Fail fast on stalled judge calls. Measured: occasional OpenAI requests get no response at all; with
  DeepEval's defaults (88.5 s per attempt, 2 attempts) one stall cost 88 s and two in a row lost the
  metric entirely (3 of ~130 metric runs in a full offline evaluation). Normal calls take 2-5 s, so
  30 s per attempt with 4 attempts retries stalls quickly and almost never gives up.
"""

import os
import re

_CITATION = re.compile(r"\s*\[\d+\]")


def strip_citations(answer: str) -> str:
    """Remove [n] markers before DeepEval scores an answer. They're UI links, not content, but DeepEval's
    statement extraction counted "[1]" as an irrelevant statement: a correct one-fact answer scored 0.5
    answer relevancy (measured on a live trace)."""
    return _CITATION.sub("", answer).strip()


DEFAULTS = {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE": "30",
    "DEEPEVAL_RETRY_MAX_ATTEMPTS": "4",
}


def configure() -> None:
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)  # explicit env vars still win
