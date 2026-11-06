"""Evaluation metrics for the arXiv RAG Curator. Pure functions, so they're unit-tested and reusable.

Retrieval (paper level: chunks are collapsed to their paper, keeping the best rank)
  hit@k        1 if any relevant paper is in the top k
  recall@k     share of relevant papers found in the top k
  precision@k  share of the top k papers that are relevant
  MRR          1 / rank of the first relevant paper
  nDCG@k       rank-aware: relevant papers near the top count more

Answers
  fact recall         share of must_include groups present (deterministic correctness proxy)
  forbidden hits      must_not_include strings that appeared (hallucination traps, leaked PII/prompt)
  citation count      valid [n] citations
  citation precision  share of cited sources that are relevant papers
  refusal detection   did the answer say "not found" instead of answering

Plus LLM-judge correctness/faithfulness (judge.py), Guardrails grounding/relevance (from the API),
behavior/route/guardrail accuracy, and latency percentiles (computed in scripts/run_eval.py).
"""

import math
import re
from statistics import mean

_CITATION = re.compile(r"\[(\d+)\]")

# Phrases the app (and the LLM under its prompt) uses when the passages don't answer the question.
REFUSAL_PATTERNS = re.compile(
    r"couldn['’]t find|could not find|can['’]t find|cannot find|"
    r"do(?:es)? not (?:contain|provide|include|mention|cover)|don['’]t (?:contain|provide|include|mention|cover)|"
    r"no (?:information|passages)|not (?:mentioned|covered|provided) in|"
    r"only answer questions about|can['’]t help with|cannot help with|"
    r"isn['’]t (?:covered|mentioned)|aren['’]t (?:covered|mentioned)",
    re.IGNORECASE,
)


# ---------- retrieval ----------


def paper_ranking(arxiv_ids_in_rank_order: list[str]) -> list[str]:
    """Chunk-level hits -> paper-level ranking (first occurrence wins)."""
    seen: list[str] = []
    for aid in arxiv_ids_in_rank_order:
        if aid not in seen:
            seen.append(aid)
    return seen


def hit_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return len([a for a in top if a in set(relevant)]) / len(top)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    for i, aid in enumerate(ranked, start=1):
        if aid in set(relevant):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    """Binary relevance nDCG."""
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, aid in enumerate(ranked[:k], start=1) if aid in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---------- answers ----------


def fact_recall(answer: str, must_include: list[list[str]]) -> float | None:
    """Each group is a list of acceptable alternatives; a group counts if any alternative appears."""
    if not must_include:
        return None
    text = answer.lower()
    found = sum(1 for group in must_include if any(alt.lower() in text for alt in group))
    return found / len(must_include)


def missing_facts(answer: str, must_include: list[list[str]]) -> list[str]:
    text = answer.lower()
    return [group[0] for group in must_include if not any(alt.lower() in text for alt in group)]


def forbidden_hits(answer: str, must_not_include: list[str]) -> list[str]:
    text = answer.lower()
    return [s for s in must_not_include if s.lower() in text]


def citations(answer: str, source_count: int) -> list[int]:
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= source_count and n not in seen:
            seen.append(n)
    return seen


def citation_precision(cited_arxiv_ids: list[str], relevant: list[str]) -> float | None:
    """Share of cited sources that come from a relevant paper. None when nothing was cited."""
    if not cited_arxiv_ids:
        return None
    rel = set(relevant)
    return sum(1 for a in cited_arxiv_ids if a in rel) / len(cited_arxiv_ids)


def is_refusal(answer: str, citation_count: int) -> bool:
    """A 'not found / can't help' answer: refusal wording and no citations."""
    return citation_count == 0 and bool(REFUSAL_PATTERNS.search(answer))


# ---------- aggregation ----------


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None
