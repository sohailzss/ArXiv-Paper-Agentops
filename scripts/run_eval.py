"""Evaluate the running app against data/golden_dataset.json.

Calls the real API (local or deployed), so it measures the whole system: retrieval, agent decisions,
answers, citations, guardrails and latency. LLM-judged answer metrics come from DeepEval
(src/eval/deepeval_metrics.py). Writes a JSON + Markdown report to eval_reports/.

    uv run python scripts/run_eval.py                                   # local API, agent mode
    uv run python scripts/run_eval.py --mode rag                        # plain RAG endpoint
    uv run python scripts/run_eval.py --base-url http://16.58.247.53 --auth arxiv:<password>
    uv run python scripts/run_eval.py --ids gd-001,gd-018 --no-judge    # quick subset, no judge cost
    uv run python scripts/run_eval.py --gate                            # exit 1 if a threshold fails (CI)

Auth can also come from EVAL_AUTH=user:password, the URL from EVAL_BASE_URL.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.eval import metrics as m  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "data" / "golden_dataset.json"
RETRIEVAL_K = (1, 3, 5)
DEEPEVAL_METRICS = (
    "correctness",
    "faithfulness",
    "answer_relevancy",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
)


# ---------- API calls ----------


def post(client: httpx.Client, path: str, payload: dict, retries: int = 8) -> tuple[dict, float]:
    """POST with retry on rate limiting (nginx allows 20 API calls/min per IP in production)."""
    for attempt in range(retries):
        started = time.perf_counter()
        r = client.post(path, json=payload)
        elapsed = time.perf_counter() - started
        if r.status_code == 429:
            time.sleep(min(30, 5 * (attempt + 1)))
            continue
        if r.status_code >= 400:
            return {"_error": f"HTTP {r.status_code}: {r.text[:300]}"}, elapsed
        return r.json(), elapsed
    return {"_error": "rate limited too many times"}, 0.0


def passage_text(source: dict) -> str:
    section = "; ".join(source.get("section_titles", []))
    return (
        f'[{source["number"]}] "{source["title"]}" (arXiv:{source["arxiv_id"]}), section: {section}\n{source["text"]}'
    )


# ---------- one case ----------


def evaluate_case(client: httpx.Client, case: dict, mode: str, judge_model: str | None) -> dict:
    exp = case["expected"]
    options = case.get("request", {})
    relevant = exp["relevant_arxiv_ids"]
    result: dict = {"id": case["id"], "category": case["category"], "difficulty": case.get("difficulty")}

    # 1) Retrieval (only meaningful when there are relevant papers to find)
    if relevant:
        payload = {"query": case["question"], "size": 10, **{k: v for k, v in options.items() if k != "top_k"}}
        search, _ = post(client, "/api/v1/hybrid-search", payload)
        if "_error" in search:
            result["retrieval_error"] = search["_error"]
        else:
            ranked = m.paper_ranking([h["arxiv_id"] for h in search["hits"]])
            result["retrieval"] = {
                **{f"hit@{k}": m.hit_at_k(ranked, relevant, k) for k in RETRIEVAL_K},
                "recall@5": m.recall_at_k(ranked, relevant, 5),
                "precision@5": m.precision_at_k(ranked, relevant, 5),
                "mrr": m.reciprocal_rank(ranked, relevant),
                "ndcg@5": m.ndcg_at_k(ranked, relevant, 5),
                "top_papers": ranked[:5],
            }

    # 2) Answer
    path = "/api/v1/agent/ask" if mode == "agent" else "/api/v1/ask"
    response, latency = post(client, path, {"query": case["question"], **options})
    if "_error" in response:
        result.update(error=response["_error"], passed=False)
        return result

    answer = response["answer"]
    sources = response.get("sources", [])
    cited_numbers = m.citations(answer, len(sources))
    cited_ids = [sources[n - 1]["arxiv_id"] for n in cited_numbers]
    refused = m.is_refusal(answer, len(cited_numbers))
    steps = [s["node"] for s in response.get("steps", [])]
    route = response.get("route")
    guardrail = response.get("guardrail") or {}
    guard_in = (guardrail.get("input") or {}).get("action")
    guard_out = guardrail.get("output") or {}
    forbidden = m.forbidden_hits(answer, exp["must_not_include"])

    # 3) Expected behavior
    behavior = exp["behavior"]
    if behavior == "answer":
        behavior_ok = not refused and len(cited_numbers) >= exp["min_citations"]
    elif behavior == "refuse_not_found":
        behavior_ok = refused or "give_up" in steps
    elif behavior == "direct":
        behavior_ok = (route == "direct") if mode == "agent" else refused
    else:  # blocked
        behavior_ok = (guard_in == "blocked") if guardrail else refused  # without guardrails: must at least refuse
    behavior_ok = behavior_ok and not forbidden

    route_ok = None
    if mode == "agent" and route is not None and not (exp["route"] == "blocked" and not guardrail):
        route_ok = route == exp["route"]
    guardrail_ok = (guard_in == exp["guardrail_input"]) if guardrail.get("input") else None

    result.update(
        behavior=behavior,
        behavior_ok=behavior_ok,
        route=route,
        route_ok=route_ok,
        refused=refused,
        guardrail_input=guard_in,
        guardrail_ok=guardrail_ok,
        grounding=guard_out.get("grounding"),
        relevance=guard_out.get("relevance"),
        fact_recall=m.fact_recall(answer, exp["must_include"]),
        missing_facts=m.missing_facts(answer, exp["must_include"]),
        forbidden_hits=forbidden,
        citations=len(cited_numbers),
        citation_precision=m.citation_precision(cited_ids, relevant) if relevant else None,
        latency_s=round(latency, 2),
        llm_calls=response.get("llm_calls"),
        rewrote=("rewrite" in steps) if mode == "agent" else None,
        steps=steps,
        answer=answer,
    )

    # 4) DeepEval metrics (LLM-judged, 0..1), against the reference answer and the passages the LLM saw
    if judge_model is not None:
        from src.eval.deepeval_metrics import score_case

        scores = score_case(
            question=case["question"],
            answer=answer,
            reference=case["reference_answer"],
            passages=[passage_text(s) for s in sources],
            should_answer=behavior == "answer",
            model=judge_model,
        )
        # Hard rule on top of the judge: declining an answerable question is never "correct".
        if behavior == "answer" and refused and "correctness" in scores:
            scores["correctness"] = {"score": 0.0, "reason": "declined an answerable question"}
        result["deepeval"] = scores
        result.update({f"deepeval_{name}": s["score"] for name, s in scores.items()})

    fr = result["fact_recall"]
    result["passed"] = bool(behavior_ok and (fr is None or fr >= 0.8))
    return result


# ---------- aggregation ----------


def summarize(results: list[dict], mode: str) -> dict:
    ok = [r for r in results if "error" not in r]
    retrieval = [r["retrieval"] for r in ok if "retrieval" in r]
    answerable = [r for r in ok if r["behavior"] == "answer"]
    declines = [r for r in ok if r["behavior"] in ("refuse_not_found", "direct", "blocked")]
    latencies = [r["latency_s"] for r in ok]

    def rate(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    summary = {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "pass_rate": rate(ok, "passed"),
        "retrieval_hit_at_1": m.avg([x["hit@1"] for x in retrieval]),
        "retrieval_hit_at_5": m.avg([x["hit@5"] for x in retrieval]),
        "retrieval_recall_at_5": m.avg([x["recall@5"] for x in retrieval]),
        "retrieval_precision_at_5": m.avg([x["precision@5"] for x in retrieval]),
        "retrieval_mrr": m.avg([x["mrr"] for x in retrieval]),
        "retrieval_ndcg_at_5": m.avg([x["ndcg@5"] for x in retrieval]),
        "fact_recall": m.avg([r["fact_recall"] for r in ok]),
        **{f"deepeval_{name}": m.avg([r.get(f"deepeval_{name}") for r in ok]) for name in DEEPEVAL_METRICS},
        "guardrail_grounding": m.avg([r.get("grounding") for r in ok]),
        "guardrail_relevance": m.avg([r.get("relevance") for r in ok]),
        "citation_precision": m.avg([r["citation_precision"] for r in ok]),
        "behavior_accuracy": rate(ok, "behavior_ok"),
        "route_accuracy": rate(ok, "route_ok"),
        "guardrail_accuracy": rate(ok, "guardrail_ok"),
        "false_refusal_rate": (sum(1 for r in answerable if r["refused"]) / len(answerable)) if answerable else None,
        "decline_accuracy": rate(declines, "behavior_ok"),
        "forbidden_violations": sum(len(r["forbidden_hits"]) for r in ok),
        "latency_p50_s": m.percentile(latencies, 50),
        "latency_p95_s": m.percentile(latencies, 95),
        "avg_llm_calls": m.avg([r.get("llm_calls") for r in ok]) if mode == "agent" else None,
        "rewrite_rate": rate(ok, "rewrote") if mode == "agent" else None,
    }
    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in ok}):
        rows = [r for r in ok if r["category"] == cat]
        by_category[cat] = {
            "cases": len(rows),
            "pass_rate": rate(rows, "passed"),
            "fact_recall": m.avg([r["fact_recall"] for r in rows]),
            "deepeval_correctness": m.avg([r.get("deepeval_correctness") for r in rows]),
            "deepeval_faithfulness": m.avg([r.get("deepeval_faithfulness") for r in rows]),
        }
    summary["by_category"] = by_category
    return summary


def check_gate(summary: dict, thresholds: dict) -> list[str]:
    failures = []
    for metric, limit in thresholds.items():
        value = summary.get(metric)
        if value is None:
            continue
        bad = value > limit if metric == "false_refusal_rate" else value < limit
        if bad:
            failures.append(f"{metric} = {value:.3f} (threshold {limit})")
    return failures


# ---------- report ----------


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if v <= 1.0 else f"{v:.2f}"
    return str(v)


def markdown_report(meta: dict, summary: dict, results: list[dict], failures: list[str]) -> str:
    lines = [
        f"# Evaluation report: {meta['mode']} mode",
        "",
        f"- API: `{meta['base_url']}` · dataset v{meta['dataset_version']} · {meta['timestamp']}",
        f"- Judge: {meta['judge'] or 'off'}",
        f"- Gate: {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {fmt(v)} |" for k, v in summary.items() if k != "by_category"]
    lines += [
        "",
        "## By category",
        "",
        "| Category | Cases | Pass rate | Fact recall | DeepEval correctness | DeepEval faithfulness |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cat, s in summary["by_category"].items():
        cells = [cat, s["cases"], fmt(s["pass_rate"]), fmt(s["fact_recall"])]
        cells += [fmt(s["deepeval_correctness"]), fmt(s["deepeval_faithfulness"])]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines += [
        "",
        "## Cases",
        "",
        "| ID | Category | Pass | Behavior | Hit@5 | MRR | Facts | DE corr/faith/rel | Grounding | Latency | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['id']} | {r['category']} | ERROR | | | | | | | | {r['error'][:80]} |")
            continue
        ret = r.get("retrieval", {})
        notes = []
        if r["missing_facts"]:
            notes.append("missing: " + ", ".join(r["missing_facts"]))
        if r["forbidden_hits"]:
            notes.append("FORBIDDEN: " + ", ".join(r["forbidden_hits"]))
        if not r["behavior_ok"]:
            notes.append(f"expected {r['behavior']}, route={r['route']}, refused={r['refused']}")
        passed, behavior = ("yes" if r["passed"] else "NO"), ("ok" if r["behavior_ok"] else "WRONG")
        lines.append(
            f"| {r['id']} | {r['category']} | {passed} | {behavior} | "
            f"{fmt(ret.get('hit@5'))} | {fmt(ret.get('mrr'))} | {fmt(r['fact_recall'])} | "
            f"{fmt(r.get('deepeval_correctness'))}/{fmt(r.get('deepeval_faithfulness'))}/"
            f"{fmt(r.get('deepeval_answer_relevancy'))} | {fmt(r['grounding'])} | "
            f"{r['latency_s']}s | {'; '.join(notes)} |"
        )
    return "\n".join(lines) + "\n"


# ---------- main ----------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--auth", default=os.getenv("EVAL_AUTH"), help="user:password for the nginx login")
    parser.add_argument("--mode", choices=["agent", "rag"], default="agent")
    parser.add_argument("--ids", help="comma-separated golden ids to run")
    parser.add_argument("--no-judge", action="store_true", help="skip DeepEval's LLM-judged metrics (no judge cost)")
    # Judge choice, measured: gpt-4o-mini ignored the GEval steps and misjudged a verbatim-supported claim;
    # gpt-4.1 judged well but hung inside DeepEval's faithfulness/relevancy calls in 2 of 3 rounds. gpt-4o
    # judged sensibly and never hung (gpt-4.1-mini and gpt-4o-mini didn't hang either).
    parser.add_argument("--judge-model", default="gpt-4o", help="judge model for DeepEval (default gpt-4o)")
    parser.add_argument("--out", default="eval_reports")
    parser.add_argument("--gate", action="store_true", help="exit 1 if any threshold fails")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = dataset["goldens"]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    judge_model = None if args.no_judge else args.judge_model
    judge_name = f"DeepEval ({judge_model})" if judge_model else None

    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    client = httpx.Client(base_url=args.base_url.rstrip("/"), auth=auth, timeout=240)

    print(f"Evaluating {len(cases)} goldens against {args.base_url} ({args.mode} mode, judge: {judge_name or 'off'})\n")
    results = []
    for case in cases:
        r = evaluate_case(client, case, args.mode, judge_model)
        results.append(r)
        if "error" in r:
            print(f"  {r['id']}  ERROR  {r['error'][:100]}")
            continue
        ret = r.get("retrieval", {})
        judge = (
            f" corr={fmt(r.get('deepeval_correctness'))} faith={fmt(r.get('deepeval_faithfulness'))}"
            if judge_model
            else ""
        )
        print(
            f"  {r['id']}  {'PASS' if r['passed'] else 'FAIL'}  {r['category']:<22} "
            f"hit@5={fmt(ret.get('hit@5')):<5} facts={fmt(r['fact_recall']):<5}{judge}  {r['latency_s']}s"
            + (f"  missing={r['missing_facts']}" if r["missing_facts"] else "")
            + ("" if r["behavior_ok"] else f"  BEHAVIOR expected={r['behavior']} route={r['route']}")
        )

    summary = summarize(results, args.mode)
    failures = check_gate(summary, dataset.get("thresholds", {}))
    meta = {
        "mode": args.mode,
        "base_url": args.base_url,
        "dataset_version": dataset.get("version"),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "judge": judge_name,
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"eval_{args.mode}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    (out / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results}, indent=2), encoding="utf-8"
    )
    (out / f"{stem}.md").write_text(markdown_report(meta, summary, results, failures), encoding="utf-8")

    print("\nSummary")
    for k, v in summary.items():
        if k != "by_category":
            print(f"  {k:<26} {fmt(v)}")
    print(f"\nGate: {'PASSED' if not failures else 'FAILED'}")
    for f in failures:
        print(f"  - {f}")
    print(f"\nReport: {out / (stem + '.md')}")
    return 1 if (args.gate and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
