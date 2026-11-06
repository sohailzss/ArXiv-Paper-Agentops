"""Gradio front end for the RAG API. Talks to FastAPI over HTTP only, so it needs no
database or OpenSearch access and stays out of the API's Docker image.

    docker compose up -d                              # API on :8000
    uv run --group ui python ui/gradio_app.py         # UI on http://localhost:7860

Point it at another API with RAG_API_URL=http://host:8000.
"""

import json
import os
from collections.abc import Iterator

import gradio as gr
import httpx

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = [
    "Which papers benchmark coding agents, and what do they measure?",
    "How can robots learn motion skills from simulation?",
    "What is a compositional knowledge architecture for organizations?",
]


def iter_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:
    """Parse server-sent events into (event_name, json_data) pairs."""
    event, data = "message", []
    for line in lines:
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield event, json.loads("\n".join(data))


def format_sources(sources: list[dict], cited: list[int] | None = None) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for s in sources:
        mark = " **(cited)**" if cited and s["number"] in cited else ""
        section = "; ".join(s["section_titles"])
        lines.append(f"**[{s['number']}]** [{s['title']}]({s['url']}) · _{section}_ · [PDF]({s['pdf_url']}){mark}")
    return "\n\n".join(lines)


def format_guardrail(guardrail: dict | None) -> str:
    """One status line for the AWS Guardrails result (empty when guardrails are off)."""
    if not guardrail:
        return ""
    parts = []
    if (i := guardrail.get("input")) and i["action"] != "none":
        parts.append(
            f"question {i['action']} ({', '.join(i['reasons'])})" if i["reasons"] else f"question {i['action']}"
        )
    if o := guardrail.get("output"):
        if o.get("grounding") is not None:
            parts.append(f"grounding {o['grounding']:.2f} · relevance {o['relevance']:.2f}")
        if o["action"] not in ("none", "flagged"):
            parts.append(f"answer {o['action']}")
    return "Guardrails: " + (" · ".join(parts) if parts else "passed")


FLAG_WARNING = (
    "> **Check this answer:** AWS Guardrails' grounding check found parts of it may not be supported "
    "by the cited passages.\n\n"
)


def format_step(step: dict) -> str:
    d = step.get("detail", {})
    match step["node"]:
        case "guardrail":
            stage, action = d.get("stage"), d.get("action")
            if stage == "input":
                return f"1. **Guardrail (question):** {action}" + (
                    f" ({', '.join(d['reasons'])})" if d.get("reasons") else ""
                )
            score = f", grounding {d['grounding']:.2f}" if d.get("grounding") is not None else ""
            return f"1. **Guardrail (answer):** {action}{score}"
        case "route":
            where = "search the papers" if d.get("route") == "search" else "answer directly"
            return f"1. **Route:** {where} _({d.get('reason', '')})_"
        case "retrieve":
            return f"1. **Search #{d.get('attempt')}:** `{d.get('query')}` → {d.get('found')} passages"
        case "grade":
            return f"1. **Grade:** {len(d.get('relevant', []))} of {d.get('of')} passages relevant"
        case "rewrite":
            return f"1. **Rewrite query:** `{d.get('query')}`"
        case "generate":
            return f"1. **Answer** from {d.get('passages')} passages"
        case "direct":
            return "1. **Answer directly** (no search needed)"
        case "give_up":
            return f"1. **Stop:** nothing relevant after {d.get('attempts')} searches"
    return f"1. {step['node']}"


def ask(question: str, mode: str, top_k: float, categories: str, date_from: str, date_to: str, use_agent: bool = True):
    question = (question or "").strip()
    if not question:
        yield "Type a question first.", "", ""
        return

    payload: dict = {"query": question, "mode": mode, "top_k": int(top_k)}
    if cats := [c.strip() for c in (categories or "").split(",") if c.strip()]:
        payload["categories"] = cats
    if date_from and date_from.strip():
        payload["date_from"] = date_from.strip()
    if date_to and date_to.strip():
        payload["date_to"] = date_to.strip()

    endpoint = "/api/v1/agent/stream" if use_agent else "/api/v1/ask/stream"
    answer, sources, status, steps = "", [], "Thinking..." if use_agent else "Searching...", []
    yield answer, "", status
    try:
        with httpx.stream("POST", f"{API_URL}{endpoint}", json=payload, timeout=240) as response:
            if response.status_code != 200:
                detail = response.read().decode(errors="replace")
                try:
                    detail = json.loads(detail).get("detail", detail)
                except ValueError:
                    pass
                yield f"**Error {response.status_code}:** {detail}", "", ""
                return
            for event, data in iter_sse(response.iter_lines()):
                if event == "step":  # agent mode: show each decision as it happens
                    steps.append(format_step(data))
                    status = "**Agent steps**\n" + "\n".join(steps)
                    yield answer, format_sources(sources), status
                elif event == "sources":
                    sources = data["sources"]
                    if not use_agent:
                        status = f"Retrieved {len(sources)} passages in {data['retrieval_ms']} ms · writing answer..."
                    yield answer, format_sources(sources), status
                elif event == "token":
                    answer += data["text"]
                    yield answer, format_sources(sources), status
                elif event == "guardrail":
                    # The answer was already streamed; the output check can replace it or flag it.
                    if data.get("replace_answer"):
                        answer = data["replace_answer"]
                    if data["check"]["action"] == "flagged":
                        answer = FLAG_WARNING + answer
                    yield answer, format_sources(sources), status
                elif event == "done" and use_agent:
                    summary = (
                        f"{data['llm_calls']} LLM calls · {len(data['search_queries'])} searches · "
                        f"{data['total_ms']} ms · cited {len(data['cited'])} of {len(sources)} sources"
                    )
                    if data.get("trace_id"):
                        summary += f" · trace `{data['trace_id'][:8]}`"
                    if guard := format_guardrail(data.get("guardrail")):
                        summary += f"\n\n{guard}"
                    status = "**Agent steps**\n" + "\n".join(steps) + f"\n\n{summary}"
                    yield answer, format_sources(sources, data["cited"]), status
                elif event == "done":
                    cited = f"cited {len(data['cited'])} of {len(sources)} sources"
                    if data.get("cached"):
                        status = f"**Served from cache** (no search or LLM call) · {cited}"
                    else:
                        status = (
                            f"LLM: {data['llm_model'] or 'not called'} · retrieval {data['retrieval_ms']} ms · "
                            f"generation {data['generation_ms']} ms · {cited}"
                        )
                    if data.get("trace_id"):
                        status += f" · trace `{data['trace_id'][:8]}`"
                    if guard := format_guardrail(data.get("guardrail")):
                        status += f"\n\n{guard}"
                    yield answer, format_sources(sources, data["cited"]), status
                elif event == "error":
                    yield answer + f"\n\n**Error:** {data['detail']}", format_sources(sources), ""
                    return
    except httpx.HTTPError as exc:
        yield f"**Can't reach the API at {API_URL}:** {exc}", "", ""


def build() -> gr.Blocks:
    with gr.Blocks(title="arXiv RAG Curator") as demo:
        gr.Markdown(
            "# arXiv RAG Curator\n"
            "Ask questions about recently ingested cs.AI papers. Answers cite their sources as [n]."
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question", placeholder="e.g. How are coding agents evaluated?", lines=2)
                use_agent = gr.Checkbox(
                    value=True,
                    label="Agent mode (route, grade passages, rewrite and retry; slower but more careful)",
                )
                with gr.Row():
                    submit = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
                gr.Examples(EXAMPLES, inputs=question)
            with gr.Column(scale=2):
                with gr.Accordion("Retrieval settings", open=False):
                    mode = gr.Radio(["hybrid", "bm25", "vector"], value="hybrid", label="Search mode")
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Passages given to the LLM")
                    categories = gr.Textbox(label="Categories (comma-separated)", placeholder="cs.AI, cs.LG")
                    with gr.Row():
                        date_from = gr.Textbox(label="From (YYYY-MM-DD)")
                        date_to = gr.Textbox(label="To (YYYY-MM-DD)")
        status = gr.Markdown()
        answer = gr.Markdown(label="Answer")
        gr.Markdown("### Sources")
        sources = gr.Markdown()

        inputs = [question, mode, top_k, categories, date_from, date_to, use_agent]
        outputs = [answer, sources, status]
        submit.click(ask, inputs, outputs)
        question.submit(ask, inputs, outputs)
        clear.click(lambda: ("", "", "", ""), None, [question, answer, sources, status])
    return demo


if __name__ == "__main__":
    build().queue().launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"), server_port=int(os.getenv("GRADIO_PORT", "7860"))
    )
