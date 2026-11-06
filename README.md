# Agentic RAG arXiv Curator

An agentic RAG system over arXiv AI papers, built phase by phase: ingestion, hybrid search, RAG, caching, a LangGraph agent, then deployment to AWS.

> Inspired by the open-source *arXiv Paper Curator* course project (MIT). Rebuilt from scratch with Airflow 3, a switchable OpenAI/Bedrock LLM layer, and an EC2 deployment.

## Status

| Phase | What | Status |
| --- | --- | --- |
| 0 | Tools, accounts, keys | done |
| 1 | FastAPI, OpenSearch, Neon, Docker Compose, health checks | done |
| 2 | arXiv client, Docling PDF parsing, Airflow 3 ingestion DAG | done |
| 3 | BM25 keyword search (OpenSearch, `/api/v1/search`) | done |
| 4 | Chunking, OpenAI/Jina embeddings (switchable), hybrid RRF search | done |
| 5 | RAG with OpenAI/Bedrock (switchable) + SSE streaming + Gradio UI | done |
| 6 | Redis answer cache + Langfuse tracing (tokens, cost, latency per step) | done |
| 7 | LangGraph agent: route, grade passages, rewrite + retry, honest give-up | done |
| 8 | Deploy to AWS EC2 (Nginx + basic auth, private ports, IAM role) | done |

## Architecture so far

```text
arXiv API --> Airflow DAG (Mon-Fri 06:00 UTC)
                 |  fetch metadata (3 s rate limit)
                 |  download PDFs (cached in data/arxiv_pdfs)
                 |  parse with Docling -> markdown + sections
                 v
            Neon Postgres (papers table, source of truth)
                 |  index task: every paper, full text when parsed
                 v
            OpenSearch  arxiv-papers -> arxiv-papers-v1        (paper-level BM25)
                        arxiv-papers-chunks-<provider>-1024-v1  (chunks: BM25 text + kNN vector)
                 ^                       ^
                 |  /api/v1/search       |  /api/v1/hybrid-search
            FastAPI  (papers, BM25)      (chunks: BM25 + vectors fused with RRF;
                 |                        query embedded with OpenAI or Jina)
                 |
                 |  /api/v1/ask, /api/v1/ask/stream
                 v
            top-k chunks -> numbered passages -> LLM (OpenAI or Bedrock) -> answer with [n] citations
                 ^
            Gradio UI (ui/gradio_app.py, :7860) streams from /ask/stream
```

## Containers

| Service | URL | Purpose |
| --- | --- | --- |
| api | http://localhost:8000/docs | FastAPI |
| opensearch | http://localhost:9200 | Search engine |
| opensearch-dashboards | http://localhost:5601 | Search UI / Dev Tools |
| airflow | http://localhost:8080 | DAGs (login `admin` / `AIRFLOW_ADMIN_PASSWORD`) |
| Gradio UI (not in Docker) | http://localhost:7860 | Ask questions: `uv run --group ui python ui/gradio_app.py` |
| airflow-db | internal | Airflow's own metadata |
| redis | localhost:6379 | RAG answer cache (128 MB, LRU, no persistence) |

## Run it

```powershell
uv sync --group ingest                       # Python deps (+ Docling for notebooks)
Copy-Item .env.example .env                  # then fill in real values
docker compose up --build -d
uv run python ui/gradio_app.py         Gradio front end (talks to the API over HTTP only)
ui/Dockerfile            UI image for production
compose.prod.yml         EC2 overrides: nginx + ui, private ports, log caps
deploy/nginx.conf        basic auth, rate limit, SSE-friendly proxy
deploy/production.env    non-secret production settings (rendered into .env on deploy)
deploy/remote_deploy.sh  server-side deploy step: install .env, rebuild, wait for /health
deploy/ci-iam-policy.json least-privilege policy for the deploy job's IAM user
.github/workflows/       ci-cd.yml (CI + deploy), eval.yml (weekly golden evaluation)
src/eval/                metrics (retrieval, answers, behaviour) + DeepEval metrics
data/golden_dataset.json 25 goldens + thresholds
scripts/
  run_eval.py            run the goldens against the API, write a report
  test_connections.py    check Neon, OpenSearch, arXiv, Jina, OpenAI
  index_papers.py        backfill / rebuild the search indexes (papers + chunks)    # all should be OK
uv run python ui/gradio_app.py         Gradio front end (talks to the API over HTTP only)
ui/Dockerfile            UI image for production
compose.prod.yml         EC2 overrides: nginx + ui, private ports, log caps
deploy/nginx.conf        basic auth, rate limit, SSE-friendly proxy
deploy/production.env    non-secret production settings (rendered into .env on deploy)
deploy/remote_deploy.sh  server-side deploy step: install .env, rebuild, wait for /health
deploy/ci-iam-policy.json least-privilege policy for the deploy job's IAM user
.github/workflows/       ci-cd.yml (CI + deploy), eval.yml (weekly golden evaluation)
scripts/index_papers.py        # backfill papers already in Neon into OpenSearch
uv run pytest -q
```

## Search

```powershell
# Browser / quick check
curl "http://localhost:8000/api/v1/search?q=coding+agents&size=5"

# Full request: filters, paging. Empty query = newest papers first.
curl -X POST http://localhost:8000/api/v1/search -H "Content-Type: application/json" `
  -d '{"query": "llm agents", "categories": ["cs.AI"], "date_from": "2026-11-01", "size": 5}'
```

### Hybrid search (chunks)

```powershell
curl -X POST http://localhost:8000/api/v1/hybrid-search -H "Content-Type: application/json" `
  -d '{"query": "robots learning to walk from simulated data", "mode": "hybrid", "size": 5}'
```

`mode` is `hybrid` (default), `bm25` or `vector`. Each hit shows its rank in both lists
(`bm25_rank`, `vector_rank`), so you can see why it ranked where it did.

- **Chunking:** chunk 0 = title + abstract (every paper has one); full-text sections are packed into
  ~400-word chunks with 50-word overlap; references/acknowledgements are skipped. Each chunk is
  embedded with its paper title and section name for context.
- **Fusion:** Reciprocal Rank Fusion, `score = sum 1/(60 + rank)`, done in Python so it's testable and
  doesn't depend on OpenSearch search pipelines.

## Ask questions (RAG)

```powershell
# One JSON answer
curl -X POST http://localhost:8000/api/v1/ask -H "Content-Type: application/json" `
  -d '{"query": "How are coding agents being benchmarked?", "top_k": 5}'

# Streamed (server-sent events: sources, then tokens, then done)
curl -N -X POST http://localhost:8000/api/v1/ask/stream -H "Content-Type: application/json" `
  -d '{"query": "How do robots learn motion skills from simulation?"}'

# Web UI
uv run --group ui python ui/gradio_app.py      # http://localhost:7860
```

- The LLM sees only the retrieved passages, numbered `[1]..[k]`, and must cite them; `cited` in the
  response lists the sources it actually used. Off-topic questions get "not in the passages".
- If retrieval finds nothing, the LLM isn't called at all (no cost, no invented answer).
- Accepts the same `mode`, `categories`, `date_from`, `date_to` as `/hybrid-search`.

## Agent (LangGraph)

```text
START -> route --search--> retrieve -> grade --relevant--> generate -> END
           |                  ^           |--none, tries left--> rewrite --> (retrieve)
           |                  |           '--none, no tries----> give_up -> END
           '--direct--> direct -> END
```

```powershell
curl -X POST http://localhost:8000/api/v1/agent/ask -H "Content-Type: application/json" `
  -d '{"query": "How do humanoid robots learn motion skills from simulation?"}'
# Streamed: POST /api/v1/agent/stream  (events: step, sources, token, done)
```

- **route**: greetings, thanks and off-topic requests get a short direct reply (no search, no invented research).
- **grade**: one LLM call keeps only passages that actually help; weakly related hits are never cited.
- **rewrite**: if nothing relevant came back, the question is rewritten into a better search query
  (never one already tried) and searched again, up to `AGENT__MAX_RETRIEVALS` (default 2).
- **give_up**: if still nothing, it says so and lists the queries it tried, rather than answering from
  unrelated passages (which plain `/ask` would do).
- The response's `steps` and `search_queries` show every decision; each node is a span in Langfuse.
- Uses the project's own LLM client and hybrid search, so it follows `LLM_PROVIDER` and
  `EMBEDDINGS__PROVIDER`. The Gradio UI has an **Agent mode** checkbox (on by default) that shows the
  steps live.

| | `/ask` (plain RAG) | `/agent/ask` |
| --- | --- | --- |
| LLM calls | 1 | 2-6 |
| Typical latency | 3-10 s | 3-12 s |
| Cached | yes (Redis) | no |
| Off-topic / unanswerable | answers from whatever was retrieved | says so honestly |

### Cache and tracing

- **Redis cache:** `/ask` and `/ask/stream` share one cache. The key covers the normalised question,
  retrieval options, LLM + embedding models and the **chunk count**, so newly indexed papers
  invalidate old answers automatically. TTL 6 h (`REDIS__TTL_SECONDS`). A repeat question returns in
  ~25 ms instead of ~5-10 s, with `cached: true`. If Redis is down, requests just skip the cache.
- **Langfuse:** one trace per question (`rag.ask` -> `cache` -> `retrieval` -> `llm`), with the full
  prompt, answer, model, token usage, cost and time to first token. Set `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` in `.env`; without them tracing is a no-op. Each response
  includes its `trace_id`.

### Logfire (app monitoring) next to Langfuse (LLM tracing)

| | Langfuse | Logfire |
| --- | --- | --- |
| Sees | prompts, answers, tokens, cost, agent steps | HTTP requests, Neon SQL, Redis, OpenAI/Jina HTTP timings, OpenSearch + embedding spans |
| Use it to | debug answers and cost | find slow requests, errors, slow queries |

- Set `LOGFIRE__TOKEN` (a write token from logfire.pydantic.dev -> project -> Settings -> Write tokens);
  without it nothing is sent. `/ping` and `/health` aren't traced (health-check noise).
- Logfire owns the global OpenTelemetry provider; Langfuse runs on its **own** provider and starts each
  trace in an empty context, so neither receives the other's spans and Langfuse traces are always clean
  roots. The Logfire request span carries `langfuse.trace_id` to jump between the two.
- SQL is instrumented at the psycopg2 driver (the SQLAlchemy instrumentor doesn't support SQLAlchemy 2.1).
- httpx bodies and auth headers are **not** captured, so prompts and API keys never reach Logfire.

### AWS Bedrock Guardrails

Applied with the standalone `ApplyGuardrail` API, so they work with OpenAI as well as Bedrock:

| Stage | What | Effect |
| --- | --- | --- |
| Question (before search/LLM/cache) | harmful content, prompt attacks, denied topics | refused with `GUARDRAILS__BLOCKED_INPUT_MESSAGE` |
| | PII (email, phone...) | masked; the masked question is what gets searched, cached and traced |
| Answer (before caching) | harmful content, PII | blocked / masked |
| | contextual grounding + relevance vs. the passages | `flag` (default: warning, answer kept), `block`, or `off` |

- Credentials: the EC2 instance role (no keys on the server); `~/.aws` locally. No credentials = off.
- The grounding check receives the passages **exactly as the LLM saw them** (title, arXiv id, section);
  with bare text, correct answers that mention an arXiv id scored 0.08 instead of 0.98. Answers that
  cite nothing (e.g. "the passages don't cover this") skip grounding, since there's nothing to verify.
- Streaming: the answer is checked after it's streamed, so a `guardrail` event tells the client to
  flag or replace it. Every response has a `guardrail` field; each check is a span in Langfuse.
- If AWS is unreachable, `GUARDRAILS__FAIL_OPEN=true` (default) lets requests through and logs it.

### Switching LLM provider

`LLM_PROVIDER=openai` uses `OPENAI_MODEL` (default `gpt-4o-mini`). `LLM_PROVIDER=bedrock` uses the
Bedrock Converse API with `BEDROCK_MODEL_ID` and `AWS_REGION`; credentials come from the normal AWS
chain (`aws configure`, env vars, or an EC2 instance role). Restart the API after changing it:
`docker compose up -d api`. `RAG__TOP_K`, `RAG__MAX_TOKENS` and `RAG__TEMPERATURE` tune generation.

### Switching embeddings provider

Set `EMBEDDINGS__PROVIDER=openai` or `jina` in `.env`, then:

```powershell
docker compose up -d api airflow             # reload .env
uv run python ui/gradio_app.py         Gradio front end (talks to the API over HTTP only)
ui/Dockerfile            UI image for production
compose.prod.yml         EC2 overrides: nginx + ui, private ports, log caps
deploy/nginx.conf        basic auth, rate limit, SSE-friendly proxy
deploy/production.env    non-secret production settings (rendered into .env on deploy)
deploy/remote_deploy.sh  server-side deploy step: install .env, rebuild, wait for /health
deploy/ci-iam-policy.json least-privilege policy for the deploy job's IAM user
.github/workflows/       ci-cd.yml (CI + deploy), eval.yml (weekly golden evaluation)
scripts/index_papers.py        # build that provider's chunk index (once)
```

Each provider has its own index (`...-openai-1024-v1`, `...-jina-1024-v1`) and the index stores the
exact model in its `_meta`, so vectors from different models are never mixed. Switching back is
instant if the other index was already built.

### Index versioning

The app always queries the alias `arxiv-papers`. To change the mapping, bump `INDEX_VERSION` in
`src/services/opensearch/index_config.py`, build the new index, reindex, and move the alias.

## Troubleshooting

- **`could not translate host name ...neon.tech`**: your ISP/router DNS can't resolve Neon. The
  containers use 8.8.8.8 / 1.1.1.1 (see `dns:` in `compose.yml`); for scripts on your machine, set
  Windows DNS to 8.8.8.8 / 1.1.1.1.
- **PDF parsing dies silently**: Docling needs ~2.5 GB. Give Docker more memory and swap in
  `%UserProfile%\.wslconfig` (`memory=5GB`, `swap=8GB`), then `wsl --shutdown`.
- **`localhost` ports stop answering after a WSL restart**: `docker compose down; docker compose up -d`.
```

## Evaluation

`data/golden_dataset.json` holds 25 golden questions written against the indexed papers: single-paper
facts and numbers, full-text sections, multi-paper questions, vague/keyword/semantic wording, a
category filter, an unanswerable research question, off-topic, prompt injection and PII. Each golden
lists the relevant papers, facts that must appear, traps that must not, the expected behaviour and a
reference answer.

```powershell
uv run python scripts/run_eval.py                                        # local API, agent mode
uv run python scripts/run_eval.py --mode rag                             # plain RAG endpoint
uv run python scripts/run_eval.py --base-url http://<ip> --auth arxiv:<password>
uv run python scripts/run_eval.py --ids gd-001,gd-018 --no-judge --gate  # subset, no judge, CI exit code
```

| Group | Metrics |
| --- | --- |
| Retrieval (paper level) | Hit@1/3/5, Recall@5, Precision@5, MRR, nDCG@5 |
| Answer quality (DeepEval, 0-1) | GEval correctness vs. reference, Faithfulness, Answer Relevancy, Contextual Precision / Recall / Relevancy |
| Answer quality (deterministic) | fact recall (required facts present), Guardrails grounding + relevance |
| Citations | citation count, citation precision (cited papers that are relevant) |
| Behaviour & safety | behaviour accuracy, route accuracy, decline accuracy, false-refusal rate, guardrail accuracy, forbidden-string violations |
| Efficiency | latency p50/p95, LLM calls per question, rewrite rate |

DeepEval's judge defaults to `gpt-4o` (`--judge-model`). Measured alternatives: `gpt-4o-mini` ignored the
GEval steps and marked a verbatim-supported claim as unfaithful; `gpt-4.1` judged well but hung inside
DeepEval's faithfulness/relevancy calls in 2 of 3 rounds. A full run costs roughly $1-2 of judge calls;
`--no-judge` skips them. `[n]` citation markers are stripped before scoring (DeepEval counted them as
irrelevant statements), and judge calls time out after 30 s with 4 attempts (`src/services/deepeval_setup.py`).

Thresholds live in the dataset file; `--gate` exits 1 if any is missed. Reports (JSON + Markdown) go
to `eval_reports/`. `tests/eval/` validates the dataset itself offline.

### Online evals (DeepEval, scores in Langfuse)

A sample of live answers is scored in production with DeepEval's reference-free metrics, and each
score is attached to that request's Langfuse trace as `deepeval_faithfulness`,
`deepeval_answer_relevancy` and `deepeval_contextual_relevancy` (0-1, DeepEval's reason as the
comment). Langfuse then charts them over time and lets you filter low-scoring answers, next to the
prompt, tokens and cost.

- Needs Langfuse and `OPENAI_API_KEY` (judge). No extra account.
- `ONLINE_EVAL__SAMPLE_RATE` (default 0.2), `ONLINE_EVAL__JUDGE_MODEL` (default `gpt-4o`),
  `ONLINE_EVAL__MAX_PENDING` (default 20: extra samples are dropped rather than queued without limit).
- Scored in a background thread after the answer is sent: no added latency, streaming untouched.
- Only cited, passage-grounded answers are scored; refusals, direct replies, guardrail blocks and cache
  hits have nothing to judge. DeepEval's usage telemetry is opted out.
- Why not Confident AI (DeepEval's hosted platform)? Its sign-up requires an organisation email; the
  same DeepEval metrics run in-process instead and reuse the existing Langfuse dashboard.

## Deploy to AWS EC2

One `t3.large` (2 vCPU, 8 GB) in `us-east-2` (same region as Neon), Ubuntu, 40 GB gp3, Elastic IP,
an IAM role with Bedrock access (no AWS keys on the server).

```text
internet --80--> nginx (basic auth, 20 req/min per IP on /api/)
                   |-- /          -> ui (Gradio)  --> api
                   |-- /api/, /docs -> api (FastAPI)
           127.0.0.1 only: api :8000, opensearch :9200, redis :6379, airflow :8080
```

**Security group:** 22 from your IP only; 80 from anywhere (password-protected). Nothing else.

**One-time server setup** (Docker, 4 GB swap, `vm.max_map_count=262144` for OpenSearch):

```bash
sudo apt-get install -y docker.io docker-compose-v2 apache2-utils && sudo usermod -aG docker ubuntu
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo "vm.max_map_count=262144" | sudo tee /etc/sysctl.d/99-opensearch.conf && sudo sysctl --system
```

**Deploy / update** (from your machine, then on the server):

```bash
tar --exclude=./.venv --exclude=./.git --exclude='./data/*' -czf - . | ssh -i key.pem ubuntu@<ip> 'mkdir -p ~/arxiv-rag && tar -xzf - -C ~/arxiv-rag'
ssh -i key.pem ubuntu@<ip>
cd ~/arxiv-rag
htpasswd -bcB deploy/.htpasswd arxiv '<password>'        # first time only
docker compose -f compose.yml -f compose.prod.yml up -d --build
```

- UI: `http://<ip>/` - API docs: `http://<ip>/docs` - both ask for the password.
- Airflow: `ssh -i key.pem -L 8080:localhost:8080 ubuntu@<ip>`, then open http://localhost:8080.
- Stop the instance when not in use; data lives on the EBS volume and Neon.

## CI/CD (GitHub Actions)

| Workflow | When | What |
| --- | --- | --- |
| `ci-cd.yml` | every push and PR | ruff lint + format check, full pytest suite, build API/UI images (Airflow image only when `airflow/` changed), validate the production compose file, start the API image and hit `/ping` |
| `ci-cd.yml` (deploy job) | push to `main`, or "Run workflow" on `main` | open SSH for the runner's IP only, rsync the code, render `.env` from `deploy/production.env` + secrets, run `deploy/remote_deploy.sh` (rebuild, wait for `/health`), smoke-test the public URL, always close SSH |
| `eval.yml` | Mondays 09:00 UTC, or on demand | the 25 goldens against production with DeepEval and `--gate`; report in the run summary + artifact |

**Configuration** lives in GitHub -> Settings -> Environments -> `production`:

| Kind | Name | Value |
| --- | --- | --- |
| Variable | `EC2_HOST` | the Elastic IP |
| Variable | `EC2_SECURITY_GROUP_ID` | `sg-...` of the instance's security group |
| Secret | `EC2_SSH_KEY` | full contents of the `.pem` file |
| Secret | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | IAM user with only `deploy/ci-iam-policy.json` |
| Secret | `UI_PASSWORD` | nginx basic-auth password for user `arxiv` |
| Secret | `POSTGRES_DATABASE_URL`, `OPENAI_API_KEY`, `JINA_API_KEY` (optional) | app |
| Secret | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LOGFIRE__TOKEN` | tracing |
| Secret | `GUARDRAILS__ID`, `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_JWT_SECRET` | guardrails, Airflow |

Non-secret settings (models, sample rate, regions, guardrail version...) are in `deploy/production.env`,
reviewed like code. The deploy job checks every required secret first and names any that are missing.

## Project layout

```text
src/
  config.py              settings from .env (nested with __)
  main.py                FastAPI app + lifespan
  dependencies.py        Depends() providers (overridden in tests)
  db/database.py         engine, sessions, Base
  models/paper.py        papers table
  repositories/paper.py  all paper SQL
  routers/ping.py        /ping and /health
  routers/search.py      /search (GET and POST)
  routers/hybrid_search.py /hybrid-search
  routers/ask.py         /ask and /ask/stream (SSE)
  routers/agent.py       /agent/ask and /agent/stream
  schemas/               Pydantic models
  services/
    arxiv/client.py      rate-limited arXiv client + PDF download
    pdf_parser/parser.py Docling parser -> sections
    opensearch/
      index_config.py    versioned mapping (english stemming, author names)
      query_builder.py   BM25 query + filters (pure functions, reused in Phase 4)
      client.py          index setup, bulk upsert, search
    metadata_fetcher.py  fetch -> store -> download -> parse -> store
      chunk_index.py     per-provider chunk index (kNN + BM25), model check
    embeddings/client.py OpenAI + Jina behind one interface, retries, batching
    chunking.py          section-aware chunks with overlap
    hybrid_search.py     BM25 + kNN, RRF fusion
    llm/client.py        OpenAI (httpx, SSE) + Bedrock (Converse) behind one interface
    rag.py               retrieve -> prompt with numbered passages -> answer + citations
    cache.py             Redis answer cache (key includes corpus size)
    tracing.py           Langfuse traces (no-op without keys)
    observability.py     Logfire setup: FastAPI, psycopg2, httpx, Redis
    guardrails.py        AWS Bedrock Guardrails: input/output checks, grounding
    online_eval.py       DeepEval online evals on sampled live answers -> Langfuse scores
    agent/graph.py       LangGraph agent: route, retrieve, grade, rewrite, generate
    indexing.py          Postgres papers -> paper index + chunk index
airflow/
  Dockerfile, start.sh   Airflow 3 image with CPU torch + Docling
  dags/                  arxiv_paper_ingestion DAG + task helpers
ui/gradio_app.py         Gradio front end (talks to the API over HTTP only)
ui/Dockerfile            UI image for production
compose.prod.yml         EC2 overrides: nginx + ui, private ports, log caps
deploy/nginx.conf        basic auth, rate limit, SSE-friendly proxy
deploy/production.env    non-secret production settings (rendered into .env on deploy)
deploy/remote_deploy.sh  server-side deploy step: install .env, rebuild, wait for /health
deploy/ci-iam-policy.json least-privilege policy for the deploy job's IAM user
.github/workflows/       ci-cd.yml (CI + deploy), eval.yml (weekly golden evaluation)
src/eval/                metrics (retrieval, answers, behaviour) + DeepEval metrics
data/golden_dataset.json 25 goldens + thresholds
scripts/
  run_eval.py            run the goldens against the API, write a report
  test_connections.py    check Neon, OpenSearch, arXiv, Jina, OpenAI
  index_papers.py        backfill / rebuild the search indexes (papers + chunks)
tests/                   unit + API tests (no network needed)
```
