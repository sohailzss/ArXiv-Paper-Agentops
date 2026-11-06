"""Daily arXiv ingestion: Mon-Fri 06:00 UTC.

setup -> fetch_and_process -> index -> report
(index pushes the run's papers into the BM25 paper index and the chunk + embedding index.)

Trigger manually from the UI with params, e.g. {"from_date": "2026-11-02", "max_results": 5}.
"""

from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task

from arxiv_ingestion import tasks


@dag(
    dag_id="arxiv_paper_ingestion",
    description="Fetch cs.AI papers from arXiv, parse PDFs with Docling, store in Postgres, index in OpenSearch",
    schedule="0 6 * * 1-5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    params={
        "from_date": Param(None, type=["null", "string"], description="YYYY-MM-DD; blank = 3 days ending yesterday"),
        "to_date": Param(None, type=["null", "string"], description="YYYY-MM-DD; blank = same as from_date"),
        "max_results": Param(15, type="integer", minimum=1, maximum=200),
        "process_pdfs": Param(True, type="boolean"),
    },
    tags=["arxiv", "ingestion"],
)
def arxiv_paper_ingestion():
    @task
    def setup() -> dict:
        return tasks.setup_environment()

    @task(execution_timeout=timedelta(hours=2))
    def fetch_and_process(_setup: dict) -> dict:
        context = get_current_context()
        params = context["params"]
        start, end = tasks.resolve_date_window(
            context.get("logical_date"), params.get("from_date"), params.get("to_date")
        )
        return tasks.fetch_and_process(start, end, params["max_results"], params["process_pdfs"])

    @task(execution_timeout=timedelta(minutes=30))
    def index(stats: dict) -> dict:
        return tasks.index_papers(stats)

    @task
    def report(stats: dict) -> dict:
        return tasks.build_report(stats)

    report(index(fetch_and_process(setup())))


arxiv_paper_ingestion()
