import importlib.util
from datetime import date, datetime
from pathlib import Path

# Load the DAG helper module directly (it lives under airflow/dags, not src/).
_path = Path(__file__).resolve().parents[2] / "airflow" / "dags" / "arxiv_ingestion" / "tasks.py"
_spec = importlib.util.spec_from_file_location("arxiv_ingestion_tasks", _path)
tasks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tasks)


def test_default_window_is_three_days_ending_yesterday():
    assert tasks.resolve_date_window(datetime(2026, 11, 5, 6), None, None) == (date(2026, 11, 2), date(2026, 11, 4))


def test_explicit_single_date():
    assert tasks.resolve_date_window(None, "2026-10-01", None) == (date(2026, 10, 1), date(2026, 10, 1))
