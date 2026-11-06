import os

import pytest

# Tests never touch real services: point everything at throwaway defaults.
os.environ.setdefault("POSTGRES_DATABASE_URL", "sqlite://")
os.environ.setdefault("OPENSEARCH__HOST", "http://localhost:9200")
os.environ.setdefault("ARXIV__RATE_LIMIT_DELAY", "0")
# Env vars beat .env, so tests never send traces or hit a real Redis.
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["REDIS__ENABLED"] = "false"
os.environ["LOGFIRE__ENABLED"] = "false"
os.environ["GUARDRAILS__ENABLED"] = "false"
os.environ["ONLINE_EVAL__ENABLED"] = "false"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"


@pytest.fixture
def database(tmp_path):
    from src.db.database import Database

    db = Database(url=f"sqlite:///{tmp_path / 'test.db'}")
    db.create_tables()
    yield db
    db.close()
