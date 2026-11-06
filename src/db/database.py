"""SQLAlchemy engine, session factory and declarative base."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class Database:
    """Owns one engine and hands out sessions. Create one per process."""

    def __init__(self, settings: Settings | None = None, url: str | None = None):
        settings = settings or get_settings()
        db_url = url or settings.postgres_database_url
        # SQLAlchemy 2.1 maps a bare "postgresql://" to psycopg (v3); we ship psycopg2-binary.
        if db_url.startswith(("postgresql://", "postgres://")):
            db_url = "postgresql+psycopg2://" + db_url.split("://", 1)[1]
        engine_kwargs: dict = {"echo": settings.postgres_echo_sql, "future": True}
        if db_url.startswith("postgresql"):
            # pool_pre_ping: Neon closes idle connections, so test each one before use.
            engine_kwargs |= {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 5, "pool_recycle": 300}
        self.engine: Engine = create_engine(db_url, **engine_kwargs)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        # Import models so they register on Base.metadata before create_all.
        from src.models import paper  # noqa: F401

        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Session that commits on success and rolls back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        self.engine.dispose()
