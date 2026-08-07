from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apila.db"


def get_engine(db_path: str | Path | None = None, *, echo: bool = False) -> Engine:
    """SQLite engine for the point-in-time store. Pass ':memory:' for tests."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine: Engine) -> Session:
    return sessionmaker(bind=engine)()
