"""SQLite cache for The Aegis verdicts (with WAL mode for better concurrency)."""
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./aegis_cache.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _record):
    """Enable WAL mode so reads don't block writes (better concurrency)."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


class VerdictCache(Base):
    __tablename__ = "verdict_cache"

    id = Column(Integer, primary_key=True, index=True)
    pdf_filename = Column(String, index=True, nullable=False)
    content_hash = Column(String, unique=True, index=True, nullable=False)
    model_used = Column(String, nullable=False, default="gemini-2.5-flash")
    alex_output = Column(Text, nullable=False)
    sam_output = Column(Text, nullable=False)
    maya_output = Column(Text, nullable=False)
    verdict = Column(String, nullable=False)
    risk_score = Column(Integer, nullable=False, default=50)
    headline = Column(String, nullable=False, default="")
    structured_json = Column(Text, nullable=False, default="{}")
    execution_time = Column(Float, nullable=False)
    total_tokens = Column(Integer, nullable=False, default=0)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    truncated = Column(Boolean, nullable=False, default=False)
    chunked = Column(Boolean, nullable=False, default=False)
    pdf_chars = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
