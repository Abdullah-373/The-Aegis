"""SQLite cache for The Aegis verdicts."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./aegis_cache.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VerdictCache(Base):
    __tablename__ = "verdict_cache"

    id = Column(Integer, primary_key=True, index=True)
    pdf_filename = Column(String, unique=True, index=True, nullable=False)
    content_hash = Column(String, index=True, nullable=False)
    strategist_output = Column(Text, nullable=False)
    red_team_output = Column(Text, nullable=False)
    judge_output = Column(Text, nullable=False)
    verdict = Column(String, nullable=False)
    execution_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
