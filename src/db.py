import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash

project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / ".env")

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New chat")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="chats")
    messages = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    files = relationship(
        "ChatFile",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="ChatFile.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    chat = relationship("Chat", back_populates="messages")


class ChatFile(Base):
    __tablename__ = "chat_files"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    stored_path = Column(String(500), nullable=False)
    file_type = Column(String(40), nullable=False, default="pdf")
    indexed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    chat = relationship("Chat", back_populates="files")


def _normalize_database_url(url: str) -> str:
    if "channel_binding=" in url:
        parts = url.split("?")
        if len(parts) == 2:
            params = [p for p in parts[1].split("&") if not p.startswith("channel_binding=")]
            url = parts[0] + (("?" + "&".join(params)) if params else "")
    return url


def _build_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    else:
        connect_args = {"connect_timeout": 5}
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args=connect_args,
    )


def _sqlite_url() -> str:
    db_path = (project_root / "examind_local.db").as_posix()
    return f"sqlite:///{db_path}"


DATABASE_URL = os.getenv("DATABASE_URL")
engine = None
SessionLocal = None
DB_BACKEND = "unknown"


def init_db() -> str:
    """Create tables. Prefer Neon/Postgres; fall back to local SQLite if unreachable."""
    global engine, SessionLocal, DB_BACKEND, DATABASE_URL

    candidates = []
    if DATABASE_URL:
        candidates.append(("postgres", _normalize_database_url(DATABASE_URL)))
    candidates.append(("sqlite", _sqlite_url()))

    last_error = None
    for backend, url in candidates:
        try:
            test_engine = _build_engine(url)
            with test_engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            engine = test_engine
            SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
            Base.metadata.create_all(bind=engine)

            if backend == "sqlite":
                @event.listens_for(engine, "connect")
                def _set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()

            DB_BACKEND = backend
            DATABASE_URL = url
            print(f"[INFO] Database ready ({backend}).")
            return backend
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Could not connect via {backend}: {exc}")

    raise RuntimeError(f"Database init failed. Last error: {last_error}")


def get_session():
    if SessionLocal is None:
        init_db()
    return SessionLocal()
