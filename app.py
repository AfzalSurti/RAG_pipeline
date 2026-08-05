import os
import sys
import threading
import uuid
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, session
from sqlalchemy import desc
from werkzeug.utils import secure_filename

os.environ.setdefault("PYTHONNOUSERSITE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR = "CopyOfExam"
PERSIST_DIR = "faiss_store_exam"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx", ".xlsx"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_rag_lock = threading.Lock()
_rag = None
_rag_ready = False
_rag_loading = False
_rag_error = None


def _is_venv_active() -> bool:
    return ".venv" in sys.executable.replace("\\", "/")


def _get_rebuild_flag() -> bool:
    return not (
        os.path.exists(os.path.join(PERSIST_DIR, "faiss.index"))
        and os.path.exists(os.path.join(PERSIST_DIR, "metadata.pkl"))
    )


def _initialize_rag() -> None:
    global _rag, _rag_ready, _rag_loading, _rag_error
    with _rag_lock:
        if _rag_ready or _rag_loading:
            return
        _rag_loading = True

    try:
        from src.search import RAGSearch

        rebuild = _get_rebuild_flag()
        _rag = RAGSearch(
            persist_dir=PERSIST_DIR,
            data_dir=DATA_DIR,
            rebuild_index=rebuild,
        )
        _rag_ready = True
        _rag_error = None
    except Exception as exc:
        _rag_error = str(exc)
        _rag_ready = False
    finally:
        with _rag_lock:
            _rag_loading = False


def _initialize_rag_async() -> None:
    threading.Thread(target=_initialize_rag, daemon=True).start()


def _ensure_rag_ready() -> None:
    if _rag_ready:
        return
    _initialize_rag()


def _run_cli_mode() -> None:
    from src.search import RAGSearch

    if not _is_venv_active():
        print("[WARN] You are not using the project virtual environment.")
        print("[WARN] Run with: D:/RAG_pipeline/.venv/Scripts/python.exe app.py --cli")

    should_rebuild = _get_rebuild_flag()
    rag_search = RAGSearch(
        persist_dir=PERSIST_DIR,
        data_dir=DATA_DIR,
        rebuild_index=should_rebuild,
    )
    print(f"[INFO] RAG ready for exam files in: {DATA_DIR}")
    while True:
        user_query = input("\nAsk your exam query (or type 'exit'): ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue
        print("\nAnswer:", rag_search.search_and_summarize(user_query, top_k=8))


app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.secret_key = os.getenv("SECRET_KEY", "examind-dev-secret")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 14,
)


def _db():
    from src.db import get_session

    return get_session()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return fn(*args, **kwargs)

    return wrapper


def _serialize_user(user):
    return {"id": user.id, "name": user.name, "email": user.email}


def _serialize_chat(chat, include_files=False):
    data = {
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
    }
    if include_files:
        data["files"] = [
            {
                "id": f.id,
                "filename": f.filename,
                "file_type": f.file_type,
                "indexed": f.indexed,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in chat.files
        ]
    return data


def _serialize_message(msg):
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


def _get_user_chat(db, chat_id: int, user_id: int):
    from src.db import Chat

    return db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    if _rag_error:
        return jsonify({"ready": False, "loading": False, "error": _rag_error}), 500
    return jsonify({"ready": _rag_ready, "loading": _rag_loading, "error": None}), 200


@app.route("/api/auth/signup", methods=["POST"])
def api_signup():
    from src.db import User

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not name or not email or len(password) < 6:
        return jsonify({"error": "Name, email, and password (min 6 chars) are required."}), 400

    db = _db()
    try:
        if db.query(User).filter(User.email == email).first():
            return jsonify({"error": "Email already registered."}), 409
        user = User(name=name, email=email)
        user.set_password(password)
        db.add(user)
        db.commit()
        db.refresh(user)
        session.clear()
        session["user_id"] = user.id
        session.permanent = True
        return jsonify({"user": _serialize_user(user)}), 201
    except Exception as exc:
        db.rollback()
        return jsonify({"error": str(exc)}), 500
    finally:
        db.close()


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    from src.db import User

    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    db = _db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid email or password."}), 401
        session.clear()
        session["user_id"] = user.id
        session.permanent = True
        return jsonify({"user": _serialize_user(user)}), 200
    finally:
        db.close()


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True}), 200


@app.route("/api/auth/me", methods=["GET"])
def api_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"user": None}), 200

    from src.db import User

    db = _db()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            session.clear()
            return jsonify({"user": None}), 200
        return jsonify({"user": _serialize_user(user)}), 200
    finally:
        db.close()


@app.route("/api/chats", methods=["GET"])
@login_required
def api_list_chats():
    from src.db import Chat

    db = _db()
    try:
        chats = (
            db.query(Chat)
            .filter(Chat.user_id == session["user_id"])
            .order_by(desc(Chat.updated_at))
            .all()
        )
        return jsonify({"chats": [_serialize_chat(c) for c in chats]}), 200
    finally:
        db.close()


@app.route("/api/chats", methods=["POST"])
@login_required
def api_create_chat():
    from src.db import Chat, utcnow

    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "New chat").strip() or "New chat"

    db = _db()
    try:
        chat = Chat(user_id=session["user_id"], title=title[:255], updated_at=utcnow())
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return jsonify({"chat": _serialize_chat(chat, include_files=True)}), 201
    except Exception as exc:
        db.rollback()
        return jsonify({"error": str(exc)}), 500
    finally:
        db.close()


@app.route("/api/chats/<int:chat_id>", methods=["GET"])
@login_required
def api_get_chat(chat_id: int):
    from src.db import Message

    db = _db()
    try:
        chat = _get_user_chat(db, chat_id, session["user_id"])
        if not chat:
            return jsonify({"error": "Chat not found."}), 404
        messages = (
            db.query(Message)
            .filter(Message.chat_id == chat.id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return jsonify(
            {
                "chat": _serialize_chat(chat, include_files=True),
                "messages": [_serialize_message(m) for m in messages],
            }
        ), 200
    finally:
        db.close()


@app.route("/api/chats/<int:chat_id>", methods=["DELETE"])
@login_required
def api_delete_chat(chat_id: int):
    db = _db()
    try:
        chat = _get_user_chat(db, chat_id, session["user_id"])
        if not chat:
            return jsonify({"error": "Chat not found."}), 404
        db.delete(chat)
        db.commit()
        return jsonify({"ok": True}), 200
    except Exception as exc:
        db.rollback()
        return jsonify({"error": str(exc)}), 500
    finally:
        db.close()


@app.route("/api/chats/<int:chat_id>/upload", methods=["POST"])
@login_required
def api_upload(chat_id: int):
    from src.data_loader import load_single_document
    from src.db import ChatFile, Message, utcnow

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    original = secure_filename(file.filename)
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    db = _db()
    try:
        chat = _get_user_chat(db, chat_id, session["user_id"])
        if not chat:
            return jsonify({"error": "Chat not found."}), 404

        chat_upload_dir = UPLOAD_DIR / str(session["user_id"]) / str(chat_id)
        chat_upload_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex}_{original}"
        stored_path = chat_upload_dir / stored_name
        file.save(stored_path)

        _ensure_rag_ready()
        if not _rag_ready or _rag is None:
            return jsonify({"error": _rag_error or "Pipeline is still initializing. Retry shortly."}), 503

        docs = load_single_document(str(stored_path))
        chunks_added = _rag.ingest_chat_documents(chat_id, docs)

        chat_file = ChatFile(
            chat_id=chat.id,
            filename=original,
            stored_path=str(stored_path),
            file_type=suffix.lstrip("."),
            indexed=True,
        )
        db.add(chat_file)

        if chat.title in {"New chat", "Untitled"}:
            chat.title = original[:80]

        notice = (
            f"Uploaded **{original}** and indexed {chunks_added} chunks. "
            "Ask questions from this paper/chapter anytime in this chat."
        )
        db.add(Message(chat_id=chat.id, role="system", content=notice))
        chat.updated_at = utcnow()
        db.commit()
        db.refresh(chat_file)

        return jsonify(
            {
                "file": {
                    "id": chat_file.id,
                    "filename": chat_file.filename,
                    "file_type": chat_file.file_type,
                    "indexed": chat_file.indexed,
                    "chunks_added": chunks_added,
                },
                "notice": notice,
                "chat": _serialize_chat(chat, include_files=True),
            }
        ), 201
    except Exception as exc:
        db.rollback()
        return jsonify({"error": str(exc)}), 500
    finally:
        db.close()


@app.route("/api/chats/<int:chat_id>/messages", methods=["POST"])
@login_required
def api_chat_message(chat_id: int):
    from src.db import Message, utcnow

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    top_k = int(payload.get("top_k") or 8)

    if not message:
        return jsonify({"error": "Message is required."}), 400

    db = _db()
    try:
        chat = _get_user_chat(db, chat_id, session["user_id"])
        if not chat:
            return jsonify({"error": "Chat not found."}), 404

        if _rag_error:
            return jsonify({"error": f"Startup error: {_rag_error}"}), 500

        _ensure_rag_ready()
        if _rag_error:
            return jsonify({"error": f"Startup error: {_rag_error}"}), 500
        if not _rag_ready or _rag is None:
            return jsonify({"error": "Pipeline is still initializing. Please retry."}), 503

        prefer_uploads = bool(chat.files)
        answer = _rag.search_and_summarize(
            message,
            top_k=max(1, min(top_k, 20)),
            chat_id=chat_id,
            prefer_uploads=prefer_uploads,
        )

        db.add(Message(chat_id=chat.id, role="user", content=message))
        db.add(Message(chat_id=chat.id, role="assistant", content=answer))

        if chat.title in {"New chat", "Untitled"}:
            chat.title = message[:60]

        chat.updated_at = utcnow()
        db.commit()

        return jsonify({"answer": answer}), 200
    except Exception as exc:
        db.rollback()
        return jsonify({"error": str(exc)}), 500
    finally:
        db.close()


# Kept for backward compatibility — prefer /api/chats/<id>/messages
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat_legacy():
    payload = request.get_json(silent=True) or {}
    chat_id = payload.get("chat_id")
    if not chat_id:
        return jsonify({"error": "chat_id is required. Create a chat first."}), 400
    return api_chat_message(int(chat_id))


if __name__ == "__main__":
    cli_mode = "--cli" in sys.argv
    if cli_mode:
        _run_cli_mode()
    else:
        if not _is_venv_active():
            print("[WARN] You are not using the project virtual environment.")
            print("[WARN] Run with: D:/RAG_pipeline/.venv/Scripts/python.exe app.py")

        from src.db import init_db

        try:
            init_db()
            print("[INFO] Database tables ready.")
        except Exception as exc:
            print(f"[ERROR] Database init failed: {exc}")
            raise

        _initialize_rag_async()
        print("[INFO] ExamMind starting at http://127.0.0.1:8000")
        print("[INFO] Open your browser — signup/login, then chat & upload papers.")
        app.run(host="127.0.0.1", port=8000, debug=False)
