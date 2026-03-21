import os
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

os.environ.setdefault("PYTHONNOUSERSITE", "1")


DATA_DIR = "CopyOfExam"
PERSIST_DIR = "faiss_store_exam"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

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
    thread = threading.Thread(target=_initialize_rag, daemon=True)
    thread.start()


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
    if should_rebuild:
        print("[INFO] Vector index not fully available. Building index (first run can take time)...")
    else:
        print("[INFO] Using existing vector index for fast startup.")

    try:
        rag_search = RAGSearch(
            persist_dir=PERSIST_DIR,
            data_dir=DATA_DIR,
            rebuild_index=should_rebuild,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print("[HINT] For scanned PDFs, install OCR runtime (Tesseract) and ensure it is added to PATH.")
        raise

    print(f"[INFO] RAG ready for exam files in: {DATA_DIR}")
    print("[INFO] Example query: give me 3 question from theory of computation")

    while True:
        user_query = input("\nAsk your exam query (or type 'exit'): ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("[INFO] Exiting.")
            break
        if not user_query:
            continue

        answer = rag_search.search_and_summarize(user_query, top_k=8)
        print("\nAnswer:", answer)


app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    if _rag_error:
        return jsonify({"ready": False, "loading": False, "error": _rag_error}), 500
    return jsonify({"ready": _rag_ready, "loading": _rag_loading, "error": None}), 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    global _rag_error
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    top_k = int(payload.get("top_k") or 8)

    if not message:
        return jsonify({"error": "Message is required."}), 400

    if _rag_error:
        return jsonify({"error": f"Startup error: {_rag_error}"}), 500

    _ensure_rag_ready()
    if _rag_error:
        return jsonify({"error": f"Startup error: {_rag_error}"}), 500
    if not _rag_ready or _rag is None:
        return jsonify({"error": "Pipeline is still initializing. Please retry."}), 503

    try:
        answer = _rag.search_and_summarize(message, top_k=max(1, min(top_k, 20)))
        return jsonify({"answer": answer}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    cli_mode = "--cli" in sys.argv
    if cli_mode:
        _run_cli_mode()
    else:
        if not _is_venv_active():
            print("[WARN] You are not using the project virtual environment.")
            print("[WARN] Run with: D:/RAG_pipeline/.venv/Scripts/python.exe app.py")

        _initialize_rag_async()
        print("[INFO] Web chat starting at http://127.0.0.1:8000")
        print("[INFO] Open your browser and start chatting.")
        app.run(host="127.0.0.1", port=8000, debug=False)



