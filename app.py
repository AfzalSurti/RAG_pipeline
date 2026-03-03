import os
import sys

os.environ.setdefault("PYTHONNOUSERSITE", "1")

from src.search import RAGSearch

if __name__=="__main__":
    if ".venv" not in sys.executable.replace("\\", "/"):
        print("[WARN] You are not using the project virtual environment.")
        print("[WARN] Run with: D:/RAG_pipeline/.venv/Scripts/python.exe app.py")

    data_dir = "CopyOfExam"
    persist_dir = "faiss_store_exam"
    should_rebuild = not (
        os.path.exists(os.path.join(persist_dir, "faiss.index"))
        and os.path.exists(os.path.join(persist_dir, "metadata.pkl"))
    )

    if should_rebuild:
        print("[INFO] Vector index not fully available. Building index (first run can take time)...")
    else:
        print("[INFO] Using existing vector index for fast startup.")

    try:
        rag_search = RAGSearch(
            persist_dir=persist_dir,
            data_dir=data_dir,
            rebuild_index=should_rebuild,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("[HINT] For scanned PDFs, install OCR runtime (Tesseract) and ensure it is added to PATH.")
        raise

    print(f"[INFO] RAG ready for exam files in: {data_dir}")
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

    

    