from src.search import RAGSearch

if __name__=="__main__":
    data_dir = "data"
    persist_dir = "faiss_store"

    try:
        rag_search = RAGSearch(
            persist_dir=persist_dir,
            data_dir=data_dir,
            rebuild_index=False,
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

    

    