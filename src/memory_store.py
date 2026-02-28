import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer


class ConversationMemoryStore:
    """Persistent vector store for conversational memory (Q/A turns)."""

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_filename: str = "memory.index",
        metadata_filename: str = "memory_metadata.pkl",
    ):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(self.persist_dir, index_filename)
        self.metadata_path = os.path.join(self.persist_dir, metadata_filename)
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

        os.makedirs(self.persist_dir, exist_ok=True)
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as memory_file:
                self.metadata = pickle.load(memory_file)
            print(f"[INFO] Loaded conversation memory from {self.persist_dir}")
        else:
            self.index = None
            self.metadata = []

    def _save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as memory_file:
            pickle.dump(self.metadata, memory_file)

    def add_interaction(self, question: str, answer: str) -> None:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return

        memory_text = f"Question: {question}\nAnswer: {answer}"
        vector = self.model.encode([memory_text]).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatL2(vector.shape[1])

        self.index.add(vector)
        self.metadata.append(
            {
                "text": memory_text,
                "question": question,
                "answer": answer,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save()
        print("[INFO] Added interaction to persistent conversation memory.")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None or not self.metadata:
            return []

        vector = self.model.encode([query_text]).astype("float32")
        safe_top_k = max(1, min(top_k, len(self.metadata)))
        distances, indexes = self.index.search(vector, safe_top_k)
        results: List[Dict[str, Any]] = []

        for idx, dist in zip(indexes[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata": self.metadata[idx],
                }
            )

        return results