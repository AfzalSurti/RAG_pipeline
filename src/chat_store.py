import os
from pathlib import Path
from typing import Any, Dict, List

from src.vectorestore import FaissVectorStore


class ChatDocumentStore:
    """Per-chat FAISS store for uploaded exam papers / chapter PDFs."""

    def __init__(
        self,
        base_dir: str = "faiss_chat_stores",
        embedding_model: str = "all-MiniLM-L6-v2",
        model: Any = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.model = model
        self._stores: Dict[int, FaissVectorStore] = {}

    def _chat_dir(self, chat_id: int) -> Path:
        path = self.base_dir / str(chat_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_store(self, chat_id: int) -> FaissVectorStore:
        if chat_id in self._stores:
            return self._stores[chat_id]

        persist_dir = str(self._chat_dir(chat_id))
        store = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=self.embedding_model,
            model=self.model,
        )
        index_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            store.load()
        self._stores[chat_id] = store
        return store

    def ingest_documents(self, chat_id: int, documents: List[Any]) -> int:
        if not documents:
            return 0
        store = self.get_store(chat_id)
        return store.add_documents(documents)

    def query(self, chat_id: int, query_text: str, top_k: int = 6) -> List[Dict[str, Any]]:
        persist_dir = self._chat_dir(chat_id)
        if not (persist_dir / "faiss.index").exists():
            return []
        store = self.get_store(chat_id)
        if store.index is None or not store.metadata:
            return []
        return store.query(query_text, top_k=top_k)

    def has_index(self, chat_id: int) -> bool:
        return (self._chat_dir(chat_id) / "faiss.index").exists()
