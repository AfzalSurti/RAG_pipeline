from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorestore import FaissVectorStore

if __name__=="__main__":
    docs=load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    print("Vector store built successfully.")

    