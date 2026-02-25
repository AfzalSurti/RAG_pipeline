from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline


if __name__=="__main__":
    docs=load_all_documents("data")
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunks_vectors=EmbeddingPipeline().embed_chunks(chunks)
    print(chunks_vectors)

    