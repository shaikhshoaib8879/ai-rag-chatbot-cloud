import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

COLLECTION_NAME = "rag_documents"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    )


def _get_chroma() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=settings.chroma_db_path,
    )


def add_documents(docs: list[Document]) -> int:
    """Add document chunks to the vector store. Returns count added."""
    db = _get_chroma()
    db.add_documents(docs)
    return len(docs)


def similarity_search(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Search for similar chunks. Returns (document, score) pairs.

    Scores are normalized to 0-1 (higher = more similar).
    """
    db = _get_chroma()
    results = db.similarity_search_with_score(query, k=k or settings.top_k)
    return [(doc, max(0.0, 1.0 - distance / 2.0)) for doc, distance in results]


def get_collection_stats() -> dict:
    """Return stats about the current vector store."""
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        return {"total_chunks": count}
    except Exception:
        return {"total_chunks": 0}


def clear_collection() -> None:
    """Delete all documents from the vector store."""
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
