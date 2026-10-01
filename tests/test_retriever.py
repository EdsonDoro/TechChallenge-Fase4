import numpy as np
from src.retrieval.retriever import VectorRetriever

def test_retriever_returns_document():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    docs = [{"document_id": "a", "text": "entrega rápida"}, {"document_id": "b", "text": "produto ruim"}]
    retriever = VectorRetriever(embeddings, docs)
    result = retriever.search(np.array([1.0, 0.0]), top_k=1, min_score=0.0)
    assert result[0]["document_id"] == "a"

def test_retriever_applies_threshold():
    retriever = VectorRetriever(np.array([[1.0, 0.0]], dtype="float32"), [{"document_id": "a", "text": "entrega"}])
    assert retriever.search(np.array([0.0, 1.0]), top_k=1, min_score=0.5) == []
