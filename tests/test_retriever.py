import numpy as np
from src.retrieval.retriever import VectorRetriever

def test_retriever_returns_document():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    docs = [
        {"document_id": "a", "text": "entrega rápida"},
        {"document_id": "b", "text": "produto ruim"},
    ]
    retriever = VectorRetriever(embeddings, docs)
    result = retriever.search(np.array([1.0, 0.0]), top_k=1)
    assert result[0]["document_id"] == "a"
