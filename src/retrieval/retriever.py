import numpy as np
import faiss

class VectorRetriever:
    """Índice vetorial FAISS com rastreabilidade para os documentos originais."""

    def __init__(self, embeddings, documents):
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("Embeddings inválidos ou vazios.")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = documents

    def search(self, query_embedding, top_k=5, min_score=0.0):
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            item = dict(self.documents[idx])
            item["score"] = float(score)
            results.append(item)
        return results
