import json
from pathlib import Path

import faiss
import numpy as np


class VectorRetriever:
    """Busca semântica FAISS com persistência e rastreabilidade."""

    def __init__(self, embeddings=None, documents=None, index=None):
        if index is not None:
            self.index = index
            self.documents = documents or []
            return

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("Embeddings inválidos ou vazios.")
        if len(documents or []) != len(vectors):
            raise ValueError("Quantidade de documentos e embeddings deve ser igual.")

        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = documents

    def search(self, query_embedding, top_k=5, min_score=0.25):
        if not self.documents:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(max(int(top_k), 1), len(self.documents))
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < float(min_score):
                continue
            item = dict(self.documents[int(idx)])
            item["metadata"] = dict(item.get("metadata", {}))
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "documents.json").write_text(
            json.dumps(self.documents, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        index_file = directory / "index.faiss"
        documents_file = directory / "documents.json"
        if not index_file.exists() or not documents_file.exists():
            raise FileNotFoundError(
                f"Índice incompleto em {directory}. Execute scripts/build_index.py."
            )
        index = faiss.read_index(str(index_file))
        documents = json.loads(documents_file.read_text(encoding="utf-8"))
        return cls(documents=documents, index=index)

    @property
    def size(self) -> int:
        return self.index.ntotal
