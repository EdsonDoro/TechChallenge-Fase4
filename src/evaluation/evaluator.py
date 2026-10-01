from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None

def evaluate_retrieval(results, relevant_ids: set[str], k: int = 5) -> RetrievalMetrics:
    """Calcula métricas simples quando há conjunto de relevância anotado."""
    retrieved = [str(item["document_id"]) for item in results[:k]]
    if not retrieved:
        return RetrievalMetrics(precision_at_k=0.0, recall_at_k=0.0, mrr=0.0)

    hits = [doc_id in relevant_ids for doc_id in retrieved]
    precision = sum(hits) / len(retrieved)
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0

    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1 / rank
            break

    return RetrievalMetrics(precision_at_k=precision, recall_at_k=recall, mrr=mrr)
