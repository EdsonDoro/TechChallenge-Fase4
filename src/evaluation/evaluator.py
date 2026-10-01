from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float


def evaluate_retrieval(results, relevant_ids: set[str], k: int = 5) -> RetrievalMetrics:
    retrieved = [str(item["document_id"]) for item in results[:k]]
    relevant_ids = {str(value) for value in relevant_ids}

    if not retrieved:
        return RetrievalMetrics(0.0, 0.0, 0.0)

    hits = [doc_id in relevant_ids for doc_id in retrieved]
    precision = sum(hits) / len(retrieved)
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0

    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    return RetrievalMetrics(precision, recall, mrr)


def evaluate_queries(pipeline, queries: list[dict], k: int = 5) -> list[dict]:
    """Avalia um conjunto anotado: {'question': ..., 'relevant_ids': [...]}."""
    rows = []
    for item in queries:
        results = pipeline.retrieve(item["question"])
        metrics = evaluate_retrieval(results, set(item["relevant_ids"]), k=k)
        rows.append(
            {
                "question": item["question"],
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "mrr": metrics.mrr,
                "retrieved_ids": [r["document_id"] for r in results[:k]],
            }
        )
    return rows
