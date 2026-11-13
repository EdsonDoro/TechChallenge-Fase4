"""Métricas reprodutíveis para avaliação da recuperação semântica."""
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float


def evaluate_retrieval(results, relevant_ids: set[str], k: int = 5) -> RetrievalMetrics:
    """Calcula Precision@K, Recall@K e MRR sobre o ranking bruto.

    Precision@K usa K como denominador mesmo quando o retriever retorna
    menos de K itens. Isso mantém a definição padrão da métrica.
    """
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")

    retrieved = [str(item["document_id"]) for item in results[:k]]
    relevant_ids = {str(value) for value in relevant_ids}

    if not relevant_ids:
        raise ValueError("relevant_ids não pode ser vazio para avaliar recall.")

    hits = [doc_id in relevant_ids for doc_id in retrieved]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant_ids)

    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit),
        0.0,
    )
    return RetrievalMetrics(precision, recall, reciprocal_rank)


def evaluate_queries(pipeline, queries: list[dict], k: int = 5) -> list[dict]:
    """Avalia consultas anotadas sem aplicar guardrails do pipeline.

    Cada item deve conter 'question' e uma lista não vazia de 'relevant_ids'.
    """
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")

    rows = []
    for item in queries:
        question = item["question"]
        relevant_ids = item.get("relevant_ids", [])
        if not relevant_ids:
            raise ValueError(f"A consulta não possui relevant_ids: {question}")

        results = pipeline.retrieve_candidates(question)
        metrics = evaluate_retrieval(results, set(relevant_ids), k=k)
        rows.append(
            {
                "question": question,
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "mrr": metrics.mrr,
                "retrieved_ids": [r["document_id"] for r in results[:k]],
            }
        )
    return rows
