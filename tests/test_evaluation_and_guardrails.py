from src.evaluation.evaluator import evaluate_retrieval
from src.rag.pipeline import RAGPipeline


def test_precision_at_k_uses_k_even_when_fewer_results_are_returned():
    results = [{"document_id": "a"}]
    metrics = evaluate_retrieval(results, {"a"}, k=5)
    assert metrics.precision_at_k == 0.2
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0


def test_evaluator_rejects_invalid_k_and_empty_gold():
    import pytest

    with pytest.raises(ValueError):
        evaluate_retrieval([], {"a"}, k=0)
    with pytest.raises(ValueError):
        evaluate_retrieval([], set(), k=5)


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] if "entrega" in text.lower() else [0.0, 1.0] for text in texts]


class LowScoreRetriever:
    def search(self, embedding, top_k, min_score):
        return [{"document_id": "low", "text": "Comentário genérico", "score": 0.30}]


class FakeGenerator:
    def generate(self, question, evidence):
        raise AssertionError("O gerador não deve ser chamado sem evidência suficiente.")


def test_in_domain_question_with_weak_retrieval_is_insufficient_evidence():
    pipeline = RAGPipeline(
        FakeEmbedder(),
        LowScoreRetriever(),
        FakeGenerator(),
        min_scope_score=0.55,
        min_domain_score=0.45,
        domain_anchors=("Problemas de entrega relatados pelos clientes da Olist",),
    )
    result = pipeline.ask("Quais problemas de entrega foram relatados?")
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["evidence"] == []
    assert "Não há evidências suficientes" in result["answer"]
