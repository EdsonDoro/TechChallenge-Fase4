from src.rag.pipeline import RAGPipeline


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0]]


class FakeRetriever:
    def search(self, embedding, top_k, min_score):
        return [{"document_id": "1", "text": "Entrega demorou", "score": 0.9}]


class FakeGenerator:
    def generate(self, question, evidence):
        return "A evidência indica atraso na entrega [1]."


def test_pipeline_returns_answer_and_evidence():
    result = RAGPipeline(FakeEmbedder(), FakeRetriever(), FakeGenerator()).ask(
        "Houve problemas de entrega?"
    )
    assert result["answer"]
    assert result["evidence"]
    assert result["has_evidence"] is True
    assert result["evidence_status"] == "supported"


class OutOfScopeRetriever:
    def search(self, embedding, top_k, min_score):
        return [{"document_id": "2", "text": "Entrega rápida", "score": 0.35}]


def test_pipeline_rejects_low_similarity_without_domain_support():
    result = RAGPipeline(
        FakeEmbedder(), OutOfScopeRetriever(), FakeGenerator(), min_scope_score=0.55
    ).ask("Qual é a previsão do tempo para amanhã?")
    assert result["has_evidence"] is False
    assert result["evidence_status"] == "out_of_scope"
    assert "fora do escopo" in result["answer"]
    assert result["retrieved_candidates"]
