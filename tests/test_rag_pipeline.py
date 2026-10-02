from src.rag.pipeline import RAGPipeline

class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] if "Olist" in text or "entrega" in text.lower() else [0.0, 1.0] for text in texts]

class FakeRetriever:
    def search(self, embedding, top_k, min_score):
        return [{"document_id": "1", "text": "Entrega demorou", "score": 0.9}]

class FakeGenerator:
    def generate(self, question, evidence):
        return "A evidência indica atraso na entrega [1]."

def test_pipeline_returns_answer_and_evidence():
    result = RAGPipeline(
        FakeEmbedder(), FakeRetriever(), FakeGenerator(),
        domain_anchors=("Quais problemas os clientes relatam sobre entrega na Olist?",),
        min_domain_score=0.45,
    ).ask("Houve problemas de entrega?")
    assert result["answer"]
    assert result["evidence"]
    assert result["has_evidence"] is True
    assert result["evidence_status"] == "supported"
    assert result["domain_score"] >= 0.45

class OutOfScopeRetriever:
    def search(self, embedding, top_k, min_score):
        return [{"document_id": "2", "text": "Entrega rápida", "score": 0.90}]

def test_pipeline_rejects_out_of_scope_even_with_high_retrieval_score():
    result = RAGPipeline(
        FakeEmbedder(), OutOfScopeRetriever(), FakeGenerator(),
        min_scope_score=0.55, min_domain_score=0.90,
        domain_anchors=("Quais problemas os clientes relatam sobre entrega na Olist?",),
    ).ask("Qual é a previsão do tempo para amanhã?")
    assert result["has_evidence"] is False
    assert result["evidence_status"] == "out_of_scope"
    assert "fora do escopo" in result["answer"]
    assert result["retrieved_candidates"]
    assert result["evidence"] == []
