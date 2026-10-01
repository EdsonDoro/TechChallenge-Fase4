from src.rag.pipeline import RAGPipeline

class FakeEmbedder:
    def encode(self, texts): return [[1.0, 0.0]]

class EmptyRetriever:
    def search(self, embedding, top_k, min_score): return []

class FakeGenerator:
    def generate(self, question, evidence):
        return "Não há evidências suficientes na base de conhecimento para responder a esta pergunta."

def test_pipeline_handles_no_evidence():
    result = RAGPipeline(FakeEmbedder(), EmptyRetriever(), FakeGenerator()).ask("Pergunta fora da base")
    assert result["has_evidence"] is False
    assert "Não há evidências suficientes" in result["answer"]
