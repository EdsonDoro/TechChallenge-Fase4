class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0]]

class FakeRetriever:
    def search(self, embedding, top_k, min_score):
        return [{"document_id": "1", "text": "Entrega demorou", "score": 0.9}]

class FakeGenerator:
    def generate(self, question, evidence):
        return "A evidência indica atraso na entrega."

def test_pipeline_returns_answer_and_evidence():
    from src.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline(FakeEmbedder(), FakeRetriever(), FakeGenerator())
    result = pipeline.ask("Houve problemas de entrega?")
    assert result["answer"]
    assert result["evidence"]
