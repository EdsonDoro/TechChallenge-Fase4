import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.embeddings.embedder import EmbeddingModel
from src.generation.llm import LLMGenerator
from src.rag.pipeline import RAGPipeline
from src.retrieval.retriever import VectorRetriever


def main():
    parser = argparse.ArgumentParser(description="Consulta o RAG do Olist.")
    parser.add_argument("question", help="Pergunta em linguagem natural.")
    args = parser.parse_args()

    retriever = VectorRetriever.load(settings.data_dir / "vectorstore")
    embedder = EmbeddingModel(settings.embedding_model)
    generator = LLMGenerator(
        model=settings.llm_model,
        provider=settings.llm_provider,
        base_url=settings.ollama_base_url if settings.llm_provider == "ollama" else None,
    )
    pipeline = RAGPipeline(
        embedder, retriever, generator, settings.top_k, settings.min_relevance_score
    )
    result = pipeline.ask(args.question)

    print("\nRESPOSTA\n")
    print(result["answer"])
    print("\nEVIDÊNCIAS\n")
    for item in result["evidence"]:
        print(f"[{item['document_id']}] score={item['score']:.4f} | {item['text']}")


if __name__ == "__main__":
    main()
