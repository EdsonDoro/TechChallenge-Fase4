"""Executa a avaliação quantitativa do retrieval com o conjunto anotado."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.settings import Settings
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.evaluation.evaluator import evaluate_queries
from src.retrieval.retriever import VectorRetriever
from src.rag.pipeline import RAGPipeline


ROOT = Path(__file__).resolve().parents[1]
GOLD_FILE = ROOT / "data" / "evaluation" / "retrieval_gold.json"
INDEX_DIR = ROOT / "data" / "vectorstore"


def main():
    settings = Settings()
    with GOLD_FILE.open(encoding="utf-8") as f:
        queries = json.load(f)

    retriever = VectorRetriever.load(INDEX_DIR)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    # O gerador não é necessário para avaliar a recuperação.
    pipeline = RAGPipeline(
        embedder,
        retriever,
        generator=None,
        top_k=settings.top_k,
        min_relevance_score=settings.min_relevance_score,
        min_scope_score=settings.min_scope_score,
        min_domain_score=settings.min_domain_score,
    )

    rows = evaluate_queries(pipeline, queries, k=settings.top_k)
    df = pd.DataFrame(rows)

    summary = {
        "queries": len(df),
        f"precision_at_{settings.top_k}": float(df[f"precision_at_k"].mean()),
        f"recall_at_{settings.top_k}": float(df[f"recall_at_k"].mean()),
        "mrr": float(df["mrr"].mean()),
    }

    print("=== AVALIAÇÃO QUANTITATIVA DO RETRIEVAL ===")
    print(df[["question", "precision_at_k", "recall_at_k", "mrr"]].to_string(index=False))
    print("\nMédias:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    output = ROOT / "data" / "evaluation" / "retrieval_results.json"
    output.write_text(
        json.dumps({"summary": summary, "details": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultados salvos em: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
