"""Executa a avaliação quantitativa do retrieval com um conjunto de referência."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from src.embeddings.embedder import EmbeddingModel
from src.evaluation.evaluator import evaluate_queries
from src.retrieval.retriever import VectorRetriever
from src.rag.pipeline import RAGPipeline

GOLD_FILE = ROOT / "data" / "evaluation" / "retrieval_gold.json"
INDEX_DIR = ROOT / "data" / "vectorstore"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=GOLD_FILE,
        help="JSON de referência; aceita tanto a lista original quanto o formato silver.",
    )
    args = parser.parse_args()

    settings = Settings()
    with args.gold_file.open(encoding="utf-8") as file:
        payload = json.load(file)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    queries = payload.get("queries", []) if isinstance(payload, dict) else payload
    if not queries:
        raise ValueError(f"Nenhuma consulta válida encontrada em {args.gold_file}")

    retriever = VectorRetriever.load(INDEX_DIR)
    embedder = EmbeddingModel(settings.embedding_model)
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

    provisional = sum(
        item.get("annotation_status", metadata.get("annotation_status", "provisional")) != "reviewed"
        for item in queries
    )
    if provisional:
        print(
            f"ATENÇÃO: {provisional}/{len(queries)} perguntas têm anotações "
            "provisórias ou automáticas; interprete as métricas como diagnóstico."
        )

    rows = evaluate_queries(pipeline, queries, k=settings.top_k)
    df = pd.DataFrame(rows)
    summary = {
        "queries": len(df),
        f"precision_at_{settings.top_k}": float(df["precision_at_k"].mean()),
        f"recall_at_{settings.top_k}": float(df["recall_at_k"].mean()),
        "mrr": float(df["mrr"].mean()),
        "annotation_status": metadata.get("annotation_status", "provisional"),
        "gold_file": str(args.gold_file),
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
