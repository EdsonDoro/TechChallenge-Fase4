"""Anota automaticamente candidatos de retrieval usando o LLM local do Ollama.

Os rótulos gerados são SILVER LABELS (provisórios), não ground truth humano.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
INPUT_FILE = ROOT / "data" / "evaluation" / "retrieval_annotation_candidates.csv"
OUTPUT_FILE = ROOT / "data" / "evaluation" / "retrieval_annotation_auto.csv"
SILVER_FILE = ROOT / "data" / "evaluation" / "retrieval_silver.json"

SYSTEM_PROMPT = """Você é um avaliador de relevância para um sistema de busca semântica
sobre avaliações de clientes do e-commerce brasileiro Olist.
Classifique cada comentário APENAS em relação à pergunta fornecida.
Use somente o texto do comentário; não invente contexto.
' relevant ' significa que o comentário contém informação que ajuda diretamente
a responder à pergunta. 'not_relevant' significa que não ajuda a respondê-la.
'uncertain' deve ser usado quando o texto é ambíguo, insuficiente ou contraditório.
Não use a nota do review nem a similaridade como prova de relevância.
Responda exclusivamente com JSON válido no formato:
{"items":[{"document_id":"...","label":"relevant|not_relevant|uncertain",
"confidence":0.0,"reason":"justificativa curta"}]}.
A confiança deve refletir a certeza semântica da classificação, entre 0 e 1.
"""

def parse_model_response(content: str, expected_ids: set[str]) -> dict[str, dict]:
    """Extrai e valida os rótulos retornados pelo modelo."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("O modelo não retornou um objeto JSON válido.")
        payload = json.loads(match.group(0))

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Resposta JSON sem lista 'items'.")

    parsed = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        document_id = str(item.get("document_id", "")).strip()
        label = str(item.get("label", "")).strip().lower()
        if document_id not in expected_ids:
            continue
        if label not in {"relevant", "not_relevant", "uncertain"}:
            label = "uncertain"
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        parsed[document_id] = {
            "auto_label": label,
            "auto_confidence": confidence,
            "auto_reason": str(item.get("reason", "")).strip(),
        }

    missing = expected_ids - set(parsed)
    for document_id in missing:
        parsed[document_id] = {
            "auto_label": "uncertain",
            "auto_confidence": 0.0,
            "auto_reason": "O modelo não retornou uma classificação válida para este documento.",
        }
    return parsed


def classify_group(client, model: str, question: str, rows: list[dict]) -> dict[str, dict]:
    candidates = [
        {"document_id": row["document_id"], "comment": row.get("comment", "")}
        for row in rows
    ]
    user_prompt = (
        f"PERGUNTA:\n{question}\n\n"
        "CLASSIFIQUE INDIVIDUALMENTE OS COMENTÁRIOS ABAIXO. "
        "Retorne exatamente um item por document_id.\n"
        + json.dumps(candidates, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_model_response(content, {row["document_id"] for row in rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--silver-output", type=Path, default=SILVER_FILE)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    args = parser.parse_args()

    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence deve estar entre 0 e 1.")
    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de candidatos não encontrado: {args.input}")

    # Importação local: permite testar as funções sem depender do cliente OpenAI.
    from openai import OpenAI

    settings = Settings()
    client = OpenAI(
        base_url=settings.ollama_base_url,
        api_key="ollama",
        timeout=180.0,
    )

    with args.input.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("O CSV de candidatos está vazio.")

    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["question"], []).append(row)

    annotations = {}
    for index, (question, group_rows) in enumerate(grouped.items(), start=1):
        print(f"[{index}/{len(grouped)}] Classificando {len(group_rows)} comentários: {question}")
        try:
            annotations.update({(question, doc_id): value for doc_id, value in classify_group(client, settings.llm_model, question, group_rows).items()})
        except Exception as exc:
            # Não interrompe os demais grupos; os itens deste grupo ficam explicitamente incertos.
            print(f"  ERRO: {exc}")
            for row in group_rows:
                annotations[(question, row["document_id"])] = {
                    "auto_label": "uncertain",
                    "auto_confidence": 0.0,
                    "auto_reason": f"Falha na chamada ao modelo: {type(exc).__name__}",
                }

    annotated_rows = []
    silver_queries = []
    for question, group_rows in grouped.items():
        relevant_ids = []
        for row in group_rows:
            result = annotations.get((question, row["document_id"]), {
                "auto_label": "uncertain",
                "auto_confidence": 0.0,
                "auto_reason": "Sem classificação.",
            })
            enriched = {**row, **result}
            enriched["annotation_method"] = "ollama_llm_auto"
            enriched["annotation_status"] = "auto_provisional"
            enriched["accepted_for_silver"] = (
                result["auto_label"] == "relevant"
                and result["auto_confidence"] >= args.min_confidence
            )
            annotated_rows.append(enriched)
            if enriched["accepted_for_silver"]:
                relevant_ids.append(row["document_id"])

        if relevant_ids:
            silver_queries.append({
                "question": question,
                "relevant_ids": relevant_ids,
                "annotation_status": "auto_provisional",
                "annotation_method": "ollama_llm_auto",
                "min_confidence": args.min_confidence,
                "annotation_note": (
                    "Rótulos automáticos provisórios; documentos relevantes "
                    "selecionados por LLM local e limiar de confiança."
                ),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(annotated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(annotated_rows)

    silver_payload = {
        "metadata": {
            "annotation_status": "auto_provisional",
            "annotation_method": "ollama_llm_auto",
            "model": settings.llm_model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_confidence": args.min_confidence,
            "source_file": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
            "warning": (
                "Conjunto silver gerado automaticamente. Não equivale a ground truth "
                "revisado por humanos e não deve ser apresentado como validação definitiva."
            ),
        },
        "queries": silver_queries,
    }
    args.silver_output.parent.mkdir(parents=True, exist_ok=True)
    args.silver_output.write_text(
        json.dumps(silver_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = {}
    for item in annotated_rows:
        counts[item["auto_label"]] = counts.get(item["auto_label"], 0) + 1
    accepted = sum(bool(item["accepted_for_silver"]) for item in annotated_rows)
    print("\n=== ANOTAÇÃO AUTOMÁTICA CONCLUÍDA ===")
    print(f"Comentários analisados: {len(annotated_rows)}")
    print(f"Classificações: {counts}")
    print(f"Relevantes aceitos no conjunto silver: {accepted}")
    print(f"Perguntas com pelo menos um relevante: {len(silver_queries)}")
    print(f"CSV anotado: {args.output}")
    print(f"Conjunto silver: {args.silver_output}")
    print("ATENÇÃO: os rótulos são provisórios e gerados por IA.")


if __name__ == "__main__":
    main()
