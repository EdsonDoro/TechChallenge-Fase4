import re
from typing import Any

import pandas as pd

TEXT_COLUMN = "review_comment_message"

_METADATA_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "product_id",
    "customer_id",
]


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa avaliações e preserva metadados úteis para rastreabilidade."""
    data = df.copy()
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Coluna obrigatória ausente: {TEXT_COLUMN}")

    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").map(clean_text)
    data = data[data[TEXT_COLUMN].str.len() > 0].copy()

    if "review_id" in data.columns:
        data["document_id"] = data["review_id"].astype(str)
    else:
        data["document_id"] = [str(i) for i in range(len(data))]

    for column in ("review_creation_date", "review_answer_timestamp"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    return data.reset_index(drop=True)


def _metadata(row: pd.Series) -> dict:
    result = {}
    for column in _METADATA_COLUMNS:
        if column in row.index:
            value = row[column]
            if pd.isna(value):
                value = None
            elif isinstance(value, pd.Timestamp):
                value = value.isoformat()
            else:
                value = str(value)
            result[column] = value
    return result


def build_documents(df: pd.DataFrame) -> list[dict]:
    """Cria documentos rastreáveis. Uma avaliação é uma unidade semântica."""
    if "document_id" not in df.columns:
        raise ValueError("A coluna document_id deve existir antes da criação dos documentos.")

    documents = []
    for _, row in df.iterrows():
        documents.append(
            {
                "document_id": str(row["document_id"]),
                "text": str(row[TEXT_COLUMN]),
                "metadata": _metadata(row),
            }
        )
    return documents
