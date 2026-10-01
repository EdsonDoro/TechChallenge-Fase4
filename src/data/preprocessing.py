import re
import pandas as pd

TEXT_COLUMN = "review_comment_message"

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def prepare_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara avaliações para a base de conhecimento sem descartar os metadados."""
    data = df.copy()
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Coluna obrigatória ausente: {TEXT_COLUMN}")

    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").map(clean_text)
    data = data[data[TEXT_COLUMN].str.len() > 0].copy()
    data["document_id"] = data.get(
        "review_id",
        pd.Series(range(len(data)), index=data.index, dtype="int64")
    ).astype(str)

    return data.reset_index(drop=True)

def build_documents(df: pd.DataFrame) -> list[dict]:
    """Transforma avaliações em documentos rastreáveis para recuperação."""
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "document_id": str(row["document_id"]),
            "text": row[TEXT_COLUMN],
            "metadata": {
                "review_id": row.get("review_id"),
                "order_id": row.get("order_id"),
                "review_score": row.get("review_score"),
                "review_creation_date": row.get("review_creation_date"),
            }
        })
    return documents
