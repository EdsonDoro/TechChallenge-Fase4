import re
from typing import Any
import pandas as pd

TEXT_COLUMN = "review_comment_message"

_METADATA_COLUMNS = [
    "review_id", "order_id", "review_score", "review_creation_date",
    "review_answer_timestamp", "product_id", "customer_id", "customer_unique_id",
    "customer_city", "customer_state", "order_status", "order_purchase_timestamp",
    "order_delivered_customer_date", "order_estimated_delivery_date",
    "delivery_days", "delivery_delay_days", "product_category_name",
    "product_category_name_english", "seller_id", "seller_city", "seller_state",
    "order_value", "payment_type",
]


def clean_text(value: Any) -> str:
    if pd.isna(value): return ""
    return re.sub(r"\s+", " ", str(value).strip())


def prepare_reviews(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f"Coluna obrigatória ausente: {TEXT_COLUMN}")
    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").map(clean_text)
    data = data[data[TEXT_COLUMN].str.len() > 0].copy()
    data["document_id"] = data["review_id"].astype(str) if "review_id" in data.columns else [str(i) for i in range(len(data))]
    for column in ("review_creation_date", "review_answer_timestamp", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"):
        if column in data.columns: data[column] = pd.to_datetime(data[column], errors="coerce")
    return data.reset_index(drop=True)


def _metadata(row: pd.Series) -> dict:
    result = {}
    for column in _METADATA_COLUMNS:
        if column not in row.index: continue
        value = row[column]
        if pd.isna(value): value = None
        elif isinstance(value, pd.Timestamp): value = value.isoformat()
        elif isinstance(value, (float, int)) and pd.isna(value): value = None
        else: value = str(value)
        result[column] = value
    return result


def build_documents(df: pd.DataFrame) -> list[dict]:
    if "document_id" not in df.columns: raise ValueError("A coluna document_id deve existir antes da criação dos documentos.")
    return [{"document_id": str(row["document_id"]), "text": str(row[TEXT_COLUMN]), "metadata": _metadata(row)} for _, row in df.iterrows()]


def enrich_reviews(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une avaliações às dimensões do Olist sem multiplicar documentos por item do pedido."""
    reviews = raw["reviews"].copy()
    orders = raw["orders"].copy()
    customers = raw["customers"].copy()
    items = raw["items"].copy()
    products = raw["products"].copy()
    sellers = raw["sellers"].copy()
    payments = raw["payments"].copy()
    categories = raw["categories"].copy()

    item_order = items.groupby("order_id", as_index=False).agg(
        product_id=("product_id", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
        seller_id=("seller_id", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    )
    product_map = products[["product_id", "product_category_name"]].merge(categories, on="product_category_name", how="left")
    item_categories = items[["order_id", "product_id"]].merge(product_map, on="product_id", how="left")
    category_order = item_categories.groupby("order_id", as_index=False).agg(
        product_category_name=("product_category_name", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
        product_category_name_english=("product_category_name_english", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
    )
    seller_order = items[["order_id", "seller_id"]].merge(sellers[["seller_id", "seller_city", "seller_state"]], on="seller_id", how="left")
    seller_summary = seller_order.groupby("order_id", as_index=False).agg(
        seller_city=("seller_city", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
        seller_state=("seller_state", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
    )
    payment_summary = payments.groupby("order_id", as_index=False).agg(
        payment_type=("payment_type", lambda s: "|".join(pd.Series(s.dropna().astype(str).unique()).tolist())),
    )

    data = reviews.merge(orders, on="order_id", how="left", suffixes=("", "_order"))
    data = data.merge(customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]], on="customer_id", how="left")
    data = data.merge(item_order, on="order_id", how="left")
    data = data.merge(category_order, on="order_id", how="left")
    data = data.merge(seller_summary, on="order_id", how="left")
    data = data.merge(payment_summary, on="order_id", how="left")

    for col in ("order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"):
        data[col] = pd.to_datetime(data[col], errors="coerce")
    data["delivery_days"] = (data["order_delivered_customer_date"] - data["order_purchase_timestamp"]).dt.total_seconds() / 86400
    data["delivery_delay_days"] = (data["order_delivered_customer_date"] - data["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    return prepare_reviews(data)
