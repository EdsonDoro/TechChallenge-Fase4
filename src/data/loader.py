from pathlib import Path
import pandas as pd


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Carrega a tabela de avaliações do Olist."""
    return load_csv(path)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, low_memory=False)


def load_olist_raw(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Carrega as tabelas Olist necessárias ao enriquecimento do corpus."""
    root = Path(data_dir)
    names = {
        "reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "categories": "product_category_name_translation.csv",
    }
    return {key: load_csv(root / filename) for key, filename in names.items()}
