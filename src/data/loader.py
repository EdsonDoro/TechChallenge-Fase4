from pathlib import Path
import pandas as pd


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Carrega a tabela de avaliações do Olist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, low_memory=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    return load_reviews(path)
