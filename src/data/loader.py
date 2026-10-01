from pathlib import Path
import pandas as pd

def load_reviews(path: str | Path) -> pd.DataFrame:
    """Carrega as avaliações da base Olist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)
