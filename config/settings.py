from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / os.getenv("DATA_DIR", "data")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

settings = Settings()
