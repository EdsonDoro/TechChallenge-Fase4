from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / os.getenv("DATA_DIR", "data")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
    min_scope_score: float = float(os.getenv("MIN_SCOPE_SCORE", "0.55"))
    min_domain_score: float = float(os.getenv("MIN_DOMAIN_SCORE", "0.45"))


settings = Settings()
