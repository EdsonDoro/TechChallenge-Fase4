import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config.settings import settings
from src.data.loader import load_olist_raw
from src.data.preprocessing import enrich_reviews, build_documents
from src.embeddings.embedder import EmbeddingModel
from src.retrieval.retriever import VectorRetriever

RAW_DIR = settings.data_dir / "raw"
INDEX_DIR = settings.data_dir / "vectorstore"

def main():
    raw = load_olist_raw(RAW_DIR)
    prepared = enrich_reviews(raw)
    documents = build_documents(prepared)
    print(f"Avaliações com comentário: {len(documents)}")
    embedder = EmbeddingModel(settings.embedding_model)
    embeddings = embedder.encode([doc["text"] for doc in documents])
    retriever = VectorRetriever(embeddings, documents)
    retriever.save(INDEX_DIR)
    print(f"Índice salvo em: {INDEX_DIR} | vetores: {retriever.size}")

if __name__ == "__main__": main()
