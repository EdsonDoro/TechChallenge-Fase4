import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.data.loader import load_reviews
from src.data.preprocessing import prepare_reviews, build_documents
from src.embeddings.embedder import EmbeddingModel
from src.retrieval.retriever import VectorRetriever

DATA_FILE = settings.data_dir / "raw" / "olist_order_reviews_dataset.csv"
INDEX_DIR = settings.data_dir / "vectorstore"

def main():
    print(f"Carregando: {DATA_FILE}")
    df = load_reviews(DATA_FILE)
    prepared = prepare_reviews(df)
    documents = build_documents(prepared)
    print(f"Documentos válidos: {len(documents)}")
    embedder = EmbeddingModel(settings.embedding_model)
    embeddings = embedder.encode([doc["text"] for doc in documents])
    retriever = VectorRetriever(embeddings, documents)
    retriever.save(INDEX_DIR)
    print(f"Índice salvo em: {INDEX_DIR}")
    print(f"Vetores indexados: {retriever.size}")

if __name__ == "__main__":
    main()
