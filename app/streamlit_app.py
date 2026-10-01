import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.settings import settings
from src.embeddings.embedder import EmbeddingModel
from src.generation.llm import LLMGenerator
from src.rag.pipeline import RAGPipeline
from src.retrieval.retriever import VectorRetriever


st.set_page_config(page_title="Voice of Customer RAG", page_icon="🔎", layout="wide")
st.title("Voice of Customer Intelligence — RAG")
st.caption("Respostas fundamentadas em avaliações de clientes do dataset Olist.")


@st.cache_resource
def load_pipeline():
    index_dir = settings.data_dir / "vectorstore"
    retriever = VectorRetriever.load(index_dir)
    embedder = EmbeddingModel(settings.embedding_model)
    generator = LLMGenerator(settings.llm_model)
    return RAGPipeline(
        embedder,
        retriever,
        generator,
        top_k=settings.top_k,
        min_relevance_score=settings.min_relevance_score,
    )


if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY não configurada. A recuperação pode ser testada, mas a geração exige uma chave da OpenAI.")

question = st.text_area(
    "Pergunta",
    placeholder="Ex.: Quais são os principais problemas relatados sobre a entrega?",
    height=100,
)

if st.button("Consultar", type="primary", disabled=not question.strip()):
    try:
        pipeline = load_pipeline()
        result = pipeline.ask(question)

        st.subheader("Resposta")
        st.write(result["answer"])

        st.subheader("Evidências recuperadas")
        if not result["evidence"]:
            st.info("Nenhuma evidência ultrapassou o limiar de relevância.")
        else:
            for item in result["evidence"]:
                metadata = item.get("metadata", {})
                with st.expander(
                    f"[{item['document_id']}] similaridade={item['score']:.3f}"
                ):
                    st.write(item["text"])
                    st.json(metadata)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Erro ao executar o RAG: {exc}")
