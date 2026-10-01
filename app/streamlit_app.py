import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Voice of Customer RAG", layout="wide")
st.title("Voice of Customer Intelligence — RAG")

st.markdown(
    "Faça perguntas sobre as avaliações de clientes e visualize as evidências "
    "recuperadas que sustentam a resposta."
)

question = st.text_area(
    "Pergunta",
    placeholder="Ex.: Quais são os principais problemas relacionados à entrega?"
)

if st.button("Consultar") and question.strip():
    st.info(
        "A interface está preparada para conectar o pipeline em "
        "`src/rag/pipeline.py`. Configure a base, o índice vetorial e o provedor "
        "de LLM antes da execução completa."
    )
