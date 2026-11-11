"""Orquestração do fluxo RAG e validação de suficiência das evidências."""
from __future__ import annotations
import numpy as np

DEFAULT_DOMAIN_ANCHORS = (
    "Quais problemas os clientes relatam nas avaliações da Olist?",
    "Quais reclamações aparecem nas avaliações sobre pedidos e entregas?",
    "Como os clientes avaliam produtos, pedidos e a experiência de compra?",
    "Quais aspectos da experiência de compra geram satisfação ou insatisfação?",
    "Quais problemas de entrega são mencionados pelos clientes?",
    "O que os clientes comentam sobre produtos, vendedores e atendimento?",
)

class RAGPipeline:
    """Pipeline: pergunta -> retrieval -> validação -> contexto -> geração."""

    OUT_OF_SCOPE_MESSAGE = (
        "A pergunta parece estar fora do escopo da base de conhecimento Olist "
        "ou não possui evidências suficientes relacionadas ao tema. "
        "Não é possível responder com segurança usando as avaliações disponíveis."
    )
    INSUFFICIENT_EVIDENCE_MESSAGE = (
        "Não há evidências suficientes na base de conhecimento para responder a esta pergunta."
    )

    def __init__(
        self, embedder, retriever, generator, top_k=5,
        min_relevance_score=0.25, min_scope_score=0.55,
        min_domain_score=0.45, domain_anchors=None,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.min_scope_score = min_scope_score
        self.min_domain_score = min_domain_score
        self.domain_anchors = tuple(domain_anchors or DEFAULT_DOMAIN_ANCHORS)
        self._anchor_embeddings = self._encode(self.domain_anchors)

    def _encode(self, texts):
        vectors = np.asarray(self.embedder.encode(list(texts)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def retrieve_candidates(self, question):
        """Retorna o retrieval bruto após apenas o limiar mínimo configurado."""
        if not question or not question.strip():
            return []
        query_embedding = self.embedder.encode([question.strip()])[0]
        return self.retriever.search(
            query_embedding, top_k=self.top_k, min_score=self.min_relevance_score
        )

    def domain_score(self, question):
        if not question or not question.strip():
            return 0.0
        query = self._encode([question.strip()])[0]
        return float(np.max(self._anchor_embeddings @ query))

    def validate_evidence(self, question, candidates):
        if not candidates:
            return [], "insufficient_evidence"
        top_score = float(candidates[0].get("score", 0.0))
        domain_score = self.domain_score(question)
        if top_score < self.min_scope_score or domain_score < self.min_domain_score:
            return [], "out_of_scope"
        return candidates, "supported"

    def retrieve(self, question):
        candidates = self.retrieve_candidates(question)
        evidence, _ = self.validate_evidence(question, candidates)
        return evidence

    @staticmethod
    def build_context(evidence):
        """Formata as evidências aceitas no contexto que será enviado ao LLM."""
        blocks = []
        for item in evidence:
            metadata = item.get("metadata", {})
            blocks.append(
                f"[{item['document_id']}] "
                f"score={item.get('score', 0):.4f} "
                f"review_score={metadata.get('review_score')} "
                f"data={metadata.get('review_creation_date')}\n"
                f"{item['text']}"
            )
        return "\n\n".join(blocks)

    def ask(self, question):
        candidates = self.retrieve_candidates(question)
        evidence, status = self.validate_evidence(question, candidates)
        context = self.build_context(evidence)

        if status == "out_of_scope":
            answer = self.OUT_OF_SCOPE_MESSAGE
        else:
            answer = self.generator.generate(question, evidence)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "context": context,
            "retrieved_candidates": candidates,
            "has_evidence": bool(evidence),
            "evidence_status": status,
            "top_retrieval_score": float(candidates[0]["score"]) if candidates else None,
            "domain_score": self.domain_score(question) if question and question.strip() else 0.0,
        }
