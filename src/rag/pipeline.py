"""Orquestração do fluxo RAG e validação de suficiência das evidências."""
import re
import unicodedata

_STOPWORDS = {
    "a","ao","aos","as","com","como","da","das","de","do","dos","e","em","essa",
    "esse","esta","este","eu","foi","há","isso","na","nas","no","nos","o","os",
    "para","por","que","qual","quais","se","sobre","são","um","uma","umas","uns",
    "é","tem","têm","mais","menos","me","minha","meu","ou","onde","quando","quanto",
    "quantos","cliente","clientes",
}

def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return {t for t in re.findall(r"[a-z0-9]{3,}", normalized) if t not in _STOPWORDS}


class RAGPipeline:
    """Pipeline: pergunta -> retrieval -> validação -> geração."""

    OUT_OF_SCOPE_MESSAGE = (
        "A pergunta parece estar fora do escopo da base de conhecimento Olist "
        "ou não possui evidências suficientes relacionadas ao tema. "
        "Não é possível responder com segurança usando as avaliações disponíveis."
    )
    INSUFFICIENT_EVIDENCE_MESSAGE = (
        "Não há evidências suficientes na base de conhecimento para responder a esta pergunta."
    )

    def __init__(self, embedder, retriever, generator, top_k=5,
                 min_relevance_score=0.25, min_scope_score=0.55):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.min_scope_score = min_scope_score

    def retrieve_candidates(self, question: str) -> list[dict]:
        if not question or not question.strip():
            return []
        query_embedding = self.embedder.encode([question.strip()])[0]
        return self.retriever.search(
            query_embedding, top_k=self.top_k, min_score=self.min_relevance_score
        )

    @staticmethod
    def _lexical_support(question: str, candidates: list[dict]) -> set[str]:
        return _tokens(question) & _tokens(
            " ".join(item.get("text", "") for item in candidates)
        )

    def validate_evidence(self, question: str, candidates: list[dict]) -> tuple[list[dict], str]:
        if not candidates:
            return [], "insufficient_evidence"
        top_score = float(candidates[0].get("score", 0.0))
        lexical_support = self._lexical_support(question, candidates)
        if top_score >= self.min_scope_score or lexical_support:
            return candidates, "supported"
        return [], "out_of_scope"

    def retrieve(self, question: str) -> list[dict]:
        candidates = self.retrieve_candidates(question)
        evidence, _ = self.validate_evidence(question, candidates)
        return evidence

    def ask(self, question: str) -> dict:
        candidates = self.retrieve_candidates(question)
        evidence, status = self.validate_evidence(question, candidates)
        if status == "out_of_scope":
            answer = self.OUT_OF_SCOPE_MESSAGE
        else:
            answer = self.generator.generate(question, evidence)
        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "retrieved_candidates": candidates,
            "has_evidence": bool(evidence),
            "evidence_status": status,
            "top_retrieval_score": float(candidates[0]["score"]) if candidates else None,
        }
