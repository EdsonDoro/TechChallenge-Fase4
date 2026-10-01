import os

from openai import OpenAI


SYSTEM_PROMPT = """Você é um analista de Voice of Customer.
Use exclusivamente as evidências recuperadas da base Olist.
Não invente fatos, estatísticas, causas ou avaliações.
Se as evidências não sustentarem a pergunta, responda que não há evidências suficientes.
Diferencie claramente observações das avaliações dos clientes de inferências.
Sempre inclua os IDs [document_id] das evidências usadas.
Responda em português do Brasil, de forma objetiva.
"""


class LLMGenerator:
    """Gerador compatível com OpenAI e Ollama local.

    Com LLM_PROVIDER=ollama, a geração ocorre localmente e não exige
    OPENAI_API_KEY nem créditos de API.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        self.model = model

        if self.provider == "ollama":
            self.base_url = base_url or os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",
            )
        elif self.provider == "openai":
            self.base_url = base_url
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(
                f"LLM_PROVIDER='{self.provider}' não suportado. "
                "Use 'ollama' ou 'openai'."
            )

    @staticmethod
    def _context(evidence: list[dict]) -> str:
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

    def generate(self, question: str, evidence: list[dict]) -> str:
        if not evidence:
            return "Não há evidências suficientes na base de conhecimento para responder a esta pergunta."

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Pergunta: {question}\n\n"
                        f"Evidências recuperadas:\n{self._context(evidence)}\n\n"
                        "Produza uma resposta fundamentada e cite os IDs entre colchetes."
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()


class EvidenceOnlyGenerator:
    """Gerador determinístico para testes e ambientes sem LLM."""

    def generate(self, question: str, evidence: list[dict]) -> str:
        if not evidence:
            return "Não há evidências suficientes na base de conhecimento para responder a esta pergunta."
        ids = ", ".join(f"[{item['document_id']}]" for item in evidence)
        return f"Foram recuperadas {len(evidence)} evidências relevantes para a pergunta. Evidências: {ids}."
