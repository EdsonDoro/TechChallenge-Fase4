from openai import OpenAI

SYSTEM_PROMPT = """Você é um analista de Voice of Customer.
Responda somente com base nas evidências fornecidas.
Se as evidências não forem suficientes, diga explicitamente que não há evidências suficientes na base.
Não invente fatos, avaliações ou estatísticas.
Sempre preserve a rastreabilidade das evidências utilizadas.
"""

class LLMGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def generate(self, question: str, evidence: list[dict]) -> str:
        if not evidence:
            return "Não há evidências suficientes na base de conhecimento para responder a esta pergunta."

        context = "\n\n".join(
            f"[{item['document_id']}] {item['text']}" for item in evidence
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Pergunta: {question}\n\n"
                        f"Evidências recuperadas:\n{context}\n\n"
                        "Responda de forma objetiva e cite os IDs das evidências usadas."
                    ),
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
