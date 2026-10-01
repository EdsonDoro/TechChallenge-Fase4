class RAGPipeline:
    def __init__(self, embedder, retriever, generator, top_k=5, min_relevance_score=0.25):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score

    def ask(self, question: str) -> dict:
        query_embedding = self.embedder.encode([question])[0]

        evidence = self.retriever.search(
            query_embedding,
            top_k=self.top_k,
            min_score=self.min_relevance_score,
        )

        answer = self.generator.generate(question, evidence)

        return {
            "question": question,
            "answer": answer,
            "evidence": evidence,
        }
