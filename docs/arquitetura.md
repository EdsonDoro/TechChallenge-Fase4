# Arquitetura

```text
                         ┌──────────────────┐
                         │ Pergunta usuário │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Processamento    │
                         │ da pergunta      │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Embedding        │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Índice vetorial  │
                         │ / recuperação    │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Evidências       │
                         │ relevantes       │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Construção do    │
                         │ contexto         │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Modelo de        │
                         │ linguagem        │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Resposta +       │
                         │ evidências       │
                         └──────────────────┘
```

## Componentes

- `src/data`: ingestão e preparação.
- `src/embeddings`: representação vetorial.
- `src/retrieval`: busca semântica.
- `src/generation`: geração controlada por evidências.
- `src/rag`: orquestração do fluxo.
- `src/evaluation`: avaliação da recuperação.
- `app`: interface de demonstração.
