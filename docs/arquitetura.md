# Arquitetura da solução

## 1. Visão arquitetural

A solução foi organizada como um pipeline RAG em duas etapas principais:

1. **Construção da base de conhecimento** — transformação das tabelas Olist em documentos enriquecidos, geração dos embeddings e criação do índice vetorial.
2. **Consulta** — transformação da pergunta em embedding, recuperação das evidências, montagem do contexto e geração da resposta.

A arquitetura busca manter uma separação clara entre dados, representação vetorial, recuperação, geração e orquestração.

## 2. Fluxo completo

```text
                         CONSTRUÇÃO DA BASE
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
   CSVs da Olist                                  Preparação / enriquecimento
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                         Documentos semânticos
                                  │
                                  ▼
                              Embeddings
                                  │
                                  ▼
                           Índice FAISS
                                  │
                                  │
══════════════════════════════════╪══════════════════════════════════
                                  │
                                  │          CONSULTA
                                  │
                                  ▼
                         Pergunta do usuário
                                  │
                                  ▼
                         Embedding da pergunta
                                  │
                                  ▼
                       Busca por similaridade
                                  │
                                  ▼
                    Top-K + limiar de relevância
                                  │
                                  ▼
                       Evidências recuperadas
                                  │
                                  ▼
                       Construção do contexto
                                  │
                                  ▼
                         LLM / Llama 3.2
                                  │
                                  ▼
                    Resposta + IDs das evidências
```

## 3. Camada de dados

A entrada principal é a tabela de avaliações:

`olist_order_reviews_dataset.csv`

O comentário da avaliação é o texto utilizado na recuperação. As tabelas relacionais são utilizadas para contextualização:

- pedidos;
- clientes;
- itens;
- produtos;
- categorias;
- vendedores;
- pagamentos.

As relações de cardinalidade múltipla são agregadas por pedido antes de formar o documento final.

Essa estratégia evita que uma mesma avaliação apareça repetida no índice apenas porque possui vários itens ou vendedores.

## 4. Documento semântico

Cada avaliação válida origina um documento.

O documento possui:

- `document_id`;
- texto limpo;
- metadados da avaliação;
- metadados do pedido;
- informações de entrega;
- informações de produto/categoria;
- informações comerciais disponíveis.

O texto da avaliação continua sendo o núcleo semântico. Os demais campos servem para enriquecer o contexto e a rastreabilidade.

## 5. Embeddings

O componente `src/embeddings/embedder.py` utiliza Sentence Transformers com:

`paraphrase-multilingual-MiniLM-L12-v2`

O mesmo modelo é utilizado para documentos e perguntas, garantindo que ambos estejam no mesmo espaço vetorial.

Os embeddings são normalizados antes da indexação e da consulta.

## 6. Recuperação

O componente `src/retrieval/retriever.py` utiliza:

- FAISS;
- `IndexFlatIP`;
- vetores normalizados;
- Top-K;
- limiar mínimo de relevância.

O índice e os documentos associados são persistidos em:

```text
data/vectorstore/
├── index.faiss
└── documents.json
```

A persistência permite reconstruir o ambiente sem manter os embeddings apenas na memória.

## 7. Orquestração RAG

O componente `src/rag/pipeline.py` coordena:

1. validação da pergunta;
2. geração do embedding;
3. recuperação;
4. aplicação do critério de relevância;
5. validação de suficiência da evidência, separando candidatos recuperados de evidências aceitas;
6. envio apenas das evidências aceitas ao gerador;
7. retorno da resposta, status e rastreabilidade.

Essa camada não implementa a lógica específica do modelo de linguagem. Ela apenas orquestra os componentes.

## 8. Geração

O componente `src/generation/llm.py` disponibiliza:

- `LLMGenerator`: geração utilizando LLM;
- `EvidenceOnlyGenerator`: gerador determinístico para testes e ambientes sem LLM.

O padrão da aplicação é **Ollama + Llama 3.2**.

O prompt de sistema estabelece que o modelo:

- deve utilizar somente as evidências recebidas;
- não deve inventar informações;
- deve reconhecer insuficiência de evidências;
- deve citar os identificadores dos documentos;
- deve responder em português.

## 9. Controle de insuficiência e fora do escopo

A arquitetura possui três mecanismos complementares:

### Recuperação

Se nenhum documento ultrapassar `MIN_RELEVANCE_SCORE`, a lista de evidências fica vazia.

### Validação de suficiência

A recuperação vetorial produz candidatos. O pipeline verifica o score máximo e o apoio lexical entre a pergunta e os textos recuperados. Quando a similaridade é baixa e não existe apoio lexical, os candidatos não são tratados como evidência suficiente e a consulta é classificada como `out_of_scope`.

### Geração

Somente evidências aceitas são enviadas ao LLM. Quando não há evidência suficiente, a resposta informa a limitação em vez de preencher a lacuna com conhecimento externo.

Essa separação é importante porque um índice vetorial sempre consegue retornar vizinhos; vizinho recuperado não significa, por si só, evidência adequada.

## 10. Interfaces

Existem três formas principais de executar a solução:

### Notebook

`notebooks/00_rag_completo.ipynb`

É a demonstração acadêmica da jornada completa e foi executado de ponta a ponta.

### CLI

`scripts/query_rag.py`

Permite realizar perguntas diretamente pelo terminal.

### Streamlit

`app/streamlit_app.py`

Fornece uma interface para consulta e visualização das evidências recuperadas.

## 11. Responsabilidade dos módulos

| Módulo | Responsabilidade |
|---|---|
| `src/data/loader.py` | leitura dos CSVs |
| `src/data/preprocessing.py` | limpeza, joins, enriquecimento e documentos |
| `src/embeddings/embedder.py` | geração de embeddings |
| `src/retrieval/retriever.py` | índice e recuperação vetorial |
| `src/generation/llm.py` | geração fundamentada |
| `src/rag/pipeline.py` | orquestração |
| `src/evaluation/evaluator.py` | métricas de recuperação |
| `scripts/build_index.py` | construção do índice |
| `scripts/query_rag.py` | consulta via terminal |
| `app/streamlit_app.py` | demonstração interativa |

## 12. Racional arquitetural

A separação por componentes foi adotada para evitar que o notebook se torne o único lugar onde a solução existe.

O notebook demonstra a jornada completa, enquanto os módulos em `src/` concentram as implementações reutilizáveis. Isso permite testar componentes isoladamente, executar a solução por CLI ou Streamlit e substituir tecnologias específicas sem reescrever todo o pipeline.

Essa organização também facilita a explicação acadêmica do projeto, pois cada etapa do RAG possui uma responsabilidade técnica identificável.
