# Estrutura do projeto

## 1. Objetivo da organização

A estrutura do repositório foi definida para separar a **implementação reutilizável**, a **demonstração acadêmica**, a **execução operacional** e a **documentação**.

Essa separação é importante porque o notebook apresenta a jornada completa do trabalho, enquanto os módulos Python permitem reutilizar a mesma lógica em scripts e na aplicação web.

## 2. Estrutura atual

```text
TechChallenge-Fase4/
│
├── app/
│   └── streamlit_app.py
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/                 # CSVs da Olist — não versionados
│   ├── processed/           # artefatos intermediários, quando utilizados
│   └── vectorstore/         # índice FAISS + documentos persistidos
│
├── docs/
│   ├── arquitetura.md
│   ├── decisoes_tecnicas.md
│   └── roteiro_video.md
│
├── notebooks/
│   └── 00_rag_completo.ipynb
│
├── scripts/
│   ├── build_index.py
│   └── query_rag.py
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── rag/
│   │   └── pipeline.py
│   │
│   └── evaluation/
│       └── evaluator.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_retriever.py
│   ├── test_rag_pipeline.py
│   └── test_pipeline_no_evidence.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── PROJECT_STRUCTURE.md
└── README.md
```

## 3. Responsabilidade das pastas

### `src/data/`

Responsável pela leitura e preparação dos dados.

`loader.py` carrega os CSVs da Olist.

`preprocessing.py` realiza limpeza, relacionamentos, agregações, enriquecimento e construção dos documentos semânticos.

### `src/embeddings/`

Concentra a geração de embeddings para documentos e perguntas.

### `src/retrieval/`

Implementa o índice vetorial e a recuperação por similaridade.

### `src/generation/`

Isola a comunicação com o LLM e também disponibiliza um gerador determinístico utilizado em testes.

### `src/rag/`

Orquestra o fluxo completo de consulta.

### `src/evaluation/`

Concentra as métricas utilizadas para avaliar a recuperação.

### `scripts/`

Contém os pontos de execução pelo terminal:

- `build_index.py`: prepara o corpus, gera embeddings e persiste o índice;
- `query_rag.py`: carrega o índice e executa uma pergunta.

### `app/`

Contém a aplicação Streamlit para demonstração interativa.

### `notebooks/`

Contém o notebook final da entrega:

`00_rag_completo.ipynb`

Esse notebook consolida a jornada completa do projeto e foi executado de ponta a ponta.

**A versão executada do notebook é um artefato final da entrega e não deve ser alterada nas atualizações de documentação.**

### `tests/`

Contém testes automatizados dos principais comportamentos da solução.

### `docs/`

Registra decisões arquiteturais, justificativas técnicas e o roteiro da demonstração em vídeo.

## 4. Fluxo entre os componentes

```text
CSV Olist
   │
   ▼
src/data
   │
   ▼
Documentos enriquecidos
   │
   ▼
src/embeddings
   │
   ▼
Vetores
   │
   ▼
src/retrieval
   │
   ▼
Evidências
   │
   ▼
src/rag
   │
   ├──────────────► src/generation ─────► Resposta
   │
   └──────────────► Evidências
```

As mesmas implementações são utilizadas pelos diferentes pontos de entrada:

```text
                 ┌──► Notebook
                 │
src/ ────────────┼──► CLI
                 │
                 └──► Streamlit
```

## 5. Artefatos que não devem ser versionados

Os dados originais da Olist, credenciais e artefatos temporários de ambiente não devem ser enviados ao Git.

Em particular:

- CSVs originais;
- `.env` com credenciais;
- ambientes virtuais;
- caches;
- arquivos temporários.

O índice vetorial pode ser reconstruído com:

```bash
python scripts/build_index.py
```

## 6. Relação com a entrega acadêmica

A estrutura foi organizada para permitir demonstrar cada etapa solicitada no desafio:

| Requisito conceitual | Local da implementação |
|---|---|
| Base de conhecimento | `src/data/` |
| Preparação dos textos | `src/data/preprocessing.py` |
| Embeddings | `src/embeddings/` |
| Indexação | `src/retrieval/` |
| Recuperação | `src/retrieval/` |
| Contexto RAG | `src/rag/` |
| Geração | `src/generation/` |
| Ausência de evidência | `src/rag/` + `src/generation/` |
| Avaliação | `src/evaluation/` |
| Demonstração | `notebooks/00_rag_completo.ipynb` |
| Execução CLI | `scripts/query_rag.py` |
| Interface | `app/streamlit_app.py` |
| Justificativas | `docs/decisoes_tecnicas.md` |
| Arquitetura | `docs/arquitetura.md` |

## 7. Princípio de manutenção

O notebook demonstra o projeto; os módulos `src/` implementam o projeto.

Essa distinção evita duplicação de lógica e torna a entrega mais próxima de uma solução de engenharia de dados/IA reutilizável, sem perder o caráter didático exigido por uma pós-graduação.
