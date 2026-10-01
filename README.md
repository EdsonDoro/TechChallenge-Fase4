# Tech Challenge — Fase 4 | Voice of Customer Intelligence com RAG

Solução de **Retrieval-Augmented Generation (RAG)** para exploração inteligente das avaliações de clientes do **Brazilian E-Commerce Public Dataset by Olist**.

## Objetivo

Construir uma solução que permita fazer perguntas em linguagem natural sobre as avaliações dos clientes, recuperar evidências relevantes da base e gerar respostas fundamentadas nessas evidências.

Fluxo principal:

**Pergunta → processamento → embeddings → recuperação semântica → contexto → LLM → resposta com evidências**

## Estrutura

```text
TechChallenge-Fase4/
├── app/
│   └── streamlit_app.py
├── config/
│   └── settings.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── vectorstore/
├── docs/
│   ├── arquitetura.md
│   ├── decisoes_tecnicas.md
│   └── roteiro_video.md
├── notebooks/
│   ├── 01_exploracao_olist.ipynb
│   ├── 02_preparacao_base.ipynb
│   ├── 03_embeddings_indexacao.ipynb
│   └── 04_avaliacao_rag.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── generation/
│   │   └── llm.py
│   ├── rag/
│   │   └── pipeline.py
│   ├── evaluation/
│   │   └── evaluator.py
│   └── utils/
│       └── logging.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Base de dados

Utilize o Brazilian E-Commerce Public Dataset by Olist, disponibilizado no Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data

Os arquivos da base **não devem ser versionados no Git**. Coloque os CSVs em `data/raw/`.

O corpus textual principal é `olist_order_reviews_dataset.csv`. A implementação atual utiliza também as demais tabelas para enriquecer cada evidência sem duplicar avaliações: `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_customers_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `olist_order_payments_dataset.csv` e `product_category_name_translation.csv`. A tabela `olist_geolocation_dataset.csv` permanece disponível para extensões geográficas.

### Relacionamento utilizado no corpus

```text
reviews (review_id, order_id, review_score, comentário)
              │
              └── order_id ──> orders ──> customer_id ──> customers
                                  │
                                  ├── order_items ──> products ──> category translation
                                  │              └── sellers
                                  └── payments
```

Os itens, categorias, vendedores e formas de pagamento são agregados por pedido antes do enriquecimento. Assim, uma avaliação continua sendo uma única unidade semântica no índice vetorial.

## Requisitos atendidos pela estrutura

### 1. Construção da base de conhecimento
- seleção dos dados;
- limpeza e preparação dos textos;
- tratamento de valores ausentes;
- organização dos documentos;
- geração de embeddings;
- armazenamento/indexação.

### 2. Pipeline RAG
- processamento da pergunta;
- representação vetorial;
- recuperação semântica;
- seleção de documentos;
- construção de contexto;
- geração da resposta.

### 3. Respostas fundamentadas
O pipeline deve retornar as evidências recuperadas juntamente com a resposta, mantendo identificadores das avaliações e metadados disponíveis.

### 4. Perguntas sem evidências suficientes
O pipeline possui um limiar de relevância configurável e deve recusar respostas quando a recuperação não atingir evidência mínima suficiente.

### 5. Metadados
A estrutura suporta enriquecimento com:
- nota da avaliação;
- período;
- pedido;
- entrega;
- categoria de produto.

### 6. Entrega
O repositório contempla:
- código-fonte;
- instruções;
- README;
- arquitetura;
- tecnologias;
- estratégia RAG;
- preparação da base;
- decisões técnicas;
- exemplos;
- avaliação;
- roteiro para vídeo.

## Instalação

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Depois:

```bash
pip install -r requirements.txt
copy .env.example .env
```

Configure as variáveis de ambiente no `.env`.

## Execução da aplicação

```bash
streamlit run app/streamlit_app.py
```

## Testes

```bash
pytest -q
```

## Organização recomendada da implementação

1. Baixar a base Olist.
2. Colocar os CSVs em `data/raw/`.
3. Executar a preparação da base.
4. Gerar embeddings.
5. Criar o índice vetorial.
6. Executar consultas de recuperação.
7. Construir o contexto.
8. Gerar resposta com o LLM.
9. Exibir evidências.
10. Avaliar recuperação e respostas.
11. Registrar exemplos no README e no vídeo.

## Observação

O PDF do desafio não impõe uma tecnologia específica para embeddings, banco vetorial ou LLM. A equipe deve justificar tecnicamente as escolhas realizadas.

## Execução do RAG completo

1. Coloque todos os CSVs do dataset em `data/raw/`, mantendo estes nomes: `olist_order_reviews_dataset.csv`, `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_customers_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `olist_order_payments_dataset.csv`, `product_category_name_translation.csv` e, opcionalmente, `olist_geolocation_dataset.csv`.
2. Crie o ambiente e instale as dependências:
   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # Linux/macOS
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copie `.env.example` para `.env` e informe `OPENAI_API_KEY` para habilitar a geração.
4. Construa a base vetorial:
   ```bash
   python scripts/build_index.py
   ```
   O processo gera `data/vectorstore/index.faiss` e `data/vectorstore/documents.json`.
5. Consulte pelo terminal:
   ```bash
   python scripts/query_rag.py "Quais são os principais problemas relacionados à entrega?"
   ```
6. Execute a aplicação:
   ```bash
   streamlit run app/streamlit_app.py
   ```

### Componentes implementados

- **Pré-processamento:** limpeza, remoção de avaliações sem comentário, junção das tabelas relacionais e preservação de metadados.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2`, adequado para consultas em português sem depender de API para a etapa vetorial.
- **Indexação:** FAISS com similaridade por produto interno e normalização dos vetores.
- **Persistência:** índice e documentos são salvos em `data/vectorstore/`.
- **Retrieval:** Top-K configurável com limiar mínimo de relevância.
- **Generation:** LLM recebe somente o contexto recuperado e é instruído a citar IDs das evidências.
- **Controle de insuficiência:** quando não há evidências acima do limiar, o pipeline responde explicitamente que a base não sustenta a pergunta.
- **Rastreabilidade:** cada evidência mantém `document_id` e metadados da avaliação.
- **Avaliação:** `precision@k`, `recall@k` e MRR estão disponíveis em `src/evaluation/evaluator.py`.
- **Testes:** cobertura de recuperação, pipeline e cenário sem evidência em `tests/`.
- **Notebook final:** `notebooks/00_rag_completo.ipynb`.

> O arquivo CSV e os artefatos vetoriais não devem ser versionados. O índice pode ser reconstruído deterministically com `scripts/build_index.py`.

