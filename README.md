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

O arquivo principal para o desafio é `olist_order_reviews_dataset.csv`. Os demais CSVs podem ser usados para enriquecimento por metadados, como produto, pedido, entrega e categoria.

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
