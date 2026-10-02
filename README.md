# Tech Challenge — Fase 4 | Voice of Customer Intelligence com RAG

## 1. Visão geral

Este projeto apresenta uma solução de **Retrieval-Augmented Generation (RAG)** aplicada ao problema de **Voice of Customer (VoC)**, utilizando o **Brazilian E-Commerce Public Dataset by Olist**.

A proposta é transformar avaliações textuais de clientes em uma base de conhecimento consultável em linguagem natural. Em vez de depender de leitura manual de milhares de comentários, o sistema representa as avaliações como vetores, recupera semanticamente as evidências mais relacionadas à pergunta e utiliza essas evidências como contexto para a geração da resposta.

O trabalho foi estruturado para demonstrar não somente o funcionamento do RAG, mas também as decisões de engenharia e ciência de dados envolvidas na construção da solução: preparação da base, enriquecimento dos documentos, representação vetorial, recuperação, geração controlada e rastreabilidade das evidências.

### Problema abordado

Avaliações de clientes contêm informações importantes sobre experiência de compra, entrega, produtos e atendimento, porém o volume e a natureza não estruturada desses textos dificultam sua exploração direta.

O objetivo técnico é permitir perguntas como:

> "Quais são os principais problemas relacionados à entrega?"

e produzir uma resposta baseada nos comentários efetivamente recuperados, mantendo os identificadores das avaliações utilizadas como evidência.

### Fluxo da solução

**Pergunta → embedding da pergunta → recuperação semântica → evidências → construção do contexto → LLM → resposta fundamentada**

O fluxo é implementado de forma modular em `src/` e demonstrado de ponta a ponta no notebook final.

---

## 2. Base de dados

A fonte é o **Brazilian E-Commerce Public Dataset by Olist**, disponibilizado publicamente no Kaggle:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data

Os arquivos da base não são versionados no Git. Eles devem ser colocados em `data/raw/`.

O corpus textual principal é:

- `olist_order_reviews_dataset.csv`

Para enriquecer semanticamente cada avaliação, a implementação também utiliza:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_order_payments_dataset.csv`
- `product_category_name_translation.csv`

O arquivo `olist_geolocation_dataset.csv` pode permanecer disponível para extensões geográficas, mas não é necessário para o pipeline atual.

### Por que enriquecer as avaliações?

O comentário da avaliação é a evidência textual principal. Entretanto, informações como nota, data, pedido, entrega, categoria e vendedor acrescentam contexto para interpretação e rastreabilidade.

A implementação evita multiplicar uma avaliação no índice quando existem vários itens, vendedores ou categorias associados ao mesmo pedido. Essas relações são agregadas antes da construção dos documentos.

Assim, **uma avaliação continua representando uma unidade semântica no índice vetorial**.

---

## 3. Preparação da base de conhecimento

A preparação ocorre em `src/data/preprocessing.py`.

As principais etapas são:

1. validação da coluna de comentário;
2. remoção de avaliações sem conteúdo textual útil;
3. limpeza de espaços e normalização de whitespace;
4. conversão de datas;
5. junção com dados de pedido, cliente, itens, produto, vendedor e pagamento;
6. agregação das relações de cardinalidade múltipla por pedido;
7. cálculo de informações derivadas de entrega;
8. preservação de metadados;
9. criação de um `document_id` rastreável;
10. transformação final em documentos semânticos.

A decisão de **não fragmentar artificialmente cada avaliação em chunks** é intencional. As avaliações são textos relativamente curtos e cada comentário constitui uma unidade semântica natural. Dividir esses textos poderia separar uma observação de seu contexto e aumentar a quantidade de documentos sem ganho proporcional de recuperação.

---

## 4. Representação vetorial

Os documentos são transformados em embeddings utilizando:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

A escolha considera principalmente:

- suporte multilíngue;
- adequação para consultas em português;
- execução local;
- ausência de dependência de uma API paga para embeddings;
- integração simples com Sentence Transformers.

Os vetores são normalizados antes da busca.

---

## 5. Recuperação semântica

A recuperação utiliza **FAISS** com `IndexFlatIP`.

O produto interno, combinado com vetores normalizados, funciona como similaridade cosseno.

O processo é:

1. gerar o embedding da pergunta;
2. normalizar o vetor;
3. consultar o índice;
4. recuperar os documentos mais similares;
5. aplicar `top_k`;
6. aplicar um limiar mínimo de relevância;
7. retornar texto, similaridade e metadados.

As configurações padrão são:

- `TOP_K=5`
- `MIN_RELEVANCE_SCORE=0.25`

Esses parâmetros ficam configuráveis no ambiente para permitir experimentação sem alterar o código.

---

## 6. Geração da resposta

A geração utiliza **Ollama + Llama 3.2 localmente** como configuração padrão.

Essa escolha foi feita para que a entrega seja reproduzível sem depender de créditos de uma API externa. A arquitetura, entretanto, mantém OpenAI como alternativa configurável.

O modelo recebe:

- a pergunta original;
- somente as evidências recuperadas;
- os identificadores dos documentos;
- metadados relevantes.

O prompt instrui o modelo a:

- utilizar exclusivamente as evidências fornecidas;
- não inventar fatos;
- diferenciar observação de inferência;
- declarar quando não existem evidências suficientes;
- citar os `document_id` utilizados;
- responder em português do Brasil.

Portanto, o LLM não é utilizado como fonte primária de conhecimento. Seu papel é **sintetizar e comunicar o contexto recuperado**.

---

## 7. Controle de ausência de evidência

Um requisito importante da solução é evitar respostas aparentemente plausíveis para perguntas que a base não consegue sustentar.

O pipeline possui um limiar mínimo de relevância. Quando nenhuma evidência ultrapassa esse limiar, a etapa de geração recebe uma lista vazia e retorna explicitamente que não há evidências suficientes para responder.

Esse comportamento é importante em um sistema de VoC porque uma resposta inventada pode transformar uma hipótese do modelo em uma falsa conclusão sobre a experiência dos clientes.

O cenário sem evidência também é coberto por teste automatizado.

---

## 8. Rastreabilidade

Cada documento mantém um identificador associado à avaliação original.

As evidências recuperadas preservam:

- `document_id`;
- texto da avaliação;
- score de similaridade;
- nota da avaliação;
- datas;
- pedido;
- entrega;
- categoria;
- outros metadados disponíveis.

A interface Streamlit apresenta essas evidências junto da resposta. O script de consulta também imprime os documentos recuperados.

Isso permite percorrer o caminho:

**pergunta → evidência recuperada → resposta**

e torna a saída auditável.

---

## 9. Organização do código

A implementação foi separada por responsabilidade:

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
│   └── 00_rag_completo.ipynb
├── scripts/
│   ├── build_index.py
│   └── query_rag.py
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
│   └── evaluation/
│       └── evaluator.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_retriever.py
│   ├── test_rag_pipeline.py
│   └── test_pipeline_no_evidence.py
├── .env.example
├── .gitignore
├── requirements.txt
├── PROJECT_STRUCTURE.md
└── README.md
```

---

## 10. Notebook final

O principal artefato didático da entrega é:

`notebooks/00_rag_completo.ipynb`

Ele consolida a jornada completa do projeto:

1. preparação do ambiente;
2. descoberta e validação dos arquivos;
3. carregamento dos dados;
4. exploração da estrutura;
5. construção do corpus enriquecido;
6. geração de embeddings;
7. construção do índice FAISS;
8. recuperação semântica;
9. configuração do LLM;
10. geração de resposta;
11. tratamento de ausência de evidência;
12. exemplos de consultas;
13. persistência e recarga do índice;
14. conclusão.

**Este notebook foi executado de ponta a ponta e a versão versionada no repositório representa a execução final do trabalho.**

O código de produção utilizado pelo notebook está em `src/`; portanto, o notebook funciona como demonstração reprodutível da solução, enquanto a lógica principal permanece organizada e reutilizável nos módulos Python.

---

## 11. Execução

### Ambiente

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instalação:

```bash
pip install -r requirements.txt
copy .env.example .env
```

### Dados

Coloque os CSVs da Olist em `data/raw/`.

### Índice vetorial

```bash
python scripts/build_index.py
```

O script gera:

- `data/vectorstore/index.faiss`
- `data/vectorstore/documents.json`

### LLM local

Instale o Ollama e disponibilize o modelo:

```bash
ollama run llama3.2
```

O padrão do projeto é:

```text
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Não é necessário fornecer créditos de API para o caminho padrão.

### Consulta pelo terminal

```bash
python scripts/query_rag.py "Quais são os principais problemas relacionados à entrega?"
```

### Aplicação web

```bash
streamlit run app/streamlit_app.py
```

### Testes

```bash
pytest -q
```

---

## 12. Avaliação

O projeto possui `src/evaluation/evaluator.py` com suporte às métricas:

- Precision@K;
- Recall@K;
- MRR.

A estrutura foi preparada para avaliação com perguntas anotadas e documentos relevantes esperados.

Nesta versão, essas métricas são disponibilizadas como infraestrutura de avaliação; **não são apresentados neste README resultados experimentais que não tenham sido efetivamente executados e registrados**.

Além da avaliação quantitativa da recuperação, o projeto contempla verificações qualitativas importantes para o requisito de RAG:

- relevância das evidências;
- coerência entre contexto e resposta;
- rastreabilidade;
- comportamento em ausência de evidência.

---

## 13. Testes automatizados

Os testes cobrem principalmente:

- preparação dos documentos;
- busca vetorial;
- aplicação do limiar de relevância;
- orquestração do pipeline RAG;
- cenário sem evidência.

Eles complementam a validação manual realizada no notebook e nas consultas pelo terminal.

---

## 14. Decisões técnicas resumidas

| Tema | Decisão | Justificativa |
|---|---|---|
| Unidade semântica | Uma avaliação por documento | O comentário é uma unidade curta e semanticamente coerente |
| Enriquecimento | Dados relacionais agregados por pedido | Acrescenta contexto sem duplicar documentos |
| Embeddings | MiniLM multilíngue | Execução local e suporte adequado ao português |
| Vetores | FAISS | Simplicidade, desempenho e persistência local |
| Similaridade | Produto interno com vetores normalizados | Equivalente à similaridade cosseno nesse cenário |
| Recuperação | Top-K + limiar | Equilibra cobertura e controle de evidência |
| LLM padrão | Ollama + Llama 3.2 | Execução local sem créditos de API |
| Evidência | `document_id` + metadados | Rastreabilidade e auditabilidade |
| Arquitetura | Código modular + notebook final | Separa implementação, reutilização e demonstração |

As decisões são detalhadas em [docs/decisoes_tecnicas.md](docs/decisoes_tecnicas.md).

---

## 15. Limitações e extensões

A solução atual prioriza clareza, reprodutibilidade e aderência ao escopo do desafio.

Possíveis evoluções incluem:

- avaliação quantitativa mais extensa com conjunto anotado;
- combinação de busca lexical e semântica;
- filtros por metadados;
- reranking;
- avaliação automatizada da fidelidade das respostas;
- exploração geográfica utilizando a tabela de geolocalização;
- comparação sistemática entre diferentes modelos de embedding e LLM.

Essas possibilidades são tratadas como extensões e não como funcionalidades já realizadas.

---

## 16. Conclusão

O projeto implementa uma solução RAG completa para Voice of Customer sobre avaliações da Olist.

A principal preocupação metodológica foi manter a resposta vinculada à evidência recuperada. Para isso, o projeto combina preparação estruturada dos dados, embeddings multilíngues, recuperação vetorial, geração controlada por contexto, tratamento explícito de insuficiência de evidência e rastreabilidade por documento.

A entrega também separa a lógica de produção em módulos Python e utiliza o notebook `00_rag_completo.ipynb` como demonstração acadêmica da jornada completa, permitindo relacionar as decisões de ciência de dados com sua implementação prática.
