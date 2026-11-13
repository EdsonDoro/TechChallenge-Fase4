# Notebooks

A jornada experimental está dividida em quatro notebooks temáticos e um notebook integrado.

## Ordem recomendada

| Ordem | Notebook | Conteúdo |
|---|---|---|
| 1 | `01_exploracao_olist.ipynb` | Dimensões, tipos, ausências, distribuição das notas e amostra de comentários |
| 2 | `02_preparacao_base.ipynb` | Limpeza textual, enriquecimento relacional, metadados e documentos semânticos |
| 3 | `03_embeddings_indexacao.ipynb` | Modelo de embeddings, vetorização, índice FAISS, persistência e consulta de inspeção |
| 4 | `04_avaliacao_rag.ipynb` | Métricas de retrieval, auditoria de resultados, evidência insuficiente e fora do domínio |

O notebook `00_rag_completo.ipynb` mantém a demonstração integrada de ponta a ponta.

## Pré-requisitos

1. Criar e ativar o ambiente Python do projeto.
2. Instalar as dependências com `pip install -r requirements.txt`.
3. Baixar o Brazilian E-Commerce Public Dataset by Olist.
4. Colocar os CSVs originais em `data/raw/`.
5. Abrir os notebooks a partir da raiz do repositório ou da pasta `notebooks/`.

Os arquivos originais da Olist não são versionados no Git. Os notebooks de preparação e indexação carregam as tabelas necessárias por meio de `src/data/loader.py`.

## Artefatos produzidos

- O notebook 02 grava `data/processed/reviews_enriched.csv` para inspeção.
- O notebook 03 grava `data/vectorstore/index.faiss` e `data/vectorstore/documents.json`.
- O notebook 04 grava `data/evaluation/retrieval_results_notebook.json`.

Os artefatos podem estar ignorados pelo Git e devem ser gerados localmente antes de executar as etapas dependentes.

## Reutilização e manutenção

Os notebooks não duplicam as regras centrais da aplicação. Eles importam as implementações de `src/` para leitura, preparação, embeddings, recuperação e avaliação.

As células de geração de embeddings e construção do FAISS podem consumir memória e levar alguns minutos. A avaliação quantitativa mede a qualidade do retrieval em relação ao conjunto anotado; por isso, a revisão manual das anotações é parte necessária da interpretação dos resultados.
