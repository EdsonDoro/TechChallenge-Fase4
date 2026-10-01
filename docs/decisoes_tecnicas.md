# Decisões técnicas

Este documento deve registrar, durante o desenvolvimento, as decisões e experimentos realizados pela equipe.

## Dados
- Fonte: Brazilian E-Commerce Public Dataset by Olist.
- Foco: avaliações textuais.
- Enriquecimentos possíveis: nota, período, pedido, entrega e categoria.

## Pré-processamento
Registrar:
- critérios de seleção;
- tratamento de nulos;
- normalização;
- estratégia de chunking, se aplicável.

## Embeddings
Registrar:
- modelo escolhido;
- motivo da escolha;
- idioma/cobertura;
- dimensão dos vetores.

## Recuperação
Registrar:
- banco/índice vetorial;
- similaridade;
- top-k;
- limiar de relevância;
- eventuais filtros por metadados.

## Geração
Registrar:
- modelo;
- prompt;
- temperatura;
- regras para evitar respostas sem evidências.

## Avaliação
Registrar:
- conjunto de perguntas;
- documentos esperados;
- métricas;
- análise qualitativa;
- casos sem evidência.
