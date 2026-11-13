# Roteiro da demonstração em vídeo

## Objetivo

Apresentar, de forma objetiva e técnica, como o projeto transforma avaliações de clientes da Olist em uma solução de Voice of Customer baseada em RAG.

A demonstração deve privilegiar **o problema, as decisões e a evidência de funcionamento**, em vez de apenas navegar pelos arquivos do repositório.

## 1. Contextualização do problema

Apresentar:

- o problema de explorar grande volume de avaliações textuais;
- a proposta de utilizar RAG para consultar esse conhecimento em linguagem natural;
- o objetivo de produzir respostas fundamentadas nas avaliações.

Mensagem principal:

> O sistema não busca apenas gerar uma resposta; ele busca recuperar evidências e utilizar essas evidências para fundamentar a resposta.

## 2. Dados utilizados

Mostrar:

- dataset Olist;
- tabela de avaliações;
- principais tabelas utilizadas para enriquecimento;
- relacionamento entre avaliação e pedido.

Explicar que uma avaliação continua sendo uma única unidade semântica, mesmo quando o pedido possui vários itens, categorias ou vendedores.

## 3. Preparação da base

Demonstrar no notebook:

- validação dos arquivos;
- limpeza dos comentários;
- remoção de comentários vazios;
- enriquecimento com tabelas relacionais;
- criação dos documentos;
- preservação dos metadados.

Explicar por que não foi aplicado chunking nas avaliações.

## 4. Embeddings e índice

Mostrar:

- modelo `paraphrase-multilingual-MiniLM-L12-v2`;
- geração dos embeddings;
- normalização;
- índice FAISS;
- persistência em `data/vectorstore/`.

Explicar que o mesmo modelo representa documentos e perguntas.

## 5. Recuperação semântica

Executar uma pergunta representativa, por exemplo:

> "Quais são os principais problemas relacionados à entrega?"

Mostrar:

- embedding da pergunta;
- documentos recuperados;
- scores;
- `document_id`;
- metadados.

A ideia é evidenciar que a recuperação acontece por significado, e não somente por correspondência literal de palavras.

## 6. Geração da resposta

Mostrar a resposta produzida pelo Llama 3.2 via Ollama.

Explicar que o LLM recebe a pergunta e as evidências recuperadas como contexto.

Destacar as instruções do prompt para:

- não inventar informações;
- usar apenas as evidências;
- citar os documentos utilizados.

## 7. Rastreabilidade

Mostrar pelo menos uma resposta acompanhada dos documentos que a sustentam.

Explicar o caminho:

**pergunta → evidências → resposta**

Esse é um dos pontos centrais da solução.

## 8. Casos sem evidência e fora do escopo

Executar dois cenários:

1. uma pergunta sem evidência suficiente na base;
2. uma pergunta claramente fora do domínio, por exemplo: "Qual é a previsão do tempo para amanhã?".

Mostrar a diferença entre candidatos recuperados pelo FAISS e evidências aceitas. No segundo cenário, mesmo que o FAISS retorne vizinhos, o guardrail deve produzir `out_of_scope`, `Evidências aceitas: 0` e `Tem evidência: False`.

Explicar o papel conjunto de `MIN_RELEVANCE_SCORE`, `MIN_SCOPE_SCORE` e `MIN_DOMAIN_SCORE`.

## 9. Execução fora do notebook

Mostrar rapidamente a mesma solução pela CLI:

```bash
python scripts/query_rag.py "Quais são os principais problemas relacionados à entrega?"
```

Se desejado, mostrar a interface Streamlit:

```bash
streamlit run app/streamlit_app.py
```

## 10. Decisões técnicas

Comentar brevemente:

- por que a avaliação é o documento;
- por que foi utilizado um embedding multilíngue;
- por que FAISS foi suficiente;
- por que o Llama 3.2 local foi adotado como padrão;
- por que existe um limiar de relevância;
- por que a resposta mantém as evidências.

## 11. Encerramento

Finalizar retomando o objetivo:

**transformar avaliações não estruturadas em conhecimento consultável, com respostas fundamentadas e rastreáveis.**

O encerramento pode mencionar extensões futuras, como avaliação anotada mais ampla, busca híbrida e reranking, deixando claro que são evoluções e não componentes obrigatórios da implementação atual.
