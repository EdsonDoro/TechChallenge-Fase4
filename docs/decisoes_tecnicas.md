# Decisões técnicas

## 1. Objetivo deste documento

Este documento registra as principais decisões tomadas durante a implementação da solução de Voice of Customer com RAG.

A intenção é explicar **por que** cada tecnologia ou estratégia foi adotada, e não apenas listar as ferramentas utilizadas.

A solução foi construída priorizando quatro características:

- fundamentação das respostas em evidências;
- reprodutibilidade;
- execução local;
- separação clara entre preparação, recuperação e geração.

---

## 2. Fonte e modelagem dos dados

### Decisão

Utilizar o **Brazilian E-Commerce Public Dataset by Olist**, tendo as avaliações como núcleo da base de conhecimento e utilizando as demais tabelas para enriquecimento.

### Justificativa

A avaliação contém a informação textual diretamente relacionada à percepção do cliente. Entretanto, o texto isolado perde parte do contexto disponível na base.

Por isso, a implementação relaciona avaliações a:

- pedidos;
- clientes;
- itens;
- produtos;
- categorias;
- vendedores;
- pagamentos.

### Cuidados com duplicação

Uma avaliação pode estar relacionada a múltiplos itens e vendedores. Fazer joins ingênuos poderia transformar uma avaliação em várias linhas.

Para evitar isso, as relações de múltiplos registros são agregadas por `order_id` antes da construção dos documentos.

**Decisão:** uma avaliação válida corresponde a uma unidade semântica no índice.

---

## 3. Limpeza e preparação textual

### Decisão

Aplicar limpeza mínima e preservadora de informação.

A função `clean_text`:

- remove espaços nas extremidades;
- normaliza sequências de whitespace;
- mantém o conteúdo textual e sua capitalização.

Também são removidas avaliações sem comentário textual útil.

### Justificativa

Como os textos são avaliações reais de clientes, transformações agressivas poderiam eliminar informação relevante para a busca semântica.

Não foi aplicada uma normalização que descaracterizasse o texto original.

---

## 4. Chunking

### Decisão

Não realizar chunking das avaliações.

### Justificativa

Cada comentário é relativamente curto e funciona como uma unidade semântica natural.

Fragmentar cada avaliação em vários pedaços poderia:

- aumentar artificialmente o número de documentos;
- separar partes semanticamente relacionadas;
- dificultar a rastreabilidade;
- aumentar a complexidade da recuperação.

Nesse contexto, manter uma avaliação por documento é mais simples e coerente com o objetivo de recuperar evidências de clientes.

---

## 5. Embeddings

### Decisão

Utilizar:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### Justificativa

A escolha considera:

1. suporte multilíngue;
2. adequação para textos em português;
3. execução local;
4. ausência de custo de API;
5. integração direta com Sentence Transformers.

O mesmo modelo transforma documentos e perguntas, garantindo que a recuperação compare representações no mesmo espaço vetorial.

---

## 6. Índice vetorial

### Decisão

Utilizar **FAISS** com `IndexFlatIP`.

### Justificativa

FAISS atende ao problema sem introduzir uma infraestrutura externa de banco vetorial.

Para o escopo da entrega, a simplicidade de um índice local é uma vantagem: o artefato pode ser reconstruído pelo script `scripts/build_index.py`.

Os vetores são normalizados. Dessa forma, o produto interno representa a similaridade cosseno.

### Persistência

São mantidos:

- `index.faiss`;
- `documents.json`.

O primeiro contém o índice vetorial e o segundo mantém o mapeamento entre posição no índice e documento/evidências.

---

## 7. Estratégia de recuperação

### Decisão

Utilizar recuperação semântica com:

- `TOP_K=5`;
- `MIN_RELEVANCE_SCORE=0.25`;
- `MIN_SCOPE_SCORE=0.55`;
- `MIN_DOMAIN_SCORE=0.45`.

### Justificativa

O Top-K limita a quantidade de evidências enviada ao LLM e reduz contexto irrelevante.

O limiar mínimo adiciona uma segunda proteção: não basta retornar os vizinhos mais próximos; eles precisam superar um nível mínimo de similaridade configurável.

Os valores foram mantidos configuráveis para permitir experimentação.

### Limitação

Os valores atuais são parâmetros de configuração e não devem ser interpretados como valores universalmente ótimos. Uma avaliação anotada mais extensa poderia comparar diferentes combinações de K e limiar.

---

## 8. LLM

### Decisão

Adotar **Ollama + Llama 3.2** como configuração padrão.

### Motivo

Durante a execução do projeto, a geração via API externa apresentou limitação relacionada a créditos disponíveis. Para tornar a solução executável e reproduzível sem depender de créditos de API, foi adotado um LLM local.

O projeto mantém a possibilidade de utilizar OpenAI como alternativa configurável.

### Benefícios da escolha

- execução local;
- ausência de custo de API no caminho padrão;
- maior reprodutibilidade do ambiente;
- compatibilidade com a interface utilizada pelo projeto.

---

## 9. Prompt e groundedness

### Decisão

O LLM recebe exclusivamente a pergunta e as evidências recuperadas.

O prompt estabelece regras explícitas:

- não inventar fatos;
- não inventar estatísticas;
- não inventar causas;
- reconhecer ausência de evidência;
- diferenciar observações de inferências;
- citar `document_id`.

### Justificativa

O objetivo do RAG não é apenas obter uma resposta plausível, mas relacionar a resposta ao conteúdo recuperado.

Assim, o LLM é tratado como componente de síntese, e não como substituto da base de conhecimento.

---

## 10. Controle de perguntas sem evidência e fora do escopo

### Decisão

Separar candidatos recuperados de evidências consideradas suficientes. O pipeline usa o `MIN_RELEVANCE_SCORE` para eliminar vizinhos muito fracos e, em seguida, aplica um guardrail de domínio e suficiência: o melhor score recuperado precisa atingir `MIN_SCOPE_SCORE` e a pergunta precisa atingir `MIN_DOMAIN_SCORE` contra as âncoras semânticas do domínio.

Quando não há candidatos, o status é `insufficient_evidence`. Quando há candidatos, mas a similaridade é baixa e não existe suporte lexical, o status é `out_of_scope`.

### Justificativa

FAISS sempre retorna os vizinhos mais próximos disponíveis. Portanto, apenas verificar se a lista de resultados não está vazia não é suficiente para concluir que a pergunta é respondível pela base. O segundo estágio reduz esse falso positivo e evita enviar contexto irrelevante ao LLM.

Os valores `MIN_SCOPE_SCORE=0.55` e `MIN_DOMAIN_SCORE=0.45` são configuráveis e devem ser tratados como guardrails heurísticos, não como classificador perfeito de domínio. Uma avaliação anotada pode ser usada futuramente para calibrar esse parâmetro.

---

## 11. Rastreabilidade

### Decisão

Manter `document_id` e metadados junto às evidências recuperadas.

### Justificativa

A rastreabilidade é importante para um projeto de Voice of Customer porque uma conclusão deve poder ser relacionada aos comentários que a sustentam.

A solução permite visualizar:

**pergunta → documentos recuperados → resposta**

tanto pela CLI quanto pelo Streamlit.

---

## 12. Organização do código

### Decisão

Separar a implementação em módulos:

- dados;
- embeddings;
- retrieval;
- geração;
- pipeline;
- avaliação.

### Justificativa

Essa separação evita concentrar toda a lógica no notebook.

O notebook `00_rag_completo.ipynb` funciona como artefato demonstrativo e acadêmico, enquanto `src/` contém a implementação reutilizável.

---

## 13. Notebook como artefato de demonstração

O notebook final foi consolidado em um único fluxo para tornar a jornada do projeto fácil de acompanhar.

Ele demonstra:

- ambiente;
- dados;
- preparação;
- embeddings;
- FAISS;
- recuperação;
- configuração do LLM;
- geração;
- ausência de evidência;
- persistência;
- conclusão.

**O notebook acompanha a implementação atual e demonstra explicitamente os cenários com evidência, insuficiência e fora do escopo.**

---

## 14. Avaliação

Foi criada uma camada de avaliação em `src/evaluation/evaluator.py` com:

- Precision@K;
- Recall@K;
- MRR.

Essas métricas foram escolhidas porque permitem avaliar objetivamente se os documentos relevantes estão sendo recuperados e em que posição aparecem.

### Limitação atual

A infraestrutura de métricas está implementada, mas este repositório não apresenta resultados experimentais que não tenham sido efetivamente executados sobre um conjunto anotado.

Portanto, resultados futuros devem ser registrados com:

- perguntas utilizadas;
- documentos considerados relevantes;
- configuração de K;
- configuração do limiar;
- métricas obtidas;
- análise dos erros.

---

## 15. Testes automatizados

Os testes verificam componentes centrais da solução:

- pré-processamento;
- recuperação;
- limiar;
- pipeline;
- ausência de evidência.

Eles não substituem a avaliação de qualidade com dados anotados, mas ajudam a garantir que mudanças estruturais não quebrem o comportamento esperado.

---

## 16. Alternativas consideradas

### OpenAI

Foi mantida como opção de configuração, mas não como dependência obrigatória.

### Banco vetorial externo

Não foi utilizado porque o escopo atual não exige infraestrutura distribuída. FAISS local é suficiente para demonstrar a estratégia de recuperação.

### Chunking

Não foi utilizado porque a avaliação individual já funciona como unidade semântica adequada.

### Busca híbrida

Não foi incorporada à implementação atual. Pode ser uma evolução futura combinando recuperação lexical e semântica.

---

## 17. Síntese das decisões

| Componente | Escolha |
|---|---|
| Fonte | Olist |
| Documento | Uma avaliação por documento |
| Enriquecimento | Dados relacionais agregados por pedido |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Índice | FAISS IndexFlatIP |
| Similaridade | Produto interno com vetores normalizados |
| Retrieval | Top-K + limiar |
| LLM padrão | Ollama + Llama 3.2 |
| Persistência | FAISS + JSON |
| Interface | Notebook + CLI + Streamlit |
| Avaliação | Precision@K, Recall@K e MRR |
| Controle de evidência | limiar de relevância + validação semântica de domínio + prompt grounded |
