RAG
Retrieve - Augment - Generate - A mechanism to determine whether retrieval is needed


## Retrieve
- Sparse retrieval: 
* word-based and is applied in text retrieval mostly. S
* `TF-IDF`, `BM25`
Example: Let's say the query (question) is about an ASEP competition. Then word by word we find passages that are about ASEP keyword.

- Dense retrieval:
* We can use dense retrievers. In practice we embed all our database in embedding space, and then we use a similarity (like cosine) and we find the closest ones (closest vectors probably). 
* `word2vec`, `encoders`, `embedders`

- Both retrievers:
* we use both usually. Sparse for exact match, dense for similar match.
* Then we either fuse, concat or weight out their outputs into probably a database.

- Vector database:
* It is a database that stores things into a vector form. The upside of them is a very fast search (eg `Pinecone`, `ElasticSearch`)
*  or similarly a vecor indexing library (`eg. FAISS`)

- Dense Passage Retriever:
* It uses for example a BERT-based backbone and is specifically pretrained for the OpenQA task with question-answer pair data.

In general the route is:
1. Many documents into chucnks, 
2. then chunks into lexicological sparse storage or floating dense storage (maybe not true)
3. Use sparse (tf-idf, bm25) and dense retrievers
4. then vector database for speed (FAISS)
5. choose top K chunks, add them to the query, and then it Generates

Sidenote: Another way of dense retriver is Contriever, which uses **contrastive** learning instead of **unsupervised** learning
(- Contrastive is doing something like simmilary and dissimilarity, so it bring similar texts closer, and pushes dissimlar away.. Codex could input here. It uses contrastive loss, with positive and negative samples.)

- Retrieval Granularity:
How we choose/break chunks. For example per word, per paragraph, per chapter, per section, per subsection, per token

- Pre-Retrieval Enhancement:
This is another idea, that we update our query Q into an enhanced query Q'. I am not sure how we do it but it is useful. A small codex input is useful

- Post-Retrieval Enhancement:
Improve relevance, Increase conciseness, Enhance coherence, Reduce noise, Manage redundancy

- Generation:
Design of generator depends on the task such as: question answering, summarization, dialog systems, long-form content writing
We use parameter accessivle (white box) and inaccesible (black box)
* **Parameter-accessible (white-box)**: These typically use open-source
models (like many variants of Llama, Mistral, T5, BART available on
Hugging Face). In this case we have access to the model weights, can
fine-tune them extensively, and understand their internal workings more
deeply.
* **Parameter-inaccessible (black-box)**: These are commercial LLMs
accessed via APIs (like OpenAI’s GPT series, Anthropic’s Claude, Google’s
Gemini through their APIs). We send an input (prompt + retrieved
context), and we get an output. We do not have access to the model
weights and have limited control over the generation process beyond
prompting and some API parameters.

- input layer integration:
It is a common way to integrate retrieved information/documents is to
combine them with the original input/query and jointly pass them to the
generator

- Router (or router LLM):
Is an intermediate LLm to help the system decide whether to retrive or not

## Evaluation

RAG evaluation should be separated into **retrieval evaluation** and
**generation evaluation**. Otherwise, when the final answer is wrong, we cannot
tell which part of the system failed.

### Evaluation dataset

Create a small set of questions before tuning the RAG. For each question, store:

* The question.
* The expected document or chunk containing the evidence.
* The expected answer, when there is one.
* Whether the question is answerable from the corpus.

The first project can begin with 15-25 questions containing straightforward,
paraphrased, ambiguous and unanswerable examples.

### Retrieval evaluation

Evaluate retrieval before connecting the LLM:

* **Hit@k:** Did at least one correct chunk appear in the top `k` results?
* **Recall@k:** How much of the expected relevant evidence appeared in the top
  `k` results?
* **MRR (Mean Reciprocal Rank):** How high was the first correct result ranked?
* Inspect failures manually to determine whether they were caused by parsing,
  chunking, the embedding model or the retrieval settings.

### Generation evaluation

After adding the LLM, evaluate:

* **Answer correctness:** Did it answer the question correctly?
* **Faithfulness/grounding:** Is every claim supported by the retrieved chunks?
* **Citation accuracy:** Do the cited chunks actually support the answer?
* **Refusal accuracy:** Does it return an `insufficient_evidence` response when
  the corpus cannot answer the question?

The two main failure categories are:

1. **Retrieval failure:** The correct evidence was not retrieved.
2. **Generation failure:** The evidence was retrieved, but the LLM ignored,
   misunderstood or contradicted it.

For hybrid RAG, compare sparse-only, dense-only and hybrid retrieval using the
same questions and metrics.

## Useful libraries and frameworks

### Pydantic

* **Where it is used:** Defines and validates structured Python inputs and
  outputs. In a RAG application it can validate API requests, chunk metadata,
  evaluation records and structured LLM answers such as `answer`, `citations`
  and `insufficient_evidence`.
* **When to use it:** Add it after the basic retrieval pipeline works, when the
  project starts producing structured outputs or gets a FastAPI interface.
* **Industry position:** A widely adopted Python validation standard and the
  natural choice with FastAPI.
* **Alternatives:** Standard-library `dataclasses` or `TypedDict` for simpler
  internal structures; Marshmallow is another validation/serialization library.
* **Documentation:** <https://docs.pydantic.dev/latest/>

### LangGraph

* **Where it is used:** Models multi-step, stateful LLM workflows as nodes and
  edges. It is useful for routers, tool calls, retries, loops, human approval and
  agentic RAG.
* **When to use it:** Later, when implementing agents or adaptive/agentic RAG.
  It is unnecessary for the first dense RAG.
* **Industry position:** A popular framework for controlled agent workflows,
  but not a required standard.
* **Alternatives:** Plain Python functions/state machines, Haystack pipelines or
  LlamaIndex workflows.
* **Documentation:** <https://docs.langchain.com/oss/python/langgraph/overview>

### LangChain

* **Where it is used:** Provides integrations and abstractions for models,
  embedding providers, document loaders, vector stores, tools and agent loops.
  It can connect the components of a RAG application quickly.
* **When to use it:** After implementing the basic pipeline with direct library
  calls, so that its abstractions do not hide the mechanics being learned.
* **Industry position:** Widely known and used, with a large integration
  ecosystem, but it is not the single industry standard or automatically the
  best choice. It adds abstractions rather than making the system non-black-box.
* **Alternatives:** Plain Python with direct model/database clients, LlamaIndex
  for a RAG-focused framework, or Haystack for explicit pipelines.
* **Documentation:** <https://docs.langchain.com/oss/python/langchain/overview>

### LlamaIndex

* **Where it is used:** Focuses on loading, transforming, indexing and
  retrieving private/domain documents for RAG. Its `Document`/`Node` model maps
  naturally to documents and chunks.
* **When to use it:** When document ingestion and retrieval become more complex,
  or when rapid RAG experimentation matters more than implementing every step
  manually.
* **Industry position:** A popular RAG/data framework, but not a universal
  standard.
* **Alternatives:** LangChain, Haystack or a custom pipeline using Sentence
  Transformers and a vector index/database directly.
* **Documentation:** <https://docs.llamaindex.ai/>

### Recommendation for the first RAG

Start with direct, visible components such as plain Python, a sentence-embedding
model and FAISS. Add Pydantic for structured records and outputs. Do not add
LangChain, LlamaIndex or LangGraph until the basic indexing, retrieval and
evaluation pipeline is understood.



### Uni Assignment
