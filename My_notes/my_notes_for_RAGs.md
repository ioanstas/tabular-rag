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

1. we have csv files -> we create them manually probably or download also. We probably need relatively small many tables so the table summaries make sense? First step is to create/find the tables. Second is to create the evaluation dataset for the tables by using an LLM probably and ask the llm "according to the table description (if we have the description) and the table, what is a question that a user could ask that would retrieve this  table?". Then the LLM gives us some questions, then we keep the questions somewhere to know which question corresponds to which table, and we create a dataset. For example 100 tables (5 rows, 5 cols) and 1 summary(.txt) for each made by LLM -> 1 question per each (ground truth). (a question maps to the ID of a table). The question acts like a label ground truth for our csv-summary pair. Do i evaluate the retrieval right after retrieval? Just before the llm gives the answer. During the hybrid search block we probably add the retrieval evaluation using those 100 questions?  Find metric for retrieval if my retrieval isnt 1st but it actually is in the top k.
Another idea would be is to have 1 big cvs, only 1, and create a summary for each row
2. breaks into sparse and dense embeddings in the following way
3. for the sparse embeddings we only need the csv files
4. for the dense embedding we create summaries from the csv files, then dense embeddings
5. Then they both converge into a hybrid search block which  (Qdrant?)
6. then you ask same question? t
7. then again hybrid search
8. then relevant knowledge block


0. Additional notes and idea: maybe add pre-retrieval method, post-retrieval, re-ranking
0.1. 
- For the evaluation first i need to learn how evalluation is usually done
- also i have to make a model create questions or a dataset that evaluates my RAG according to a colleague? So this is a way to test the retrieval of the RAG. 
Colleague says there is no need to do evaluation on the answer but i might do it for . Some ideas are `exact match`, `semantic similarity`, and maybe something else? Or another idea for the evaluadtion of the answer is to use LLM as a judge and ask how good you think you answered the question
- Maybe i can evaluate hybrid search?
- Use MTEB leaderboard to choose a model?

The whole tabular rag is happening because text to sql doesnt work very well for a RAG. Perhaps i can even implement PNEUMA immediately or even take code parts?
Use openrouter?
Use hugging face -> get a model  like mistral


## Offline Block 
  1. csv tables + manifest
    - We create the manifest in this part i think to help us (need more info)
  2. Ingestion and Validation
    - Q: What do we do? Read the manifest and load all 60 CSV tables with their IDs and metadata.
    - Q: What do we check? Files exist, CSVs load, IDs are unique, and row/column counts match the manifest.
    - Q: Why? Ensure the source data is complete, consistent, and traceable before processing it.
  3. Deterministic representation
    - Q: What do we do? Convert each table into text containing its ID, title, domain, column names, and all rows.
    - Q: What do we preserve? Exact values and stable row/column order—the same input always produces the same text.
    - Q: What do we save? One record per table in artifacts/sparse_documents.jsonl, containing table_id and
      serialized_text.
  4. BM25 index and sparse-retrieval baseline
    - Q: What do we load? The 60 serialized table documents and their table_id values.
    - Q: What do we build? Tokenize the text into searchable terms and build a BM25 index. Keep each document linked to its table ID.
    - Q: How do we search? Tokenize the question using the same rules, score the documents, and return the top-ranked table IDs.
    - Q: What do we check? Try about five sample questions and inspect whether the expected tables appear near the top.
    - Q: Why? Establish a simple, measurable retrieval baseline before adding semantic search.

    BM25:
    Offline part:
      1. loads documents (jsonl to be exact) then converts it to a sparseDocument
      2. tokenizes, converts them into bm25 counts
      3. Returns a SparseIndex with `build_bm25(documents)` And this is the end of offline mode
    Online part:
      1. `search_bm25(index, question, top_k)`
  5. Dense embeddings side/ index:
    - Q: What is an embedding? A numerical vector representing aspects of a text’s meaning. Related texts should have similar vectors.
    - Q: What do we embed? One descriptive summary per table. We prepared these summaries separately and saved them in artifacts/table_summaries.jsonl.
    - Q: Why summaries? They describe what each table contains in natural language, helping retrieval when questions use different wording.
    - Q: Do we train a model? No. We use a pretrained embedding model to encode our summaries. This is inference, not training.
    - Q: What do we build? A DenseIndex containing the vectors, ordered table IDs, and model/settings metadata.
    - Q: What do we save? artifacts/dense_index.npz. Our current matrix has shape (60, 384): 60 summaries, each represented by 384 numbers.
    - Q: What do we check? Every table has one vector; IDs remain aligned; values are finite; normalized vectors have length approximately 1; inputs aren’t truncated; saving/loading preserves the data.
    - Q: Why? Prepare reusable semantic representations for online retrieval.
  
  
  # Choosing an embedding model
    - Q: What criteria matter?
      - Retrieval quality: does it retrieve the correct tables on reviewed development questions? Compare Recall@k and MRR.
      - Language and domain: does it support our text’s language and terminology?
      - Input length: can it encode the complete summaries without truncation?
      - Speed and resources: how much CPU/GPU memory and processing time does it require?
      - Deployment and cost: local hosting versus API charges, network dependence, and data-handling requirements.
      - Vector size: more dimensions require more storage and computation; they don’t automatically mean better retrieval.
    - Q: What did we choose and why?
      sentence-transformers/all-MiniLM-L6-v2: a compact English model suitable for an initial local baseline. It produces 384-dimensional vectors and has a default limit of 256 wordpieces. Our summaries fit
      within that limit. We haven’t yet established that it gives the best retrieval results. Model card

  ```text   
  Option                      How it works
  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Hugging Face/local model    Download model weights, then encode text on your machine. This is our approach.
  ──────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   OpenAI API                  Send text to a dedicated embedding model such as text-embedding-3-small or text-embedding-3-large. These are separate from GPT chat models. OpenAI documentation
  ──────────────────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Claude/Anthropic            Anthropic currently doesn’t offer its own embedding model. Its documentation demonstrates Voyage AI, a separate embedding provider. Claude can still generate the final answer.
                               Anthropic documentation
  ```
