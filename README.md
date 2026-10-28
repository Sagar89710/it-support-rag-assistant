# Enterprise IT Support RAG and LangGraph Assistant

A portfolio project demonstrating a document-grounded IT support assistant built with Python, LangChain, FAISS, and LangGraph.

## What it demonstrates

- Markdown document loading and recursive text chunking
- Embedding-based semantic retrieval with FAISS
- Source citations for retrieved knowledge-base sections
- LangGraph state management and conditional routing
- Answer generation, response validation, and fallback handling
- A local demo mode that runs without an API key
- Optional OpenAI mode for LLM-generated answers
- Automated tests and a small retrieval evaluation set

## Architecture

1. Load the support knowledge base.
2. Split the content into overlapping chunks.
3. Create embeddings and index chunks in FAISS.
4. Retrieve the most relevant chunks for a user question.
5. Generate a grounded answer using the retrieved context.
6. Validate that the answer contains support context and citations.
7. Route unsupported or invalid responses to a safe fallback.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py app.py "How do I reset my password?"
```

The default local mode uses deterministic keyword-hash embeddings and a deterministic answer generator, so the pipeline can be demonstrated without sending data to an external API. FAISS is still used for vector indexing and retrieval, with a lightweight lexical reranker for reliable offline topic matching.

To use OpenAI embeddings and answer generation, copy `.env.example` to `.env`, replace `your_api_key_here` with your own API key, and run:

```powershell
py app.py "What should I include when escalating an incident?" --openai
```

Never commit `.env` or share the API key. The project reads `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_EMBEDDING_MODEL` from environment variables.

## Quality checks

Run the automated tests and offline evaluation before sharing changes:

```powershell
py -m pytest -q
py evaluate.py
```

The tests verify knowledge-base loading, a cited answer for a supported question, and fallback behavior for an unsupported question. The evaluation script reports retrieval accuracy for representative support questions.

## Example output

```text
Answer: Based on the support knowledge base: Employees can reset a forgotten password from the company identity portal. Select Forgot password, verify the registered mobile number, and follow the reset link. Passwords must be at least 12 characters and must not reuse the previous five passwords.
Sources: support_knowledge_base.md [Password Reset], support_knowledge_base.md [Software Installation]
Validation: passed
```

## Design notes

- FAISS stores vectors for semantic similarity search; it does not generate answers.
- Chunk overlap helps preserve context across chunk boundaries.
- The graph keeps retrieval, generation, validation, and fallback as separate testable nodes.
- The validator checks whether the response is grounded and cited; unsupported questions receive a fallback instead of an invented answer.
- The evaluation set measures whether expected support sections are retrieved for representative questions.
- In production, the local embeddings and deterministic generator can be replaced or supplemented with managed embeddings, an LLM, access controls, monitoring, and an evaluation dataset.
