# Model-Agnostic Port of Anthropic's RAG Course

This repository ports the **RAG and Agentic Search** section of Anthropic's
*Claude with the Anthropic API* course to a provider-agnostic Python project.
It replaces Claude and Voyage AI dependencies with OpenAI-compatible chat
providers and Google AI Studio embeddings while preserving the course's core
retrieval concepts.

The project is intentionally educational: the retrieval algorithms and indexes
are implemented directly so their behavior and trade-offs remain visible.

## Course Correlation

| Anthropic course section | Project implementation |
| --- | --- |
| Introducing Retrieval Augmented Generation | `lessons/01_generation_boundary.py`, `lessons/04_grounded_rag.py` |
| Text Chunking Strategies | `lessons/05_text_chunking.py`, `src/rag_learn/chunking.py` |
| Text Embeddings | `lessons/03_semantic_retrieval.py`, `src/rag_learn/embeddings.py` |
| The Full RAG Flow | `lessons/04_grounded_rag.py`, `lessons/06_chunked_rag.py` |
| Implementing the RAG Flow | `src/rag_learn/indexes.py`, `src/rag_learn/rag.py` |
| BM25 Lexical Search (`004_bm25.ipynb`) | `lessons/07_bm25_search.py`, `src/rag_learn/bm25.py`, `BM25Index` |
| A Multi-Index RAG Pipeline (`005_hybrid.ipynb`) | `lessons/08_hybrid_search.py`, `src/rag_learn/hybrid.py`, `Retriever` |
| Contextual Retrieval | `lessons/09_contextual_retrieval.py`, `src/rag_learn/contextual.py` |

The course-aligned sample document is stored at `data/report.md`.

## Architecture

```text
Documents
  -> chunking
  -> BM25Index + VectorIndex
  -> Reciprocal Rank Fusion Retriever
  -> relevant context
  -> grounded generation with citations
```

Generation providers are attempted in this order:

1. Google AI Studio Gemma 4 26B A4B
2. OpenRouter free-model router
3. OpenRouter Kimi K2.6 free

Google's `gemini-embedding-001` is used for embeddings. One embedding model is
used consistently because vectors from different models are not comparable.

## Setup

Requires Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env.local
```

Add provider credentials to `.env.local`:

```dotenv
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
```

`.env.local` is ignored by Git. Keep `.env.example` committed with empty
values so required configuration remains documented without exposing secrets.

## Run

Run any lesson from the repository root:

```powershell
.\.venv\Scripts\python.exe lessons\08_hybrid_search.py
.\.venv\Scripts\python.exe lessons\09_contextual_retrieval.py
```

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Key Modules

- `retrieval.py`: document model and basic keyword retrieval
- `chunking.py`: fixed-size and structure-aware chunking
- `embeddings.py`: embedding provider and cosine similarity
- `indexes.py`: reusable in-memory BM25 and vector indexes
- `hybrid.py`: general multi-index retriever and Reciprocal Rank Fusion
- `rag.py`: grounded answer generation and citations
- `contextual.py`: LLM-generated context added before indexing

## Scope and Limitations

This is a learning project, not a production retrieval service:

- Indexes are in memory and are rebuilt between runs.
- Retrieval thresholds and RRF parameters are not tuned on an evaluation set.
- Contextualization adds preprocessing cost and may introduce generated errors.
- There is no persistent vector database, metadata filtering, or ingestion job.
