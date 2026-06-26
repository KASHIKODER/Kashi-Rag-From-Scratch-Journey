# RAG From Scratch — 17 Retrieval-Augmented Generation Techniques Implemented

A hands-on implementation of 17 RAG (Retrieval-Augmented Generation) techniques, built end-to-end using 100% free and open-source tools — no paid APIs required.

This repository documents my journey through LangChain's "RAG From Scratch" curriculum, migrated entirely from OpenAI to **Groq + HuggingFace**, along with every production error encountered and how it was resolved.

---

## What's Inside

| # | Technique | Description |
|---|-----------|-------------|
| 1-4 | **Foundations** | Indexing, retrieval, and generation — the core RAG pipeline |
| 5 | **Multi-Query** | Generates multiple reworded versions of a question to improve retrieval coverage |
| 6 | **RAG-Fusion** | Combines multi-query retrieval with Reciprocal Rank Fusion (RRF) |
| 7 | **Decomposition** | Breaks a complex question into sub-questions, answered sequentially or in parallel |
| 8 | **Step-Back Prompting** | Generates a more abstract version of the question to retrieve broader context |
| 9 | **HyDE** | Retrieves using a hypothetical generated answer instead of the raw question |
| 10 | **Routing** | Logical (LLM function-calling) and Semantic (embedding similarity) routing to the right data source |
| 11 | **Query Construction** | Converts natural language into structured metadata filters |
| 12 | **Multi-Representation Indexing** | Retrieves from document summaries while returning the full document for generation |
| 13 | **RAPTOR** | Recursive clustering and summarization, building a hierarchical tree of abstractions |
| 14 | **ColBERT** | Token-level embeddings with MaxSim scoring for fine-grained retrieval |
| 15 | **Re-Ranking** | Two-stage retrieval using Cohere's reranking model on top of broad vector search |
| 16-17 | **Agentic RAG (CRAG / Self-RAG)** | A LangGraph-based agent that decides when to retrieve, grades document relevance, and rewrites queries on failure |

---

## Tech Stack

**LLM Inference:** Groq (`llama-3.3-70b-versatile`)
**Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
**Frameworks:** LangChain, LangGraph
**Vector Stores:** ChromaDB, InMemoryVectorStore
**Re-Ranking:** Cohere Rerank
**Clustering (RAPTOR):** UMAP, scikit-learn (Gaussian Mixture Models)
**Language:** Python 3.10

> All tools used are free-tier — this project was built without any paid API usage.

---

## Repository Structure

```
rag-from-scratch/
├── part_1_to_11.py          # Foundations through Query Construction
├── part_12_to_17.py         # Advanced indexing through Agentic RAG
├── notes/
│   ├── 01_foundations.md
│   ├── 02_query_translation.md
│   ├── 03_advanced_indexing.md
│   └── 04_agentic_rag.md
├── errors_and_fixes.md      # Every error hit during development, and the fix
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-from-scratch.git
cd rag-from-scratch
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
```

Free keys available at:
- Groq: https://console.groq.com
- Cohere: https://dashboard.cohere.com

### 5. Run

```bash
python part_1_to_11.py
python part_12_to_17.py
```

---

## Key Learnings

- **Retrieval quality determines RAG performance more than the LLM itself** — most failures traced back to chunking, embedding choice, or retrieval strategy rather than generation.
- **Library APIs change fast.** Several imports from the original course (`langchain.prompts`, `langchain.load`, `langchain.storage`) have since moved to `langchain_core` and `langchain_classic` — every migration is documented in [`errors_and_fixes.md`](./errors_and_fixes.md).
- **Agentic RAG (LangGraph) is the most production-relevant pattern** covered here — giving the LLM control over *when* to retrieve, with built-in relevance grading and query-rewriting on failure, closely mirrors how real-world RAG systems are now being architected.
- **Free-tier APIs are sufficient for learning and prototyping** — Groq's inference speed combined with HuggingFace's local embeddings made it possible to implement every technique without spending money on OpenAI credits.

---

## What's Next

- [ ] Add evaluation metrics using RAGAS (faithfulness, answer relevance, context precision)
- [ ] Scale Multi-Document RAG with metadata filtering across 10+ documents
- [ ] Deploy as a FastAPI service with a simple Streamlit frontend
- [ ] Apply Multi-Representation Indexing and Agentic RAG patterns to [ProofLayer AI](#) — an explainable AI research workspace

---

## Acknowledgements

Based on the **"RAG From Scratch"** course by Lance Martin (LangChain), originally taught using OpenAI. This implementation independently migrates the entire pipeline to free, open-source alternatives.

---

## Connect

If you found this useful or want to discuss RAG architecture, feel free to connect on [LinkedIn](#) or open an issue.
