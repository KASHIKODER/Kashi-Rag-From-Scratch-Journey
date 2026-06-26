
# 📓 RAG From Scratch — Code Breakdown (Part -2) 12 to 17

---

## 🗂️ PART 12 — Multi-Representation Indexing

### Cell 7 — Do Documents Load Karna

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())
```

**Line-by-line:**
- Pehla `WebBaseLoader` — Agents wala blog post load karta hai
- `docs.extend(...)` — **dusra document add karta hai same list mein** (extend = list ke andar items append karna, replace nahi)
- Ab `docs` mein **2 alag documents** hain (Agents + Human Data Quality)

**Yeh pehle se different hai:** Pehle (Part 1-11) hum **chunks** banate the splitting karke. Yahan hum **poore documents** rakh rahe hain — chunking nahi ho rahi shuru mein.

---

### Cell 8 — Har Document Ka Summary Banana ⭐ Core Idea

```python
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-3.5-turbo", max_retries=0)
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})
```

**Line-by-line:**
- `{"doc": lambda x: x.page_content}` — yeh **lambda function** har Document object se sirf uska text content (`page_content`) nikaalta hai
- `ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")` — yeh simple prompt LLM ko bolta hai summary banane ko
- `chain.batch(docs, {"max_concurrency": 5})` — **naya concept yahan hai:**
  - `.batch()` — LangChain ka method jo **multiple inputs ek saath process** karta hai (yahan: 2 documents)
  - `{"max_concurrency": 5}` — max **5 requests parallel** mein chal sakti hain (agar zyada documents hote)

**Result:** `summaries` ek list hai — `[summary_of_agents_doc, summary_of_human_data_doc]`

---

### Cell 9 — Multi-Vector Retriever Setup ⭐⭐ Sabse Important Concept

```python
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Vectorstore — sirf summaries store karega
vectorstore = Chroma(collection_name="summaries",
                     embedding_function=OpenAIEmbeddings())

# Doc Store — poore raw documents store karega
store = InMemoryByteStore()
id_key = "doc_id"

# Retriever — dono ko jodta hai
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Summaries ko doc_id ke saath label karna
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
```

**Line-by-line — yeh sabse important part hai:**

1. `vectorstore = Chroma(...)` — yeh sirf **summaries** ko embed karke store karega (semantic search ke liye)
2. `store = InMemoryByteStore()` — yeh ek simple **key-value store** hai jo **poore raw documents** rakhega (RAM mein, temporary)
3. `id_key = "doc_id"` — yeh wo **link/reference field** hai jo dono stores ko connect karta hai
4. `doc_ids = [str(uuid.uuid4()) for _ in docs]` — har document ke liye ek **unique random ID** generate kiya (`uuid4()` — universally unique identifier)
5. `summary_docs = [...]` — har summary ko ek `Document` object mein wrap kiya, **metadata mein doc_id attach kiya** — taaki pata chale "yeh summary kis original document ka hai"
6. `retriever.vectorstore.add_documents(summary_docs)` — summaries ko vector store mein daala (embedding ke saath)
7. `retriever.docstore.mset(list(zip(doc_ids, docs)))` — **`mset` matlab "multiple set"** — yeh ek dictionary jaisa mapping banata hai: `{doc_id_1: full_document_1, doc_id_2: full_document_2}`

**Visual Samajh:**
```
VECTOR STORE (summaries, searchable):          DOC STORE (full docs, by ID):
┌─────────────────────────────┐                ┌──────────────────────────┐
│ doc_id: "abc-123"           │   ──links──→    │ "abc-123": [FULL Agent   │
│ summary: "This post about   │                │  blog post, 5000 words]  │
│  AI agents discusses..."    │                │                          │
├─────────────────────────────┤                ├──────────────────────────┤
│ doc_id: "xyz-789"           │   ──links──→    │ "xyz-789": [FULL Human   │
│ summary: "This post covers  │                │  Data post, 3000 words]  │
│  human data quality..."     │                │                          │
└─────────────────────────────┘                └──────────────────────────┘
```

**Kyun yeh powerful hai?** Search **chote, optimized summaries** pe hoti hai (fast, accurate), par jo LLM ko milta hai woh **poora original document** hai (complete context, no information loss from chunking).

---

### Cell 10 — Vector Store Pe Direct Search (Sirf Summary)

```python
query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query, k=1)
sub_docs[0]
```

**Kya ho raha hai:** Yeh **sirf vector store** pe search kar raha hai (summary store), retriever object use nahi kar raha. Result milega — **summary** wala chota Document, poora document nahi.

---

### Cell 11 — Retriever Se Search (Poora Document)

```python
retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
retrieved_docs[0].page_content[0:500]
```

**Yahan farak dikhega:** Ab hum `retriever` use kar rahe hain (na ki sirf `vectorstore`). Internally yeh:
1. Query ko embed karta hai
2. **Summaries** mein similarity search karta hai
3. Best matching summary ka `doc_id` nikaalta hai
4. Us `doc_id` se **doc store** mein lookup karta hai
5. **Poora original document** return karta hai

`page_content[0:500]` — sirf first 500 characters print karega (poora document bahut bada hai dikhane ke liye).

**Yahi hai trick:** Summary se search, full document return.

---

## 🌲 PART 13 — RAPTOR

```markdown
## Part 13: RAPTOR

Deep dive video: https://www.youtube.com/watch?v=jbGchdTL7d0
Paper: https://arxiv.org/pdf/2401.18059.pdf
Full code: https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
```

**⚠️ Important baat:** Is notebook mein **RAPTOR ka actual code nahi hai** — sirf links hain. Lance ne yeh implementation **alag se ek dedicated cookbook notebook** mein rakha hai (uska link upar diya hai), kyunki RAPTOR ka code bahut lamba hai (clustering algorithms, recursive tree building, etc.) jo woh ek standalone notebook deserve karta hai.

**Agar tujhe RAPTOR actually code karna hai**, woh GitHub link (`cookbook/RAPTOR.ipynb`) open karna padega — main usko alag se breakdown kar sakta hun agar tu woh notebook upload kare.

---

## 🐻 PART 14 — ColBERT

### Cell 15-16 — Setup

```python
! pip install -U ragatouille

from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
```

**Line-by-line:**
- `ragatouille` — yeh library hai jo ColBERT ko **easy-to-use** banati hai (raw ColBERT implement karna complex hai)
- `RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")` — pretrained ColBERT v2.0 model download/load karta hai (yeh HuggingFace se aata hai)

---

### Cell 17 — Wikipedia Se Data Lena

```python
import requests

def get_wikipedia_page(title: str):
    """Retrieve the full text content of a Wikipedia page."""
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"]
```

**Line-by-line:**
- `requests.get(URL, params=params, headers=headers)` — Wikipedia ke **official API** ko HTTP request bhejta hai (yeh standard Python web-scraping nahi, balki Wikipedia ka apna API hai)
- `params` — query parameters: konsa page chahiye, format JSON mein, plain text extract chahiye
- `headers = {"User-Agent": ...}` — Wikipedia request karne walon ko identify karna chahta hai (best practice)
- `next(iter(data["query"]["pages"].values()))` — yeh JSON response ke complex nested structure se **page data nikaalta hai** (Wikipedia API thoda awkward JSON structure deta hai)

---

### Cell 18 — Indexing

```python
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)
```

**Line-by-line:**
- `collection=[full_document]` — Miyazaki ka poora Wikipedia page (text string)
- `max_document_length=180` — har chunk max **180 tokens** ka hoga (ColBERT internally khud split karta hai)
- `split_documents=True` — automatically document ko chunks mein todega before indexing

**Internally kya ho raha hai:** ColBERT document ko tokens mein todta hai, **har token ka apna vector** banata hai (jaisa humne notes mein discuss kiya tha — single vector nahi, token-level vectors).

---

### Cell 19-20 — Search Karna

```python
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
results
```

**Yeh direct RAGatouille API use karta hai** — `k=3` top 3 matches return karega.

```python
retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")
```

**Yahan dikhata hai:** `as_langchain_retriever()` — RAGatouille ke model ko **LangChain-compatible retriever** mein convert kar deta hai. Iska fayda — ab tu isse normal LangChain chains mein use kar sakta hai (jaisa humne Part 1-11 mein retriever use kiya tha).

---

## 🏆 PART 15 — Re-Ranking (Notebook 15-18)

### Cell 8 — Standard Indexing (Same Jaisa Pehle)

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(blog_docs)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
```

Yeh wahi standard indexing hai jo Part 1 mein dekhi thi — naya kuch nahi.

---

### Cell 9-12 — RAG-Fusion Wapas Use Karna

```python
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
```

**Yeh exactly wahi RAG-Fusion code hai jo Part 6 mein tha.** Lance yahan dikhata hai ki **Re-ranking RAG-Fusion ka extension hai** — RRF (Reciprocal Rank Fusion) khud ek **basic re-ranking technique** hai jo humne already implement kiya tha.

```python
def reciprocal_rank_fusion(results: list[list], k=60):
    # ... same code as Part 6
```

**Concept Recap:** RRF documents ko unki **rank position** ke basis pe re-score karta hai across multiple query results — yeh already ek re-ranking method hai.

---

### Cell 14-15 — Cohere Re-Rank ⭐ Naya Concept

```python
from langchain_community.llms import Cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Re-rank
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(question)
```

**Line-by-line — yeh naya pattern hai:**

- `retriever = vectorstore.as_retriever(search_kwargs={"k": 10})` — pehle **10 documents** fetch karo (broad net daalo, zyada candidates)
- `compressor = CohereRerank()` — yeh **Cohere ka specialized re-ranking model** hai. Yeh ek dusra ML model hai jo **specifically trained hai relevance scoring ke liye** (embedding similarity se zyada accurate)
- `ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)` — yeh **wrapper pattern** hai:
  1. Pehle `base_retriever` chalega (10 docs fetch karega)
  2. Phir `compressor` (Cohere Rerank) un 10 docs ko **dobara score** karega relevance ke hisaab se
  3. Sirf **best, sabse relevant** docs return honge (typically top 3-5)

**Yeh kyun better hai simple KNN search se?**

| Simple Vector Search | Cohere Re-Rank |
|----------------------|----------------|
| Embedding similarity (fast, approximate) | Specialized cross-attention model (slower, more accurate) |
| Sirf "kitna close hai vector space mein" dekhta hai | Actually **question aur document ko saath mein padhta hai** relevance judge karne ke liye |
| Single-shot retrieval | Two-stage: broad retrieval → precise re-ranking |

**Real-world analogy:** Vector search jaisa hai **library mein topic ke hisaab se kitabein dhundhna** (broad match). Re-ranking jaisa hai **ek expert librarian se poochna** "in 10 kitabon mein se kaunsi sabse zyada relevant hai meri specific query ke liye" (precise match).

---

## 🔄 PART 16 — CRAG (Sirf Reference, Code Nahi)

```markdown
## 16 - Retrieval (CRAG)

Deep Dive: https://www.youtube.com/watch?v=E2shqsYwxck
Notebooks:
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_mistral.ipynb
```

**⚠️ Honest baat:** Is notebook mein CRAG ka **actual code bilkul nahi hai** — sirf 3 lines ka reference hai jo bahar ke LangGraph repository ki taraf point karta hai.

**Yeh wahi CRAG hai** jo maine tujhe **pehle ke notes mein already detail se explain kiya tha** (Video 15 — Active RAG with LangGraph) — woh tab maine transcript se samjhaya tha. Agar tujhe **actual CRAG code** chahiye, woh upar diye GitHub links se alag notebook download karna padega.

---

## 🔁 PART 17 — Self-RAG (Sirf Reference, Code Nahi)

```markdown
## 17 - Retrieval (Self-RAG)

Notebooks:
https://github.com/langchain-ai/langgraph/tree/main/examples/rag
https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_mistral_nomic.ipynb
```

**Same situation:** Sirf links hain, code nahi. Self-RAG **CRAG ka cousin concept** hai — dono "active/adaptive RAG" family mein aate hain jo maine pehle explain kiya tha. Self-RAG specifically **hallucination grading aur answer-quality grading** pe zyada focus karta hai (jo humne Adaptive RAG ke notes mein dekha tha).

---

## 📏 PART 18 — Impact of Long Context (Sirf Reference, Code Nahi)

```markdown
## 18 - Impact of long context

Deep dive: https://www.youtube.com/watch?v=SsHUNfhF32s
Slides: https://docs.google.com/presentation/...
```

