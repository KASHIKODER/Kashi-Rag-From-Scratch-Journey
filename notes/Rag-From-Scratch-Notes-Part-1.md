Pehle teeno notebooks dekhta hun.Bhai poora code dekh liya — teeno notebooks. Ab full breakdown deta hun, topic by topic.

---

# 📓 RAG From Scratch — Complete Code Breakdown
### Notebooks 1-4, 5-9, 10-11 (Sab Cover Kiya)

---

## 🛠️ PART 0 — Environment Setup (Sab Notebooks Mein Common)

```python
! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
```

**Kya install ho raha hai:**
| Package | Kaam |
|---------|------|
| `langchain_community` | Document loaders, vector stores ke community integrations |
| `tiktoken` | OpenAI ka tokenizer — text ko tokens mein count karta hai |
| `langchain-openai` | OpenAI models (GPT, embeddings) ko LangChain se connect karta hai |
| `langchainhub` | Pre-built prompts download karne ke liye (jaise `rlm/rag-prompt`) |
| `chromadb` | Local vector database |
| `langchain` | Core framework |

```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = <your-api-key>
os.environ['OPENAI_API_KEY'] = <your-api-key>
```

**Kya ho raha hai:**
- `LANGCHAIN_TRACING_V2` — LangSmith tracing ON kar raha hai (har step ka log milega)
- `LANGCHAIN_API_KEY` — LangSmith ka API key (free signup pe milta hai)
- `OPENAI_API_KEY` — OpenAI ka API key (paid hai, ₹ lagega)

> ⚠️ **Important:** `<your-api-key>` ko actual key se replace karna hoga — yeh placeholder hai.

---

## 📦 PART 1 — Overview / Quick Start (Notebook 1, Cell 7)

```python
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#### INDEXING ####
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/...",),
    ...
)
```

Yeh **full RAG pipeline ka preview** hai — ek hi cell mein indexing + retrieval + generation. Baaki cells mein yeh step-by-step explain hoga, isliye is cell ko detail mein nahi todenge.

---

## 🔢 PART 2 — Indexing (Notebook 1)

### Cell 9-11 — Token Counting

```python
question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string(question, "cl100k_base")
```

**Line-by-line:**
- `tiktoken.get_encoding("cl100k_base")` — yeh GPT-3.5/GPT-4 ka tokenizer hai
- `encoding.encode(string)` — text ko token IDs ki list mein convert karta hai
- `len(...)` — kitne tokens hain count karta hai

**Kyun zaruri hai?** Embedding models aur LLMs ki **token limit** hoti hai. Pehle se pata hona chahiye tera text kitna "bhara" hai.

---

### Cell 13 — Embeddings Generate Karna

```python
from langchain_openai import OpenAIEmbeddings
embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)
```

**Line-by-line:**
- `OpenAIEmbeddings()` — OpenAI ka embedding model load karta hai (default: `text-embedding-ada-002`)
- `embd.embed_query(question)` — text ko ek **vector** (numbers ki list) mein convert karta hai
- `len(query_result)` — output: **1536** (vector ki dimension)

**Matlab:** "What kinds of pets do I like?" → `[0.023, -0.041, 0.018, ...]` (1536 numbers)

---

### Cell 15 — Cosine Similarity

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)
```

**Line-by-line:**
- `np.dot(vec1, vec2)` — dono vectors ka dot product
- `np.linalg.norm(vec1)` — vector ki "length" (magnitude)
- Formula: `cosine_similarity = (A·B) / (|A| × |B|)`

**Output range:** -1 to 1, jahan **1 = identical meaning**, **0 = unrelated**

**Real-world analogy:** Socho do arrows kisi room mein khade hain. Agar dono **same direction** mein point kar rahe hain → similarity 1 ke kareeb. Agar **perpendicular** hain → similarity 0. Yahi concept hai — "What kinds of pets do I like?" aur "My favorite pet is a cat." same direction mein point karenge kyunki dono pets ke baare mein hain.

---

### Cell 17 — Document Loading

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
```

**Line-by-line:**
- `WebBaseLoader` — kisi bhi webpage ko scrape karke load karta hai
- `web_paths` — kaunsa URL load karna hai (yahan: Lilian Weng ka "AI Agents" blog post)
- `bs_kwargs` — BeautifulSoup ke parameters
- `bs4.SoupStrainer(class_=(...))` — **sirf** in HTML classes ka content nikaalo (post-content, post-title, post-header) — baaki ads/navbar/footer ignore karo
- `loader.load()` — actual scraping karta hai, `Document` objects ki list return karta hai

---

### Cell 19 — Text Splitting (Chunking)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

splits = text_splitter.split_documents(blog_docs)
```

**Line-by-line:**
- `RecursiveCharacterTextSplitter` — document ko chote chunks mein todta hai
- `from_tiktoken_encoder` — chunk size **tokens mein** measure karta hai (characters mein nahi)
- `chunk_size=300` — har chunk max 300 tokens ka hoga
- `chunk_overlap=50` — consecutive chunks ke beech 50 tokens overlap honge (context na toote isliye)

**Visual example:**
```
Original: [................................................500 tokens.....]

Chunk 1: [...300 tokens...]
Chunk 2:           [...overlap 50...300 tokens...]
Chunk 3:                                [...overlap 50...]
```

**Kyun overlap?** Agar important sentence chunk boundary pe kat jaaye, overlap usse dono chunks mein partially capture kar leta hai.

---

### Cell 21 — Vector Store Mein Index Karna

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
```

**Line-by-line:**
- `Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())` — har split ko embed karta hai aur ChromaDB (local vector database) mein store karta hai
- `vectorstore.as_retriever()` — vector store ko ek "Retriever" object mein convert karta hai — yeh object questions le sakta hai aur relevant chunks return kar sakta hai

**Internally kya ho raha hai:**
```
splits = [chunk1, chunk2, chunk3, ...]
    ↓ (each chunk embedded)
vectorstore = {
    chunk1: [0.01, 0.5, ...],
    chunk2: [0.3, -0.2, ...],
    ...
}
```

---

## 🔍 PART 3 — Retrieval (Notebook 1, Cell 23-25)

```python
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("What is Task Decomposition?")
len(docs)
```

**Line-by-line:**
- `search_kwargs={"k": 1}` — sirf **1 sabse relevant chunk** return karo
- `retriever.get_relevant_documents("...")` — question embed karta hai, vector store mein KNN search karta hai, top-k chunks return karta hai
- `len(docs)` → output: `1` (jaisa expect kiya)

**Yeh deprecated method hai** — naye LangChain versions mein use hota hai:
```python
docs = retriever.invoke("What is Task Decomposition?")
```

---

## ✍️ PART 4 — Generation (Notebook 1, Cell 27-34)

### Cell 27-28 — Prompt + LLM Define Karna

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

**Line-by-line:**
- `template` — ek string with **placeholders** `{context}` and `{question}`
- `ChatPromptTemplate.from_template(template)` — yeh template ko ek LangChain "Prompt" object banata hai jo dict accept kar sakta hai
- `ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)` — LLM define kiya
  - `temperature=0` — **deterministic output** (same input → same output har baar, randomness zero)

---

### Cell 29-30 — Chain Banake Run Karna

```python
chain = prompt | llm

chain.invoke({"context":docs,"question":"What is Task Decomposition?"})
```

**Line-by-line:**
- `prompt | llm` — yeh **LCEL (LangChain Expression Language)** syntax hai. `|` pipe operator hai — pehle prompt run hoga, fir uska output LLM ko jayega
- `chain.invoke({"context": docs, "question": "..."})` — dictionary pass karke chain run karta hai

**Flow:**
```
{"context": docs, "question": "What is Task Decomposition?"}
        ↓
   prompt.format(...) → "Answer the question based on...\n\n[doc content]\n\nQuestion: What is..."
        ↓
   llm.invoke(formatted_prompt) → AIMessage(content="Task decomposition is...")
```

---

### Cell 31-32 — LangChain Hub Se Prompt Pull Karna

```python
from langchain import hub
prompt_hub_rag = hub.pull("rlm/rag-prompt")
prompt_hub_rag
```

**Kya ho raha hai:** LangChain Hub se ek **pre-built, community-tested RAG prompt** download kar raha hai. Apna prompt likhne ki zarurat nahi — already optimized prompt mil jaata hai.

---

### Cell 34 — Full Automated RAG Chain ⭐ (Most Important)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")
```

**Yeh sabse important pattern hai — poori course mein baar baar aayega.**

**Line-by-line:**
- `{"context": retriever, "question": RunnablePassthrough()}` — yeh ek **dictionary chain** hai:
  - `"context": retriever` → jo bhi input aaya (question), woh retriever ko jayega, retriever docs return karega
  - `"question": RunnablePassthrough()` → input jaisa hai waisa hi aage pass ho jayega (no modification)
- `| prompt` → dono values (`context` aur `question`) prompt template mein fill ho jaate hain
- `| llm` → filled prompt LLM ko jaata hai
- `| StrOutputParser()` → LLM ka output (AIMessage object) ko plain **string** mein convert karta hai

**Visual Flow:**
```
Input: "What is Task Decomposition?"
        │
        ├──→ retriever("What is...") ──→ [doc1, doc2, doc3]  ──┐
        │                                                       ├──→ context, question
        └──→ passthrough("What is...") ──→ "What is..."  ───────┘
                                                    ↓
                                              prompt.format(context=docs, question="What is...")
                                                    ↓
                                              llm.invoke(formatted_prompt)
                                                    ↓
                                              StrOutputParser() → "Task decomposition is the process of..."
```

Yeh **pura RAG pipeline ek single line mein** — bina manually retriever call kiye!

---

## 🔄 PART 5 — Multi-Query (Notebook 2)

### Cell 9 — Multi-Query Prompt

```python
from langchain.prompts import ChatPromptTemplate

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Line-by-line:**
- Prompt LLM ko bolta hai: "1 question lo, 5 different versions banao"
- `| (lambda x: x.split("\n"))` — yeh **custom function step** hai. LLM output ek string hai jaisa:
  ```
  "1. What is...\n2. How does...\n3. Explain..."
  ```
  `.split("\n")` isko Python **list** mein todta hai: `["1. What is...", "2. How does...", "3. Explain..."]`

---

### Cell 10 — Retrieval Per Question + Unique Union

```python
from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
```

**Line-by-line:**
- `dumps(doc)` — har Document object ko **JSON string** mein convert karta hai (kyunki Document objects directly compare/hash nahi ho sakte, par strings ho sakti hain)
- `flattened_docs = [... for sublist in documents for doc in sublist]` — yeh **nested list flatten** kar raha hai:
  ```python
  documents = [[doc1, doc2], [doc2, doc3], [doc1, doc4]]
  # flatten karne ke baad:
  flattened_docs = [doc1_str, doc2_str, doc2_str, doc3_str, doc1_str, doc4_str]
  ```
- `set(flattened_docs)` — duplicates remove karta hai (Python set mein automatically unique values rehti hain)
- `loads(doc)` — JSON string ko wapas Document object mein convert karta hai

- `retriever.map()` — yeh **important hai**. `.map()` ka matlab: agar input ek **list of questions** hai, toh retriever har question pe **independently** chalega, aur result hoga **list of lists**:
  ```
  Input: ["Q1", "Q2", "Q3", "Q4", "Q5"]
  retriever.map() → [[docs_for_Q1], [docs_for_Q2], [docs_for_Q3], [docs_for_Q4], [docs_for_Q5]]
  ```

**Full Flow:**
```
question
   ↓
generate_queries → [Q1, Q2, Q3, Q4, Q5]  (5 rephrased questions)
   ↓
retriever.map() → [[docs_Q1], [docs_Q2], [docs_Q3], [docs_Q4], [docs_Q5]]
   ↓
get_unique_union() → [unique_doc1, unique_doc2, ...]  (flattened + deduplicated)
```

---

### Cell 11 — Final RAG Chain (Multi-Query)

```python
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})
```

**Naya concept — `itemgetter`:**
- `itemgetter("question")` — input dictionary se sirf `"question"` key ka value nikaalta hai
- Pehle wale chain mein `RunnablePassthrough()` use hua tha (jab input sirf string tha)
- Yahan input ek **dictionary** hai `{"question": "..."}`, isliye `itemgetter("question")` use karte hain specific key nikalne ke liye

**Difference samajh:**
```python
# Method 1 (input is plain string):
RunnablePassthrough()  # passes whole input as-is

# Method 2 (input is dict):
itemgetter("question")  # extracts just the "question" key's value
```

---

## 🔀 PART 6 — RAG Fusion (Notebook 2)

### Cell 13-14 — Query Generation

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

Same pattern jaisa Multi-Query — bas 4 queries banata hai instead of 5, aur prompt thoda different hai.

---

### Cell 15 — Reciprocal Rank Fusion (RRF) ⭐ Naya Concept

```python
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results
```

**Line-by-line:**
- `fused_scores = {}` — har unique document ka final score store karega
- `for docs in results:` — har query ke retrieval result pe loop
- `for rank, doc in enumerate(docs):` — har document ki **rank/position** nikaalta hai (0, 1, 2, 3...)
- `fused_scores[doc_str] += 1 / (rank + k)` — **RRF formula**: jitna upar rank hai (chota number), utna zyada score milega

**RRF Formula Samajh:**
```
score = 1 / (rank + k)

rank=0 (top result), k=60 → 1/60 = 0.0167
rank=5,              k=60 → 1/65 = 0.0154
rank=10,             k=60 → 1/70 = 0.0143
```

**Kyun yeh useful hai?** Agar ek document **multiple queries mein top pe** aata hai, toh uska total score badhta jayega — matlab woh **consistently relevant** hai across different phrasings.

**Example:**
```
Query 1 results: [docA(rank0), docB(rank1), docC(rank2)]
Query 2 results: [docB(rank0), docA(rank1), docD(rank2)]

docA score = 1/(0+60) + 1/(1+60) = 0.0167 + 0.0164 = 0.0331
docB score = 1/(1+60) + 1/(0+60) = 0.0164 + 0.0167 = 0.0331
docC score = 1/(2+60) = 0.0161
docD score = 1/(2+60) = 0.0161

Final ranking: docA/docB tied at top, then docC/docD
```

- `sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)` — scores ke basis pe descending order mein sort karta hai

---

## 🧩 PART 7 — Decomposition (Notebook 2)

### Cell 19-21 — Sub-Questions Generate Karna

```python
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)
generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question":question})
```

Pehle jaisa pattern — bas prompt ka focus **breaking down into smaller pieces** hai, "different perspectives" nahi.

---

### Cell 23-25 — Sequential Answering (Q&A Pairs Build Karna) ⭐ Naya Pattern

```python
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)
```

**Yeh prompt mein 3 placeholders hain:**
- `{question}` — current sub-question
- `{q_a_pairs}` — pehle ke sub-questions ke answers
- `{context}` — current question ke liye retrieved docs

```python
def format_qa_pair(question, answer):
    """Format Q and A pair"""
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

q_a_pairs = ""
for q in questions:
    rag_chain = (
    {"context": itemgetter("question") | retriever, 
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())

    answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
```

**Yeh sabse important loop hai — line by line samajh:**

1. `q_a_pairs = ""` — empty string se start (koi prior answers nahi hain abhi)
2. `for q in questions:` — har sub-question pe loop chalta hai (3 questions: Q1, Q2, Q3)
3. **First iteration (Q1):**
   - `rag_chain.invoke({"question": Q1, "q_a_pairs": ""})` — kyunki q_a_pairs khaali hai, sirf Q1 answer hoga retrieved context se
   - Answer milta hai → `format_qa_pair(Q1, answer1)` → `"Question: Q1\nAnswer: answer1"`
   - `q_a_pairs` ab update ho jaata hai: `"\n---\nQuestion: Q1\nAnswer: answer1"`
4. **Second iteration (Q2):**
   - `rag_chain.invoke({"question": Q2, "q_a_pairs": q_a_pairs})` — ab `q_a_pairs` mein Q1's answer hai!
   - Matlab Q2 answer karte time, LLM ko Q1 ka context bhi milta hai
   - Naya answer milta hai, `q_a_pairs` mein **append** ho jaata hai
5. **Third iteration (Q3):** same — ab Q1 + Q2 dono ka context milta hai

**Yeh "building up" pattern hai — jaisa hum manually notes mein samjha tha.**

---

### Cell 28-29 — Parallel Answering (Independent)

```python
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt_rag = hub.pull("rlm/rag-prompt")

def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """RAG on each sub-question"""
    sub_questions = sub_question_generator_chain.invoke({"question": question})
    
    rag_results = []
    for sub_question in sub_questions:
        retrieved_docs = retriever.get_relevant_documents(sub_question)
        answer = (prompt_rag | llm | StrOutputParser()).invoke(
            {"context": retrieved_docs, "question": sub_question}
        )
        rag_results.append(answer)
    
    return rag_results, sub_questions
```

**Yeh Sequential se kaise different hai:**
- Yahan har sub-question **independently** answer hota hai — koi `q_a_pairs` nahi pass ho rahi
- Sirf apna khud ka retrieved context use karta hai

```python
def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

context = format_qa_pairs(questions, answers)

template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)
```

**Final step:** Saare independent Q&A pairs ko ek saath combine karke ek **synthesis prompt** banata hai jo final answer deta hai.

**Sequential vs Parallel — Key Difference:**

| Sequential | Parallel |
|-----------|----------|
| Q2 ko Q1 ka answer pata hai | Har question independent hai |
| Slower (one-by-one) | Faster (sab ek saath chal sakte hain) |
| Use when: questions related hain ek dusre se | Use when: questions independent hain |

---

## 🔙 PART 8 — Step-Back Prompting (Notebook 2)

### Cell 32 — Few-Shot Examples Setup

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

**Line-by-line:**
- `examples` — yeh do **demonstration pairs** hain — "yeh specific question diya, aapko isse abstract karna hai"
- `example_prompt` — template define karta hai ki har example kaise format hoga (human asks, AI responds)
- `FewShotChatMessagePromptTemplate` — yeh LangChain ka special class hai jo automatically **multiple examples ko ek prompt mein chain** kar deta hai

**Concept:** Few-shot prompting matlab — LLM ko 2-3 examples dikhao "yeh karna hai" se pehle real question poochna. Isse LLM pattern samajh jaata hai bina explicit instructions ke.

```python
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "What is task decomposition for LLM agents?"
generate_queries_step_back.invoke({"question": question})
```

**Output expect karo:** `"what is the process of task decomposition?"` (zyada abstract/general version)

---

### Cell 34 — Combining Both Contexts

```python
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        "step_back_context": generate_queries_step_back | retriever,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)
```

**Line-by-line:**
- `"normal_context": RunnableLambda(lambda x: x["question"]) | retriever` — original question se retrieval
- `"step_back_context": generate_queries_step_back | retriever` — pehle step-back question generate hota hai, fir uska retrieval
- Dono contexts prompt mein fill ho jaate hain — LLM ko **dono levels ka info** milta hai (specific + general)

**`RunnableLambda` kya hai?** Yeh ek custom Python function (lambda) ko LangChain chain ke andar use karne deta hai. `lambda x: x["question"]` matlab input dictionary se "question" key nikaalo.

---

## 🌫️ PART 9 — HyDE (Notebook 2)

### Cell 36-38 — Hypothetical Document Generate Karna

```python
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
)

question = "What is task decomposition for LLM agents?"
generate_docs_for_retrieval.invoke({"question":question})
```

**Kya ho raha hai:** LLM ko bolte hain — "ek hypothetical academic passage likho jo iss question ka answer de" (bina actual data dekhe, sirf apni training knowledge se).

```python
retrieval_chain = generate_docs_for_retrieval | retriever 
retrieved_docs = retrieval_chain.invoke({"question":question})
```

**Yeh interesting hai:** `generate_docs_for_retrieval` output ek **string** (hypothetical passage) hai. Yeh string seedha `retriever` ko pipe ho jaati hai — retriever isi hypothetical passage ko embed karke vector search karta hai (real question ko nahi!).

```python
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retrieved_docs,"question":question})
```

Final step — retrieved (real) docs + original question → normal RAG generation.

---

## 🧭 PART 10 — Routing (Notebook 3)

### Cell 7 — Logical Routing Setup

```python
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)
```

**Line-by-line — yeh naya aur important concept hai:**
- `class RouteQuery(BaseModel):` — yeh **Pydantic model** hai. Pydantic Python ki library hai jo data validation karta hai
- `datasource: Literal["python_docs", "js_docs", "golang_docs"]` — yeh field sirf **in 3 exact values** mein se ek le sakta hai. Koi aur value diya toh error aayega
- `Field(..., description="...")` — `...` matlab yeh field **required** hai (optional nahi), aur description LLM ko batati hai field ka matlab kya hai
- `llm.with_structured_output(RouteQuery)` — yeh **LLM ko force karta hai** ki output sirf RouteQuery schema follow kare — koi random text nahi, sirf valid `datasource` value

**Pydantic kyun use karte hain?** Bina isse, LLM kuch bhi text return kar sakta hai jaise *"I think this should go to python documentation"* — jisse parse karna mushkil hai. Pydantic se output hamesha **clean, predictable structure** mein aata hai:
```python
RouteQuery(datasource="python_docs")
```

```python
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

router = prompt | structured_llm
```

---

### Cell 9-14 — Router Test + Conditional Routing

```python
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

result = router.invoke({"question": question})
result.datasource
```

**Output:** `"python_docs"` (kyunki code Python syntax hai)

```python
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        return "chain for js_docs"
    else:
        return "golang_docs"

from langchain_core.runnables import RunnableLambda

full_chain = router | RunnableLambda(choose_route)
full_chain.invoke({"question": question})
```

**Line-by-line:**
- `choose_route(result)` — yeh function `RouteQuery` object lekar check karta hai `.datasource` field ki value kya hai, aur uske basis pe ek **string** return karta hai (kis chain ko aage call karna hai)
- `router | RunnableLambda(choose_route)` — pehle router chalega (LLM decide karega datasource), fir `choose_route` function decide karega actual routing logic

**Real implementation mein:** `return "chain for python_docs"` ki jagah actual retriever/chain call hota hai, jaise `return python_docs_retriever.invoke(question)`.

---

### Cell 17 — Semantic Routing

```python
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you...
{query}"""

embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    return PromptTemplate.from_template(most_similar)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)
```

**Line-by-line:**
- `embeddings.embed_documents(prompt_templates)` — dono prompts (physics, math) ko embed karta hai — list of vectors
- `prompt_router(input)` function:
  - User ka query embed karta hai
  - `cosine_similarity([query_embedding], prompt_embeddings)` — query ki similarity dono prompts se compute karta hai
  - `similarity.argmax()` — sabse zyada similarity wala **index** nikaalta hai (0 ya 1)
  - `prompt_templates[similarity.argmax()]` — woh prompt select karta hai
- Yeh function `RunnableLambda` mein wrap hota hai taaki LangChain chain mein use ho sake

**Semantic Routing vs Logical Routing:**

| Logical Routing | Semantic Routing |
|-----------------|-------------------|
| LLM function-calling se decide karta hai | Embedding similarity se decide hota hai |
| Costlier (LLM call lagti hai) | Cheaper (sirf embedding comparison) |
| Better for complex categorization | Better for simple topic matching |

---

## 🏗️ PART 11 — Query Construction (Notebook 3)

### Cell 21 — Metadata Dekhna

```python
from langchain_community.document_loaders import YoutubeLoader

docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
).load()

docs[0].metadata
```

**Kya ho raha hai:** Ek YouTube video load karta hai aur uski metadata dekhta hai — jaise `title`, `publish_date`, `view_count`, `length`, etc.

---

### Cell 23 — Schema Define Karna

```python
import datetime
from typing import Literal, Optional, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, inclusive.",
    )
```

**Line-by-line:**
- `content_search: str` — yeh **required** field hai (no default), semantic search ke liye
- `title_search: str` — bhi required, title pe semantic search ke liye
- `min_view_count: Optional[int] = None` — yeh **optional** field hai. Agar user ne view count ka zikr nahi kiya, toh `None` rahega
- `Optional[datetime.date]` — date type field, optional

**Yeh schema basically batata hai LLM ko:** "Yeh saari information fields hain jo tum extract kar sakte ho user ke natural language question se."

---

### Cell 25 — Query Analyzer Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm
```

Same pattern jaisa Routing mein dekha — `with_structured_output()` LLM ko force karta hai ki TutorialSearch schema follow kare.

---

### Cell 26-29 — Testing

```python
query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()
```
**Output (approx):**
```
content_search: "rag from scratch"
title_search: "rag from scratch"
```

```python
query_analyzer.invoke(
    {"question": "videos on chat langchain published in 2023"}
).pretty_print()
```
**Output (approx):**
```
content_search: "chat langchain"
title_search: "chat langchain"
earliest_publish_date: 2023-01-01
latest_publish_date: 2023-12-31
```

**Yeh dikhata hai:** Same natural language question se, LLM automatically samajh jaata hai kaunse fields fill karne hain — semantic search wale fields **always** fill hote hain, par date/view-count/length wale fields **sirf jab user ne unka zikr kiya ho**.

---

# 💻 PART 12 — Apne System Mein Kaise Run Karna Hai

## Step 1 — Python Install Karo (Agar Nahi Hai)

```bash
python --version
```
Agar nahi hai, [python.org](https://python.org) se download karo (3.10+ recommended).

---

## Step 2 — Virtual Environment Banao (Recommended)

```bash
python -m venv rag_env

# Windows:
rag_env\Scripts\activate

# Mac/Linux:
source rag_env/bin/activate
```

**Kyun zaruri hai?** Isse tere system ka Python clean rehta hai — saare packages isolated environment mein install hote hain.

---

## Step 3 — Jupyter Notebook Install Karo

```bash
pip install jupyter notebook
```

---

## Step 4 — Required Packages Install Karo

```bash
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain youtube-transcript-api pytube
```

---

## Step 5 — API Keys Lo

### OpenAI API Key (Paid — Required)
1. Jao: https://platform.openai.com/api-keys
2. Account banao, **billing add karo** (minimum $5 credit)
3. "Create new secret key" pe click karo
4. Key copy kar lo (sirf ek baar dikhti hai!)

### LangSmith API Key (Free — Optional but recommended)
1. Jao: https://smith.langchain.com
2. Free signup karo
3. Settings → API Keys → Create API Key

---

## Step 6 — Notebook Mein Keys Daalo

```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'ls__xxxxxxxxxxxxxxxxxxxx'  # tera actual key
os.environ['OPENAI_API_KEY'] = 'sk-xxxxxxxxxxxxxxxxxxxx'      # tera actual key
```


```


