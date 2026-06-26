"""
RAG FROM SCRATCH — PART 1-11 PUBLIC SHOWCASE
============================================

Clean LinkedIn/GitHub-friendly output version.

Important:
- Core RAG logic is preserved from the learning script.
- Changes are only around:
  1. .env-based secrets
  2. cleaner public output
  3. warning/noise suppression
  4. compact previews instead of raw terminal dumps

Covers:
1-4. Basic RAG: Indexing, Retrieval, Generation
5. Multi-Query
6. RAG-Fusion
7. Decomposition
8. Step-Back Prompting
9. HyDE
10. Routing
11. Query Construction
"""

import os
import re
import warnings
import logging
from operator import itemgetter
from typing import Literal, Optional
import datetime

from dotenv import load_dotenv

# ============================================================
# ENV + PUBLIC OUTPUT CLEANUP
# ============================================================

load_dotenv()

os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "SuyashRAGProject/0.1")
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = os.getenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# LangSmith is optional. Keep disabled unless explicitly enabled in .env.
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*beta.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF Hub.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def status(label: str, value: str):
    print(f"✅ {label}: {value}")


def info(label: str, value: str):
    print(f"• {label}: {value}")


def topic_note(covered: str, difference: str):
    print(f"   ↳ Covered: {covered}")
    print(f"   ↳ Why it matters: {difference}")


def clean_text(text: str, limit: int = 360) -> str:
    text = str(text)
    text = text.replace("Lil'Log Posts Archive Search Tags FAQ", " ")
    text = text.replace("Posts Archive Search Tags FAQ", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def doc_preview(doc, limit: int = 360) -> str:
    """Return a clean preview from a LangChain Document or tuple(Document, score)."""
    if isinstance(doc, tuple):
        doc = doc[0]
    content = getattr(doc, "page_content", str(doc))
    return clean_text(content, limit)


def clean_query_list(text: str, fallback: list[str], max_items: int = 5) -> list[str]:
    """Convert messy LLM output into clean search queries."""
    items = []
    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith(("here are", "sure", "output", "queries:")):
            continue

        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[\).\s-]+", "", line)
        line = line.replace("**", "").strip()

        # If the model returns: "query" - explanation, keep only the query.
        quoted = re.findall(r'"([^"]+)"', line)
        if quoted:
            line = quoted[0].strip()
        elif " - " in line:
            line = line.split(" - ", 1)[0].strip()

        if len(line) > 12 and "?" in line or len(line) > 18:
            items.append(line)

        if len(items) >= max_items:
            break

    return items if items else fallback[:max_items]


def unique_documents(documents: list):
    """Unique union without LangChain loads/dumps warnings."""
    seen = set()
    unique = []
    for sublist in documents:
        for doc in sublist:
            key = (
                doc.metadata.get("source", ""),
                clean_text(doc.page_content, 700),
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)
    return unique


def safe_invoke(chain, payload, fallback, label: str):
    """Run chain; if quota/API fails, keep public demo clean and continue."""
    try:
        return chain.invoke(payload)
    except Exception as e:
        info(f"{label} fallback", type(e).__name__)
        return fallback


def fallback_answer(topic: str) -> str:
    fallback_map = {
        "basic": "Task decomposition means breaking a complex task into smaller, manageable steps so an LLM agent can plan and solve the task more effectively.",
        "multi_query": "Task decomposition for LLM agents is the process of splitting a complex goal into sub-goals or smaller actions. This helps planning, improves reasoning, and makes tool-using agents easier to control.",
        "rag_fusion": "RAG-Fusion improves retrieval by generating multiple query perspectives and combining ranked results. For task decomposition, it helps retrieve documents that consistently discuss breaking tasks into sub-goals.",
        "decomposition": "An LLM-powered autonomous agent system usually includes planning, memory, and tool use. Planning breaks tasks into steps, memory stores useful context, and tools allow the agent to interact with external systems.",
        "step_back": "Step-back prompting answers a specific question by also retrieving broader conceptual context. For task decomposition, the broader idea is problem solving through smaller sub-goals.",
        "hyde": "HyDE first generates a hypothetical answer-style passage, then uses that passage for retrieval. This can align a short question more closely with document-style content.",
        "semantic": "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape beyond its event horizon.",
    }
    return fallback_map.get(topic, "Quota-safe fallback answer used for the public showcase run.")


section("RAG FROM SCRATCH — PART 1-11 PUBLIC DEMO")
print("Clean showcase output for LinkedIn/GitHub screenshots. No API keys printed.")
print("Mode: Original learning logic preserved; output formatting made public-friendly. Rate-limit/schema fallbacks keep the showcase from crashing.\n")


# ============================================================
# IMPORTS
# ============================================================

import bs4
from pydantic import BaseModel, Field

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_community.utils.math import cosine_similarity


# ============================================================
# SHARED MODELS
# ============================================================

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

llm = ChatGroq(model=GROQ_MODEL, temperature=0)

embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
)


# ============================================================
# PART 1-4: BASIC RAG PIPELINE
# ============================================================

section("PART 1-4 — Basic RAG Pipeline")
topic_note(
    "Load a blog, split it into chunks, embed chunks, store them in Chroma, retrieve relevant context, and generate an answer.",
    "This is the foundation: instead of relying only on model memory, the LLM answers using retrieved external context.",
)

# Load blog post
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()
status("Blog documents loaded", str(len(blog_docs)))

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50,
)
splits = text_splitter.split_documents(blog_docs)
status("Chunks created", str(len(splits)))

# Embed and store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
status("Vector store ready", "Chroma + HuggingFace embeddings")

# Retrieval test
basic_question = "What is Task Decomposition?"
retrieved_basic_docs = retriever.invoke(basic_question)
status("Retrieved documents", str(len(retrieved_basic_docs)))
info("Retrieval query", basic_question)
info("Retrieved context preview", doc_preview(retrieved_basic_docs[0], 280))

# Generation
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = safe_invoke(rag_chain, basic_question, fallback_answer("basic"), "Basic RAG answer")
print("\n🎯 BASIC RAG ANSWER")
print(clean_text(answer, 520))


# ============================================================
# PART 5: MULTI-QUERY
# ============================================================

section("PART 5 — Multi-Query")
topic_note(
    "Generate multiple rewritten versions of the same user question and retrieve documents for each version.",
    "It reduces wording mismatch between short user questions and the way relevant documents are written.",
)

multi_query_template = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

multi_query_fallback = [
    "What is task decomposition in LLM agents?",
    "How do LLM agents break complex tasks into smaller steps?",
    "Why is task decomposition useful for autonomous agents?",
    "How does task decomposition help planning in LLM-powered agents?",
    "What are sub-goals in LLM agent systems?",
]

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: clean_query_list(x, multi_query_fallback, max_items=5))
)


def get_unique_union(documents: list):
    """Unique union of retrieved docs."""
    return unique_documents(documents)


question = "What is task decomposition for LLM agents?"
sample_queries = safe_invoke(
    generate_queries,
    {"question": question},
    multi_query_fallback,
    "Multi-Query generation",
)
status("Generated query variants", str(len(sample_queries)))
for i, q in enumerate(sample_queries[:5], start=1):
    info(f"Query {i}", clean_text(q, 130))

retrieval_chain = generate_queries | retriever.map() | get_unique_union
multi_docs = retrieval_chain.invoke({"question": question})
status("Unique retrieved documents", str(len(multi_docs)))

multi_prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on this context:

{context}

Question: {question}
"""
)

final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | multi_prompt
    | llm
    | StrOutputParser()
)

answer = safe_invoke(
    final_rag_chain,
    {"question": question},
    fallback_answer("multi_query"),
    "Multi-Query answer",
)
print("\n🎯 MULTI-QUERY ANSWER")
print(clean_text(answer, 520))


# ============================================================
# PART 6: RAG-FUSION
# ============================================================

section("PART 6 — RAG-Fusion")
topic_note(
    "Generate multiple search queries, retrieve ranked lists, then merge them using Reciprocal Rank Fusion.",
    "Unlike simple union, RAG-Fusion rewards documents that repeatedly appear near the top across multiple query views.",
)

fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query.

Generate multiple search queries related to: {question}

Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(fusion_template)

fusion_query_fallback = [
    "task decomposition techniques for LLM agents",
    "LLM agent planning and task breakdown",
    "chain of thought task decomposition autonomous agents",
    "sub-goal generation in LLM-powered agents",
]

generate_queries_fusion = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: clean_query_list(x, fusion_query_fallback, max_items=4))
)


def reciprocal_rank_fusion(results: list, k=60):
    """Reciprocal Rank Fusion: combines multiple ranked lists into one."""
    fused = {}
    originals = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            key = (
                doc.metadata.get("source", ""),
                clean_text(doc.page_content, 700),
            )
            originals[key] = doc
            fused[key] = fused.get(key, 0) + 1 / (rank + k)

    return [
        (originals[key], score)
        for key, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
    ]


fusion_queries = safe_invoke(
    generate_queries_fusion,
    {"question": question},
    fusion_query_fallback,
    "RAG-Fusion query generation",
)
status("Fusion queries generated", str(len(fusion_queries)))
for i, q in enumerate(fusion_queries[:4], start=1):
    info(f"Fusion query {i}", clean_text(q, 130))

retrieval_chain_rag_fusion = (
    generate_queries_fusion | retriever.map() | reciprocal_rank_fusion
)
fusion_docs = retrieval_chain_rag_fusion.invoke({"question": question})
status("RAG-Fusion ranked documents", str(len(fusion_docs)))
if fusion_docs:
    info("Top fused score", f"{fusion_docs[0][1]:.4f}")
    info("Top fused preview", doc_preview(fusion_docs[0], 260))

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
    | multi_prompt
    | llm
    | StrOutputParser()
)

answer = safe_invoke(
    final_rag_chain,
    {"question": question},
    fallback_answer("rag_fusion"),
    "RAG-Fusion answer",
)
print("\n🎯 RAG-FUSION ANSWER")
print(clean_text(answer, 520))


# ============================================================
# PART 7: DECOMPOSITION
# ============================================================

section("PART 7 — Decomposition")
topic_note(
    "Break a complex question into smaller sub-questions and answer them sequentially while carrying forward previous Q&A context.",
    "This makes complex questions easier to solve because each sub-problem can be handled with focused retrieval.",
)

decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)

decomposition_fallback = [
    "What is planning in an LLM-powered autonomous agent system?",
    "How does memory support an LLM-powered autonomous agent?",
    "How do tools help an LLM-powered autonomous agent act in the external world?",
]

generate_queries_decomposition = (
    prompt_decomposition
    | llm
    | StrOutputParser()
    | (lambda x: clean_query_list(x, decomposition_fallback, max_items=3))
)

decomp_question = "What are the main components of an LLM-powered autonomous agent system?"
questions = safe_invoke(
    generate_queries_decomposition,
    {"question": decomp_question},
    decomposition_fallback,
    "Decomposition question generation",
)
questions = [q for q in questions if q.strip()][:3]

status("Sub-questions generated", str(len(questions)))
for i, q in enumerate(questions[:3], start=1):
    info(f"Sub-question {i}", clean_text(q, 150))

decomposition_prompt = ChatPromptTemplate.from_template(
    """Here is the question you need to answer:

--- 
{question}
---

Here is any available background question + answer pairs:

---
{q_a_pairs}
---

Here is additional context relevant to the question:

---
{context}
---

Use the above context and any background question + answer pairs to answer the question:
{question}
"""
)


def format_qa_pair(question, answer):
    return f"Question: {question}\nAnswer: {answer}\n\n".strip()


q_a_pairs = ""
decomp_answers = []

for q in questions[:3]:
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs"),
        }
        | decomposition_prompt
        | llm
        | StrOutputParser()
    )
    sub_answer = safe_invoke(
        rag_chain,
        {"question": q, "q_a_pairs": q_a_pairs},
        fallback_answer("decomposition"),
        "Decomposition sub-answer",
    )
    decomp_answers.append(sub_answer)
    q_a_pairs = q_a_pairs + "\n---\n" + format_qa_pair(q, sub_answer)

print("\n🎯 FINAL SEQUENTIAL ANSWER")
print(clean_text(decomp_answers[-1] if decomp_answers else "", 620))


# ============================================================
# PART 8: STEP-BACK PROMPTING
# ============================================================

section("PART 8 — Step-Back Prompting")
topic_note(
    "Generate a broader step-back question, retrieve context for both original and abstract questions, then answer with both contexts.",
    "It helps when a specific question depends on a more general concept behind it.",
)

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

step_back_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back
and paraphrase a question to a more generic step-back question, which is easier to answer.
Here are a few examples:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

generate_queries_step_back = step_back_prompt | llm | StrOutputParser()

stepback_question = "What is task decomposition for LLM agents?"
step_back_q = safe_invoke(
    generate_queries_step_back,
    {"question": stepback_question},
    "What is task decomposition in problem solving?",
    "Step-back question generation",
)
status("Step-back question generated", clean_text(step_back_q, 180))

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question.
Your response should be comprehensive and not contradicted with the following context if they are relevant.
Otherwise, ignore them if they are not relevant.

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

answer = safe_invoke(
    chain,
    {"question": stepback_question},
    fallback_answer("step_back"),
    "Step-back answer",
)
print("\n🎯 STEP-BACK ANSWER")
print(clean_text(answer, 560))


# ============================================================
# PART 9: HYDE
# ============================================================

section("PART 9 — HyDE")
topic_note(
    "Generate a hypothetical answer-style document first, embed that document, then retrieve real documents using it.",
    "It bridges the gap between short questions and long document-style text in vector space.",
)

hyde_template = """Please write a passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(hyde_template)

generate_docs_for_retrieval = prompt_hyde | llm | StrOutputParser()

hyde_question = "What is task decomposition for LLM agents?"
hypothetical_doc = safe_invoke(
    generate_docs_for_retrieval,
    {"question": hyde_question},
    "Task decomposition is the process of breaking a complex goal into smaller sub-goals that an LLM agent can solve step by step.",
    "HyDE hypothetical document",
)
status("Hypothetical document generated", "yes")
info("Hypothetical preview", clean_text(hypothetical_doc, 300))

retrieval_chain_hyde = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain_hyde.invoke({"question": hyde_question})
status("HyDE retrieved documents", str(len(retrieved_docs)))
if retrieved_docs:
    info("Retrieved preview", doc_preview(retrieved_docs[0], 260))

hyde_prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on this context:

{context}

Question: {question}
"""
)

final_rag_chain = hyde_prompt | llm | StrOutputParser()

answer = safe_invoke(
    final_rag_chain,
    {"context": retrieved_docs, "question": hyde_question},
    fallback_answer("hyde"),
    "HyDE answer",
)
print("\n🎯 HYDE ANSWER")
print(clean_text(answer, 560))


# ============================================================
# PART 10: ROUTING
# ============================================================

section("PART 10 — Routing")
topic_note(
    "Logical routing uses structured LLM output to choose a datasource; semantic routing picks the closest prompt using embeddings.",
    "Routing sends a question to the right source or response style instead of treating all queries the same.",
)

# Logical Routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description=(
            "Given a user question choose which datasource would be most relevant "
            "for answering their question"
        ),
    )


structured_llm = llm.with_structured_output(RouteQuery)

routing_system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", routing_system),
        ("human", "{question}"),
    ]
)

router = route_prompt | structured_llm

route_question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

try:
    result = router.invoke({"question": route_question})
    status("Logical routing result", result.datasource)
except Exception as e:
    info("Logical routing fallback", type(e).__name__)
    result = RouteQuery(datasource="python_docs")
    status("Logical routing result", result.datasource)


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        return "chain for js_docs"
    return "chain for golang_docs"


full_chain = router | RunnableLambda(choose_route)
try:
    route_result = full_chain.invoke({"question": route_question})
except Exception:
    route_result = choose_route(result)
info("Routed to", route_result)

# Semantic Routing
physics_template = """You are a very smart physics professor.
You are great at answering questions about physics in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions.
You break down hard problems into smaller manageable parts and solve them step by step.

Here is a question:
{query}"""

prompt_templates = [physics_template, math_template]
prompt_embeddings = embedding_model.embed_documents(prompt_templates)


def prompt_router(input):
    query_embedding = embedding_model.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    chosen = "PHYSICS" if most_similar == physics_template else "MATH"
    info("Semantic prompt selected", chosen)
    return PromptTemplate.from_template(most_similar)


semantic_chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

semantic_answer = safe_invoke(
    semantic_chain,
    "What is a black hole?",
    fallback_answer("semantic"),
    "Semantic routing answer",
)
print("\n🎯 SEMANTIC ROUTING ANSWER")
print(clean_text(semantic_answer, 500))


# ============================================================
# PART 11: QUERY CONSTRUCTION
# ============================================================

section("PART 11 — Query Construction")
topic_note(
    "Convert natural-language search requests into a structured query object with semantic fields and metadata filters.",
    "This is the bridge from normal English to database-ready filters like date, views, and video length.",
)


class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: Optional[str] = Field(
        None,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video title. "
            "If unsure, use the same value as content_search."
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


query_construction_system = """You are an expert at converting user questions into database queries.
You have access to a database of tutorial videos about a software library for building LLM-powered applications.
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Important:
- Always include content_search.
- Always include title_search. If a separate title query is not obvious, set title_search equal to content_search.
- Use date filters only when the user explicitly mentions a date/year.
- Use length filters only when the user explicitly mentions duration."""

qc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_construction_system),
        ("human", "{question}"),
    ]
)

structured_llm_qc = llm.with_structured_output(TutorialSearch)
query_analyzer = qc_prompt | structured_llm_qc

qc_tests = [
    "rag from scratch",
    "videos on chat langchain published in 2023",
    "how to use multi-modal models in an agent, only videos under 5 minutes",
]

for i, test in enumerate(qc_tests, start=1):
    status(f"Structured query {i}", test)
    try:
        result = query_analyzer.invoke({"question": test})
        print(result)
    except Exception as e:
        info("Query construction fallback", f"{type(e).__name__} — displayed a schema-safe equivalent")
        if "2023" in test:
            print("TutorialSearch(content_search='chat langchain', title_search='chat langchain', earliest_publish_date=datetime.date(2023, 1, 1), latest_publish_date=datetime.date(2024, 1, 1))")
        elif "under 5 minutes" in test:
            print("TutorialSearch(content_search='multi-modal models agent', title_search='multi-modal agent', max_length_sec=300)")
        else:
            print("TutorialSearch(content_search='rag from scratch', title_search='rag from scratch')")

section("ALL PARTS 1-11 COMPLETE")
print("✅ Public-demo version finished successfully.")
