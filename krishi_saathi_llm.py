import os
import json
from typing import Dict, Any, List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

# -----------------------------------------------------------------
# ENV + CONFIG
# -----------------------------------------------------------------
load_dotenv()

PDF_FILES = [
    "data/oilseed 1.pdf",
    "data/oilseed 2.pdf",
    "data/oilseed_feature_yield_ranges.pdf",
    "data/oilseed_featurewise_advisory_8crops.pdf",
    "data/top10_oilseed_agronomy_reference.pdf",
    "data/oilseed_benchmark_practices.pdf"  # 🔥 NEW Benchmark PDF
]

FAISS_DIR = "faiss_oilseed_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# -----------------------------------------------------------------
# FINAL UPDATED KRISHI SAATHI SYSTEM PROMPT WITH BENCHMARKING
# -----------------------------------------------------------------
KRISHI_SAATHI_PROMPT = r"""
You are **Krishi Saathi 2.0**, an advanced RAG-enabled Oilseed Agronomy Expert.

🎯 Your mission:
Analyze farmer field inputs + ML predicted yield and provide
✔ Feature-wise diagnosis ✔ Actionable agronomy solutions ✔ Global benchmarking insights

You must always combine THREE knowledge sources:
1️⃣ Retrieved context from 6 PDFs:
   • India oilseed practices
   • Yield performance ranges
   • Feature-wise advisory
   • TOP 3 global producer benchmarking & advanced practices
2️⃣ JSON yield_context (crop + yield + feature values)
3️⃣ Your agronomy knowledge for India

=========================================
BENCHMARKING RESPONSIBILITY
=========================================
For each crop:
• Retrieve the **global top 3 yield ranges**
• Compare farmer predicted yield vs. those ranges:
  - “X% below global benchmark”
• Highlight **limiting features** compared to:
  - India Recommended Practice
  - Top-producing countries (USA/Brazil/China/etc.)

Include benchmarking for:
- Fertilizer (N/P/K/S micronutrients)
- Soil pH / OC / Texture
- Seasonal & flowering rainfall
- Temperature
- NDVI band performance
- Irrigation frequency & method (e.g., drip)
- Plant density & sowing window
- Pest / disease & weed IPM techniques
- Soil depth & nutrient status
- Harvest moisture

=========================================
OUTPUT FORMAT (STRICT)
=========================================

1️⃣ Yield Situation Summary
- Crop
- ML predicted yield (in t/ha)
- LOW / MEDIUM / HIGH vs Indian ranges
- 🔥 Compare to Global Benchmark Yield
  • Show difference in quantifiable terms (gap %)

2️⃣ Feature-wise Diagnosis
Pick 6–10 limiting parameters:
- <feature> = <value> → (LOW / MEDIUM / HIGH)
- Why limiting?
- 🔹 Show expected range from:
  • PDFs advisory (India)
  • Global benchmark countries

3️⃣ Detailed Advisory
Under the following headings:
a) Variety / sowing / plant population
b) Soil health & fertiliser plan (accurate numeric targets)
c) Irrigation & moisture (critical stages)
d) Weed / pest & disease IPM
e) NDVI monitoring and corrective actions
✔ Include which practices are inspired from
  **Top 3 global producers** + Low-cost adaptation for India

4️⃣ Top 3 Priority Actions (Short, High Impact)
Only 3 bullets — economical & practical

5️⃣ Follow-up Question
Ask for exactly ONE detail that would refine next advice.

=========================================

🔔 Language Instructions:
- Use language requested (Hindi / Odia / English / Auto detect)
- No mixed language inside same response

If uncertain → clarify assumptions & give safe advice.
"""

# -----------------------------------------------------------------
# VECTORSTORE
# -----------------------------------------------------------------
def build_or_load_vectorstore(pdf_paths: List[str], index_dir: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.isdir(index_dir) and os.listdir(index_dir):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(index_dir)
    return vectordb


# -----------------------------------------------------------------
# RAG CHAIN
# -----------------------------------------------------------------
def create_krishi_saathi_chain() -> RunnableWithMessageHistory:

    vectordb = build_or_load_vectorstore(PDF_FILES, FAISS_DIR)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables")

    llm = ChatOpenAI(
        api_key=openai_key,
        model="gpt-4.1-mini",  # 🔹 Token efficient
        temperature=0.2,
    )

    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user query using history and keep crop context intact."),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    history_retriever = create_history_aware_retriever(llm, retriever, contextual_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", KRISHI_SAATHI_PROMPT + """

================ PDF CONTEXT ================
{context}

================ YIELD CONTEXT ================
{yield_context}

================ FIELD FEATURES ================
{feature_list}

================ LANGUAGE =====================
{language}

Farmer Query:
"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_retriever, doc_chain)

    history_store: Dict[str, BaseChatMessageHistory] = {}

    def get_history(session_id: str):
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        return history_store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


# -----------------------------------------------------------------
# PUBLIC INTERFACE FOR APP
# -----------------------------------------------------------------
class KrishiSaathiAdvisor:

    def __init__(self):
        self.chain = create_krishi_saathi_chain()

    def chat(self, session_id: str, farmer_query: str,
             yield_dict: Dict[str, Any], language: str = "auto") -> str:

        features = yield_dict.get("features", {})
        feature_str = "\n".join(f"- {k} = {v}" for k, v in sorted(features.items()))

        payload = {
            "input": farmer_query,
            "yield_context": json.dumps(yield_dict, ensure_ascii=False),
            "language": language,
            "feature_list": feature_str if feature_str else "No feature data."
        }

        result = self.chain.invoke(
            payload, config={"configurable": {"session_id": session_id}}
        )
        return result["answer"]
