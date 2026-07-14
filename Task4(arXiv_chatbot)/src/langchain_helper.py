"""
langchain_helper.py (Task 4 — arXiv CS Research Assistant)
=============================================================
Same overall shape as Task 1/3's langchain_helper.py (CSV → embed → FAISS →
retrieval → LLM), but with two deliberate architecture changes required by
this task's brief ("use open source LLM for explanation generation"):

  1. EMBEDDINGS: local sentence-transformers model instead of Gemini's API.
     Runs entirely on your machine, no API key, no quota, no rate limits.
  2. LLM: a local open-source model via Hugging Face `transformers`,
     instead of Gemini. Same reasoning — zero quota, zero API cost, fully
     offline once the model is downloaded once.

This directly solves the recurring problem from Task 1/3 (free-tier Gemini
quota exhaustion): with everything local, there is no daily/per-minute
request cap to hit at all.

FOLLOW-UP QUESTIONS
---------------------
Task 1/3 used RetrievalQA, which has no memory of previous turns — every
question is answered in isolation. This task's brief explicitly requires
"the ability to handle follow-up questions," so this uses LangChain's
ConversationalRetrievalChain instead, which takes a running chat_history
list and rephrases follow-up questions using that context before retrieval
(e.g. "what about its limitations?" gets resolved against whatever paper
was just discussed).

MODEL CHOICE (free-tier / no-GPU consideration)
--------------------------------------------------
  Embeddings: sentence-transformers/all-MiniLM-L6-v2
    - ~80MB download, runs fast on CPU, widely used, good general quality.
  LLM: google/flan-t5-base
    - ~250M parameters, ~1GB download, instruction-tuned (good at
      "answer this given this context" style prompts out of the box),
      runs on CPU (slowly — expect several seconds per answer, not
      Gemini-speed). This is the honest tradeoff of going fully local
      and free: noticeably lower quality and speed than a large hosted
      model, in exchange for zero quota/cost. A larger local model
      (e.g. Mistral-7B via Ollama) would answer better but needs
      several GB of RAM/VRAM most laptops don't have spare — flan-t5-base
      was chosen specifically to be runnable on a typical machine with no
      GPU. See README for how to swap in a bigger model if you have the
      hardware for it.

NOTE ON THIS BUILD ENVIRONMENT
---------------------------------
This code was written and its FAISS/retrieval mechanics were tested using
a local deterministic fake-embedding stand-in, the same technique used for
Tasks 1 and 3 — but for a different reason here: this sandbox's network
access is restricted to PyPI/GitHub/npm and cannot reach huggingface.co,
so the real sentence-transformers/flan-t5 model weights could not be
downloaded and run here. The pipeline structure (CSVLoader → FAISS →
ConversationalRetrievalChain) is identical to the already-proven Task 1/3
pattern; what's untested is specifically the real local model behavior,
which needs to be verified once on your own machine with internet access
to huggingface.co.
"""

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # same Windows/FAISS fix as Task 1/3

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

BASE_DIR = Path(__file__).resolve().parent.parent  # task4/
DATASET_CSV = BASE_DIR / "dataset" / "arxiv_cs_sample.csv"
FAISS_INDEX_DIR = str(BASE_DIR / "faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"


def get_embeddings():
    """Local, free, no API key. Downloads the model once on first use
    (~80MB), then runs fully offline from a local cache thereafter."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_llm():
    """
    Local, free, open-source LLM via a Hugging Face transformers pipeline.
    device=-1 forces CPU (safe default — works everywhere, just slower
    than a GPU). If you have a CUDA GPU available, change to device=0 for
    a significant speedup.
    """
    from transformers import pipeline

    hf_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        max_new_tokens=256,
        device=-1,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


def create_vector_db():
    """Load arxiv_cs_sample.csv, embed locally, save FAISS index."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} not found.\n"
            "Run the preprocessing step first (from inside task4/):\n"
            "  python src/preprocess_arxiv.py --input path/to/arxiv-metadata-oai-snapshot.json"
        )

    loader = CSVLoader(file_path=str(DATASET_CSV), source_column="prompt", encoding="utf-8")
    docs = loader.load()

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(FAISS_INDEX_DIR)

    print(f"[create_vector_db] arXiv FAISS index built with {len(docs)} papers.")
    return len(docs)


def get_conversational_chain():
    """
    Load the saved FAISS index and return a ConversationalRetrievalChain —
    unlike Task 1/3's RetrievalQA, this accepts a running chat_history so
    follow-up questions ("what about its limitations?") can be resolved
    against the previous turn.

    Usage:
        chain = get_conversational_chain()
        result = chain.invoke({"question": "What is a transformer?", "chat_history": []})
        chat_history = [(question, result["answer"])]
        result2 = chain.invoke({"question": "What are its downsides?", "chat_history": chat_history})
    """
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_prompt = PromptTemplate(
        template="""You are a research assistant explaining computer science concepts using
        the paper abstracts in the context below. Answer clearly, in your own words, for
        someone learning the topic. If the context doesn't contain the answer, say you don't
        have relevant papers on that topic rather than guessing.

        CONTEXT: {context}

        QUESTION: {question}

        ANSWER:""",
        input_variables=["context", "question"],
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
    )
    return chain


def summarize_text(text: str, llm=None) -> str:
    """
    Summarize a single paper abstract (or any retrieved text) using the
    same local LLM — a separate, simpler call than the conversational
    chain, for the 'summarize this paper' UI feature.
    """
    llm = llm or get_llm()
    prompt = (
        "Summarize the following research abstract in 2-3 plain-English "
        f"sentences for someone new to the topic:\n\n{text}\n\nSummary:"
    )
    return llm.invoke(prompt)
