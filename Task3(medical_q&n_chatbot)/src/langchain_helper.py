"""
langchain_helper.py (Task 3 — Medical Q&A)
============================================
Same architecture as Task 1's langchain_helper.py:
    CSVLoader → Gemini embeddings → FAISS → RetrievalQA → Gemini LLM

Differences from Task 1:
  - Points at dataset/medquad_dataset.csv instead of dataset/dataset.csv
  - FAISS index saved to faiss_index/ inside task3/ (separate from Task 1's
    own faiss_index/, since these are standalone folders)
  - Prompt template tuned for medical Q&A: answer only from context, say
    "I don't have information on that" if not found, remind the user to
    consult a professional for personal medical decisions
  - No incremental update scheduler — Task 3 uses a fixed, one-time-built
    dataset (MedQuAD doesn't change), so that part of Task 1 isn't reused here
"""

import os
from pathlib import Path

# Windows / FAISS OMP crash fix — must be set before numpy/faiss import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # task3/
DATASET_CSV = BASE_DIR / "dataset" / "medquad_dataset.csv"
FAISS_INDEX_DIR = str(BASE_DIR / "faiss_index")

# NOTE: embedding model requires "models/" prefix with langchain-google-genai 2.x —
# omitting it causes every embed call to fail with:
#   400 BatchEmbedContentsRequest.model: unexpected model name format
GEMINI_LLM_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.1,
    )


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


def create_vector_db():
    """
    Load medquad_dataset.csv, embed with Gemini, save FAISS index.

    Embeds in small batches with a delay between batches and retry-with-
    backoff on 429 "quota exceeded" errors, rather than firing all
    embedding calls at once. FAISS.from_documents() with no throttling
    reliably hits free-tier per-minute rate limits on a few hundred+ rows
    (confirmed: 2,366 rows failed partway through with a 429 in real
    testing) — this batches the work instead.
    """
    import time

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{DATASET_CSV} not found.\n"
            "Run the preprocessing step first (from inside task3/):\n"
            "  python src/preprocess_medquad.py"
        )

    loader = CSVLoader(file_path=str(DATASET_CSV), source_column="prompt", encoding="utf-8")
    docs = loader.load()

    embeddings = get_embeddings()

    BATCH_SIZE = 20          # small batch per embedding call
    DELAY_BETWEEN_BATCHES = 8  # seconds — conservative for a ~5-15 RPM free-tier limit
    MAX_RETRIES = 5

    vectordb = None
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if vectordb is None:
                    vectordb = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    vectordb.add_documents(batch)
                print(f"[create_vector_db] Batch {batch_num}/{total_batches} embedded "
                      f"({len(batch)} docs, {i + len(batch)}/{len(docs)} total).")
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES:
                    wait = DELAY_BETWEEN_BATCHES * attempt  # linear backoff
                    print(f"[create_vector_db] Rate limited on batch {batch_num}, "
                          f"attempt {attempt}/{MAX_RETRIES}. Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise

        if i + BATCH_SIZE < len(docs):
            time.sleep(DELAY_BETWEEN_BATCHES)

    vectordb.save_local(FAISS_INDEX_DIR)
    print(f"[create_vector_db] Medical FAISS index built with {len(docs)} QA pairs.")
    return len(docs)


def get_qa_chain():
    """Load the saved FAISS index and return a RetrievalQA chain."""
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt_template = """You are a medical information assistant answering questions using
    ONLY the context below, sourced from NIH health resources (MedQuAD dataset).

    Rules:
    - If the answer is not present in the context, say "I don't have information about
      that in my knowledge base." Do not invent medical facts.
    - Do not recommend specific medications, dosages, or personal treatment decisions —
      remind the user to consult a healthcare professional for those.
    - Prefer the wording from the context where possible.

    CONTEXT: {context}

    QUESTION: {question}

    ANSWER:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain
