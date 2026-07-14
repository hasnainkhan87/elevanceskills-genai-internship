"""
langchain_helper.py
====================
Core LangChain logic for the Nullclass FAQ chatbot.

This file does two things on top of the original training project:

1. MIGRATION: GooglePalm + HuggingFaceInstructEmbeddings -> Gemini.
   `GooglePalm` is fully retired. `text-embedding-004` (the previous free
   Gemini embedding model) was ALSO shut down (Jan 14, 2026). The current
   free-tier replacements are:
       - LLM:        gemini-2.5-flash      (chat / Q&A generation)
       - Embeddings: gemini-embedding-001  (replaces text-embedding-004)
   Both model names are defined once below — if Google changes free-tier
   availability again, this is the only place you need to edit.

2. NEW FEATURE: an incremental knowledge-base updater.
   - create_vector_db()             -> ORIGINAL full build. Reads the master
                                        dataset/dataset.csv and builds a fresh
                                        FAISS index from scratch. Kept as-is
                                        for first-time setup (the "Create
                                        Knowledgebase" button).
   - update_vector_db_incremental() -> NEW. Scans a watched folder
                                        (new_sources/) for new or changed
                                        CSV/TXT files, works out which
                                        individual FAQ rows have never been
                                        embedded before, embeds ONLY those
                                        new rows, and merges them into the
                                        EXISTING FAISS index. The index is
                                        never rebuilt from scratch and
                                        nothing is ever re-embedded.

See README.md -> "Dynamic Knowledge Base Updates" for the full design
rationale (why a watched folder instead of URL scraping, why a two-level
hash dedup, etc).
"""

import os
import json
import hashlib
import datetime as dt
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

load_dotenv()  # take environment variables from .env (GOOGLE_API_KEY)

# --------------------------------------------------------------------------
# Paths — resolved relative to THIS file, so it doesn't matter whether you
# run `streamlit run src/main.py` from the project root, from src/, etc.
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # project root
DATASET_CSV = BASE_DIR / "dataset" / "dataset.csv"          # original master FAQ file
NEW_SOURCES_DIR = BASE_DIR / "new_sources"                  # <-- the "watched folder"
FAISS_INDEX_DIR = str(BASE_DIR / "faiss_index")
MANIFEST_PATH = BASE_DIR / "kb_manifest.json"

NEW_SOURCES_DIR.mkdir(exist_ok=True)  # make sure the watched folder exists

# --------------------------------------------------------------------------
# Gemini model names (free tier, as of June 2026 — see README for flags)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Manifest helpers — kb_manifest.json is the small bookkeeping file that
# makes incremental updates possible. It tracks which file contents and
# which individual FAQ rows have already been embedded.
# --------------------------------------------------------------------------
def _load_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "ingested_row_hashes": {},  # row_hash -> {"source": ..., "ingested_at": ...}
        "file_hashes": {},          # relative file path -> sha256 of file bytes
        "total_documents": 0,
        "last_updated": None,
        "history": [],              # log of every build/update run
    }


def _save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _hash_text(text: str) -> str:
    """Content hash for a single FAQ row (prompt+response). This is the
    real dedup key — see README for why row-level beats file-level alone."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    """Whole-file hash, used purely as a cheap fast-path: if a watched
    file's bytes are byte-for-byte identical to last time, skip opening
    and parsing it at all."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------
def _load_csv_documents(path: Path):
    """Load a prompt/response CSV the same way the original project does.
    Falls back to cp1252 if the file isn't valid UTF-8 — the real Nullclass
    dataset.csv in this repo was exported from Excel/Windows and contains
    cp1252 punctuation (curly quotes), which raises UnicodeDecodeError under
    plain utf-8."""
    last_err = None
    for encoding in ("utf-8", "cp1252"):
        try:
            loader = CSVLoader(file_path=str(path), source_column="prompt", encoding=encoding)
            return loader.load()
        except (UnicodeDecodeError, RuntimeError) as e:
            # Newer langchain-community wraps the UnicodeDecodeError in a
            # RuntimeError, so we check the cause rather than the outer type.
            if isinstance(e, RuntimeError) and not isinstance(e.__cause__, UnicodeDecodeError):
                raise
            last_err = e
            continue
    raise ValueError(f"Could not decode {path} as utf-8 or cp1252: {last_err}")


def _load_txt_documents(path: Path):
    """Very simple .txt ingestion: split on blank lines into paragraphs,
    one Document per paragraph. Good enough for short FAQ-style snippets;
    swap in a real text splitter if you start dropping long articles in
    new_sources/."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [Document(page_content=c, metadata={"source": path.name}) for c in chunks]


# --------------------------------------------------------------------------
# 1) ORIGINAL full build — unchanged behaviour, new embeddings
# --------------------------------------------------------------------------
def create_vector_db(embeddings=None):
    """Full (re)build of the FAISS index from the master dataset/dataset.csv.
    This is the ORIGINAL "Create Knowledgebase" button behaviour, kept for
    first-time setup. It also seeds kb_manifest.json so the incremental
    updater below knows these rows are already embedded and will never
    re-add or re-embed them."""
    embeddings = embeddings or get_embeddings()
    docs = _load_csv_documents(DATASET_CSV)

    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(FAISS_INDEX_DIR)

    manifest = _load_manifest()
    now = dt.datetime.now().isoformat(timespec="seconds")
    for doc in docs:
        row_hash = _hash_text(doc.page_content)
        manifest["ingested_row_hashes"][row_hash] = {
            "source": "dataset/dataset.csv",
            "ingested_at": now,
        }
    manifest["file_hashes"][str(DATASET_CSV.relative_to(BASE_DIR))] = _hash_file(DATASET_CSV)
    manifest["total_documents"] = len(manifest["ingested_row_hashes"])
    manifest["last_updated"] = now
    manifest["history"].append(
        {"timestamp": now, "action": "full_rebuild", "documents_added": len(docs)}
    )
    _save_manifest(manifest)

    print(f"[create_vector_db] Built FAISS index with {len(docs)} FAQs and seeded manifest.")
    return len(docs)


# --------------------------------------------------------------------------
# 2) NEW — incremental updater
# --------------------------------------------------------------------------
def update_vector_db_incremental(sources_dir: Path = None, embeddings=None, verbose=True):
    """
    Scan `sources_dir` (default: new_sources/) for new or changed CSV/TXT
    files and merge ONLY the genuinely new FAQ rows into the existing FAISS
    index — without rebuilding it and without re-embedding anything that
    was already ingested.

    Dedup strategy (two levels, see README for full rationale):
      1. File-level hash  -> cheap fast-path. If a watched file's content
         hash is identical to last run, skip it entirely: no parsing, no
         embedding calls, no API quota used.
      2. Row-level hash   -> the real dedup guard. Every individual FAQ row
         (prompt+response) is hashed. Only rows never seen before are kept
         for embedding. This correctly handles someone *appending* new
         rows to a file that was already partially ingested, or the same
         row showing up in two different files.

    Returns a summary dict and prints a human-readable log line, so this
    can be called from the scheduler, a Streamlit button, or manually.
    """
    sources_dir = Path(sources_dir) if sources_dir else NEW_SOURCES_DIR
    embeddings = embeddings or get_embeddings()
    manifest = _load_manifest()
    now = dt.datetime.now().isoformat(timespec="seconds")

    new_docs = []                  # list of (row_hash, Document) — genuinely new
    files_skipped_unchanged = []
    files_scanned = []
    rows_skipped_duplicate = 0

    candidate_files = sorted(p for p in sources_dir.glob("*") if p.suffix.lower() in (".csv", ".txt"))

    for path in candidate_files:
        rel_path = str(path.relative_to(BASE_DIR))
        file_hash = _hash_file(path)

        if manifest["file_hashes"].get(rel_path) == file_hash:
            files_skipped_unchanged.append(rel_path)
            continue  # fast path — nothing changed in this file since last check

        files_scanned.append(rel_path)
        try:
            docs = _load_csv_documents(path) if path.suffix.lower() == ".csv" else _load_txt_documents(path)
        except Exception as e:
            if verbose:
                print(f"[update] ERROR reading {rel_path}: {e}")
            continue

        for doc in docs:
            row_hash = _hash_text(doc.page_content)
            if row_hash in manifest["ingested_row_hashes"]:
                rows_skipped_duplicate += 1
                continue
            doc.metadata["source_file"] = rel_path
            new_docs.append((row_hash, doc))

        manifest["file_hashes"][rel_path] = file_hash  # remember even if 0 new rows came from it

    if new_docs:
        docs_to_embed = [d for _, d in new_docs]
        if not Path(FAISS_INDEX_DIR).exists():
            # No index yet at all (e.g. someone deleted faiss_index/) — build
            # one from just these documents. Still not a "rebuild" in the
            # sense the task means: we are not re-embedding any prior data,
            # because there is none yet.
            vectordb = FAISS.from_documents(documents=docs_to_embed, embedding=embeddings)
        else:
            vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            vectordb.add_documents(docs_to_embed)  # <-- only the NEW docs get embedded here
        vectordb.save_local(FAISS_INDEX_DIR)

        for row_hash, doc in new_docs:
            manifest["ingested_row_hashes"][row_hash] = {
                "source": doc.metadata.get("source_file"),
                "ingested_at": now,
            }
        manifest["total_documents"] = len(manifest["ingested_row_hashes"])

    manifest["last_updated"] = now
    manifest["history"].append(
        {
            "timestamp": now,
            "action": "incremental_update",
            "files_scanned": files_scanned,
            "files_skipped_unchanged": files_skipped_unchanged,
            "documents_added": len(new_docs),
            "rows_skipped_duplicate": rows_skipped_duplicate,
        }
    )
    _save_manifest(manifest)

    summary = {
        "timestamp": now,
        "files_scanned": files_scanned,
        "files_skipped_unchanged": files_skipped_unchanged,
        "new_documents_added": len(new_docs),
        "rows_skipped_duplicate": rows_skipped_duplicate,
        "total_documents_in_kb": manifest["total_documents"],
    }

    if verbose:
        print(f"[{now}] Knowledge base update check")
        print(f"  Files scanned (changed):     {files_scanned or 'none'}")
        print(f"  Files skipped (unchanged):   {files_skipped_unchanged or 'none'}")
        print(f"  New FAQ rows embedded+added: {len(new_docs)}")
        print(f"  Duplicate rows skipped:      {rows_skipped_duplicate}")
        print(f"  Total documents in KB now:   {manifest['total_documents']}")

    return summary


def get_kb_status():
    """Small read-only helper for the Streamlit UI: last update time, doc
    count, and recent history — without touching FAISS or calling the API."""
    manifest = _load_manifest()
    return {
        "faiss_index_exists": Path(FAISS_INDEX_DIR).exists(),
        "last_updated": manifest.get("last_updated"),
        "total_documents": manifest.get("total_documents", 0),
        "history": manifest.get("history", [])[-5:],
    }


# --------------------------------------------------------------------------
# Q&A chain — same retrieval-QA design as the original project, pointed at
# Gemini instead of PaLM/HuggingFace.
# --------------------------------------------------------------------------
def get_qa_chain():
    embeddings = get_embeddings()
    vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # NOTE: the original project called `vectordb.as_retriever(score_threshold=0.7)`.
    # I tested this directly against a FAISS retriever and confirmed it is
    # silently swallowed as a no-op in current langchain — VectorStoreRetriever
    # only recognizes `search_type` / `search_kwargs`, so that threshold never
    # actually did anything, even in the original repo. The "real" fix is
    # `search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7}`,
    # but FAISS's default relevance-score conversion is calibrated for
    # normalized embeddings around ~1536 dims (e.g. OpenAI ada-002) and can
    # behave unpredictably with gemini-embedding-001's 3072-dim vectors —
    # I don't have API access in this environment to calibrate a real
    # threshold against actual Gemini embedding scores, so I'm not shipping
    # an untested cutoff that could silently return zero results. Plain
    # top-k similarity search (below) is robust regardless of embedding
    # scale, and the prompt template already instructs Gemini to say
    # "I don't know" when the retrieved context doesn't contain the answer.
    # If you want score-based filtering, test real thresholds on your data
    # first, then switch to the search_type/search_kwargs form above.
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

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


if __name__ == "__main__":
    import sys

    if "--update" in sys.argv:
        update_vector_db_incremental()
    else:
        create_vector_db()
        chain = get_qa_chain()
        print(chain.invoke({"query": "hello?"}))
