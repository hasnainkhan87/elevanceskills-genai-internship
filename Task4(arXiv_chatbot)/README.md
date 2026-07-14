# Task 4 — arXiv CS Research Assistant (open-source LLM)

An extension of the same retrieval architecture used in Tasks 1 and 3
(CSV → embeddings → FAISS → LangChain retrieval → LLM → Streamlit), with
one deliberate architectural change required by this task's brief: **the
embedding model and the LLM are both open-source and run locally**,
instead of Google Gemini's hosted API.

## Problem statement

Build a domain-expert chatbot over a subset of the arXiv dataset
(computer science papers) that can: answer complex questions, summarize
papers, explain concepts, handle follow-up questions, search papers, and
show a simple concept visualization — using an open-source LLM rather than
a hosted API.

## Why this is architecturally different from Task 1/3 (and why)

| | Task 1 / Task 3 | Task 4 |
|---|---|---|
| Embeddings | Gemini API (`gemini-embedding-001`) | Local `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Gemini API (`gemini-2.5-flash`) | Local `google/flan-t5-base` (Hugging Face `transformers`) |
| Chain type | `RetrievalQA` (no memory) | `ConversationalRetrievalChain` (tracks chat history for follow-ups) |
| API key needed | Yes (`GOOGLE_API_KEY`) | No |
| Quota/rate limits | Yes — hit repeatedly during Task 3's build | None |

This swap was made for two reasons: (1) the task brief explicitly says
"use open source LLM for explanation generation," and (2) Task 3's build
repeatedly hit Gemini's free-tier daily embedding quota (1,000
requests/day) during development — a fully local stack has no quota to
run out of. Everything downstream of the embedding/LLM swap — the CSV
shape, FAISS indexing, the retriever, the Streamlit layout pattern — is
the same architecture as Tasks 1 and 3, just with a different model
backend. **Worth flagging to your mentor:** this is a bigger architectural
deviation than Task 3's dataset swap, since it changes the LLM/embedding
provider itself, not just the data — confirm it's acceptable under your
internship's "same architecture" rule the same way Task 3's dataset
substitution was confirmed.

## Dataset: arXiv (Kaggle, Cornell University)

The full snapshot (`arxiv-metadata-oai-snapshot.json`) is a single JSONL
file (one JSON object per line, not a JSON array), currently around
2.8 million papers and 1.1GB+. Relevant fields: `id`, `title`, `abstract`,
`categories` (space-separated tags like `cs.CV cs.LG`), `authors`,
`update_date`.

### Why streaming + reservoir sampling, not `pandas.read_json()`

The full file is far too large to load into memory at once on a typical
laptop. `preprocess_arxiv.py` reads it **one line at a time** — memory use
stays flat regardless of file size — and filters to papers with at least
one category starting with `cs.` (configurable via `CATEGORY_PREFIXES`).

Because the total number of matching cs.* papers isn't known until the
whole file has been scanned, plain `random.sample()` can't be used (it
needs the full list up front). Instead the script uses **reservoir
sampling** — a streaming algorithm that produces a uniform random sample
of fixed size from a stream of unknown length, in a single pass, holding
at most `max_rows` candidates in memory at any time. This is the correct
tool for "sample N items from a huge stream."

### Sample size

`DEFAULT_MAX_ROWS = 300`. Given Task 3's real-world lesson (even a modest
dataset can take a long time to fully embed on a constrained machine —
though for a different reason here, since local embedding has no API
quota, the constraint is just CPU speed), 300 is a reasonable starting
point that's easy to lower (`--max_rows 100`) if a first build feels slow
on your hardware, or raise if it doesn't.

### Verified, real output

Tested against a synthetic 2,000-line JSONL fixture (built to mimic the
real schema, including malformed/empty lines to test robustness):
- Correctly skipped an unparseable line without crashing
- Correctly filtered to only papers with a `cs.*` category — verified
  zero false positives among the sampled output
- Correctly reproduced an identical sample across two separate runs with
  the same seed

## Concept extraction

`concept_extraction.py` — same curated-keyword pattern as Task 3's
`medical_ner.py`, adapted to a CS/ML vocabulary across four categories:
MODEL (transformer, CNN, GAN, ...), TASK (classification, translation,
...), METHOD (gradient descent, transfer learning, ...), DATASET_EVAL
(F1 score, benchmark, accuracy, ...).

**Why keywords again:** same reasoning as Task 3 — a real
research-concept extraction model needs domain-specific training data
that isn't available as a free, ready-to-use tool. A curated list is
deterministic, testable without a model download, and satisfies the
brief's "information extraction" requirement at a basic level.

**Verified, real output** from the module's self-test
(`python src/concept_extraction.py`):
```
We propose a convolutional neural network for image classification trained
using stochastic gradient descent, achieving state of the art accuracy on
the benchmark dataset.
 -> [{'term': 'convolutional neural network', 'category': 'MODEL'},
     {'term': 'stochastic gradient descent', 'category': 'METHOD'},
     {'term': 'image classification', 'category': 'TASK'},
     {'term': 'benchmark dataset', 'category': 'DATASET_EVAL'},
     {'term': 'state of the art', 'category': 'DATASET_EVAL'},
     {'term': 'accuracy', 'category': 'DATASET_EVAL'}]
```
Longest-match-first correctly picked "convolutional neural network" as
one term rather than also separately matching "neural network" inside it.

## Methodology

1. `preprocess_arxiv.py` streams the raw arXiv JSONL and reservoir-samples
   a cs.*-filtered subset into `dataset/arxiv_cs_sample.csv`
2. Clicking "Create Knowledgebase" loads that CSV, embeds every row
   locally with `sentence-transformers/all-MiniLM-L6-v2`, and saves a
   FAISS index — no network call beyond the one-time model download
3. A user question is embedded the same way; FAISS returns the top-4
   nearest papers by meaning
4. Those are passed, along with the running chat history, to a
   `ConversationalRetrievalChain` backed by a local `flan-t5-base` model —
   this is what enables follow-up questions ("what about its
   limitations?") to be understood in context of the previous turn
5. `concept_extraction.py` tags the user's question with recognized
   CS/ML terms, shown as colored tags in the UI
6. The Paper Search tab does plain keyword filtering over the CSV — no
   LLM involved, so it's instant regardless of model speed
7. The Concepts tab runs `concept_frequency()` across all loaded
   abstracts and shows a bar chart of the most common recognized terms

## Free-tier / open-source model tradeoffs (read before running)

- **Model size vs. hardware:** `flan-t5-base` (~250M params) was chosen
  specifically to run on a CPU-only laptop with no GPU. It will be
  noticeably slower (several seconds per answer) and less fluent than
  Gemini. If you have a GPU or more RAM available, swapping to a larger
  model (e.g. `google/flan-t5-large`, or a 7B model via Ollama) would
  improve answer quality — change `LLM_MODEL_NAME` in `langchain_helper.py`
  and, for very large models, consider `device=0` in `get_llm()` to use a
  GPU.
- **First run downloads models:** the first time `create_vector_db()` or
  `get_llm()` runs, it downloads the embedding model (~80MB) and the LLM
  (~1GB) from Hugging Face and caches them locally. This needs internet
  access once; every run after that is fully offline.
- **No daily/per-minute quota**, unlike Gemini — this was the whole point
  of the architecture change, given Task 3's repeated 429 errors.

## What was verified vs. what wasn't

**Verified, with real captured output:**
- The streaming/reservoir-sampling preprocessing logic (tested against a
  2,000-line synthetic fixture with malformed lines, confirmed correct
  category filtering and reproducible sampling)
- The concept extraction module (self-test output above)
- The FAISS/CSVLoader/`ConversationalRetrievalChain` pipeline mechanics —
  tested end-to-end using local deterministic fake embeddings and
  LangChain's built-in `FakeListLLM`, confirming the chain builds
  correctly, retrieves the right documents, accepts and passes
  `chat_history` for follow-ups, and `summarize_text()` runs correctly

**Not verified — needs to be run once on your own machine:**
- The real `sentence-transformers` and `flan-t5-base` model behavior —
  this build environment's network access doesn't reach huggingface.co,
  so the actual embedding quality and LLM answer quality haven't been
  observed directly, only the surrounding pipeline logic
- A real run against the full arXiv snapshot at scale (the fixture test
  used a small synthetic file, not the actual 1.1GB+ dataset)
- The Streamlit UI rendering in a browser, including the tabs, chat
  interface, paper search, and concept bar chart

## Project structure

```
task4/
├── dataset/
│   └── arxiv_cs_sample.csv     # generated by preprocess_arxiv.py (not committed)
├── src/
│   ├── main.py                 # Streamlit UI: chat, search, concepts
│   ├── langchain_helper.py     # local embeddings + local LLM + conversational retrieval
│   ├── concept_extraction.py   # keyword-based CS/ML concept tagging
│   └── preprocess_arxiv.py     # arXiv JSONL → filtered/sampled CSV
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
cd task4

# 1. Download the arXiv metadata snapshot from Kaggle:
#    https://www.kaggle.com/datasets/Cornell-University/arxiv
#    (arxiv-metadata-oai-snapshot.json — requires a free Kaggle account)
#    Place it anywhere; you'll pass its path to the preprocessing script.

# 2. Install dependencies (torch download is large — a few GB)
pip install -r requirements.txt

# 3. Preprocess (streams the file — can take a few minutes for a 1GB+ file)
python src/preprocess_arxiv.py --input path/to/arxiv-metadata-oai-snapshot.json

# 4. (Windows only, every session) fix the FAISS/OpenMP conflict
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 5. Run the app
streamlit run src/main.py
```

In the browser: click **"Create Knowledgebase"** — the first click will
also download the embedding model and LLM from Hugging Face (a few
hundred MB to ~1GB total), so it may take several minutes the first time
only. After that, ask a question in the Chat tab, try a follow-up, search
papers by keyword, and check the Concepts tab for a frequency chart.
