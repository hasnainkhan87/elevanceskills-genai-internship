# Task 3 — Medical Q&A Chatbot (MedQuAD)

An extension of the same architecture used in Task 1 (Nullclass FAQ chatbot):
CSV → Gemini embeddings → FAISS → LangChain RetrievalQA → Gemini LLM →
Streamlit UI — pointed at a different, domain-specific dataset (MedQuAD)
instead of the original e-learning FAQ data.

## Problem statement

The brief: build a medical Q&A chatbot using the MedQuAD dataset, with basic
retrieval, basic medical entity recognition (symptoms/diseases/treatments),
and a simple Streamlit interface — reusing the same pipeline pattern as
Task 1 rather than building a new architecture from scratch.

## Why this reuses Task 1's architecture (not a new project)

| Task 1 (Nullclass FAQ) | Task 3 (Medical Q&A) |
|---|---|
| `dataset/dataset.csv` (prompt, response) | `dataset/medquad_dataset.csv` (prompt, response, focus, source, qtype) |
| `CSVLoader(source_column="prompt")` | Same, unchanged |
| `gemini-2.5-flash` LLM, `models/gemini-embedding-001` embeddings | Identical model names, identical wrapper functions `get_llm()` / `get_embeddings()` |
| `FAISS.from_documents()` → `save_local()` | Identical |
| `RetrievalQA.from_chain_type(chain_type="stuff", ...)` | Identical chain type and structure |
| `create_vector_db()` / `get_qa_chain()` function names | Same names, same signatures |
| Streamlit: title, sidebar status, button, question box, answer | Same layout, adds a disclaimer box and entity tags |

The only genuinely new pieces are `preprocess_medquad.py` (XML → CSV, since
MedQuAD ships as XML rather than a ready CSV) and `medical_ner.py` (keyword
entity tagging, which the FAQ chatbot never needed). Everything downstream
of the CSV — loading, embedding, indexing, retrieval, generation, UI — is
the same pipeline, just pointed at different data.

## Dataset: MedQuAD

[MedQuAD](https://github.com/abachaa/MedQuAD) contains 47,457 medical
question-answer pairs from 12 NIH websites, distributed as XML files, one
per document, each containing one or more `<QAPair>` elements with a
`<Question qtype="...">` and an `<Answer>`.

### Folders excluded

Folders `10_MPlus_ADAM_QA`, `11_MPlusDrugs_QA`, and `12_MPlusHerbsSupplements_QA`
have their `<Answer>` content **removed by the dataset's own authors** to
respect MedlinePlus copyright — those XML files exist but the answer text
is empty. Embedding empty answers would produce meaningless vectors, so
these three folders are skipped entirely.

### Folders used, and sampling plan

| Folder | Files used | Why |
|---|---|---|
| `1_CancerGov_QA` | all 116 | small enough to use in full |
| `3_GHR_QA` | 200 of 1086 (sampled) | full folder too large for a free-tier demo build |
| `5_NIDDK_QA` | all 157 | small enough to use in full |
| `6_NINDS_QA` | 200 of 277 (sampled) | kept close to full size, capped for consistency |
| `7_SeniorHealth_QA` | all 48 | small enough to use in full |
| `8_NHLBI_QA_XML` | all 88 | small enough to use in full |

Sampling uses `random.seed(42)` for reproducibility. On top of the
folder-level sampling, **each individual XML file contributes at most 3 QA
pairs** — some MedQuAD answers run past 4,000 words, and without this cap a
handful of very long documents would dominate the embedded set and skew
retrieval toward them. Answers under 20 characters are also dropped as
likely data artifacts.

This keeps the total dataset in the low thousands of QA pairs rather than
tens of thousands, which matters directly for free-tier Gemini embedding
quota — embedding the full 47,457-pair dataset in one run would very likely
exceed a free account's daily request cap.

### Preprocessing pipeline

`src/preprocess_medquad.py`:
1. Walks each of the 6 valid folders under `medquad/` (the cloned repo)
2. Parses each XML file with `xml.etree.ElementTree` (standard library —
   no extra XML dependency needed)
3. Extracts `<Focus>` (the disease/topic name), then for each `<QAPair>`
   extracts the question, answer, and `qtype` attribute
4. Cleans whitespace with a regex collapse (`re.sub(r"\s+", " ", text)`) —
   raw MedQuAD XML has inconsistent indentation and line breaks inside
   answer text
5. Writes `dataset/medquad_dataset.csv` with columns:
   `prompt, response, focus, source, qtype`
   (`prompt`/`response` match Task 1's column names exactly, so the same
   `CSVLoader(source_column="prompt")` call works with zero changes)

**Verified:** I built a realistic 4-QA-pair MedQuAD-format XML fixture
(matching the real repo's structure, confirmed against actual MedQuAD XML
content) and ran this script against it. Result: correctly capped at 3 QA
pairs from that file (the 4th, a "how to diagnose" question, was correctly
dropped by the per-file cap), correctly skipped a separate test case with
a too-short answer, and correctly printed a warning rather than crashing
when a folder was missing.

## Medical entity recognition

`src/medical_ner.py` — a curated keyword dictionary (~130 terms across
SYMPTOM / DISEASE / TREATMENT), not spaCy.

### Why keywords instead of spaCy

spaCy's general-purpose models (`en_core_web_sm` etc.) are trained on news
and web text — they recognize PERSON, ORG, GPE, DATE, and similar
categories, but have no built-in notion of "symptom" vs "disease" vs
"treatment." Asking `en_core_web_sm` to tag "fever" or "chemotherapy"
returns nothing useful, since those categories were never part of its
training objective. A real biomedical NER model (scispaCy, medspaCy,
BioBERT-based taggers) would do this correctly, but scispaCy's biomedical
models are 400MB+ downloads and can be finicky to install, and BioBERT-class
inference isn't available on a free tier without extra setup. A curated
keyword list directly satisfies the brief's own wording — "**basic** medical
entity recognition" — is fully deterministic, fully testable without a
model download, and runs in microseconds.

**Known limitation:** this approach only recognizes terms already in the
list. It will miss rare diseases, misspellings, or medical slang that
aren't in the ~130-term dictionary. It is not a substitute for a trained
biomedical NER model in a production system.

### How it works

- Case-insensitive whole-word/phrase matching (`\b...\b` regex boundaries)
- Longest-match-first: multi-word terms like "heart attack" or "type 2
  diabetes" are checked before shorter overlapping terms, and once a span
  is claimed by a longer match, shorter terms can't also match inside it
- Returns `[{"term": "fever", "category": "SYMPTOM"}, ...]`

**Verified, real output** from the module's built-in self-test
(`python src/medical_ner.py`):

```
I have a fever and a headache, could this be the flu?
 -> [{'term': 'headache', 'category': 'SYMPTOM'}, {'term': 'fever', 'category': 'SYMPTOM'}]

My doctor mentioned type 2 diabetes and prescribed insulin therapy.
 -> [{'term': 'type 2 diabetes', 'category': 'DISEASE'}, {'term': 'insulin therapy', 'category': 'TREATMENT'}]

What are the symptoms of a heart attack?
 -> [{'term': 'heart attack', 'category': 'DISEASE'}]

Nothing medical in this sentence at all.
 -> []
```

(Worth noting: while testing this, "heart attack" initially came back
mis-tagged as SYMPTOM rather than DISEASE — a heart attack is itself a
medical event, not merely a symptom of something else. That was a genuine
bug in the first draft of the keyword list, caught by actually running the
self-test rather than assuming the list was correct, and fixed before this
build was finalized.)

## Methodology

1. `preprocess_medquad.py` converts the cloned MedQuAD XML into
   `dataset/medquad_dataset.csv`
2. Clicking "Create Knowledgebase" in the UI loads that CSV, embeds every
   row with `gemini-embedding-001`, and saves a FAISS index to
   `faiss_index/`
3. A user question is embedded the same way; FAISS returns the top-4
   nearest QA pairs by meaning (not exact keyword match)
4. Those 4 are "stuffed" into a prompt template along with the question and
   sent to `gemini-2.5-flash`, which writes an answer grounded in that
   retrieved context — instructed to say "I don't have information about
   that" if the context doesn't actually contain the answer, and to avoid
   recommending specific medications or dosages
5. Separately, `medical_ner.py` scans the *question itself* (not the answer)
   for known symptom/disease/treatment terms and displays them as tags in
   the UI, so the user can see what the system recognized their question to
   be about

## What was verified vs. what wasn't

**Verified, with real captured output:**
- The XML → CSV preprocessing logic (fixture-tested against real MedQuAD
  XML structure, per-file cap and short-answer skip both confirmed working)
- The entity recognition module (self-test output above, including catching
  and fixing a real categorization bug)
- The FAISS/CSVLoader/RetrievalQA pipeline mechanics — tested end-to-end
  using a local deterministic stand-in for the embedding model (no network
  access to Google's API in the environment this was built in), confirming
  the CSV loads correctly with all 5 columns, FAISS indexes it, and
  metadata is preserved

**Not verified — needs to be run once with a real API key:**
- An actual Gemini LLM call and real embedding call
- The full MedQuAD clone + real preprocessing run at scale (the fixture
  test above used a small hand-built XML sample, not the real 47,457-pair
  repo)
- The Streamlit UI rendering in a browser
- Real sample Q&A output from the live system (see "Setup" below — once
  you've run it, capture 2–3 real question/answer pairs here for your
  submission)

## Real-world issue found and fixed after initial testing

The first version of this build called `FAISS.from_documents()` on all
2,366 rows in a single shot, with zero throttling. When actually run
against the real Gemini API (not the fake-embedding stand-in used for
local testing), this failed partway through with a `429 quota exceeded`
error — free-tier embedding is rate-limited per minute, and firing 2,366
back-to-back calls blows through that limit almost immediately.

Two fixes were applied in response:

1. **`preprocess_medquad.py`** now caps the total dataset at `MAX_TOTAL_ROWS
   = 100` (randomly sampled across all 6 folders, not just the first ones
   processed, so every folder still contributes). This was first set to 300,
   which avoided the outright 429 crash but still took a noticeably long
   time to build in practice — free-tier throughput turned out to be less
   generous than the published numbers suggested. 100 rows keeps a real
   one-time build meaningfully faster while still giving a genuine
   multi-folder sample across all 6 topic areas.
2. **`langchain_helper.py`**'s `create_vector_db()` now embeds in batches of
   20 with an 8-second delay between batches, and retries with backoff up
   to 5 times if a batch hits a 429. This was verified with a test harness
   that simulated a 429 failure on the second batch and confirmed the
   retry correctly recovered and all documents still ended up in the final
   index.

If your account has a higher free-tier or paid quota, `MAX_TOTAL_ROWS` and
`BATCH_SIZE`/`DELAY_BETWEEN_BATCHES` can be raised — check your actual
limit at [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)
before doing so.

## Project structure

```
task3/
├── medquad/                    # cloned MedQuAD repo (not committed — see .gitignore)
├── dataset/
│   └── medquad_dataset.csv     # generated by preprocess_medquad.py (not committed)
├── src/
│   ├── main.py                 # Streamlit UI
│   ├── langchain_helper.py     # retrieval/generation pipeline (same pattern as Task 1)
│   ├── medical_ner.py          # keyword-based entity recognition
│   └── preprocess_medquad.py   # MedQuAD XML → CSV converter
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```bash
cd task3

# 1. Clone MedQuAD (only needed once)
git clone https://github.com/abachaa/MedQuAD.git medquad

# 2. API key
cp .env.example .env
# edit .env and paste your real Gemini API key (the same AIzaSy... key
# used in Task 1 — do not use an "AQ."-prefixed key, see note below)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Preprocess the dataset
python src/preprocess_medquad.py
# generates dataset/medquad_dataset.csv

# 5. (Windows only, every session) fix the FAISS/OpenMP conflict
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 6. Run the app
streamlit run src/main.py
```

In the browser: click **"Create Knowledgebase"** once, wait for it to
finish, then ask a medical question.

### A note on API keys

If your Gemini API key starts with `AQ.` rather than `AIzaSy...`, it is not
a standard API key and will likely fail with 400/401 errors against this
code — this happened in Task 1's setup and was resolved by generating a
proper `AIzaSy...` key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey). The same
working key from Task 1 can be reused here.

## Medical disclaimer

The Streamlit sidebar displays a persistent warning that this is an
educational project, not medical advice, and that answers come from a
sampled subset of a public dataset plus an AI model — both of which can be
incomplete, outdated, or wrong. This is repeated as a caption under every
answer. This matters because MedQuAD, while sourced from NIH sites, is
being partially sampled and re-generated through an LLM here rather than
shown verbatim, and free-tier LLMs can still produce incorrect or
outdated-sounding phrasing even when grounded in real source text.
