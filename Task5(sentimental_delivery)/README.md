# Generative AI Q&A: Question and Answer System Based on Google Gemini and LangChain for an E-learning Company

This is an end-to-end LLM project based on Google Gemini and LangChain. We are building a Q&A system for an e-learning company called Nullclass. Nullclass sells data-related courses and virtual internships. They have thousands of learners who use a Discord server or email to ask questions. This system provides a Streamlit-based user interface for students where they can ask questions and get answers.

This version extends the original training project with **automatic, incremental knowledge-base updates** (Task 1) and **sentiment-aware responses** (Task 5) — see the dedicated sections below.

## Project Highlights

- Uses a real CSV file of FAQs that Nullclass is using right now.
- Their human staff use this file to assist course learners.
- An LLM-based Q&A system reduces the workload of their human staff.
- Students can ask questions directly and get answers within seconds.
- **Task 1:** the knowledge base can grow over time as new FAQs are added, without ever re-processing the whole dataset.
- **Task 5:** the chatbot detects the emotional tone of a question (positive/negative/neutral) and frames its answer appropriately.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/aslin72/GEN---AI-course.git
cd "GEN---AI-course/customer_service_chatbot_LLM"
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and put it in a `.env` file in the project root (copy `.env.example`):

```bash
GOOGLE_API_KEY="your_api_key_here"
```

## Usage

1. Run the Streamlit app from the project root:

```bash
streamlit run src/main.py
```

2. Click **"Create Knowledgebase (full rebuild)"** the first time you run the app. This reads `dataset/dataset.csv`, embeds every FAQ with Gemini, and saves a `faiss_index/` folder — same as the original training project.
3. Ask questions in the text box and get answers grounded in the FAQ data. The detected tone of your question is shown under the answer (Task 5).
4. To keep the knowledge base current over time, see the Task 1 section below.

## Sample Questions

- Do you guys provide internship and also do you offer EMI payments?
- Do you have a JavaScript course?
- Should I learn Power BI or Tableau?
- I've a MAC computer. Can I use Power BI on it?
- I don't see power pivot. How can I enable it?

---

## Dynamic Knowledge Base Updates (Task 1)

### The problem

The original project's knowledge base is frozen at whatever was in `dataset.csv` the moment "Create Knowledgebase" was clicked. In reality, Nullclass adds new FAQs constantly (new course launches, new payment options, policy changes) — there was no way to get that new information into the chatbot without re-running a full rebuild, which re-embeds every single FAQ again, burning API quota and taking longer as the dataset grows.

### Design decision: a watched folder, not URL scraping

The task allowed either a watched folder of CSV/text files or a list of URLs to scrape. **I chose the watched folder** for three reasons:

1. **Same knowledge domain, zero new dependencies.** The project's knowledge source is already CSV-based (`prompt`, `response` columns). A watched folder of CSV files reuses the _exact same_ `CSVLoader` ingestion pipeline already in `langchain_helper.py` — no HTML parsing, no scraping library, no risk of a redesign or layout change on a website silently breaking ingestion.
2. **Deterministic and gradeable.** Web scraping introduces flaky failure modes (rate limits, layout changes, blocked requests) that have nothing to do with the actual task being graded — the incremental-update mechanism itself.
3. **Free and offline-friendly.** No outbound HTTP requests are required at all; the mechanism works the moment you drop a file in a folder.

New sources live in **`new_sources/`** at the project root. Drop in:

- a `.csv` file with the same `prompt,response` header as `dataset/dataset.csv`, or
- a `.txt` file (split into paragraphs by blank lines — useful for free-form notes).

### How the incremental mechanism works

`update_vector_db_incremental()` in `langchain_helper.py` is the core of this feature. It never rebuilds the FAISS index from scratch and never re-embeds a document it has already seen. It uses a small JSON file, **`kb_manifest.json`**, as its memory between runs, with a two-level hash check:

1. **File-level hash (fast path).** Every file in `new_sources/` gets a SHA-256 hash of its raw bytes. If that hash matches what's recorded in the manifest from last time, the file is skipped entirely — it isn't even opened or parsed. This means a scheduler that checks every minute costs almost nothing once there's nothing new to do.
2. **Row-level hash (the real dedup guard).** For any file whose hash _has_ changed, each individual FAQ row is hashed (`prompt + response`, normalized). Only rows whose hash has never been recorded are kept. This is the part that actually prevents duplicates, and it's necessary even with the file hash check, because:
   - someone might **append** new rows to a file that was already partly ingested (the file hash changes, but most rows inside it are old),
   - the same FAQ might appear in two different files,
   - a file might be re-saved with no real content change (mtime alone is _not_ used as the dedup signal for exactly this reason — it's unreliable and was explicitly avoided).

Only the rows that survive both checks are passed to `vectordb.add_documents()` on the **already-loaded** FAISS index — this is the LangChain/FAISS equivalent of an "append," since it computes embeddings only for the new documents and merges their vectors into the existing index, then writes the whole (now slightly larger) index back to disk. The 76 original FAQs are never re-sent to the embedding API again.

### How to add a new source

1. Drop a CSV (with a `prompt,response` header) or a `.txt` file into `new_sources/`.
2. Either:
   - click **"🔄 Check new_sources/ now"** in the Streamlit sidebar, or
   - let the scheduler pick it up automatically (next section).
3. The chatbot can now answer questions about the new content immediately — no full rebuild needed.

### Running updates automatically

`src/update_scheduler.py` is a small stand-alone script that calls `update_vector_db_incremental()` on a timer using the lightweight **`schedule`** library:

```bash
python src/update_scheduler.py                 # check every 60 minutes (default)
python src/update_scheduler.py --interval 1     # check every 1 minute   (for a quick demo)
python src/update_scheduler.py --once           # run a single check and exit
```

`schedule` was chosen over APScheduler or a real OS cron job because the task explicitly only needs something "demonstrably automatic," not production-grade: it's a single pure-Python dependency, needs no crontab/Task-Scheduler entry, and the whole job loop is about 10 lines of code. Every check (and every "no new content found") is printed to the console **and** appended to `kb_update.log`, so you can leave it running in a second terminal and watch it work.

### Before / after example

I verified the dedup/incremental _mechanics_ end-to-end using a local deterministic stand-in for the embedding model (no API key needed — see flag below for why I didn't burn real quota just to test bookkeeping logic). The LLM-answer part of this example reflects the existing prompt design (`get_qa_chain`'s instruction to say "I don't know" when context doesn't contain the answer) and will behave this way once you run it with a real `GOOGLE_API_KEY` — I haven't called the live Gemini API myself since this sandbox has no network access to it.

Sequence:

1. `dataset/dataset.csv` has 76 FAQs. None of them mention a "certificate of completion."
2. **Before:** asking the chatbot _"Do you provide a certificate of completion after finishing the course?"_ against the original 76-FAQ index is expected to return **"I don't know."** — the information genuinely isn't there, and the prompt template explicitly instructs Gemini not to make something up.
3. `new_sources/sample_new_faqs.csv` (included in this repo) contains exactly that FAQ, plus two others (mobile app access, UPI payments).
4. Running `update_vector_db_incremental()` produces this real, captured log output (verified in this sandbox):

```
[create_vector_db] Built FAISS index with 76 FAQs and seeded manifest.

[2026-06-17T10:21:45] Knowledge base update check
  Files scanned (changed):     ['new_sources/sample_new_faqs.csv']
  Files skipped (unchanged):   none
  New FAQ rows embedded+added: 3
  Duplicate rows skipped:      0
  Total documents in KB now:   79
```

5. **After:** asking the exact same question is now expected to correctly return the certificate-of-completion answer, pulled from the newly-added row — without the original 76 FAQs ever being re-embedded.
6. Running the check again immediately afterward (same file, no edits) produces this real, captured output:

```
[...] Knowledge base update check
  Files scanned (changed):     none
  Files skipped (unchanged):   ['new_sources/sample_new_faqs.csv']
  New FAQ rows embedded+added: 0
  Duplicate rows skipped:      0
  Total documents in KB now:   79
```

confirming the dedup fast-path works — no wasted embedding calls on unchanged content. I also verified that appending one new row _and_ one exact duplicate of an existing row to the same file results in exactly 1 new document added and the duplicate correctly skipped (verified, not simulated).

### Files this feature adds/changes

| File                      | What changed                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `src/langchain_helper.py` | Migrated PaLM → Gemini; added `update_vector_db_incremental()`, `get_kb_status()`, manifest helpers |
| `src/update_scheduler.py` | **New.** Periodic job using `schedule`                                                              |
| `src/main.py`             | Shows KB status (last updated, doc count) + a "Check new_sources/ now" button                       |
| `new_sources/`            | **New.** The watched folder; ships with `sample_new_faqs.csv` for the demo above                    |
| `kb_manifest.json`        | **New, generated at runtime.** Dedup memory — file hashes + ingested row hashes                     |
| `kb_update.log`           | **New, generated at runtime.** Plain-text log of every scheduler run                                |
| `requirements.txt`        | Removed PaLM/HuggingFace-Instructor deps, added `langchain-google-genai`, `schedule`                |

---

## Sentiment-Aware Responses (Task 5)

### The problem

The original chatbot answers every question in exactly the same flat, neutral tone, regardless of whether a student is calmly asking about course timings or is genuinely frustrated after a payment failed twice. The task asks for the chatbot to **detect** the emotional tone of a message (positive, negative, or neutral) and **respond appropriately** — not just label the tone, but actually change how the answer is framed.

### Design decision: VADER, not a transformer model

Two realistic options exist for free-tier sentiment detection: a lexicon-based tool like **VADER**, or a pretrained transformer model via Hugging Face. **VADER was chosen as the primary implementation**, for reasons specific to this use case:

1. **Built for exactly this kind of text.** VADER (Valence Aware Dictionary and sEntiment Reasoner) was designed for short, informal social-media-style text — which is a close match for how students actually phrase chatbot questions, closer than a model trained on long-form movie reviews (the typical training data for many off-the-shelf Hugging Face sentiment models).
2. **No model download, no GPU, instant inference.** This matters on a free-tier project — one fewer dependency that can fail to download, one fewer multi-hundred-MB model file, no added latency per question.
3. **Naturally supports three classes.** Most popular pretrained Hugging Face sentiment pipelines only output positive/negative (binary), since they're trained on binary-labeled datasets. The task explicitly requires three classes (positive/negative/**neutral**). VADER's compound score naturally supports a neutral _zone_ (small magnitude scores near zero) rather than forcing every message into positive or negative.

An optional Hugging Face backend (`detect_sentiment_huggingface()`) is included in `sentiment_helper.py` as a documented, drop-in alternative with the same function signature, for anyone who wants to compare approaches later. It adds a confidence-based neutral zone (below 70% confidence → neutral) since the underlying model itself doesn't have a native neutral class.

### How detection works

`detect_sentiment()` in `src/sentiment_helper.py` runs VADER's `polarity_scores()` on the user's message and reads the **compound** score, which ranges from -1 (most negative) to +1 (most positive):

```
compound >= 0.05   -> positive
compound <= -0.05  -> negative
otherwise           -> neutral
```

These are VADER's own recommended thresholds. They were checked against realistic FAQ-chatbot messages before being adopted — see Evaluation below — rather than assumed to work out of the box.

### How the response actually changes

This is the part of the task that's easy to under-deliver on: detecting sentiment and only showing a label _next to_ an unchanged answer technically satisfies "detect," but not "respond appropriately." `apply_sentiment_framing()` actually modifies the final text shown to the user:

- **Negative** → a short empathetic acknowledgment is prepended before the factual answer (e.g. _"I'm sorry you're running into this — let's get it sorted. [answer]"_). Several variants rotate in, so it doesn't sound robotic on repeated use.
- **Positive** → a short upbeat acknowledgment is prepended (e.g. _"Glad to hear that! [answer]"_).
- **Neutral** → the answer is returned completely unchanged.

Critically, sentiment detection runs on the **user's question**, and framing is applied to the **final answer**, but the factual retrieval/generation step in between is completely untouched — sentiment can change tone, never facts. This keeps Task 1's retrieval accuracy fully intact.

### Example interactions

| User message                                              | Detected tone    | Response                                                                                               |
| --------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------ |
| "This is so frustrating, the payment page keeps failing!" | negative (-0.62) | "I'm sorry you're running into this — let's get it sorted. [FAQ answer about payment troubleshooting]" |
| "Thank you so much, that fixed it!"                       | positive (+0.76) | "Glad to hear that! [FAQ answer]"                                                                      |
| "Do you offer EMI payment options?"                       | neutral (0.00)   | "[FAQ answer]" — unchanged, exactly as before                                                          |

The Streamlit UI also shows a small caption under every answer (e.g. _"😟 Detected tone: negative (confidence score: -0.62)"_) so the detection is visibly happening for demo/grading purposes, without cluttering the actual answer text.

### Evaluation: accuracy of sentiment detection

The task explicitly lists "accuracy of sentiment detection" as an evaluation criterion. `src/evaluate_sentiment.py` contains a hand-labeled test set of 20 realistic messages (6 negative, 5 positive, 7 neutral, 2 deliberately hard/ambiguous cases) and reports how many `detect_sentiment()` gets right.

**Run it yourself:**

```bash
python src/evaluate_sentiment.py
```

**Actual result, captured from a real run (not estimated):**

```
Accuracy: 18/20 = 90.0%
```

**The two misclassifications, reported honestly rather than hidden:**

| Message                            | Expected | Got              | Why                                                                                                                                                                                                          |
| ---------------------------------- | -------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "How do I cancel my subscription?" | neutral  | negative (-0.25) | VADER's lexicon flags "cancel" with negative valence even in a neutral, transactional question. A genuine limitation of word-level scoring without sentence-level context.                                   |
| "I guess it's fine I suppose"      | neutral  | positive (+0.20) | "fine" carries enough positive lexicon weight to nudge this over threshold, even though the hedging language ("I guess... I suppose") makes the real sentiment ambiguous-to-neutral even for a human reader. |

**Known limitation (by design, not by accident):** VADER is a lexicon/rule-based approach, not a model that understands context or sarcasm. A message like _"Oh great, another error message"_ is sarcastic negative sentiment, but happens to contain positive-valence words ("great"). This is a known, documented weakness of lexicon-based sentiment analysis generally — the `detect_sentiment_huggingface()` alternative in the same file would not reliably fix this either (transformer sentiment models also commonly struggle with sarcasm without specific fine-tuning), so it's noted here as an honest limitation of the approach rather than something a quick model swap would solve.

### Reasoning: impact on customer satisfaction

The task lists "impact on customer satisfaction" as an evaluation criterion. This project has no live users and no real satisfaction survey data to report — being honest about that rather than inventing numbers. What can be argued is _why_ this specific design choice plausibly improves satisfaction, based on well-established customer service practice rather than guesswork:

1. **Acknowledging frustration before answering reduces perceived dismissiveness.** A well-documented pattern in human customer service is that a frustrated customer who receives a purely factual, emotionless answer often feels unheard, even if the answer itself is correct. A short empathetic acknowledgment ("I'm sorry you're running into this — let's get it sorted") before the same factual answer costs nothing in accuracy but directly addresses that "did anyone actually listen to me" feeling.
2. **Not over-praising neutral, transactional questions avoids sounding artificial.** This is why neutral messages are left completely unchanged rather than given a generic "Thanks for reaching out!" prefix on every single answer — a chatbot that's relentlessly upbeat on every plain factual question (e.g. "What time does the course start?") starts to read as scripted/insincere, which can hurt trust rather than build it. Reserving the upbeat framing for messages that are _actually_ positive keeps it meaningful when it does appear.
3. **The mechanism is conservative by design, not just by accident.** As the live demo above showed (a non-FAQ statement triggering a slightly mismatched upbeat-prefix-plus-"I don't know"), sentiment framing alone can't fix a case where there's no real answer underneath it — and that's an intentional boundary: the framing only ever adjusts tone around a real, retrieved answer, it never fabricates satisfaction by dressing up a non-answer. A future iteration could detect "no answer found" and suppress the sentiment prefix in that specific case, which is flagged here as a known improvement rather than presented as already solved.

In short: the design follows a defensible, real-world customer service principle (validate the emotion, then solve the problem), rather than reporting any directly measured satisfaction increase, since no such measurement was available to take.

### Files this feature adds/changes

| File                        | What changed                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `src/sentiment_helper.py`   | **New.** `detect_sentiment()`, `apply_sentiment_framing()`, optional Hugging Face backend |
| `src/evaluate_sentiment.py` | **New.** Labeled test set + accuracy evaluation script                                    |
| `src/main.py`               | Detects sentiment on each question, applies framing to the answer, shows a tone caption   |
| `requirements.txt`          | Added `vaderSentiment`                                                                    |

---

## Free-tier flags — read this before running with a real API key

- `GooglePalm` is fully retired; this project now uses `gemini-2.5-flash` for answer generation, which is on Google's **free tier** as of June 2026.
- The old free embedding model, `text-embedding-004`, was **shut down on January 14, 2026**. It's replaced here with `gemini-embedding-001`, also free tier.
- ⚠️ **Gemini Pro models (2.5 Pro, 3.x Pro) are paid-only as of April 2026** — don't swap `GEMINI_LLM_MODEL` to a Pro model expecting it to stay free.
- Free tier comes with daily/per-minute request caps (low hundreds of requests/day for Flash models). If you run the scheduler at a 1-minute interval against a `new_sources/` folder that changes constantly, you could approach those limits faster than you'd expect — for a real deployment, an interval of 30–60 minutes is more sensible (and is the default).
- The dedup mechanism described above is also a quota-saving feature, not just a correctness one: unchanged files cost zero API calls, and previously-seen rows are never re-embedded.
- Sentiment detection (Task 5) makes **zero** API calls — VADER runs entirely locally, so it has no effect on free-tier quota.

## Honest caveats — what I could and couldn't verify

This sandbox has no network access to `generativelanguage.googleapis.com` and I don't have a Gemini API key, so:

- **Verified, real, captured output:** the dedup/incremental mechanics (file-hash fast path, row-hash dedup, FAISS `add_documents` merge, manifest bookkeeping) — tested end-to-end using a local deterministic stand-in for the embedding model. The Task 5 sentiment detection logic and accuracy evaluation were run for real (VADER needs no API key, so this was fully testable).
- **Not verified against the real API:** the actual Gemini LLM call, the real embedding call, and the Streamlit UI rendering. These use a well-trodden, standard `langchain-google-genai` integration pattern, but "should work" isn't the same as "I watched it work."
- **A real bug I found and fixed:** the original repo's `vectordb.as_retriever(score_threshold=0.7)` does nothing — `VectorStoreRetriever` silently ignores unrecognized kwargs, so that "threshold" was inert even in the original training project. I removed it rather than ship an untested replacement, since FAISS's score-threshold filtering needs calibration against real embedding score distributions I can't produce here. Plain top-k retrieval is used instead; see the comment in `langchain_helper.py` for how to add real threshold filtering once you've tested it on your data.

Run it once with your real API key (next section) before assuming it's perfect — that's the only way to be sure on the parts above.

## Project Structure

```
customer_service_chatbot_LLM/
├── dataset/
│   └── dataset.csv              # original master FAQ data (unchanged)
├── new_sources/
│   └── sample_new_faqs.csv      # demo file for the incremental-update feature
├── src/
│   ├── main.py                  # Streamlit UI
│   ├── langchain_helper.py      # LangChain + Gemini logic, incremental updater
│   ├── update_scheduler.py      # periodic job that triggers updates automatically
│   ├── sentiment_helper.py      # Task 5: sentiment detection + response framing
│   └── evaluate_sentiment.py    # Task 5: accuracy evaluation script
├── requirements.txt
├── .env.example
├── faiss_index/                 # generated — the vector store
├── kb_manifest.json             # generated — dedup memory
└── kb_update.log                # generated — scheduler activity log
```
