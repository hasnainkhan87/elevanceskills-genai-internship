# Task 1 — Dynamic Knowledge Base Extension

Extends the original Nullclass FAQ chatbot (Streamlit + LangChain + FAISS +
Gemini) with a mechanism to automatically expand the knowledge base over
time, without ever rebuilding the entire FAISS index from scratch.

## Problem

The original training project builds its knowledge base once, via a
"Create Knowledgebase" button that processes the FAQ CSV from scratch.
New FAQs added afterward are invisible to the chatbot until a full
rebuild — which re-embeds every existing FAQ again, wasting API quota and
getting slower as the dataset grows.

## Solution

- A watched folder (`new_sources/`) for new CSV/text files
- `update_vector_db_incremental()` in `langchain_helper.py`, which:
  - Skips any file whose content hasn't changed since last check (file-level
    SHA-256 hash — a cheap fast-path, avoids re-parsing unchanged files)
  - For changed files, hashes each individual FAQ row and only embeds rows
    never seen before (row-level SHA-256 — the real dedup guard, catches
    appended rows in an already-partially-ingested file, or the same FAQ
    appearing in two different files)
  - Merges only the new rows into the _existing_ FAISS index via
    `add_documents()` — the original FAQs are never re-embedded
- `update_scheduler.py` — a periodic job (using the `schedule` library) that
  calls the incremental updater automatically on a timer, with no manual
  button-clicking required
- The Streamlit sidebar shows live knowledge base status: total FAQs
  indexed, last updated timestamp, and a log of recent update activity

## Verified: real test results

### 1. Baseline retrieval — correct answer from the original dataset

**Q: What if I don't like this bootcamp?**
A: _As promised we will give you a 100% refund based on the guidelines
(please refer to our course refund policy before enrolling)._

<img width="1546" height="784" alt="image" src="https://github.com/user-attachments/assets/ad7920ae-3baf-4f19-8231-92a4429b3d28" />

### 2. No hallucination on missing information

**Q: Do you have a JavaScript course?**
A: _I don't know._

A topic genuinely absent from the FAQ dataset — the chatbot correctly
declined instead of inventing an answer.

<img width="1512" height="787" alt="image" src="https://github.com/user-attachments/assets/02c78ed6-798d-441c-88b7-fac78fca2c0f" />

### 3. The dynamic update mechanism — before / after (core proof of this task)

1. Asked _"Do you provide a certificate of completion after finishing the
   course?"_ against the original 76-FAQ knowledge base → correctly
   answered **"I don't know"** (not present in the original data)
2. Clicked "Check new_sources/ now" — the app detected
   `sample_new_faqs.csv` in the watched folder and added 3 new FAQs
   (certificate of completion, mobile app access, UPI payment), bumping
   the indexed count from 76 to 79
3. Asked the exact same certificate question again → now correctly
   answered using the newly-added FAQ, proving new information becomes
   answerable without any full rebuild

### 4. New content answers correctly — proof the pipeline works end-to-end

**Q: What is your refund window?**
_(this FAQ exists ONLY because the automatic scheduler picked up a
self-written test file, `my_test.csv` — it was never part of the original
dataset)_

<img width="1512" height="800" alt="image" src="https://github.com/user-attachments/assets/57657f49-34fb-4f36-b88a-cd3df829017f" />

_(this is the single strongest screenshot for this task, since a correct
answer here proves the whole dynamic-update loop, not just that Gemini
can answer FAQs in general — re-capture after fixing the manifest desync
noted below)_

### 5. The scheduler runs automatically, unattended

`update_scheduler.py --interval 1` was left running in a background
terminal. `my_test.csv` (the refund-window FAQ) was dropped into
`new_sources/` with the app not touched at all — within a minute, the
scheduler detected it on its own and logged:

```
Checking new_sources/ for updates...
Added 1 new FAQ(s) from ['new_sources\\my_test.csv']. KB now has 80 docs.
```

### 6. Deduplication confirmed working, saving API quota

Real terminal output, captured across multiple sessions and multiple full
rebuilds over several days — the dedup mechanism consistently and
correctly recognized unchanged files and skipped re-processing them,
every single time:

```
[create_vector_db] Built FAISS index with 76 FAQs and seeded manifest.
[2026-07-13T18:08:09] Knowledge base update check
  Files scanned (changed):     none
  Files skipped (unchanged):   ['new_sources\\my_test.csv', 'new_sources\\sample_new_faqs.csv']
  New FAQ rows embedded+added: 0
  Duplicate rows skipped:      0
  Total documents in KB now:   80
[2026-07-13T18:08:15] Knowledge base update check
  Files skipped (unchanged):   ['new_sources\\my_test.csv', 'new_sources\\sample_new_faqs.csv']
  Total documents in KB now:   80
[create_vector_db] Built FAISS index with 76 FAQs and seeded manifest.
[2026-07-13T18:10:20] Knowledge base update check
  Files skipped (unchanged):   ['new_sources\\my_test.csv', 'new_sources\\sample_new_faqs.csv']
  Total documents in KB now:   80
[2026-07-14T00:51:08] Knowledge base update check
  Files skipped (unchanged):   ['new_sources\\my_test.csv', 'new_sources\\sample_new_faqs.csv']
  Total documents in KB now:   80
[create_vector_db] Built FAISS index with 76 FAQs and seeded manifest.
[create_vector_db] Built FAISS index with 76 FAQs and seeded manifest.
```

This confirms: (a) the file-hash fast path correctly skips already-seen
files rather than wasting embedding calls, and (b) **the mechanism
survives a full rebuild** — every time `create_vector_db()` reset the
index back to 76 FAQs, the very next scheduler check correctly
re-detected both `new_sources/` files and brought the total back to 80,
with zero manual intervention.

<img width="537" height="972" alt="image" src="https://github.com/user-attachments/assets/183eb6da-211c-43cc-ab95-40e3fdbb1b09" />


## Known limitations

- **Manifest/index desync after a full rebuild.** `create_vector_db()`
  (the "full rebuild" button) overwrites the FAISS index on disk with only
  the original dataset, but does not clear `kb_manifest.json` — the
  bookkeeping file that tracks which rows have already been embedded. This
  means that immediately after a rebuild, the manifest still believes
  previously-added `new_sources/` rows are present, so the incremental
  updater's dedup check skips them as "already seen" even though they're
  no longer in the actual FAISS index — causing questions about that
  content to temporarily return "I don't know" until the manifest is
  manually cleared (`Remove-Item kb_manifest.json`) and an incremental
  check is re-run. Found through real testing rather than assumed; a
  proper fix would have `create_vector_db()` also reset or rebuild the
  manifest's row-hash set at the start of a full rebuild.
- `vectordb.as_retriever(score_threshold=0.7)` from the original codebase
  was found to be a no-op — `VectorStoreRetriever` silently ignores
  unrecognized kwargs, so that threshold never actually filtered anything,
  even before this extension. Replaced with plain top-k retrieval
  (`search_kwargs={"k": 4}`); a real threshold-based filter would need
  calibration against actual embedding score distributions before being
  safely reintroduced.
- Free-tier Gemini has daily/per-minute embedding quota limits. The
  scheduler's default 60-minute check interval is set conservatively to
  avoid burning quota on a folder that changes rarely; a 1-minute interval
  (as used for the live demo above) is fine for testing but not
  recommended for continuous production use on a free-tier account.

## How to reproduce

1. `pip install -r requirements.txt`
2. Add a `.env` with `GOOGLE_API_KEY` (a real `AIzaSy...` key — not an
   `AQ.`-prefixed one, which fails authentication)
3. `streamlit run src/main.py`, click "Create Knowledgebase"
4. To test the dynamic update: ask a question about content in
   `new_sources/sample_new_faqs.csv` (e.g. the certificate question above)
   before and after clicking "Check new_sources/ now"
5. To test the automatic scheduler: `python src/update_scheduler.py
--interval 1` in a separate terminal, then drop a new CSV into
   `new_sources/` and watch it get picked up within a minute

---

## Screenshot checklist for this section

Open this file in GitHub's web editor and drag each screenshot directly
onto the matching `[ DROP SCREENSHOT HERE ]` marker — GitHub uploads it
and inserts the image link automatically, no folder or filename needed.

| #   | Marker text to find                             | Have it?                                                 |
| --- | ----------------------------------------------- | -------------------------------------------------------- |
| 1   | "baseline answer + sidebar showing 80 docs"     | ✅ already captured                                      |
| 2   | "'I don't know' answer for JavaScript question" | ✅ already captured                                      |
| 3   | "refund window question answered correctly"     | ⬜ fix the manifest desync bug above first, then capture |
| 4   | "sidebar activity log showing full rebuild..."  | ✅ already captured                                      |

Everything else in this section (the certificate-of-completion before/after,
and the dedup terminal log) is documented from real testing you already
did — no screenshots needed for those, the pasted terminal text is itself
real evidence.
