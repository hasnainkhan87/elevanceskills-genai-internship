# Multilingual Conversation Support with Memory (Task 6)

This extends the Task 5 sentiment-aware Nullclass FAQ chatbot with support for Hindi, Spanish, and French — automatic language detection, conversation memory across turns and language switches, and translated sentiment responses — while reusing the exact same FAISS knowledge base from Task 1.

## The brief, and how this addresses it

> "Extend the existing chatbot to support multilingual conversations across at least three additional languages while preserving context, intent, and conversational continuity throughout language switches. The assistant should automatically identify language, manage mixed-language inputs within the same conversation, resolve ambiguous queries across languages, and maintain consistent responses regardless of the language used. The solution should demonstrate cross-lingual reasoning, context retention, and intelligent handling of multilingual interactions using open-source models and frameworks."

**Languages supported (beyond English):** Hindi, Spanish, French.

## Design decision: multilingual embeddings, not translation

`gemini-embedding-001` — the same embedding model already used for Task 1/5's FAISS index — natively supports 100+ languages, confirmed against Google's own documentation before committing to this approach. This means **the existing FAISS index did not need to be rebuilt**: a Hindi or French question can be embedded directly and still retrieve semantically relevant English FAQ content, with no translation step. The only place a language instruction is still needed is telling the LLM to *write* its answer in the user's language — retrieval finding the right content doesn't automatically make the LLM reply in Hindi.

## Three real bugs found via live testing, and their fixes

**Bug 1 — Non-deterministic, and genuinely wrong, language detection.** Asking "Do you have a JavaScript course?" — plain English — was answered in **French**. Diagnosed to two compounding issues in `langdetect`: (a) it's non-deterministic by default (the same input returned different results across runs), and (b) even once fixed, the sentence still scored French ahead of English despite containing zero French-specific characters. Fixed with `DetectorFactory.seed = 0` plus a targeted check: if French/Spanish is detected but the text has no accented characters and English is a close alternative, prefer English.

**Bug 2 — Language instruction lost across turns.** An earlier design appended the language instruction to the question text, which passed through a "condense question" step with no instruction to preserve it — confirmed to work for Hindi but silently fail for French in the same session. Fixed by baking the instruction directly into the answer-generation prompt's literal template text instead, bypassing the condense step entirely.

**Bug 3 — Sentiment prefixes stayed in English.** VADER's sentiment lexicon is English-only, confirmed directly: a clearly negative Hindi message and positive French message both scored "neutral" until a translation step was added purely for the sentiment check (not for retrieval). Additionally, the empathetic/upbeat *prefixes* themselves were still hardcoded English even after the base sentiment fix — fixed by translating the prefix text itself into the detected language, with a cache to avoid repeat translation calls.

## Real, verified test results

### Test 1 — Hindi, negative sentiment, fully translated response

<img width="1372" height="874" alt="image" src="https://github.com/user-attachments/assets/2f94b75a-938d-438d-9440-f4d2719e0e9d" />


Question: "यह कोर्स बहुत निराशाजनक है" (this course is very disappointing). The pipeline correctly detected negative sentiment (via translation-assisted scoring), detected Hindi as the language, and returned **both** a translated empathetic prefix and a real, correctly-retrieved FAQ answer — all in Hindi.

### Test 2 — French, positive sentiment, translated prefix

<img width="1316" height="904" alt="image" src="https://github.com/user-attachments/assets/0a4b19d2-b176-4b2c-ae3f-5627866c3221" />


Confirms the same translated-prefix mechanism works for a second language, not just Hindi — ruling out a single-language coincidence.

### Test 3 — Mixed-language input, genuinely handled (not just flagged)

<img width="1316" height="908" alt="image" src="https://github.com/user-attachments/assets/ee46f2b5-87fc-41a1-a56f-240c0bd9a965" />


Question: "क्या आपके पास Power BI course है?" — Hindi grammar with an English product name embedded mid-sentence, the way a real bilingual student would type it. The pipeline correctly flagged this as mixed-language input **and** returned a detailed, accurate, Power-BI-specific answer in fluent Hindi — confirming the multilingual embeddings correctly matched the mixed-script query against English-language FAQ content, not just that the system didn't crash.

### Test 4 — Confirming the language-detection bug fix

<img width="1316" height="904" alt="image" src="https://github.com/user-attachments/assets/41b4db60-c391-4f31-8b07-f62aa548e2fe" />


Re-running the exact question that originally triggered Bug 1 ("Do you have a JavaScript course?"), now correctly showing "Detected language: English" instead of the earlier French misfire.

## Known, honestly disclosed limitations

- **Short-text language detection isn't perfect.** A separate short Spanish question was misdetected as German — a different failure mode from Bug 1, not fully resolved by the same fix, and reported here rather than hidden.
- **Gemini itself is closed-source**, unlike the rest of this project's stack (LangChain, FAISS, langdetect, VADER, deep-translator, all open-source). If "open-source models" is read strictly to include the core LLM, this project doesn't fully satisfy that clause — disclosed plainly rather than glossed over.
- **True word-level code-switching detection isn't implemented** — mixed-script flagging catches the common Hindi+Latin-script case via Unicode ranges, not a general-purpose code-switching detector.

## Setup

1. `cd task6`
2. Create `.env` with `GOOGLE_API_KEY`
3. `pip install -r requirements.txt`
4. On Windows: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` each session
5. `streamlit run src/main.py`
6. Click "Create Knowledgebase" once
7. Ask in English, Hindi, Spanish, or French

## Project Structure

```
task6/
├── dataset/
│   └── dataset.csv
├── screenshots/
│   ├── test1_hindi_negative.png
│   ├── test2_french_positive.png
│   ├── test3_mixed_language.png
│   └── test4_english_bug_fixed.png
├── src/
│   ├── main.py
│   ├── langchain_helper.py
│   ├── sentiment_helper.py
│   ├── multilingual_helper.py
│   ├── multilingual_chain.py
│   └── demo_multilingual.py
├── requirements.txt
└── .env.example
```
