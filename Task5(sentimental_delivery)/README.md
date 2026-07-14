# Sentiment-Aware Responses (Task 5)

This extends the Task 1 Nullclass FAQ chatbot to detect the emotional tone of a student's question (positive, negative, or neutral) and frame its response appropriately — while keeping the underlying factual answer completely unchanged.

## The brief, and how this addresses it

> "Integrate sentiment analysis into the chatbot to detect and respond appropriately to customer emotions during interactions. Expected Outcome: A chatbot that can recognize and address positive, negative, or neutral sentiments in user messages. Evaluation Criteria: Accuracy of sentiment detection, appropriateness of responses to different sentiments, impact on customer satisfaction."

All three named evaluation criteria are addressed directly, not just the two easy-to-measure ones — see below.

## Design decision: VADER, not a transformer model

VADER (Valence Aware Dictionary and sEntiment Reasoner) was chosen over a Hugging Face transformer model for three concrete reasons: it's built specifically for short, informal text (a close match for how students actually phrase chatbot questions), it needs no model download or GPU (important for a free-tier-conscious project), and it naturally supports a genuine three-way split (positive/negative/**neutral**) via a compound score, rather than most off-the-shelf transformer sentiment models which are binary-only and need an artificial confidence-based neutral zone bolted on.

## How the response actually changes (not just gets labeled)

`apply_sentiment_framing()` actually modifies the final text shown to the user — a negative message gets a short empathetic acknowledgment prepended before the factual answer, a positive message gets a brief upbeat acknowledgment, and a neutral message is returned completely unchanged. Critically, sentiment detection runs on the *question*, and framing is applied to the *final answer*, but the factual retrieval/generation step in between is completely untouched — sentiment can change tone, never facts.

## Real, verified test results

### Negative sentiment

<img width="1372" height="874" alt="image" src="https://github.com/user-attachments/assets/d1ea5e51-8eee-499f-8d90-5501bc5ed77b" />


A frustrated question correctly triggers an empathetic prefix before the factual FAQ answer, with the detected tone and confidence score shown transparently in the UI caption.

### Positive sentiment

<img width="1523" height="784" alt="image" src="https://github.com/user-attachments/assets/2d5d380a-4113-423c-91f1-2b982f13e361" />


A clearly positive message correctly triggers an upbeat prefix, confirming the framing logic works in both directions, not just for negative cases.

### Neutral sentiment

<img width="1316" height="904" alt="image" src="https://github.com/user-attachments/assets/8ababe01-a723-4e02-aa8a-a3a99eed1373" />


A plain factual question is answered with no prefix at all — confirming the system doesn't over-apply emotional framing to ordinary transactional questions, which would read as artificial/scripted if every single answer got a "Glad to help!" tacked on.

## Evaluation: accuracy of sentiment detection

![Accuracy evaluation](screenshots/test4_accuracy.png)

`src/evaluate_sentiment.py` runs a hand-labeled test set of 20 realistic messages (6 negative, 5 positive, 7 neutral, 2 deliberately hard/ambiguous cases) against `detect_sentiment()`.

**Real, captured result: 18/20 = 90.0% accuracy.**

**The two misclassifications, reported honestly rather than hidden:**
- "How do I cancel my subscription?" — expected neutral, detected negative. VADER's lexicon flags "cancel" with negative valence even in a neutral, transactional question — a genuine limitation of word-level scoring without full sentence context.
- "I guess it's fine I suppose" — expected neutral, detected positive. The hedging language makes the true sentiment ambiguous even for a human reader; "fine" carries enough positive lexicon weight to tip the score.

**Known limitation, by design, not by accident:** VADER cannot reliably detect sarcasm, since it scores words rather than understanding intent — a documented, general weakness of lexicon-based sentiment analysis, not something a quick model swap would necessarily fix either.

## Reasoning: impact on customer satisfaction

This project has no live users or real satisfaction data, so rather than inventing numbers, the design choice is defended on established customer service principle: acknowledging frustration before answering reduces the "did anyone actually listen to me" feeling that a purely factual, emotionless response can create, even when the answer itself is correct. Equally, neutral messages are deliberately left unframed — a chatbot that's relentlessly upbeat on every plain factual question starts to read as scripted and insincere, which can hurt trust rather than build it. The framing is conservative by design: it only ever adjusts tone around a real, retrieved answer, never fabricates warmth around a non-answer.

## Setup

1. `cd task5`
2. Create `.env` with `GOOGLE_API_KEY`
3. `pip install -r requirements.txt`
4. On Windows: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` each session
5. `streamlit run src/main.py`
6. Click "Create Knowledgebase" once
7. Run `python src/evaluate_sentiment.py` to see the accuracy evaluation yourself

## Project Structure

```
task5/
├── dataset/
│   └── dataset.csv
├── screenshots/
│   ├── test1_negative.png
│   ├── test2_positive.png
│   ├── test3_neutral.png
│   └── test4_accuracy.png
├── src/
│   ├── main.py
│   ├── langchain_helper.py
│   ├── sentiment_helper.py
│   └── evaluate_sentiment.py
├── requirements.txt
└── .env.example
```
