# Multi-Modal Course Support Assistant (Task 2)

This extends the Task 1 Nullclass FAQ chatbot with image understanding. Students can upload a screenshot (e.g. a payment error) or a photo of course material, ask a question about it, and get an answer grounded in both the image content and the same FAQ knowledge base from Task 1.

## The brief, and how this addresses it

> "Develop a multi-modal AI assistant capable of understanding and reasoning over both text and image inputs. The assistant should analyze visual content, extract relevant information, maintain conversational context across multiple interactions, and generate evidence-based responses. The solution should demonstrate contextual reasoning, ambiguity handling, response validation, and intelligent decision-making rather than simple model inference or direct output generation from a single model."

This project builds a deliberate **3-stage pipeline** rather than a single "send image to LLM, show the answer" call:

1. **Vision extraction + ambiguity assessment** (one API call) — analyze the image, extract a short search topic, and explicitly decide whether the input is too unclear to answer confidently.
2. **Retrieval** (free, local) — search the same FAISS knowledge base from Task 1 using the extracted topic.
3. **Answer generation + self-validation** (one API call) — generate the answer, and require the model to explicitly state whether its own answer is actually consistent with the image and retrieved content.

## Why Groq instead of Gemini (architecture change, mentor-approved)

Tasks 1, 5, and 6 all use Google Gemini. This task uses **Groq** instead, for the vision and text generation steps. Two reasons:

1. **Gemini's free tier was hit hard during Task 6 development** — a real `429 ResourceExhausted` error, 20 requests/day cap on `gemini-2.5-flash`. Task 2 needs 2 API calls per image question, so that quota would exhaust in 10 questions.
2. **The mentor explicitly approved switching architecture for this task**: *"It's not necessary to use the same architecture."*

Retrieval still uses Gemini (Task 1's existing FAISS index/embeddings, unchanged); only the reasoning/generation layer uses Groq (`qwen/qwen3.6-27b`, chosen over the alternative Llama 4 Scout since Groq's own docs flag Scout as a "Preview Model").

## Two real bugs found via live testing, and their fixes

**Bug 1 — Token limit:** the first live test against a text-dense dashboard screenshot crashed with `max completion tokens reached before generating a valid document`. Fixed by raising `max_completion_tokens` from 1024 to 2048.

**Bug 2 — Wrong exception type caught:** a later live test crashed with `groq.BadRequestError: json_validate_failed`. This happens server-side, before any local JSON parsing, so the original retry logic (which only caught `json.JSONDecodeError`) never triggered. Fixed by wrapping the API call itself in `try/except groq.BadRequestError`, verified with mocked tests using the actual exception type.

## Real, verified test — and a genuine limitation found through it

### Turn 1: establishing context

![Power BI question](screenshots/test1_powerbi_question.png)

<img width="1456" height="818" alt="image" src="https://github.com/user-attachments/assets/3dbd2476-7ad3-468c-97e9-748e9961ac05" />

Image: a screenshot of an unrelated ElevanceSkills internship dashboard (a "Gen AI Customer Service Bot" course page).
Question: "Do you have a Power BI course?"

The model correctly recognized the question was about Power BI — a completely different topic from what's shown in the image — and answered using real, relevant FAQ content about the course's project-based structure and upgrade policy, while still acknowledging the image's actual content. Validation: Passed, with the note confirming the answer is "fully grounded in the provided FAQ context."

### Turn 2: a vague follow-up, with the same image still attached

<img width="1512" height="804" alt="image" src="https://github.com/user-attachments/assets/4b074d57-391f-4939-8715-513111700804" />


Question: "What about the price?" — deliberately vague, testing whether the assistant remembers Turn 1 was about Power BI, without repeating the course name.

**Result, reported honestly:** the assistant answered about the *image's* course (the Gen AI Customer Service Bot) instead of correctly carrying forward that Turn 1 was about Power BI. This is a genuine, understood limitation, not a bug that was missed:

**Why this happens:** every turn in this pipeline re-sends whatever image is currently attached. The vision model has two competing signals on a follow-up — the text-based conversation history (memory) and the literal image in front of it. On Turn 1, the question clearly named an unrelated topic, so the model correctly treated the image as background. On Turn 2, the vague question gave the model nothing to anchor to *except* the image, so it defaulted to the image over the text-based memory.

**This is disclosed as an honest, unresolved architectural limitation** rather than patched over with a prompt tweak. The conversation-memory mechanism itself is correctly built and wired (verified with mocked tests confirming context reaches both API calls), but this live test shows the mechanism doesn't reliably win out over a re-attached image when the follow-up is ambiguous. A real fix would require either allowing image-less follow-up questions, or having the pipeline explicitly judge each turn whether the current image is actually relevant before using it — both meaningful architectural changes reserved for a future iteration.

## What's verified vs. not yet verified

**Verified with mocked/fake Groq responses:** clear-input extraction, ambiguous-input handling, malformed JSON handling (both exception types), the ambiguity short-circuit saving a real API call, validation-failure surfacing, conversation memory accumulation, context-passing mechanics, backward compatibility of all changes.

**Verified against the real Groq API (live testing):** image understanding accuracy, JSON mode reliability, both real bugs found and fixed, and the multi-turn memory limitation described above.

**Not yet verified:** behavior on a genuinely blurry/corrupted image; latency under real-world load.

## Setup

1. `cd task2`
2. Create `.env` with **two** keys:
   - `GOOGLE_API_KEY` — same Gemini key used in Tasks 1/5/6
   - `GROQ_API_KEY` — a free key from console.groq.com
3. `pip install -r requirements.txt`
4. On Windows: `$env:KMP_DUPLICATE_LIB_OK="TRUE"` each session
5. `streamlit run src/main.py`
6. Click "Create Knowledgebase" once
7. Upload an image, ask a question

## Project Structure

```
task2/
├── dataset/
│   └── dataset.csv
├── screenshots/
│   ├── test1_powerbi_question.png
│   └── test2_followup_limitation.png
├── src/
│   ├── main.py
│   ├── langchain_helper.py
│   ├── vision_helper.py
│   └── vision_chain.py
├── requirements.txt
└── .env.example
```
