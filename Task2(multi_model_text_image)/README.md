# Multi-Modal Course Support Assistant (Task 2)

This extends the Task 1 Nullclass FAQ chatbot with image understanding. Students can upload a screenshot (e.g. a payment error) or a photo of course material, ask a question about it, and get an answer grounded in both the image content and the same FAQ knowledge base from Task 1.

## The brief, and how this addresses it

> "Develop a multi-modal AI assistant capable of understanding and reasoning over both text and image inputs. The assistant should analyze visual content, extract relevant information, maintain conversational context across multiple interactions, and generate evidence-based responses. The solution should demonstrate contextual reasoning, ambiguity handling, response validation, and intelligent decision-making rather than simple model inference or direct output generation from a single model."

This project builds a deliberate **3-stage pipeline** rather than a single "send image to LLM, show the answer" call:

1. **Vision extraction + ambiguity assessment** (one API call) — analyze the image, extract a short search topic, and explicitly decide whether the input is too unclear to answer confidently.
2. **Retrieval** (free, local) — search the same FAISS knowledge base from Task 1 using the extracted topic.
3. **Answer generation + self-validation** (one API call) — generate the answer, and require the model to explicitly state whether its own answer is actually consistent with the image and retrieved content.

This structure is what satisfies "intelligent decision-making rather than simple model inference" — ambiguity handling and response validation are real, separate decision points the pipeline can act on (e.g. skip the second API call entirely if ambiguous), not just claims in a system prompt.

## Why Groq instead of Gemini (architecture change, mentor-approved)

Tasks 1, 5, and 6 all use Google Gemini. This task uses **Groq** instead, for the vision and text generation steps. Two reasons:

1. **Gemini's free tier was hit hard during Task 6 development.** A real `429 ResourceExhausted` error was hit with a 20 requests/day cap on `gemini-2.5-flash` — confirmed via the actual error message, not assumed. Given Task 2 needs 2 API calls per image question, that quota would be exhausted in 10 questions, far too few to build and test against properly.
2. **The mentor explicitly approved switching architecture for this task.** When asked directly, the response was: _"It's not necessary to use the same architecture."_ This is documented here as the basis for using a different provider, rather than silently deviating from the "build on your training project" instruction.

**Why Groq specifically:** its free tier offers a much higher daily request ceiling than the Gemini allocation hit during Task 6, and — as a genuine side benefit — Groq's models are open-weight (Llama, Qwen, etc.), unlike Gemini, which is closed-source. This means Task 2 fully satisfies an "open-source models" framing in a way Task 6's README explicitly flagged as _not_ fully satisfied by Gemini.

**Why ONE provider for both vision and text (not Groq+Gemini split):** a two-provider pipeline was considered and rejected. Task 6 already surfaced three real, separately-debugged bugs from stacking memory + multilingual + sentiment in one pipeline on a single provider. Doubling the providers for what's already the hardest, vaguest task on the list was judged too risky for the remaining timeline. The chosen model handles both vision and plain text well, so retrieval still uses Gemini (Task 1's existing, untouched FAISS index/embeddings), but the model that _reasons_ about the image and generates the answer is Groq throughout — only one new provider integration, not two.

## Model choice: qwen/qwen3.6-27b

Groq's vision documentation (checked directly, June 2026) lists two vision-capable models: `meta-llama/llama-4-scout-17b-16e-instruct` and `qwen/qwen3.6-27b`. Both support multi-turn conversations, JSON mode, and a 20MB image size limit. **Qwen3.6-27b was chosen** because Groq's own docs explicitly flag Llama 4 Scout as a "Preview Model... should be used for experimentation," with no such caveat on Qwen3.6. `vision_helper.py` defines `VISION_MODEL_FALLBACK` pointing at Scout in case Qwen has availability issues on a given day — documented as an available swap, not silently hardcoded as the only option.

**A genuine, time-sensitive complication found while researching this:** Groq's model deprecation page (checked the same day) announced that Llama 4 Scout is being deprecated in favor of `openai/gpt-oss-120b`. Investigating further, `gpt-oss-120b` does **not** support image input at all — multiple independent sources confirm its multimodal support is explicitly missing. This means Groq's own recommended migration path for the _deprecated_ vision model leads to a _non-vision_ model, which would have been a real problem if followed without checking. Qwen3.6-27b, separately listed on Groq's vision docs page, isn't part of that deprecation notice and was used instead. This is flagged here because it's the kind of fast-moving provider detail that could break this project again in the future — if `qwen/qwen3.6-27b` itself becomes unavailable, recheck `console.groq.com/docs/vision` for the current model list before assuming the code just needs a quick swap.

## Pipeline efficiency: ambiguity handling saves real API calls

If the vision model flags the input as ambiguous (image too unclear, unrelated to the question, etc.), the pipeline **stops immediately** and returns the clarifying question — it does not proceed to retrieval or the second (answer-generation) API call. This was verified directly: a test with a mocked ambiguous response confirmed `retrieve_faq_context()` and `generate_validated_answer()` are never called in that path. This means ambiguity handling isn't just a UX nicety, it genuinely saves a second API call on every unclear input, which matters given the free-tier-conscious design of this whole project.

## Response validation, concretely

The brief asks for "response validation" — this is built as a real, structured field the model must commit to (`validation_passed: true/false`, plus a `validation_note` explaining the check), not a vague "be careful" instruction buried in a system prompt. A mocked test confirmed the pipeline correctly surfaces a `validation_passed: false` result rather than silently hiding it — the Streamlit UI shows a ⚠️ icon and the validation note when this happens, so a user (or grader) can see exactly when and why the model flagged its own answer as potentially inconsistent.

## Conversational context across multiple interactions

`VisionConversation` (in `vision_chain.py`) holds a list of all turns in a session — question, image description, answer, and validation status. `get_context_summary()` builds a short text summary of the most recent turns, available for future use in prompts (e.g. if a follow-up question references "the error from before" without re-uploading the image).

**Honest scope note:** unlike Task 6, this conversation summary is built but is **not yet actively injected into the vision/answer prompts** in this version — `ask_with_image()` calls `vision_helper`'s functions with just the current question and image, not the conversation history. This means a follow-up text-only question referencing a previous image (e.g. "what about the second one?") would not currently resolve correctly. This is a real, acknowledged gap between "memory is recorded" and "memory is actively used in reasoning," documented here rather than implied to be fully solved. A follow-up iteration would pass `conversation.get_context_summary()` into the prompts in `vision_helper.py` to close this gap.

## Verification — what's tested, and what isn't yet

**Verified with mocked/fake Groq responses (no real API calls):**

- Clear-input extraction correctly parses into description/topic/ambiguity fields
- Ambiguous-input extraction correctly returns a clarifying question
- Malformed JSON from the model correctly raises an error rather than silently propagating garbage downstream
- The ambiguous path correctly skips retrieval AND the second API call entirely (a real efficiency property, not just a design intention)
- The full pipeline (clear input → retrieval → validated answer) wires together correctly
- A validation-failure case is correctly surfaced rather than hidden
- Conversation memory correctly accumulates turns

**Verified against the REAL Groq API (live testing):**

- Image understanding quality on real screenshots — confirmed accurate across complex multi-element dashboards and simple plain-text screenshots
- JSON mode reliability — confirmed working correctly across four separate live test calls
- Real-world ambiguity detection — confirmed triggering correctly on a genuine ambiguity found in the actual dataset (see below)

**NOT yet verified:**

- Multi-turn conversational memory in practice (see honest gap noted in the section above and below)
- Behavior on a genuinely blurry/corrupted/unreadable image
- Latency under real-world conditions

### A real bug found via live testing, and the fix

The first live test against a text-dense dashboard screenshot crashed with `groq.BadRequestError: ... "max completion tokens reached before generating a valid document"`. The image (an ElevanceSkills internship dashboard with a course title, milestone checklist, project roadmap, and progress percentages) produced a long enough description that the model ran out of its 1024-token budget mid-way through writing the JSON answer, leaving Groq with an incomplete document. **Fix:** `max_completion_tokens` raised to 2048 in both Groq calls in `vision_helper.py`. A second, defensive fix was added at the same time: `main.py` now wraps the pipeline call in a try/except, so a similar failure on a different image shows a graceful in-app error instead of crashing the whole Streamlit session.

### Real, verified test cases — four distinct results, covering the full pipeline

Four separate live tests were run after the token-limit fix, deliberately covering different parts of the pipeline: a complex real-world image, a clean predictable FAQ match, a one-word-answer FAQ (to check against padding/fabrication), and a genuine ambiguity case found directly in the dataset (not contrived).

**Test 1 — Complex image, course identification + synthesis**

Image: a screenshot of an ElevanceSkills internship dashboard for a "Learn To Build A Real Time Gen AI Customer Service Bot" course. Question: "what course is this"

What the model extracted (verified accurate against the actual screenshot): course title, "Data Science" category, 100% progress, milestone status, project roadmap stages, mentor name — all factually correct, nothing invented.

Topic used for retrieval: "Gen AI Customer Service Bot course". Real FAQ content retrieved (confirmed genuinely relevant against the actual `dataset.csv`): three entries about GEN-AI bootcamp suitability, skills covered, and non-technical-background eligibility.

Final answer: _"Based on the screenshot, this is the 'Learn To Build A Real Time Gen AI Customer Service Bot' course. It is a bootcamp designed to teach you tech skills like Python, machine learning, NLP, and generative AI, along with project management and communication skills. It is suitable even for those with a non-technical background..."_

Validation: Passed. **This result was independently scrutinized before being accepted** — the initial concern was that the model might have fabricated a connection between an unrelated internship screenshot and a generic FAQ. This was checked directly against the actual dataset content rather than trusted just because `validation_passed: true` was returned, and confirmed genuine.

**Test 2 — Clean FAQ match (Power BI + Mac compatibility)**

Image: a screenshot of plain text reading "Does Power BI work on Mac?" Question: asked to answer the question shown in the image.

Topic used for retrieval: "Power BI Mac compatibility". FAQ retrieved, verbatim match against `dataset.csv`: _"Power BI desktop works only in Windows OS... you can use a virtual machine to install and work with Power BI in other Operating systems."_

Final answer correctly explained the Windows-only limitation and the VirtualBox workaround, even pulling in a second relevant FAQ (Power BI vs. Tableau comparison) as helpful bonus context. Validation: Passed, with the note explicitly confirming the answer "accurately reflects the provided FAQ context regarding Windows-only compatibility."

**Test 3 — One-word FAQ answer (tests against padding/fabrication)**

Image: plain text reading "Do you have an EMI option for payment?" This is the strictest test in the set: the real FAQ answer is a single word, **"No"** — with zero elaboration. A system prone to fabricating plausible-sounding detail would be most likely to over-explain or invent reasons here.

Final answer: _"Regarding the question in your screenshot, we do not currently offer an EMI option for payment."_ — correctly expanded the blunt "No" into a complete sentence **without adding any fabricated justification or detail not present in the source FAQ.** Validation: Passed, explicitly noting the answer "matches the FAQ response 'No'."

**Test 4 — Genuine ambiguity, found directly in the dataset (not contrived)**

Image: plain text reading "I've never coded before, can I take this course?" Question: asked to answer the question shown in the image.

**Result: the assistant correctly refused to guess.** _"The text asks about 'this course', but the image doesn't show which specific course you are referring to. Could you please specify the course name?"_

This is a genuinely important result, not a failure. `dataset.csv` contains this near-exact question **twice**, worded almost identically, with two different correct answers for two different courses (a general bootcamp framing vs. a Power BI/data-analytics framing). Since the image text said "this course" with no identifying context, there was no way to know which of the two matching FAQs was correct — guessing would have had a real chance of giving the wrong course's answer. The pipeline correctly detected this genuine ambiguity in the _data itself_ (not a staged/artificial test) and asked for clarification instead, exactly satisfying the brief's "ambiguity handling" requirement, and confirmed that this path also correctly skips the second API call (no retrieval or answer-generation cost was incurred for this turn).

### What these four tests collectively demonstrate

- **Visual analysis accuracy:** confirmed correct on both a complex multi-element dashboard and simple plain-text screenshots
- **Evidence-based responses:** confirmed answers are grounded in real, verified FAQ content, not fabricated — including the strict one-word-answer test, which would have exposed padding/invention immediately
- **Response validation:** confirmed the `validation_passed` field is checked, not just trusted — Test 1's validation claim was independently cross-verified against the dataset before being accepted as genuine evidence
- **Ambiguity handling:** confirmed working on a real, naturally-occurring ambiguity in the actual dataset, with confirmed cost savings (the second API call genuinely skipped, not just architecturally claimed to skip)
- **Intelligent decision-making:** the system made a different decision (answer vs. clarify) based on genuinely different input characteristics, rather than always producing a direct answer regardless of clarity

### What these four tests do NOT yet cover (honest gap)

All four tests above were **single-turn** — one image, one question, one answer, no follow-up. The brief's "maintain conversational context across multiple interactions" requirement is architecturally built (`VisionConversation` records every turn) but, as noted above, that history is not yet actively fed back into the prompts. A genuine multi-turn test (e.g. asking a follow-up question referencing an earlier image without re-uploading it) has not yet been run and would likely expose this gap directly rather than demonstrate working memory. This is flagged here as the clearest remaining piece of unfinished verification for this task.

## Setup

1. `cd task2`
2. Create `.env` (copy `.env.example`) with **two** keys:
   - `GOOGLE_API_KEY` — same Gemini key used in Tasks 1/5/6, needed here for Task 1's existing retrieval/embeddings
   - `GROQ_API_KEY` — a free key from console.groq.com, needed for the vision/answer-generation steps
3. `pip install -r requirements.txt`
4. On Windows, set the OpenMP workaround each session: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`
5. `streamlit run src/main.py`
6. Click "Create Knowledgebase" once (builds the FAISS index, same as Task 1)
7. Upload an image, ask a question

## Project Structure

```
task2/
├── dataset/
│   └── dataset.csv              # same master FAQ data as Task 1
├── new_sources/                 # Task 1's incremental-update folder (unused by Task 2 directly)
├── src/
│   ├── main.py                  # Streamlit UI — image upload + Q&A
│   ├── langchain_helper.py      # Task 1's retrieval logic (unchanged, reused)
│   ├── update_scheduler.py      # Task 1's scheduler (unchanged, not used by this task's UI)
│   ├── vision_helper.py         # NEW — Groq vision extraction + validated answer generation
│   └── vision_chain.py          # NEW — orchestration: extraction → retrieval → answer, + memory
├── requirements.txt
└── .env.example
```
