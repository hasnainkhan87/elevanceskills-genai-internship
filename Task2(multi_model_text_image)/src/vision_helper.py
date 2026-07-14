"""
vision_helper.py
=================
Task 2 — Multi-modal AI assistant (text + image) for the Nullclass
FAQ chatbot.

WHAT THIS DOES (matches the task brief):
  1. Analyze an uploaded image (screenshot of an error, photo of course
     material, a chart, etc.) and extract relevant information.
  2. Assess whether the image/question pair is ambiguous BEFORE
     attempting to answer — if it is, return a clarifying question
     instead of guessing.
  3. Hand off a short, retrieval-friendly topic string so the SAME
     FAISS knowledge base from Task 1 can be queried.

WHY GROQ (not Gemini, unlike Tasks 1/5/6):
  Gemini's free tier was hit hard during Task 6 development (a 20
  requests/day cap on gemini-2.5-flash, confirmed via a real 429 error).
  The mentor explicitly said switching architecture/provider is fine for
  this task. Groq's free tier offers a much higher daily request
  ceiling, and its vision models are genuinely open-weight, unlike
  Gemini, which is closed-source.

CONVERSATION CONTEXT SUPPORT: both extract_image_content() and
generate_validated_answer() accept an optional conversation_context
parameter (default ""), so a multi-turn conversation can resolve
ambiguity using prior turns rather than re-asking for information
already given.

RETRY LOGIC (extract_image_content only): a real failure was found via
live testing — Groq's API itself rejected a request with
`groq.BadRequestError` ("json_validate_failed", empty failed_generation).
An earlier version of this retry loop only caught json.JSONDecodeError
(a LOCAL parsing failure), which structurally could never catch this —
the real error happens at the API call itself, server-side, before any
local parsing is attempted. Fixed by wrapping the API call in a
try/except groq.BadRequestError as well. Verified with mocked tests
using the exact real exception type.
"""

import os
import json
import groq
from groq import Groq

VISION_MODEL_PRIMARY = "qwen/qwen3.6-27b"
VISION_MODEL_FALLBACK = "meta-llama/llama-4-scout-17b-16e-instruct"


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. Add it to your .env file "
            "(see .env.example) — this is a SEPARATE key from GOOGLE_API_KEY, "
            "since Task 2 uses Groq instead of Gemini for vision."
        )
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Stage 1: Vision extraction + ambiguity assessment (ONE Groq call, with retry)
# --------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are analyzing an image submitted by a student to an e-learning company's support chatbot, alongside their question.

{conversation_context_block}

Look at the image carefully and respond with a JSON object with EXACTLY these fields:

{{
  "description": "A factual, detailed description of what is actually visible in the image — text, error messages, charts, course names, UI elements, etc. Be specific, not generic.",
  "extracted_topic": "A SHORT phrase (2-6 words) capturing the main subject to search a course-FAQ knowledge base with — e.g. 'payment page error', 'Power BI course', 'certificate download'. If nothing in the image relates to courses/payments/learning, set this to null.",
  "is_ambiguous": true or false — true if the image is unclear, unrelated to the question, too low quality to read, OR if it's genuinely unclear what the user wants to know about it EVEN AFTER considering the conversation context above (if any was provided),
  "clarifying_question": "If is_ambiguous is true, a SHORT, specific question to ask the user to resolve the ambiguity. If is_ambiguous is false, set this to null."
}}

IMPORTANT: if conversation context was provided above and it resolves what would otherwise be ambiguous (e.g. the user previously specified which course they meant, and this new image/question is clearly a follow-up about that same course), use that context to set is_ambiguous to false rather than asking again for information already given.

Respond with ONLY the JSON object, no other text, no markdown code fences."""


def extract_image_content(
    image_base64: str,
    user_question: str,
    image_mime_type: str = "image/jpeg",
    conversation_context: str = "",
    max_retries: int = 2,
) -> dict:
    """
    Stage 1 of the pipeline: send the image + question to Groq's vision
    model, get back a structured extraction.

    Returns:
        {
            "description": str,
            "extracted_topic": str or None,
            "is_ambiguous": bool,
            "clarifying_question": str or None,
            "raw_model_response": str,
        }

    Raises:
        ValueError if the model still fails after all retry attempts.
    """
    client = get_groq_client()

    if conversation_context:
        context_block = (
            f"CONVERSATION CONTEXT FROM EARLIER IN THIS SESSION (use this to "
            f"resolve ambiguity if it helps — e.g. if the course/topic was "
            f"already established earlier, don't ask for it again):\n"
            f"{conversation_context}\n"
        )
    else:
        context_block = "(This is the first turn in the conversation — no prior context.)"

    last_error = None
    raw_text = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL_PRIMARY,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT.format(conversation_context_block=context_block)
                                + f"\n\nThe user's question is: {user_question}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                temperature=0.2,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
        except groq.BadRequestError as e:
            # The real failure hit live: Groq's own server rejects the
            # response before returning it. json.loads() is never
            # reached. Catching it HERE, around the API call itself,
            # is what makes the retry actually work for this case.
            last_error = e
            raw_text = "(no response body — Groq rejected the request server-side)"
            continue

        raw_text = response.choices[0].message.content

        try:
            parsed = json.loads(raw_text)
            return {
                "description": parsed.get("description", ""),
                "extracted_topic": parsed.get("extracted_topic"),
                "is_ambiguous": bool(parsed.get("is_ambiguous", False)),
                "clarifying_question": parsed.get("clarifying_question"),
                "raw_model_response": raw_text,
            }
        except json.JSONDecodeError as e:
            last_error = e
            continue

    raise ValueError(
        f"Vision model did not return valid JSON after {max_retries + 1} attempts. "
        f"Last raw response: {raw_text!r}. Original error: {last_error}"
    )


# --------------------------------------------------------------------------
# Stage 3: Answer generation + self-validation (ONE Groq call)
# --------------------------------------------------------------------------
ANSWER_PROMPT_TEMPLATE = """You are a helpful course support assistant. You have an image description and (if available) relevant FAQ content. Generate a helpful answer, AND validate that your answer is actually consistent with what was seen in the image before finalizing it.

{conversation_context_block}

IMAGE DESCRIPTION: {image_description}

USER'S QUESTION: {question}

RELEVANT FAQ CONTEXT (may be empty if nothing matched): {faq_context}

Respond with a JSON object with EXACTLY these fields:
{{
  "answer": "Your helpful answer to the user, grounded in the FAQ context and consistent with the image description. If conversation context was provided above, use it to keep your answer consistent with what was already discussed. If the FAQ context doesn't contain relevant information, say so honestly rather than guessing.",
  "validation_passed": true or false — true if your answer is genuinely consistent with and grounded in the image description, FAQ context, AND any conversation context provided, false otherwise,
  "validation_note": "A short note explaining your validation check"
}}

Respond with ONLY the JSON object, no other text, no markdown code fences."""


def generate_validated_answer(
    image_description: str,
    question: str,
    faq_context: str,
    conversation_context: str = "",
) -> dict:
    """
    Stage 3 of the pipeline: generate the final answer AND a structured
    self-validation check, in a single call.

    Returns:
        {
            "answer": str,
            "validation_passed": bool,
            "validation_note": str,
            "raw_model_response": str,
        }
    """
    client = get_groq_client()

    if conversation_context:
        context_block = (
            f"CONVERSATION CONTEXT FROM EARLIER IN THIS SESSION (use this to "
            f"keep your answer consistent with what was already discussed):\n"
            f"{conversation_context}\n"
        )
    else:
        context_block = "(This is the first turn in the conversation — no prior context.)"

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        conversation_context_block=context_block,
        image_description=image_description,
        question=question,
        faq_context=faq_context or "(no matching FAQ content found)",
    )

    response = client.chat.completions.create(
        model=VISION_MODEL_PRIMARY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=2048,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Answer model did not return valid JSON despite JSON mode being requested. "
            f"Raw response: {raw_text!r}. Original error: {e}"
        )

    return {
        "answer": parsed.get("answer", ""),
        "validation_passed": bool(parsed.get("validation_passed", False)),
        "validation_note": parsed.get("validation_note", ""),
        "raw_model_response": raw_text,
    }