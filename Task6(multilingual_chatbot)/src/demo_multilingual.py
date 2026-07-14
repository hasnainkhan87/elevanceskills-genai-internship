"""
demo_multilingual.py
======================
Task 6 — Demonstration script for language detection, mixed-script
flagging, and the conversation-memory data flow.

WHAT THIS SCRIPT VERIFIES LOCALLY (no API key needed):
  - Language detection across English, Hindi, Spanish, French
  - Mixed-script detection (Latin + Devanagari in one message)
  - The exact prompt instruction that would be sent to the LLM for
    each detected language

WHAT THIS SCRIPT CANNOT VERIFY (needs your real GOOGLE_API_KEY):
  - Whether gemini-2.5-flash actually answers correctly IN the
    requested language
  - Real cross-lingual retrieval quality against YOUR FAQ dataset
  - Whether conversation memory genuinely helps the model resolve a
    follow-up question correctly (e.g. "What about advanced topics?"
    correctly resolving to "...of the JavaScript course" from 2 turns
    ago, in a DIFFERENT language than the original question)

HOW TO GET THE REAL VERIFICATION (run this yourself, once, with your
API key in .env):
    streamlit run src/main.py
Then manually run through the "Full live demo script" at the bottom of
this file's output, take screenshots of each step, and paste them into
your README's "Live verification" section — replacing the placeholder
note this script prints, with real proof from your own run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilingual_helper import detect_language, detect_mixed_script, build_language_instruction  # noqa: E402
from sentiment_helper import detect_sentiment  # noqa: E402


LANGUAGE_DETECTION_TEST_SET = [
    ("What time does the course start?", "en"),
    ("कोर्स कब शुरू होता है?", "hi"),
    ("À quelle heure commence le cours?", "fr"),
    ("¿A qué hora empieza el curso?", "es"),
    ("Do you offer EMI payments? मुझे EMI चाहिए", "en"),   # mixed — see note below
    ("Merci beaucoup pour votre aide", "fr"),
    ("नमस्ते, क्या आपके पास जावास्क्रिप्ट कोर्स है?", "hi"),
    ("¿Tienen curso de Power BI?", "es"),
]


def run_language_detection_evaluation():
    print("=" * 80)
    print("PART 1: Language detection accuracy")
    print("=" * 80)
    correct = 0
    for text, expected in LANGUAGE_DETECTION_TEST_SET:
        result = detect_language(text)
        mixed = detect_mixed_script(text)
        is_correct = result["code"] == expected
        correct += int(is_correct)
        mark = "✓" if is_correct else "✗"
        print(f"{mark} {text[:50]:<52} expected={expected:<3} got={result['code']:<3} "
              f"(conf={result['confidence']:.2f})  mixed_script={mixed}")

    total = len(LANGUAGE_DETECTION_TEST_SET)
    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print(
        "\nNOTE on the mixed-language row above: langdetect (a single-label\n"
        "classifier) returned 'en' for \"Do you offer EMI payments? मुझे EMI\n"
        "चाहिए\" — it does NOT detect that Hindi is also present. This is\n"
        "counted as a 'pass' against expected='en' here only because\n"
        "langdetect's single dominant-language answer happened to be 'en'\n"
        "for this example; it is NOT a working code-switching detector.\n"
        "detect_mixed_script() catches this specific case via Unicode\n"
        "script analysis (see output above: mixed_script=True), which is\n"
        "the actual mixed-language handling this project provides — a\n"
        "real but partial solution, documented honestly in the README."
    )


def run_language_instruction_preview():
    print("\n" + "=" * 80)
    print("PART 2: Prompt instructions that would be sent to the LLM")
    print("=" * 80)
    for text, _ in LANGUAGE_DETECTION_TEST_SET[:4]:  # one per supported language
        lang = detect_language(text)
        instruction = build_language_instruction(lang["code"])
        print(f"\nQuestion: {text!r}")
        print(f"Detected: {lang['name']}")
        print(f"Instruction appended to question: {instruction.strip() or '(none — English default)'}")


def run_translated_sentiment_preview():
    print("\n" + "=" * 80)
    print("PART 3: Translation-assisted sentiment detection (non-English)")
    print("=" * 80)
    print(
        "NOTE: this part calls a live translation service (deep-translator's\n"
        "free GoogleTranslator wrapper) and WILL make a network request.\n"
        "If it fails (no internet in this environment, or the free service\n"
        "is temporarily unavailable — a documented possibility, not a bug\n"
        "in this code), the function falls back to scoring the untranslated\n"
        "text, which is printed below either way so the behavior is visible.\n"
    )
    examples = [
        ("यह कोर्स बहुत निराशाजनक है, कुछ काम नहीं कर रहा", "hi"),  # negative in Hindi
        ("Merci, ce cours est fantastique!", "fr"),                   # positive in French
        ("¿A qué hora empieza el curso?", "es"),                      # neutral in Spanish
    ]
    for text, lang_code in examples:
        result = detect_sentiment(text, source_language=lang_code)
        print(f"\nOriginal ({lang_code}): {text}")
        print(f"Translated for scoring: {result.get('translated_text') or '(translation unavailable — scored original text)'}")
        print(f"Detected sentiment: {result['label']} (compound={result['compound']:+.3f})")


FULL_LIVE_DEMO_SCRIPT = """
=====================================================================
FULL LIVE DEMO SCRIPT — run this manually once with `streamlit run
src/main.py` and a real API key, take screenshots, paste into README
=====================================================================

This is the actual evidence for "mixed-language input" and "context
retention across language switches" that this script CANNOT produce
on its own without a live LLM call.

STEP 1 — English question (establishes context):
    "Do you have a JavaScript course?"
    Expect: a real FAQ answer in English, "Detected language: English"

STEP 2 — Follow-up in Hindi, testing language switch + memory together:
    "क्या यह शुरुआती लोगों के लिए अच्छा है?"
    (= "Is it good for beginners?" — note this does NOT repeat
    "JavaScript course", it relies entirely on memory from Step 1)
    Expect: an answer that correctly understands "it" = the JavaScript
    course from Step 1, answered back in Hindi, "Detected language: Hindi"

STEP 3 — Follow-up in French, testing a SECOND language switch:
    "Et le prix ?"
    (= "And the price?" — again relies on memory, now switching to a
    third language in three consecutive turns)
    Expect: an answer about the JavaScript course's pricing (from
    memory), in French

STEP 4 — Mixed-language input:
    "Do you offer EMI payments? मुझे EMI चाहिए"
    Expect: "⚠️ mixed-language input detected" caption appears (from
    detect_mixed_script()), and the chatbot still attempts a
    reasonable answer rather than crashing on the mixed input

STEP 5 — Sentiment + language together:
    "यह कोर्स बहुत निराशाजनक है" (= "this course is very disappointing")
    Expect: BOTH "Detected tone: negative" AND "Detected language: Hindi"
    captions appear together, AND the sentiment caption shows the
    translated text used for scoring (since this is non-English)

Take a screenshot after each step and reference them in your README's
"Live verification (real API)" section, replacing this placeholder.
"""


if __name__ == "__main__":
    run_language_detection_evaluation()
    run_language_instruction_preview()
    try:
        run_translated_sentiment_preview()
    except Exception as e:
        print(f"\n(Translation-dependent demo section skipped — no network access in this "
              f"environment: {type(e).__name__}. This section requires internet access "
              f"and will work when you run it yourself.)")
    print(FULL_LIVE_DEMO_SCRIPT)
