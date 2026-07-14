"""
evaluate_sentiment.py
=======================
Task 5 — Evaluation script for sentiment detection accuracy.

The task brief lists "accuracy of sentiment detection" as an explicit
evaluation criterion. This script provides real, reproducible evidence
for that: a small hand-labeled test set of realistic FAQ-chatbot-style
messages, run through detect_sentiment(), with accuracy reported.

Run with:
    python src/evaluate_sentiment.py

The test set below was written by hand to cover:
  - clearly negative messages (frustration, complaints, anger)
  - clearly positive messages (gratitude, satisfaction)
  - neutral factual questions (plain FAQ-style queries with no emotional
    content) — including ones that could be MISTAKEN for negative if you
    only looked at keywords (e.g. "refund", "cancel") but are phrased
    neutrally
  - a couple of harder/ambiguous cases, included on purpose and noted as
    known limitations rather than hidden from the results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_helper import detect_sentiment  # noqa: E402

# Each tuple: (message, expected_label)
LABELED_TEST_SET = [
    # --- Negative ---
    ("This is so frustrating, nothing works!", "negative"),
    ("I am really annoyed, I paid for this and it still does not work", "negative"),
    ("This is the worst experience ever, I want my money back", "negative"),
    ("Why is this so confusing, I've tried three times and failed", "negative"),
    ("I'm disappointed, the course content is outdated", "negative"),
    ("Terrible support, nobody replied to my email for a week", "negative"),

    # --- Positive ---
    ("Thank you so much, this is amazing!", "positive"),
    ("Great, finally got it working, thanks!", "positive"),
    ("I love this course so much", "positive"),
    ("Excellent explanation, that really helped me understand", "positive"),
    ("You guys are awesome, thanks for the quick fix", "positive"),

    # --- Neutral (plain factual FAQ questions, including ones with
    #     "negative-sounding" keywords that should NOT trigger negative) ---
    ("What time does the course start?", "neutral"),
    ("Can I get a refund?", "neutral"),
    ("Do you offer EMI payment options?", "neutral"),
    ("How do I cancel my subscription?", "neutral"),
    ("Is there a JavaScript course available?", "neutral"),
    ("What is the duration of the internship program?", "neutral"),
    ("Where can I download the certificate?", "neutral"),

    # --- Harder / ambiguous cases (kept in on purpose — see README for
    #     how these are discussed as known limitations, not hidden) ---
    ("Oh great, another error message", "positive"),   # sarcasm — VADER will likely get this WRONG
    ("I guess it's fine I suppose", "neutral"),         # genuinely ambiguous even for a human
]


def run_evaluation():
    correct = 0
    results = []

    for message, expected in LABELED_TEST_SET:
        detected = detect_sentiment(message)
        is_correct = detected["label"] == expected
        correct += int(is_correct)
        results.append((message, expected, detected["label"], detected["compound"], is_correct))

    total = len(LABELED_TEST_SET)
    accuracy = correct / total * 100

    print(f"{'Message':<55} {'Expected':<10} {'Detected':<10} {'Score':>7}  {'Result'}")
    print("-" * 95)
    for message, expected, detected_label, compound, is_correct in results:
        mark = "✓" if is_correct else "✗"
        print(f"{message[:53]:<55} {expected:<10} {detected_label:<10} {compound:+.3f}  {mark}")

    print("-" * 95)
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%\n")

    misses = [(m, e, d) for m, e, d, c, ok in results if not ok]
    if misses:
        print("Misclassified messages (see README 'Known Limitations' for discussion):")
        for message, expected, detected_label in misses:
            print(f"  - {message!r}: expected={expected}, got={detected_label}")

    return accuracy, results


if __name__ == "__main__":
    run_evaluation()
