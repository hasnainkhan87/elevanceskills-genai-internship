"""
concept_extraction.py
=======================
Basic information extraction for CS/ML research questions and abstracts —
same curated-keyword pattern as Task 3's medical_ner.py, adapted to a
computer science vocabulary instead of medical terms.

WHY KEYWORDS AGAIN (same reasoning as Task 3)
------------------------------------------------
A real research-concept extraction model would need training data specific
to CS/ML terminology (similar to how medical NER needs biomedical training
data) — not something a general-purpose free-tier tool provides out of the
box. A curated list of ~100 common ML/CS terms, categorized, is fully
deterministic, testable without any model download, and directly satisfies
the brief's "information extraction" requirement at the "basic" level the
task actually asks for.

CATEGORIES
-----------
MODEL       — architectures / model families (transformer, CNN, LSTM, ...)
TASK        — problem types (classification, segmentation, translation, ...)
METHOD      — techniques / algorithms (gradient descent, backpropagation, ...)
DATASET_EVAL — datasets and evaluation concepts (benchmark, accuracy, F1 score, ...)
"""

import re

MODEL_TERMS = [
    "convolutional neural network", "recurrent neural network",
    "long short-term memory", "generative adversarial network",
    "graph neural network", "variational autoencoder",
    "transformer", "attention mechanism", "self-attention",
    "encoder-decoder", "autoencoder", "cnn", "rnn", "lstm", "gru",
    "gan", "bert", "gpt", "resnet", "vgg", "u-net", "vae",
    "diffusion model", "vision transformer", "neural network",
    "deep learning", "reinforcement learning agent",
]

TASK_TERMS = [
    "image classification", "object detection", "semantic segmentation",
    "machine translation", "sentiment analysis", "named entity recognition",
    "question answering", "text summarization", "speech recognition",
    "image generation", "anomaly detection", "recommendation system",
    "clustering", "classification", "regression", "segmentation",
    "translation", "generation", "detection", "recognition",
    "forecasting", "retrieval",
]

METHOD_TERMS = [
    "gradient descent", "stochastic gradient descent", "backpropagation",
    "transfer learning", "reinforcement learning", "supervised learning",
    "unsupervised learning", "self-supervised learning", "few-shot learning",
    "zero-shot learning", "fine-tuning", "regularization", "dropout",
    "batch normalization", "data augmentation", "hyperparameter tuning",
    "cross-validation", "ensemble learning", "active learning",
    "federated learning", "contrastive learning", "knowledge distillation",
    "pruning", "quantization",
]

DATASET_EVAL_TERMS = [
    "benchmark dataset", "training set", "validation set", "test set",
    "f1 score", "precision", "recall", "accuracy", "loss function",
    "cross entropy", "mean squared error", "confusion matrix",
    "roc curve", "auc", "bleu score", "perplexity", "state of the art",
    "ablation study", "overfitting", "underfitting", "benchmark",
]

CATEGORY_MAP = [
    ("MODEL", MODEL_TERMS),
    ("TASK", TASK_TERMS),
    ("METHOD", METHOD_TERMS),
    ("DATASET_EVAL", DATASET_EVAL_TERMS),
]


def _build_term_index():
    all_terms = []
    seen = set()
    for category, terms in CATEGORY_MAP:
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            all_terms.append((key, category))
    all_terms.sort(key=lambda t: len(t[0]), reverse=True)  # longest match first
    return all_terms


_TERM_INDEX = _build_term_index()


def extract_concepts(text: str) -> list:
    """
    Scan `text` for known CS/ML concept terms.

    Returns: [{"term": "transformer", "category": "MODEL"}, ...]

    Same longest-match-first, non-overlapping-span logic as Task 3's
    medical_ner.py — e.g. "convolutional neural network" is matched as one
    term rather than also separately matching "neural network" inside it.
    """
    if not text:
        return []

    text_lower = text.lower()
    matched_spans = []
    results = []
    seen_terms = set()

    for term, category in _TERM_INDEX:
        pattern = r"\b" + re.escape(term) + r"\b"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            if any(not (end <= s or start >= e) for s, e in matched_spans):
                continue
            matched_spans.append((start, end))
            if term not in seen_terms:
                seen_terms.add(term)
                results.append({"term": term, "category": category})
            break

    return results


def concept_frequency(texts: list) -> dict:
    """
    Given a list of texts (e.g. a batch of retrieved abstracts), return a
    dict of {term: count} across all of them — used for the concept
    visualization feature (a simple bar chart of what came up most).
    """
    freq = {}
    for text in texts:
        for entity in extract_concepts(text):
            freq[entity["term"]] = freq.get(entity["term"], 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    test_cases = [
        "We propose a convolutional neural network for image classification "
        "trained using stochastic gradient descent, achieving state of the "
        "art accuracy on the benchmark dataset.",
        "This transformer-based model uses self-attention for machine "
        "translation and is evaluated using BLEU score.",
        "Nothing technical in this sentence at all.",
    ]
    for t in test_cases:
        print(t)
        print(" ->", extract_concepts(t))
        print()
