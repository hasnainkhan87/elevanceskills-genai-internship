"""
medical_ner.py
===============
Basic medical entity recognition using a curated keyword dictionary —
deliberately NOT spaCy or any external NER library.

WHY KEYWORDS INSTEAD OF SPACY
------------------------------
spaCy's general-purpose models (en_core_web_sm/md/lg) are trained on
newswire and web text; they recognise PERSON, ORG, GPE, DATE, etc., but
have no built-in concept of "symptom" vs "disease" vs "treatment" — asking
en_core_web_sm to tag "fever" or "chemotherapy" returns nothing useful,
since those aren't categories it was trained on. A real medical NER model
(scispaCy, medspaCy, BioBERT-based taggers) would do this properly, but
those either need a heavier download (scispaCy's biomedical models are
400MB+) or paid hosted inference. A well-built curated keyword list directly
satisfies the brief's own wording — "basic medical entity recognition" —
is fully deterministic and testable, needs zero extra downloads, and runs
in microseconds. That tradeoff (recall on rare/misspelled terms) is the
documented limitation — see README.

MATCHING RULES
--------------
- Case-insensitive
- Longest-match-first: "heart attack" is checked before "heart" so multi-word
  terms aren't shadowed by a shorter term that happens to be a substring
- A term is only counted once even if it appears multiple times in the text
"""

import re

# --------------------------------------------------------------------------
# Curated term lists (~130 terms total across 3 categories)
# --------------------------------------------------------------------------

SYMPTOM_TERMS = [
    "abdominal pain", "back pain", "chest pain", "shortness of breath",
    "difficulty breathing", "night sweats", "weight loss",
    "weight gain", "blurred vision", "loss of appetite", "joint pain",
    "muscle weakness", "irregular heartbeat",
    "fever", "cough", "fatigue", "nausea", "vomiting", "headache",
    "dizziness", "swelling", "rash", "bleeding", "weakness", "numbness",
    "tingling", "insomnia", "diarrhea", "constipation", "seizure",
    "tremor", "anxiety", "depression", "confusion", "chills", "cramps",
    "itching", "bruising", "sweating", "palpitations", "wheezing",
    "sore throat", "runny nose", "congestion", "bloating", "hives",
]

DISEASE_TERMS = [
    "type 1 diabetes", "type 2 diabetes", "multiple sclerosis",
    "chronic fatigue syndrome", "chronic kidney disease",
    "heart disease", "high blood pressure", "sickle cell disease",
    "coronary artery disease", "rheumatoid arthritis", "heart attack",
    "cancer", "diabetes", "hypertension", "asthma", "arthritis",
    "alzheimer", "parkinson", "stroke", "obesity", "hiv", "aids",
    "hepatitis", "tuberculosis", "pneumonia", "anemia", "epilepsy",
    "lupus", "fibromyalgia", "covid", "influenza", "malaria",
    "thyroid", "leukemia", "lymphoma", "melanoma", "osteoporosis",
    "cirrhosis", "psoriasis", "eczema", "glaucoma", "cataract",
    "migraine", "bronchitis", "sinusitis", "gastritis", "ulcer",
]

TREATMENT_TERMS = [
    "physical therapy", "bone marrow transplant", "stem cell transplant",
    "radiation therapy", "hormone therapy", "blood transfusion",
    "insulin therapy", "kidney dialysis",
    "medication", "surgery", "therapy", "vaccine", "chemotherapy",
    "radiation", "antibiotics", "antidepressant", "painkiller",
    "immunotherapy", "dialysis", "transplant", "physiotherapy",
    "rehabilitation", "dosage", "treatment", "prescription",
    "injection", "infusion", "biopsy", "screening", "vaccination",
    "surgery", "medication", "insulin", "steroid", "antiviral",
    "anesthesia", "catheter", "pacemaker", "stent", "bypass",
]

CATEGORY_MAP = [
    ("SYMPTOM", SYMPTOM_TERMS),
    ("DISEASE", DISEASE_TERMS),
    ("TREATMENT", TREATMENT_TERMS),
]


def _build_term_index():
    """
    Flatten all (term, category) pairs and sort by term length descending,
    so longest-match-first works correctly (e.g. 'heart attack' is checked
    before the shorter, unrelated-category term 'heart' would be, if such
    a term existed).
    """
    all_terms = []
    seen = set()
    for category, terms in CATEGORY_MAP:
        for term in terms:
            key = term.lower()
            if key in seen:
                continue  # first category to claim a term wins (avoids duplicates)
            seen.add(key)
            all_terms.append((term.lower(), category))
    all_terms.sort(key=lambda t: len(t[0]), reverse=True)
    return all_terms


_TERM_INDEX = _build_term_index()


def detect_medical_entities(text: str) -> list:
    """
    Scan `text` for known medical terms and return a list of matches.

    Returns:
        [{"term": "fever", "category": "SYMPTOM"}, {"term": "diabetes", "category": "DISEASE"}, ...]

    Longest terms are matched first and matched spans are removed from
    consideration so a shorter term can't also match inside an
    already-matched longer term (e.g. "heart attack" won't also produce
    a separate spurious match for "heart" if "heart" were in the list).
    """
    if not text:
        return []

    text_lower = text.lower()
    matched_spans = []  # list of (start, end) already claimed
    results = []
    seen_terms = set()

    for term, category in _TERM_INDEX:
        # word-boundary regex so "fever" doesn't match inside "feverish" partially wrong,
        # but does match "feverish" as containing "fever" only if desired — here we require
        # the term itself to appear as a whole word/phrase boundary.
        pattern = r"\b" + re.escape(term) + r"\b"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            # skip if this span overlaps a span already claimed by a longer term
            if any(not (end <= s or start >= e) for s, e in matched_spans):
                continue
            matched_spans.append((start, end))
            if term not in seen_terms:
                seen_terms.add(term)
                results.append({"term": term, "category": category})
            break  # only need one occurrence per term to record it once

    return results


if __name__ == "__main__":
    # quick self-test
    test_cases = [
        "I have a fever and a headache, could this be the flu?",
        "My doctor mentioned type 2 diabetes and prescribed insulin therapy.",
        "What are the symptoms of a heart attack?",
        "Nothing medical in this sentence at all.",
    ]
    for t in test_cases:
        print(t)
        print(" ->", detect_medical_entities(t))
        print()
