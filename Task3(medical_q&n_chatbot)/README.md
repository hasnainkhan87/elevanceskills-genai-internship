# Task 3 — Medical Q&A Chatbot (MedQuAD)

An extension of the same architecture developed in **Task 1**, adapted to answer
medical questions using the **MedQuAD** dataset. The chatbot combines **Gemini
Embeddings**, **FAISS**, **LangChain RetrievalQA**, and a **Streamlit UI** to
provide context-grounded medical answers while performing **basic medical
entity recognition** on user queries.

---

## Problem

The original chatbot built in Task 1 was designed around a small e-learning FAQ
dataset. While its Retrieval-Augmented Generation (RAG) pipeline could easily
be reused, it could not:

- Answer medical-domain questions
- Process MedQuAD's XML dataset
- Identify diseases, symptoms, or treatments mentioned by the user
- Provide a medical-specific interface and disclaimer

The objective of this task was therefore to extend the existing architecture
into a domain-specific Medical Q&A Assistant without redesigning the entire
pipeline.

---

## Solution

The chatbot reuses the retrieval pipeline developed in Task 1 while replacing
the underlying dataset and adding medical-specific preprocessing.

The implementation includes:

- **XML → CSV preprocessing** using `preprocess_medquad.py`
- **Gemini Embeddings (`models/gemini-embedding-001`)**
- **FAISS Vector Database** for semantic retrieval
- **LangChain RetrievalQA** with Gemini Flash
- **Keyword-based Medical Entity Recognition**
- **Streamlit UI** with educational medical disclaimer
- Dataset sampling and batched embedding generation to remain compatible with
  Gemini free-tier API limits

Unlike Task 1, only the preprocessing stage and entity recognition module are
new. The retrieval, embedding, indexing, and generation pipeline remains
unchanged.

---

# Verified: real test results

## 1. Correct retrieval of genetic information about Lynch syndrome

**Question**

> **What are the genetic changes related to Lynch syndrome?**

The chatbot successfully retrieved the relevant MedQuAD context and generated a
detailed explanation covering the five major genes associated with Lynch
syndrome (**MLH1, MSH2, MSH6, PMS2, and EPCAM**), while explaining their role
in DNA repair.

This demonstrates that semantic retrieval correctly located the appropriate
medical documents before answer generation.

**Screenshot**

<img width="1384" height="868" alt="image" src="https://github.com/user-attachments/assets/fbd0016c-085a-4a79-b92f-fe46b3c35443" />


---

## 2. Medical entity recognition working alongside answer generation

**Question**

> **How to prevent Liver Hepatocellular Cancer?**

The chatbot generated a medically relevant answer discussing:

- Hepatitis B and C
- Cirrhosis
- Aflatoxin exposure
- Hepatitis B vaccination

Additionally, the interface correctly detected **Cancer** as a medical disease
and displayed it under **Detected Medical Terms**.

This verifies that both systems operate independently:

- **RetrievalQA** generates the answer
- **medical_ner.py** recognizes medical entities from the user's question

**Screenshot**

<img width="1394" height="868" alt="image" src="https://github.com/user-attachments/assets/50ffc563-5075-4f2f-970a-50ec5d2658d3" />


---

## 3. Successful retrieval for another medical topic

**Question**

> **What is Smoking and the Digestive System?**

The chatbot successfully retrieved context explaining the relationship between
smoking and digestive disorders including:

- GERD
- Crohn's Disease
- Peptic ulcers
- Pancreatitis
- Liver disease
- Gallstones

This demonstrates that the retrieval pipeline performs consistently across
different medical topics rather than memorizing a single domain.

**Screenshot**

<img width="1388" height="868" alt="image" src="https://github.com/user-attachments/assets/2cac8781-e88a-4f0d-9659-0a6ecabf6ebf" />


---

## 4. Semantic understanding of paraphrased questions

Instead of repeating the wording used inside the dataset, the chatbot was asked:

> **Which genes are linked to Lynch syndrome?**

Although phrased differently, the chatbot still retrieved the correct context
and identified:

- MLH1
- MSH2
- MSH6
- PMS2
- EPCAM

This confirms that retrieval is based on **semantic similarity** rather than
exact keyword matching.

**Screenshot**

<img width="1392" height="868" alt="image" src="https://github.com/user-attachments/assets/a2810d98-7710-45c0-92e3-cb20a7109c8b" />


---

## Dataset

The chatbot uses the **MedQuAD** dataset containing over **47,000**
medical Question-Answer pairs collected from multiple **NIH websites**.

Since embedding the complete dataset would exceed the Gemini free-tier API
limits, preprocessing performs controlled sampling while maintaining coverage
across multiple medical specialties.

During preprocessing:

- XML files are parsed into structured Question-Answer pairs
- Metadata such as **Focus**, **Source**, and **Question Type** is preserved
- Very short or incomplete answers are discarded
- The processed data is exported as `dataset/medquad_dataset.csv`

The generated CSV follows the same format as Task 1, allowing the existing
retrieval pipeline to be reused without modification.

---

## Medical Entity Recognition

The chatbot includes a lightweight medical entity recognizer implemented in
`medical_ner.py`.

Instead of relying on heavyweight biomedical NLP models, the implementation
uses a curated dictionary of approximately **130 medical keywords** grouped
into:

- **Symptoms**
- **Diseases**
- **Treatments**

The recognizer performs:

- Case-insensitive matching
- Whole-word matching
- Longest-match-first detection

Detected entities are displayed beneath each answer, allowing users to see
which medical concepts were recognized in their question.

---

## Real-world issue found during implementation

While building the knowledge base, embedding the complete sampled dataset in
one operation exceeded the **Gemini free-tier embedding quota**, producing
multiple **HTTP 429 (Quota Exceeded)** errors.

To solve this, the implementation was modified to:

- Generate embeddings in **small batches**
- Add retry logic with exponential backoff
- Reduce the total sampled dataset size for demonstration purposes

These changes significantly improved reliability while remaining compatible
with the free-tier API limits.

---

## Known limitations

- This chatbot is intended **only for educational purposes** and should not be
  considered medical advice.
- The keyword-based entity recognizer only detects terms present in its
  dictionary and may miss uncommon diseases or spelling variations.
- The chatbot uses only a sampled subset of MedQuAD, so some medical topics may
  not be available.
- Response quality ultimately depends on the retrieved context and Gemini's
  generated output.

---

## How to reproduce

1. Install the project dependencies

```bash
pip install -r requirements.txt
```

2. Configure your Gemini API Key inside `.env`

3. Generate the processed MedQuAD dataset

```bash
python src/preprocess_medquad.py
```

4. Launch the Streamlit application

```bash
streamlit run src/main.py
```

5. Click **Create Knowledgebase**

6. Ask medical questions such as:

- What are the genetic changes related to Lynch syndrome?
- How to prevent Liver Hepatocellular Cancer?
- What is Smoking and the Digestive System?
- Which genes are linked to Lynch syndrome?

---

## Screenshot checklist

| # | Marker text to find | Have it? |
|---|----------------------|-----------|
| 1 | "Lynch syndrome genetic changes answered correctly" | ✅ |
| 2 | "Medical entity recognition detecting Cancer" | ✅ |
| 3 | "Smoking and Digestive System answer" | ✅ |
| 4 | "Semantic retrieval using paraphrased Lynch syndrome question" | ✅ |
