# Task 4 — arXiv CS Research Assistant (Open-Source LLM)

An extension of the Retrieval-Augmented Generation (RAG) architecture used in
**Tasks 1 and 3**, adapted for answering research questions over a sampled subset
of **arXiv Computer Science papers**.

Unlike the previous tasks, this chatbot replaces the hosted Gemini models with
**fully open-source embedding and language models running locally**, eliminating
API keys, daily quotas, and external dependencies while adding conversational
memory, paper search, and concept visualization.

---

## Problem

The previous chatbot architecture was capable of retrieving answers from
structured datasets but was limited to FAQ-style conversations.

The objective of this task was to build a research assistant capable of:

- Answering questions about Computer Science research papers
- Supporting follow-up questions through conversational memory
- Searching papers by keywords
- Extracting important research concepts
- Visualizing concept frequency across the dataset
- Running entirely on open-source models without relying on external APIs

---

## Solution

The implementation reuses the Retrieval-Augmented Generation pipeline developed
in previous tasks while replacing the hosted Gemini models with local
open-source alternatives.

The solution includes:

- **Sentence-Transformers (`all-MiniLM-L6-v2`)** for local embeddings
- **FAISS** vector database for semantic retrieval
- **FLAN-T5 Base** running locally for response generation
- **ConversationalRetrievalChain** for multi-turn conversations
- **Paper Search** tab for keyword-based paper lookup
- **Concept Extraction** using keyword-based NLP
- **Concept Frequency Visualization**
- **Streamlit UI** with Chat, Paper Search, and Concepts tabs

Unlike previous tasks, the entire pipeline executes locally without requiring
API keys or internet connectivity after the initial model download.

---

# Verified: real test results

## 1. Research question answered using retrieved papers

**Question**

> **What is postoperative delirium?**

The chatbot successfully retrieved multiple relevant research papers from the
knowledge base and generated a summarized explanation describing postoperative
delirium (POD) while listing the source papers used to generate the answer.

This demonstrates that semantic retrieval correctly identifies relevant papers
before response generation.

**Screenshot**

<img width="1031" height="841" alt="image" src="https://github.com/user-attachments/assets/36730d89-1f3f-47ca-9766-09135cd6df27" />


---

## 2. Conversational memory supports follow-up questions

Instead of asking another complete question, the chatbot was asked:

> **What are its main challenges?**

Although the question did not explicitly mention postoperative delirium, the
assistant correctly understood the reference using the previous conversation
history and generated a context-aware response.

This verifies that **ConversationalRetrievalChain** successfully maintains chat
history, allowing natural multi-turn conversations.

**Screenshot**

<img width="760" height="702" alt="image" src="https://github.com/user-attachments/assets/9f9bb096-fa41-4648-8d8b-29a75dd7b069" />


---

## 3. Context-aware follow-up using previous discussion

A second follow-up question was tested:

> **Is postoperative delirium the same thing every time, or are there different types?**

The chatbot correctly continued the existing conversation without requiring the
topic to be reintroduced, explaining the existence of multiple postoperative
delirium phenotypes discussed in the retrieved research papers.

This further confirms that the chatbot understands conversational context rather
than treating every question independently.

**Screenshot**

<img width="806" height="573" alt="image" src="https://github.com/user-attachments/assets/18443d35-ef31-4ce1-b37d-f0a077ceec37" />


---

## 4. Paper Search functionality

The **Paper Search** tab was tested using the keyword:

> **privacy**

The application instantly displayed all matching research papers whose titles
contained the requested keyword.

Unlike the chatbot itself, this feature performs direct keyword searching over
the processed dataset rather than semantic retrieval, making searches
effectively instantaneous.

**Screenshot**

<img width="887" height="862" alt="image" src="https://github.com/user-attachments/assets/ed89db5e-50e6-460d-b8ac-7baef4008215" />


---

## 5. Concept frequency visualization

The **Concepts** tab analyzes all loaded research papers and extracts common
Computer Science concepts using a curated keyword dictionary.

The resulting bar chart displays frequently occurring concepts including:

- Accuracy
- Classification
- Deep Learning
- Detection
- Generation
- Retrieval
- Reinforcement Learning
- Transformer

This provides a quick overview of the dominant research topics contained within
the sampled arXiv dataset.

**Screenshot**

<img width="810" height="510" alt="image" src="https://github.com/user-attachments/assets/fed96b0f-9d23-4542-adb7-0146eefdacf6" />


---

## Dataset

The chatbot uses a sampled subset of the **arXiv Computer Science Dataset**
published by Cornell University.

Since the complete metadata snapshot contains millions of papers and exceeds
typical development hardware limits, preprocessing performs streaming and
reservoir sampling to create a manageable subset while preserving a uniform
distribution across Computer Science categories.

During preprocessing:

- JSONL records are streamed one line at a time
- Only Computer Science (`cs.*`) papers are retained
- Reservoir sampling selects a representative subset
- Metadata including title, abstract, authors, categories, and publication date
  is preserved
- The processed dataset is exported as `dataset/arxiv_cs_sample.csv`

---

## Concept Extraction

The chatbot includes a lightweight concept extraction module implemented in
`concept_extraction.py`.

Instead of relying on heavyweight research NLP models, the implementation uses a
curated keyword dictionary covering common Computer Science concepts grouped
into:

- **Models**
- **Tasks**
- **Methods**
- **Evaluation Metrics**

Detected concepts are used both for keyword highlighting and for generating the
Concept Frequency visualization displayed in the Streamlit interface.

---

## Real-world issue found during implementation

Unlike the previous Gemini-based implementations, this chatbot does not suffer
from API quota limitations.

However, running the entire pipeline locally introduced different practical
constraints.

The first execution downloads both the embedding model and FLAN-T5 language
model from Hugging Face, resulting in a large one-time download and slower
initial setup.

To keep the application responsive on CPU-only systems:

- A lightweight embedding model was selected
- FLAN-T5 Base was chosen instead of significantly larger LLMs
- Dataset sampling limits preprocessing and indexing time

These trade-offs provide a fully offline chatbot while maintaining reasonable
performance on standard hardware.

---

## Known limitations

- Answer quality depends on the sampled subset of arXiv papers rather than the
  complete dataset.
- FLAN-T5 Base provides lower-quality responses than larger commercial LLMs but
  remains lightweight enough for local execution.
- Concept extraction relies on a curated keyword dictionary and may miss
  uncommon terminology.
- The first application launch requires downloading the Hugging Face models
  before offline execution becomes available.

---

## How to reproduce

1. Download the arXiv metadata snapshot from Kaggle.

2. Install the project dependencies.

```bash
pip install -r requirements.txt
```

3. Generate the sampled Computer Science dataset.

```bash
python src/preprocess_arxiv.py --input path/to/arxiv-metadata-oai-snapshot.json
```

4. Launch the Streamlit application.

```bash
streamlit run src/main.py
```

5. Click **Create Knowledgebase**.

6. Try the following features:

- Ask a research question
- Continue with follow-up questions
- Search papers using keywords
- View the Concept Frequency visualization

---

## Screenshot checklist

| # | Marker text to find | Have it? |
|---|----------------------|-----------|
| 1 | "Research question answered using retrieved papers" | ✅ |
| 2 | "Conversational follow-up question" | ✅ |
| 3 | "Context-aware follow-up explaining POD types" | ✅ |
| 4 | "Paper Search using privacy keyword" | ✅ |
| 5 | "Concept Frequency visualization" | ✅ |
