# ElevanceSkills Generative AI Internship — Nullclass Customer Service Chatbot

This repository contains all six task submissions for the Generative AI internship through ElevanceSkills (client: Nullclass). Every task extends the same original training project — a Streamlit-based customer support chatbot for Nullclass, an e-learning company selling data science courses and virtual internships — following the internship's core requirement to build on the training pipeline rather than create unrelated standalone projects.

**Repo:** https://github.com/hasnainkhan87/elevanceskills-genai-internship

## The Original Training Project

The base project is a Retrieval-Augmented Generation (RAG) chatbot built with:
- **LangChain** for orchestration
- **Google Gemini** (`gemini-2.5-flash` for generation, `gemini-embedding-001` for embeddings)
- **FAISS** as the vector store
- **Streamlit** for the user interface
- A CSV of real Nullclass FAQs as the knowledge base

Every task below extends this same architecture with a new capability, reusing the retrieval pipeline wherever possible rather than rebuilding it from scratch.

## Tasks

| Task | Folder | What it adds |
|---|---|---|
| **1** | [`Task1(customer_service_chatbot_LLM)/`](./Task1(customer_service_chatbot_LLM)) | Dynamic, incremental knowledge base updates — new FAQs are embedded and merged into the FAISS index without ever rebuilding it from scratch |
| **2** | [`Task2(multi_model_text_image)/`](./Task2(multi_model_text_image)) | Multi-modal support — students can upload an image (screenshot, course material) and get an answer grounded in both the image and the FAQ knowledge base, with explicit ambiguity handling and response validation |
| **3** | [`Task3(medical_q&n_chatbot)/`](./Task3(medical_q&n_chatbot)) | A specialized medical Q&A chatbot built on the MedQuAD dataset, with basic medical entity recognition (symptoms/diseases/treatments) |
| **4** | [`Task4(arXiv_chatbot)/`](./Task4(arXiv_chatbot)) | A domain-expert chatbot trained on a computer science subset of the arXiv papers dataset, for discussing and summarizing research topics |
| **5** | [`Task5(sentimental_delivery)/`](./Task5(sentimental_delivery)) | Sentiment-aware responses — detects positive/negative/neutral tone in a question and frames the answer's empathy accordingly, without altering the underlying facts |
| **6** | [`Task6(multilingual_chatbot)/`](./Task6(multilingual_chatbot)) | Multilingual conversation support (English, Hindi, Spanish, French) with memory across turns and language switches, building on Task 5's sentiment layer |

Each task folder is a **complete, independently runnable project** — you can `cd` into any one of them and run it on its own without needing the others.

## Architecture Philosophy

Rather than treating each task as an isolated demo, this repo follows a layered approach where possible: Task 6 builds on Task 5, which builds on Task 1, so the most advanced folder (`Task6/`) demonstrates dynamic knowledge base updates, sentiment awareness, *and* multilingual support working together in a single chatbot. Tasks 2, 3, and 4 are kept as standalone extensions of Task 1's core architecture, since combining every capability into one pipeline was judged to add more debugging risk than value given the project timeline — this tradeoff is documented in each task's own README.

## A Note on Architecture Changes

Task 2 uses **Groq** instead of Google Gemini for its vision and answer-generation steps. This was an intentional, mentor-approved deviation: Gemini's free tier was hit hard during Task 6's development (a real `429` rate-limit error at 20 requests/day), and the mentor explicitly confirmed it wasn't necessary to keep the exact same architecture across every task. This decision, along with the reasoning behind it, is documented in full in Task 2's README.

## Honesty and Documentation Standard

Every task's README follows the same standard: real, captured evidence (not just architectural claims), honest disclosure of bugs found during live testing and how they were fixed, and clear acknowledgment of any known limitations rather than overselling what was built. Several tasks include screenshots of real test runs against the live Gemini/Groq APIs as evidence.

## Getting Started

Each task folder contains its own `README.md` with full setup instructions, but the general pattern across all of them is:

1. `cd` into the task folder
2. Create a `.env` file (copy `.env.example`) with your API key(s)
3. `pip install -r requirements.txt`
4. On Windows, run `$env:KMP_DUPLICATE_LIB_OK="TRUE"` each session (a known `faiss-cpu`/OpenMP conflict workaround)
5. `streamlit run src/main.py`
6. Click "Create Knowledgebase" in the app (builds the FAISS index on first run)

## About This Internship

This work was completed as part of the Generative AI internship offered through ElevanceSkills, with Nullclass as the training project client. The internship's core instruction was to extend the original training project with each task's required feature rather than building unrelated new projects — the folder structure and architectural decisions throughout this repo reflect that constraint.
