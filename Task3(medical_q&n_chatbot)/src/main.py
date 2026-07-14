import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, FAISS_INDEX_DIR
from medical_ner import detect_medical_entities
from pathlib import Path

st.set_page_config(page_title="Medical Q&A Assistant", page_icon="🏥")

st.title("Medical Q&A Assistant 🏥")

with st.sidebar:
    st.subheader("📚 Knowledge Base Status")
    if Path(FAISS_INDEX_DIR).exists():
        st.success("FAISS index found")
    else:
        st.warning("No FAISS index yet — click 'Create Knowledgebase' first")

    st.markdown("---")
    st.subheader("⚠️ Medical Disclaimer")
    st.warning(
        "This is an **educational internship project**, not a substitute for "
        "professional medical advice, diagnosis, or treatment. Answers are "
        "generated from a sample of the public MedQuAD dataset (NIH sources) "
        "and an AI language model — they may be incomplete or out of date. "
        "**Always consult a qualified healthcare provider** for any medical "
        "concern."
    )

btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Building FAISS index from dataset/medquad_dataset.csv..."):
        n = create_vector_db()
    st.success(f"Knowledge base created with {n} medical QA pairs.")
    st.rerun()

question = st.text_input("Ask a medical question:")

if question:
    if not Path(FAISS_INDEX_DIR).exists():
        st.error("Please click 'Create Knowledgebase' first.")
    else:
        with st.spinner("Searching medical knowledge base..."):
            chain = get_qa_chain()
            response = chain.invoke({"query": question})

        st.header("Answer")
        st.write(response["result"])

        entities = detect_medical_entities(question)
        if entities:
            st.subheader("Detected medical terms in your question")
            icon_map = {"SYMPTOM": "🔴", "DISEASE": "🔵", "TREATMENT": "🟢"}
            tags = " &nbsp; ".join(
                f"{icon_map.get(e['category'], '⚪')} **{e['term']}** `{e['category']}`"
                for e in entities
            )
            st.markdown(tags)
        else:
            st.caption("No specific symptoms, diseases, or treatments recognized in the question.")

        st.caption(
            "Reminder: this answer is generated from a sample dataset and an AI "
            "model, and is for educational purposes only — not medical advice."
        )
