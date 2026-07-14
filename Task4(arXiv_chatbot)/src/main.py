import streamlit as st
import pandas as pd
from pathlib import Path

from langchain_helper import (
    get_conversational_chain,
    create_vector_db,
    summarize_text,
    FAISS_INDEX_DIR,
    DATASET_CSV,
)
from concept_extraction import extract_concepts, concept_frequency

st.set_page_config(page_title="arXiv CS Research Assistant", page_icon="📚")
st.title("arXiv CS Research Assistant 📚")
st.caption(
    "A domain-expert chatbot over a sampled subset of arXiv computer science "
    "papers — runs on a local, open-source embedding model and LLM (no API "
    "key, no usage quota)."
)

# --------------------------------------------------------------------------
# Sidebar: knowledge base status + build button
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📚 Knowledge Base Status")
    if Path(FAISS_INDEX_DIR).exists():
        st.success("FAISS index found")
    else:
        st.warning("No FAISS index yet — click 'Create Knowledgebase' first")

    if st.button("Create Knowledgebase"):
        with st.spinner(
            "Building FAISS index locally (downloads the embedding model on "
            "first run — this can take a few minutes)..."
        ):
            n = create_vector_db()
        st.success(f"Knowledge base created with {n} papers.")
        st.rerun()

    st.markdown("---")
    st.caption(
        "Running fully locally: sentence-transformers for embeddings, "
        "a small open-source LLM (flan-t5-base) for answers. Expect "
        "answers to take longer and be less polished than a hosted API — "
        "that's the tradeoff for zero API cost/quota."
    )

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------
# Tabs: Chat / Paper search / Concept visualization
# --------------------------------------------------------------------------
tab_chat, tab_search, tab_concepts = st.tabs(["💬 Chat", "🔍 Paper Search", "📊 Concepts"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples

# --------------------------------------------------------------------------
# Chat tab — conversational Q&A with follow-up support
# --------------------------------------------------------------------------
with tab_chat:
    if not Path(FAISS_INDEX_DIR).exists():
        st.info("Build the knowledge base from the sidebar first.")
    else:
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        question = st.chat_input("Ask about a CS/ML concept or paper...")
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking (local model — may take a moment)..."):
                    chain = get_conversational_chain()
                    result = chain.invoke({
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                    })
                answer = result["answer"]
                st.write(answer)

                entities = extract_concepts(question)
                if entities:
                    icon_map = {"MODEL": "🟣", "TASK": "🟡", "METHOD": "🟢", "DATASET_EVAL": "🔵"}
                    tags = " &nbsp; ".join(
                        f"{icon_map.get(e['category'], '⚪')} **{e['term']}** `{e['category']}`"
                        for e in entities
                    )
                    st.caption("Detected concepts:")
                    st.markdown(tags)

                with st.expander("Source papers used"):
                    for doc in result.get("source_documents", []):
                        st.write(f"**{doc.metadata.get('source', 'Unknown title')}**")
                        st.caption(doc.page_content[:300] + "...")

            st.session_state.chat_history.append((question, answer))

# --------------------------------------------------------------------------
# Paper search tab — plain filter over the loaded CSV, no LLM involved
# --------------------------------------------------------------------------
with tab_search:
    st.subheader("Search papers by keyword")
    if not DATASET_CSV.exists():
        st.info("Run the preprocessing script first to generate the paper dataset.")
    else:
        df = pd.read_csv(DATASET_CSV)
        query = st.text_input("Search titles and abstracts:")
        if query:
            mask = (
                df["prompt"].str.contains(query, case=False, na=False)
                | df["response"].str.contains(query, case=False, na=False)
            )
            results = df[mask]
            st.write(f"{len(results)} matching papers")
            for _, row in results.iterrows():
                with st.expander(row["prompt"]):
                    st.caption(f"Categories: {row['categories']} | Authors: {row['authors']}")
                    st.write(row["response"])
                    if st.button("Summarize this abstract", key=f"sum_{row['arxiv_id']}"):
                        with st.spinner("Summarizing with local LLM..."):
                            summary = summarize_text(row["response"])
                        st.info(summary)
        else:
            st.dataframe(df[["prompt", "categories", "authors"]].head(20))

# --------------------------------------------------------------------------
# Concept visualization tab — simple frequency chart, no LLM involved
# --------------------------------------------------------------------------
with tab_concepts:
    st.subheader("Concept frequency across the loaded papers")
    if not DATASET_CSV.exists():
        st.info("Run the preprocessing script first to generate the paper dataset.")
    else:
        df = pd.read_csv(DATASET_CSV)
        freq = concept_frequency(df["response"].tolist())
        if freq:
            freq_df = pd.DataFrame(
                {"term": list(freq.keys())[:20], "count": list(freq.values())[:20]}
            ).set_index("term")
            st.bar_chart(freq_df)
        else:
            st.info("No recognized concepts found in the current dataset sample.")
