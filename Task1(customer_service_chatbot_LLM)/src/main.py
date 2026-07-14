import streamlit as st
from langchain_helper import (
    get_qa_chain,
    create_vector_db,
    update_vector_db_incremental,
    get_kb_status,
)

st.title(" CUSTOMER SERVICE CHATBOT 🤖")

status = get_kb_status()

with st.sidebar:
    st.subheader("📚 Knowledge Base Status")
    if status["faiss_index_exists"]:
        st.success("FAISS index found")
    else:
        st.warning("No FAISS index yet — click 'Create Knowledgebase' first")
    st.write(f"**Total FAQs indexed:** {status['total_documents']}")
    st.write(f"**Last updated:** {status['last_updated'] or 'never'}")

    if status["history"]:
        st.caption("Recent update activity:")
        for h in reversed(status["history"]):
            tag = "🆕 full rebuild" if h["action"] == "full_rebuild" else "🔄 incremental check"
            st.caption(f"{h['timestamp']} — {tag} — +{h['documents_added']} doc(s)")

col1, col2 = st.columns(2)

with col1:
    if st.button("Create Knowledgebase (full rebuild)"):
        with st.spinner("Building FAISS index from dataset/dataset.csv... this may take a moment"):
            n = create_vector_db()
        st.success(f"Knowledge base created with {n} FAQs.")
        st.rerun()

with col2:
    if st.button("🔄 Check new_sources/ now"):
        with st.spinner("Scanning new_sources/ for new or changed files..."):
            summary = update_vector_db_incremental()
        if summary["new_documents_added"] > 0:
            st.success(
                f"Added {summary['new_documents_added']} new FAQ(s) "
                f"from {summary['files_scanned']}."
            )
        else:
            st.info("No new content found — knowledge base already up to date.")
        st.rerun()

st.caption(
    "Tip: drop a new CSV (columns: prompt, response) or .txt file into "
    "new_sources/ and either click the button above or let "
    "`update_scheduler.py` pick it up automatically."
)

question = st.text_input("Question: ")

if question:
    if not status["faiss_index_exists"]:
        st.error("Please click 'Create Knowledgebase' first.")
    else:
        chain = get_qa_chain()
        response = chain.invoke({"query": question})

        st.header("Answer")
        st.write(response["result"])
