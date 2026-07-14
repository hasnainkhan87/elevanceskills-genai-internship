import streamlit as st
from langchain_helper import (
    create_vector_db,
    update_vector_db_incremental,
    get_kb_status,
)
from sentiment_helper import detect_sentiment, apply_sentiment_framing
from multilingual_chain import ask_multilingual
from multilingual_helper import detect_language

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

    st.divider()
    st.subheader("🌐 Conversation")
    st.caption(
        "This chatbot now remembers earlier turns (Task 6) — ask a "
        "follow-up question and it will use the context above, even if "
        "you switch languages."
    )
    if st.button("🗑️ Reset conversation"):
        st.session_state.pop("mqa_memory", None)
        st.session_state.pop("chat_log", None)
        st.rerun()

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
    "`update_scheduler.py` pick it up automatically. Ask in English, "
    "Hindi, Spanish, or French — the chatbot detects your language "
    "automatically and answers back in the same language."
)

# --- Task 6: conversation memory must survive Streamlit reruns ---
# Streamlit re-executes this whole script on every interaction, so a
# plain local variable for the memory would reset every time the user
# asks a new question. st.session_state persists across reruns within
# the same browser session, so the memory is created ONCE and reused.
#
# NOTE: we only store MEMORY in session_state, not a pre-built chain.
# The chain itself is rebuilt fresh on every question (inside
# ask_multilingual()) with that question's own detected-language
# instruction baked into its answer prompt — necessary because
# appending the instruction to the question text was found (via real
# user testing) to NOT reliably survive the chain's internal
# "condense question" step.
if status["faiss_index_exists"] and "mqa_memory" not in st.session_state:
    st.session_state["mqa_memory"] = None  # ask_multilingual() creates it on first use
    st.session_state["chat_log"] = []

question = st.text_input("Question: ")

if question:
    if not status["faiss_index_exists"]:
        st.error("Please click 'Create Knowledgebase' first.")
    else:
        # --- Task 5: sentiment detection (language-aware) ---
        lang_result = detect_language(question)
        sentiment = detect_sentiment(question, source_language=lang_result["code"])

        # --- Task 6: retrieval + generation, memory-aware, multilingual ---
        result = ask_multilingual(st.session_state["mqa_memory"], question)
        st.session_state["mqa_memory"] = result["memory"]

        # --- Task 5: apply emotional framing to the FINAL answer ---
        framed_answer = apply_sentiment_framing(
    result["answer"], sentiment, target_language=result["detected_language"]["code"]
)

        st.session_state["chat_log"].append({
            "question": question,
            "answer": framed_answer,
            "language": result["detected_language"]["name"],
            "mixed_script": result["mixed_script_detected"],
            "sentiment": sentiment["label"],
        })

        st.header("Answer")
        st.write(framed_answer)

        sentiment_emoji = {"positive": "🙂", "negative": "😟", "neutral": "😐"}
        mixed_flag = " ⚠️ mixed-language input detected" if result["mixed_script_detected"] else ""
        st.caption(
            f"{sentiment_emoji.get(sentiment['label'], '')} "
            f"Detected tone: **{sentiment['label']}** "
            f"(score: {sentiment['compound']:+.2f})"
            f"  |  🌐 Detected language: **{result['detected_language']['name']}**"
            f"{mixed_flag}"
        )
        if sentiment.get("translated_text"):
            st.caption(f"_(Sentiment was checked on a translated copy: \"{sentiment['translated_text']}\")_")

# --- Conversation history display ---
if st.session_state.get("chat_log"):
    st.divider()
    st.subheader("🕘 Conversation history (this session)")
    for turn in reversed(st.session_state["chat_log"][:-1]):
        st.markdown(f"**Q ({turn['language']}):** {turn['question']}")
        st.markdown(f"**A:** {turn['answer']}")
        st.caption("—")