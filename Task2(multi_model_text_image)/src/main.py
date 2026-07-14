import base64
import streamlit as st
from langchain_helper import create_vector_db, get_kb_status
from vision_chain import VisionConversation, ask_with_image

st.title("📸 Multi-Modal Course Support Assistant")

status = get_kb_status()

with st.sidebar:
    st.subheader("📚 Knowledge Base Status")
    if status["faiss_index_exists"]:
        st.success("FAISS index found (reused from Task 1)")
    else:
        st.warning("No FAISS index yet — click 'Create Knowledgebase' first")
    st.write(f"**Total FAQs indexed:** {status['total_documents']}")

    st.divider()
    st.subheader("🧠 How this works")
    st.caption(
        "1. Upload an image (screenshot, photo of course material, etc.) "
        "and ask a question.\n"
        "2. The assistant first checks if the image+question is clear "
        "enough to answer — if not, it asks YOU a clarifying question "
        "instead of guessing.\n"
        "3. If clear, it searches the same FAQ knowledge base from "
        "Task 1, generates an answer, and validates that answer "
        "against what's actually in the image before showing it to you."
    )

    if st.button("🗑️ Reset conversation"):
        st.session_state.pop("vision_conversation", None)
        st.rerun()

if st.button("Create Knowledgebase (full rebuild)"):
    with st.spinner("Building FAISS index from dataset/dataset.csv..."):
        n = create_vector_db()
    st.success(f"Knowledge base created with {n} FAQs.")
    st.rerun()

st.divider()

# --- Conversation memory must survive Streamlit reruns ---
if "vision_conversation" not in st.session_state:
    st.session_state["vision_conversation"] = VisionConversation()

uploaded_image = st.file_uploader(
    "Upload an image (screenshot, photo of course material, error message, etc.)",
    type=["png", "jpg", "jpeg"],
)
question = st.text_input("Your question about the image:")

if uploaded_image and question:
    if not status["faiss_index_exists"]:
        st.error("Please click 'Create Knowledgebase' first.")
    else:
        image_bytes = uploaded_image.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = uploaded_image.type or "image/jpeg"

        try:
            with st.spinner("Analyzing image and checking for ambiguity..."):
                result = ask_with_image(
                    image_base64=image_base64,
                    question=question,
                    conversation=st.session_state["vision_conversation"],
                    image_mime_type=mime_type,
                )
        except Exception as e:
            # A REAL error was hit during live testing: a text-dense
            # image caused the model to run out of its token budget
            # before finishing valid JSON (groq.BadRequestError:
            # "max completion tokens reached before generating a valid
            # document"). max_completion_tokens was increased in
            # vision_helper.py to reduce how often this happens, but
            # showing a graceful error here — instead of letting the
            # whole Streamlit app crash with a raw traceback — is the
            # right fallback regardless of the exact cause, since a
            # single image/question pair should never be able to break
            # the entire app for the rest of the session.
            st.error(
                f"Something went wrong analyzing this image: {type(e).__name__}. "
                f"This can happen with very text-dense or complex images. "
                f"Try a simpler/clearer image, or rephrase your question."
            )
            with st.expander("Technical details (for debugging)"):
                st.code(str(e))
            result = None

        if result is not None:
            st.image(uploaded_image, caption="Your uploaded image", width=300)

            if result["is_ambiguous"]:
                st.warning("🤔 The assistant needs more information:")
                st.write(result["answer"])
                st.caption(
                    "This is the AMBIGUITY HANDLING stage — the assistant "
                    "stopped here rather than guessing, and did not spend an "
                    "extra API call generating a speculative answer."
                )
            else:
                st.header("Answer")
                st.write(result["answer"])

                validation_icon = "✅" if result["validation_passed"] else "⚠️"
                st.caption(
                    f"{validation_icon} **Response validation:** "
                    f"{'Passed' if result['validation_passed'] else 'Flagged as potentially inconsistent'} "
                    f"— {result['validation_note']}"
                )

                with st.expander("🔍 See the reasoning behind this answer"):
                    st.write(f"**What the model saw in the image:** {result['image_description']}")
                    st.write(f"**Topic used to search the FAQ knowledge base:** {result['extracted_topic'] or '(none — nothing course-related detected)'}")
                    st.write(f"**FAQ content retrieved:** {result['faq_context_used'] or '(none found)'}")

# --- Conversation history, demonstrating "context across multiple interactions" ---
conversation = st.session_state["vision_conversation"]
if conversation.turns:
    st.divider()
    st.subheader("🕘 Conversation history (this session)")
    for i, turn in enumerate(reversed(conversation.turns[:-1] if (uploaded_image and question) else conversation.turns), 1):
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(f"**Image showed:** {turn['image_description'][:120]}...")
        st.markdown(f"**A:** {turn['answer']}")
        st.caption("—")
        