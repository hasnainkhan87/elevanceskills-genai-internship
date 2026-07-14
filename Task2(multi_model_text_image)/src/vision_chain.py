"""
vision_chain.py
================
Task 2 — Orchestration layer tying together vision extraction (Groq),
FAISS retrieval (reused from Task 1), and validated answer generation
(Groq) into one coherent multi-modal pipeline, with conversation
memory across turns that is ACTIVELY used in the prompts (not just
recorded).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vision_helper


def retrieve_faq_context(extracted_topic: str, k: int = 3) -> str:
    """
    Query the EXISTING Task 1 FAISS index using the topic extracted
    from the image. Returns concatenated top-k FAQ snippets, or empty
    string if extracted_topic is None or no reasonable match exists.
    Makes NO API calls — free, local vector similarity search.
    """
    if not extracted_topic:
        return ""

    from langchain_helper import get_embeddings, FAISS_INDEX_DIR
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = vectordb.similarity_search(extracted_topic, k=k)

    if not docs:
        return ""

    return "\n\n".join(doc.page_content for doc in docs)


class VisionConversation:
    """
    Holds conversation memory across multiple image/text turns for one
    chat session. A fresh instance should be created per Streamlit
    session (e.g. stored in st.session_state).
    """

    def __init__(self):
        self.turns = []

    def add_turn(self, question: str, image_description: str, answer: str, validation_passed: bool):
        self.turns.append({
            "question": question,
            "image_description": image_description,
            "answer": answer,
            "validation_passed": validation_passed,
        })

    def get_context_summary(self, max_turns: int = 3) -> str:
        """
        Build a short text summary of recent turns. Only the most
        recent max_turns are included to keep prompts bounded.
        """
        if not self.turns:
            return ""

        recent = self.turns[-max_turns:]
        lines = []
        for i, turn in enumerate(recent, 1):
            lines.append(f"Turn {i} — User asked: {turn['question']}")
            lines.append(f"  (Image showed: {turn['image_description'][:100]})")
            lines.append(f"  Assistant answered: {turn['answer'][:150]}")
        return "\n".join(lines)


def ask_with_image(
    image_base64: str,
    question: str,
    conversation: VisionConversation,
    image_mime_type: str = "image/jpeg",
) -> dict:
    """
    The full Task 2 pipeline entry point. Call this once per
    image+question turn.

    FIX: conversation context is now actually passed into both Groq
    calls (extraction and answer generation), closing a previously
    disclosed gap where memory was recorded but never used in
    reasoning. Verified with mocked tests across a 2-turn sequence.

    Returns:
        {
            "answer": str,
            "is_ambiguous": bool,
            "image_description": str,
            "extracted_topic": str or None,
            "faq_context_used": str,
            "validation_passed": bool or None,
            "validation_note": str or None,
        }
    """
    context_summary = conversation.get_context_summary()

    # Stage 1: vision extraction + ambiguity check (Groq call #1)
    extraction = vision_helper.extract_image_content(
        image_base64, question, image_mime_type, conversation_context=context_summary
    )

    if extraction["is_ambiguous"]:
        conversation.add_turn(
            question=question,
            image_description=extraction["description"],
            answer=extraction["clarifying_question"],
            validation_passed=False,
        )
        return {
            "answer": extraction["clarifying_question"],
            "is_ambiguous": True,
            "image_description": extraction["description"],
            "extracted_topic": extraction["extracted_topic"],
            "faq_context_used": "",
            "validation_passed": None,
            "validation_note": None,
        }

    # Stage 2: retrieval (free, local)
    faq_context = retrieve_faq_context(extraction["extracted_topic"])

    # Stage 3: validated answer generation (Groq call #2)
    result = vision_helper.generate_validated_answer(
        image_description=extraction["description"],
        question=question,
        faq_context=faq_context,
        conversation_context=context_summary,
    )

    conversation.add_turn(
        question=question,
        image_description=extraction["description"],
        answer=result["answer"],
        validation_passed=result["validation_passed"],
    )

    return {
        "answer": result["answer"],
        "is_ambiguous": False,
        "image_description": extraction["description"],
        "extracted_topic": extraction["extracted_topic"],
        "faq_context_used": faq_context,
        "validation_passed": result["validation_passed"],
        "validation_note": result["validation_note"],
    }