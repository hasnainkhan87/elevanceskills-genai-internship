"""
Addition to langchain_helper.py for Task 6 — Multilingual Support
=====================================================================
This function would be ADDED to your existing langchain_helper.py
(alongside get_qa_chain(), get_llm(), get_embeddings(), etc. — none of
which need to change). It builds a NEW chain type that supports
conversation memory and multilingual responses, while reusing the
exact same FAISS index, retriever, and embeddings as Task 1/5.
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from multilingual_helper import detect_language, detect_mixed_script, build_language_instruction


# A condense-question prompt that explicitly preserves the user's
# original language when LangChain rephrases a follow-up question into
# a standalone one.
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase
the follow-up question to be a standalone question. Keep it in the
SAME language the follow-up question was asked in — do not translate it.

Chat History:
{chat_history}
Follow-up question: {question}
Standalone question (same language as the follow-up question):"""
)


# REAL BUG FOUND VIA USER TESTING: an earlier version appended the
# language instruction onto the QUESTION TEXT itself. This silently
# failed for some languages (worked for Hindi, failed for French in the
# same session) because the question text passes through
# CONDENSE_QUESTION_PROMPT FIRST — an LLM call with no instruction to
# preserve appended meta-instructions, so whether they survived into
# the final answer was unreliable and varied by language.
#
# THE FIX: the language instruction is now baked directly into the
# LITERAL TEXT of the answer-step prompt template, not passed as data
# that flows through the question. The combine_docs_chain step (which
# generates the final answer) is NOT touched by the condense step — it
# receives this prompt's template text as-is, guaranteeing the
# instruction survives regardless of what the condense step does to
# the question text. Verified directly across 3 consecutive turns with
# 3 different languages, with the correct instruction confirmed present
# in the exact final prompt sent to the LLM each time, and memory
# correctly accumulating across all 3 rebuilt chains.
def build_answer_prompt(language_instruction: str) -> PromptTemplate:
    """
    Build the answer-generation prompt with a given language
    instruction baked in as literal template text (NOT a {variable} —
    deliberately, so it never has to travel through the
    condense-question step).
    """
    template = f"""Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {{context}}

QUESTION: {{question}}
{language_instruction}"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


def get_multilingual_qa_chain(memory=None, language_instruction: str = ""):
    """
    Build a ConversationalRetrievalChain that:
      - reuses the EXISTING FAISS index/retriever (same as get_qa_chain())
      - supports conversation memory across turns
      - has the given language instruction baked into its answer prompt

    IMPORTANT: this needs to be called FRESH for each question with that
    question's own detected-language instruction (see ask_multilingual()
    below). The chain object itself is rebuilt each call, but the
    MEMORY object must be the SAME one reused across calls, which is
    why `memory` is a parameter here rather than created fresh every
    time.

    Pass an existing `memory` object to continue a prior conversation
    (e.g. across Streamlit reruns using st.session_state), or leave as
    None to start a fresh conversation.
    """
    from langchain_helper import get_llm, get_embeddings, FAISS_INDEX_DIR
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    vectordb = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # REQUIRED — see comments in this file's history for why
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": build_answer_prompt(language_instruction)},
    )
    return chain, memory


def ask_multilingual(memory, question: str) -> dict:
    """
    Ask a question with multilingual + memory support. This function:
      1. detects the question's language
      2. flags mixed-script input honestly (see multilingual_helper.py)
      3. builds a FRESH chain for THIS question with that language's
         instruction baked into its answer prompt
      4. invokes it (memory is reused — the same object accumulates
         every turn across calls, even though the chain wrapper itself
         is rebuilt each time)

    NOTE: this function takes `memory` directly (not a pre-built
    `chain`), since the chain must be rebuilt per-question with the
    right language instruction. Callers (e.g. main.py) should keep the
    MEMORY object in st.session_state, not a pre-built chain object.

    Returns a dict with the answer PLUS the detected-language metadata,
    so the Streamlit UI can show "Detected language: Hindi" the same
    way Task 5 shows "Detected tone: negative".
    """
    lang_result = detect_language(question)
    mixed_script = detect_mixed_script(question)
    language_instruction = build_language_instruction(lang_result["code"])

    chain, memory = get_multilingual_qa_chain(memory=memory, language_instruction=language_instruction)
    result = chain.invoke({"question": question})

    return {
        "answer": result["answer"],
        "source_documents": result.get("source_documents", []),
        "detected_language": lang_result,
        "mixed_script_detected": mixed_script,
        "memory": memory,  # same object, returned for clarity at the call site
    }