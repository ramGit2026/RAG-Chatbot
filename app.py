"""Streamlit interface for the ServiceDesk GenAI Copilot."""

from __future__ import annotations

import json

import streamlit as st

from src.agent import generate_support_analysis
from src.config import (
    APP_NAME,
    DEFAULT_TOP_K,
    MAX_FILE_SIZE_MB,
)
from src.retrieval import (
    build_knowledge_base,
    load_embedding_model,
    retrieve_relevant_chunks,
)
from src.ticketing import build_ticket


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the embedding model for the first time...")
def get_embedding_model():
    return load_embedding_model()


def initialize_state() -> None:
    defaults = {
        "knowledge_base": None,
        "messages": [],
        "tickets": [],
        "processed_file_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No sufficiently relevant source passages were found.")
        return

    with st.expander("Evidence retrieved from the knowledge base", expanded=False):
        for index, source in enumerate(sources, start=1):
            location = f"Page {source['page']}" if source.get("page") else "Text document"
            st.markdown(
                f"**{index}. {source['source']}** · {location} · "
                f"similarity {source['score']:.3f}"
            )


def render_ticket(ticket: dict) -> None:
    st.markdown(f"**{ticket['ticket_id']}** · {ticket['status']}")
    st.json(ticket, expanded=False)
    st.download_button(
        label="Download ticket JSON",
        data=json.dumps(ticket, indent=2, ensure_ascii=False),
        file_name=f"{ticket['ticket_id'].lower()}.json",
        mime="application/json",
        key=f"download_{ticket['ticket_id']}",
    )


initialize_state()

st.title("🛠️ ServiceDesk GenAI Copilot")
st.caption(
    "An enterprise-style IT support assistant powered by RAG, semantic search, "
    "Gemini, and structured ticket triage."
)

with st.sidebar:
    st.header("Knowledge base")
    st.write("Upload your organization's IT policies, FAQs, and troubleshooting guides.")
    uploaded_files = st.file_uploader(
        "Knowledge documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help=f"PDF, TXT, or Markdown. Maximum {MAX_FILE_SIZE_MB} MB per file.",
    )

    if st.button("Build / refresh knowledge base", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one document first.")
        else:
            try:
                file_payloads = [
                    {"name": item.name, "content": item.getvalue()}
                    for item in uploaded_files
                ]
                with st.spinner("Extracting documents and building the vector index..."):
                    model = get_embedding_model()
                    kb = build_knowledge_base(file_payloads, model)
                st.session_state.knowledge_base = kb
                st.session_state.processed_file_names = kb.source_names
                st.session_state.messages = []
                st.success(
                    f"Knowledge base ready: {len(kb.chunks)} chunks "
                    f"from {len(kb.source_names)} document(s)."
                )
            except Exception as exc:
                st.error(f"Could not build the knowledge base: {exc}")

    if st.session_state.knowledge_base is not None:
        st.divider()
        st.success("Knowledge base loaded")
        st.caption(f"{len(st.session_state.knowledge_base.chunks)} searchable chunks")
        for filename in st.session_state.processed_file_names:
            st.caption(f"📄 {filename}")
        if st.button("Clear knowledge base and chat", use_container_width=True):
            st.session_state.knowledge_base = None
            st.session_state.messages = []
            st.session_state.processed_file_names = []
            st.rerun()

    st.divider()
    st.caption("Model: Gemini · Embeddings: Sentence Transformers · Vector search: FAISS")
    st.caption("Demo mode: ticket drafts are downloadable; they are not sent to a live ITSM system.")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = ""

if not api_key:
    st.warning(
        "Gemini API key not configured. Add GEMINI_API_KEY to "
        "`.streamlit/secrets.toml` locally or the app Secrets settings after deployment."
    )

if st.session_state.knowledge_base is None:
    st.info(
        "Start by uploading an IT support handbook, password policy, VPN guide, "
        "or help-desk FAQ in the sidebar, then click **Build / refresh knowledge base**."
    )
    st.markdown("#### Example employee requests")
    col1, col2, col3 = st.columns(3)
    col1.write("🔐 I cannot sign in after resetting my password.")
    col2.write("🌐 My office VPN keeps disconnecting.")
    col3.write("💻 How do I request a replacement laptop?")
else:
    overview = st.columns(3)
    overview[0].metric("Knowledge chunks", len(st.session_state.knowledge_base.chunks))
    overview[1].metric("Documents", len(st.session_state.processed_file_names))
    overview[2].metric("Draft tickets", len(st.session_state.tickets))

    st.divider()
    st.subheader("Employee support chat")
    st.caption("Ask a question about the uploaded company support documents.")

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                analysis = message.get("analysis", {})
                st.caption(
                    f"Category: {analysis.get('category', 'Other')} · "
                    f"Priority: {analysis.get('priority', 'Medium')} · "
                    f"Sentiment: {analysis.get('sentiment', 'Neutral')}"
                )
                render_sources(analysis.get("retrieved_sources", []))
                actions = analysis.get("recommended_actions", [])
                if actions:
                    with st.expander("Recommended next steps"):
                        for action in actions:
                            st.markdown(f"- {action}")

                ticket_key = f"ticket_for_message_{message_index}"
                if message.get("ticket"):
                    st.success(f"Draft ticket created: {message['ticket']['ticket_id']}")
                elif st.button("Prepare support ticket", key=ticket_key):
                    ticket = build_ticket(analysis, message.get("question", ""))
                    message["ticket"] = ticket
                    st.session_state.tickets.append(ticket)
                    st.rerun()

    question = st.chat_input(
        "Describe your IT issue or ask about a company procedure...",
        disabled=not bool(api_key),
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching company guidance and preparing a response..."):
                embedding_model = get_embedding_model()
                retrieved = retrieve_relevant_chunks(
                    question,
                    st.session_state.knowledge_base,
                    embedding_model,
                    top_k=DEFAULT_TOP_K,
                )
                analysis = generate_support_analysis(
                    question=question,
                    retrieved_chunks=retrieved,
                    api_key=api_key,
                )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": analysis["answer"],
                    "analysis": analysis,
                    "question": question,
                }
            )
        except Exception as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I could not complete the request. Check the API key, "
                        "model access, and application logs, then try again.\n\n"
                        f"Technical detail: `{exc}`"
                    ),
                    "analysis": {},
                    "question": question,
                }
            )
        st.rerun()

if st.session_state.tickets:
    st.divider()
    with st.expander(f"Support ticket drafts ({len(st.session_state.tickets)})"):
        for ticket in reversed(st.session_state.tickets):
            render_ticket(ticket)
            st.divider()

with st.expander("About this project"):
    st.markdown(
        """
        **ServiceDesk GenAI Copilot** is a portfolio demonstration of a grounded
        enterprise support workflow. It combines retrieval-augmented generation,
        semantic search, LLM-based issue triage, and downloadable structured ticket drafts.

        It does not connect to a real ticketing platform, identity provider, or company
        network. Use synthetic or approved documents; do not upload confidential data.
        """
    )
