import streamlit as st

from config import settings
from rag.chain import stream_response
from rag.document_loader import load_and_split
from rag.vector_store import add_documents, clear_collection, get_collection_stats

st.set_page_config(page_title="RAG AI Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .stAppDeployButton {display: none;}
    #MainMenu {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── API key handling ────────────────────────────────────────────────────────
groq_key = ""

if hasattr(st, "secrets"):
    groq_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_key and settings.groq_api_key:
    groq_key = settings.groq_api_key

# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG AI Chatbot")

    # API key
    st.subheader("API Key")

    if groq_key:
        st.success("Groq API key: Configured")
    else:
        groq_key = st.text_input(
            "Enter Groq API key",
            type="password",
            help="Get a free key at https://console.groq.com/keys",
        )

    if groq_key:
        settings.groq_api_key = groq_key

    api_ready = bool(groq_key)

    st.divider()

    # Document upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        disabled=not api_ready,
    )

    if uploaded_files and api_ready:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                for f in new_files:
                    try:
                        chunks = load_and_split(f)
                        add_documents(chunks)
                        st.session_state.processed_files.add(f.name)
                        st.toast(f"Added {f.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")

    st.divider()

    # Knowledge base stats
    st.subheader("Knowledge Base")
    stats = get_collection_stats()
    st.metric("Document Chunks", stats["total_chunks"])
    st.caption(f"Files processed: {len(st.session_state.processed_files)}")

    if stats["total_chunks"] > 0:
        if st.button("Clear Knowledge Base", type="secondary"):
            clear_collection()
            st.session_state.processed_files.clear()
            st.rerun()

    st.divider()

    # Config display
    with st.expander("Configuration"):
        st.text(f"LLM: {settings.llm_model} (Groq)")
        st.text(f"Embeddings: {settings.embedding_model} (local)")
        st.text(f"Chunk size: {settings.chunk_size}")
        st.text(f"Chunk overlap: {settings.chunk_overlap}")
        st.text(f"Top-K results: {settings.top_k}")
        st.text(f"Temperature: {settings.temperature}")

# ── Main chat area ──────────────────────────────────────────────────────────
st.header("Chat")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for doc, score in msg["sources"]:
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", None)
                    label = f"**{source}**"
                    if page is not None:
                        label += f" (Page {int(page) + 1})"
                    label += f" — relevance: {score:.2f}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", disabled=not api_ready):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        text_stream, sources = stream_response(prompt, st.session_state.messages[:-1])
        response = st.write_stream(text_stream)

        if sources:
            with st.expander("View Sources"):
                for doc, score in sources:
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", None)
                    label = f"**{source}**"
                    if page is not None:
                        label += f" (Page {int(page) + 1})"
                    label += f" — relevance: {score:.2f}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

elif not api_ready:
    st.info("Please enter your Groq API key in the sidebar to get started. Get a free key at https://console.groq.com/keys")
