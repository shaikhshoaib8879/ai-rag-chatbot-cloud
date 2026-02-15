from collections.abc import Generator

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from rag.prompts import NO_DOCS_PROMPT, RAG_SYSTEM_PROMPT
from rag.vector_store import get_collection_stats, similarity_search


def _format_context(results: list) -> str:
    """Format retrieved chunks into a labeled context string."""
    parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", None)
        label = f"[Source {i}: {source}"
        if page is not None:
            label += f", Page {int(page) + 1}"
        label += f"] (relevance: {score:.2f})"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_chat_history(messages: list[dict]) -> str:
    """Format Streamlit chat history into a string for the prompt."""
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_prompt(query: str, chat_history: list[dict]) -> tuple[str, str, list]:
    """Build system instruction and user query. Returns (system, user_query, sources)."""
    stats = get_collection_stats()
    source_results = []

    if stats["total_chunks"] == 0:
        return NO_DOCS_PROMPT, query, source_results

    source_results = similarity_search(query)
    context = _format_context(source_results)
    history_str = _format_chat_history(chat_history)

    system_content = RAG_SYSTEM_PROMPT.format(
        context=context, chat_history=history_str
    )
    return system_content, query, source_results


def stream_response(
    query: str, chat_history: list[dict]
) -> tuple[Generator[str, None, None], list]:
    """Run the RAG pipeline and return a (text_stream, source_docs) tuple."""
    system_prompt, user_query, source_results = _build_prompt(query, chat_history)

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        system_instruction=system_prompt,
    )

    messages = [HumanMessage(content=user_query)]

    def generate() -> Generator[str, None, None]:
        try:
            for chunk in llm.stream(messages):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n\n**Error:** {e}"

    return generate(), source_results
