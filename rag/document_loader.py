import tempfile
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile

from config import settings


def load_and_split(uploaded_file: UploadedFile) -> list:
    """Parse an uploaded file and split it into chunks with metadata."""
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix in (".txt", ".md"):
        loader = TextLoader(tmp_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    Path(tmp_path).unlink(missing_ok=True)
    return chunks
