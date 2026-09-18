from pathlib import Path
import pickle

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import POLICY_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, RAG_TOP_K


FAISS_INDEX_PATH = VECTORSTORE_DIR / "index"
_embeddings = None


def get_embeddings():
    """Load the embedding model once."""
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL, model_kwargs = {"device": "cpu"}, encode_kwargs = {"normalize_embeddings": True})
    
    return _embeddings


def load_policy_documents() -> list[Document]:
    """Load all supported policy documents."""
    documents = []

    if not POLICY_DIR.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

    for file_path in POLICY_DIR.iterdir():
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension == ".md":
            text = file_path.read_text(encoding="utf-8")
            documents.append(Document(page_content = text, metadata={"source": file_path.name, "type": "policy"}))


        elif extension == ".txt":
            text = file_path.read_text(encoding="utf-8")

            documents.append(Document(page_content = text,metadata = {"source": file_path.name, "type": "policy"}))

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 100)
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore() -> None:
    """Build a local FAISS vector store from policy documents."""
    documents = load_policy_documents()

    if not documents:
        raise ValueError("No policy documents found.")

    chunks = split_documents(documents)

    print(f"Loaded {len(documents)} document(s).")
    print(f"Created {len(chunks)} chunk(s).")

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents = True, exist_ok = True)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print( f"FAISS vector store saved to: " f"{FAISS_INDEX_PATH}")


def load_vectorstore() -> FAISS:
    """Load an existing local FAISS index."""

    index_file = (VECTORSTORE_DIR / "index.faiss")

    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index does not exist.\n"
            "Run:\n"
            "python scripts/ingest_sop.py"
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization = True)

    return vectorstore


def search_sop(query: str, k: int = RAG_TOP_K) -> list[Document]:
    """Perform semantic search over the SOP knowledge base."""
    vectorstore = load_vectorstore()
    documents = vectorstore.similarity_search(query, k=k)
    return documents


def format_sop_results(documents: list[Document]) -> str:
    if not documents:
        return "No relevant SOP information found."

    output = []

    for i, document in enumerate(documents, start = 1):
        source = document.metadata.get("source", "Unknown")

        output.append(
            f"[SOP Result {i}]\n"
            f"Source: {source}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(output)


if __name__ == "__main__":
    build_vectorstore()
    results = search_sop("What should happen if the temperature goes above 4 degrees?")
    print(format_sop_results(results))