"""RAG tools for the Sprints Advisor: ingest curriculum documents into ChromaDB
and search them. Supports .md, .txt and .pdf files."""
import logging
import time
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "curriculum_docs"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "sprints_curriculum"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
LATENCY_BUDGET_MS = 1000


@lru_cache(maxsize=1)
def _embedding_function() -> DefaultEmbeddingFunction:
    """Load the local embedding model once (no API key or GPU needed)."""
    return DefaultEmbeddingFunction()


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    """Open the persistent ChromaDB store once."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_collection():
    """Return the curriculum collection, creating it if needed."""
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _read_file(path: Path) -> str:
    """Read the text of a curriculum file (.md, .txt or .pdf)."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def ingest_documents() -> int:
    """Load every curriculum file, split it into chunks, embed and store them.
    Rebuilds the collection from scratch, so it is safe to re-run.
    Returns the number of chunks stored."""
    try:
        _client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        for i, chunk in enumerate(splitter.split_text(_read_file(path))):
            ids.append(f"{path.stem}-{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk": i})

    if not documents:
        raise FileNotFoundError(f"No curriculum documents found in {DOCS_DIR}")
    _get_collection().add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def retrieve(query: str, k: int = 3) -> tuple[list[dict], float]:
    """Return the k most similar chunks and the retrieval time in milliseconds.
    Each hit has: text, source, distance (lower = more similar)."""
    start = time.perf_counter()
    result = _get_collection().query(query_texts=[query], n_results=k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > LATENCY_BUDGET_MS:
        logger.warning("Retrieval took %.0f ms (budget %d ms)", elapsed_ms, LATENCY_BUDGET_MS)
    hits = [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
    return hits, elapsed_ms


@tool
def search_curriculum(query: str) -> str:
    """Search the Sprints curriculum documents (program overview, modules,
    grading policy, career support, FAQ) and return the most relevant passages
    with their source file. Use it to answer learner questions about the bootcamp."""
    try:
        hits, _ = retrieve(query)
    except Exception as e:
        return f"Unable to verify: curriculum search failed ({e})."
    if not hits:
        return "Unable to verify: no relevant curriculum content was found."
    return "\n\n".join(
        f"[{h['source']}] (distance {h['distance']:.2f}) {h['text']}" for h in hits
    )


if __name__ == "__main__":
    print(f"Ingested {ingest_documents()} chunks")
    retrieve("warm up")  # first call loads the model, so it is not timed below
    hits, ms = retrieve("What is the attendance policy?")
    print(f"Retrieval took {ms:.0f} ms")
    for h in hits:
        print(h["source"], round(h["distance"], 2), h["text"][:80].replace("\n", " "))