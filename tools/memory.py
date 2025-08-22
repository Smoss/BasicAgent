import os
import uuid
from datetime import datetime
from typing import Any, List, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.tools import tool


def _get_memory_store() -> Chroma:
    """Create or return the persistent Chroma collection for long-term memories."""
    db_root = os.path.join(".", "chroma_db")
    os.makedirs(db_root, exist_ok=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=-1)
    return Chroma(
        collection_name="memories",
        persist_directory=db_root,
        embedding_function=embeddings,
    )


@tool
def memory_add(text: str, tags: Optional[str] = None) -> str:
    """Store a piece of information for future recall. Include optional comma-separated tags."""
    store = _get_memory_store()
    memory_id = str(uuid.uuid4())
    metadata: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if tags:
        metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    doc = Document(page_content=text, metadata=metadata, id=memory_id)
    store.add_documents(documents=[doc], ids=[memory_id])
    return f"memory_saved:id={memory_id}"


@tool
def memory_search(query: str, k: int = 5) -> str:
    """Search stored memories for items relevant to the query. Returns a compact JSON string."""
    store = _get_memory_store()
    results = store.similarity_search_with_score(query, k=k)
    compact: List[dict] = []
    for doc, score in results:
        compact.append(
            {
                "id": getattr(doc, "id", None),
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata,
            }
        )
    # Return as JSON-like string to keep tool outputs small and robust
    try:
        import json

        return json.dumps(compact, ensure_ascii=False)
    except Exception:
        return str(compact)
