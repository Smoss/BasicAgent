from hashlib import md5
from typing import List, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_voice_retriever(path: str) -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=-1)
    store = Chroma(
        collection_name="voice_lines",
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    docs: List[Document] = []
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            text = ln.strip()
            if not text:
                continue
            text = text[:200]
            doc_id = md5(text.encode("utf-8")).hexdigest()
            ids.append(doc_id)
            docs.append(Document(page_content=text, metadata={}, id=doc_id))

    if ids:
        try:
            store.delete(ids=ids)
        except Exception:
            # Best-effort cleanup; proceed to add
            pass
        store.add_documents(documents=docs, ids=ids)

    return store


def select_voice_lines(store: Chroma, query: Optional[str], k: int) -> List[str]:
    retriever = store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query or "")
    return [d.page_content for d in results]
