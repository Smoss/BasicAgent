import os
import hashlib
from typing import Any, Iterable, Optional

from langchain.tools import Tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from agent.types import DocumentIdTitle
from config import CHROMA_DB_PATH, EMBEDDING_MODEL


def _stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:40]


def get_lore_store(collection_name: str = "lore") -> Chroma:
    db_root = CHROMA_DB_PATH
    os.makedirs(db_root, exist_ok=True)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_gpu=-1)
    return Chroma(
        collection_name=collection_name,
        persist_directory=db_root,
        embedding_function=embeddings,
    )


def build_lore_doc(
    *,
    summary: str,
    title: str,
    url: str,
    wiki: str,
    character_name: str,
    raw_source: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
    doc_id: Optional[str] = None,
) -> Document:
    metadata: dict[str, Any] = {
        "title": title,
        "url": url,
        "wiki": wiki,
        "character": character_name,
        "source": "fandom",
    }
    if raw_source:
        metadata["raw_source_present"] = True
    if extra_metadata:
        metadata.update(extra_metadata)

    if doc_id is None:
        doc_id = _stable_id(wiki, title, character_name)
    return Document(page_content=summary, metadata=metadata, id=doc_id)


def upsert_lore_documents(
    docs: Iterable[Document],
    *,
    collection_name: str = "lore",
) -> list[str]:
    store = get_lore_store(collection_name)
    ids: list[str] = []
    add_docs: list[Document] = []
    add_ids: list[str] = []

    for doc in docs:
        doc_id = getattr(doc, "id", None) or _stable_id(doc.page_content)
        # Try to detect if the document already exists by id
        try:
            existing = store.get(ids=[doc_id])
            if existing and existing.get("ids"):
                ids.append(doc_id)
                continue
        except Exception:
            # If get by id fails, fall back to add
            pass
        add_docs.append(doc)
        add_ids.append(doc_id)
        ids.append(doc_id)

    if add_docs:
        store.add_documents(documents=add_docs, ids=add_ids)
    return ids


def search_lore(
    query: str,
    *,
    k: int = 5,
    collection_name: str = "lore",
    where: Optional[dict[str, Any]] = None,
) -> list[tuple[Document, float]]:
    store = get_lore_store(collection_name)
    # LangChain's similarity_search_with_score doesn't accept where, so we use retriever if filter needed
    if where:
        retriever = store.as_retriever(search_kwargs={"k": k, "filter": where})
        docs = retriever.invoke(query)
        # Scores not available via retriever; return zero scores
        return [(d, 0.0) for d in docs]
    return store.similarity_search_with_score(query, k=k)


def build_lore_retriever(
    *,
    collection_name: str = "lore",
    search_k: int = 5,
    where: Optional[dict[str, Any]] = None,
):
    store = get_lore_store(collection_name)
    kwargs: dict[str, Any] = {"k": search_k}
    if where:
        kwargs["filter"] = where
    return store.as_retriever(search_kwargs=kwargs)


def get_all_documents(
    *,
    store: Chroma,
) -> list[DocumentIdTitle]:
    """Get all document titles from the vector database"""

    # Get all documents from the collection
    all_docs = store.get()
    titles: list[DocumentIdTitle] = []
    for metadata, id in zip(all_docs.get("metadatas", []), all_docs.get("ids", [])):  # type: ignore
        if metadata:
            title = metadata.get("title", "Unknown")
            doc_id = id
            titles.append(DocumentIdTitle(doc_id=doc_id, doc_title=title))
    return titles


def build_lore_retriever_tool(
    *,
    store: Chroma,
    collection_name: str = "lore",
) -> Tool:
    def _retrieve_lore_by_ids(ids: list[str]) -> list[Document]:
        return store.get_by_ids(ids)

    return Tool(
        name=f"retrieve_{collection_name}_lore",
        description=f"Retrieve {collection_name} documents from the vector database using the list of ids provided. Ids are of the form md5 hashes. The ids are provided in the form of a list of strings.",
        func=_retrieve_lore_by_ids,
    )
