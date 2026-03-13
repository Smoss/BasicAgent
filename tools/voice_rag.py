import logging
from hashlib import md5
from typing import Any, List, Optional
from langchain_core.retrievers import BaseRetriever
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DB_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_voice_store(character_name: str) -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_gpu=-1)
    return Chroma(
        collection_name=f"{character_name}_voice_lines",
        persist_directory=f"{CHROMA_DB_PATH}/{character_name}",
        embedding_function=embeddings,
    )


def populate_voice_lines(path: str, character_name: str):
    store = get_voice_store(character_name)
    df = pd.read_csv(path)
    docs = df.apply(  # type: ignore
        lambda row: Document(
            page_content=row["voice_line"],
            metadata={
                "trigger": row["trigger"],
                "context": row["context"],
            },
            id=md5(row["voice_line"].encode("utf-8")).hexdigest(),
        ),
        axis=1,
    ).tolist()
    ids: List[str] = store.get().get("ids", [])  # type: ignore

    # Delete existing voice lines
    if ids:
        logger.info("Deleting %d voice lines", len(ids))
        store.delete(ids=ids)

    # Add new voice lines
    if docs:
        unique_docs = {doc.id: doc for doc in docs}
        logger.info("Adding %d voice lines", len(unique_docs))
        store.add_documents(
            documents=list(unique_docs.values()), ids=list(unique_docs.keys())
        )  # type: ignore


def build_voice_retriever(
    character_name: str,
    max_voice_lines: int = 6,
    where: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    store = get_voice_store(character_name)

    kwargs: dict[str, Any] = {"k": max_voice_lines}
    if where:
        kwargs["filter"] = where
    return store.as_retriever(search_kwargs=kwargs)


def select_voice_lines(store: Chroma, query: Optional[str], k: int) -> List[str]:
    retriever = store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query or "")
    return [d.page_content for d in results]
