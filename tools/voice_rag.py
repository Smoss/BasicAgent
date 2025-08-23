from hashlib import md5
from typing import List, Optional
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_voice_retriever(path: str, character_name: str) -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=-1)
    store = Chroma(
        collection_name=f"{character_name}_voice_lines",
        persist_directory=f"./chroma_db/{character_name}",
        embedding_function=embeddings,
    )

    docs: List[Document] = []
    ids: List[str] = []
    df = pd.read_csv(path)
    documents = df.apply(lambda row: Document(page_content=row["voice_line"], metadata={'idx': row.index, 'trigger': row["trigger"], 'context': row["context"]}, id=md5(row["voice_line"].encode("utf-8")).hexdigest()), axis=1).tolist()

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
