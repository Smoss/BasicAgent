import os
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import pandas as pd

df = pd.read_csv("reviews.csv")
embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=-1)

db_location = "./chroma_db"
add_docs = not os.path.exists(db_location)

if add_docs:
    documents = df.apply(  # type: ignore
        lambda row: Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"date": row["Date"], "rating": row["Rating"]},
            id=str(row.index),
        ),
        axis=1,
    ).tolist()
    ids = df.index.map(str).tolist()

vector_store = Chroma(
    collection_name="reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_docs:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
