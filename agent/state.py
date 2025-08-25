from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int
    retrieved_lore_docs: list[Document]  # Documents from initial vector search
    retrieved_style_docs: list[Document]  # Style/voice line documents
    retrieved_context_docs: list[Document]  # Additional context documents
