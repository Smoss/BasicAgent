from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from tools.helldivers.training_manual_types import (
    CampaignPlanet,
    CurrentStatus,
    MajorOrder,
    News,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int
    retrieved_lore_docs: list[Document]  # Documents from initial vector search
    retrieved_style_docs: list[Document]  # Style/voice line documents
    retrieved_context_docs: list[Document]  # Additional context documents
    retrieved_planet_lore: list[Document]  # Planet lore documents
    active_campaigns: list[CampaignPlanet]  # Active campaigns
    active_major_orders: list[MajorOrder]  # Active major orders
    past_week_news: list[News]  # News from the past week
    current_status: CurrentStatus  # Current status, including time and current events
    tool_messages: Annotated[list[BaseMessage], add_messages]


class StateUpdate(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int
    retrieved_lore_docs: list[Document]
    retrieved_style_docs: list[Document]
    retrieved_context_docs: list[Document]
    retrieved_planet_lore: list[Document]
    active_campaigns: list[CampaignPlanet]
    active_major_orders: list[MajorOrder]
    past_week_news: list[News]
    current_status: CurrentStatus
    tool_messages: Annotated[list[BaseMessage], add_messages]
