from unittest.mock import MagicMock

from agent.helldivers.nodes import route_lore_retrieval


def test_routes_to_lore_retrieval_when_tool_calls_present():
    mock_msg = MagicMock()
    mock_msg.tool_calls = [{"name": "retrieve_planet_lore", "args": {"ids": ["abc"]}}]
    state = {"tool_messages": [mock_msg]}
    assert route_lore_retrieval(state) == "lore_retrieval"  # type: ignore


def test_routes_to_chatbot_when_no_tool_calls():
    mock_msg = MagicMock()
    mock_msg.tool_calls = []
    state = {"tool_messages": [mock_msg]}
    assert route_lore_retrieval(state) == "chatbot"  # type: ignore


def test_routes_to_chatbot_when_tool_messages_empty():
    state = {"tool_messages": []}
    assert route_lore_retrieval(state) == "chatbot"  # type: ignore


def test_routes_to_chatbot_when_tool_messages_missing():
    state = {}
    assert route_lore_retrieval(state) == "chatbot"  # type: ignore
