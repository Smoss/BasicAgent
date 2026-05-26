from langchain.tools import Tool
from agent.state import State, StateUpdate
from tools.helldivers.training_manual_api import (
    get_campaigns,
    get_current_status,
    get_major_orders,
    get_past_week_news,
)


def route_lore_retrieval(state: State) -> str:
    if (tool_messages := state.get("tool_messages", [])) and tool_messages[
        -1
    ].tool_calls:  # type: ignore
        return "lore_retrieval"

    return "chatbot"


class LoreRetrievalNode:
    def __init__(self, tools: dict[str, Tool]):
        self.tools = {tool.name: tool for tool in tools.values()}
        self.tools_to_state_key = {tool.name: key for key, tool in self.tools.items()}

    def __call__(self, state: State) -> StateUpdate:
        if tool_messages := state.get("tool_messages"):
            tool_message = tool_messages[-1]
        else:
            raise ValueError("No tool messages in state")

        result: StateUpdate = {"tool_messages": []}
        for tool_call in tool_message.tool_calls:  # type: ignore
            tool_name = tool_call["name"]
            if tool_name not in self.tools:
                raise ValueError(f"Tool {tool_name} not found")
            tool_args = tool_call["args"]
            tool_inputs = {}
            if "__arg1" in tool_args:
                raw_ids = tool_args["__arg1"]
            elif "ids" in tool_args:
                raw_ids = tool_args["ids"]
            else:
                raise ValueError(f"Tool {tool_name} has no ids")

            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            tool_inputs["ids"] = raw_ids
            call_result = self.tools[tool_name](tool_inputs)  # type: ignore
            result[self.tools_to_state_key[tool_name]] = call_result  # type: ignore
        return result


def retrieve_campaigns(_: State) -> StateUpdate:
    """Step 3: Retrieve campaigns"""
    campaigns = get_campaigns()
    return {"active_campaigns": campaigns}  # type: ignore


def retrieve_major_orders(_: State) -> StateUpdate:
    """Step 4: Retrieve major orders"""
    major_orders = get_major_orders()
    return {"active_major_orders": major_orders}  # type: ignore


def retrieve_current_status(_: State) -> StateUpdate:
    """Step 5: Retrieve current status"""
    current_status = get_current_status()
    return {"current_status": current_status}  # type: ignore


def retrieve_news(state: State) -> StateUpdate:
    """Step 6: Retrieve news"""
    news = get_past_week_news(state)
    return {"past_week_news": news}  # type: ignore
