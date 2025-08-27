from langchain.tools import Tool
from agent.state import State
from tools.helldivers.training_manual_api import (
    get_campaigns,
    get_current_status,
    get_major_orders,
    get_past_week_news,
)

def route_lore_retrieval(state: State) -> State:
    if (tool_messages := state.get("tool_messages", [])) and tool_messages[-1].tool_calls:
        return "lore_retrieval"

    return "chatbot"

class LoreRetrievalNode:
    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}
    
    def __call__(self, state: State) -> State:
        if tool_messages := state.get("tool_messages"):
            tool_message = tool_messages[-1]
        else:
            raise ValueError("No tool messages in state")
            
        result = {
            "tool_messages": []
        }
        for tool_call in tool_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name not in self.tools:
                raise ValueError(f"Tool {tool_name} not found")
            tool_args = tool_call["args"]
            tool_inputs = {}
            if '__arg1' in tool_args:
                raw_ids = tool_args['__arg1']
            elif 'ids' in tool_args:
                raw_ids = tool_args['ids']
            else:
                raise ValueError(f"Tool {tool_name} has no ids")
            
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            tool_inputs['ids'] = raw_ids
            match tool_name:
                case "retrieve_planet_lore":
                    result["retrieved_planet_lore"] = self.tools["retrieve_planet_lore"](tool_inputs)
                case "retrieve_context_lore":
                    result["retrieved_context_docs"] = self.tools["retrieve_context_lore"](tool_inputs)
                case _:
                    raise ValueError(f"Tool {tool_name} not found")
        return result

def retrieve_campaigns(_: State) -> State:
    """Step 3: Retrieve campaigns"""
    campaigns = get_campaigns()
    return {"active_campaigns": campaigns}


def retrieve_major_orders(_: State) -> State:
    """Step 4: Retrieve major orders"""
    major_orders = get_major_orders()
    return {"active_major_orders": major_orders}


def retrieve_current_status(_: State) -> State:
    """Step 5: Retrieve current status"""
    current_status = get_current_status()
    return {"current_status": current_status}


def retrieve_news(state: State) -> State:
    """Step 6: Retrieve news"""
    news = get_past_week_news(state)
    return {"past_week_news": news}

def additional_documents_nodes(_: State) -> State:
    """Step 7: Retrieve additional documents"""
    
    return {"additional_documents": additional_documents}
