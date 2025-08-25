

from agent.state import State
from tools.helldivers.training_manual_api import get_campaigns, get_current_status, get_major_orders, get_past_week_news


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