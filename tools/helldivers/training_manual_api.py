import logging

from agent.state import State
import requests

from config import HELLDIVERS_API_BASE_URL
from tools.helldivers.training_manual_types import (
    CampaignPlanet,
    CurrentStatus,
    MajorOrder,
    MajorOrderTask,
    News,
    Planet,
)

logger = logging.getLogger(__name__)


def get_campaigns() -> list[CampaignPlanet]:
    """Get a list of all active planets that are currently being contested"""
    campaigns_raw = requests.get(f"{HELLDIVERS_API_BASE_URL}war/campaign")
    campaigns = campaigns_raw.json()
    campaigns = [CampaignPlanet(**campaign) for campaign in campaigns]
    planets = get_all_planets()
    for campaign in campaigns:
        campaign.environmentals = planets[campaign.planetIndex].environmentals
    return campaigns


def get_major_orders() -> list[MajorOrder]:
    """Get a list of all major orders that are currently active"""
    major_orders_raw = requests.get(f"{HELLDIVERS_API_BASE_URL}war/major-orders")
    major_orders = major_orders_raw.json()
    major_orders = [
        MajorOrder(
            id=major_order["id32"],
            expiresIn=major_order["expiresIn"],
            overrideBrief=major_order["setting"]["overrideBrief"],
            description=major_order["setting"]["taskDescription"],
            tasks=[
                MajorOrderTask(
                    type=task["type"],
                    target_value=task["values"][2],
                    current_value=progress,
                )
                for task, progress in zip(
                    major_order["setting"]["tasks"], major_order["progress"]
                )
            ],
        )
        for major_order in major_orders
    ]
    return major_orders


def get_current_status() -> CurrentStatus:
    """Get the current status, including time and current events"""
    current_status_raw = requests.get(f"{HELLDIVERS_API_BASE_URL}war/status")
    current_status = current_status_raw.json()
    current_status = CurrentStatus(**current_status)
    return current_status


def get_past_week_news(state: State) -> list[News]:
    """Get the past week's news"""
    news_dict = {}
    for i in range(7):
        daily_news_raw = requests.get(
            f"{HELLDIVERS_API_BASE_URL}war/news?from={state['current_status'].time - i * 86400}"
        ).json()
        daily_news = [News(**news) for news in daily_news_raw]
        for news in daily_news:
            news_dict[news.id] = news
    past_week_news = list(news_dict.values())
    past_week_news.sort(key=lambda x: x.published)
    return past_week_news


def get_all_planets() -> list[Planet]:
    """Get a list of all planets in the game"""
    planets_raw = requests.get(f"{HELLDIVERS_API_BASE_URL}planets")
    planets = planets_raw.json()
    logger.info("Found %d planets", len(planets))
    planets = [Planet(**planet) for planet in planets.values()]
    return planets
