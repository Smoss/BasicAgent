from datetime import timedelta
from pydantic import BaseModel
from typing import Any, Optional


class Biome(BaseModel):
    slug: str
    description: str


class Environmental(BaseModel):
    name: str
    description: str


class Planet(BaseModel):
    name: str
    sector: str
    biome: Biome | None = None
    environmentals: list[Environmental]


class MajorOrderTask(BaseModel):
    type: int
    target_value: int
    current_value: int


class CampaignPlanet(BaseModel):
    planetIndex: int
    name: str
    faction: str
    players: int
    percentage: float
    defense: bool
    majorOrder: bool = False
    biome: Biome | None = None
    expireDateTime: Optional[float] = None


class MajorOrder(BaseModel):
    id: int
    expiresIn: int
    overrideBrief: str
    description: str
    tasks: list[MajorOrderTask]


class CurrentEvent(BaseModel):
    eventId: int
    id32: int
    title: str
    message: str


class CurrentStatus(BaseModel):
    globalEvents: list[CurrentEvent]
    time: int


class News(BaseModel):
    id: int
    published: int
    type: int
    tagIds: list[Any]
    message: str


def convert_planet_list(active_campaigns: list[CampaignPlanet]) -> str:
    current_planets = ""
    for campaign in active_campaigns:
        direction = (
            "under attack by"
            if campaign.defense
            else "being defended against an assault by"
        )
        reclaimed = "reclaimed" if not campaign.defense else "invaded"
        current_planets += f"{campaign.name} is {direction} {campaign.faction} with {round(campaign.players, -2):,} players. The planet is {campaign.percentage}% {reclaimed}.\n"
        current_planets += (
            f"The planet's biome is a {campaign.biome.slug}. Described as: {campaign.biome.description}"
            if campaign.biome
            else ""
        )
    return current_planets


def convert_major_order_list(active_major_orders: list[MajorOrder]) -> str:
    current_major_orders = ""
    for major_order in active_major_orders:
        description = f" - {major_order.description}" if major_order.description else ""
        current_major_orders += f"{major_order.overrideBrief}{description}\n"
        for task in major_order.tasks:
            if task.target_value > 0 and task.current_value != task.target_value:
                current_major_orders += f"Currently we are at {task.current_value / task.target_value * 100:.2f}% of the way to our goal.\n"
            elif task.target_value == task.current_value:
                current_major_orders += "We have completed the major order.\n"

        expiration_timer = timedelta(seconds=major_order.expiresIn)
        current_major_orders += f"The major order needs to be completed in {expiration_timer.days} days, {expiration_timer.seconds // 3600} hours, {expiration_timer.seconds // 60 % 60} minutes.\n"
        current_major_orders += "\n\n"
    return current_major_orders


def convert_news_list(past_week_news: list[News]) -> str:
    current_news = ""
    for news in past_week_news:
        current_news += f"{news.message}\n"
    return current_news


def convert_current_event_list(current_events: list[CurrentEvent]) -> str:
    current_event_str = ""
    for event in current_events:
        if event.message and event.title:
            current_event_str += f"{event.title}: {event.message}\n"
    return current_event_str
