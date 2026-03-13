from tools.helldivers.training_manual_types import (
    Biome,
    CampaignPlanet,
    CurrentEvent,
    Environmental,
    MajorOrder,
    MajorOrderTask,
    News,
    convert_current_event_list,
    convert_major_order_list,
    convert_news_list,
    convert_planet_list,
)


from typing import Any


def _make_campaign(**overrides: Any) -> CampaignPlanet:
    defaults: dict[str, Any] = dict(
        planetIndex=0,
        name="Malevelon Creek",
        faction="Automatons",
        players=12345,
        percentage=45.5,
        defense=True,
        biome=Biome(slug="swamp", description="A murky swamp world"),
        environmentals=[
            Environmental(name="Acid Storms", description="Corrosive rain")
        ],
    )
    defaults.update(overrides)
    return CampaignPlanet(**defaults)


def test_convert_planet_list_defense():
    campaigns = [_make_campaign(defense=True)]
    result = convert_planet_list(campaigns)
    assert "Malevelon Creek" in result
    assert "under attack by" in result
    assert "Automatons" in result
    assert "12,300" in result  # rounded to nearest 100
    assert "invaded" in result


def test_convert_planet_list_offense():
    campaigns = [_make_campaign(defense=False)]
    result = convert_planet_list(campaigns)
    assert "being defended against an assault by" in result
    assert "reclaimed" in result


def test_convert_planet_list_no_biome():
    campaigns = [_make_campaign(biome=None)]
    result = convert_planet_list(campaigns)
    assert "Malevelon Creek" in result


def test_convert_major_order_list():
    orders = [
        MajorOrder(
            id=1,
            expiresIn=90061,  # 1 day, 1 hour, 1 minute, 1 second
            overrideBrief="Liberate the sector",
            description="Free all planets",
            tasks=[MajorOrderTask(type=1, target_value=100, current_value=50)],
        )
    ]
    result = convert_major_order_list(orders)
    assert "Liberate the sector" in result
    assert "Free all planets" in result
    assert "50.00%" in result
    assert "1 days" in result


def test_convert_major_order_completed():
    orders = [
        MajorOrder(
            id=2,
            expiresIn=3600,
            overrideBrief="Done",
            description="",
            tasks=[MajorOrderTask(type=1, target_value=10, current_value=10)],
        )
    ]
    result = convert_major_order_list(orders)
    assert "completed" in result


def test_convert_news_list():
    news = [
        News(id=1, published=100, type=0, tagIds=[], message="War update 1"),
        News(id=2, published=200, type=0, tagIds=[], message="War update 2"),
    ]
    result = convert_news_list(news)
    assert "War update 1" in result
    assert "War update 2" in result


def test_convert_current_event_list():
    events = [
        CurrentEvent(eventId=1, id32=1, title="Alert", message="Enemy approaching"),
        CurrentEvent(eventId=2, id32=2, title="", message="No title event"),
    ]
    result = convert_current_event_list(events)
    assert "Alert: Enemy approaching" in result
    # Event with empty title should not appear
    assert "No title event" not in result
