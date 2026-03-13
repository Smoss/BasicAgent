import asyncio
from hashlib import md5
import os
from typing import List, Optional

import fandom  # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from agent.types import WikiPage
from tools.helldivers.training_manual_api import get_all_planets
from tools.lore_db import build_lore_doc, upsert_lore_documents
from utils.persona_config import load_persona_config

MAX_CHARACTERS = 4000

SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    (
        "You distill Fandom wiki pages into concise lore notes for roleplay agents.\n"
        "Keep critical facts, factions, relationships, locations, timeline beats, and actionable details.\n"
        "- Tone: neutral, compact, in-universe accurate.\n"
        f"- Max {MAX_CHARACTERS} characters.\n"
        "- Include the page title at the top as 'Title: <title>'.\n\n"
        "Title: {title}\n"
        "URL: {url}\n"
        "Raw Summary:\n{raw}\n\n"
        "Concise Lore Notes:"
    )
)


def _slugify(title: str) -> str:
    title = title.replace('"', "_")
    return title.replace(" ", "_")


async def fetch_planet_page(wiki: str, planet_name: str) -> WikiPage:
    url = f"https://{wiki}.fandom.com/wiki/{_slugify(planet_name)}"
    content = ""
    try:
        print(f"Fetching page '{planet_name}' from {wiki}")
        fandom.set_wiki(wiki)
        page = fandom.page(planet_name)
        content = page.plain_text
    except Exception as e:
        print(f"Failed to fetch page '{planet_name}': {e}")
        return WikiPage(title=planet_name, url=url, raw="")
    print(f"Fetched page '{planet_name}' from {wiki}")
    return WikiPage(title=planet_name, url=url, raw=content)


async def fetch_planet_pages(wiki: str, planet_names: List[str]) -> List[WikiPage]:
    """Fetch Fandom pages for planet names using plain text content"""
    try:
        fandom.set_wiki(wiki)
    except Exception as e:
        raise RuntimeError(f"Failed to set fandom wiki '{wiki}': {e}")

    pages = await asyncio.gather(
        *[fetch_planet_page(wiki, planet_name) for planet_name in planet_names]
    )
    print(f"Fetched {len(pages)} planet pages from {len(planet_names)} planet names")
    return pages


def summarize_pages(
    model: str, pages: List[WikiPage], save_dir: Optional[str] = None
) -> List[WikiPage]:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print("-" * 80)
    print(f"Summarizing {len(pages)} planet pages")

    llm = OllamaLLM(model=model, num_gpu=-1, num_ctx=16384, reasoning=False)
    print(f"Using model: {model}")
    print(f"Using save directory: {save_dir}")
    print(f"Using prompt template: {SUMMARIZE_PROMPT}")

    summarized: List[WikiPage] = []
    for p in pages:
        title = p.title
        url = p.url
        raw = p.raw
        if not raw:
            summarized.append(WikiPage(title=title, url=url, raw=""))
            continue

        prompt = SUMMARIZE_PROMPT.format_messages(title=title, url=url, raw=raw)
        if save_dir:
            with open(
                os.path.join(save_dir, f"{_slugify(title)}_prompt.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(prompt))

        summary = llm.invoke(prompt)
        text = summary if isinstance(summary, str) else str(summary)

        if save_dir:
            with open(
                os.path.join(save_dir, f"{_slugify(title)}_summary.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)

        summarized.append(WikiPage(title=title, url=url, raw=text))
    return summarized


def store_planet_pages(
    *,
    pages: List[WikiPage],
    wiki: str,
    character_name: str,
) -> List[str]:
    docs = [
        build_lore_doc(
            summary=p.raw,
            title=p.title,
            url=p.url,
            wiki=wiki,
            character_name=character_name,
            extra_metadata={"ingest_kind": "planet_summary"},
            doc_id=md5(p.title.encode("utf-8")).hexdigest(),
        )
        for p in pages
    ]
    return upsert_lore_documents(docs, collection_name=f"{character_name}_planets")


def ingest_planet_lore(
    *, persona_dir: str, model: str = "gpt-oss:20b", save_dir: Optional[str] = None
) -> dict:
    cfg = load_persona_config(persona_dir)

    # Get all planets from the API
    planets = get_all_planets()
    planet_names = [planet.name.replace(" ", "_") for planet in planets]

    print(f"Found {len(planet_names)} planets to ingest")
    print(f"Planet names: {planet_names[:5]}...")  # Show first 5 as preview

    # Fetch Fandom pages for each planet
    pages = asyncio.run(fetch_planet_pages(cfg.fandom_wiki, planet_names))

    # Summarize the pages using LLM
    summarized = summarize_pages(model, pages, save_dir)

    # Store in ChromaDB
    ids = store_planet_pages(
        pages=summarized, wiki=cfg.fandom_wiki, character_name=cfg.character_name
    )

    return {
        "persona": cfg.character_name,
        "wiki": cfg.fandom_wiki,
        "planets": planet_names,
        "stored_ids": ids,
        "count": len(ids),
        "successful_fetches": len(pages),
        "total_planets": len(planet_names),
    }
