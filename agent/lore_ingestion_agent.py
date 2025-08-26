import os
from typing import List, Optional

import fandom  # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from tools.lore_db import build_lore_doc, upsert_lore_documents
from utils.persona_config import load_persona_config


SUMMARIZE_PROMPT = ChatPromptTemplate.from_template(
    (
        "You distill Fandom wiki pages into concise lore notes for roleplay agents.\n"
        "Keep critical facts, factions, relationships, locations, timeline beats, and actionable details.\n"
        "- Tone: neutral, compact, in-universe accurate.\n"
        "- Max ~1800 characters.\n"
        "- Include the page title at the top as 'Title: <title>'.\n\n"
        "Title: {title}\n"
        "URL: {url}\n"
        "Raw Summary:\n{raw}\n\n"
        "Concise Lore Notes:"
    )
)


def load_topics(persona_dir: str) -> List[str]:
    topics_path = os.path.join(persona_dir, "fandom_topics.txt")
    with open(topics_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def _slugify(title: str) -> str:
    title = title.replace('"', "_")
    return title.replace(" ", "_")


def flatten_content(title: str, url: str, section: dict) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if "content" in section:
        out.append({"title": title, "url": url, "raw": section.get("content", "")})

    for sub_section in section.get("sections", []):
        out.extend(
            flatten_content(
                f"{title} - {sub_section.get('title', '')}", url, sub_section
            )
        )
    return out


def fetch_fandom_pages(wiki: str, titles: list[str]) -> list[dict]:
    try:
        fandom.set_wiki(wiki)
    except Exception as e:
        raise RuntimeError(f"Failed to set fandom wiki '{wiki}': {e}")

    out: list[dict] = []
    for t in titles:
        url = f"https://{wiki}.fandom.com/wiki/{_slugify(t)}"
        content = ""
        try:
            content = fandom.page(t).content
        except Exception as e:
            # Leave raw empty if not found
            print(f"Failed to fetch page '{t}': {e}")
            pass
        if content:
            assert isinstance(content, dict)
            flattened = flatten_content(t, url, content)
            out.extend(flattened)
    print(f"Fetched {len(out)} sections from {len(titles)} pages")
    return out


def summarize_pages(
    model: str, pages: list[dict], save_dir: Optional[str] = None
) -> list[dict[str, str]]:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print("-" * 80)
    print(f"Summarizing {len(pages)} pages")

    llm = OllamaLLM(model=model, num_gpu=-1, num_ctx=16384, reasoning=False)
    print(f"Using model: {model}")
    print(f"Using save directory: {save_dir}")
    print(f"Using prompt template: {SUMMARIZE_PROMPT}")
    summarized: list[dict] = []
    for p in pages:
        title = p.get("title", "")
        url = p.get("url", "")
        raw = p.get("raw", "")
        if not raw:
            summarized.append({"title": title, "url": url, "summary": ""})
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
        summarized.append({"title": title, "url": url, "summary": text})
    return summarized


def store_pages(
    *,
    pages: list[dict],
    wiki: str,
    character_name: str,
) -> list[str]:
    docs = [
        build_lore_doc(
            summary=p.get("summary", ""),
            title=p.get("title", ""),
            url=p.get("url", ""),
            wiki=wiki,
            character_name=character_name,
            extra_metadata={"ingest_kind": "fandom_summary"},
        )
        for p in pages
    ]
    return upsert_lore_documents(docs, collection_name=f"{character_name}_lore")


def ingest_persona_fandom(
    *, persona_dir: str, model: str = "gpt-oss:20b", save_dir: Optional[str] = None
) -> dict:
    cfg = load_persona_config(persona_dir)
    topics = load_topics(persona_dir)

    pages = fetch_fandom_pages(cfg.fandom_wiki, topics)
    summarized = summarize_pages(model, pages, save_dir)
    ids = store_pages(
        pages=summarized, wiki=cfg.fandom_wiki, character_name=cfg.character_name
    )

    return {
        "persona": cfg.character_name,
        "wiki": cfg.fandom_wiki,
        "topics": topics,
        "stored_ids": ids,
        "count": len(ids),
        # "saved_dir": saved.get("dir"),
        # "saved_files_count": len(saved.get("files", [])),
    }
