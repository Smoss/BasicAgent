from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.tools import Tool
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.tools import tool
import fandom  # type: ignore

import json

load_dotenv()


@tool
def save_to_txt(text: str, filename: str = "research_output.txt") -> str:
    """Save the research output to a text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n{text}\n--- End of Research Output ---"
    with open(filename, "a") as file:
        file.write(formatted_text)

    return f"Research output saved to {filename}"


save_tool = Tool(
    name="save_to_txt",
    description="Save the research output to a text file",
    func=save_to_txt,
)

# @tool('search_tool')
# def search_tool(query: str) -> str:
#     """Search the web for information"""
#     return TavilySearch(max_results=5).invoke({"query": query})

search_tool = DuckDuckGoSearchResults(max_results=5, output_format="list")  # type: ignore
# search_tool = Tool(
#     name="search_tool", description="Search the web for information", func=search.run
# )

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())  # type: ignore


# --- Fandom tool builder ---
def build_fandom_search_tool(default_wiki: str) -> Tool:
    """
    Search the wiki and return a list of candidate article titles.
    Output JSON: { "wiki": str, "titles": [str, ...] }
    """

    def _fandom_search(query: str) -> str:
        try:
            fandom.set_wiki(default_wiki)
        except Exception as e:
            return json.dumps(
                {
                    "error": f"failed_to_set_wiki: {e}",
                    "wiki": default_wiki,
                }
            )

        results = fandom.search(query) or []
        titles = [title for title, _ in results]
        return json.dumps({"wiki": default_wiki, "titles": titles}, ensure_ascii=False)

    return Tool(
        name="fandom_search",
        description=(
            "Search the configured Fandom wiki and return JSON titles. "
            "Input: query string. Output JSON: {wiki, titles:[str,...]}."
        ),
        func=_fandom_search,
    )


def build_fandom_pages_tool(default_wiki: str):
    """
    Fetch summaries for a list of Fandom article titles.

    Input:
      - titles: list[str] (article titles)
      - sentences: int = -1 (max number of sentences per summary; -1 = full summary)

    Output JSON:
      {
        "wiki": str,
        "pages": [
          { "title": str, "url": str, "summary": str }
        ]
      }
    """

    def _slugify(title: str) -> str:
        return title.replace(" ", "_")

    def fandom_pages(titles: list[str], sentences: int = -1) -> str:
        try:
            fandom.set_wiki(default_wiki)
        except Exception as e:
            return json.dumps(
                {"error": f"failed_to_set_wiki: {e}", "wiki": default_wiki}
            )

        pages_out: list[dict] = []
        for t in (titles or [])[:10]:
            try:
                summ = fandom.summary(t, sentences=sentences)
                url = f"https://{default_wiki}.fandom.com/wiki/{_slugify(t)}"
                pages_out.append({"title": t, "url": url, "summary": summ})
                # print(f"Fetched summary for {t}: {summ}")
            except Exception:
                pages_out.append(
                    {
                        "title": t,
                        "url": f"https://{default_wiki}.fandom.com/wiki/{_slugify(t)}",
                        "summary": "",
                    }
                )

        return json.dumps(
            {"wiki": default_wiki, "pages": pages_out}, ensure_ascii=False
        )

    return Tool(
        name="fandom_pages",
        description=(
            "Fetch summaries for a list of Fandom article titles. "
            "Input: titles: list[str] (article titles) "
            "sentences: int = -1 (max number of sentences per summary; -1 = full summary) "
            "Output JSON: {wiki, pages:[{title,url,summary},...]}."
        ),
        func=fandom_pages,
    )
