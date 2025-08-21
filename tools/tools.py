from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.tools import Tool
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.tools import tool

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

search_tool = DuckDuckGoSearchResults(max_results=5, output_format="list")
# search_tool = Tool(
#     name="search_tool", description="Search the web for information", func=search.run
# )

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
