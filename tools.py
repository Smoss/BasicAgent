from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(text: str, filename: str="research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n{text}\n--- End of Research Output ---"
    with open(filename, "a") as file:
        file.write(formatted_text)
    
    return f"Research output saved to {filename}"

save_tool = Tool(
    name="save_to_txt",
    description="Save the research output to a text file",
    func=save_to_txt
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_tool",
    description="Search the web for information",
    func=search.run
)

api_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="wiki_tool",
    description="Search Wikipedia for information",
    func=api_wrapper.run
)