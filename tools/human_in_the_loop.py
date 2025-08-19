

from typing import Annotated
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt


@tool
def human_assist(query: str) -> str:
    """Request assistance from a human."""
    try:
        human_response = interrupt({"query": query})
    except GraphInterrupt as e:
        print('Exception', e)
        # return Command(update={"response": [ToolMessage(str(e), tool_call_id=tool_call_id)]})
    # human_response = interrupt({"query": query})
    return human_response["data"]
# def human_assist(
#     name: str,
#     birthday: str,
#     tool_call_id: Annotated[str, InjectedToolCallId]
#     ) -> Command:
#     """Ask a human for assistance"""
#     print(name, birthday, tool_call_id)
#     print("Howdy, human!")
#     try:
#         human_response = interrupt(
#             {
#                 "query": f"Is the name {name} and birthday {birthday} correct?",
#             }
#         )
#     except GraphInterrupt as e:
#         print(e)
#         return Command(update={"response": [ToolMessage(str(e), tool_call_id=tool_call_id)]})
#     print("WHY AM I HERE?")
#     print(human_response)
#     if human_response.get("correct", "").lower().startswith("y"):
#         verified_name = name
#         verified_birthday = birthday
#         response = "Correct"
#     else:
#         verified_name = human_response.get("name", name)
#         verified_birthday = human_response.get("birthday", birthday)
#         response = f"Made a correction: {human_response}"

#     state_update = {
#         "name": verified_name,
#         "birthday": verified_birthday,
#         "response": [ToolMessage(response, tool_call_id=tool_call_id)],
#     }
#     return Command(update=state_update)