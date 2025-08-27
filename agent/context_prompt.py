from agent.types import DocumentIdTitle


    # "Additional context:
    # {additional_context}

    # Please provide any required tool calls to retrieve the additional documents and planets. Use the ids from the DocumentIdTitle objects to retrieve the documents and planets."

    

    # Already retrieved documents:
    # {"\n".join([f"- {title}: {content}" for title, content in zip(already_retrieved_titles, already_retrieved_content)])}

    

    # All available documents in the retrieve_context_lore database:
    # {"\n".join([str(doc) for doc in all_available_documents])}
def create_context_prompt(
    system_prompt_text: str,
    user_text: str,
    already_retrieved_titles: list[str],
    already_retrieved_content: list[str],
    all_available_documents: list[DocumentIdTitle],
    all_available_planets: list[DocumentIdTitle],
    additional_context: str,
) -> str:
    return f"""You are a context retrieval assistant for a character AI system.

    System prompt for the character:
    {system_prompt_text}

    All available planets in the retrieve_planet_lore database:
    {"\n".join([str(planet) for planet in all_available_planets])}
    "Additional context:
    {additional_context}

    User message: {user_text}
    You must make a tool call to retrieve the additional planet information. The argument to the tool call is a list of ids, even if there is only one.
    """
