from agent.types import DocumentIdTitle


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

    All available planets in the retrieve_planet_lore database:
    {"\n".join([str(planet) for planet in all_available_planets])}
    "Additional context:
    {additional_context}

    User message: {user_text}
    You must make a tool call to retrieve the additional planet information. The argument to the tool call is a list of ids, even if there is only one.
    If the player references a planet or asks questions about planets, you must make a tool call to retrieve the additional planet information.
    """
