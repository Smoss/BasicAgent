from agent.types import DocumentIdTitle


def create_context_prompt(
    system_prompt_text: str,
    user_text: str,
    already_retrieved_titles: list[str],
    already_retrieved_content: list[str],
    all_available_titles: list[DocumentIdTitle],
    additional_context: str,
    num_documents: int = 5,
) -> str:
    return f"""You are a context retrieval assistant for a character AI system.

    System prompt for the character:
    {system_prompt_text}

    User message: {user_text}

    Already retrieved documents:
    {"\n".join([f"- {title}: {content}" for title, content in zip(already_retrieved_titles, already_retrieved_content)])}

    All available documents in the database:
    {"\n".join([f"- {title.doc_title} (ID: {title.doc_id})" for title in all_available_titles])}

    Additional context:
    {additional_context}

    Based on the user message, system prompt, and already retrieved documents, select at most {num_documents} additional documents from the available list that would provide useful context.
    Be sure to properly format the output as a list of DocumentIdTitle objects. Keep the document IDs as md5 hashes.
    Focus on gaps in knowledge or areas that would provide useful context."""
