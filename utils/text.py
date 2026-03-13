from config import DISCORD_MESSAGE_LIMIT


def chunk_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split a long message into chunks that fit within the character limit.

    Splits on newlines when possible, falls back to character-level splitting
    for single lines that exceed the limit.
    """
    if len(text) <= limit:
        return [text]

    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > limit:  # +1 for newline
            if current_chunk:
                chunks.append(current_chunk.rstrip())
                current_chunk = line
            else:
                # Single line exceeds limit — split by character count
                chunks.append(line[:limit])
                current_chunk = line[limit:]
        else:
            current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.rstrip())

    return chunks
