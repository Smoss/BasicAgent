import os

DEFAULT_CONTEXT_MODEL = os.environ.get("DEFAULT_CONTEXT_MODEL", "gpt-oss:20b")
DEFAULT_CHAT_MODEL = os.environ.get("DEFAULT_CHAT_MODEL", "gpt-oss:20b")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
HELLDIVERS_API_BASE_URL = os.environ.get(
    "HELLDIVERS_API_BASE_URL", "https://helldiverstrainingmanual.com/api/v1/"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
DISCORD_MESSAGE_LIMIT = int(os.environ.get("DISCORD_MESSAGE_LIMIT", "1900"))
DEFAULT_CONTEXT_WINDOW = int(os.environ.get("DEFAULT_CONTEXT_WINDOW", "8192"))
