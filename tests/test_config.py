import importlib


def test_config_defaults():
    import config

    importlib.reload(config)
    assert config.DEFAULT_CONTEXT_MODEL == "gpt-oss:20b"
    assert config.DEFAULT_CHAT_MODEL == "gpt-oss:20b"
    assert config.CHROMA_DB_PATH == "./chroma_db"
    assert (
        config.HELLDIVERS_API_BASE_URL == "https://helldiverstrainingmanual.com/api/v1/"
    )
    assert config.EMBEDDING_MODEL == "nomic-embed-text"
    assert config.DISCORD_MESSAGE_LIMIT == 1900
    assert config.DEFAULT_CONTEXT_WINDOW == 8192


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("DEFAULT_CONTEXT_MODEL", "test-model:7b")
    monkeypatch.setenv("CHROMA_DB_PATH", "/tmp/test_chroma")
    monkeypatch.setenv("DISCORD_MESSAGE_LIMIT", "2000")
    monkeypatch.setenv("DEFAULT_CONTEXT_WINDOW", "4096")

    import config

    importlib.reload(config)

    assert config.DEFAULT_CONTEXT_MODEL == "test-model:7b"
    assert config.CHROMA_DB_PATH == "/tmp/test_chroma"
    assert config.DISCORD_MESSAGE_LIMIT == 2000
    assert config.DEFAULT_CONTEXT_WINDOW == 4096

    # Clean up: reload with original env
    monkeypatch.delenv("DEFAULT_CONTEXT_MODEL")
    monkeypatch.delenv("CHROMA_DB_PATH")
    monkeypatch.delenv("DISCORD_MESSAGE_LIMIT")
    monkeypatch.delenv("DEFAULT_CONTEXT_WINDOW")
    importlib.reload(config)
