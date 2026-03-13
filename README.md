# Monterrey v1

A character-driven Discord bot powered by LangGraph that roleplays as configurable personas using triple RAG (lore, voice lines, live game data) and local LLMs via Ollama. Built for the Helldivers 2 universe but extensible to other settings.

## Architecture

```
User Message (Discord / CLI)
        │
        ▼
┌─────────────────────────────────────────┐
│          LangGraph State Machine        │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │      Retrieval Phase            │    │
│  │  1. Lore RAG (Fandom wiki)     │    │
│  │  2. Voice RAG (style lines)    │    │
│  │  3. Live data (Helldivers API) │    │
│  │  4. Context retrieval (LLM)    │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │      Chat Phase                 │    │
│  │  System prompt + context +      │    │
│  │  voice exemplars → response     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
        │
        ▼
    Bot Response
```

**Multi-model strategy:** A smaller model handles context retrieval decisions while a larger model generates responses. Embeddings use `nomic-embed-text` via Ollama.

## Tech Stack

| Layer | Components |
|-------|-----------|
| Agent Framework | LangGraph, LangChain |
| LLM Inference | Ollama (local) |
| Vector DB | ChromaDB (persistent) |
| Embeddings | nomic-embed-text (Ollama) |
| External Data | Helldivers Training Manual API, Fandom Wiki |
| Discord | discord.py |
| Package Manager | uv |

## Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.com/)** installed and running with models pulled:
  ```bash
  ollama pull nomic-embed-text   # embeddings (required)
  ollama pull gemma3:12b         # chat model (or your preferred model)
  ollama pull qwen3:8b           # context model (or your preferred model)
  ```
- **Discord bot token** (for Discord mode)

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Create a persona directory** (e.g. `personas/democracy_officer/`):
   ```
   personas/democracy_officer/
   ├── config.yaml           # Character config
   ├── system_prompt.txt     # Character personality prompt
   ├── voice_lines.txt       # Optional: voice lines for style (one per line)
   └── fandom_topics.txt     # Topics to ingest from Fandom wiki (one per line)
   ```

   Example `config.yaml`:
   ```yaml
   character_name: democracy_officer
   system_prompt_path: system_prompt.txt
   voice_lines_path: voice_lines.txt
   fandom_wiki: helldivers
   ```

3. **Set environment variables** (optional):
   ```bash
   # Create a .env file
   DISCORD_BOT_TOKEN=your_token_here
   ```

4. **Ingest data** into ChromaDB:
   ```bash
   # Ingest Fandom wiki lore
   make ingest-fandom-wiki config_path=personas/democracy_officer

   # Ingest Helldivers planet data
   make ingest-planet-lore config_path=personas/democracy_officer

   # Embed voice lines from CSV
   make embed-voice-lines config_path=personas/democracy_officer
   ```

## Usage

### Discord Bot

```bash
make run-discord token=YOUR_TOKEN config_path=personas/democracy_officer
```

Options:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `token` | `DISCORD_BOT_TOKEN` env var | Discord bot token |
| `config_path` | `personas/democracy_officer` | Persona directory |
| `chat_model` | `gemma3:12b` | Model for response generation |
| `context_model` | `qwen3:8b` | Model for context retrieval |
| `context_window` | `16384` | LLM context size |
| `max_voice_lines` | `3` | Voice exemplars injected per response |
| `mention_only` | `True` | Only reply when mentioned |
| `allowed_channels` | (all) | Comma-separated channel ID whitelist |
| `debug` | `False` | Enable debug logging |

### CLI Agent

```bash
make run-agent model=gpt-oss:20b
```

## Persona Configuration

Personas are defined by a directory containing a `config.yaml` and supporting files. The `personas/` directory is gitignored — create your own locally.

**`config.yaml` fields:**
- `character_name` — Name used for ChromaDB collections and display
- `system_prompt_path` — Path (relative to persona dir) to the system prompt
- `voice_lines_path` — Optional path to voice lines file
- `fandom_wiki` — Fandom wiki identifier for lore ingestion

## Triple RAG System

1. **Lore RAG** — Fetches and summarizes Fandom wiki pages, stores embeddings in ChromaDB. Retrieved via vector search at query time.
2. **Voice RAG** — Embeds character voice lines for semantic retrieval, injecting style exemplars into the prompt.
3. **Planet RAG** — Fetches planet data from the Helldivers API and Fandom, with LLM-driven selection of relevant planets per query.

## Make Commands

| Command | Description |
|---------|-------------|
| `make format` | Auto-format with ruff |
| `make lint` | Lint (ruff) and type check (mypy) |
| `make run-agent` | Run the interactive CLI agent |
| `make run-discord` | Start the Discord bot |
| `make embed-voice-lines` | Load voice lines into ChromaDB |
| `make ingest-fandom-wiki` | Fetch and summarize Fandom wiki pages |
| `make ingest-planet-lore` | Fetch planet data from Helldivers API + Fandom |
