test:
	uv run pytest tests/ -v

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run mypy .

run-agent:
	if [ -z "$(model)" ]; then \
		echo "Defaulting to model: gpt-oss:20b"; \
	else \
		echo "Running agent with model: $(model)"; \
	fi
	uv run python -m scripts.agent_cli run_agent --model=$(if $(model),$(model),gpt-oss:20b)

embed-voice-lines:
	uv run python -m scripts.embed_voice_lines \
		--config_path=$(if $(config_path),$(config_path),personas\democracy_officer)

run-discord:
	uv run python -m scripts.discord_bridge run \
		--context_model=$(if $(context_model),$(context_model),qwen3:8b) \
		--chat_model=$(if $(chat_model),$(chat_model),gemma3:12b) \
		--mention_only=$(if $(mention_only),$(mention_only),True) \
		--config_path=$(if $(config_path),$(config_path),personas\democracy_officer) \
		--max_voice_lines=$(if $(max_voice_lines),$(max_voice_lines),13) \
		--context_window=$(if $(context_window),$(context_window),20000) \
		$(if $(allowed_channels),--allowed_channels="$(allowed_channels)",) \
		$(if $(token),--token="$(token)",) \
		$(if $(debug),--debug=$(debug),)

ingest-fandom-wiki:
	uv run python -m scripts.ingest_fandom_lore \
		--model=$(if $(model),$(model),gemma3:4b) \
		--persona_dir=$(if $(persona_dir),$(persona_dir),personas\democracy_officer) \
		--save_dir=$(if $(save_dir),$(save_dir),personas\democracy_officer\debug)

ingest-planet-lore:
	uv run python -m scripts.ingest_planet_lore \
		--model=$(if $(model),$(model),gemma3:4b) \
		--persona_dir=$(if $(persona_dir),$(persona_dir),personas\democracy_officer) \
		--save_dir=$(if $(save_dir),$(save_dir),personas\democracy_officer\debug\planet_lore)