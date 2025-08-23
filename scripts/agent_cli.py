import fire  # type: ignore

from agent.character_agent import SimpleCharacterAgent


def main():
    # Expose the Agent class via Fire. Example:
    # python scripts/agent_cli.py run_agent --model=gpt-oss:20b
    fire.Fire(SimpleCharacterAgent)


if __name__ == "__main__":
    main()
