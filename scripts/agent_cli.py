import fire  # type: ignore

from agent.agent import Agent


def main():
    # Expose the Agent class via Fire. Example:
    # python scripts/agent_cli.py run_agent --model=gpt-oss:20b
    fire.Fire(Agent)


if __name__ == "__main__":
    main()
