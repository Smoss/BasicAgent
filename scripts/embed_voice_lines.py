import yaml
import fire  # type: ignore

from tools.voice_rag import populate_voice_lines


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    populate_voice_lines(config["voice_lines_path"], config["character_name"])


if __name__ == "__main__":
    fire.Fire(main)
