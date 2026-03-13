import fire  # type: ignore

from tools.voice_rag import populate_voice_lines
from utils.persona_config import load_persona_config


def main(config_path: str):
    persona_config = load_persona_config(config_path)
    assert persona_config.voice_lines_path

    populate_voice_lines(persona_config.voice_lines_path, persona_config.character_name)


if __name__ == "__main__":
    fire.Fire(main)
