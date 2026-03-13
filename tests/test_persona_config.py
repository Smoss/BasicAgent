import os
import tempfile

import yaml

from utils.persona_config import load_persona_config


def test_load_persona_config():
    data = {
        "character_name": "test_char",
        "system_prompt_path": "/tmp/prompt.txt",
        "voice_lines_path": "/tmp/voice.csv",
        "fandom_wiki": "helldivers",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(data, f)

        config = load_persona_config(tmpdir)
        assert config.character_name == "test_char"
        assert config.system_prompt_path == "/tmp/prompt.txt"
        assert config.voice_lines_path == "/tmp/voice.csv"
        assert config.fandom_wiki == "helldivers"


def test_load_persona_config_optional_voice_lines():
    data = {
        "character_name": "no_voice",
        "system_prompt_path": "/tmp/prompt.txt",
        "fandom_wiki": "helldivers",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(data, f)

        config = load_persona_config(tmpdir)
        assert config.character_name == "no_voice"
        assert config.voice_lines_path is None
