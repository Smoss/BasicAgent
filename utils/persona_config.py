import os
from pydantic import BaseModel
import yaml


class PersonaConfig(BaseModel):
    character_name: str
    system_prompt_path: str
    voice_lines_path: str | None
    fandom_wiki: str


def load_persona_config(persona_dir: str) -> PersonaConfig:
    cfg_path = os.path.join(persona_dir, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PersonaConfig(
        character_name=data["character_name"],
        system_prompt_path=data.get("system_prompt_path", ""),
        voice_lines_path=data.get("voice_lines_path"),
        fandom_wiki=data.get("fandom_wiki", ""),
    )
