import json
import os

import fire  # type: ignore

from agent.planet_ingestion_agent import ingest_planet_lore


def main(persona_dir: str, model: str = "gpt-oss:20b", save_dir: str | None = None):
    if not os.path.isdir(persona_dir):
        raise SystemExit(f"Persona directory not found: {persona_dir}")
    result = ingest_planet_lore(persona_dir=persona_dir, model=model, save_dir=save_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
