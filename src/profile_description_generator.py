from __future__ import annotations

import json
from pathlib import Path
from typing import List

from clients.gemini_client import GeminiClient
from models import DescriptionResponse, Profile
from prompts.profile_description_generation_prompt import CHARACTER_PROFILE_GENERATION_PROMPT

PERSONALITY_TYPES: List[str] = [
    "INTJ",
    "INTP",
    "ENTJ",
    "ENTP",
    "INFJ",
    "INFP",
    "ENFJ",
    "ENFP",
    "ISTJ",
    "ISFJ",
    "ESTJ",
    "ESFJ",
    "ISTP",
    "ISFP",
    "ESTP",
    "ESFP",
]

OUTPUT_FILE = Path(__file__).with_name("profile_descriptions.json")


def build_prompt(personality_type: str) -> str:
    base_prompt = CHARACTER_PROFILE_GENERATION_PROMPT.format(personality_type=personality_type)
    return (
        f"{base_prompt}\n"
        "\n"
        "Output Format (JSON):\n"
        "{\n"
        "    \"description\": \"A 2-sentence description matching the character profile.\"\n"
        "}\n"
        "Only return valid JSON with a single field named \"description\"."
    )


def main() -> None:
    client = GeminiClient()
    profiles: List[Profile] = []

    for personality_type in PERSONALITY_TYPES:
        prompt = build_prompt(personality_type)
        response = client.generate_structured_json(prompt, DescriptionResponse.model_json_schema())
        description_response = DescriptionResponse.model_validate(response)
        profiles.append(
            Profile(
                personality_type=personality_type,
                description=description_response.description,
            )
        )

    OUTPUT_FILE.write_text(json.dumps([profile.model_dump() for profile in profiles], indent=2, ensure_ascii=False))
    print(f"Saved {len(profiles)} profile descriptions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
