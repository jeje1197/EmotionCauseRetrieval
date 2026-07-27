from __future__ import annotations

import json
from pathlib import Path
from typing import List

from clients.gemini_client import GeminiClient
from models import Profile, ProfileEvents
from prompts.profile_event_generation_prompt import CHARACTER_PROFILE_EVENT_GENERATION_PROMPT

INPUT_FILE = Path(__file__).with_name("profile_descriptions.json")
OUTPUT_FILE = Path(__file__).with_name("profiles_with_events.json")


def build_prompt(description: str) -> str:
    return CHARACTER_PROFILE_EVENT_GENERATION_PROMPT.format(description=description)


def main() -> None:
    client = GeminiClient()
    profiles_data = json.loads(INPUT_FILE.read_text())
    profiles: List[Profile] = []

    for profile_data in profiles_data:
        prompt = build_prompt(profile_data["description"])
        response = client.generate_structured_json(prompt, ProfileEvents.model_json_schema())
        profile_events = ProfileEvents.model_validate(response)

        profiles.append(
            Profile(
                personality_type=profile_data["personality_type"],
                description=profile_data["description"],
                events=profile_events.events,
                questions=profile_events.questions,
            )
        )

    OUTPUT_FILE.write_text(json.dumps([profile.model_dump() for profile in profiles], indent=2, ensure_ascii=False))
    print(f"Saved {len(profiles)} full profiles to {OUTPUT_FILE}")
