from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import List, Tuple

from clients.gemini_client import GeminiClient
from models import Profile, ProfileEvents
from prompts.profile_event_generation_prompt import CHARACTER_PROFILE_EVENT_GENERATION_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = DATA_DIR / "profile_descriptions.json"
OUTPUT_FILE = DATA_DIR / "profiles_with_events.json"

TARGET_EVENT_SCHEDULE: List[Tuple[int, int]] = [
    (11, 14),
    (12, 15),
    (13, 11),
    (14, 12),
    (15, 13),
    (11, 15),
    (12, 11),
    (13, 12),
    (14, 13),
    (15, 14),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 11),
    (11, 13),
]

QUESTION_ORDER_SCHEDULE: List[str] = [
    "factual first, affective second" if index % 2 == 0 else "affective first, factual second"
    for index in range(len(TARGET_EVENT_SCHEDULE))
]


def build_prompt(description: str, target_event_ids: Tuple[int, int], question_order: str) -> str:
    first_question_type = "factual_causal"
    second_question_type = "affective_causal"
    if question_order.startswith("affective"):
        first_question_type, second_question_type = second_question_type, first_question_type

    return CHARACTER_PROFILE_EVENT_GENERATION_PROMPT.format(
        description=description,
        first_target_event_id=target_event_ids[0],
        second_target_event_id=target_event_ids[1],
        question_order=question_order,
        first_question_type=first_question_type,
        second_question_type=second_question_type,
    )


def _generate_profile_events(args: Tuple[int, dict]) -> Profile:
    index, profile_data = args
    personality_type = profile_data["personality_type"]
    target_event_ids = TARGET_EVENT_SCHEDULE[index]
    question_order = QUESTION_ORDER_SCHEDULE[index]

    logger.info("Generating events for personality type %s using targets %s and order %s", personality_type, target_event_ids, question_order)
    prompt = build_prompt(profile_data["description"], target_event_ids, question_order)
    logger.debug("Prompt for %s: %s", personality_type, prompt)

    client = GeminiClient()
    response = client.generate_structured_json(prompt, ProfileEvents.model_json_schema())
    profile_events = ProfileEvents.model_validate(response)

    return Profile(
        personality_type=personality_type,
        description=profile_data["description"],
        events=profile_events.events,
        questions=profile_events.questions,
    )


def main() -> None:
    logger.info("Starting profile event generation")
    profiles_data = json.loads(INPUT_FILE.read_text())

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        profiles = list(executor.map(_generate_profile_events, enumerate(profiles_data)))

    OUTPUT_FILE.write_text(json.dumps([profile.model_dump() for profile in profiles], indent=2, ensure_ascii=False))
    logger.info("Saved %d full profiles to %s", len(profiles), OUTPUT_FILE)


if __name__ == "__main__":
    main()
