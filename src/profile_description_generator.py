from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, List, Tuple

from clients.gemini_client import GeminiClient
from models import DescriptionResponse, Profile
from prompts.profile_description_generation_prompt import CHARACTER_PROFILE_GENERATION_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "profile_descriptions.json"


def build_prompt(personality_type: str, prior_characters: List[Tuple[str, str]]) -> str:
    previously_used_names = ", ".join(name for name, _ in prior_characters) or "None"
    previously_used_occupations = ", ".join(occupation for _, occupation in prior_characters) or "None"

    return CHARACTER_PROFILE_GENERATION_PROMPT.format(
        personality_type=personality_type,
        previously_used_names=previously_used_names,
        previously_used_occupations=previously_used_occupations,
    )


def generate_description(personality_type: str, prior_characters: List[Tuple[str, str]]) -> Profile:
    logger.info("Generating description for personality type %s", personality_type)
    prompt = build_prompt(personality_type, prior_characters)
    logger.debug("Prompt for %s: %s", personality_type, prompt)

    response = _retry_generate(prompt, DescriptionResponse.model_json_schema())
    description_response = DescriptionResponse.model_validate(response)

    return Profile(
        personality_type=personality_type,
        description=description_response.description,
    )


def _retry_generate(prompt: str, schema: Any, max_attempts: int = 3, backoff_seconds: float = 2.0) -> Any:
    client = GeminiClient()
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.info("Attempt %d/%d for structured generation", attempt, max_attempts)
            response = client.generate_structured_json(prompt, schema)
            logger.debug("Raw response on attempt %d: %s", attempt, response)
            return response
        except Exception as exc:
            logger.warning("Generation failed on attempt %d/%d: %s", attempt, max_attempts, exc)
            if attempt >= max_attempts:
                logger.error("All generation attempts failed")
                raise
            sleep_time = backoff_seconds * attempt
            logger.info("Retrying in %.1f seconds", sleep_time)
            time.sleep(sleep_time)


def extract_name_and_role(description: str) -> Tuple[str, str]:
    first_sentence = description.split(".", 1)[0].strip()

    name_match = re.match(r"^([A-Z][a-zA-Z]+)", first_sentence)
    occupation = "Unknown"
    name = "Unknown"

    if name_match:
        name = name_match.group(1)

    occupation_match = re.search(r"is an? ([^,\.]+)", first_sentence)
    if occupation_match:
        occupation = occupation_match.group(1).strip()
    else:
        alternate_match = re.search(r"^\w+ is ([^,\.]+)", first_sentence)
        if alternate_match:
            occupation = alternate_match.group(1).strip()

    if "who values" in first_sentence:
        occupation = first_sentence.split("who values", 1)[0]
        occupation = occupation.replace(f"{name} is a", "").replace(f"{name} is an", "").strip()

    if not occupation:
        occupation = "Unknown"

    return name, occupation


def main() -> None:
    logger.info("Starting profile description generation")
    client = GeminiClient()
    profiles: List[Profile] = []
    prior_characters: List[Tuple[str, str]] = []

    for personality_type in PERSONALITY_TYPES:
        profile = generate_description(personality_type, prior_characters)
        profiles.append(profile)

        name, occupation = extract_name_and_role(profile.description)
        prior_characters.append((name, occupation))
        logger.info("Recorded character %s with occupation %s", name, occupation)

    OUTPUT_FILE.write_text(json.dumps([profile.model_dump() for profile in profiles], indent=2, ensure_ascii=False))
    logger.info("Saved %d profile descriptions to %s", len(profiles), OUTPUT_FILE)


if __name__ == "__main__":
    main()
