"""Make one Gemini VAD request without loading a dataset."""

from __future__ import annotations

import json

from src.clients.gemini_client import GeminiClient
from src.scripts.embed_dataset_helpers import score_vad


def main() -> None:
    utterance = "I'm delighted that we finally finished the project together."
    scores = score_vad(GeminiClient(), utterance)
    print(f"Utterance: {utterance}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
