"""Gemini client wrapper for structured Gemini SDK calls."""

import json
from typing import Any, Dict

from google import genai


class GeminiClient:
    def __init__(self, model: str = "gemini-3.6-flash") -> None:
        self.client = genai.Client()
        self.model = model

    def generate_structured_json(self, prompt: str, schema: Dict[str, Any]) -> Any:
        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        )

        output_text = getattr(interaction, "output_text", None)
        if output_text is None:
            output_text = getattr(interaction, "output", None)

        if output_text is None:
            raise RuntimeError("Gemini interaction did not return any output text.")

        if isinstance(output_text, str):
            return json.loads(output_text)

        return output_text
