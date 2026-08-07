"""Gemini client wrapper for structured Gemini SDK calls."""

import json
from typing import Any, Dict, Iterable, List

from google import genai
from google.genai import types


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

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        if not getattr(response, "embeddings", None):
            raise RuntimeError("Embedding request did not return embeddings.")
        return [list(embedding.values) for embedding in response.embeddings]
