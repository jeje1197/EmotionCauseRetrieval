from pydantic import BaseModel, Field


class ProfileEvent(BaseModel):
    id: int = Field(description="Unique integer id of the event in chronological order.")
    description: str = Field(description="A 1-sentence description of an event.")
