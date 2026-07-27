from pydantic import BaseModel, Field


class DescriptionResponse(BaseModel):
    description: str = Field(description="A 2-sentence profile description.")
