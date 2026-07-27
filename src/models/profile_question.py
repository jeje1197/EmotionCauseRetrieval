from pydantic import BaseModel, Field


class ProfileQuestion(BaseModel):
    question: str = Field(description="A question about the character based on a later event.")
    target_event_id: int = Field(description="The id of the event that the question refers to.")
    answer_event_id: int = Field(description="The id of the prior event that answers the question.")
