from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List

from .profile_event import ProfileEvent
from .profile_question import ProfileQuestion


class Profile(BaseModel):
    personality_type: str = Field(description="MBTI personality type.")
    description: str = Field(description="Generated character description.")
    events: List[ProfileEvent] = Field(default_factory=list, description="Generated life events for the profile.")
    questions: List[ProfileQuestion] = Field(default_factory=list, description="Generated questions linked to prior events.")
