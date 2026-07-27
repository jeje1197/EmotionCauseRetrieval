from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List

from .profile_event import ProfileEvent
from .profile_question import ProfileQuestion


class ProfileEvents(BaseModel):
    events: List[ProfileEvent] = Field(description="A chronological list of events in the character's life.")
    questions: List[ProfileQuestion] = Field(description="Questions that can be answered by prior events.")
