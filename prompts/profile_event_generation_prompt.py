CHARACTER_PROFILE_EVENT_GENERATION_PROMPT = """\
Based on the character description provided below, generate a list of 10 events that occur in the character's life and 2 questions that can be answered by the events.

Event Guidelines:
Each event should be a 1-sentence description of an observation, action, or occurrence in the character's life.
The events should be realistic and take place in chronological order. 

Question Guidelines:
- Each question should be based on one of the events that took place and directly answerable by a single prior event.
- The first question should be a factual question that follows logical causality, such as Q: "Why did [Character] carry an umbrella around today?" A: "[Character] noticed that the street looked damp while on their morning walk."
- The second question should be emotionally grounded based on an event that took place, such as Q: "Why did [Character] hesistate before going home today?" A:"[Character] felt overwhelmed by the amount of homework they had to finish."

Character Description: {description}
Output Format (JSON):
{{
    "events": [
        {{"id": 1, "description": "A 1-sentence description of an event."}}
    ],
    "questions": [
        {{"question": "A 1-sentence factual question.", "target_event_id": 5, "answer_event_id": 1}},
        {{"question": "A 1-sentence emotional question.", "target_event_id": 8, "answer_event_id": 3}},
    ]
}

Important:
- Each event should have a integer unique ID 1 through 10.
- There should be exactly 10 events and 2 questions in the output.
- The target_event_id should reference the ID of the event that the question is based on.
- The answer_event_id should reference the ID of the event that explains the behavior or action in the question.
- Only events with IDs < target_event_id should be considered as potential answer events.
- Return only the JSON object. Do not include any additional text or explanations.

Your Output:
"""