CHARACTER_PROFILE_EVENT_GENERATION_PROMPT = """\
Based on the character description provided below, generate a list of 10 events that occur in the character's life and 2 questions that can be answered by the events.

Event Guidelines:
Each event should be a 1-sentence description of an observation, action, or occurrence in the character's life.
The events should be realistic and take place in chronological order. 

Question Guidelines:
- Each question should be based on one of the events that took place and directly answerable by a single prior event.
- The first question should be a factual question that follows logical causality, such as Q: "Why did [Character] carry an umbrella around today?" A: "[Character] noticed that the street looked damp while on their morning walk."
- The second question should be emotionally grounded based on an event that took place, such as Q: "Why did [Character] hesitate before going home today?" A:"[Character] felt overwhelmed by the amount of homework they had to finish."
- The single most relevant event should be selected as the answer.

Character Description: {description}
Output Format (JSON):
{{
    "events": [
        {{"id": 1, "description": "A 1-sentence description of an event."}}
    ],
    "questions": [
        {{"question": "A 1-sentence factual question.", "target_event_id": 7, "answer_event_id": 5}},
        {{"question": "A 1-sentence emotional question.", "target_event_id": 8, "answer_event_id": 2}}
    ]
}}

Important:
- Each event should be given a unique integer ID from 1 through 10 in the order they occur.
- There should be exactly 10 events and 2 questions in the output.
- The target_event_id should reference the ID of the event that the question is based on.
- The target_event_id should be greater than 6 to ensure that each question has at least 6 prior events to choose from as potential answers.
- The answer_event_id should reference the ID of the single prior event that best explains the behavior, action, feeling, or occurence mentioned in the question.
- The answer_event_id should be less than the target_event_id to ensure that the answer is based on a prior event.
- Only events with IDs less than the target_event_id should be considered as potential answer events.
- Return only the JSON object. Do not include any additional text or explanations.

Your Output:
"""