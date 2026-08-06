CHARACTER_PROFILE_EVENT_GENERATION_PROMPT = """\
Based on the character description provided below, generate a chronological history of exactly 15 events from the character's life and 2 questions that can be answered using that history.

Event Guidelines:
- Each event should be a single-sentence description of an observation, action, experience, or occurrence in the character's life.
- The events should be realistic, internally consistent, and presented in chronological order.
- The history should span multiple periods or stages of the character's life rather than concentrating all events within one narrow period, setting, or activity.
- Events may include childhood or adolescent memories, education, work, relationships, hobbies, personal achievements, setbacks, everyday activities, emotional experiences, and life milestones.
- Include a mixture of mundane, moderately significant, and highly meaningful events.
- Vary sentence structure, temporal framing, setting, circumstances, emotional intensity, and event type.
- Reflect the intersection of the character's personality and life circumstances without making every event a stereotypical expression of the character's personality.
- Do not follow a fixed narrative template such as childhood, education, career problem, emotional crisis, and successful resolution.
- Do not require the final event to resolve the character's challenges or end the history positively.

Causal-Pair Guidelines:
- Construct two distinct target-and-answer event pairs within the history.
- Each target event should describe an action, reaction, decision, feeling, or behavior that is best explained by one earlier event.
- The answer event should provide enough information for the relationship to be understandable without requiring unsupported assumptions.
- Long-range emotional or experiential connections are allowed, but the history must establish the relevant connection clearly enough that the selected answer is more defensible than the other candidates.
- Do not make every factual relationship an immediate practical response.
- Do not make every emotional relationship a repetition of the same explicitly named emotion.
- Emotional effects may include hesitation, confidence, avoidance, protectiveness, gratitude, attachment, renewed motivation, caution, relief, or a bittersweet reaction, as well as directly stated emotions.
- A prior event may lead to a different later emotion or behavior rather than merely repeating the same emotional state.

Distractor Guidelines:
- For each question, include at least 3 plausible distractor events among the eligible prior events.
- Plausible distractors may be:
  1. topically similar but emotionally or causally mismatched,
  2. emotionally similar but topically or causally mismatched, or
  3. topically and emotionally similar but not the actual cause.
- Despite these distractors, one prior event must remain the single best answer.
- Avoid making the answer identifiable only because it repeats distinctive words from the question or target event.
- Use natural paraphrases where appropriate, while preserving a clear causal connection.

Question Guidelines:
- Each question should ask why the character performed an action, made a decision, reacted in a particular way, or experienced a response described in its target event.
- Each question must be directly answerable by one single prior event.
- The first question should test primarily situational or factual causality.
- The second question should test primarily emotionally or experientially grounded causality.
- Vary the wording of the questions. Do not always phrase them as “Why did [Character] feel [emotion]?”
- A question with ordinary behavioral wording may have an emotionally grounded answer, and an explicitly emotional reaction may have a concrete situational explanation.
- Do not quote or closely repeat the answer event in the question.
- The single most relevant prior event should be selected as the answer.

Examples of acceptable question forms include:
- "Why did [Character] bring an extra charger to the meeting?"
- "What prompted [Character] to reconsider the invitation?"
- "Why was returning to the building especially meaningful to [Character]?"
- "What earlier experience shaped [Character]'s cautious response?"
- "Why did an otherwise successful moment feel bittersweet to [Character]?"

Character Description: {description}
Question Order: {question_order}
First Target Event ID: {first_target_event_id}
Second Target Event ID: {second_target_event_id}

Output Format (JSON):
{{
    "events": [
        {{"id": 1, "description": "A single-sentence description of an event."}}
    ],
    "questions": [
        {{
            "question": "A single-sentence question.",
            "question_type": "{first_question_type}",
            "target_event_id": {first_target_event_id},
            "answer_event_id": "A single prior event ID."
        }},
        {{
            "question": "A single-sentence question.",
            "question_type": "{second_question_type}",
            "target_event_id": {second_target_event_id},
            "answer_event_id": "A single prior event ID."
        }}
    ]
}}

Important:
- Return exactly 15 events and exactly 2 questions.
- Give each event a unique integer ID from 1 through 15 in chronological order.
- The target_event_id must reference the event containing the action, reaction, decision, feeling, or behavior asked about.
- Each target_event_id must be between 11 and 15, inclusive, so that at least 10 prior events are available as candidates.
- Use two different target_event_id values.
- Vary the target positions across generated character profiles rather than consistently choosing the same IDs.
- Do not automatically place the factual target before the affective target.
- The answer_event_id must identify the single prior event that best explains the target.
- The answer_event_id must be less than its corresponding target_event_id.
- Only events with IDs lower than the target_event_id may be considered as answer candidates.
- The two questions should normally use different answer events.
- Validate that every referenced event ID exists.
- Return valid JSON only, without Markdown, commentary, or explanations.

Your Output:
"""