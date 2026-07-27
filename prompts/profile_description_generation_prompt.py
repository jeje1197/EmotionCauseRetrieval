CHARACTER_PROFILE_GENERATION_PROMPT = """\
Generate a 2-sentence description for a character profile based on the Myers-Briggs (MBTI) personality type {personality_type} in the following format:

[Character name] is a(n) [occupation / social role] who values [values] and enjoys [hobbies]. [Character] is currently [current focus in life].

Important:
- Do not include the MBTI type in the description.
- Replace each placeholder with appropriate content.
- Only return the description. Do not add any additional text or explanations.

Description:
"""