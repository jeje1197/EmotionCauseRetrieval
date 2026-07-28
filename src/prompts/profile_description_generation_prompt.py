CHARACTER_PROFILE_GENERATION_PROMPT = """\
Generate a 2-sentence description for a character profile based on the Myers-Briggs (MBTI) personality type {personality_type} in the following format:

[Character name] is a(n) [occupation / social role] who values [values] and enjoys [hobbies]. [Character] is currently [current focus in life].

Important:
- Replace each placeholder with appropriate content.
- Do not include the MBTI type in the description.
- Only provide a first name for the character.
- Do not use a name that has been used in previous descriptions.
- Do not use an occupation that has been used in previous descriptions.
- For the occupation, choose a realistic job or social role that aligns with the personality type, but it doesn't have to be limited to the most likely role for that personality type. It can be a creative or unconventional choice or social role such as "student," "parent," or "community leader" if it fits the character.
- Only return the description. Do not add any additional text or explanations.

Previously Used Names: {previously_used_names}
Previously Used Occupations: {previously_used_occupations}

Your Output:
"""