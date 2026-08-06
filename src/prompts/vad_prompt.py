VAD_PROMPT = """\
For the following user query, generate a score for each dimension of the VAD emotion model. The VAD emotion model is a three-dimensional framework used to map feelings on continuous axes. Here are the three dimensions that compose the model.

# VAD Emotion Model Dimensions
Valence (V): Measures how positive or negative an emotion feels, ranging from unhappy or sad to happy and joyful.
Arousal (A): Measures intensity or energy levels, ranging from calm and bored to excited and stimulated.
Dominance (D): Measures feeling in control or empowered versus feeling submissive or helpless.


# Output Format (JSON)
{
	"valence": number,
	"arousal": number,
	"dominance": number
}

# Important:
- Always output the three dimensions with valid floating point numbers.
- Each dimensional score should be a floating point number in the range of -1 to 1.
- Return JSON only. Do not add any other text or explanations.


# Data to Score
User Query: It was a battle to the end, but Spain managed to win it all.
Your Response:\
"""