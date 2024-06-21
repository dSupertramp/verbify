sentiment_analysis_prompt = """
Perform a comprehensive sentiment analysis on the provided text.
"""


summarize_prompt = """
Perform a comprehensive summarize on the provided text.
"""

translate_prompt = """
Perform a comprehensive translation on the provided text in the following language:
{language}
"""

correction_prompt = """
Perform a comprehensive grammar and style correction on the provided text.
"""

ner_prompt = """
Perform a comprehensive NER (Named Entity Recognition) on the provided text.
For each element, you have to define the element, its type, the starting position and end position.

This is an example:

Text:
Barack Obama was born in Hawaii.

Expected output:
{"element": "Barack Obama", "type": "PERSON", "start_pos": 0, "end_pos": 12},
{"element": "Hawaii", "type": "LOCATION", "start_pos": 22, "end_pos": 28}
"""


contextual_text_generation_prompt = """
Perform a comprehenesive context-aware generation of the given text, from a maximum of {max_length} characters.
"""
