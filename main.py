from verbify.sentiment import sentiment_analysis
from verbify.summarization import summarize
from verbify.translation import translate
from verbify.correction import correct
from verbify.ner import ner
from verbify.generation import contextual_text_generation
from llm.groq import GroqLLM
from dotenv import load_dotenv
import os

load_dotenv()

text = "The car is broken."
llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"), text=text)
# ner = ner(llm=llm)
generated_text = contextual_text_generation(llm=llm, max_length=50)
print(generated_text)