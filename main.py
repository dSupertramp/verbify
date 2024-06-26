from verbify.sentiment import sentiment_analysis
from verbify.summarization import summarize
from verbify.translation import translate
from verbify.correction import correct
from verbify.ner import ner
from verbify.generation import contextual_text_generation
from verbify.classification import classify
from verbify.llm.groq import GroqLLM
from verbify.llm.openai import OpenAILLM
from dotenv import load_dotenv
import os

load_dotenv()

text = "The latest iPhone model has many new features."
llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), text=text)
print(classify(llm=llm))
