# summarize_single_chain.py

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    api_key=api_key
)

summary_template = PromptTemplate.from_template(
    "Summarize the following text in a concise paragraph:\n\nText: {text}\nSummary:"
)

text_input = "Machine learning models are designed to learn from data without explicit programming. They have various applications, such as image recognition, natural language processing, and recommendation systems."

summary_chain = summary_template | llm
summary = summary_chain.invoke({'text':text_input})
print(summary)