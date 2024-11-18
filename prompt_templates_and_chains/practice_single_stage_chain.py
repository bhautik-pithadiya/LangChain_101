# practice_single_stage_chain.py

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(
    model = 'gpt-3.5-turbo-instruct',
    api_key=api_key
)

tips_template = PromptTemplate.from_template(
    'List three practical tips for improving {topic}:\n'
)

# chain
chain = tips_template | llm

# invoke
print(chain.invoke({'topic': 'Procastination'}))