# multi_stage_chain_summ_q_a.py

from langchain_core.runnables import RunnableSequence
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define the LLM
llm = OpenAI(
    model = 'gpt-3.5-turbo-instruct',
    api_key=api_key
)

# Step 1 : Summarization prompt
summary_template = PromptTemplate.from_template(
    'Summarize the following text in a concise paragraph:\n\nText: {text}\nSummary:'
)

# Step 2: Q&A prompt
qa_template = PromptTemplate.from_template(
    "Based on the summary provided, answer the following question:\n\nSummary: {summary}\nQuestion: {question}\nAnswer: " 
)

# Create the chain
summarization_chain = summary_template | llm
qa_chain = qa_template | llm

multi_stage_chain = RunnableSequence(first=summarization_chain,last=qa_chain)

# Input text and question

input_data = {
    "text": "Artificial intelligence is transforming industries by enabling machines to learn from data. It has applications in healthcare, finance, and other sectors.",
    "question": "What are some applications of AI?"
}

# Run the chain
output = multi_stage_chain.invoke({'text':input_data['text']})

print(output)